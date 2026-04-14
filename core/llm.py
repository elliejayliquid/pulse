"""
LLM client — OpenAI-compatible API wrapper.

Supports local llama-server and cloud APIs (OpenAI, OpenRouter, Anthropic, etc.)
via the OpenAI SDK's base_url mechanism.

Two call modes:
- chat()           → heartbeat/scheduled tasks (returns PulseResponse with JSON parsing)
- chat_with_tools() → Telegram conversations (tool-calling loop, returns plain text)
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, APIConnectionError

logger = logging.getLogger(__name__)


def extract_think_content(text: str) -> str:
    """Pull out the reasoning content from `<think>...</think>` blocks.

    Mirror image of `strip_think_tags`: instead of removing the thinking
    blocks, we collect them. Used to optionally show the model's chain
    of thought to the user (e.g. in Telegram via expandable blockquote).

    Returns the joined reasoning text, or "" if there's nothing to find.
    Handles the same quirks as strip_think_tags: standard tags, partial
    tags (cut off mid-think), missing opening tag, and plain-text
    "Thinking Process:" blocks from LM Studio.
    """
    if not text:
        return ""
    parts: list[str] = []
    # 1. Standard <think>...</think> tags
    for m in re.finditer(r'<think>([\s\S]*?)</think>', text):
        parts.append(m.group(1).strip())
    # 2. Missing opening tag (LM Studio strips <think> but leaves </think>)
    if not parts:
        m = re.match(r'^([\s\S]*?)</think>', text)
        if m:
            parts.append(m.group(1).strip())
    # 3. Plain-text "Thinking Process:" block
    if not parts:
        m = re.match(
            r'^Thinking(?:\s+Process)?:\s*\n([\s\S]*?)(?=\n\{|\Z)',
            text
        )
        if m:
            parts.append(m.group(1).strip())
    return "\n\n".join(p for p in parts if p)


def strip_think_tags(text: str) -> str:
    """Remove thinking/reasoning blocks from model output.

    Handles two formats:
    - <think>...</think> tags (Qwen 3, DeepSeek, etc.)
    - Plain-text "Thinking Process:" blocks (Qwen 3.5 via LM Studio when
      the template doesn't support <think> tags properly)
    """
    if not text:
        return text
    # 1. Standard <think>...</think> tags
    cleaned = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
    # 2. Partial think tags (model cut off mid-think, never closed)
    cleaned = re.sub(r'<think>[\s\S]*$', '', cleaned)
    # 3. Missing opening tag (LM Studio strips <think> but leaves </think>)
    cleaned = re.sub(r'^[\s\S]*?</think>\s*', '', cleaned)
    # 4. Plain-text thinking blocks (e.g. "Thinking Process:\n1. **Analyze**...")
    cleaned = re.sub(
        r'^Thinking(?:\s+Process)?:\s*\n[\s\S]*?(?=\n\{|\Z)',
        '', cleaned
    )
    
    # 5. Strip EOS tokens if they leaked through
    cleaned = cleaned.replace('</s>', '')
    
    return cleaned.strip()


@dataclass
class PulseResponse:
    """Parsed response from the companion's heartbeat/task."""
    thinking: str = ""
    action: str = "silent"         # notify | schedule | silent
    message: str = ""
    schedule_task: str = ""
    schedule_when: str = ""
    raw: str = ""

    @classmethod
    def from_llm_output(cls, text: str) -> "PulseResponse":
        """Parse structured JSON response from the LLM."""
        # Strip think tags before parsing JSON
        cleaned = strip_think_tags(text)
        resp = cls(raw=cleaned)

        # Try to extract JSON from the response
        try:
            # Handle responses that might have text before/after JSON
            # Check for array first (some models return [{...}, {...}])
            data = None
            arr_start = cleaned.find("[")
            json_start = cleaned.find("{")
            if arr_start >= 0 and (json_start < 0 or arr_start < json_start):
                arr_end = cleaned.rfind("]") + 1
                if arr_end > arr_start:
                    try:
                        items = json.loads(cleaned[arr_start:arr_end])
                        if isinstance(items, list) and items:
                            # Find the notify/action object (skip tool-like entries)
                            data = None
                            for item in reversed(items):
                                if isinstance(item, dict) and (
                                    "message" in item or "thinking" in item
                                    or item.get("action") in ("notify", "schedule", "silent")
                                ):
                                    data = item
                                    break
                            if not data:
                                data = items[-1]  # fallback to last item
                            # Unwrap "args" wrapper if model used tool-call style
                            if "args" in data and isinstance(data["args"], dict):
                                data.update(data.pop("args"))
                            logger.info(f"Parsed action from JSON array ({len(items)} items)")
                    except json.JSONDecodeError:
                        pass  # fall through to single-object parsing

            if data is None:
                if json_start >= 0:
                    json_end = cleaned.rfind("}") + 1
                    if json_end > json_start:
                        data = json.loads(cleaned[json_start:json_end])

            if data is None:
                # No JSON found — treat entire response as a message
                logger.warning("No JSON found in response, treating as notification")
                resp.action = "notify"
                resp.message = cleaned.strip()
                return resp

            resp.thinking = data.get("thinking", "")
            resp.action = data.get("action", "silent")
            resp.message = data.get("message", "")

            schedule = data.get("schedule")
            if schedule and isinstance(schedule, dict):
                resp.schedule_task = schedule.get("task", "")
                resp.schedule_when = schedule.get("when", "")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # If JSON fails, treat it as a notification with the raw text
            resp.action = "notify"
            resp.message = cleaned.strip()

        return resp


class LLMClient:
    """OpenAI-compatible client for local llama-server or cloud APIs."""

    def __init__(self, endpoint: str, model_name: str = "default",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 top_p: float = 1.0,
                 api_key: str = "", usage_tracker=None,
                 reasoning: bool = False, reasoning_effort: str = "",
                 provider_type: str = "local"):
        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key or "not-needed",
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        self._available = None
        self._usage = usage_tracker
        self._provider_name = ""  # set by engine for usage logging
        # Build extra_body for cloud providers that support reasoning control
        # Only OpenRouter uses the reasoning meta-parameter; OpenAI handles
        # reasoning natively per model and would reject unknown fields.
        # OpenAI newer models (GPT-5.x, o-series) require max_completion_tokens
        # instead of max_tokens. Other providers and local servers use max_tokens.
        self._max_tokens_param = (
            "max_completion_tokens" if provider_type == "openai"
            else "max_tokens"
        )
        self._extra_body = {}
        if provider_type == "openrouter":
            if reasoning:
                # OpenRouter: prefer explicit effort tier when set, else
                # just enable reasoning at the default budget. Effort tiers
                # are provider-agnostic: each backend maps low/medium/high
                # to its own internal token budget. For models like Grok
                # 4.1 Fast that ship with a small default budget, bumping
                # this is the difference between a 200-char restatement
                # and a real reasoning trace.
                if reasoning_effort in ("low", "medium", "high"):
                    self._extra_body["reasoning"] = {"effort": reasoning_effort}
                else:
                    self._extra_body["reasoning"] = {"enabled": True}
            else:
                self._extra_body["reasoning"] = {"effort": "none"}

        # Captured reasoning content from the most recent call. The engine
        # reads this after each chat()/chat_with_tools() to optionally
        # surface the model's chain of thought to the user. Empty string
        # if the model didn't expose any reasoning. Sources we look at:
        #   - <think>...</think> tags inside the response text
        #   - non-standard `message.reasoning_content` (DeepSeek)
        #   - non-standard `message.reasoning` (OpenRouter)
        self.last_reasoning: str = ""

    def _collect_reasoning(self, message, text: str) -> str:
        """Pull reasoning from wherever the provider chose to put it.

        Three sources, in priority order — first non-empty wins:
          1. `message.reasoning_content` — DeepSeek's official API field
          2. `message.reasoning` — OpenRouter's reasoning field (when
             `extra_body.reasoning.enabled = true` was set)
          3. `<think>...</think>` blocks inside the regular text content
             — universal fallback for local thinking models (Qwen3-Thinking,
             DeepSeek R1, GLM-Z1, etc.) that emit reasoning inline.

        Returns "" if nothing was found.
        """
        # OpenAI SDK exposes unknown fields via the message object's __dict__
        # or via getattr — both work because the SDK uses pydantic models
        # that pass through extras.
        for attr in ("reasoning_content", "reasoning"):
            val = getattr(message, attr, None)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return extract_think_content(text)

    def _track(self, response):
        """Log token usage from an API response (no-op if no tracker)."""
        if self._usage and hasattr(response, "usage") and response.usage:
            self._usage.record(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                provider=self._provider_name,
                model=self.model_name,
            )
            # Log prompt cache stats (OpenAI automatic caching)
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            if cached:
                logger.info(
                    f"Prompt cache: {cached} cached of "
                    f"{response.usage.prompt_tokens} prompt tokens "
                    f"({cached * 100 // response.usage.prompt_tokens}% hit)"
                )

    def is_available(self) -> bool:
        """Check if the LLM server is reachable."""
        try:
            self.client.models.list()
            self._available = True
            return True
        except (APIConnectionError, Exception) as e:
            logger.debug(f"LLM server not available: {e}")
            self._available = False
            return False

    def chat(self, messages: list[dict]) -> Optional[PulseResponse]:
        """Send messages to the LLM and get a parsed response.

        Used for heartbeat ticks and scheduled tasks (expects JSON format).

        Args:
            messages: OpenAI-format messages list

        Returns:
            PulseResponse or None if LM Studio is unavailable
        """
        # Reset captured reasoning at the start of each call so a previous
        # call's chain of thought never leaks forward.
        self.last_reasoning = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                **{self._max_tokens_param: self.max_tokens},
                **({"extra_body": self._extra_body} if self._extra_body else {}),
            )

            self._track(response)
            finish = getattr(response.choices[0], "finish_reason", "unknown")
            message = response.choices[0].message
            text = message.content or ""
            self.last_reasoning = self._collect_reasoning(message, text)
            logger.info(f"LLM response ({len(text)} chars, finish={finish})")
            logger.debug(f"Raw response: {text[:200]}...")

            # Empty content = model spent all tokens on reasoning, treat as silent
            if not text.strip():
                logger.info("Empty response (reasoning model used all tokens?) — treating as silent.")
                return PulseResponse(raw="", action="silent")

            return PulseResponse.from_llm_output(text)

        except APIConnectionError:
            logger.warning("LLM server is not running or not reachable")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        skill_registry, max_rounds: int = 5) -> tuple[Optional[str], list[str]]:
        """Chat with tool-calling support for Telegram conversations.

        Implements an agentic loop:
        1. Send messages + tool definitions to llama-server
        2. If model returns tool_calls → execute them, feed results back
        3. Repeat until model gives a final text response (or max rounds)

        Args:
            messages: OpenAI-format messages list
            tools: Tool definitions (from SkillRegistry.get_all_tools())
            skill_registry: SkillRegistry instance for executing tool calls
            max_rounds: Max tool-calling rounds before forcing a text response

        Returns:
            Tuple of (final response text, list of tool names used)
        """
        # Work on a copy so we don't mutate the caller's messages
        msgs = list(messages)
        tools_used = []
        # Reset captured reasoning at the start of each call so a previous
        # call's chain of thought never leaks forward.
        self.last_reasoning = ""

        for round_num in range(max_rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,
                    tools=tools if tools else None,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    **{self._max_tokens_param: self.max_tokens},
                    **({"extra_body": self._extra_body} if self._extra_body else {}),
                )

                self._track(response)
                choice = response.choices[0]
                message = choice.message
                text = message.content or ""
                tool_calls = getattr(message, "tool_calls", None)
                finish = getattr(choice, "finish_reason", "unknown")

                # Debug: log full response when empty
                if not text and not tool_calls:
                    logger.warning(f"Empty response from LLM. Choice: {choice}")

                # If no tool calls, this is the final response
                if not tool_calls:
                    logger.info(f"LLM final response ({len(text)} chars, round {round_num + 1}, finish={finish})")
                    self.last_reasoning = self._collect_reasoning(message, text)
                    cleaned = strip_think_tags(text)
                    return (cleaned if cleaned else None), tools_used

                # Model wants to call tools
                logger.info(f"LLM requested {len(tool_calls)} tool call(s) (round {round_num + 1})")

                # Add the assistant's message (with tool_calls) to the conversation
                # We need to serialize tool_calls properly for the next API call
                assistant_msg = {"role": "assistant", "content": text or ""}
                tool_calls_list = []
                for tc in tool_calls:
                    tool_calls_list.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                assistant_msg["tool_calls"] = tool_calls_list
                msgs.append(assistant_msg)

                # Execute each tool call and add results
                for tc in tool_calls:
                    func_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                        logger.warning(f"Failed to parse tool arguments for {func_name}: {tc.function.arguments}")

                    logger.info(f"Executing tool: {func_name}({args})")
                    tools_used.append(func_name)
                    result = skill_registry.execute(func_name, args)

                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": func_name,
                        "content": str(result),
                    })

            except APIConnectionError:
                logger.warning("LLM server disconnected during tool loop")
                return None, tools_used
            except Exception as e:
                logger.error(f"Tool-calling loop failed (round {round_num + 1}): {e}")
                return None, tools_used

        # Exhausted max rounds — force a final response without tools
        logger.warning(f"Tool loop hit max rounds ({max_rounds}), requesting final response...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                temperature=self.temperature,
                top_p=self.top_p,
                **{self._max_tokens_param: self.max_tokens},
                **({"extra_body": self._extra_body} if self._extra_body else {}),
            )
            self._track(response)
            text = response.choices[0].message.content or ""
            return (strip_think_tags(text) or None), tools_used
        except Exception as e:
            logger.error(f"Final response after tool loop failed: {e}")
            return None, tools_used
