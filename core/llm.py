"""
LLM client — OpenAI-compatible API wrapper.

Supports local llama-server and cloud APIs (OpenAI, OpenRouter, Anthropic, etc.)
via the OpenAI SDK's base_url mechanism.

Two call modes:
- chat()           → heartbeat/scheduled tasks (returns PulseResponse with JSON parsing)
- chat_with_tools() → Telegram conversations (tool-calling loop, returns plain text)
"""

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, APIConnectionError, BadRequestError

logger = logging.getLogger(__name__)


def tool_definition_name(tool: dict) -> str:
    """Return the function name from an OpenAI-format tool definition."""
    return tool.get("function", {}).get("name", "")


def extend_tools_unique(tools: list[dict], new_tools: list[dict]) -> list[dict]:
    """Append tool definitions by name, skipping duplicates.

    Some providers reject duplicate function definitions with a hard 400.
    This can happen when search_tools is called more than once and two
    searches return the same on-demand skill.
    """
    existing = {tool_definition_name(tool) for tool in tools}
    added = []
    for tool in new_tools:
        name = tool_definition_name(tool)
        if not name or name in existing:
            if name:
                logger.debug(f"Skipping duplicate tool definition: {name}")
            continue
        tools.append(tool)
        added.append(tool)
        existing.add(name)
    return added


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


def empty_reply_nudge(autonomous: bool = False, truncated: bool = False) -> str:
    """Build the one-shot nudge sent when a tool-calling turn ends with no text.

    ``autonomous`` turns (heartbeats, scheduled tasks) have no user message to
    answer, so they're asked for their usual JSON decision instead — telling a
    heartbeat to "reply to the user" pushes it into notifying as plain text.
    ``truncated`` means the output budget ran out (usually all spent thinking),
    so the model is told to think briefly rather than retry the same plan.
    """
    parts = ["[SYSTEM:"]
    if truncated:
        parts.append(
            "Your last turn ran out of output space before you wrote anything — "
            "it was all spent thinking. Keep your thinking short this time."
        )
    elif autonomous:
        parts.append("You used tools but ended your turn without a decision.")
    else:
        parts.append("You performed a tool action but ended your turn without replying to the user.")
    if autonomous:
        parts.append(
            "This is your own time, not a reply to a message. Finish now with your "
            "usual JSON decision (notify, schedule or silent — silent is fine). "
            "Anything unfinished can be done in one short step or left for later.]"
        )
    else:
        parts.append(
            "Do not be silent. Respond now in text, in character, directly to the "
            "user's latest message.]"
        )
    return " ".join(parts)


def detect_tool_loop(recent_calls: list[tuple[str, str]]) -> Optional[str]:
    """Check recent tool calls for stuck patterns.
    
    Returns a warning message if a loop is detected, None otherwise.
    """
    if len(recent_calls) < 4:
        return None
    
    last_5 = recent_calls[-5:]
    
    # Pattern 1: exact same (tool, args) repeated 3+ times in last 5
    counts = Counter(last_5)
    for call, count in counts.items():
        if count >= 3:
            return (
                f"⚠️ LOOP DETECTED: You've called {call[0]} with identical arguments "
                f"{count} times. You appear to be stuck. Stop calling tools and "
                f"produce your final response now."
            )
    
    # Pattern 2: same tool name 5 times in a row (even with different args)
    if len(last_5) >= 5:
        names = [c[0] for c in last_5]
        # search_tools is expected infrastructure: a resident may legitimately
        # open several different tool drawers in one turn. Exact duplicate
        # searches are still caught by Pattern 1 above.
        if len(set(names)) == 1 and names[0] != "search_tools":
            return (
                f"⚠️ LOOP DETECTED: You've called {names[0]} 5 times in a row. "
                f"Review your approach — are you making progress? If not, stop "
                f"and produce your final response."
            )
    
    return None


@dataclass
class PulseResponse:
    """Parsed response from the companion's heartbeat/task."""
    thinking: str = ""
    continuity: str = ""       # compact episode note for the resident's future self
    action: str = "silent"         # notify | schedule | silent
    message: str = ""
    schedule_task: str = ""
    schedule_when: str = ""
    raw: str = ""

    @staticmethod
    def _parse_labeled_response(text: str) -> Optional[dict]:
        """Parse loose label-style heartbeat replies.

        Some models occasionally answer with markdown-ish labels instead of the
        required JSON, e.g.:

            **thinking:** ...
            **action:** silent

        Treat that as structured output rather than leaking it as a notification.
        """
        if not text:
            return None

        label_re = re.compile(
            r'(?im)^\s*(?:\*\*)?\s*(thinking|continuity|action|message|schedule)\s*:\s*(?:\*\*)?\s*'
        )
        matches = list(label_re.finditer(text))
        if not matches:
            return None

        fields: dict[str, str] = {}
        for i, match in enumerate(matches):
            key = match.group(1).lower()
            value_start = match.end()
            value_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[value_start:value_end].strip()
            if value:
                fields[key] = value

        action = fields.get("action", "").strip().lower()
        action = re.split(r'\s+', action, maxsplit=1)[0].strip('.,')
        if action not in ("notify", "schedule", "silent"):
            return None

        return {
            "thinking": fields.get("thinking", ""),
            "continuity": fields.get("continuity", ""),
            "action": action,
            "message": fields.get("message", ""),
        }

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
                                    or "continuity" in item
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
                data = cls._parse_labeled_response(cleaned)

            if data is None:
                # No JSON (or labeled fields) found in the heartbeat response.
                # Do NOT push the raw text to the user: that leaks the model's
                # private thinking (deliberation prose, raw JSON envelopes,
                # markdown fences) into the chat. Stay silent and log the raw
                # for diagnosis instead; the model gets another chance on the
                # next heartbeat.
                preview = cleaned[:200].replace("\n", " ")
                logger.warning(
                    "Heartbeat response had no parsable JSON; staying "
                    f"silent rather than leak thinking. Raw preview: {preview!r}"
                )
                resp.action = "silent"
                resp.message = ""
                return resp

            resp.thinking = data.get("thinking", "")
            resp.continuity = data.get("continuity", "")
            resp.action = data.get("action", "silent")
            resp.message = data.get("message", "")

            schedule = data.get("schedule")
            if schedule and isinstance(schedule, dict):
                resp.schedule_task = schedule.get("task", "")
                resp.schedule_when = schedule.get("when", "")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            data = cls._parse_labeled_response(cleaned)
            if data:
                resp.thinking = data.get("thinking", "")
                resp.continuity = data.get("continuity", "")
                resp.action = data.get("action", "silent")
                resp.message = data.get("message", "")
            else:
                # Same reason as above: a malformed response is not something
                # the user should see. Stay silent; log the raw for diagnosis.
                preview = cleaned[:200].replace("\n", " ")
                logger.warning(
                    f"Heartbeat response failed JSON parse ({e}); staying "
                    f"silent rather than leak thinking. Raw preview: {preview!r}"
                )
                resp.action = "silent"
                resp.message = ""

        return resp


class LLMClient:
    """OpenAI-compatible client for local llama-server or cloud APIs."""

    def __init__(self, endpoint: str, model_name: str = "default",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 top_p: float = 1.0, top_k: int | None = None,
                 api_key: str = "", usage_tracker=None,
                 reasoning: bool = False, reasoning_effort: str = "",
                 provider_type: str = "local",
                 provider_routing: dict | None = None):
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

            # OpenRouter provider routing: pin to specific backend providers
            # (e.g. {"order": ["mistral"], "allow_fallbacks": false,
            # "require_parameters": true}). Used when a model's behavior
            # varies by backend host and you want consistency. See OpenRouter
            # docs at openrouter.ai/docs/provider-routing.
            if provider_routing:
                self._extra_body["provider"] = provider_routing

        # top_k is a llama.cpp / OpenRouter sampler (not part of the base
        # OpenAI schema), so it rides in extra_body. Only sent when set.
        if top_k is not None:
            self._extra_body["top_k"] = top_k

        # Captured reasoning content from the most recent call. The engine
        # reads this after each chat()/chat_with_tools() to optionally
        # surface the model's chain of thought to the user. Empty string
        # if the model didn't expose any reasoning. Sources we look at:
        #   - <think>...</think> tags inside the response text
        #   - non-standard `message.reasoning_content` (DeepSeek)
        #   - non-standard `message.reasoning` (OpenRouter)
        self.last_reasoning: str = ""
        # Last error message from a failed call — read by the Telegram
        # channel to surface specific failure reasons to the user.
        self.last_error: str = ""

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
            # OpenAI automatic caching: cached tokens are a SUBSET of
            # prompt_tokens (unlike Anthropic, where the buckets are
            # disjoint). Recorded so cost_report.py can bill the cached
            # portion at the discounted rate instead of guessing.
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            self._usage.record(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                provider=self._provider_name,
                model=self.model_name,
                cache_read_input_tokens=cached or 0,
            )
            # Log prompt cache stats (OpenAI/OpenRouter automatic caching)
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details:
                cached = getattr(details, "cached_tokens", 0) or 0
                written = getattr(details, "cache_write_tokens", 0) or 0
                prompt_total = response.usage.prompt_tokens or 0
                if cached or written:
                    parts = []
                    if cached:
                        pct = cached * 100 // prompt_total if prompt_total else 0
                        parts.append(f"{cached} read ({pct}% hit)")
                    if written:
                        parts.append(f"{written} written")
                    logger.info(
                        f"Prompt cache: {', '.join(parts)} of "
                        f"{prompt_total} prompt tokens"
                    )
                logger.debug(f"Prompt token details: {details}")

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

    @staticmethod
    def _merge_leading_system_messages(messages: list[dict]) -> list[dict]:
        """Normalize ContextManager messages for OpenAI-compatible endpoints.

        - Strips "cache_hint" markers (Anthropic-only prompt-cache breakpoints;
          some providers reject unknown message fields).
        - Merges consecutive user messages into one (the heartbeat prompt splits
          stable/volatile context into two user turns for Anthropic caching;
          chat templates expect alternating roles).
        - Merges consecutive system messages at the start into one. OpenAI/
          OpenRouter automatic prompt caching is prefix-based — the cache key is
          the token sequence from the start of the request. Multiple system
          messages can break this because some providers restart the prefix
          counter at each system boundary. Merging them into a single block
          gives the largest possible stable prefix for cache hits.
        """
        normalized = []
        for msg in messages:
            if "cache_hint" in msg:
                msg = {k: v for k, v in msg.items() if k != "cache_hint"}
            prev = normalized[-1] if normalized else None
            if (prev is not None
                    and msg.get("role") == "user" and prev.get("role") == "user"
                    and isinstance(msg.get("content"), str)
                    and isinstance(prev.get("content"), str)):
                normalized[-1] = {**prev, "content": prev["content"] + "\n\n" + msg["content"]}
            else:
                normalized.append(msg)
        messages = normalized

        if not messages or messages[0].get("role") != "system":
            return messages
        system_parts = []
        rest_start = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
                rest_start = i + 1
            else:
                break
        if len(system_parts) <= 1:
            return messages
        merged = [{"role": "system", "content": "\n\n".join(system_parts)}]
        merged.extend(messages[rest_start:])
        return merged

    def _create_completion(self, **kwargs):
        """chat.completions.create with our extra_body, tolerating models that
        can't turn reasoning off.

        Always-reasoning models (e.g. Mistral Large 4 on OpenRouter) answer
        reasoning effort "none" with a 400. Drop the switch and retry once; from
        then on this client sends no reasoning setting, so the model reasons at
        its default, the closest to "off" it allows.
        """
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if not self._reasoning_off_rejected(e):
                raise
            logger.warning(
                f"{self.model_name} can't turn reasoning off — retrying without the "
                "reasoning setting (set model.reasoning: true to choose an effort)."
            )
            self._extra_body.pop("reasoning", None)
            if self._extra_body:
                kwargs["extra_body"] = self._extra_body
            else:
                kwargs.pop("extra_body", None)
            return self.client.chat.completions.create(**kwargs)

    def _reasoning_off_rejected(self, error: BadRequestError) -> bool:
        if self._extra_body.get("reasoning") != {"effort": "none"}:
            return False
        # OpenRouter wraps the provider's message in error.metadata.raw
        detail = f"{error} {getattr(error, 'body', '')}".lower()
        return "reasoning" in detail

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
        self.last_error = ""
        try:
            # Build kwargs — only include penalties if non-zero (Google API rejects them)
            kwargs = dict(
                model=self.model_name,
                messages=self._merge_leading_system_messages(messages),
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            if self.frequency_penalty:
                kwargs["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty:
                kwargs["presence_penalty"] = self.presence_penalty
            response = self._create_completion(**kwargs)

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
            self.last_error = "LLM server is not running or not reachable"
            logger.warning(self.last_error)
            return None
        except Exception as e:
            self.last_error = f"LLM call failed: {e}"
            logger.error(self.last_error)
            return None

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        skill_registry, max_rounds: int = 5, loop_mode: str = "capped",
                        autonomous: bool = False) -> tuple[Optional[str], list[str]]:
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
            loop_mode: "capped" or "unlimited"

        Returns:
            Tuple of (final response text, list of tool names used)
        """
        # Merge leading system messages for better prefix caching, then copy
        msgs = list(self._merge_leading_system_messages(messages))
        tools_used = []
        # Reset captured reasoning at the start of each call so a previous
        # call's chain of thought never leaks forward.
        self.last_reasoning = ""
        self.last_error = ""

        if loop_mode == "unlimited":
            effective_max = 999  # effectively unlimited, loop detector is the safety net
        else:
            effective_max = max_rounds

        recent_calls: list[tuple[str, str]] = []  # (tool_name, args_hash)
        loop_warned = False
        empty_nudged = False

        for round_num in range(effective_max):
            try:
                # Build kwargs — only include penalties if non-zero (Google API rejects them)
                kwargs = dict(
                    model=self.model_name,
                    messages=msgs,
                    tools=tools if tools else None,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                if self.frequency_penalty:
                    kwargs["frequency_penalty"] = self.frequency_penalty
                if self.presence_penalty:
                    kwargs["presence_penalty"] = self.presence_penalty
                response = self._create_completion(**kwargs)

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
                    cleaned = strip_think_tags(text).strip()
                    # A turn that used tools but ended with no reply means housekeeping
                    # silently replaced responding to the user (common on cold wake-ups),
                    # and a truncated one means the budget went to thinking.
                    # Nudge once to actually reply instead of dropping the message.
                    truncated = finish == "length"
                    if (
                        not cleaned
                        and (tools_used or truncated)
                        and not empty_nudged
                        and round_num + 1 < effective_max
                    ):
                        empty_nudged = True
                        logger.info(
                            f"Empty final response ({'truncated' if truncated else 'after tool work'})"
                            " — nudging model to reply."
                        )
                        msgs.append({
                            "role": "user",
                            "content": empty_reply_nudge(autonomous, truncated),
                        })
                        continue
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

                    if func_name == "search_tools" and hasattr(skill_registry, 'search_tools'):
                        found_tools, result = skill_registry.search_tools(args.get("query", ""))
                        if found_tools:
                            extend_tools_unique(tools, found_tools)
                    else:
                        result = skill_registry.execute(func_name, args)

                    result_str = str(result)
                    preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    logger.info(f"Tool result [{func_name}]: {preview}")

                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": func_name,
                        "content": result_str,
                    })

                    args_str = tc.function.arguments
                    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
                    recent_calls.append((func_name, args_hash))

                loop_warning = detect_tool_loop(recent_calls)
                if loop_warning:
                    if loop_warned:
                        logger.warning("Loop persisted after warning — forcing stop")
                        break
                    loop_warned = True
                    msgs.append({"role": "user", "content": f"[SYSTEM: {loop_warning}]"})

                if loop_mode == "capped" and round_num == effective_max - 3:
                    wind_down_msg = (
                        "⚠️ You have 2 tool rounds remaining. Wrap up your current work — "
                        "call paint_finish() if painting, or produce your final response."
                    )
                    msgs.append({"role": "user", "content": f"[SYSTEM: {wind_down_msg}]"})

            except APIConnectionError:
                self.last_error = "LLM server disconnected during tool loop"
                logger.warning(self.last_error)
                return None, tools_used
            except Exception as e:
                self.last_error = f"Tool-calling loop failed (round {round_num + 1}): {e}"
                logger.error(self.last_error)
                return None, tools_used

        # Exhausted rounds or loop force-stop — produce clean silent response
        logger.warning(f"Tool loop ended after {round_num + 1} rounds (mode={loop_mode})")

        try:
            response = self._create_completion(
                model=self.model_name,
                messages=msgs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            self._track(response)
            text = response.choices[0].message.content or ""
            cleaned = strip_think_tags(text)
            if cleaned:
                return cleaned, tools_used
        except Exception as e:
            self.last_error = f"Final response after tool loop failed: {e}"
            logger.error(self.last_error)

        logger.warning("Forced final response was empty/failed — returning clean silent")
        return '{"action": "silent", "thinking": "Tool loop limit reached, wrapping up."}', tools_used
