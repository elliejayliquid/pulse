"""
LLM client — native Anthropic API.

Uses the Anthropic Python SDK directly (not via OpenAI compatibility shim)
to get native features: prompt caching, proper tool_use format, etc.

Same interface as LLMClient so the engine can swap them transparently.
"""

import json
import logging
from typing import Optional

from anthropic import Anthropic, APIConnectionError, APIStatusError

from core.llm import PulseResponse, extract_think_content, strip_think_tags

logger = logging.getLogger(__name__)


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Anthropic format.

    OpenAI:  {"type": "function", "function": {"name", "description", "parameters"}}
    Anthropic: {"name", "description", "input_schema"}
    """
    converted = []
    for tool in tools:
        func = tool.get("function", {})
        converted.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Key differences:
    - System prompt is a separate parameter, not a message
    - Anthropic doesn't support "system" role in messages array
    - Tool results use "tool_result" content blocks

    Returns (system_prompt, anthropic_messages).
    """
    system_parts = []
    anthropic_msgs = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content)

        elif role == "assistant":
            # Check if this assistant message has tool_calls (OpenAI format)
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # Convert to Anthropic content blocks
                blocks = []
                # Include any text content first
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    try:
                        tool_input = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tool_input = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": tool_input,
                    })
                anthropic_msgs.append({"role": "assistant", "content": blocks})
            else:
                anthropic_msgs.append({"role": "assistant", "content": content or ""})

        elif role == "tool":
            # OpenAI tool results → Anthropic tool_result content block
            # Anthropic requires tool_result to be in a "user" role message
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content or "",
            }
            # Merge consecutive tool results into a single user message
            if (anthropic_msgs and
                    anthropic_msgs[-1].get("role") == "user" and
                    isinstance(anthropic_msgs[-1].get("content"), list) and
                    anthropic_msgs[-1]["content"] and
                    anthropic_msgs[-1]["content"][-1].get("type") == "tool_result"):
                anthropic_msgs[-1]["content"].append(tool_result_block)
            else:
                anthropic_msgs.append({
                    "role": "user",
                    "content": [tool_result_block],
                })

        elif role == "user":
            anthropic_msgs.append({"role": "user", "content": content or ""})

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, anthropic_msgs


class AnthropicClient:
    """Native Anthropic API client with prompt caching and tool_use support."""

    def __init__(self, endpoint: str, model_name: str = "claude-sonnet-4-20250514",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 api_key: str = "", usage_tracker=None,
                 reasoning: bool = False, provider_type: str = "anthropic",
                 cache_ttl: str = "5m"):
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._available = None
        self._usage = usage_tracker
        self._provider_name = "anthropic"
        self._enable_caching = True  # prompt caching for cost savings
        # Cache TTL: "5m" (default) or "1h". The 1h tier costs 2x to write
        # vs 1.25x for 5m, but saves the rewrite cost across longer gaps.
        # Pays off if a written cache is read at least one extra time —
        # ideal for personas with heartbeat intervals ≤60 min.
        self._cache_ttl = cache_ttl if cache_ttl in ("5m", "1h") else "5m"

        # Captured reasoning from the most recent call. Mirrors LLMClient's
        # interface so the engine can read either client uniformly. For now
        # we only capture from <think>...</think> tags in text content;
        # native Anthropic thinking blocks (which require a separate
        # request param + tool-loop passthrough) are a follow-up.
        self.last_reasoning: str = ""

    def _track(self, response):
        """Log token usage from an Anthropic API response."""
        if self._usage and hasattr(response, "usage") and response.usage:
            self._usage.record(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                provider=self._provider_name,
                model=self.model_name,
            )
            # Log cache stats if available
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            if cache_read or cache_create:
                # Anthropic's input_tokens excludes cached tokens, so total =
                # input_tokens (uncached) + cache_read + cache_create
                total = (response.usage.input_tokens or 0) + cache_read + cache_create
                hit_pct = cache_read * 100 // total if total else 0
                logger.info(
                    f"Prompt cache: {cache_read} read, {cache_create} created "
                    f"of {total} total tokens ({hit_pct}% hit)"
                )

    def is_available(self) -> bool:
        """Check if the Anthropic API is reachable."""
        try:
            # Lightweight check — just see if we can hit the API
            self.client.messages.create(
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            self._available = True
            return True
        except (APIConnectionError, APIStatusError, Exception) as e:
            logger.debug(f"Anthropic API not available: {e}")
            self._available = False
            return False

    def _build_system_blocks(self, system_text: str) -> list[dict]:
        """Build system prompt as content blocks with optional caching.

        Marks the system prompt for prompt caching — Anthropic caches
        the prefix up to the cache_control breakpoint, giving ~90% cost
        reduction on repeated calls with the same system prompt.
        """
        if not system_text:
            return []

        block = {"type": "text", "text": system_text}
        if self._enable_caching:
            cache_control = {"type": "ephemeral"}
            if self._cache_ttl == "1h":
                cache_control["ttl"] = "1h"
            block["cache_control"] = cache_control
        return [block]

    def chat(self, messages: list[dict]) -> Optional[PulseResponse]:
        """Send messages and get a parsed PulseResponse.

        Used for heartbeat ticks and scheduled tasks (expects JSON format).
        """
        try:
            system_text, anthropic_msgs = _convert_messages(messages)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self._build_system_blocks(system_text),
                messages=anthropic_msgs,
                temperature=self.temperature,
            )

            self._track(response)
            stop = response.stop_reason or "unknown"

            # Extract text from content blocks
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            logger.info(f"Anthropic response ({len(text)} chars, stop={stop})")
            logger.debug(f"Raw response: {text[:200]}...")

            if not text.strip():
                logger.info("Empty response — treating as silent.")
                return PulseResponse(raw="", action="silent")

            return PulseResponse.from_llm_output(text)

        except APIConnectionError:
            logger.warning("Anthropic API not reachable")
            return None
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            return None

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        skill_registry, max_rounds: int = 5) -> tuple[Optional[str], list[str]]:
        """Chat with native Anthropic tool_use support.

        Implements the same agentic loop as LLMClient but using
        Anthropic's native tool_use/tool_result content blocks.

        Returns:
            Tuple of (final response text, list of tool names used)
        """
        system_text, anthropic_msgs = _convert_messages(messages)
        anthropic_tools = _openai_tools_to_anthropic(tools) if tools else []
        tools_used = []
        # Reset captured reasoning per call so a previous chain of thought
        # never leaks forward into the next reply.
        self.last_reasoning = ""

        system_blocks = self._build_system_blocks(system_text)

        for round_num in range(max_rounds):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    system=system_blocks,
                    messages=anthropic_msgs,
                    tools=anthropic_tools if anthropic_tools else None,
                    temperature=self.temperature,
                )

                self._track(response)
                stop = response.stop_reason or "unknown"

                # Separate text and tool_use blocks
                text_parts = []
                tool_uses = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)

                text = "\n".join(text_parts)

                # If no tool calls, this is the final response
                if not tool_uses or stop == "end_turn":
                    logger.info(f"Anthropic final response ({len(text)} chars, round {round_num + 1}, stop={stop})")
                    self.last_reasoning = extract_think_content(text)
                    cleaned = strip_think_tags(text)
                    return (cleaned if cleaned else None), tools_used

                # Model wants to call tools
                logger.info(f"Anthropic requested {len(tool_uses)} tool call(s) (round {round_num + 1})")

                # Add the assistant's full response (text + tool_use blocks)
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                anthropic_msgs.append({"role": "assistant", "content": assistant_content})

                # Execute tools and collect results
                tool_results = []
                for tu in tool_uses:
                    func_name = tu.name
                    args = tu.input or {}
                    logger.info(f"Executing tool: {func_name}({args})")
                    tools_used.append(func_name)
                    result = skill_registry.execute(func_name, args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": str(result),
                    })

                # Add tool results as a user message
                anthropic_msgs.append({"role": "user", "content": tool_results})

            except APIConnectionError:
                logger.warning("Anthropic API disconnected during tool loop")
                return None, tools_used
            except Exception as e:
                logger.error(f"Anthropic tool loop failed (round {round_num + 1}): {e}")
                return None, tools_used

        # Exhausted max rounds — force final response without tools
        logger.warning(f"Tool loop hit max rounds ({max_rounds}), requesting final response...")
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=system_blocks,
                messages=anthropic_msgs,
                temperature=self.temperature,
            )
            self._track(response)
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            return (strip_think_tags(text) or None), tools_used
        except Exception as e:
            logger.error(f"Final response after tool loop failed: {e}")
            return None, tools_used
