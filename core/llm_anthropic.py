"""
LLM client — native Anthropic API.

Uses the Anthropic Python SDK directly (not via OpenAI compatibility shim)
to get native features: prompt caching, proper tool_use format, etc.

Same interface as LLMClient so the engine can swap them transparently.
"""

import hashlib
import json
import logging
import time
from typing import Any, Optional

from anthropic import Anthropic, APIConnectionError, APIStatusError, InternalServerError

from core.llm import (
    PulseResponse,
    extract_think_content,
    strip_think_tags,
    detect_tool_loop,
    extend_tools_unique,
)

logger = logging.getLogger(__name__)

CACHE_DIAGNOSTICS_BETA = "cache-diagnosis-2026-04-07"
_MISSING = object()


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


def _convert_messages(messages: list[dict]) -> tuple[list[str], list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Key differences:
    - System prompt is a separate parameter, not a message
    - Anthropic doesn't support "system" role in messages array
    - Tool results use "tool_result" content blocks

    Returns (system_parts, anthropic_messages).
    system_parts is an ordered list so the caller can apply cache_control
    selectively (e.g. only on the first stable block).
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
            if isinstance(content, list):
                # Multi-part content (e.g. text + image) — convert from OpenAI format
                blocks = []
                for part in content:
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        # OpenAI: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                        # Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # Parse data URI: data:<media_type>;base64,<data>
                            header, b64_data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                },
                            })
                        else:
                            # URL-based image
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                },
                            })
                anthropic_msgs.append({"role": "user", "content": blocks})
            else:
                anthropic_msgs.append({"role": "user", "content": content or ""})

    return system_parts, anthropic_msgs


def _obj_get(obj: Any, name: str, default=None):
    """Read either SDK objects or plain dicts returned by tests/mocks."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict) and name in model_extra:
        return model_extra[name]
    pydantic_extra = getattr(obj, "__pydantic_extra__", None)
    if isinstance(pydantic_extra, dict) and name in pydantic_extra:
        return pydantic_extra[name]
    return getattr(obj, name, default)


class AnthropicClient:
    """Native Anthropic API client with prompt caching and tool_use support."""

    def __init__(self, endpoint: str, model_name: str = "claude-sonnet-4-20250514",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 top_p: float = 1.0, top_k: int | None = None,
                 api_key: str = "", usage_tracker=None,
                 reasoning: bool = False, provider_type: str = "anthropic",
                 cache_ttl: str = "5m", cache_automatic: bool = False,
                 cache_diagnostics: bool = False):
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Anthropic recommends altering either temperature or top_p, not both.
        # We only forward top_p when it's been explicitly set (non-default).
        self.top_p = top_p
        # Anthropic natively supports top_k; only forwarded when explicitly set.
        self.top_k = top_k
        self._available = None
        self._usage = usage_tracker
        self._provider_name = "anthropic"
        self._enable_caching = True  # prompt caching for cost savings
        # Cache TTL: "5m" (default) or "1h". The 1h tier costs 2x to write
        # vs 1.25x for 5m, but saves the rewrite cost across longer gaps.
        # Pays off if a written cache is read at least one extra time —
        # ideal for personas with heartbeat intervals ≤60 min.
        self._cache_ttl = cache_ttl if cache_ttl in ("5m", "1h") else "5m"
        self._cache_automatic = bool(cache_automatic)
        self._cache_diagnostics = bool(cache_diagnostics)
        self._diagnostics_disabled = False
        self._diagnostics_previous_ids: dict[str, str] = {}

        # Captured reasoning from the most recent call. Mirrors LLMClient's
        # interface so the engine can read either client uniformly. For now
        # we only capture from <think>...</think> tags in text content;
        # native Anthropic thinking blocks (which require a separate
        # request param + tool-loop passthrough) are a follow-up.
        self.last_reasoning: str = ""
        # Last error message from a failed call — read by the Telegram
        # channel to surface specific reasons ("API 500", "disconnected",
        # etc.) instead of a generic "LLM didn't respond" message.
        self.last_error: str = ""

    def _cache_control(self) -> dict:
        control = {"type": "ephemeral"}
        if self._cache_ttl == "1h":
            control["ttl"] = "1h"
        return control

    def _create_kwargs(self, *, system, messages, tools=None,
                       automatic_cache: bool = False) -> dict:
        """Build Anthropic request kwargs without deprecated sampler params.

        Newer Claude models reject some generation controls such as
        temperature. Keep native Anthropic calls conservative: max_tokens is
        required, while model-specific tuning stays out of the request.
        """
        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if self._enable_caching and automatic_cache:
            kwargs["cache_control"] = self._cache_control()
        return kwargs

    def _track(self, response):
        """Log token usage from an Anthropic API response."""
        usage = _obj_get(response, "usage")
        if self._usage and usage:
            cache_read = _obj_get(usage, "cache_read_input_tokens", 0) or 0
            cache_create = _obj_get(usage, "cache_creation_input_tokens", 0) or 0
            cache_creation = _obj_get(usage, "cache_creation")
            cache_5m = _obj_get(cache_creation, "ephemeral_5m_input_tokens", 0) or 0
            cache_1h = _obj_get(cache_creation, "ephemeral_1h_input_tokens", 0) or 0
            if not cache_create and (cache_5m or cache_1h):
                cache_create = cache_5m + cache_1h

            self._usage.record(
                prompt_tokens=_obj_get(usage, "input_tokens", 0) or 0,
                completion_tokens=_obj_get(usage, "output_tokens", 0) or 0,
                provider=self._provider_name,
                model=self.model_name,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_create,
                cache_creation_5m_input_tokens=cache_5m,
                cache_creation_1h_input_tokens=cache_1h,
            )
            # Log cache stats if available
            if cache_read or cache_create:
                # Anthropic's input_tokens excludes cached tokens, so total =
                # input_tokens (uncached) + cache_read + cache_create
                input_tokens = _obj_get(usage, "input_tokens", 0) or 0
                total = input_tokens + cache_read + cache_create
                hit_pct = cache_read * 100 // total if total else 0
                breakdown = ""
                if cache_5m or cache_1h:
                    breakdown = f" ({cache_5m} 5m, {cache_1h} 1h)"
                logger.info(
                    f"Prompt cache: {cache_read} read, {cache_create} created "
                    f"{breakdown} of {total} total tokens ({hit_pct}% hit)"
                )

    def _diagnostics_key(self, system) -> str:
        """Group related calls for Anthropic cache diagnostics comparisons."""
        if isinstance(system, list) and system:
            first = _obj_get(system[0], "text", "")
        else:
            first = system or ""
        payload = json.dumps(
            {"model": self.model_name, "stable_system": first},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _diagnostics_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return (
            "cache-diagnosis" in text
            or "diagnostic" in text
            or "betas" in text
            or "beta" in text
        )

    def _log_cache_diagnostics(self, response, previous_id: str | None) -> None:
        diagnostics = _obj_get(response, "diagnostics", _MISSING)
        usage = _obj_get(response, "usage")
        cache_read = _obj_get(usage, "cache_read_input_tokens", 0) or 0

        if diagnostics is _MISSING:
            logger.info(
                "Anthropic cache diagnostics: response field absent "
                "(beta header may not have been accepted)"
            )
            return

        if diagnostics is None:
            if previous_id:
                logger.info(
                    "Anthropic cache diagnostics: no divergence detected "
                    f"(cache_read_input_tokens={cache_read})"
                )
            else:
                logger.info(
                    "Anthropic cache diagnostics: baseline captured; "
                    "next matching call can be compared"
                )
            return

        reason = _obj_get(diagnostics, "cache_miss_reason")
        if reason is None:
            logger.info("Anthropic cache diagnostics: comparison pending/inconclusive")
            return

        reason_type = _obj_get(reason, "type", "unknown")
        missed_tokens = _obj_get(reason, "cache_missed_input_tokens", 0) or 0
        logger.info(
            "Anthropic cache diagnostics: "
            f"{reason_type} (~{missed_tokens} missed input tokens)"
        )

    def _create_message_with_diagnostics(self, create_kwargs: dict, previous_id: str | None):
        """Send diagnostics using typed beta params or the SDK escape hatch."""
        try:
            beta_messages = getattr(getattr(self.client, "beta", None), "messages", None)
            if beta_messages is None:
                raise AttributeError("Anthropic SDK has no beta.messages API")
            return beta_messages.create(
                **create_kwargs,
                diagnostics={"previous_message_id": previous_id},
                betas=[CACHE_DIAGNOSTICS_BETA],
            )
        except AttributeError as e:
            logger.debug(f"Anthropic typed cache diagnostics unavailable: {e}")
        except TypeError as e:
            if not self._diagnostics_error(e):
                raise
            logger.debug(f"Anthropic typed cache diagnostics unsupported by SDK: {e}")

        extra_headers = dict(create_kwargs.pop("extra_headers", {}) or {})
        existing_beta = extra_headers.get("anthropic-beta") or extra_headers.get("Anthropic-Beta")
        if existing_beta:
            betas = [part.strip() for part in existing_beta.split(",") if part.strip()]
            if CACHE_DIAGNOSTICS_BETA not in betas:
                betas.append(CACHE_DIAGNOSTICS_BETA)
            extra_headers["anthropic-beta"] = ",".join(betas)
        else:
            extra_headers["anthropic-beta"] = CACHE_DIAGNOSTICS_BETA

        extra_body = dict(create_kwargs.pop("extra_body", {}) or {})
        extra_body["diagnostics"] = {"previous_message_id": previous_id}

        return self.client.messages.create(
            **create_kwargs,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

    def _create_message(self, create_kwargs: dict, diagnostics_key: str | None = None):
        """Create a message, optionally adding Anthropic cache diagnostics."""
        if (
            not self._cache_diagnostics
            or self._diagnostics_disabled
            or not diagnostics_key
        ):
            return self.client.messages.create(**create_kwargs)

        previous_id = self._diagnostics_previous_ids.get(diagnostics_key)
        try:
            response = self._create_message_with_diagnostics(dict(create_kwargs), previous_id)
        except TypeError as e:
            if not self._diagnostics_error(e):
                raise
            self._diagnostics_disabled = True
            logger.warning(f"Anthropic cache diagnostics unsupported by installed SDK; retrying normally: {e}")
            return self.client.messages.create(**create_kwargs)
        except APIStatusError as e:
            if not self._diagnostics_error(e):
                raise
            self._diagnostics_disabled = True
            logger.warning(f"Anthropic cache diagnostics rejected by API; retrying normally: {e}")
            return self.client.messages.create(**create_kwargs)

        response_id = _obj_get(response, "id")
        if response_id:
            self._diagnostics_previous_ids[diagnostics_key] = response_id
        self._log_cache_diagnostics(response, previous_id)
        return response

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

    def _build_system_blocks(self, system_parts: list[str]) -> list[dict]:
        """Build system prompt as content blocks with selective caching.

        Only the FIRST block (stable persona/instructions) gets cache_control.
        Subsequent blocks (dynamic context: time, memories, journal) are left
        uncached so they don't bust the prefix hash every call.
        """
        if not system_parts:
            return []

        blocks = []
        for i, text in enumerate(system_parts):
            if not text:
                continue
            block = {"type": "text", "text": text}
            if self._enable_caching and i == 0:
                block["cache_control"] = self._cache_control()
            blocks.append(block)
        return blocks

    def chat(self, messages: list[dict]) -> Optional[PulseResponse]:
        """Send messages and get a parsed PulseResponse.

        Used for heartbeat ticks and scheduled tasks (expects JSON format).
        """
        self.last_error = ""
        try:
            system_parts, anthropic_msgs = _convert_messages(messages)
            system_blocks = self._build_system_blocks(system_parts)
            diagnostics_key = self._diagnostics_key(system_blocks)

            create_kwargs = self._create_kwargs(
                system=system_blocks,
                messages=anthropic_msgs,
            )
            # Retry loop for transient 500 errors
            response = None
            for attempt in range(3):
                try:
                    response = self._create_message(create_kwargs, diagnostics_key)
                    break  # Success!
                except InternalServerError as e:
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)  # 2s, 4s
                        logger.warning(f"Anthropic 500 (attempt {attempt + 1}/3), retrying in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        self.last_error = f"Anthropic API error after 3 retries: {e}"
                        logger.error(self.last_error)
                        return None

            if not response:
                return None

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
            self.last_error = "Anthropic API not reachable"
            logger.warning(self.last_error)
            return None
        except Exception as e:
            self.last_error = f"Anthropic call failed: {e}"
            logger.error(self.last_error)
            return None

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        skill_registry, max_rounds: int = 5, loop_mode: str = "capped") -> tuple[Optional[str], list[str]]:
        """Chat with native Anthropic tool_use support.

        Implements the same agentic loop as LLMClient but using
        Anthropic's native tool_use/tool_result content blocks.

        Returns:
            Tuple of (final response text, list of tool names used)
        """
        system_parts, anthropic_msgs = _convert_messages(messages)
        anthropic_tools = _openai_tools_to_anthropic(tools) if tools else []
        tools_used = []
        # Reset captured reasoning and error per call so a previous chain
        # of thought / error never leaks forward into the next reply.
        self.last_error = ""
        self.last_reasoning = ""

        if loop_mode == "unlimited":
            effective_max = 999
        else:
            effective_max = max_rounds

        recent_calls: list[tuple[str, str]] = []
        loop_warned = False

        system_blocks = self._build_system_blocks(system_parts)
        diagnostics_key = self._diagnostics_key(system_blocks)

        for round_num in range(effective_max):
            try:
                create_kwargs = self._create_kwargs(
                    system=system_blocks,
                    messages=anthropic_msgs,
                    tools=anthropic_tools if anthropic_tools else None,
                    automatic_cache=self._cache_automatic,
                )
                
                # Retry loop for transient 500 errors
                response = None
                for attempt in range(3):
                    try:
                        response = self._create_message(create_kwargs, diagnostics_key)
                        break  # Success!
                    except InternalServerError as e:
                        if attempt < 2:
                            wait = 2 ** (attempt + 1)  # 2s, 4s
                            logger.warning(f"Anthropic 500 (attempt {attempt + 1}/3), retrying in {wait}s: {e}")
                            time.sleep(wait)
                        else:
                            logger.error(f"Anthropic 500 after 3 attempts: {e}")
                            return None, tools_used

                if not response:
                    return None, tools_used

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

                    if func_name == "search_tools" and hasattr(skill_registry, 'search_tools'):
                        found_tools, result = skill_registry.search_tools(args.get("query", ""))
                        if found_tools:
                            added_tools = extend_tools_unique(tools, found_tools)
                            anthropic_tools.extend(_openai_tools_to_anthropic(added_tools))
                    else:
                        result = skill_registry.execute(func_name, args)

                    result_str = str(result)
                    preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    logger.info(f"Tool result [{func_name}]: {preview}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_str,
                    })

                    args_str = json.dumps(args)
                    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
                    recent_calls.append((func_name, args_hash))

                loop_warning = detect_tool_loop(recent_calls)
                if loop_warning:
                    if loop_warned:
                        logger.warning("Loop persisted after warning — forcing stop")
                        break
                    loop_warned = True
                    tool_results.append({"type": "text", "text": f"[SYSTEM: {loop_warning}]"})

                if loop_mode == "capped" and round_num == effective_max - 3:
                    wind_down_msg = (
                        "⚠️ You have 2 tool rounds remaining. Wrap up your current work — "
                        "call paint_finish() if painting, or produce your final response."
                    )
                    tool_results.append({"type": "text", "text": f"[SYSTEM: {wind_down_msg}]"})

                # Add tool results as a user message
                anthropic_msgs.append({"role": "user", "content": tool_results})

            except APIConnectionError:
                self.last_error = "Anthropic API disconnected during tool loop"
                logger.warning(self.last_error)
                return None, tools_used
            except Exception as e:
                self.last_error = f"Anthropic tool loop failed (round {round_num + 1}): {e}"
                logger.error(self.last_error)
                return None, tools_used

        # Exhausted rounds or loop force-stop — produce clean silent response
        logger.warning(f"Tool loop ended after {round_num + 1} rounds (mode={loop_mode})")

        try:
            create_kwargs = self._create_kwargs(
                system=system_blocks,
                messages=anthropic_msgs,
                automatic_cache=self._cache_automatic,
            )

            # Retry loop for transient 500 errors
            response = None
            for attempt in range(3):
                try:
                    response = self._create_message(create_kwargs, diagnostics_key)
                    break
                except InternalServerError as e:
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Anthropic 500 in final response (attempt {attempt + 1}/3), retrying in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        self.last_error = f"Anthropic 500 in final response after 3 retries: {e}"
                        logger.error(self.last_error)
                        return '{"action": "silent", "thinking": "Tool loop limit reached, wrapping up."}', tools_used

            if not response:
                return '{"action": "silent", "thinking": "Tool loop limit reached, wrapping up."}', tools_used
                
            self._track(response)
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            cleaned = strip_think_tags(text)
            if cleaned:
                return cleaned, tools_used
        except Exception as e:
            self.last_error = f"Final response after tool loop failed: {e}"
            logger.error(self.last_error)

        logger.warning("Forced final response was empty/failed — returning clean silent")
        return '{"action": "silent", "thinking": "Tool loop limit reached, wrapping up."}', tools_used
