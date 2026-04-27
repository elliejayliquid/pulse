"""
LLM client — native OpenAI Responses API.

Uses the OpenAI Responses API (/v1/responses) to support reasoning models
(o-series, gpt-5.x) alongside tool-calling, which is not supported
on the standard Chat Completions endpoint for these models.

Same interface as LLMClient so the engine can swap them transparently.
"""

import json
import logging
import time
from typing import Optional

from openai import OpenAI, APIError, APIConnectionError

from core.llm import PulseResponse, strip_think_tags

logger = logging.getLogger(__name__)


def _openai_tools_to_responses(tools: list[dict]) -> list[dict]:
    """Convert OpenAI Chat Completions tool format to Responses API format.

    Chat Completions: {"type": "function", "function": {"name", "description", "parameters"}}
    Responses API:    {"type": "function", "name", "description", "parameters"}
    """
    converted = []
    for tool in tools:
        func = tool.get("function", {})
        converted.append({
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def _convert_messages_for_responses(messages: list[dict]) -> list[dict]:
    """Convert Chat Completions messages to Responses API input format.

    Key difference: content block types use 'input_text'/'output_text'
    instead of 'text'. Messages with plain string content are fine as-is.
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # String content — pass through as-is
        if isinstance(content, str):
            converted.append(msg)
            continue

        # Array content — convert type: "text" to the right Responses type
        if isinstance(content, list):
            new_content = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                elif block.get("type") == "text":
                    # assistant → output_text, everything else → input_text
                    new_type = "output_text" if role == "assistant" else "input_text"
                    new_content.append({
                        "type": new_type,
                        "text": block.get("text", ""),
                    })
                elif block.get("type") == "image_url":
                    # Chat Completions: {"type": "image_url", "image_url": {"url": "..."}}
                    # Responses API:    {"type": "input_image", "image_url": "..."}
                    img_data = block.get("image_url", {})
                    url = img_data.get("url", "") if isinstance(img_data, dict) else str(img_data)
                    new_content.append({
                        "type": "input_image",
                        "image_url": url,
                    })
                else:
                    new_content.append(block)
            converted.append({**msg, "content": new_content})
        else:
            converted.append(msg)

    return converted


class OpenAIResponsesClient:
    """Native OpenAI Responses API client with reasoning + tool support."""

    def __init__(self, endpoint: str, model_name: str = "gpt-5.5",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 top_p: float = 1.0, frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 api_key: str = "", usage_tracker=None,
                 reasoning: bool = False, reasoning_effort: str = ""):
        # NOTE: endpoint is accepted for interface compatibility but ignored
        # as Responses API client always talks to OpenAI directly.
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self._usage = usage_tracker
        self._provider_name = "openai"
        self._available = None
        
        # Build reasoning config — summary is always "auto" when
        # reasoning is on so we always have something to show via
        # show_reasoning. The display toggle lives in the Telegram
        # channel, not here.
        self._reasoning = None
        if reasoning:
            self._reasoning = {
                "effort": reasoning_effort or "medium",
                "summary": "auto",
            }
        
        # Same public interface as LLMClient/AnthropicClient
        self.last_reasoning: str = ""
        self.last_error: str = ""

    def chat(self, messages: list[dict]) -> Optional[PulseResponse]:
        """Simple chat call (no tools) — used for heartbeats and tasks."""
        self.last_reasoning = ""
        self.last_error = ""
        try:
            create_kwargs = dict(
                model=self.model_name,
                input=_convert_messages_for_responses(messages),
                max_output_tokens=self.max_tokens,
            )
            
            # OpenAI reasoning models (o-series, gpt-5.x) reject all sampling
            # parameters (temperature, top_p, frequency_penalty, presence_penalty).
            is_reasoning_model = self.model_name.startswith(("gpt-5", "o1", "o3"))
            
            if self._reasoning:
                create_kwargs["reasoning"] = self._reasoning
                
            if not is_reasoning_model:
                create_kwargs["temperature"] = self.temperature
                if self.top_p < 1.0:
                    create_kwargs["top_p"] = self.top_p
                if self.frequency_penalty != 0.0:
                    create_kwargs["frequency_penalty"] = self.frequency_penalty
                if self.presence_penalty != 0.0:
                    create_kwargs["presence_penalty"] = self.presence_penalty

            # Retry loop — same pattern as AnthropicClient
            response = None
            for attempt in range(3):
                try:
                    response = self.client.responses.create(**create_kwargs)
                    break
                except (APIError, APIConnectionError) as e:
                    if attempt < 2:
                        wait = 2 ** (attempt + 1)  # 2s, 4s
                        logger.warning(f"OpenAI error (attempt {attempt + 1}/3), retrying in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        self.last_error = f"OpenAI API error after 3 retries: {e}"
                        logger.error(self.last_error)
                        return None

            if not response:
                return None

            self._track(response)
            text = response.output_text or ""
            self.last_reasoning = self._collect_reasoning(response)
            
            if not text.strip():
                return PulseResponse(raw="", action="silent")
            return PulseResponse.from_llm_output(text)
        except Exception as e:
            self.last_error = f"OpenAI call failed: {e}"
            logger.error(self.last_error)
            return None

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        skill_registry, max_rounds: int = 5) -> tuple[Optional[str], list[str]]:
        """Conversation call with tool-calling loop."""
        input_items = _convert_messages_for_responses(messages)
        tools_used = []
        self.last_reasoning = ""
        self.last_error = ""
        
        for round_num in range(max_rounds):
            try:
                create_kwargs = dict(
                    model=self.model_name,
                    input=input_items,
                    tools=_openai_tools_to_responses(tools),
                    max_output_tokens=self.max_tokens,
                )
                
                is_reasoning_model = self.model_name.startswith(("gpt-5", "o1", "o3"))
                
                if self._reasoning:
                    create_kwargs["reasoning"] = self._reasoning
                    
                if not is_reasoning_model:
                    create_kwargs["temperature"] = self.temperature
                    if self.top_p < 1.0:
                        create_kwargs["top_p"] = self.top_p
                    if self.frequency_penalty != 0.0:
                        create_kwargs["frequency_penalty"] = self.frequency_penalty
                    if self.presence_penalty != 0.0:
                        create_kwargs["presence_penalty"] = self.presence_penalty

                logger.info(f"Responses API kwargs (excl. input/tools): reasoning={create_kwargs.get('reasoning')}, model={create_kwargs.get('model')}")

                # Retry loop — same pattern as AnthropicClient
                response = None
                for attempt in range(3):
                    try:
                        response = self.client.responses.create(**create_kwargs)
                        break
                    except (APIError, APIConnectionError) as e:
                        if attempt < 2:
                            wait = 2 ** (attempt + 1)
                            logger.warning(f"OpenAI error (attempt {attempt + 1}/3), retrying in {wait}s: {e}")
                            time.sleep(wait)
                        else:
                            self.last_error = f"OpenAI API error after 3 retries: {e}"
                            logger.error(self.last_error)
                            return None, tools_used

                if not response:
                    return None, tools_used

                self._track(response)

                # Check for function calls in output
                function_calls = [item for item in response.output 
                                 if item.type == "function_call"]
                
                if not function_calls:
                    # Final text response
                    text = response.output_text or ""
                    self.last_reasoning = self._collect_reasoning(response)
                    return (strip_think_tags(text) or None), tools_used
                
                # Append ALL output items (reasoning + function_calls + any
                # text) to input for the next round. The Responses API
                # requires that reasoning items accompany their paired
                # function_call items — dropping them causes a 400 error.
                input_items.extend(response.output)

                # Execute tools and feed results back
                for fc in function_calls:
                    try:
                        args = json.loads(fc.arguments)
                    except json.JSONDecodeError:
                        args = {}
                        logger.warning(f"Failed to parse tool arguments for {fc.name}: {fc.arguments}")
                    tools_used.append(fc.name)
                    result = skill_registry.execute(fc.name, args)
                    
                    # Append tool result for this function call
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": str(result),
                    })

            except Exception as e:
                self.last_error = f"OpenAI tool loop failed (round {round_num + 1}): {e}"
                logger.error(self.last_error)
                return None, tools_used

        # Exhausted max rounds — force final response without tools
        logger.warning(f"Tool loop hit max rounds ({max_rounds}), requesting final response...")
        try:
            # Omit sampling params here too
            kwargs = dict(
                model=self.model_name,
                input=input_items,
                max_output_tokens=self.max_tokens,
            )
            is_reasoning_model = self.model_name.startswith(("gpt-5", "o1", "o3"))
            if not is_reasoning_model:
                kwargs["temperature"] = self.temperature
                if self.top_p < 1.0:
                    kwargs["top_p"] = self.top_p
                if self.frequency_penalty != 0.0:
                    kwargs["frequency_penalty"] = self.frequency_penalty
                if self.presence_penalty != 0.0:
                    kwargs["presence_penalty"] = self.presence_penalty
                    
            response = self.client.responses.create(**kwargs)
            self._track(response)
            text = response.output_text or ""
            return (strip_think_tags(text) or None), tools_used
        except Exception as e:
            self.last_error = f"Final response after tool loop failed: {e}"
            logger.error(self.last_error)
            return None, tools_used

    def _collect_reasoning(self, response) -> str:
        """Extract reasoning summaries from Responses API output items."""
        parts = []
        item_types = [getattr(item, "type", "?") for item in response.output]
        logger.info(f"Response output items: {item_types}")
        for item in response.output:
            if getattr(item, "type", None) == "reasoning":
                summaries = getattr(item, "summary", [])
                logger.info(f"Reasoning item found, summaries: {summaries}")
                for s in summaries:
                    text = getattr(s, "text", "")
                    if text:
                        parts.append(text.strip())
        result = "\n\n".join(parts)
        if result:
            logger.info(f"Collected reasoning summary: {len(result)} chars")
        else:
            logger.info("No reasoning summary found in response")
        return result

    def _track(self, response):
        """Record token usage if a tracker is provided."""
        if self._usage and hasattr(response, "usage") and response.usage:
            self._usage.record(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                provider=self._provider_name,
                model=self.model_name,
            )
            # Reasoning tokens are separate from output tokens
            reasoning_tokens = getattr(response.usage, "reasoning_tokens", 0)
            if reasoning_tokens:
                logger.info(f"Reasoning tokens: {reasoning_tokens}")
            # Log prompt cache stats (OpenAI automatic caching)
            # Responses API: input_tokens_details.cached_tokens
            details = getattr(response.usage, "input_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            if cached:
                input_tokens = response.usage.input_tokens or 1
                logger.info(
                    f"Prompt cache: {cached} cached of "
                    f"{input_tokens} input tokens "
                    f"({cached * 100 // input_tokens}% hit)"
                )

    def is_available(self) -> bool:
        """Check if the API is reachable."""
        try:
            self.client.models.list()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False
