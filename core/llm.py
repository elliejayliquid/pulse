"""
LM Studio client - OpenAI-compatible API wrapper.

Talks to whatever model is loaded in LM Studio at localhost:1234.
If LM Studio isn't running, retries gracefully.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, APIConnectionError

logger = logging.getLogger(__name__)


@dataclass
class PulseResponse:
    """Parsed response from Nova."""
    thinking: str = ""
    action: str = "silent"         # notify | post_lor | schedule | silent
    message: str = ""
    lor_category: str = "general"
    lor_title: str = ""
    schedule_task: str = ""
    schedule_when: str = ""
    raw: str = ""

    @classmethod
    def from_llm_output(cls, text: str) -> "PulseResponse":
        """Parse structured JSON response from the LLM."""
        resp = cls(raw=text)

        # Try to extract JSON from the response
        try:
            # Handle responses that might have text before/after JSON
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(text[json_start:json_end])
            else:
                # No JSON found — treat entire response as a message
                logger.warning("No JSON found in response, treating as notification")
                resp.action = "notify"
                resp.message = text.strip()
                return resp

            resp.thinking = data.get("thinking", "")
            resp.action = data.get("action", "silent")
            resp.message = data.get("message", "")
            resp.lor_category = data.get("lor_category", "general")
            resp.lor_title = data.get("lor_title", "")

            schedule = data.get("schedule")
            if schedule and isinstance(schedule, dict):
                resp.schedule_task = schedule.get("task", "")
                resp.schedule_when = schedule.get("when", "")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # If JSON fails, treat it as a notification with the raw text
            resp.action = "notify"
            resp.message = text.strip()

        return resp


class LLMClient:
    """OpenAI-compatible client for LM Studio."""

    def __init__(self, endpoint: str, model_name: str = "default",
                 temperature: float = 0.7, max_tokens: int = 1024):
        self.client = OpenAI(
            base_url=endpoint,
            api_key="not-needed"  # LM Studio doesn't require a key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._available = None

    def is_available(self) -> bool:
        """Check if LM Studio is reachable."""
        try:
            self.client.models.list()
            self._available = True
            return True
        except (APIConnectionError, Exception) as e:
            logger.debug(f"LM Studio not available: {e}")
            self._available = False
            return False

    def chat(self, messages: list[dict]) -> Optional[PulseResponse]:
        """Send messages to the LLM and get a parsed response.

        Args:
            messages: OpenAI-format messages list
                      [{"role": "system", "content": "..."}, ...]

        Returns:
            PulseResponse or None if LM Studio is unavailable
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            text = response.choices[0].message.content
            logger.info(f"LLM response ({len(text)} chars)")
            logger.debug(f"Raw response: {text[:200]}...")

            return PulseResponse.from_llm_output(text)

        except APIConnectionError:
            logger.warning("LM Studio is not running or not reachable")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
