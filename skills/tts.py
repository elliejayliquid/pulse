"""
TTS skill — lets the companion send voice messages.

The companion decides when to speak aloud instead of (or alongside) text.
Uses Qwen3-TTS VoiceDesign to generate speech from a voice description
defined in config, combined with per-message emotion/style.

The generated audio is queued as `pending_voice` — the Telegram channel
picks it up and sends it as a voice message. The text is preserved in
conversation context but NOT sent as a visible text message.
"""

import logging
from pathlib import Path

from skills.base import BaseSkill

logger = logging.getLogger(__name__)

# Shared engine instance across skill reloads (one model in VRAM)
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from core.tts import TTSEngine
        _engine = TTSEngine()
    return _engine


class TTSSkill(BaseSkill):
    name = "tts"

    def __init__(self, config: dict):
        super().__init__(config)
        tts_config = config.get("tts", {})
        self.voice_description = tts_config.get("voice_description", "")
        self.pending_voice: Path | None = None  # OGG path for Telegram to send
        self.pending_voice_text: str | None = None  # text to log in conversation

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "speak",
                    "description": (
                        "Send a voice message instead of text. Use this when a voice "
                        "note feels more natural or personal — a warm greeting, a reaction, "
                        "something playful, or a moment that deserves more than text. "
                        "The listener will ONLY hear your voice (no text shown), so make "
                        "sure your words carry the full meaning. Keep it concise — voice "
                        "messages should feel spontaneous, not like a speech."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "What to say aloud. Write naturally, as you'd actually speak."
                            },
                            "emotion": {
                                "type": "string",
                                "description": (
                                    "How to say it — your tone and feeling. Examples: "
                                    "'warmly', 'with dry humor', 'gently', 'excited', "
                                    "'teasing', 'softly and fond'. Leave empty for neutral."
                                )
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name != "speak":
            return f"Unknown tool: {tool_name}"

        text = arguments.get("text", "").strip()
        if not text:
            return "Nothing to say."

        emotion = arguments.get("emotion", "")

        if not self.voice_description:
            return "Voice not configured — add tts.voice_description to config."

        engine = _get_engine()

        # execute() runs in a worker thread (via asyncio.to_thread in engine),
        # so asyncio.run() is safe here — no event loop in this thread.
        import asyncio
        try:
            ogg_path = asyncio.run(engine.speak(text, self.voice_description, emotion))
        except Exception as e:
            logger.error(f"[TTS] Generation failed: {e}", exc_info=True)
            return f"Voice generation failed: {e}"

        # Queue for Telegram to pick up
        self.pending_voice = ogg_path
        self.pending_voice_text = text

        logger.info(f"[TTS] Queued voice message: {text[:50]}...")
        return f"Voice message generated. The listener will hear you say: \"{text}\""
