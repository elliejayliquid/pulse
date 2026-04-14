"""
TTS skill — lets the companion send voice messages.

The companion decides when to speak aloud instead of (or alongside) text.
Supports two modes:
  - **Design** (audition): generates speech from a voice description in config,
    combined with per-message emotion.  Slightly different each time.
  - **Clone** (production): reproduces a locked-in reference voice consistently.
    Activated when voice_sample + voice_sample_text are set in config.

The generated audio is queued as `pending_voice` — the Telegram channel
picks it up and sends it as a voice message.
"""

import logging
import re
from pathlib import Path

from skills.base import BaseSkill

logger = logging.getLogger(__name__)

# Shared engine instance across skill reloads (one model in VRAM)
_engine = None


# Pre-compiled patterns for chat-shaped markup the TTS engine can't handle.
# Bold runs FIRST (its **markers** would otherwise be parsed as nested italics),
# then italic-asterisk, then italic-underscore.
_RE_BOLD = re.compile(r'\*\*([^*\n]+)\*\*')
_RE_ITALIC_AST = re.compile(r'\*([^*\n]+)\*')
_RE_ITALIC_UND = re.compile(r'(?<!\w)_([^_\n]+)_(?!\w)')
_RE_DOUBLE_COMMA = re.compile(r',\s*,')
_RE_WHITESPACE = re.compile(r'\s+')


def clean_for_tts(text: str) -> str:
    """Strip chat-shaped markup so the TTS engine doesn't read it literally.

    - **bold** → bold (markers removed, text kept as plain narration)
    - *action* → , action , (roleplay markers become brief pauses around
      narrated action text — audiobook-style stage directions)
    - _word_ → , word ,

    This runs ONLY at the boundary where text reaches the TTS engine. The
    original text is preserved everywhere else (conversation history,
    Telegram messages, journal entries) so the companion still sees what
    they actually wrote.
    """
    if not text:
        return text
    text = _RE_BOLD.sub(r'\1', text)
    text = _RE_ITALIC_AST.sub(r', \1,', text)
    text = _RE_ITALIC_UND.sub(r', \1,', text)
    text = _RE_DOUBLE_COMMA.sub(',', text)
    text = _RE_WHITESPACE.sub(' ', text)
    return text.strip()


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
        # Queue of (ogg_path, spoken_text) tuples — supports multiple speak()
        # calls per turn. Telegram drains the queue in order; engine joins the
        # texts into conversation history so nothing said aloud is forgotten.
        self.pending_voices: list[tuple[Path, str]] = []

        # Clone mode: reference clip + transcript
        voice_sample = tts_config.get("voice_sample", "")
        self.voice_sample: Path | None = Path(voice_sample) if voice_sample else None
        self.voice_sample_text: str | None = tts_config.get("voice_sample_text", "") or None

        if self.voice_sample:
            if not self.voice_sample.exists():
                logger.warning(f"[TTS] voice_sample not found: {self.voice_sample}")
                self.voice_sample = None
                self.voice_sample_text = None
            else:
                logger.info(f"[TTS] Clone mode — reference: {self.voice_sample}")

    @property
    def clone_mode(self) -> bool:
        return self.voice_sample is not None and self.voice_sample_text is not None

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
                        "Keep it concise — voice messages should feel spontaneous, not like a speech. "
                        "You MUST call this function — never write [speak] as text."
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

        if not self.voice_description and not self.clone_mode:
            return "Voice not configured — add tts.voice_description to config."

        engine = _get_engine()

        # Clean ONLY the copy we hand to the engine. The original `text` keeps
        # its asterisks/markdown so conversation history and the return value
        # show what the companion actually wrote.
        cleaned = clean_for_tts(text)

        # execute() runs in a worker thread (via asyncio.to_thread in engine),
        # so asyncio.run() is safe here — no event loop in this thread.
        import asyncio
        try:
            ogg_path = asyncio.run(engine.speak(
                text=cleaned,
                voice_description=self.voice_description,
                emotion=emotion,
                ref_audio_path=self.voice_sample,
                ref_text=self.voice_sample_text,
            ))
        except Exception as e:
            logger.error(f"[TTS] Generation failed: {e}", exc_info=True)
            return f"Voice generation failed: {e}"

        # Queue for Telegram to pick up (FIFO — multiple calls per turn allowed).
        # Store ORIGINAL text so conversation history reflects intent, not the
        # TTS-cleaned version.
        self.pending_voices.append((ogg_path, text))

        mode = "clone" if self.clone_mode else "design"
        logger.info(
            f"[TTS/{mode}] Queued voice message ({len(self.pending_voices)} in queue): {text[:50]}..."
        )
        #return f"Voice message generated. The listener will hear you say: \"{text}\""
        #return "Voice message delivered. No need to repeat or rephrase in text — your voice carries it."
        return f"Voice message generated & delivered. The listener will hear you say: \"{text}\". No need to repeat or rephrase this in text — your voice carries it."

    def synthesize(self, text: str, emotion: str = "") -> Path | None:
        """Generate a voice clip on demand without queuing.

        Used by channels (e.g. Telegram 🔊 button) to convert any text to
        voice without involving the companion's tool-calling loop. Returns
        the OGG path; the caller is responsible for cleanup.

        This is a sync method — call it via asyncio.to_thread from async
        contexts so it doesn't block the event loop during inference.
        """
        text = text.strip()
        if not text:
            return None
        if not self.voice_description and not self.clone_mode:
            return None

        # Clean only the engine input — the Telegram message we're synthesizing
        # from is unchanged; the user is replying to the original.
        cleaned = clean_for_tts(text)
        if not cleaned:
            return None

        engine = _get_engine()
        import asyncio
        try:
            ogg_path = asyncio.run(engine.speak(
                text=cleaned,
                voice_description=self.voice_description,
                emotion=emotion,
                ref_audio_path=self.voice_sample,
                ref_text=self.voice_sample_text,
            ))
        except Exception as e:
            logger.error(f"[TTS] On-demand synthesis failed: {e}", exc_info=True)
            return None

        mode = "clone" if self.clone_mode else "design"
        logger.info(f"[TTS/{mode}] On-demand synthesis: {cleaned[:50]}...")
        return ogg_path

    @property
    def is_configured(self) -> bool:
        """True if voice generation is possible (either design or clone mode)."""
        return bool(self.voice_description) or self.clone_mode

    def shutdown(self):
        """Free the voice model from VRAM.

        Called from pulse.py during graceful shutdown so the TTS model is
        released *before* llama-server finishes its in-flight inference,
        giving the LLM extra headroom for the wind-down. Safe to call even
        if the model was never loaded.
        """
        global _engine
        if _engine is not None:
            _engine.unload()

    async def warmup(self):
        """Preload the voice model into VRAM at startup.

        Without this, the first 🔊 tap (or first speak() call) pays a cold-start
        tax of 20-40s while Qwen3 weights stream off disk into VRAM. With this,
        the load happens in the background during Pulse startup, alongside
        llama-server warmup, so the first user-visible generation is fast.

        Safe to fail — if warmup errors out, on-demand synthesis will still
        attempt to load the model on first use. We just log and move on.
        """
        if not self.is_configured:
            return
        try:
            engine = _get_engine()
            if self.clone_mode:
                # Clone mode also builds the voice prompt from the reference
                # clip, which is the other slow first-call cost
                await engine._ensure_clone_loaded(self.voice_sample, self.voice_sample_text)
            else:
                await engine._ensure_design_loaded()
            logger.info("[TTS] Warmup complete — voice model preloaded.")
        except Exception as e:
            logger.warning(f"[TTS] Warmup failed (will retry on first synthesis): {e}")
