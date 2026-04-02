"""
Text-to-speech engine via Qwen3-TTS VoiceDesign.

Generates speech from text + voice description + emotion. Each companion
defines a voice_description in their persona; the emotion is passed per-call.
Output is OGG/Opus (Telegram voice format).

Model loading strategy: load on first speak(), keep loaded for reuse,
unload explicitly via unload() or when Pulse shuts down.
"""

import asyncio
import logging
import platform
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Qwen3-TTS VoiceDesign — 1.7B, supports voice design from text descriptions
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def _find_ffmpeg() -> Path:
    """Find ffmpeg: system PATH first, then whisper's downloaded copy."""
    system = shutil.which("ffmpeg")
    if system:
        return Path(system)
    suffix = ".exe" if platform.system() == "Windows" else ""
    whisper_ffmpeg = Path("data/whisper") / f"ffmpeg{suffix}"
    if whisper_ffmpeg.exists():
        return whisper_ffmpeg
    raise FileNotFoundError(
        "ffmpeg not found. Enable voice transcription first (downloads ffmpeg), "
        "or install ffmpeg manually."
    )


class TTSEngine:
    """Qwen3-TTS VoiceDesign wrapper — load once, generate many."""

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def ensure_loaded(self):
        """Load the TTS model if not already loaded."""
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            logger.info(f"[TTS] Loading model {MODEL_ID}...")
            self._model = await asyncio.to_thread(self._load_model)
            logger.info("[TTS] Model loaded.")

    @staticmethod
    def _load_model():
        import torch
        from qwen_tts import Qwen3TTSModel
        return Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("[TTS] Model unloaded.")

    async def speak(self, text: str, voice_description: str,
                    emotion: str = "") -> Path:
        """Generate speech and return path to OGG file.

        Args:
            text: What to say.
            voice_description: Persona's base voice description.
            emotion: Optional emotion/style instruction (e.g. "warmly", "with dry humor").

        Returns:
            Path to a temporary .ogg file ready for Telegram.
        """
        await self.ensure_loaded()

        # Combine base voice description with per-call emotion
        instruct = voice_description.strip()
        if emotion:
            instruct = f"{instruct}, {emotion.strip()}"

        logger.info(f"[TTS] Generating: {text[:60]}... | voice: {instruct[:60]}...")

        # Generate WAV in a thread (model inference is blocking)
        wavs, sr = await asyncio.to_thread(
            self._model.generate_voice_design,
            text=text,
            language="English",
            instruct=instruct,
        )

        # Write WAV to temp file
        import soundfile as sf
        wav_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(wav_path), wavs[0], sr)

        # Convert WAV -> OGG/Opus for Telegram voice messages
        ogg_path = wav_path.with_suffix(".ogg")
        await self._wav_to_ogg(wav_path, ogg_path)

        # Clean up WAV
        try:
            wav_path.unlink()
        except Exception:
            pass

        logger.info(f"[TTS] Generated {ogg_path} ({ogg_path.stat().st_size / 1024:.0f}KB)")
        return ogg_path

    @staticmethod
    async def _wav_to_ogg(wav_path: Path, ogg_path: Path):
        """Convert WAV to OGG/Opus using ffmpeg."""
        ffmpeg = _find_ffmpeg()
        proc = await asyncio.create_subprocess_exec(
            str(ffmpeg),
            "-i", str(wav_path),
            "-c:a", "libopus",
            "-b:a", "64k",
            "-y",
            str(ogg_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg WAV->OGG failed: {stderr.decode()[:500]}")
