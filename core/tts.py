"""
Text-to-speech engine via Qwen3-TTS.

Supports two modes:
  - **Design** (audition): VoiceDesign 1.7B — generates speech from a natural
    language voice description + emotion.  Each call may sound slightly
    different.  Use this to explore voices until you find one you love.
  - **Clone** (production): Base 0.6B — generates speech that faithfully
    reproduces a short reference clip.  Consistent voice every time, lower
    VRAM, faster inference.  No per-call emotion control.

Mode is chosen automatically: if a voice reference clip is provided, clone
mode is used; otherwise design mode.

Output is OGG/Opus (Telegram voice format).
"""

import asyncio
import logging
import platform
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


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
    """Qwen3-TTS wrapper — supports both VoiceDesign and voice-clone modes."""

    def __init__(self):
        self._design_model = None   # VoiceDesign 1.7B
        self._clone_model = None    # Base 0.6B
        self._clone_prompt = None   # cached clone prompt (from reference clip)
        self._lock = asyncio.Lock()
        self._active_mode = None    # "design" or "clone"

    @property
    def is_loaded(self) -> bool:
        return self._design_model is not None or self._clone_model is not None

    # ------------------------------------------------------------------
    # Design mode (VoiceDesign 1.7B)
    # ------------------------------------------------------------------

    async def _ensure_design_loaded(self):
        if self._design_model is not None:
            return
        async with self._lock:
            if self._design_model is not None:
                return
            logger.info(f"[TTS] Loading design model {DESIGN_MODEL_ID}...")
            self._design_model = await asyncio.to_thread(self._load_design)
            self._active_mode = "design"
            logger.info("[TTS] Design model loaded.")

    @staticmethod
    def _load_design():
        import torch
        from qwen_tts import Qwen3TTSModel
        return Qwen3TTSModel.from_pretrained(
            DESIGN_MODEL_ID,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Clone mode (Base 0.6B)
    # ------------------------------------------------------------------

    async def _ensure_clone_loaded(self, ref_audio_path: Path, ref_text: str):
        """Load clone model and build the voice prompt from a reference clip."""
        if self._clone_model is not None and self._clone_prompt is not None:
            return
        async with self._lock:
            if self._clone_model is not None and self._clone_prompt is not None:
                return

            # If we had the design model loaded, unload it to free VRAM
            if self._design_model is not None:
                logger.info("[TTS] Unloading design model to make room for clone model...")
                del self._design_model
                self._design_model = None
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            logger.info(f"[TTS] Loading clone model {CLONE_MODEL_ID}...")
            self._clone_model = await asyncio.to_thread(self._load_clone)

            logger.info(f"[TTS] Building voice prompt from {ref_audio_path}...")
            self._clone_prompt = await asyncio.to_thread(
                self._build_clone_prompt, self._clone_model, ref_audio_path, ref_text
            )
            self._active_mode = "clone"
            logger.info("[TTS] Clone model + voice prompt ready.")

    @staticmethod
    def _load_clone():
        import torch
        from qwen_tts import Qwen3TTSModel
        return Qwen3TTSModel.from_pretrained(
            CLONE_MODEL_ID,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

    @staticmethod
    def _build_clone_prompt(model, ref_audio_path: Path, ref_text: str):
        """Create a reusable voice clone prompt from a reference audio file."""
        import soundfile as sf
        audio_data, sr = sf.read(str(ref_audio_path))
        return model.create_voice_clone_prompt(
            ref_audio=(audio_data, sr),
            ref_text=ref_text,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def unload(self):
        """Free GPU memory for both models."""
        unloaded = []
        if self._design_model is not None:
            del self._design_model
            self._design_model = None
            unloaded.append("design")
        if self._clone_model is not None:
            del self._clone_model
            self._clone_model = None
            self._clone_prompt = None
            unloaded.append("clone")
        if unloaded:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info(f"[TTS] Unloaded: {', '.join(unloaded)}")

    async def speak(self, text: str, voice_description: str,
                    emotion: str = "",
                    ref_audio_path: Path | None = None,
                    ref_text: str | None = None) -> Path:
        """Generate speech and return path to OGG file.

        If ref_audio_path + ref_text are provided, uses clone mode (0.6B).
        Otherwise uses design mode (1.7B) with voice_description + emotion.
        """
        if ref_audio_path and ref_text:
            return await self._speak_clone(text, ref_audio_path, ref_text)
        else:
            return await self._speak_design(text, voice_description, emotion)

    async def _speak_design(self, text: str, voice_description: str,
                            emotion: str = "") -> Path:
        """Generate via VoiceDesign — unique voice from description."""
        await self._ensure_design_loaded()

        instruct = voice_description.strip()
        if emotion:
            instruct = f"{instruct}, {emotion.strip()}"

        logger.info(f"[TTS/design] Generating: {text[:60]}... | voice: {instruct[:60]}...")

        wavs, sr = await asyncio.to_thread(
            self._design_model.generate_voice_design,
            text=text,
            language="English",
            instruct=instruct,
        )

        return await self._wavs_to_ogg(wavs[0], sr)

    async def _speak_clone(self, text: str, ref_audio_path: Path,
                           ref_text: str) -> Path:
        """Generate via voice cloning — consistent voice from reference clip."""
        await self._ensure_clone_loaded(ref_audio_path, ref_text)

        logger.info(f"[TTS/clone] Generating: {text[:60]}...")

        wavs, sr = await asyncio.to_thread(
            self._clone_model.generate_voice_clone,
            text=text,
            language="English",
            voice_clone_prompt=self._clone_prompt,
        )

        return await self._wavs_to_ogg(wavs[0], sr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _wavs_to_ogg(wav_data, sample_rate) -> Path:
        """Write WAV data to temp file, convert to OGG, return OGG path."""
        import soundfile as sf

        wav_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(wav_path), wav_data, sample_rate)

        ogg_path = wav_path.with_suffix(".ogg")
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

        # Clean up WAV
        try:
            wav_path.unlink()
        except Exception:
            pass

        logger.info(f"[TTS] Generated {ogg_path} ({ogg_path.stat().st_size / 1024:.0f}KB)")
        return ogg_path
