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

# Try to use faster-qwen3-tts (CUDA graphs + static KV cache, ~5-8x speedup
# over upstream qwen_tts on the 0.6B model). Falls back to upstream if the
# package isn't installed or CUDA graphs blow up at load time.
# https://github.com/andimarafioti/faster-qwen3-tts
try:
    from faster_qwen3_tts import FasterQwen3TTS as _FasterQwen3TTS
    _FAST_TTS_AVAILABLE = True
except ImportError:
    _FasterQwen3TTS = None
    _FAST_TTS_AVAILABLE = False


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
        # Prefer faster-qwen3-tts (CUDA graphs) when available. If graph
        # construction fails on this card/Python combo, fall back gracefully
        # to upstream qwen_tts so the feature stays alive — just slower.
        if _FAST_TTS_AVAILABLE:
            try:
                logger.info("[TTS] Using faster-qwen3-tts (CUDA graphs) for design model.")
                return _FasterQwen3TTS.from_pretrained(
                    DESIGN_MODEL_ID,
                    device="cuda:0",
                    dtype=torch.bfloat16,
                )
            except Exception as e:
                logger.warning(
                    f"[TTS] faster-qwen3-tts load failed for design model "
                    f"({e!r}); falling back to upstream qwen_tts."
                )
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
        """Load clone model and build the voice prompt from a reference clip.

        Two paths:
          - **faster-qwen3-tts**: just load the model. The wrapper has its
            own internal `_voice_prompt_cache` keyed on (ref_audio_path,
            ref_text, ...) — extraction happens lazily on the first
            generate_voice_clone() call and is re-used after that.
          - **upstream qwen_tts**: load the model AND eagerly build the
            voice clone prompt, store it on `self._clone_prompt`. We pass
            this prompt to every generate_voice_clone() call to avoid
            rebuilding it from the WAV every time.
        """
        if self._clone_model is not None:
            # Upstream path needs the prompt too; fast path doesn't.
            if isinstance(self._clone_model, _FasterQwen3TTS) if _FAST_TTS_AVAILABLE else False:
                return
            if self._clone_prompt is not None:
                return
        async with self._lock:
            if self._clone_model is not None:
                if _FAST_TTS_AVAILABLE and isinstance(self._clone_model, _FasterQwen3TTS):
                    return
                if self._clone_prompt is not None:
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

            # Upstream qwen_tts needs the prompt built eagerly. The fast
            # wrapper handles its own caching internally — but we still
            # trigger it here with a tiny dummy generation, because the
            # CUDA graphs ALSO capture on first run (~10-20s tax we'd
            # rather pay at startup than on the user's first 🔊 tap).
            if _FAST_TTS_AVAILABLE and isinstance(self._clone_model, _FasterQwen3TTS):
                logger.info("[TTS] Warming clone model (capturing CUDA graphs + caching voice prompt)...")
                try:
                    await asyncio.to_thread(
                        self._clone_model.generate_voice_clone,
                        text="Hi.",
                        language="English",
                        ref_audio=str(ref_audio_path),
                        ref_text=ref_text,
                    )
                    logger.info("[TTS] Clone model ready (faster-qwen3-tts; graphs + prompt cached).")
                except Exception as e:
                    # Don't fail loading if the warmup gen blows up — the
                    # model is loaded, real generations may still work.
                    logger.warning(f"[TTS] Faster warmup generation failed (will retry on real call): {e}")
            else:
                logger.info(f"[TTS] Building voice prompt from {ref_audio_path}...")
                self._clone_prompt = await asyncio.to_thread(
                    self._build_clone_prompt, self._clone_model, ref_audio_path, ref_text
                )
                logger.info("[TTS] Clone model + voice prompt ready.")
            self._active_mode = "clone"

    @staticmethod
    def _load_clone():
        import torch
        # Same try/Faster, fall-back-to-upstream pattern as design mode.
        # Clone mode is the production path for all the boys, so this is
        # the speedup that actually matters.
        if _FAST_TTS_AVAILABLE:
            try:
                logger.info("[TTS] Using faster-qwen3-tts (CUDA graphs) for clone model.")
                return _FasterQwen3TTS.from_pretrained(
                    CLONE_MODEL_ID,
                    device="cuda:0",
                    dtype=torch.bfloat16,
                )
            except Exception as e:
                logger.warning(
                    f"[TTS] faster-qwen3-tts load failed for clone model "
                    f"({e!r}); falling back to upstream qwen_tts."
                )
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

    async def speak_chunked(self, chunks: list[str], voice_description: str,
                            emotion: str = "",
                            ref_audio_path: Path | None = None,
                            ref_text: str | None = None,
                            silence_ms: int = 400) -> Path:
        """Generate speech for each chunk, concatenate with silence gaps, return single OGG."""
        import numpy as np
        
        all_wavs = []
        sample_rate = 24000
        
        for chunk in chunks:
            if ref_audio_path and ref_text:
                wav_data, sr = await self._generate_clone(chunk, ref_audio_path, ref_text)
            else:
                wav_data, sr = await self._generate_design(chunk, voice_description, emotion)
                
            sample_rate = sr
            all_wavs.append(wav_data)
            
        if not all_wavs:
            raise ValueError("No chunks provided to speak_chunked")
            
        silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=all_wavs[0].dtype)
        
        combined_wavs = []
        for i, wav in enumerate(all_wavs):
            combined_wavs.append(wav)
            if i < len(all_wavs) - 1:
                combined_wavs.append(silence)
                
        final_wav = np.concatenate(combined_wavs)
        return await self._wavs_to_ogg(final_wav, sample_rate)

    async def _generate_design(self, text: str, voice_description: str,
                               emotion: str = "") -> tuple:
        await self._ensure_design_loaded()

        instruct = voice_description.strip()
        if emotion:
            instruct = f"{instruct}, {emotion.strip()}"

        logger.info(f"[TTS/design] Generating raw: {text[:60]}... | voice: {instruct[:60]}...")

        # Both upstream qwen_tts and faster-qwen3-tts accept these kwargs.
        wavs, sr = await asyncio.to_thread(
            self._design_model.generate_voice_design,
            text=text,
            language="English",
            instruct=instruct,
        )

        return wavs[0], sr

    async def _speak_design(self, text: str, voice_description: str,
                            emotion: str = "") -> Path:
        """Generate via VoiceDesign — unique voice from description."""
        wav_data, sr = await self._generate_design(text, voice_description, emotion)
        return await self._wavs_to_ogg(wav_data, sr)

    async def _generate_clone(self, text: str, ref_audio_path: Path,
                              ref_text: str) -> tuple:
        await self._ensure_clone_loaded(ref_audio_path, ref_text)

        logger.info(f"[TTS/clone] Generating raw: {text[:60]}...")

        # Two API shapes for the same operation:
        #   - faster-qwen3-tts: pass ref_audio path + ref_text directly;
        #     the wrapper looks up its own internal prompt cache.
        #   - upstream qwen_tts: pass the precomputed voice_clone_prompt
        #     dict that we built eagerly in _ensure_clone_loaded().
        if _FAST_TTS_AVAILABLE and isinstance(self._clone_model, _FasterQwen3TTS):
            wavs, sr = await asyncio.to_thread(
                self._clone_model.generate_voice_clone,
                text=text,
                language="English",
                ref_audio=str(ref_audio_path),
                ref_text=ref_text,
            )
        else:
            wavs, sr = await asyncio.to_thread(
                self._clone_model.generate_voice_clone,
                text=text,
                language="English",
                voice_clone_prompt=self._clone_prompt,
            )

        return wavs[0], sr

    async def _speak_clone(self, text: str, ref_audio_path: Path,
                           ref_text: str) -> Path:
        """Generate via voice cloning — consistent voice from reference clip."""
        wav_data, sr = await self._generate_clone(text, ref_audio_path, ref_text)
        return await self._wavs_to_ogg(wav_data, sr)

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
