"""
Voice transcription via whisper.cpp — auto-downloads binary, model, and ffmpeg.

Usage:
    transcriber = Transcriber(config)
    await transcriber.ensure_ready()
    text = await transcriber.transcribe("voice.ogg")

All downloads are lazy — nothing is fetched until the first voice message.
"""

import asyncio
import logging
import platform
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MODEL = "base"
WHISPER_VERSION = "1.8.4"

# Download URLs
WHISPER_BIN_URL = (
    f"https://github.com/ggml-org/whisper.cpp/releases/download/"
    f"v{WHISPER_VERSION}/whisper-bin-x64.zip"
)
WHISPER_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin"
)
FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
    "latest/ffmpeg-master-latest-win64-gpl.zip"
)


class Transcriber:
    """Local voice transcription using whisper.cpp CLI."""

    def __init__(self, config: dict):
        voice_cfg = config.get("voice", {})
        self.model_name = voice_cfg.get("whisper_model", DEFAULT_MODEL)
        self.language = voice_cfg.get("language", "auto")

        # Store everything under data/whisper/ (derive from memories path)
        memories_dir = config.get("paths", {}).get("memories", "data/memories")
        data_root = str(Path(memories_dir).parent)
        self.data_dir = Path(data_root) / "whisper"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._ready = False

    @property
    def whisper_exe(self) -> Path:
        suffix = ".exe" if platform.system() == "Windows" else ""
        return self.data_dir / f"whisper-cli{suffix}"

    @property
    def model_path(self) -> Path:
        return self.data_dir / f"ggml-{self.model_name}.bin"

    @property
    def ffmpeg_exe(self) -> Path:
        # Use system ffmpeg if available, otherwise our downloaded one
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return Path(system_ffmpeg)
        suffix = ".exe" if platform.system() == "Windows" else ""
        return self.data_dir / f"ffmpeg{suffix}"

    def is_ready(self) -> bool:
        return self.whisper_exe.exists() and self.model_path.exists() and self.ffmpeg_exe.exists()

    async def ensure_ready(self):
        """Download whisper.cpp, model, and ffmpeg if not present."""
        if self._ready:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        if not self.whisper_exe.exists():
            tasks.append(self._download_whisper())
        if not self.model_path.exists():
            tasks.append(self._download_model())
        if not self.ffmpeg_exe.exists():
            tasks.append(self._download_ffmpeg())

        if tasks:
            logger.info(f"Downloading {len(tasks)} component(s) for voice transcription...")
            await asyncio.gather(*tasks)

        self._ready = True
        logger.info(f"Voice transcription ready (model: {self.model_name})")

    async def transcribe(self, ogg_path: str | Path) -> str:
        """Transcribe an .ogg voice file to text.

        Converts ogg → wav via ffmpeg, then runs whisper-cli.
        Returns the transcribed text.
        """
        if not self._ready:
            await self.ensure_ready()

        ogg_path = Path(ogg_path)
        wav_path = ogg_path.with_suffix(".wav")

        try:
            # Convert ogg → wav
            await self._convert_to_wav(ogg_path, wav_path)

            # Run whisper
            text = await self._run_whisper(wav_path)
            return text.strip()
        finally:
            # Clean up temp files
            for f in (wav_path, ogg_path):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _convert_to_wav(self, input_path: Path, output_path: Path):
        """Convert audio to WAV using ffmpeg."""
        cmd = [
            str(self.ffmpeg_exe),
            "-i", str(input_path),
            "-ar", "16000",       # whisper expects 16kHz
            "-ac", "1",           # mono
            "-y",                 # overwrite
            str(output_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {stderr.decode()[:500]}")

        logger.debug(f"Converted {input_path.name} → {output_path.name}")

    async def _run_whisper(self, wav_path: Path) -> str:
        """Run whisper-cli on a WAV file and return the transcribed text."""
        cmd = [
            str(self.whisper_exe),
            "-m", str(self.model_path),
            "-f", str(wav_path),
            "--no-timestamps",
            "-l", self.language,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Whisper failed: {stderr.decode()[:500]}")

        # whisper-cli outputs transcription to stdout, progress/info to stderr
        text = stdout.decode("utf-8", errors="replace")
        # Filter out whisper info lines
        lines = [
            line for line in text.splitlines()
            if line.strip()
            and not line.startswith("whisper_")
            and not line.startswith("main:")
            and not line.startswith("system_info")
        ]
        result = " ".join(lines).strip()
        logger.info(f"Transcribed ({len(result)} chars): {result[:80]}...")
        return result

    # --- Download helpers ---

    async def _download_file(self, url: str, dest: Path, label: str) -> Path:
        """Download a file with progress logging."""
        import urllib.request

        logger.info(f"[Voice] Downloading {label}...")

        def _download():
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / 1_048_576
            logger.info(f"[Voice] {label} downloaded ({size_mb:.1f} MB)")

        await asyncio.to_thread(_download)
        return dest

    async def _download_whisper(self):
        """Download and extract whisper.cpp CLI binary."""
        zip_path = self.data_dir / "_whisper.zip"
        try:
            await self._download_file(WHISPER_BIN_URL, zip_path, "whisper.cpp")
            await asyncio.to_thread(self._extract_whisper, zip_path)
        finally:
            zip_path.unlink(missing_ok=True)

    def _extract_whisper(self, zip_path: Path):
        """Extract whisper-cli exe and DLLs from the zip."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                name = Path(info.filename).name.lower()
                if name in ("whisper-cli.exe", "main.exe"):
                    # Extract to our known path
                    data = zf.read(info.filename)
                    self.whisper_exe.write_bytes(data)
                    logger.info(f"[Voice] Extracted {name}")
                elif name.endswith(".dll"):
                    # DLLs need to be alongside the exe
                    dll_dest = self.data_dir / name
                    data = zf.read(info.filename)
                    dll_dest.write_bytes(data)
                    logger.info(f"[Voice] Extracted {name}")

    async def _download_model(self):
        """Download whisper GGML model from HuggingFace."""
        url = WHISPER_MODEL_URL.format(model=self.model_name)
        await self._download_file(url, self.model_path, f"whisper model ({self.model_name})")

    async def _download_ffmpeg(self):
        """Download ffmpeg static build and extract ffmpeg.exe."""
        if platform.system() != "Windows":
            logger.warning(
                "[Voice] Auto-download only supports Windows. "
                "Please install ffmpeg manually: apt install ffmpeg / brew install ffmpeg"
            )
            return

        zip_path = self.data_dir / "_ffmpeg.zip"
        try:
            await self._download_file(FFMPEG_URL, zip_path, "ffmpeg")
            await asyncio.to_thread(self._extract_ffmpeg, zip_path)
        finally:
            zip_path.unlink(missing_ok=True)

    def _extract_ffmpeg(self, zip_path: Path):
        """Extract just ffmpeg.exe from the (large) ffmpeg zip."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                name = Path(info.filename).name.lower()
                if name == "ffmpeg.exe":
                    data = zf.read(info.filename)
                    target = self.data_dir / "ffmpeg.exe"
                    target.write_bytes(data)
                    logger.info(f"[Voice] Extracted ffmpeg.exe")
                    return
        raise RuntimeError("ffmpeg.exe not found in downloaded archive")
