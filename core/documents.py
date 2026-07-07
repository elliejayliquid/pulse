"""
Document inbox helpers — shared by the Telegram channel and the documents skill.

Incoming files are saved to the persona's inbox (personas/<p>/data/inbox/),
and text is extracted for the model: small documents are injected inline into
the conversation, larger ones are read on demand via the documents skill.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Extensions treated as plain text (decoded as UTF-8, errors replaced).
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json",
    ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html",
    ".htm", ".py", ".js", ".ts", ".css", ".sh", ".bat", ".ps1", ".sql",
    ".c", ".cpp", ".h", ".java", ".rb", ".go", ".rs", ".php", ".srt", ".vtt",
}

# Telegram Bot API refuses downloads above 20MB; leave headroom.
MAX_DOCUMENT_BYTES = 19 * 1024 * 1024

_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._ -]+")


def inbox_dir(config: dict) -> Path:
    """Persona inbox folder, derived from the per-persona database path."""
    db_path = Path(config.get("paths", {}).get("database", "data/pulse.db"))
    return db_path.parent / "inbox"


def safe_filename(name: str) -> str:
    cleaned = _FILENAME_SANITIZER.sub("_", (name or "").strip()).strip(". ")
    return cleaned[:120] or "document"


def save_to_inbox(directory: Path, filename: str, data: bytes) -> Path:
    """Save bytes under a sanitized, collision-free name. Returns the path."""
    directory.mkdir(parents=True, exist_ok=True)
    base = safe_filename(filename)
    stem, dot, suffix = base.rpartition(".")
    if not dot:
        stem, suffix = base, ""
    target = directory / base
    counter = 2
    while target.exists():
        target = directory / (f"{stem}_{counter}.{suffix}" if suffix else f"{stem}_{counter}")
        counter += 1
    target.write_bytes(data)
    return target


def extract_text(path: Path) -> tuple[str, str]:
    """Best-effort text extraction.

    Returns (text, note). Empty text with a non-empty note means the file
    could not be read as text (unsupported format, missing dependency, or
    extraction failure) — the note is safe to show to the model.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not PDF_AVAILABLE:
            return "", "PDF text extraction is unavailable (pypdf is not installed)."
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
            text = "\n\n".join(part for part in pages if part).strip()
            if not text:
                return "", (
                    "The PDF contains no extractable text — it is likely scanned "
                    "images without OCR."
                )
            return text, ""
        except Exception as e:
            logger.warning(f"PDF extraction failed for {path.name}: {e}")
            return "", f"PDF text extraction failed: {e}"

    if suffix in TEXT_EXTENSIONS or not suffix:
        try:
            raw = path.read_bytes()
        except OSError as e:
            return "", f"Could not read the file: {e}"
        if b"\x00" in raw[:4096]:
            return "", "The file appears to be binary, not text."
        return raw.decode("utf-8", errors="replace").strip(), ""

    return "", (
        f"'{suffix}' files are not readable as text yet "
        "(supported: plain text/code formats and PDF)."
    )
