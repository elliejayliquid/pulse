"""Safe, targeted persona config editing for the GUI."""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import Any

from gui.backup import BackupManager, IDENTITY_FILES


IDENTITY_FIELDS = {
    "model": "Display Model Name",
    "voice_notes": "Voice Notes",
}

TTS_FIELDS = {
    "voice_description": "TTS Voice Description",
    "voice_sample": "TTS Voice Sample",
    "voice_sample_text": "TTS Voice Sample Text",
}


class ConfigEditor:
    """Preview and apply allowlisted persona edits."""

    def __init__(self, root: Path | str, backups: BackupManager):
        self.root = Path(root).resolve()
        self.personas_dir = self.root / "personas"
        self.backups = backups

    def preview(self, persona: str, changes: dict[str, Any]) -> dict[str, Any]:
        persona_dir = self._persona_dir(persona)
        normalized = self._normalize_changes(changes)
        rendered = self._render_files(persona_dir, normalized)
        changed_files = [
            result for result in rendered
            if result["original"] != result["updated"]
        ]
        return {
            "persona": persona,
            "has_changes": bool(changed_files),
            "changes": self._change_summary(persona_dir, changed_files),
            "diff": "\n".join(
                self._diff(result["path"], result["original"], result["updated"])
                for result in changed_files
            ),
        }

    def save(self, persona: str, changes: dict[str, Any]) -> dict[str, Any]:
        persona_dir = self._persona_dir(persona)
        normalized = self._normalize_changes(changes)
        rendered = self._render_files(persona_dir, normalized)
        changed_files = [
            result for result in rendered
            if result["original"] != result["updated"]
        ]
        if not changed_files:
            return {
                "ok": True,
                "changed": False,
                "backup": None,
                "changes": [],
                "diff": "",
            }

        backup = self.backups.create_backup(persona, reason="pre-edit")
        for result in changed_files:
            self._atomic_write(result["path"], result["updated"])

        return {
            "ok": True,
            "changed": True,
            "backup": backup,
            "changes": self._change_summary(persona_dir, changed_files),
            "diff": "\n".join(
                self._diff(result["path"], result["original"], result["updated"])
                for result in changed_files
            ),
        }

    def _persona_dir(self, persona: str) -> Path:
        if not persona or persona == "__base__":
            raise ValueError("Choose a persona before saving.")
        persona_dir = (self.personas_dir / persona).resolve()
        try:
            persona_dir.relative_to(self.personas_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Invalid persona name: {persona}") from exc
        if not persona_dir.is_dir():
            raise FileNotFoundError(f"Persona not found: {persona}")
        return persona_dir

    def _normalize_changes(self, changes: dict[str, Any]) -> dict[str, dict[str, str]]:
        if not isinstance(changes, dict):
            raise ValueError("Changes must be an object.")

        identity = changes.get("identity", {}) or {}
        tts = changes.get("tts", {}) or {}
        unknown_identity = sorted(set(identity) - set(IDENTITY_FIELDS))
        unknown_tts = sorted(set(tts) - set(TTS_FIELDS))
        if unknown_identity or unknown_tts:
            unknown = unknown_identity + [f"tts.{name}" for name in unknown_tts]
            raise ValueError(f"Unsupported field(s): {', '.join(unknown)}")

        return {
            "identity": {
                key: self._clean_text(value, IDENTITY_FIELDS[key])
                for key, value in identity.items()
            },
            "tts": {
                key: self._clean_text(value, TTS_FIELDS[key])
                for key, value in tts.items()
            },
        }

    def _clean_text(self, value: Any, label: str) -> str:
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"{label} must be text.")
        if "\x00" in value:
            raise ValueError(f"{label} cannot contain null bytes.")
        if len(value) > 20000:
            raise ValueError(f"{label} is too long.")
        return value.replace("\r\n", "\n").replace("\r", "\n")

    def _render_files(self, persona_dir: Path, changes: dict[str, dict[str, str]]) -> list[dict]:
        rendered = []
        if changes["identity"]:
            identity_path = self._identity_path(persona_dir)
            original = identity_path.read_text(encoding="utf-8") if identity_path.exists() else ""
            updated = original
            for key, value in changes["identity"].items():
                updated = _set_top_level_field(updated, key, value)
            rendered.append({"path": identity_path, "original": original, "updated": updated})

        if changes["tts"]:
            config_path = persona_dir / "config.yaml"
            original = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
            updated = original
            for key, value in changes["tts"].items():
                updated = _set_nested_field(updated, "tts", key, value)
            rendered.append({"path": config_path, "original": original, "updated": updated})
        return rendered

    def _identity_path(self, persona_dir: Path) -> Path:
        for filename in IDENTITY_FILES:
            path = persona_dir / filename
            if path.exists():
                return path
        return persona_dir / "persona.yaml"

    def _change_summary(self, persona_dir: Path, changed_files: list[dict]) -> list[dict[str, str]]:
        return [
            {
                "file": result["path"].relative_to(self.root).as_posix(),
                "summary": f"Updated {result['path'].name}",
            }
            for result in changed_files
        ]

    def _diff(self, path: Path, original: str, updated: str) -> str:
        rel = path.relative_to(self.root).as_posix()
        return "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"{rel} (current)",
            tofile=f"{rel} (proposed)",
        ))

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def _set_top_level_field(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    start, end = _top_level_range(lines, key)
    block_indicator = _block_indicator(lines[start]) if start is not None else None
    replacement = _format_field(key, value, 0, block_indicator=block_indicator)
    if start is None:
        prefix = text.rstrip("\n")
        sep = "\n" if prefix else ""
        updated = f"{prefix}{sep}{replacement}\n"
        _assert_no_duplicate_keys(updated)
        return updated
    new_lines = lines[:start] + replacement.splitlines() + lines[end:]
    updated = "\n".join(new_lines) + "\n"
    _assert_no_duplicate_keys(updated)
    return updated


def _set_nested_field(text: str, parent: str, key: str, value: str) -> str:
    lines = text.splitlines()
    parent_start, parent_end = _top_level_range(lines, parent)
    if parent_start is None:
        prefix = text.rstrip("\n")
        sep = "\n\n" if prefix else ""
        updated = f"{prefix}{sep}{parent}:\n{_format_field(key, value, 2)}\n"
        _assert_no_duplicate_keys(updated)
        return updated

    indent = _child_indent(lines, parent_start + 1, parent_end)
    field_start, field_end = _nested_range(lines, parent_start + 1, parent_end, key, indent)
    block_indicator = _block_indicator(lines[field_start]) if field_start is not None else None
    replacement = _format_field(key, value, indent, block_indicator=block_indicator)
    if field_start is None:
        insert_at = parent_end
        new_lines = lines[:insert_at] + replacement.splitlines() + lines[insert_at:]
    else:
        new_lines = lines[:field_start] + replacement.splitlines() + lines[field_end:]
    updated = "\n".join(new_lines) + "\n"
    _assert_no_duplicate_keys(updated)
    return updated


def _top_level_range(lines: list[str], key: str) -> tuple[int | None, int | None]:
    pattern = re.compile(rf"^{re.escape(key)}\s*:")
    top_level = re.compile(r"^[A-Za-z_][\w-]*\s*:")
    for i, line in enumerate(lines):
        if pattern.match(line):
            end = i + 1
            while end < len(lines) and not top_level.match(lines[end]):
                end += 1
            return i, end
    return None, None


def _nested_range(lines: list[str], start: int, end: int, key: str, indent: int) -> tuple[int | None, int | None]:
    prefix = " " * indent
    pattern = re.compile(rf"^{re.escape(prefix)}{re.escape(key)}\s*:")
    nested = re.compile(rf"^{re.escape(prefix)}[A-Za-z_][\w-]*\s*:")
    for i in range(start, end):
        if pattern.match(lines[i]):
            field_end = i + 1
            while field_end < end and not nested.match(lines[field_end]) and not re.match(r"^[A-Za-z_]", lines[field_end]):
                field_end += 1
            return i, field_end
    return None, None


def _child_indent(lines: list[str], start: int, end: int) -> int:
    for i in range(start, end):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace():
            return len(line) - len(line.lstrip(" "))
    return 2


def _block_indicator(line: str) -> str | None:
    match = re.match(r"^\s*[A-Za-z_][\w-]*\s*:\s*([|>][+-]?)\s*(?:#.*)?$", line)
    return match.group(1) if match else None


def _assert_no_duplicate_keys(text: str) -> None:
    seen: dict[tuple[str, ...], set[str]] = {}
    stack: list[tuple[int, str]] = []
    block_indent: int | None = None

    for line in text.splitlines():
        if block_indent is not None:
            if not line.strip() or (line[:1].isspace() and _indent_of(line) > block_indent):
                continue
            block_indent = None

        if line.lstrip().startswith("#"):
            continue
        match = re.match(r"^(\s*)([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if not match:
            continue

        indent = len(match.group(1))
        key = match.group(2)
        rest = match.group(3).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent_path = tuple(item[1] for item in stack)
        bucket = seen.setdefault(parent_path, set())
        if key in bucket:
            raise ValueError(f"Duplicate key '{key}' detected - manual edit needed.")
        bucket.add(key)
        stack.append((indent, key))
        if re.match(r"^[|>][+-]?(?:\s+#.*)?$", rest):
            block_indent = indent


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _format_field(key: str, value: str, indent: int, block_indicator: str | None = None) -> str:
    prefix = " " * indent
    if "\n" in value:
        lines = value.split("\n")
        body = "\n".join(f"{prefix}  {line}" if line else f"{prefix}" for line in lines)
        return f"{prefix}{key}: {block_indicator or '|'}\n{body}"
    # Known limitation: replacing an inline scalar also replaces any trailing
    # inline comment on that same line. We avoid broad YAML round-trips so
    # surrounding comments and formatting survive.
    return f"{prefix}{key}: {json.dumps(value, ensure_ascii=False)}"
