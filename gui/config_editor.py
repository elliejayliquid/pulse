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
    "name": "Companion Name",
    "user_name": "User Name",
    "model": "Display Model Name",
    "system_prompt": "System Prompt",
    "relationship_context": "Relationship Context",
    "voice_notes": "Voice Notes",
    "traits": "Traits",
}

TTS_FIELDS = {
    "voice_description": "TTS Voice Description",
    "voice_sample": "TTS Voice Sample",
    "voice_sample_text": "TTS Voice Sample Text",
}

PROVIDER_FIELDS = {
    "type": "Provider Type",
    "model": "Provider Model",
    "max_context": "Max Context",
    "base_url": "Base URL",
}

PROVIDER_TYPES = {"local", "openrouter", "openai", "anthropic", "custom"}

SERVER_FIELDS = {
    "llama_cpp_dir": "llama.cpp Directory",
    "models_dir": "Models Directory",
    "host": "Server Host",
    "port": "Server Port",
    "gpu_layers": "GPU Layers",
    "flash_attention": "Flash Attention",
    "parallel": "Parallel Slots",
}

MODEL_FIELDS = {
    "model_file": "Model File",
    "mmproj_file": "Vision Projector",
    "max_context": "Max Context",
    "temperature": "Temperature",
    "max_response_tokens": "Max Response Tokens",
    "frequency_penalty": "Frequency Penalty",
    "presence_penalty": "Presence Penalty",
    "top_p": "Top P",
    "top_k": "Top K",
    "reasoning": "Reasoning",
    "reasoning_effort": "Reasoning Effort",
    "show_reasoning": "Show Reasoning",
    "max_tool_rounds": "Max Tool Rounds",
}

MODEL_LIMITS = {
    "temperature": (0.0, 2.0),
    "max_response_tokens": (256, 32768),
    "frequency_penalty": (0.0, 2.0),
    "presence_penalty": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, 1000),
    "max_context": (1024, 1048576),
    "max_tool_rounds": (1, 32),
}

SERVER_LIMITS = {
    "port": (1024, 65535),
    "gpu_layers": (-1, 512),
    "parallel": (1, 8),
}

REASONING_EFFORTS = {"", "low", "medium", "high"}

CONTEXT_BUDGET_FIELDS = {
    "recent_tail_exchanges": "Recent Tail Exchanges",
}

CONTEXT_BUDGET_LIMITS = {
    "recent_tail_exchanges": (1, 10),
}

HEARTBEAT_FIELDS = {
    "interval_minutes": "Heartbeat Interval",
    "randomize": "Randomize Heartbeat",
    "interval_min_minutes": "Heartbeat Min Interval",
    "interval_max_minutes": "Heartbeat Max Interval",
    "quiet_hours_start": "Quiet Hours Start",
    "quiet_hours_end": "Quiet Hours End",
    "debug": "Heartbeat Debug",
}

HEARTBEAT_LIMITS = {
    "interval_minutes": (1, 1440),
    "interval_min_minutes": (1, 1440),
    "interval_max_minutes": (1, 1440),
    "quiet_hours_start": (0, 23),
    "quiet_hours_end": (0, 23),
}

DEV_TICK_FIELDS = {
    "enabled": "Dev Tick",
    "interval_minutes": "Dev Tick Interval",
    "schedule_time": "Dev Tick Schedule Time",
    "max_rounds": "Dev Tick Max Rounds",
}

DEV_TICK_LIMITS = {
    "interval_minutes": (1, 10080),
    "max_rounds": (1, 32),
}

CHANNEL_NAMES = {"telegram", "toast", "lor"}

LOR_CHANNEL_FIELDS = {
    "enabled": "LoR Channel",
    "author_name": "LoR Author Name",
    "model_name": "LoR Model Name",
    "context_initial_lookback_hours": "LoR Initial Lookback Hours",
}

CONTEXT_FIELDS = {
    "inject_skills": "Context Injection Skills",
}

PATH_FIELDS = {
    "lor_data": "LoR Data Folder",
}


class ConfigEditor:
    """Preview and apply allowlisted persona edits."""

    def __init__(self, root: Path | str, backups: BackupManager):
        self.root = Path(root).resolve()
        self.personas_dir = self.root / "personas"
        self.backups = backups

    def preview(self, persona: str, changes: dict[str, Any]) -> dict[str, Any]:
        persona_dir = self._persona_dir(persona)
        normalized = self._normalize_changes(changes, persona_dir)
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
        normalized = self._normalize_changes(changes, persona_dir)
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

    def _normalize_changes(self, changes: dict[str, Any], persona_dir: Path) -> dict[str, dict[str, Any]]:
        if not isinstance(changes, dict):
            raise ValueError("Changes must be an object.")

        identity = changes.get("identity", {}) or {}
        tts = changes.get("tts", {}) or {}
        provider = changes.get("provider", {}) or {}
        server = changes.get("server", {}) or {}
        model = changes.get("model", {}) or {}
        context_budget = changes.get("context_budget", {}) or {}
        heartbeat = changes.get("heartbeat", {}) or {}
        dev_tick = changes.get("dev_tick", {}) or {}
        skills = changes.get("skills", {}) or {}
        channels = changes.get("channels", {}) or {}
        context = changes.get("context", {}) or {}
        paths = changes.get("paths", {}) or {}
        if not isinstance(provider, dict):
            raise ValueError("Provider changes must be an object.")
        if not isinstance(server, dict):
            raise ValueError("Server changes must be an object.")
        if not isinstance(model, dict):
            raise ValueError("Model changes must be an object.")
        if not isinstance(context_budget, dict):
            raise ValueError("Context budget changes must be an object.")
        if not isinstance(dev_tick, dict):
            raise ValueError("Dev tick changes must be an object.")
        if not isinstance(skills, dict):
            raise ValueError("Skills changes must be an object.")
        if not isinstance(channels, dict):
            raise ValueError("Channels changes must be an object.")
        if not isinstance(context, dict):
            raise ValueError("Context changes must be an object.")
        if not isinstance(paths, dict):
            raise ValueError("Path changes must be an object.")
        valid_skills = self._editable_skill_names(persona_dir)
        unknown_identity = sorted(set(identity) - set(IDENTITY_FIELDS))
        unknown_tts = sorted(set(tts) - set(TTS_FIELDS))
        unknown_provider = sorted(set(provider) - set(PROVIDER_FIELDS))
        unknown_server = sorted(set(server) - set(SERVER_FIELDS))
        unknown_model = sorted(set(model) - set(MODEL_FIELDS))
        unknown_context_budget = sorted(set(context_budget) - set(CONTEXT_BUDGET_FIELDS))
        unknown_heartbeat = sorted(set(heartbeat) - set(HEARTBEAT_FIELDS))
        unknown_dev_tick = sorted(set(dev_tick) - set(DEV_TICK_FIELDS))
        unknown_skills = sorted(set(skills) - valid_skills)
        unknown_channels = sorted(set(channels) - CHANNEL_NAMES)
        unknown_context = sorted(set(context) - set(CONTEXT_FIELDS))
        unknown_paths = sorted(set(paths) - set(PATH_FIELDS))
        if (
            unknown_identity
            or unknown_tts
            or unknown_provider
            or unknown_server
            or unknown_model
            or unknown_context_budget
            or unknown_heartbeat
            or unknown_dev_tick
            or unknown_skills
            or unknown_channels
            or unknown_context
            or unknown_paths
        ):
            unknown = (
                unknown_identity
                + [f"tts.{name}" for name in unknown_tts]
                + [f"provider.{name}" for name in unknown_provider]
                + [f"server.{name}" for name in unknown_server]
                + [f"model.{name}" for name in unknown_model]
                + [f"context_budget.{name}" for name in unknown_context_budget]
                + [f"heartbeat.{name}" for name in unknown_heartbeat]
                + [f"dev_tick.{name}" for name in unknown_dev_tick]
                + [f"skills.{name}" for name in unknown_skills]
                + [f"channels.{name}" for name in unknown_channels]
                + [f"context.{name}" for name in unknown_context]
                + [f"paths.{name}" for name in unknown_paths]
            )
            raise ValueError(f"Unsupported field(s): {', '.join(unknown)}")

        normalized_identity = {}
        for key, value in identity.items():
            if key == "traits":
                normalized_identity[key] = self._clean_traits(value)
            else:
                normalized_identity[key] = self._clean_text(value, IDENTITY_FIELDS[key])

        return {
            "identity": normalized_identity,
            "tts": {
                key: self._clean_text(value, TTS_FIELDS[key])
                for key, value in tts.items()
            },
            "provider": {
                key: self._clean_provider_value(key, value)
                for key, value in provider.items()
            },
            "server": {
                key: self._clean_server_value(key, value)
                for key, value in server.items()
            },
            "model": {
                key: self._clean_model_value(key, value)
                for key, value in model.items()
            },
            "context_budget": {
                key: self._clean_context_budget_value(key, value)
                for key, value in context_budget.items()
            },
            "heartbeat": {
                key: self._clean_heartbeat_value(key, value)
                for key, value in heartbeat.items()
            },
            "dev_tick": {
                key: self._clean_dev_tick_value(key, value)
                for key, value in dev_tick.items()
            },
            "skills": {
                key: self._clean_skill_value(key, value)
                for key, value in skills.items()
            },
            "channels": {
                key: self._clean_channel_value(key, value)
                for key, value in channels.items()
            },
            "context": {
                key: self._clean_context_value(key, value, valid_skills)
                for key, value in context.items()
            },
            "paths": {
                key: self._clean_text(value, PATH_FIELDS[key])
                for key, value in paths.items()
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

    def _clean_traits(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("Traits must be a list.")
        if len(value) > 20:
            raise ValueError("Too many traits (max 20).")
        cleaned = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Each trait must be text.")
            text = item.replace("\r\n", "\n").replace("\r", "\n").strip()
            if "\x00" in text:
                raise ValueError("Trait cannot contain null bytes.")
            if not text:
                continue
            if len(text) > 200:
                raise ValueError(f"Trait too long (max 200 chars): {text[:30]}...")
            lower = text.lower()
            if lower in seen:
                continue
            seen.add(lower)
            cleaned.append(text)
        return cleaned

    def _clean_provider_value(self, key: str, value: Any) -> str | int:
        label = PROVIDER_FIELDS[key]
        if key == "type":
            if not isinstance(value, str) or value not in PROVIDER_TYPES:
                raise ValueError(f"{label} must be one of: {', '.join(sorted(PROVIDER_TYPES))}.")
            return value
        if key == "model":
            return self._clean_text(value, label)
        if key == "base_url":
            return self._clean_text(value, label)
        if type(value) is not int:
            raise ValueError(f"{label} must be a whole number.")
        if not 1024 <= value <= 1048576:
            raise ValueError(f"{label} must be between 1024 and 1048576.")
        return value

    def _clean_server_value(self, key: str, value: Any) -> str | int | bool:
        label = SERVER_FIELDS[key]
        if key in ("llama_cpp_dir", "models_dir", "host"):
            return self._clean_text(value, label)
        if key == "flash_attention":
            if type(value) is not bool:
                raise ValueError(f"{label} must be true or false.")
            return value
        if type(value) is not int:
            raise ValueError(f"{label} must be a whole number.")
        low, high = SERVER_LIMITS[key]
        if not low <= value <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")
        return value

    def _clean_model_value(self, key: str, value: Any) -> str | int | float | bool:
        label = MODEL_FIELDS[key]
        if key in ("model_file", "mmproj_file"):
            return self._clean_text(value, label)
        if key in ("reasoning", "show_reasoning"):
            if type(value) is not bool:
                raise ValueError(f"{label} must be true or false.")
            return value
        if key == "reasoning_effort":
            if not isinstance(value, str) or value not in REASONING_EFFORTS:
                raise ValueError(f"{label} must be one of: low, medium, high, or empty.")
            return value
        if key in ("max_response_tokens", "max_context", "max_tool_rounds", "top_k"):
            if type(value) is not int:
                raise ValueError(f"{label} must be a whole number.")
            low, high = MODEL_LIMITS[key]
            if not low <= value <= high:
                raise ValueError(f"{label} must be between {low} and {high}.")
            return value
        if not isinstance(value, (int, float)) or type(value) is bool:
            raise ValueError(f"{label} must be a number.")
        cleaned = round(float(value), 2)
        low, high = MODEL_LIMITS[key]
        if not low <= cleaned <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")
        return cleaned

    def _clean_context_budget_value(self, key: str, value: Any) -> int:
        label = CONTEXT_BUDGET_FIELDS[key]
        if type(value) is not int:
            raise ValueError(f"{label} must be a whole number.")
        low, high = CONTEXT_BUDGET_LIMITS[key]
        if not low <= value <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")
        return value

    def _clean_heartbeat_value(self, key: str, value: Any) -> int | bool:
        label = HEARTBEAT_FIELDS[key]
        if key in ("randomize", "debug"):
            if type(value) is not bool:
                raise ValueError(f"{label} must be true or false.")
            return value
        if type(value) is not int:
            raise ValueError(f"{label} must be a whole number.")
        low, high = HEARTBEAT_LIMITS[key]
        if not low <= value <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")
        return value

    def _clean_dev_tick_value(self, key: str, value: Any) -> int | bool | str:
        label = DEV_TICK_FIELDS[key]
        if key == "enabled":
            if type(value) is not bool:
                raise ValueError(f"{label} must be true or false.")
            return value
        if key == "schedule_time":
            cleaned = self._clean_text(value, label).strip()
            if not cleaned:
                return ""
            if not re.match(r"^\d{2}:\d{2}$", cleaned):
                raise ValueError(f"{label} must be empty or use HH:MM.")
            hour, minute = (int(part) for part in cleaned.split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(f"{label} must be a valid 24-hour time.")
            return cleaned
        if type(value) is not int:
            raise ValueError(f"{label} must be a whole number.")
        low, high = DEV_TICK_LIMITS[key]
        if not low <= value <= high:
            raise ValueError(f"{label} must be between {low} and {high}.")
        return value

    def _clean_channel_value(self, key: str, value: Any) -> dict[str, Any]:
        if type(value) is bool:
            return {"enabled": value}
        if not isinstance(value, dict):
            raise ValueError(f"{key.title()} channel must be true or false.")

        allowed = {"enabled"}
        if key == "lor":
            allowed = set(LOR_CHANNEL_FIELDS)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "Unsupported field(s): "
                + ", ".join(f"channels.{key}.{field}" for field in unknown)
            )

        cleaned: dict[str, Any] = {}
        for field, raw in value.items():
            if field == "enabled":
                if type(raw) is not bool:
                    raise ValueError(f"{key.title()} channel must be true or false.")
                cleaned[field] = raw
            elif field in ("author_name", "model_name"):
                cleaned[field] = self._clean_text(raw, LOR_CHANNEL_FIELDS[field])
            elif field == "context_initial_lookback_hours":
                if type(raw) is not int:
                    raise ValueError("LoR Initial Lookback Hours must be a whole number.")
                if not 1 <= raw <= 8760:
                    raise ValueError("LoR Initial Lookback Hours must be between 1 and 8760.")
                cleaned[field] = raw
        return cleaned

    def _clean_context_value(self, key: str, value: Any, valid_skills: set[str]) -> list[str]:
        if key != "inject_skills":
            raise ValueError(f"Unsupported field: context.{key}")
        if not isinstance(value, list):
            raise ValueError("Context Injection Skills must be a list.")
        cleaned = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Context Injection Skills must contain skill names.")
            name = item.strip()
            if not name:
                continue
            if name not in valid_skills:
                raise ValueError(f"Unknown context injection skill: {name}")
            if name not in seen:
                seen.add(name)
                cleaned.append(name)
        return cleaned

    def _clean_skill_value(self, key: str, value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError(f"{key.replace('_', ' ').title()} skill must be true or false.")
        return value

    def _editable_skill_names(self, persona_dir: Path) -> set[str]:
        skills_dir = self.root / "skills"
        names = set()
        if skills_dir.exists():
            names.update(
                path.stem
                for path in skills_dir.glob("*.py")
                if path.stem not in ("__init__", "base")
            )
        config_path = persona_dir / "config.yaml"
        if config_path.exists():
            names.update(_nested_keys(config_path.read_text(encoding="utf-8"), "skills"))
        return names

    def _render_files(self, persona_dir: Path, changes: dict[str, dict[str, Any]]) -> list[dict]:
        rendered = []
        if changes["identity"]:
            identity_path = self._identity_path(persona_dir)
            original = identity_path.read_text(encoding="utf-8") if identity_path.exists() else ""
            updated = original
            for key, value in changes["identity"].items():
                updated = _set_top_level_field(updated, key, value)
            rendered.append({"path": identity_path, "original": original, "updated": updated})

        if (
            changes["tts"]
            or changes["provider"]
            or changes["server"]
            or changes["model"]
            or changes["context_budget"]
            or changes["heartbeat"]
            or changes["dev_tick"]
            or changes["skills"]
            or changes["channels"]
            or changes["context"]
            or changes["paths"]
        ):
            config_path = persona_dir / "config.yaml"
            original = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
            updated = original
            for key, value in changes["tts"].items():
                updated = _set_nested_field(updated, "tts", key, value)
            for key, value in changes["provider"].items():
                updated = _set_nested_field(updated, "provider", key, value)
            for key, value in changes["server"].items():
                updated = _set_nested_field(updated, "server", key, value)
            for key, value in changes["model"].items():
                updated = _set_nested_field(updated, "model", key, value)
            for key, value in changes["context_budget"].items():
                updated = _set_nested_field(updated, "context_budget", key, value)
            for key, value in changes["heartbeat"].items():
                updated = _set_nested_field(updated, "heartbeat", key, value)
            for key, value in changes["dev_tick"].items():
                updated = _set_nested_field(updated, "dev_tick", key, value)
            for key, value in changes["skills"].items():
                updated = _set_deep_nested_field(updated, ["skills", key, "enabled"], value)
            for key, fields in changes["channels"].items():
                for field, value in fields.items():
                    updated = _set_deep_nested_field(updated, ["channels", key, field], value)
            for key, value in changes["context"].items():
                updated = _set_nested_field(updated, "context", key, value)
            for key, value in changes["paths"].items():
                updated = _set_nested_field(updated, "paths", key, value)
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
        display_original = original if original.endswith("\n") or not original else f"{original}\n"
        display_updated = updated if updated.endswith("\n") or not updated else f"{updated}\n"
        return "".join(difflib.unified_diff(
            display_original.splitlines(keepends=True),
            display_updated.splitlines(keepends=True),
            fromfile=f"{rel} (current)",
            tofile=f"{rel} (proposed)",
        ))

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def _set_top_level_field(text: str, key: str, value: Any) -> str:
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


def _set_nested_field(text: str, parent: str, key: str, value: Any) -> str:
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
        insert_at = _nested_insert_at(lines, parent_start + 1, parent_end)
        new_lines = lines[:insert_at] + replacement.splitlines() + lines[insert_at:]
    else:
        new_lines = lines[:field_start] + replacement.splitlines() + lines[field_end:]
    updated = "\n".join(new_lines) + "\n"
    _assert_no_duplicate_keys(updated)
    return updated


def _set_deep_nested_field(text: str, path: list[str], value: Any) -> str:
    if len(path) < 2:
        raise ValueError("Nested path must contain at least two keys.")
    if len(path) == 2:
        return _set_nested_field(text, path[0], path[1], value)

    lines = text.splitlines()
    top_start, top_end = _top_level_range(lines, path[0])
    if top_start is None:
        prefix = text.rstrip("\n")
        sep = "\n\n" if prefix else ""
        updated = f"{prefix}{sep}{_format_deep_block(path, value, 0)}\n"
        _assert_no_duplicate_keys(updated)
        return updated

    block_start, block_end = top_start, top_end
    parent_indent = 0
    for depth, key in enumerate(path[1:-1], start=1):
        child_indent = _child_indent_or(lines, block_start + 1, block_end, parent_indent + 2)
        child_start, child_end = _nested_range(lines, block_start + 1, block_end, key, child_indent)
        if child_start is None:
            replacement = _format_deep_block(path[depth:], value, child_indent)
            insert_at = _nested_insert_at(lines, block_start + 1, block_end)
            new_lines = lines[:insert_at] + replacement.splitlines() + lines[insert_at:]
            updated = "\n".join(new_lines) + "\n"
            _assert_no_duplicate_keys(updated)
            return updated
        block_start, block_end = child_start, child_end
        parent_indent = child_indent

    leaf = path[-1]
    leaf_indent = _child_indent_or(lines, block_start + 1, block_end, parent_indent + 2)
    leaf_start, leaf_end = _nested_range(lines, block_start + 1, block_end, leaf, leaf_indent)
    replacement = _format_field(leaf, value, leaf_indent)
    if leaf_start is None:
        insert_at = _nested_insert_at(lines, block_start + 1, block_end)
        new_lines = lines[:insert_at] + replacement.splitlines() + lines[insert_at:]
    else:
        new_lines = lines[:leaf_start] + replacement.splitlines() + lines[leaf_end:]
    updated = "\n".join(new_lines) + "\n"
    _assert_no_duplicate_keys(updated)
    return updated


def _format_deep_block(path: list[str], value: Any, indent: int) -> str:
    if len(path) == 1:
        return _format_field(path[0], value, indent)
    prefix = " " * indent
    return f"{prefix}{path[0]}:\n{_format_deep_block(path[1:], value, indent + 2)}"


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
            if _block_indicator(lines[i]) is None and not re.match(rf"^{re.escape(prefix)}{re.escape(key)}\s*:\s*$", lines[i]):
                return i, i + 1
            field_end = i + 1
            while field_end < end and not nested.match(lines[field_end]) and not re.match(r"^[A-Za-z_]", lines[field_end]):
                field_end += 1
            return i, field_end
    return None, None


def _nested_keys(text: str, parent: str) -> set[str]:
    lines = text.splitlines()
    parent_start, parent_end = _top_level_range(lines, parent)
    if parent_start is None:
        return set()
    indent = _child_indent(lines, parent_start + 1, parent_end)
    prefix = " " * indent
    pattern = re.compile(rf"^{re.escape(prefix)}([A-Za-z_][\w-]*)\s*:")
    return {
        match.group(1)
        for line in lines[parent_start + 1:parent_end]
        if (match := pattern.match(line))
    }


def _nested_insert_at(lines: list[str], start: int, end: int) -> int:
    insert_at = end
    while insert_at > start:
        line = lines[insert_at - 1]
        stripped = line.strip()
        if not stripped or line.startswith("#"):
            insert_at -= 1
            continue
        break
    return insert_at


def _child_indent(lines: list[str], start: int, end: int) -> int:
    return _child_indent_or(lines, start, end, 2)


def _child_indent_or(lines: list[str], start: int, end: int, fallback: int) -> int:
    for i in range(start, end):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace():
            return len(line) - len(line.lstrip(" "))
    return fallback


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


def _format_field(key: str, value: Any, indent: int, block_indicator: str | None = None) -> str:
    prefix = " " * indent
    if isinstance(value, list):
        if not value:
            return f"{prefix}{key}: []"
        items = "\n".join(
            f"{prefix}  - {json.dumps(item, ensure_ascii=False)}"
            for item in value
        )
        return f"{prefix}{key}:\n{items}"
    if type(value) is bool:
        return f"{prefix}{key}: {'true' if value else 'false'}"
    if type(value) is int:
        return f"{prefix}{key}: {value}"
    if type(value) is float:
        formatted = f"{value:.2f}".rstrip("0")
        if formatted.endswith("."):
            formatted += "0"
        return f"{prefix}{key}: {formatted}"
    if "\n" in value:
        lines = value.split("\n")
        body = "\n".join(f"{prefix}  {line}" if line else f"{prefix}" for line in lines)
        return f"{prefix}{key}: {block_indicator or '|'}\n{body}"
    # Known limitation: replacing an inline scalar also replaces any trailing
    # inline comment on that same line. We avoid broad YAML round-trips so
    # surrounding comments and formatting survive.
    return f"{prefix}{key}: {json.dumps(value, ensure_ascii=False)}"
