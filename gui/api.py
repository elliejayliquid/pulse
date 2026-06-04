"""Backend bridge for the Pulse Engine desktop GUI.

Phase 1 is intentionally read-only for persona configs. The GUI can inspect
personas, merged config, identity, status files, and logs without mutating the
project.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gui.backup import BackupManager
from gui.config_editor import ConfigEditor
from core.journal_mirror import (
    journal_memory_display_text,
    search_summary_is_thin,
)
from skills.journal import VALID_ENTRY_TYPES


STATUS_STALE_AFTER_SECONDS = 120
LANTERN_STALE_HOURS = 24
LANTERN_EXPIRED_HOURS = 7 * 24
LANTERN_FIELDS = ("mode", "mood", "focus", "note", "open_thread")
CORE_ANCHOR_TEMPLATES = {
    "_self": {
        "title": "Who I Am",
        "sections": (
            "who_i_am",
            "what_im_like",
            "my_preferences",
            "how_i_present_myself",
            "what_im_working_on",
            "extra_notes",
        ),
    },
    "_user": {
        "title": "About My Human",
        "sections": (
            "who_they_are",
            "what_theyre_like",
            "their_preferences",
            "how_they_communicate",
            "extra_notes",
        ),
    },
    "_relationship": {
        "title": "Our Relationship",
        "sections": (
            "how_we_relate",
            "our_dynamic",
            "shared_context",
            "boundaries_or_norms",
            "extra_notes",
        ),
    },
}


PROVIDER_KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

PROVIDER_KEY_LABELS = {
    "OPENROUTER_API_KEY": "OpenRouter API Key",
    "OPENAI_API_KEY": "OpenAI API Key",
    "ANTHROPIC_API_KEY": "Anthropic API Key",
    "TELEGRAM_BOT_TOKEN": "Telegram Bot Token",
}

SECRET_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,80}$")
MEMORY_STATUSES = {"current", "historical", "superseded", "archived"}
MEMORY_CONFIDENCES = {"high", "medium", "low"}
MEMORY_SOURCES = {"user_defined", "model_extracted", "imported", "system"}


SKILL_ICONS = {
    "dev": "{}",
    "garden": "🌱",
    "journal": "📖",
    "lantern": "🕯",
    "lor": "🌐",
    "memory": "🧠",
    "paint": "🎨",
    "schedule": "⏰",
    "sticker": "★",
    "tasks": "✓",
    "tts": "🔊",
    "web_search": "🔍",
}


@dataclass
class ProcessInfo:
    persona: str
    pid: int
    started_at: str
    process: Any = None


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay values win."""
    merged = dict(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return "********" + value[-4:]


def _secret_label(key_name: str) -> str:
    return PROVIDER_KEY_LABELS.get(key_name, key_name)


def _env_line_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key = stripped.split("=", 1)[0].strip()
    return key if SECRET_KEY_RE.match(key) else None


def _write_env_preserving(path: Path, updates: dict[str, str | None]) -> None:
    """Apply env var updates while preserving comments, order, and unknown lines."""
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    output: list[str] = []

    for line in existing_lines:
        key = _env_line_key(line)
        if key and key in remaining:
            value = remaining.pop(key)
            if value:
                output.append(f"{key}={value}")
            continue
        output.append(line)

    for key, value in remaining.items():
        if value:
            output.append(f"{key}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    content = "\n".join(output)
    tmp.write_text((content + "\n") if content else "", encoding="utf-8")
    os.replace(tmp, path)


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Open a GUI-side SQLite connection that waits briefly on writer locks."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


class PulseAPI:
    """Backend bridge called from JS as window.pywebview.api.*."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.base_config_path = self.root / "config.yaml"
        self.personas_dir = self.root / "personas"
        self.prefs_path = self.root / "data" / "gui_prefs.json"
        self.backups = BackupManager(self.root)
        self.config_editor = ConfigEditor(self.root, self.backups)
        self.processes: dict[str, ProcessInfo] = {}
        self._window = None
        self._close_requested: list[str] | None = None
        self._force_close = False

    # Persona management

    def list_personas(self) -> list[dict]:
        personas: list[dict] = [self._base_persona_summary()]
        if self.personas_dir.exists():
            for path in sorted(self.personas_dir.iterdir()):
                if not path.is_dir() or path.name.startswith(".") or path.name == "_template":
                    continue
                personas.append(self._persona_summary(path.name))
        return personas

    def load_persona(self, name: str) -> dict:
        base_config = _load_yaml(self.base_config_path)
        base_identity = self._load_identity(None)

        if name == "__base__":
            config = base_config
            identity = base_identity
            persona_dir = self.root
            overlay = {}
        else:
            persona_dir = self.personas_dir / name
            overlay = _load_yaml(persona_dir / "config.yaml")
            config = deep_merge(base_config, overlay)
            identity = deep_merge(base_identity, self._load_identity(name))
            self._apply_persona_defaults(config, name)

        return {
            "name": name,
            "display_name": identity.get("name") or ("Base Config" if name == "__base__" else name.title()),
            "config": config,
            "overlay": overlay,
            "identity": identity,
            "summary": self._build_summary(name, config, identity),
            "skills": self._list_skills(config),
            "channels": self._list_channels(config),
            "key_status": self.get_key_status(name),
            "status": self.get_status(name),
            "core_anchors": self._core_anchor_statuses(name),
            "paths": {
                "persona_dir": _safe_rel(persona_dir, self.root),
                "config": _safe_rel(persona_dir / "config.yaml", self.root) if name != "__base__" else "config.yaml",
                "identity": self._identity_path(name),
                "avatar": self._avatar_data_uri(name),
            },
        }

    def preview_persona_save(self, name: str, changes: dict) -> dict:
        try:
            preview = self.config_editor.preview(name, changes)
            return {"ok": True, "preview": preview}
        except (FileNotFoundError, ValueError, OSError) as e:
            return {"ok": False, "error": str(e)}

    def save_persona(self, name: str, changes: dict, identity: dict | None = None) -> dict:
        try:
            return self.config_editor.save(name, changes)
        except (FileNotFoundError, ValueError, OSError) as e:
            return {"ok": False, "error": str(e)}

    def create_persona(self, name: str) -> dict:
        return {
            "ok": False,
            "error": "Persona creation is planned for a later safe-write phase.",
        }

    def delete_persona(self, name: str) -> dict:
        return {
            "ok": False,
            "error": "Persona deletion/archive is planned for a later safe-write phase.",
        }

    # Backups

    def create_backup(self, persona: str, reason: str = "manual") -> dict:
        try:
            backup = self.backups.create_backup(persona, reason=reason)
            return {"ok": True, "backup": backup}
        except (FileNotFoundError, ValueError, OSError) as e:
            return {"ok": False, "error": str(e)}

    def list_backups(self, persona: str) -> dict:
        try:
            backups = self.backups.list_backups(persona)
            return {"ok": True, "backups": backups}
        except (FileNotFoundError, ValueError, OSError) as e:
            return {"ok": False, "error": str(e)}

    def restore_backup(self, persona: str, backup_path: str) -> dict:
        try:
            return self.backups.restore(persona, backup_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            return {"ok": False, "error": str(e)}

    def restore_last_backup(self, persona: str) -> dict:
        try:
            backups = self.backups.list_backups(persona)
            if not backups:
                return {"ok": False, "error": "No backups available."}
            return self.backups.restore(persona, backups[0]["path"])
        except (FileNotFoundError, ValueError, OSError) as e:
            return {"ok": False, "error": str(e)}

    def restore_db_before_image(self, persona: str, stamp: str) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        if not re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?(?:_\d{2})?", str(stamp or "")):
            return {"ok": False, "error": "Invalid undo stamp."}

        backup_root = (self.root / "gui_data" / "db_backups" / persona).resolve()
        backup_dir = (backup_root / stamp).resolve()
        try:
            backup_dir.relative_to(backup_root)
        except ValueError:
            return {"ok": False, "error": "Invalid undo path."}
        if not backup_dir.is_dir():
            return {"ok": False, "error": "Undo snapshot not found."}

        files = sorted(backup_dir.glob("*.json"))
        if not files:
            return {"ok": False, "error": "Undo snapshot is empty."}

        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": False, "error": "Persona database not found."}

        restored = 0
        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with conn:
                    for file in files:
                        payload = json.loads(file.read_text(encoding="utf-8"))
                        if payload.get("persona") != persona:
                            raise ValueError("Undo snapshot belongs to another persona.")
                        table = str(payload.get("table") or "")
                        delete_where = self._json_restore_db_value(payload.get("delete_where"))
                        if delete_where:
                            restored += self._delete_table_rows_for_undo(conn, table, delete_where)
                        before = self._json_restore_db_value(payload.get("before"))
                        restored += self._restore_table_before_image(conn, table, before)
        except (sqlite3.Error, OSError, ValueError, json.JSONDecodeError) as e:
            return {"ok": False, "error": f"Could not restore undo snapshot: {e}"}

        return {
            "ok": True,
            "changed": True,
            "restored": restored,
            "stamp": stamp,
            "running": self._persona_is_running(persona),
        }

    # File pickers

    def pick_voice_sample(self, persona: str, current_path: str = "") -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        if not self._window:
            return {"ok": False, "error": "File dialog requires pywebview."}
        start_dir = ""
        if current_path:
            candidate = (self.root / current_path).resolve()
            if candidate.parent.is_dir():
                start_dir = str(candidate.parent)
        try:
            import webview
            result = self._window.create_file_dialog(
                webview.OPEN_DIALOG,
                directory=start_dir,
                allow_multiple=False,
                file_types=(
                    "Audio Files (*.ogg;*.wav;*.mp3;*.flac;*.opus)",
                    "All Files (*.*)",
                ),
            )
        except Exception as e:
            return {"ok": False, "error": f"File dialog failed: {e}"}
        if not result:
            return {"ok": False}
        chosen = Path(result[0])
        if not chosen.is_file():
            return {"ok": False, "error": "Selected file does not exist."}
        dest_dir = self.personas_dir / persona / "data" / "tts" / "voice_ref"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / chosen.name
        if chosen.resolve() != dest.resolve():
            shutil.copy2(chosen, dest)
        rel_path = dest.relative_to(self.root).as_posix()
        return {"ok": True, "path": rel_path}

    def pick_folder(self, current_path: str = "") -> dict:
        if not self._window:
            return {"ok": False, "error": "Folder dialog requires pywebview."}
        start_dir = ""
        if current_path:
            candidate = Path(current_path)
            if candidate.is_dir():
                start_dir = str(candidate)
            elif candidate.parent.is_dir():
                start_dir = str(candidate.parent)
        try:
            import webview
            result = self._window.create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=start_dir,
            )
        except Exception as e:
            return {"ok": False, "error": f"Folder dialog failed: {e}"}
        if not result:
            return {"ok": False}
        chosen = result[0] if isinstance(result, (list, tuple)) else result
        return {"ok": True, "path": str(chosen)}

    def _relative_model_file(self, models_dir: str, chosen_path: str) -> str:
        if not models_dir:
            raise ValueError("Set Models Dir before choosing a model file.")
        models_path = Path(models_dir).expanduser().resolve()
        if not models_path.is_dir():
            raise ValueError("Models Dir does not exist.")

        chosen = Path(chosen_path).expanduser().resolve()
        if not chosen.is_file():
            raise ValueError("Selected file does not exist.")
        if chosen.suffix.lower() != ".gguf":
            raise ValueError("Choose a .gguf model file.")
        try:
            rel_path = chosen.relative_to(models_path)
        except ValueError as e:
            raise ValueError("Choose a .gguf file inside Models Dir.") from e
        return rel_path.as_posix()

    def pick_model_file(self, models_dir: str = "", current_file: str = "") -> dict:
        if not self._window:
            return {"ok": False, "error": "File dialog requires pywebview."}
        if not models_dir:
            return {"ok": False, "error": "Set Models Dir before choosing a model file."}

        models_path = Path(models_dir).expanduser()
        if not models_path.is_dir():
            return {"ok": False, "error": "Models Dir does not exist."}

        start_dir = str(models_path)
        if current_file:
            candidate = models_path / current_file
            if candidate.parent.is_dir():
                start_dir = str(candidate.parent)
        try:
            import webview
            result = self._window.create_file_dialog(
                webview.OPEN_DIALOG,
                directory=start_dir,
                allow_multiple=False,
                file_types=(
                    "GGUF Models (*.gguf)",
                    "All Files (*.*)",
                ),
            )
        except Exception as e:
            return {"ok": False, "error": f"File dialog failed: {e}"}
        if not result:
            return {"ok": False}
        chosen = result[0] if isinstance(result, (list, tuple)) else result
        try:
            return {"ok": True, "path": self._relative_model_file(models_dir, str(chosen))}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    # Secrets/status/logs

    def _merged_config_for(self, persona: str) -> dict:
        config = _load_yaml(self.base_config_path)
        if persona and persona != "__base__":
            config = deep_merge(config, _load_yaml(self.personas_dir / persona / "config.yaml"))
        return config

    def _secret_keys_for(self, persona: str, config: dict | None = None) -> list[str]:
        config = config or self._merged_config_for(persona)
        provider = config.get("provider", {})
        provider_type = provider.get("type", "local")
        api_key_env = provider.get("api_key_env", "") or PROVIDER_KEY_MAP.get(provider_type, "")
        keys = []
        if provider_type != "local" and api_key_env:
            keys.append(api_key_env)
        keys.append("TELEGRAM_BOT_TOKEN")
        unique = []
        for key in keys:
            if key and key not in unique:
                unique.append(key)
        return unique

    def get_key_status(self, persona: str) -> dict:
        config = self._merged_config_for(persona)

        root_env = _parse_env(self.root / ".env")
        persona_env = _parse_env(self.personas_dir / persona / ".env") if persona and persona != "__base__" else {}
        env = {**root_env, **persona_env}

        provider = config.get("provider", {})
        provider_type = provider.get("type", "local")
        api_key_env = provider.get("api_key_env", "")
        expected_api_key_env = PROVIDER_KEY_MAP.get(provider_type, "")
        effective_api_key_env = api_key_env or expected_api_key_env
        provider_key_status = {
            key: bool(env.get(value))
            for key, value in PROVIDER_KEY_MAP.items()
        }
        provider_key_sources = {
            key: (
                "persona" if persona_env.get(value)
                else "inherited" if root_env.get(value)
                else "missing"
            )
            for key, value in PROVIDER_KEY_MAP.items()
        }
        channels = config.get("channels", {})
        telegram = channels.get("telegram", {})
        telegram_enabled = telegram.get("enabled", True) if isinstance(telegram, dict) else bool(telegram)
        telegram_key = "TELEGRAM_BOT_TOKEN"
        return {
            "provider_type": provider_type,
            "api_key_env": api_key_env,
            "expected_api_key_env": expected_api_key_env,
            "api_key_set": bool(effective_api_key_env and env.get(effective_api_key_env)),
            "api_key_source": (
                "persona" if effective_api_key_env and persona_env.get(effective_api_key_env)
                else "inherited" if effective_api_key_env and root_env.get(effective_api_key_env)
                else "missing"
            ),
            "provider_key_status": provider_key_status,
            "provider_key_sources": provider_key_sources,
            "telegram_key": telegram_key,
            "telegram_set": bool(env.get(telegram_key)),
            "telegram_source": (
                "persona" if persona_env.get(telegram_key)
                else "inherited" if root_env.get(telegram_key)
                else "missing"
            ),
            "telegram_enabled": telegram_enabled,
        }

    def get_secrets(self, persona: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}

        config = self._merged_config_for(persona)
        root_env = _parse_env(self.root / ".env")
        persona_env = _parse_env(persona_dir / ".env")
        provider_type = config.get("provider", {}).get("type", "local")

        secrets = []
        for key_name in self._secret_keys_for(persona, config):
            if not SECRET_KEY_RE.match(key_name):
                continue
            persona_value = persona_env.get(key_name, "")
            root_value = root_env.get(key_name, "")
            if persona_value:
                source = "persona"
                masked = _mask_secret(persona_value)
            elif root_value:
                source = "inherited"
                masked = _mask_secret(root_value)
            else:
                source = "missing"
                masked = ""
            secrets.append({
                "key": key_name,
                "label": _secret_label(key_name),
                "masked": masked,
                "source": source,
            })

        return {"ok": True, "provider_type": provider_type, "secrets": secrets}

    def reveal_secret(self, persona: str, key_name: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        if key_name not in self._secret_keys_for(persona):
            return {"ok": False, "error": "Secret key is not editable for this persona."}
        if not SECRET_KEY_RE.match(key_name):
            return {"ok": False, "error": "Invalid secret key name."}

        root_env = _parse_env(self.root / ".env")
        persona_env = _parse_env(persona_dir / ".env")
        value = persona_env.get(key_name, "") or root_env.get(key_name, "")
        return {"ok": True, "key": key_name, "value": value}

    def save_secrets(self, persona: str, updates: dict) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Cannot edit base secrets from the GUI."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        if not isinstance(updates, dict):
            return {"ok": False, "error": "Invalid secret update payload."}

        allowed = set(self._secret_keys_for(persona))
        clean_updates: dict[str, str | None] = {}
        for key, value in updates.items():
            key_name = str(key)
            if key_name not in allowed or not SECRET_KEY_RE.match(key_name):
                return {"ok": False, "error": f"Secret key is not editable: {key_name}"}
            if value is None:
                clean_updates[key_name] = None
                continue
            if not isinstance(value, str):
                return {"ok": False, "error": f"Invalid value for {key_name}."}
            cleaned = value.strip()
            if "\x00" in cleaned or "\n" in cleaned or "\r" in cleaned:
                return {"ok": False, "error": f"Invalid value for {key_name}."}
            if len(cleaned) > 10000:
                return {"ok": False, "error": f"Secret value is too long: {key_name}."}
            clean_updates[key_name] = cleaned or None

        if not clean_updates:
            return {"ok": True, "changed": False}

        self.backups.create_backup(persona, reason="pre-secret-edit")
        env_path = persona_dir / ".env"
        before = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        _write_env_preserving(env_path, clean_updates)
        after = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        return {"ok": True, "changed": before != after}

    def get_status(self, persona: str | None = None) -> dict:
        self._cleanup_processes()
        if persona is None:
            return {name: self.get_status(name) for name in self._persona_names()}

        status = self._read_status_file(persona)
        proc = self.processes.get(persona)
        if proc:
            file_phase = status.get("phase", "")
            engine_alive = file_phase in ("starting", "running")
            status.update({
                "running": True,
                "stale": False,
                "persona": persona,
                "pid": proc.pid,
                "phase": file_phase if engine_alive else "starting",
                "started_at": proc.started_at,
                "source": "gui-process-registry",
            })
        else:
            status.setdefault("running", False)
            status.setdefault("persona", persona)
            status.setdefault("phase", "stopped")
            status.setdefault("source", "status-file" if status.get("updated_at") else "none")
        return status

    def get_log_tail(self, persona: str, lines: int = 80) -> str:
        if not persona or persona == "__base__":
            return ""
        path = self.root / "logs" / f"{persona}.log"
        if not path.exists():
            return f"No log file yet: {_safe_rel(path, self.root)}"
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-max(1, min(lines, 500)):])

    def get_lantern(self, persona: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}

        config = self._merged_config_for(persona)
        self._apply_persona_defaults(config, persona)
        db_path = Path(config.get("paths", {}).get("database") or (persona_dir / "data" / f"{persona}.db"))
        if not db_path.is_absolute():
            db_path = (self.root / db_path).resolve()

        row = self._read_lantern_row(db_path, persona)
        if not row:
            return {
                "ok": True,
                "exists": False,
                "persona": persona,
                "db_path": _safe_rel(db_path, self.root),
            }

        age_hours = self._hours_since(row.get("updated_at", ""))
        stale = age_hours is None or age_hours > LANTERN_STALE_HOURS
        expired = age_hours is None or age_hours > LANTERN_EXPIRED_HOURS
        state = "expired" if expired else "stale" if stale else "current"
        fields = {
            key: row.get(key) or ""
            for key in ("mode", "mood", "focus", "note", "open_thread")
        }
        return {
            "ok": True,
            "exists": True,
            "persona": persona,
            "resident_id": row.get("resident_id") or persona,
            "state": state,
            "stale": stale,
            "expired": expired,
            "updated_at": row.get("updated_at", ""),
            "updated_at_display": self._format_timestamp(row.get("updated_at", "")),
            "age_label": self._age_label(age_hours),
            "age_hours": age_hours,
            "fields": fields,
            "empty": not any(fields.values()),
            "db_path": _safe_rel(db_path, self.root),
        }

    def set_lantern(self, persona: str, fields: dict) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        try:
            cleaned = self._clean_lantern_fields(fields or {})
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not any(cleaned.values()):
            return {"ok": False, "error": "Lantern needs at least one field. Use Dim/Clear to leave a resting note."}

        db_path = self._persona_database_path(persona)
        before = self._read_lantern_row(db_path, persona)
        if self._lantern_fields_match(before, cleaned):
            return {
                "ok": True,
                "changed": False,
                "lantern": self.get_lantern(persona),
                "running": self._persona_is_running(persona),
            }
        updated_at = datetime.now(timezone.utc).isoformat()
        try:
            self._write_db_before_image(persona, "lantern-update", "resident_lanterns", before)
            self._upsert_lantern(db_path, persona, cleaned, updated_at)
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not update lantern: {e}"}

        return {
            "ok": True,
            "changed": True,
            "lantern": self.get_lantern(persona),
            "running": self._persona_is_running(persona),
        }

    def dim_lantern(self, persona: str, note: str | None = None) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        try:
            cleaned_note = self._clean_lantern_text(note or "Lantern dimmed; no active state set.", "note")
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        fields = {
            "mode": "",
            "mood": "",
            "focus": "",
            "note": cleaned_note,
            "open_thread": "",
        }
        db_path = self._persona_database_path(persona)
        before = self._read_lantern_row(db_path, persona)
        if self._lantern_fields_match(before, fields):
            return {
                "ok": True,
                "changed": False,
                "lantern": self.get_lantern(persona),
                "running": self._persona_is_running(persona),
            }
        updated_at = datetime.now(timezone.utc).isoformat()
        try:
            self._write_db_before_image(persona, "lantern-dim", "resident_lanterns", before)
            self._upsert_lantern(db_path, persona, fields, updated_at)
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not dim lantern: {e}"}

        return {
            "ok": True,
            "changed": True,
            "lantern": self.get_lantern(persona),
            "running": self._persona_is_running(persona),
        }

    def list_memories(self, persona: str, view: str = "active",
                      kind: str = "fact", page: int = 1,
                      page_size: int = 25) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        if view not in ("active", "archived"):
            return {"ok": False, "error": "Memory view must be active or archived."}
        if isinstance(kind, int):
            page_size = page
            page = kind
            kind = "fact"
        if kind not in ("all", "fact", "journal", "session_log"):
            return {"ok": False, "error": "Memory type must be all, fact, journal, or session_log."}

        db_path = self._persona_database_path(persona)
        page = max(1, int(page or 1))
        page_size = max(5, min(int(page_size or 25), 100))
        if not db_path.exists():
            return self._empty_memory_page(persona, view, kind, page, page_size, db_path)

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                columns = self._table_columns(conn, "memories")
                if not columns:
                    return self._empty_memory_page(persona, view, kind, page, page_size, db_path)

                select_sql = self._memory_select_sql(columns)
                where = self._memory_view_where(view, kind, columns)
                total = conn.execute(f"SELECT COUNT(*) FROM memories {where}").fetchone()[0]
                offset = (page - 1) * page_size
                rows = conn.execute(
                    f"SELECT {select_sql} FROM memories {where} "
                    "ORDER BY date DESC, id DESC LIMIT ? OFFSET ?",
                    (page_size, offset),
                ).fetchall()
                all_rows = conn.execute(
                    f"SELECT {select_sql} FROM memories ORDER BY date DESC, id DESC"
                ).fetchall()
        except (sqlite3.Error, OSError) as e:
            return {"ok": False, "error": f"Could not read memories: {e}"}

        history = self._memory_history_index([dict(row) for row in all_rows])
        items = [self._format_memory_item(dict(row), history) for row in rows]
        return {
            "ok": True,
            "persona": persona,
            "view": view,
            "kind": kind,
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_more": offset + len(items) < total,
            "items": items,
            "db_path": _safe_rel(db_path, self.root),
        }

    def get_memory_detail(self, persona: str, memory_id: int) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        try:
            memory_id = int(memory_id)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Invalid memory ID."}

        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": False, "error": "Memory database not found."}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                columns = self._table_columns(conn, "memories")
                if not columns:
                    return {"ok": False, "error": "Memory table not found."}
                select_sql = self._memory_select_sql(columns)
                rows = [
                    dict(row)
                    for row in conn.execute(
                        f"SELECT {select_sql} FROM memories ORDER BY date DESC, id DESC"
                    ).fetchall()
                ]
        except (sqlite3.Error, OSError) as e:
            return {"ok": False, "error": f"Could not read memory: {e}"}

        by_id = {row["id"]: row for row in rows}
        if memory_id not in by_id:
            return {"ok": False, "error": f"Memory #{memory_id} not found."}

        superseded_by = {
            row["supersedes"]: row["id"]
            for row in rows
            if row.get("supersedes")
        }
        current_id = memory_id
        seen = set()
        while superseded_by.get(current_id) and current_id not in seen:
            seen.add(current_id)
            current_id = superseded_by[current_id]

        chain = []
        cursor = current_id
        seen.clear()
        while cursor and cursor in by_id and cursor not in seen:
            seen.add(cursor)
            chain.append(by_id[cursor])
            cursor = by_id[cursor].get("supersedes")

        chain = self._hydrate_memory_detail_rows(db_path, chain)
        history = self._memory_history_index(rows)
        return {
            "ok": True,
            "persona": persona,
            "memory_id": memory_id,
            "current_id": current_id,
            "versions": [self._format_memory_item(row, history) for row in chain],
            "db_path": _safe_rel(db_path, self.root),
        }

    def update_memory(self, persona: str, memory_id: int, changes: dict) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        try:
            memory_id = int(memory_id)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Invalid memory ID."}

        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": False, "error": "Memory database not found."}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
                if not row:
                    return {"ok": False, "error": f"Memory #{memory_id} not found."}
                before = dict(row)
                if before.get("type") == "journal":
                    return {
                        "ok": False,
                        "error": "Journal memories are mirrors. Edit the linked journal entry instead.",
                    }

                cleaned, text_changed = self._clean_memory_update(changes or {}, before)
                if not cleaned:
                    return {"ok": True, "changed": False, "memory": self.get_memory_detail(persona, memory_id)}

                stamp = self._write_db_before_image(persona, "memory-update", "memories", before)
                assignments = [f"{key} = ?" for key in cleaned]
                values = list(cleaned.values())
                if text_changed:
                    assignments.append("embedding = NULL")
                values.append(memory_id)
                with conn:
                    conn.execute(
                        f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?",
                        values,
                    )
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not update memory: {e}"}

        return {
            "ok": True,
            "changed": True,
            "undo_stamp": stamp,
            "memory": self.get_memory_detail(persona, memory_id),
            "running": self._persona_is_running(persona),
        }

    def delete_memory(self, persona: str, memory_id: int) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        try:
            memory_id = int(memory_id)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Invalid memory ID."}

        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": False, "error": "Memory database not found."}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
                if not row:
                    return {"ok": False, "error": f"Memory #{memory_id} not found."}
                deleted = dict(row)
                if deleted.get("type") == "journal":
                    return {
                        "ok": False,
                        "error": "Journal memories are mirrors. Delete the linked journal entry instead.",
                    }
                children = [
                    dict(child)
                    for child in conn.execute(
                        "SELECT * FROM memories WHERE supersedes = ?",
                        (memory_id,),
                    ).fetchall()
                ]
                stamp = self._write_db_before_image(
                    persona,
                    "memory-delete",
                    "memories",
                    [deleted] + children,
                )
                with conn:
                    conn.execute(
                        "UPDATE memories SET supersedes = ? WHERE supersedes = ?",
                        (deleted.get("supersedes"), memory_id),
                    )
                    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not delete memory: {e}"}

        return {
            "ok": True,
            "changed": True,
            "undo_stamp": stamp,
            "deleted_id": memory_id,
            "relinked": len(children),
            "running": self._persona_is_running(persona),
        }

    def delete_all_memories(self, persona: str, include_journal: bool = False) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": True, "changed": False, "deleted": 0}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                where = "" if include_journal else "WHERE type != 'journal'"
                rows = [dict(row) for row in conn.execute(f"SELECT * FROM memories {where}").fetchall()]
                if not rows:
                    return {"ok": True, "changed": False, "deleted": 0}
                stamp = self._write_db_before_image(
                    persona,
                    "memory-delete-all",
                    "memories",
                    rows,
                )
                with conn:
                    conn.execute(f"DELETE FROM memories {where}")
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not delete memories: {e}"}

        return {
            "ok": True,
            "changed": True,
            "undo_stamp": stamp,
            "deleted": len(rows),
            "running": self._persona_is_running(persona),
        }

    def list_journal_entries(self, persona: str, view: str = "active",
                             entry_type: str = "all", page: int = 1,
                             page_size: int = 25) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        if view not in ("active", "resolved", "all"):
            return {"ok": False, "error": "Journal view must be active, resolved, or all."}
        allowed_types = {
            "all", "event", "preference", "topic", "tone",
            "open_thread", "follow_up", "reflection",
        }
        if entry_type not in allowed_types:
            return {"ok": False, "error": "Journal type filter is not recognized."}

        db_path = self._persona_database_path(persona)
        page = max(1, int(page or 1))
        page_size = max(5, min(int(page_size or 25), 100))
        if not db_path.exists():
            return self._empty_journal_page(persona, view, entry_type, page, page_size, db_path)

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                columns = self._table_columns(conn, "journal_entries")
                if not columns:
                    return self._empty_journal_page(persona, view, entry_type, page, page_size, db_path)
                select_sql = self._journal_select_sql(columns)
                where, params = self._journal_where(view, entry_type, columns)
                total = conn.execute(f"SELECT COUNT(*) FROM journal_entries {where}", params).fetchone()[0]
                offset = (page - 1) * page_size
                rows = conn.execute(
                    f"SELECT {select_sql} FROM journal_entries {where} "
                    "ORDER BY date DESC, id DESC LIMIT ? OFFSET ?",
                    (*params, page_size, offset),
                ).fetchall()
        except (sqlite3.Error, OSError) as e:
            return {"ok": False, "error": f"Could not read journal entries: {e}"}

        items = [self._format_journal_entry(dict(row), detail=False) for row in rows]
        return {
            "ok": True,
            "persona": persona,
            "view": view,
            "entry_type": entry_type,
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_more": offset + len(items) < total,
            "items": items,
            "db_path": _safe_rel(db_path, self.root),
        }

    def get_journal_entry(self, persona: str, entry_id: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        entry_id = str(entry_id or "").strip()
        if not entry_id:
            return {"ok": False, "error": "Journal entry ID is required."}

        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": False, "error": "Journal database not found."}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                columns = self._table_columns(conn, "journal_entries")
                if not columns:
                    return {"ok": False, "error": "Journal table not found."}
                select_sql = self._journal_select_sql(columns)
                row = conn.execute(
                    f"SELECT {select_sql} FROM journal_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()
        except (sqlite3.Error, OSError) as e:
            return {"ok": False, "error": f"Could not read journal entry: {e}"}

        if not row:
            return {"ok": False, "error": f"Journal entry {entry_id} not found."}
        return {
            "ok": True,
            "persona": persona,
            "entry": self._format_journal_entry(dict(row), detail=True),
            "db_path": _safe_rel(db_path, self.root),
        }

    def update_journal_entry(self, persona: str, entry_id: str, changes: dict) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        entry_id = str(entry_id or "").strip()
        if not entry_id or entry_id.startswith("_"):
            return {"ok": False, "error": "Choose a transient journal entry."}
        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": False, "error": "Journal database not found."}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                entry_row = conn.execute(
                    "SELECT * FROM journal_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()
                if not entry_row:
                    return {"ok": False, "error": f"Journal entry {entry_id} not found."}
                before_entry = dict(entry_row)
                cleaned, mirror_stale = self._clean_journal_update(changes or {}, before_entry)
                if not cleaned:
                    return {"ok": True, "changed": False, "entry": self.get_journal_entry(persona, entry_id)}

                journal_file = f"entries/{entry_id}.md"
                before_mirrors = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM memories WHERE type = 'journal' AND journal_file = ?",
                        (journal_file,),
                    ).fetchall()
                ]
                stamp = self._write_db_before_image(
                    persona,
                    "journal-update-entry",
                    "journal_entries",
                    before_entry,
                )
                self._write_db_before_image(
                    persona,
                    "journal-update-memory",
                    "memories",
                    before_mirrors,
                    stamp=stamp,
                    delete_where={"type": "journal", "journal_file": journal_file} if not before_mirrors else None,
                )

                assignments = [f"{key} = ?" for key in cleaned]
                values = list(cleaned.values())
                values.append(entry_id)
                with conn:
                    conn.execute(
                        f"UPDATE journal_entries SET {', '.join(assignments)} WHERE id = ?",
                        values,
                    )
                    updated_entry = dict(before_entry)
                    updated_entry.update(cleaned)
                    if mirror_stale:
                        self._upsert_journal_mirror(conn, updated_entry)
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not update journal entry: {e}"}

        return {
            "ok": True,
            "changed": True,
            "undo_stamp": stamp,
            "entry": self.get_journal_entry(persona, entry_id),
            "running": self._persona_is_running(persona),
        }

    def delete_journal_entry(self, persona: str, entry_id: str) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        entry_id = str(entry_id or "").strip()
        if not entry_id or entry_id.startswith("_"):
            return {"ok": False, "error": "Choose a transient journal entry."}
        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": False, "error": "Journal database not found."}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                entry_row = conn.execute(
                    "SELECT * FROM journal_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()
                if not entry_row:
                    return {"ok": False, "error": f"Journal entry {entry_id} not found."}
                before_entry = dict(entry_row)
                journal_file = f"entries/{entry_id}.md"
                before_mirrors = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM memories WHERE type = 'journal' AND journal_file = ?",
                        (journal_file,),
                    ).fetchall()
                ]
                stamp = self._write_db_before_image(
                    persona,
                    "journal-delete-entry",
                    "journal_entries",
                    before_entry,
                )
                self._write_db_before_image(
                    persona,
                    "journal-delete-memory",
                    "memories",
                    before_mirrors,
                    stamp=stamp,
                )
                with conn:
                    conn.execute("DELETE FROM journal_entries WHERE id = ?", (entry_id,))
                    conn.execute(
                        "DELETE FROM memories WHERE type = 'journal' AND journal_file = ?",
                        (journal_file,),
                    )
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not delete journal entry: {e}"}

        return {
            "ok": True,
            "changed": True,
            "undo_stamp": stamp,
            "deleted_id": entry_id,
            "deleted_mirrors": len(before_mirrors),
            "running": self._persona_is_running(persona),
        }

    def delete_all_journal_entries(self, persona: str) -> dict:
        valid = self._validate_persona_for_db_write(persona)
        if not valid["ok"]:
            return valid
        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {"ok": True, "changed": False, "deleted": 0}

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                entries = [dict(row) for row in conn.execute("SELECT * FROM journal_entries").fetchall()]
                if not entries:
                    return {"ok": True, "changed": False, "deleted": 0}
                journal_files = [f"entries/{row['id']}.md" for row in entries]
                placeholders = ", ".join("?" for _ in journal_files)
                mirrors = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM memories WHERE type = 'journal' "
                        f"AND journal_file IN ({placeholders})",
                        tuple(journal_files),
                    ).fetchall()
                ]
                stamp = self._write_db_before_image(
                    persona,
                    "journal-delete-all-entries",
                    "journal_entries",
                    entries,
                )
                self._write_db_before_image(
                    persona,
                    "journal-delete-all-memories",
                    "memories",
                    mirrors,
                    stamp=stamp,
                )
                with conn:
                    conn.execute("DELETE FROM journal_entries")
                    conn.execute(
                        "DELETE FROM memories WHERE type = 'journal' "
                        f"AND journal_file IN ({placeholders})",
                        tuple(journal_files),
                    )
        except (sqlite3.Error, OSError, ValueError) as e:
            return {"ok": False, "error": f"Could not delete journal entries: {e}"}

        return {
            "ok": True,
            "changed": True,
            "undo_stamp": stamp,
            "deleted": len(entries),
            "deleted_mirrors": len(mirrors),
            "running": self._persona_is_running(persona),
        }

    def get_core_anchor(self, persona: str, anchor_id: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        if anchor_id not in CORE_ANCHOR_TEMPLATES:
            return {"ok": False, "error": "Core anchor must be _self, _user, or _relationship."}

        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return {
                "ok": True,
                "persona": persona,
                "anchor": self._empty_core_anchor(anchor_id),
                "db_path": _safe_rel(db_path, self.root),
            }

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                columns = self._table_columns(conn, "identity")
                if not columns:
                    return {
                        "ok": True,
                        "persona": persona,
                        "anchor": self._empty_core_anchor(anchor_id),
                        "db_path": _safe_rel(db_path, self.root),
                    }
                select_sql = self._core_anchor_select_sql(columns)
                row = conn.execute(
                    f"SELECT {select_sql} FROM identity WHERE id = ?",
                    (anchor_id,),
                ).fetchone()
        except (sqlite3.Error, OSError) as e:
            return {"ok": False, "error": f"Could not read core anchor: {e}"}

        anchor = (
            self._format_core_anchor(dict(row))
            if row else self._empty_core_anchor(anchor_id)
        )
        return {
            "ok": True,
            "persona": persona,
            "anchor": anchor,
            "db_path": _safe_rel(db_path, self.root),
        }

    def set_core_anchor(self, persona: str, anchor_id: str, sections: dict) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        if anchor_id not in CORE_ANCHOR_TEMPLATES:
            return {"ok": False, "error": "Core anchor must be _self, _user, or _relationship."}

        db_path = self._persona_database_path(persona)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            before = self._read_core_anchor_row(db_path, anchor_id)
            cleaned = self._clean_core_anchor_sections(anchor_id, sections or {}, before)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except (sqlite3.Error, OSError) as e:
            return {"ok": False, "error": f"Could not read core anchor: {e}"}

        if before is None and not any(value.strip() for value in cleaned.values()):
            return {
                "ok": True,
                "changed": False,
                "anchor": self._empty_core_anchor(anchor_id),
                "running": self._persona_is_running(persona),
            }

        before_sections = self._core_sections_from_row(before)
        if before is not None and before_sections == cleaned:
            current = self.get_core_anchor(persona, anchor_id)
            return {
                "ok": True,
                "changed": False,
                "anchor": current.get("anchor") or self._empty_core_anchor(anchor_id),
                "running": self._persona_is_running(persona),
            }

        template = CORE_ANCHOR_TEMPLATES[anchor_id]
        updated_at = datetime.now(timezone.utc).isoformat()
        try:
            self._write_db_before_image(persona, f"core-{anchor_id}", "identity", before)
            self._upsert_core_anchor(
                db_path,
                anchor_id,
                template["title"],
                cleaned,
                updated_at,
            )
        except (sqlite3.Error, OSError) as e:
            return {"ok": False, "error": f"Could not update core anchor: {e}"}

        current = self.get_core_anchor(persona, anchor_id)
        return {
            "ok": True,
            "changed": True,
            "anchor": current.get("anchor") or self._empty_core_anchor(anchor_id),
            "running": self._persona_is_running(persona),
        }

    def _core_anchor_statuses(self, persona: str) -> dict:
        anchor_ids = ("_self", "_user", "_relationship")
        statuses = {
            anchor_id: {"exists": False, "has_content": False}
            for anchor_id in anchor_ids
        }
        if not persona or persona == "__base__":
            return statuses
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return statuses
        db_path = self._persona_database_path(persona)
        if not db_path.exists():
            return statuses
        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                columns = self._table_columns(conn, "identity")
                if not columns:
                    return statuses
                select_sql = self._core_anchor_select_sql(columns)
                rows = conn.execute(
                    f"SELECT {select_sql} FROM identity WHERE id IN (?, ?, ?)",
                    anchor_ids,
                ).fetchall()
        except (sqlite3.Error, OSError):
            return statuses

        for row in rows:
            anchor = self._format_core_anchor(dict(row))
            statuses[anchor["id"]] = {
                "exists": True,
                "has_content": not anchor["empty"],
            }
        return statuses

    def stop_pulse(self, persona: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona before stopping Pulse."}
        if not (self.personas_dir / persona).is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}

        data_dir = self._persona_data_dir(persona)
        data_dir.mkdir(parents=True, exist_ok=True)
        sentinel = data_dir / "shutdown_requested"
        sentinel.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

        status = self.get_status(persona)
        status["phase"] = "stopping"
        return {"ok": True, "status": status}

    def start_pulse(self, persona: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona before starting Pulse."}
        if not (self.personas_dir / persona).is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}

        self._cleanup_processes()
        running = [
            name for name, info in self.processes.items()
            if info.process is not None and info.process.poll() is None
        ]
        if running:
            return {
                "ok": False,
                "error": f"Pulse is already running for {running[0]}. Stop it before starting another persona.",
            }

        status = self.get_status(persona)
        if status.get("running") and not status.get("stale"):
            return {
                "ok": False,
                "error": f"Pulse appears to already be running for {persona}. Refresh status or stop it first.",
            }

        data_dir = self._persona_data_dir(persona)
        data_dir.mkdir(parents=True, exist_ok=True)
        sentinel = data_dir / "shutdown_requested"
        if sentinel.exists():
            sentinel.unlink()

        creationflags = 0
        if os.name == "nt":
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        process = subprocess.Popen(
            [sys.executable, "pulse.py", "--persona", persona],
            cwd=str(self.root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self.processes[persona] = ProcessInfo(
            persona=persona,
            pid=process.pid,
            started_at=datetime.now(timezone.utc).isoformat(),
            process=process,
        )
        return {"ok": True, "status": self.get_status(persona)}

    # Shell helpers

    def check_close_request(self) -> list[str] | None:
        req = self._close_requested
        self._close_requested = None
        return req

    def get_running_personas(self) -> list[str]:
        self._cleanup_processes()
        running = set(
            name for name, info in self.processes.items()
            if info.process is not None and info.process.poll() is None
        )
        for name in self._persona_names():
            status = self._read_status_file(name)
            if status.get("running") and not status.get("stale"):
                running.add(name)
        return sorted(running)

    def stop_all_and_close(self) -> dict:
        for persona in list(self.processes.keys()):
            self.stop_pulse(persona)
        self._force_close = True
        if self._window:
            self._window.destroy()
        return {"ok": True}

    def close_keep_running(self) -> dict:
        self._force_close = True
        if self._window:
            self._window.destroy()
        return {"ok": True}

    def open_folder(self, persona: str) -> dict:
        persona_dir = self.root if persona == "__base__" else self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Folder not found: {persona}"}
        self._open_path(persona_dir)
        return {"ok": True}

    def _open_path(self, path: Path) -> None:
        if os.name == "nt":
            os.startfile(str(path))
        else:
            import subprocess as _sp
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            _sp.Popen([opener, str(path)])

    # Preferences

    def get_prefs(self) -> dict:
        try:
            return json.loads(self.prefs_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save_prefs(self, prefs: dict) -> dict:
        self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.prefs_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(prefs or {}, indent=2), encoding="utf-8")
        tmp.replace(self.prefs_path)
        return {"ok": True}

    # Internal helpers

    def _persona_names(self) -> list[str]:
        if not self.personas_dir.exists():
            return []
        return [
            p.name for p in sorted(self.personas_dir.iterdir())
            if p.is_dir() and not p.name.startswith(".") and p.name != "_template"
        ]

    def _cleanup_processes(self) -> None:
        finished = [
            name for name, info in self.processes.items()
            if info.process is not None and info.process.poll() is not None
        ]
        for name in finished:
            self.processes.pop(name, None)

    def _persona_data_dir(self, persona: str) -> Path:
        return self.personas_dir / persona / "data"

    def _persona_database_path(self, persona: str) -> Path:
        config = self._merged_config_for(persona)
        self._apply_persona_defaults(config, persona)
        db_path = Path(config.get("paths", {}).get("database") or self._persona_data_dir(persona) / f"{persona}.db")
        if not db_path.is_absolute():
            db_path = (self.root / db_path).resolve()
        return db_path

    def _validate_persona_for_db_write(self, persona: str) -> dict:
        if not persona or persona == "__base__":
            return {"ok": False, "error": "Choose a persona first."}
        persona_dir = self.personas_dir / persona
        if not persona_dir.is_dir():
            return {"ok": False, "error": f"Persona not found: {persona}"}
        return {"ok": True}

    def _persona_is_running(self, persona: str) -> bool:
        status = self.get_status(persona)
        return bool(status.get("running")) and not bool(status.get("stale"))

    def _clean_lantern_fields(self, fields: dict) -> dict[str, str]:
        unknown = set(fields) - set(LANTERN_FIELDS)
        if unknown:
            raise ValueError(f"Unknown lantern field: {sorted(unknown)[0]}")
        return {
            key: self._clean_lantern_text(fields.get(key, ""), key)
            for key in LANTERN_FIELDS
        }

    def _clean_lantern_text(self, value: Any, field: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"Lantern field {field} must be text.")
        if "\x00" in value:
            raise ValueError(f"Lantern field {field} contains an invalid character.")
        value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(value) > 2000:
            raise ValueError(f"Lantern field {field} is too long.")
        return value

    def _lantern_fields_match(self, row: dict | None, fields: dict[str, str]) -> bool:
        if not row:
            return False
        return all((row.get(key) or "") == (fields.get(key) or "") for key in LANTERN_FIELDS)

    def _core_sections_from_row(self, row: dict | None) -> dict[str, str]:
        if not row:
            return {}
        raw = row.get("sections") or "{}"
        try:
            sections = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            sections = {}
        if not isinstance(sections, dict):
            return {}
        return {str(key): str(value or "") for key, value in sections.items()}

    def _clean_core_anchor_sections(self, anchor_id: str, sections: dict,
                                    before: dict | None) -> dict[str, str]:
        if not isinstance(sections, dict):
            raise ValueError("Core sections must be an object.")
        template_keys = list(CORE_ANCHOR_TEMPLATES[anchor_id]["sections"])
        before_sections = self._core_sections_from_row(before)
        allowed = set(template_keys) | set(before_sections)
        for key in sections:
            if key not in allowed:
                raise ValueError(f"Unknown core section: {key}")

        ordered_keys = list(before_sections) if before_sections else template_keys[:]
        for key in sections:
            if key not in ordered_keys:
                ordered_keys.append(key)

        cleaned = {}
        total_length = 0
        for key in ordered_keys:
            value = sections.get(key, before_sections.get(key, ""))
            if not isinstance(value, str):
                raise ValueError(f"Core section {key} must be text.")
            value = value.replace("\r\n", "\n").replace("\r", "\n")
            if "\x00" in value:
                raise ValueError(f"Core section {key} contains an invalid character.")
            if len(value) > 12000:
                raise ValueError(f"Core section {key} is too long.")
            total_length += len(value)
            cleaned[key] = value
        if total_length > 50000:
            raise ValueError("Core anchor is too long.")
        return cleaned

    def _ensure_lantern_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS resident_lanterns (
                resident_id TEXT PRIMARY KEY,
                mode TEXT,
                mood TEXT,
                focus TEXT,
                note TEXT,
                open_thread TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _upsert_lantern(self, db_path: Path, persona: str,
                        fields: dict[str, str], updated_at: str) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect_sqlite(db_path)) as conn:
            self._ensure_lantern_table(conn)
            with conn:
                conn.execute(
                    """
                    INSERT INTO resident_lanterns
                        (resident_id, mode, mood, focus, note, open_thread, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(resident_id) DO UPDATE SET
                        mode = excluded.mode,
                        mood = excluded.mood,
                        focus = excluded.focus,
                        note = excluded.note,
                        open_thread = excluded.open_thread,
                        updated_at = excluded.updated_at
                    """,
                    (
                        persona,
                        fields["mode"],
                        fields["mood"],
                        fields["focus"],
                        fields["note"],
                        fields["open_thread"],
                        updated_at,
                    ),
                )

    def _ensure_identity_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identity (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                sections TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_updated TEXT
            )
            """
        )

    def _upsert_core_anchor(self, db_path: Path, anchor_id: str, title: str,
                            sections: dict[str, str], updated_at: str) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect_sqlite(db_path)) as conn:
            self._ensure_identity_table(conn)
            with conn:
                conn.execute(
                    """
                    INSERT INTO identity (id, title, sections, last_updated)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        sections = excluded.sections,
                        last_updated = excluded.last_updated
                    """,
                    (
                        anchor_id,
                        title,
                        json.dumps(sections, ensure_ascii=False),
                        updated_at,
                    ),
                )

    def _clean_memory_update(self, changes: dict, before: dict) -> tuple[dict, bool]:
        if not isinstance(changes, dict):
            raise ValueError("Memory changes must be an object.")
        allowed = {
            "text", "tags", "importance", "status", "confidence", "source",
            "time_sensitive", "valid_until",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown memory field: {sorted(unknown)[0]}")

        cleaned: dict[str, Any] = {}
        text_changed = False
        if "text" in changes:
            text = self._clean_db_text(changes.get("text"), "Memory text", max_len=50000)
            if not text.strip():
                raise ValueError("Memory text cannot be empty.")
            if text != (before.get("text") or ""):
                cleaned["text"] = text
                text_changed = True

        if "tags" in changes:
            tags = self._clean_tag_list(changes.get("tags"))
            before_tags = self._decode_tags(before.get("tags"))
            if tags != before_tags:
                cleaned["tags"] = json.dumps(tags, ensure_ascii=False)

        if "importance" in changes:
            importance = self._clean_int(changes.get("importance"), "Memory importance", 1, 10)
            if importance != before.get("importance"):
                cleaned["importance"] = importance

        for field, allowed_values in (
            ("status", MEMORY_STATUSES),
            ("confidence", MEMORY_CONFIDENCES),
            ("source", MEMORY_SOURCES),
        ):
            if field in changes:
                value = str(changes.get(field) or "").strip()
                if value not in allowed_values:
                    raise ValueError(f"Invalid memory {field}: {value}")
                if field == "status":
                    before_status = before.get("status") or ""
                    if value != before_status and "superseded" in (value, before_status):
                        raise ValueError(
                            "Superseding is managed automatically and cannot be "
                            "set or cleared here."
                        )
                if value != (before.get(field) or ""):
                    cleaned[field] = value

        if "time_sensitive" in changes:
            value = int(bool(changes.get("time_sensitive")))
            before_value = before.get("time_sensitive")
            before_bool = int(bool(before_value)) if before_value is not None else None
            if value != before_bool:
                cleaned["time_sensitive"] = value

        if "valid_until" in changes:
            value = self._clean_optional_db_text(changes.get("valid_until"), "Valid until", max_len=80)
            if value != (before.get("valid_until") or ""):
                cleaned["valid_until"] = value or None

        return cleaned, text_changed

    def _clean_journal_update(self, changes: dict, before: dict) -> tuple[dict, bool]:
        if not isinstance(changes, dict):
            raise ValueError("Journal changes must be an object.")
        allowed = {
            "title", "entry_type", "content", "why_it_mattered",
            "search_summary", "tags", "importance", "pinned", "resolved",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown journal field: {sorted(unknown)[0]}")

        cleaned: dict[str, Any] = {}
        mirror_stale = False
        textual_changed = False

        if "title" in changes:
            title = self._clean_optional_db_text(changes.get("title"), "Journal title", max_len=500)
            if title != (before.get("title") or ""):
                cleaned["title"] = title or None
                textual_changed = True

        if "entry_type" in changes:
            entry_type = str(changes.get("entry_type") or "").strip()
            if entry_type not in VALID_ENTRY_TYPES:
                raise ValueError(f"Invalid journal entry type: {entry_type}")
            if entry_type != (before.get("entry_type") or ""):
                cleaned["entry_type"] = entry_type
                textual_changed = True

        if "content" in changes:
            content = self._clean_db_text(changes.get("content"), "Journal content", max_len=120000)
            if not content.strip():
                raise ValueError("Journal content cannot be empty.")
            if content != (before.get("content") or ""):
                cleaned["content"] = content
                cleaned["summary_needs_review"] = 1
                textual_changed = True
                mirror_stale = True

        if "why_it_mattered" in changes:
            why = self._clean_optional_db_text(changes.get("why_it_mattered"), "Why it mattered", max_len=12000)
            if why != (before.get("why_it_mattered") or ""):
                cleaned["why_it_mattered"] = why or None
                textual_changed = True

        if "search_summary" in changes:
            summary = self._clean_optional_db_text(changes.get("search_summary"), "Search summary", max_len=12000)
            if summary != (before.get("search_summary") or ""):
                cleaned["search_summary"] = summary or None
                cleaned["summary_needs_review"] = int(search_summary_is_thin(summary))
                textual_changed = True
                mirror_stale = True

        if "tags" in changes:
            tags = self._clean_tag_list(changes.get("tags"))
            before_tags = self._decode_tags(before.get("tags"))
            if tags != before_tags:
                cleaned["tags"] = json.dumps(tags, ensure_ascii=False)
                textual_changed = True

        if "importance" in changes:
            importance = self._clean_int(changes.get("importance"), "Journal importance", 1, 10)
            if importance != before.get("importance"):
                cleaned["importance"] = importance

        if "pinned" in changes:
            pinned = int(bool(changes.get("pinned")))
            if pinned != int(bool(before.get("pinned"))):
                cleaned["pinned"] = pinned

        if "resolved" in changes:
            entry_type = cleaned.get("entry_type") or before.get("entry_type")
            resolved = None
            if entry_type in ("open_thread", "follow_up"):
                resolved = int(bool(changes.get("resolved")))
            if resolved != before.get("resolved"):
                cleaned["resolved"] = resolved
        elif "entry_type" in cleaned:
            if cleaned["entry_type"] in ("open_thread", "follow_up"):
                if before.get("resolved") is None:
                    cleaned["resolved"] = 0
            elif before.get("resolved") is not None:
                cleaned["resolved"] = None

        if textual_changed:
            cleaned["date"] = datetime.now(timezone.utc).isoformat()
            mirror_stale = True
        elif cleaned:
            cleaned["date"] = datetime.now(timezone.utc).isoformat()

        return cleaned, mirror_stale

    def _clean_db_text(self, value: Any, label: str, *, max_len: int) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be text.")
        if "\x00" in value:
            raise ValueError(f"{label} contains an invalid character.")
        value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(value) > max_len:
            raise ValueError(f"{label} is too long.")
        return value

    def _clean_optional_db_text(self, value: Any, label: str, *, max_len: int) -> str:
        if value is None:
            return ""
        return self._clean_db_text(value, label, max_len=max_len)

    def _clean_tag_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        raw = value.split(",") if isinstance(value, str) else value
        if not isinstance(raw, list):
            raise ValueError("Tags must be a list or comma-separated text.")
        tags = []
        seen = set()
        for item in raw:
            tag = str(item or "").strip()
            if not tag:
                continue
            if "\x00" in tag:
                raise ValueError("Tag contains an invalid character.")
            if len(tag) > 80:
                raise ValueError("Tag is too long.")
            key = tag.lower()
            if key not in seen:
                seen.add(key)
                tags.append(tag)
        if len(tags) > 50:
            raise ValueError("Too many tags.")
        return tags

    def _clean_int(self, value: Any, label: str, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{label} must be a number.") from e
        if number < minimum or number > maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")
        return number

    def _upsert_journal_mirror(self, conn: sqlite3.Connection,
                               entry: dict) -> None:
        entry_id = str(entry.get("id") or "")
        journal_file = f"entries/{entry_id}.md"
        tags = ["journal"] + self._decode_tags(entry.get("tags"))
        display_text = journal_memory_display_text({
            **entry,
            "tags": self._decode_tags(entry.get("tags")),
        })
        now = entry.get("date") or datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "SELECT id FROM memories WHERE type = 'journal' AND journal_file = ? "
            "ORDER BY id ASC LIMIT 1",
            (journal_file,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE memories SET text = ?, tags = ?, importance = ?, "
                "embedding = ?, journal_file = ?, date = ?, status = ?, "
                "confidence = ?, source = ? WHERE id = ?",
                (
                    display_text,
                    json.dumps(tags, ensure_ascii=False),
                    5,
                    None,
                    journal_file,
                    now,
                    "current",
                    "medium",
                    "model_extracted",
                    row["id"],
                ),
            )
            return

        conn.execute(
            "INSERT INTO memories "
            "(text, tags, type, importance, embedding, journal_file, date, "
            "status, confidence, source) "
            "VALUES (?, ?, 'journal', ?, ?, ?, ?, ?, ?, ?)",
            (
                display_text,
                json.dumps(tags, ensure_ascii=False),
                5,
                None,
                journal_file,
                now,
                "current",
                "medium",
                "model_extracted",
            ),
        )

    def _write_db_before_image(self, persona: str, action: str,
                               table: str, before: Any,
                               stamp: str | None = None,
                               delete_where: dict[str, Any] | None = None) -> str:
        stamp = stamp or self._new_db_backup_stamp(persona)
        safe_action = re.sub(r"[^a-zA-Z0-9_-]+", "-", action).strip("-") or "db-edit"
        out_dir = self.root / "gui_data" / "db_backups" / persona / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "persona": persona,
            "action": action,
            "table": table,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "before": self._json_safe_db_value(before),
            "delete_where": self._json_safe_db_value(delete_where),
        }
        (out_dir / f"{safe_action}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return stamp

    def _new_db_backup_stamp(self, persona: str) -> str:
        base = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        root = self.root / "gui_data" / "db_backups" / persona
        stamp = base
        counter = 2
        while (root / stamp).exists():
            stamp = f"{base}_{counter:02d}"
            counter += 1
        return stamp

    def _json_safe_db_value(self, value: Any) -> Any:
        if isinstance(value, (bytes, bytearray)):
            return {"__pulse_bytes_b64": base64.b64encode(bytes(value)).decode("ascii")}
        if isinstance(value, dict):
            return {str(key): self._json_safe_db_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_safe_db_value(item) for item in value]
        return value

    def _json_restore_db_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"__pulse_bytes_b64"}:
                return base64.b64decode(value["__pulse_bytes_b64"].encode("ascii"))
            return {key: self._json_restore_db_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_restore_db_value(item) for item in value]
        return value

    def _restore_table_before_image(self, conn: sqlite3.Connection,
                                    table: str, before: Any) -> int:
        allowed_tables = {"memories", "journal_entries", "resident_lanterns", "identity"}
        if table not in allowed_tables:
            raise ValueError(f"Unsupported undo table: {table}")
        if before is None:
            return 0

        rows = before if isinstance(before, list) else [before]
        table_columns = self._table_columns(conn, table)
        restored = 0
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Undo row is malformed.")
            if not row:
                continue
            columns = [column for column in row.keys() if column in table_columns]
            if not columns:
                continue
            placeholders = ", ".join("?" for _ in columns)
            names = ", ".join(f'"{column}"' for column in columns)
            values = [row[column] for column in columns]
            conn.execute(
                f"INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})",
                values,
            )
            restored += 1
        return restored

    def _delete_table_rows_for_undo(self, conn: sqlite3.Connection,
                                    table: str, where: dict[str, Any]) -> int:
        allowed_tables = {"memories", "journal_entries", "resident_lanterns", "identity"}
        if table not in allowed_tables:
            raise ValueError(f"Unsupported undo table: {table}")
        if not isinstance(where, dict) or not where:
            return 0
        table_columns = self._table_columns(conn, table)
        conditions = []
        values = []
        for column, value in where.items():
            if column not in table_columns:
                raise ValueError(f"Unsupported undo column: {column}")
            conditions.append(f'"{column}" = ?')
            values.append(value)
        cur = conn.execute(
            f"DELETE FROM {table} WHERE {' AND '.join(conditions)}",
            values,
        )
        return cur.rowcount

    def _empty_memory_page(self, persona: str, view: str, kind: str, page: int,
                           page_size: int, db_path: Path) -> dict:
        return {
            "ok": True,
            "persona": persona,
            "view": view,
            "kind": kind,
            "page": page,
            "page_size": page_size,
            "total": 0,
            "has_more": False,
            "items": [],
            "db_path": _safe_rel(db_path, self.root),
        }

    def _empty_journal_page(self, persona: str, view: str, entry_type: str,
                            page: int, page_size: int, db_path: Path) -> dict:
        return {
            "ok": True,
            "persona": persona,
            "view": view,
            "entry_type": entry_type,
            "page": page,
            "page_size": page_size,
            "total": 0,
            "has_more": False,
            "items": [],
            "db_path": _safe_rel(db_path, self.root),
        }

    def _table_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.Error:
            return set()
        return {row[1] for row in rows}

    def _memory_select_sql(self, columns: set[str]) -> str:
        required = [
            "id", "text", "tags", "type", "importance", "retrieval_count",
            "last_accessed", "supersedes", "journal_file", "date",
        ]
        optional = ["status", "confidence", "source", "last_confirmed", "time_sensitive", "valid_until"]
        parts = [name if name in columns else f"NULL AS {name}" for name in required + optional]
        return ", ".join(parts)

    def _journal_select_sql(self, columns: set[str]) -> str:
        fields = [
            "id", "author", "title", "entry_type", "content",
            "why_it_mattered", "search_summary", "summary_needs_review",
            "tags", "importance", "pinned", "resolved", "date",
        ]
        parts = [name if name in columns else f"NULL AS {name}" for name in fields]
        return ", ".join(parts)

    def _core_anchor_select_sql(self, columns: set[str]) -> str:
        fields = ["id", "title", "sections", "created_at", "last_updated"]
        parts = [name if name in columns else f"NULL AS {name}" for name in fields]
        return ", ".join(parts)

    def _memory_view_where(self, view: str, kind: str, columns: set[str]) -> str:
        filters = []
        if view == "archived":
            if "status" not in columns:
                return "WHERE 0"
            filters.append("status = 'archived'")
        elif "status" in columns:
            filters.append("(status IS NULL OR status != 'archived')")

        if kind != "all":
            if "type" not in columns:
                return "WHERE 0"
            filters.append(f"type = '{kind}'")

        if "supersedes" in columns:
            filters.append(
                "id NOT IN "
                "(SELECT supersedes FROM memories WHERE supersedes IS NOT NULL)"
            )
        return f"WHERE {' AND '.join(filters)}" if filters else ""

    def _journal_where(self, view: str, entry_type: str,
                       columns: set[str]) -> tuple[str, tuple]:
        filters = []
        params: list[Any] = []
        if view == "active" and "resolved" in columns:
            filters.append("(resolved IS NULL OR resolved = 0)")
        elif view == "resolved":
            if "resolved" not in columns:
                return "WHERE 0", ()
            filters.append("resolved = 1")
        if entry_type != "all":
            if "entry_type" not in columns:
                return "WHERE 0", ()
            filters.append("entry_type = ?")
            params.append(entry_type)
        return (f"WHERE {' AND '.join(filters)}" if filters else ""), tuple(params)

    def _decode_tags(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
        return []

    def _memory_history_index(self, rows: list[dict]) -> dict[int, dict]:
        by_id = {row["id"]: row for row in rows}
        superseded_by = {
            row["supersedes"]: row["id"]
            for row in rows
            if row.get("supersedes")
        }
        index = {}
        for row in rows:
            count = 1
            cursor = row.get("supersedes")
            seen = {row["id"]}
            while cursor and cursor in by_id and cursor not in seen:
                seen.add(cursor)
                count += 1
                cursor = by_id[cursor].get("supersedes")
            index[row["id"]] = {
                "version_count": count,
                "supersedes": row.get("supersedes"),
                "replaced_by": superseded_by.get(row["id"]),
            }
        return index

    def _hydrate_memory_detail_rows(self, db_path: Path, rows: list[dict]) -> list[dict]:
        journal_ids = {
            self._journal_entry_id(row.get("journal_file"))
            for row in rows
            if row.get("type") == "journal" and row.get("journal_file")
        }
        journal_ids.discard(None)
        if not journal_ids:
            return rows

        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                if not self._table_columns(conn, "journal_entries"):
                    return rows
                placeholders = ", ".join("?" for _ in journal_ids)
                entries = {
                    row["id"]: dict(row)
                    for row in conn.execute(
                        f"SELECT * FROM journal_entries WHERE id IN ({placeholders})",
                        tuple(journal_ids),
                    ).fetchall()
                }
        except (sqlite3.Error, OSError):
            return rows

        hydrated = []
        for row in rows:
            copy = dict(row)
            entry_id = self._journal_entry_id(copy.get("journal_file"))
            entry = entries.get(entry_id)
            if entry and entry.get("content"):
                copy["text"] = self._format_journal_memory_detail(entry)
                copy["detail_source"] = "journal_entry"
            hydrated.append(copy)
        return hydrated

    def _journal_entry_id(self, journal_file: str | None) -> str | None:
        if not journal_file:
            return None
        return Path(str(journal_file)).stem or None

    def _format_journal_memory_detail(self, entry: dict) -> str:
        parts = []
        title = entry.get("title")
        if title:
            parts.append(str(title).strip())
        content = entry.get("content")
        if content:
            parts.append(str(content).strip())
        why = entry.get("why_it_mattered")
        if why:
            parts.append(f"Why it mattered: {str(why).strip()}")
        return "\n\n".join(part for part in parts if part)

    def _format_journal_entry(self, row: dict, detail: bool) -> dict:
        content = row.get("content") or ""
        why = row.get("why_it_mattered") or ""
        date = row.get("date") or ""
        age_hours = self._hours_since(date)
        resolved = row.get("resolved")
        pinned = row.get("pinned")
        review = row.get("summary_needs_review")
        status = "resolved" if resolved else "active"
        if resolved is None and (row.get("entry_type") or "") not in ("open_thread", "follow_up"):
            status = "reference"
        return {
            "id": row.get("id") or "",
            "author": row.get("author") or "",
            "title": row.get("title") or self._journal_fallback_title(row),
            "entry_type": row.get("entry_type") or "entry",
            "content": content if detail else "",
            "preview": content[:260] + ("..." if len(content) > 260 else ""),
            "why_it_mattered": why if detail else "",
            "why_preview": why[:180] + ("..." if len(why) > 180 else ""),
            "search_summary": row.get("search_summary") if detail else "",
            "summary_needs_review": bool(review) if review is not None else False,
            "tags": self._decode_tags(row.get("tags")),
            "importance": row.get("importance"),
            "pinned": bool(pinned) if pinned is not None else False,
            "resolved": bool(resolved) if resolved is not None else None,
            "status": status,
            "date": date,
            "date_display": str(date)[:10] if date else "unknown",
            "age_label": self._age_label(age_hours),
        }

    def _journal_fallback_title(self, row: dict) -> str:
        entry_type = str(row.get("entry_type") or "entry").replace("_", " ").title()
        content = str(row.get("content") or "").strip()
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        return first_line[:80] if first_line else entry_type

    def _empty_core_anchor(self, anchor_id: str) -> dict:
        titles = {
            "_self": "Who I Am",
            "_user": "About My Human",
            "_relationship": "Our Relationship",
        }
        return {
            "id": anchor_id,
            "title": titles.get(anchor_id, anchor_id),
            "sections": [],
            "empty": True,
            "created_at": "",
            "created_at_display": "unknown",
            "last_updated": "",
            "last_updated_display": "unknown",
            "age_label": "unknown age",
        }

    def _format_core_anchor(self, row: dict) -> dict:
        raw_sections = row.get("sections") or "{}"
        try:
            sections = json.loads(raw_sections) if isinstance(raw_sections, str) else raw_sections
        except (TypeError, ValueError):
            sections = {}
        if not isinstance(sections, dict):
            sections = {}
        section_items = [
            {
                "key": str(key),
                "label": self._core_section_label(str(key)),
                "value": str(value or ""),
            }
            for key, value in sections.items()
        ]
        updated = row.get("last_updated") or row.get("created_at") or ""
        age_hours = self._hours_since(updated)
        return {
            "id": row.get("id") or "",
            "title": row.get("title") or self._empty_core_anchor(row.get("id") or "")["title"],
            "sections": section_items,
            "empty": not any(item["value"].strip() for item in section_items),
            "created_at": row.get("created_at") or "",
            "created_at_display": self._format_timestamp(row.get("created_at") or ""),
            "last_updated": row.get("last_updated") or "",
            "last_updated_display": self._format_timestamp(row.get("last_updated") or ""),
            "age_label": self._age_label(age_hours),
        }

    def _core_section_label(self, key: str) -> str:
        labels = {
            "what_theyre_like": "What They're Like",
            "how_they_communicate": "How They Communicate",
            "their_preferences": "Their Preferences",
            "who_they_are": "Who They Are",
            "what_im_like": "What I'm Like",
            "what_im_working_on": "What I'm Working On",
            "who_i_am": "Who I Am",
            "my_preferences": "My Preferences",
            "how_i_present_myself": "How I Present Myself",
            "how_we_relate": "How We Relate",
            "our_dynamic": "Our Dynamic",
            "shared_context": "Shared Context",
            "boundaries_or_norms": "Boundaries Or Norms",
            "extra_notes": "Extra Notes",
        }
        return labels.get(key, key.replace("_", " ").title())

    def _format_memory_item(self, row: dict, history: dict[int, dict]) -> dict:
        tags = self._decode_tags(row.get("tags"))
        text = row.get("text") or ""
        date = row.get("date") or ""
        age_hours = self._hours_since(date)
        hist = history.get(row.get("id"), {})
        return {
            "id": row.get("id"),
            "text": text,
            "preview": text[:240] + ("..." if len(text) > 240 else ""),
            "tags": tags,
            "type": row.get("type") or "fact",
            "importance": row.get("importance"),
            "date": date,
            "date_display": str(date)[:10] if date else "unknown",
            "age_label": self._age_label(age_hours),
            "status": row.get("status") or "legacy",
            "confidence": row.get("confidence") or "",
            "source": row.get("source") or "",
            "journal_file": row.get("journal_file") or "",
            "detail_source": row.get("detail_source") or "",
            "retrieval_count": row.get("retrieval_count"),
            "last_accessed": row.get("last_accessed") or "",
            "last_confirmed": row.get("last_confirmed") or "",
            "time_sensitive": bool(row.get("time_sensitive")) if row.get("time_sensitive") is not None else False,
            "valid_until": row.get("valid_until") or "",
            "supersedes": hist.get("supersedes"),
            "replaced_by": hist.get("replaced_by"),
            "version_count": hist.get("version_count", 1),
            "has_history": hist.get("version_count", 1) > 1,
        }

    def _read_core_anchor_row(self, db_path: Path, anchor_id: str) -> dict | None:
        if not db_path.exists():
            return None
        with closing(_connect_sqlite(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            columns = self._table_columns(conn, "identity")
            if not columns:
                return None
            select_sql = self._core_anchor_select_sql(columns)
            row = conn.execute(
                f"SELECT {select_sql} FROM identity WHERE id = ?",
                (anchor_id,),
            ).fetchone()
            return dict(row) if row else None

    def _read_lantern_row(self, db_path: Path, persona: str) -> dict | None:
        if not db_path.exists():
            return None
        try:
            with closing(_connect_sqlite(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM resident_lanterns WHERE resident_id = ?",
                    (persona,),
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error:
            return None

    def _hours_since(self, value: str) -> float | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600)
        except (ValueError, TypeError):
            return None

    def _age_label(self, hours: float | None) -> str:
        if hours is None:
            return "unknown age"
        if hours < 1:
            minutes = max(0, int(hours * 60))
            return f"{minutes} min old"
        if hours < 48:
            return f"{int(hours)} hours old"
        return f"{int(hours // 24)} days old"

    def _format_timestamp(self, value: str) -> str:
        if not value:
            return "Unknown"
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local = dt.astimezone()
            return local.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return str(value)

    def _base_persona_summary(self) -> dict:
        config = _load_yaml(self.base_config_path)
        return {
            "name": "__base__",
            "display_name": "Base Config",
            "provider": config.get("provider", {}).get("type", "local"),
            "model": config.get("provider", {}).get("model") or config.get("model", {}).get("model_file", ""),
            "is_base": True,
            "running": False,
        }

    def _persona_summary(self, name: str) -> dict:
        loaded = self.load_persona(name)
        summary = loaded["summary"]
        return {
            "name": name,
            "display_name": summary["display_name"],
            "provider": summary["provider_type"],
            "model": summary["model_display"],
            "is_base": False,
            "running": loaded["status"].get("running", False),
        }

    def _load_identity(self, persona: str | None) -> dict:
        if persona:
            base = self.personas_dir / persona
        else:
            base = self.root
        for filename in ("persona.yaml", "persona.yml", "persona.json"):
            path = base / filename
            if path.exists():
                if path.suffix == ".json":
                    return json.loads(path.read_text(encoding="utf-8"))
                return _load_yaml(path)
        return {}

    def _identity_path(self, persona: str) -> str:
        base = self.root if persona == "__base__" else self.personas_dir / persona
        for filename in ("persona.yaml", "persona.yml", "persona.json"):
            path = base / filename
            if path.exists():
                return _safe_rel(path, self.root)
        return ""

    def _apply_persona_defaults(self, config: dict, name: str) -> None:
        persona_dir = self.personas_dir / name
        data_dir = persona_dir / "data"
        paths = config.setdefault("paths", {})
        paths.setdefault("database", str(data_dir / f"{name}.db"))
        paths.setdefault("telegram_chat_id", str(data_dir / "telegram_chat_id.txt"))

    def _build_summary(self, name: str, config: dict, identity: dict) -> dict:
        provider = config.get("provider", {})
        model_cfg = config.get("model", {})
        server_cfg = config.get("server", {})
        provider_type = provider.get("type", "local")
        provider_model = provider.get("model", "")
        identity_model = identity.get("model", "")
        model_file = model_cfg.get("model_file", "")
        model_display = identity_model or provider_model or Path(model_file).name or "not configured"
        provider_max_context = provider.get("max_context", "")
        local_max_context = model_cfg.get("max_context", "")
        return {
            "display_name": identity.get("name") or ("Base Config" if name == "__base__" else name.title()),
            "user_name": identity.get("user_name", ""),
            "model_display": model_display,
            "provider_model": provider_model,
            "provider_type": provider_type,
            "base_url": provider.get("base_url", ""),
            "max_context": provider_max_context,
            "provider_max_context": provider_max_context,
            "local_max_context": local_max_context,
            "model_file": model_cfg.get("model_file", ""),
            "mmproj_file": model_cfg.get("mmproj_file", ""),
            "server": {
                "llama_cpp_dir": server_cfg.get("llama_cpp_dir", ""),
                "models_dir": server_cfg.get("models_dir", ""),
                "host": server_cfg.get("host", "127.0.0.1"),
                "port": server_cfg.get("port", 8012),
                "gpu_layers": server_cfg.get("gpu_layers", -1),
                "flash_attention": server_cfg.get("flash_attention", True),
                "parallel": server_cfg.get("parallel", 1),
            },
            "max_response_tokens": model_cfg.get("max_response_tokens", ""),
            "temperature": model_cfg.get("temperature", config.get("model", {}).get("temperature", 0.7)),
            "frequency_penalty": model_cfg.get("frequency_penalty", ""),
            "presence_penalty": model_cfg.get("presence_penalty", ""),
            "top_p": model_cfg.get("top_p", ""),
            "reasoning": model_cfg.get("reasoning", False),
            "reasoning_effort": model_cfg.get("reasoning_effort", ""),
            "show_reasoning": model_cfg.get("show_reasoning", False),
            "max_tool_rounds": model_cfg.get("max_tool_rounds", ""),
            "context_budget": config.get("context_budget", {}),
            "heartbeat": config.get("heartbeat", {}),
            "tts": config.get("tts", {}),
        }

    def _list_skills(self, config: dict) -> list[dict]:
        skills_dir = self.root / "skills"
        configured = config.get("skills", {})
        names = set(configured.keys())
        if skills_dir.exists():
            names.update(
                p.stem for p in skills_dir.glob("*.py")
                if p.stem not in ("__init__", "base")
            )
        result = []
        for name in sorted(names):
            enabled = configured.get(name, {}).get("enabled", True)
            result.append({
                "name": name,
                "label": name.replace("_", " ").title(),
                "enabled": bool(enabled),
                "icon": SKILL_ICONS.get(name, name[:2].title()),
            })
        return result

    def _list_channels(self, config: dict) -> list[dict]:
        channels = config.get("channels", {})
        return [
            {
                "name": name,
                "label": name.replace("_", " ").title(),
                "enabled": bool(value.get("enabled", True)) if isinstance(value, dict) else bool(value),
            }
            for name, value in sorted(channels.items())
        ]

    def _read_status_file(self, persona: str) -> dict:
        if not persona or persona == "__base__":
            return {}
        path = self.personas_dir / persona / "data" / "status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        updated = status.get("updated_at")
        status["stale"] = False
        if updated:
            try:
                dt = datetime.fromisoformat(updated)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                status["stale"] = (
                    datetime.now(timezone.utc) - dt
                ).total_seconds() > STATUS_STALE_AFTER_SECONDS
            except ValueError:
                status["stale"] = True
        return status

    def _avatar_data_uri(self, persona: str) -> str:
        if not persona or persona == "__base__":
            return ""
        base = self.personas_dir / persona
        for ext, mime in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")):
            path = base / f"avatar{ext}"
            if path.exists():
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"data:{mime};base64,{data}"
        return ""
