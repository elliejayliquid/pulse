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


STATUS_STALE_AFTER_SECONDS = 120
LANTERN_STALE_HOURS = 24
LANTERN_EXPIRED_HOURS = 7 * 24


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

    def _read_lantern_row(self, db_path: Path, persona: str) -> dict | None:
        if not db_path.exists():
            return None
        try:
            with closing(sqlite3.connect(str(db_path))) as conn:
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
