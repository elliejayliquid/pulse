"""Backend bridge for the Pulse Engine desktop GUI.

Phase 1 is intentionally read-only for persona configs. The GUI can inspect
personas, merged config, identity, status files, and logs without mutating the
project.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gui.backup import BackupManager
from gui.config_editor import ConfigEditor


STATUS_STALE_AFTER_SECONDS = 120


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

    # Secrets/status/logs

    def get_key_status(self, persona: str) -> dict:
        config = _load_yaml(self.base_config_path)
        if persona and persona != "__base__":
            config = deep_merge(config, _load_yaml(self.personas_dir / persona / "config.yaml"))

        root_env = _parse_env(self.root / ".env")
        persona_env = _parse_env(self.personas_dir / persona / ".env") if persona and persona != "__base__" else {}
        env = {**root_env, **persona_env}

        api_key_env = config.get("provider", {}).get("api_key_env", "")
        telegram_key = "TELEGRAM_BOT_TOKEN"
        return {
            "api_key_env": api_key_env,
            "api_key_set": bool(api_key_env and env.get(api_key_env)),
            "telegram_key": telegram_key,
            "telegram_set": bool(env.get(telegram_key)),
        }

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
        if os.name == "nt":
            os.startfile(str(persona_dir))
        else:
            import subprocess as _sp
            _sp.Popen(["xdg-open", str(persona_dir)])
        return {"ok": True}

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
        provider_type = provider.get("type", "local")
        provider_model = provider.get("model", "")
        identity_model = identity.get("model", "")
        model_file = model_cfg.get("model_file", "")
        model_display = identity_model or provider_model or Path(model_file).name or "not configured"
        max_context = provider.get("max_context") or model_cfg.get("max_context") or ""
        return {
            "display_name": identity.get("name") or ("Base Config" if name == "__base__" else name.title()),
            "user_name": identity.get("user_name", ""),
            "model_display": model_display,
            "provider_model": provider_model,
            "provider_type": provider_type,
            "max_context": max_context,
            "max_response_tokens": model_cfg.get("max_response_tokens", ""),
            "temperature": model_cfg.get("temperature", config.get("model", {}).get("temperature", 0.7)),
            "frequency_penalty": model_cfg.get("frequency_penalty", ""),
            "presence_penalty": model_cfg.get("presence_penalty", ""),
            "top_p": model_cfg.get("top_p", ""),
            "reasoning": model_cfg.get("reasoning", False),
            "reasoning_effort": model_cfg.get("reasoning_effort", ""),
            "show_reasoning": model_cfg.get("show_reasoning", False),
            "max_tool_rounds": model_cfg.get("max_tool_rounds", ""),
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
                "enabled": bool(value.get("enabled", False)) if isinstance(value, dict) else bool(value),
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
