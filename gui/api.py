"""Backend bridge for the Pulse Engine desktop GUI.

Phase 1 is intentionally read-only for persona configs. The GUI can inspect
personas, merged config, identity, status files, and logs without mutating the
project.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SKILL_ICONS = {
    "dev": "</>",
    "garden": "Gr",
    "journal": "Jr",
    "lantern": "Ln",
    "lor": "Lo",
    "memory": "Mm",
    "paint": "Pt",
    "schedule": "Sc",
    "sticker": "St",
    "tasks": "Tk",
    "tts": "Vo",
    "web_search": "Ws",
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
        self.processes: dict[str, ProcessInfo] = {}

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

    def save_persona(self, name: str, config: dict, identity: dict) -> dict:
        return {
            "ok": False,
            "error": "Config saving is not implemented in Phase 1. This GUI is read-only for persona files.",
        }

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
        if persona is None:
            return {name: self.get_status(name) for name in self._persona_names()}

        status = self._read_status_file(persona)
        proc = self.processes.get(persona)
        if proc:
            status.update({
                "running": True,
                "persona": persona,
                "pid": proc.pid,
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
        return {
            "ok": False,
            "error": "Graceful stop sentinel support is planned for the process-management phase.",
        }

    def start_pulse(self, persona: str) -> dict:
        return {
            "ok": False,
            "error": "Starting Pulse from the GUI is planned for the process-management phase.",
        }

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
            "reasoning": model_cfg.get("reasoning", False),
            "reasoning_effort": model_cfg.get("reasoning_effort", ""),
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
                status["stale"] = (datetime.now(timezone.utc) - dt).total_seconds() > 30
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
