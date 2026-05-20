"""Runtime status and shutdown-sentinel helpers for Pulse.

The GUI polls status.json while Pulse is running. Writes must be atomic because
the GUI may read while Pulse is updating the file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RuntimeStatus:
    """Manages per-persona GUI status files and shutdown requests."""

    def __init__(self, config: dict):
        paths = config.get("paths", {})
        status_path = paths.get("status")
        shutdown_path = paths.get("shutdown_requested")

        persona_dir = config.get("_persona_dir")
        if not status_path and persona_dir:
            status_path = str(Path(persona_dir) / "data" / "status.json")
        if not shutdown_path and persona_dir:
            shutdown_path = str(Path(persona_dir) / "data" / "shutdown_requested")

        self.status_path = Path(status_path) if status_path else None
        self.shutdown_path = Path(shutdown_path) if shutdown_path else None
        self.persona = config.get("_persona_name") or "base"
        self.provider = config.get("provider", {}).get("type", "local")
        self.model = (
            config.get("provider", {}).get("model")
            or config.get("model", {}).get("model_file")
            or ""
        )

    def clear_shutdown_request(self) -> None:
        """Remove a stale shutdown sentinel, if one exists."""
        if self.shutdown_path and self.shutdown_path.exists():
            self.shutdown_path.unlink()

    def request_shutdown(self) -> None:
        """Create the shutdown sentinel."""
        if not self.shutdown_path:
            return
        self.shutdown_path.parent.mkdir(parents=True, exist_ok=True)
        self.shutdown_path.write_text(self._now(), encoding="utf-8")

    def shutdown_requested(self, consume: bool = True) -> bool:
        """Return whether a shutdown was requested, optionally deleting it."""
        if not self.shutdown_path or not self.shutdown_path.exists():
            return False
        if consume:
            try:
                self.shutdown_path.unlink()
            except FileNotFoundError:
                pass
        return True

    def write(self, phase: str, running: bool, **extra: Any) -> None:
        """Atomically write status.json."""
        if not self.status_path:
            return

        payload = {
            "persona": self.persona,
            "pid": os.getpid(),
            "phase": phase,
            "running": running,
            "updated_at": self._now(),
            "provider": self.provider,
            "model": self.model,
        }
        payload.update({k: v for k, v in extra.items() if v is not None})

        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.status_path)

    def write_stopped(self, **extra: Any) -> None:
        """Write the final clean stopped status."""
        self.write("stopped", False, **extra)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
