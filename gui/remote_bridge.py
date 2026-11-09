"""GUI side of remote personas (designs/remote_host.md §5.3).

A persona whose PC folder has a `remote.json` marker lives on the home server.
For those, the GUI shows live status and logs from the box, sends Start/Stop
there, and syncs persona.yaml/config.yaml both ways through scripts/remote.py.

The GUI polls status every 2s; SSH is far too slow for that, so status and logs
come from a cache that a background thread refreshes (one `pulse-ctl list` for
everyone). Without gui_data/remote.json nothing here does anything.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

STATUS_TTL = 15        # seconds between box listings
LOG_TTL = 10           # seconds between log fetches (only for the persona on screen)
PENDING_TIMEOUT = 300  # a start/stop that never shows up stops being "pending" after this


def _remote():
    import scripts.remote as remote
    return remote


def _error(e: Exception) -> dict:
    return {"ok": False, "error": str(e)}


class RemoteBridge:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.personas_dir = self.root / "personas"
        self.settings_path = self.root / "gui_data" / "remote.json"
        self._lock = threading.Lock()
        self._listing: dict[str, dict] | None = None
        self._listing_at = 0.0
        self._listing_error = ""
        self._listing_busy = False
        self._logs: dict[str, tuple[str, float]] = {}
        self._logs_busy: set[str] = set()
        self._pending: dict[str, tuple[str, float]] = {}  # persona -> (starting|stopping, when)
        self._sync_rows: dict[str, list[dict]] = {}

    # --- who is remote -----------------------------------------------------

    def enabled(self) -> bool:
        return self.settings_path.is_file()

    def is_remote(self, persona: str | None) -> bool:
        return bool(persona) and persona != "__base__" and self.enabled() and (
            self.personas_dir / persona / "remote.json"
        ).is_file()

    def _settings(self) -> dict:
        return _remote().load_settings()

    def host(self) -> str:
        try:
            return self._settings().get("host", "")
        except Exception:
            return ""

    # --- status ------------------------------------------------------------

    def _refresh_listing(self) -> None:
        try:
            result = _remote().ctl(self._settings(), "list", timeout=20)
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "pulse-ctl list failed")
            listing = {p["name"]: p for p in result.get("personas", [])}
            with self._lock:
                self._listing, self._listing_error = listing, ""
        except Exception as e:  # unreachable box, ssh missing, bad settings...
            with self._lock:
                self._listing_error = str(e) or e.__class__.__name__
        finally:
            with self._lock:
                self._listing_at = time.time()
                self._listing_busy = False

    def _ensure_fresh(self, force: bool = False) -> None:
        with self._lock:
            if self._listing_busy:
                return
            if not force and time.time() - self._listing_at < STATUS_TTL:
                return
            self._listing_busy = True
        threading.Thread(target=self._refresh_listing, daemon=True).start()

    def status(self, persona: str) -> dict:
        """GUI-shaped status for a remote persona, from the cached box listing."""
        self._ensure_fresh()
        with self._lock:
            listing, error, checked = self._listing, self._listing_error, self._listing_at
            pending = self._pending.get(persona)
        base = {"persona": persona, "remote": True, "host": self.host(), "stale": False,
                "source": "remote", "remote_error": error,
                "checked_at": checked or None,
                # two small file reads, no SSH: keeps the Sync button's dot current
                "unsynced": self.unsynced(persona).get("files", [])}
        if listing is None:
            return {**base, "running": False, "unknown": True,
                    "phase": "box unreachable" if error else "checking box..."}

        entry = listing.get(persona) or {}
        status = dict(entry.get("status") or {})
        active = (entry.get("systemd") or {}).get("active", "inactive")
        if active == "active":
            phase, running = status.get("phase") or "running", True
            if phase in ("stopped", "stopping") and not pending:
                phase = "starting"
        elif active == "activating":
            phase, running = "starting", True
        elif active == "deactivating":
            phase, running = "stopping", True
        else:
            phase, running = "stopped", False

        if pending:
            want, since = pending
            reached = (want == "starting" and running and phase == "running") or (
                want == "stopping" and not running)
            if reached or time.time() - since > PENDING_TIMEOUT:
                with self._lock:
                    self._pending.pop(persona, None)
            else:
                phase, running = want, True

        status.update(base)
        status.update({"running": running, "phase": phase, "systemd": active})
        status.pop("pid", None)  # a box pid means nothing on the PC
        return status

    def control(self, persona: str, verb: str) -> dict:
        try:
            result = _remote().ctl(self._settings(), verb, persona)
        except Exception as e:
            return _error(e)
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error") or f"{verb} failed on the box"}
        with self._lock:
            self._pending[persona] = ("stopping" if verb == "stop" else "starting", time.time())
        self._ensure_fresh(force=True)
        return {"ok": True, "status": self.status(persona)}

    def running_on_box(self, persona: str) -> bool:
        with self._lock:
            entry = (self._listing or {}).get(persona) or {}
        return (entry.get("systemd") or {}).get("active") == "active"

    # --- log -----------------------------------------------------------------

    def _refresh_log(self, persona: str, lines: int) -> None:
        try:
            result = _remote().ctl(self._settings(), "log", persona, str(lines), timeout=20)
            text = result.get("log") if result.get("ok") else f"Couldn't read the box log: {result.get('error')}"
        except Exception as e:
            text = f"Couldn't read the box log: {e}"
        with self._lock:
            self._logs[persona] = (text or "", time.time())
            self._logs_busy.discard(persona)

    def log(self, persona: str, lines: int = 100) -> str:
        with self._lock:
            cached = self._logs.get(persona)
            due = cached is None or time.time() - cached[1] >= LOG_TTL
            if due and persona not in self._logs_busy:
                self._logs_busy.add(persona)
                threading.Thread(target=self._refresh_log, args=(persona, lines), daemon=True).start()
        return cached[0] if cached else f"Loading {persona}'s log from the box..."

    # --- persona.yaml / config.yaml sync --------------------------------------

    def unsynced(self, persona: str) -> dict:
        """Cheap (no SSH): files saved on the PC since the last sync."""
        try:
            return {"ok": True, "files": _remote().local_unsynced(self._settings(), persona)}
        except Exception as e:
            return _error(e)

    def sync_check(self, persona: str) -> dict:
        try:
            remote = _remote()
            rows = remote.sync_state(self._settings(), persona)
        except Exception as e:
            return _error(e)
        with self._lock:
            self._sync_rows[persona] = rows
        return {
            "ok": True,
            "persona": persona,
            "host": self.host(),
            "files": [{"file": r["file"], "state": r["state"], "diff": r["diff"]} for r in rows],
        }

    def sync_apply(self, persona: str, choices: dict[str, str]) -> dict:
        """choices: file -> "push" | "pull" | "skip", as picked in the Sync window.

        Re-checks first: if either side changed since the window showed its diff,
        nothing is written and the window reloads.
        """
        with self._lock:
            seen = {r["file"]: r["diff"] for r in self._sync_rows.get(persona, [])}
        push = [f for f, c in (choices or {}).items() if c == "push"]
        pull = [f for f, c in (choices or {}).items() if c == "pull"]
        if not push and not pull:
            return {"ok": True, "pushed": [], "pulled": [], "nothing": True}
        try:
            remote = _remote()
            settings = self._settings()
            rows = remote.sync_state(settings, persona)
            now = {r["file"]: r["diff"] for r in rows}
            if any(now.get(f) != seen.get(f) for f in push + pull):
                with self._lock:
                    self._sync_rows[persona] = rows
                return {"ok": False, "changed": True,
                        "error": "Something changed since this window loaded. Look again, then apply."}
            pushed = remote.push_files(settings, persona, rows, push)["pushed"] if push else []
            pulled = remote.pull_files(settings, persona, rows, pull) if pull else {"pulled": []}
        except Exception as e:
            return _error(e)
        self._ensure_fresh(force=True)
        return {
            "ok": True,
            "pushed": [p["file"] for p in pushed],
            "pulled": [p["file"] for p in pulled["pulled"]],
            "pc_backup": pulled.get("backup"),
            "running": self.running_on_box(persona),
        }

    # --- snapshots of the DBs (memories, journal, sessions) --------------------

    def backup_state(self, persona: str) -> dict:
        try:
            return _remote().backup_state(self._settings(), persona)
        except Exception as e:
            return _error(e)

    def pull_backup(self, persona: str) -> dict:
        try:
            return _remote().pull(self._settings(), persona)
        except Exception as e:
            return _error(e)
