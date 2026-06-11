"""
Lantern skill — a small current-state signal for residents.

The lantern is not a journal, memory, or mood tracker in the quantified sense.
It is a compact "where am I standing right now?" marker that can optionally
be injected into context later.
"""

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from skills.base import BaseSkill


STALE_HOURS = 24
EXPIRED_HOURS = 7 * 24


class LanternSkill(BaseSkill):
    name = "lantern"
    description = "Track and update persistent emotional or contextual state"
    always_loaded = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.resident_id = config.get("_persona_name") or "default"
        self.db_path = self._resolve_db_path(config)
        self._ensure_table()

    def _resolve_db_path(self, config: dict) -> str:
        paths = config.get("paths", {})
        db_path = paths.get("database")
        if db_path:
            return db_path

        persona_dir = config.get("_persona_dir")
        if persona_dir:
            return str(Path(persona_dir) / "data" / "lantern.db")

        return os.path.join("data", "lantern.db")

    def _connect(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        with closing(self._connect()) as conn:
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
            conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_lantern(self, resident_id: str | None = None):
        rid = resident_id or self.resident_id
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM resident_lanterns WHERE resident_id = ?",
                (rid,),
            ).fetchone()
        return dict(row) if row else None

    def _is_stale(self, updated_at: str) -> bool:
        try:
            dt = datetime.fromisoformat(updated_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - dt
            return age.total_seconds() > STALE_HOURS * 3600
        except Exception:
            return True

    def _is_expired(self, updated_at: str) -> bool:
        try:
            dt = datetime.fromisoformat(updated_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - dt
            return age.total_seconds() > EXPIRED_HOURS * 3600
        except Exception:
            return True

    def _hours_since(self, updated_at: str) -> float | None:
        """Return how many hours have passed since updated_at, or None on error."""
        try:
            dt = datetime.fromisoformat(updated_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - dt
            return age.total_seconds() / 3600
        except Exception:
            return None

    def _age_label(self, updated_at: str) -> str:
        hours = self._hours_since(updated_at)
        if hours is None:
            return "unknown age"
        if hours < 1:
            return f"{int(hours * 60)} minutes old"
        if hours < 48:
            return f"{int(hours)} hours old"
        return f"{int(hours // 24)} days old"

    def get_context(self) -> str:
        row = self._get_lantern()

        # No lantern has ever been set — gentle first-time nudge
        if not row:
            return (
                f"[Lantern] No lantern set for {self.resident_id} yet. "
                "When you're ready, use set_lantern to mark where you're standing — "
                "mode, mood, focus, whatever feels true right now."
            )

        parts = []
        for label, key in [
            ("mode", "mode"),
            ("mood", "mood"),
            ("focus", "focus"),
            ("open thread", "open_thread"),
            ("note", "note"),
        ]:
            value = row.get(key)
            if value:
                parts.append(f"{label}: {value}")

        stale = self._is_stale(row["updated_at"])
        expired = self._is_expired(row["updated_at"])

        if not parts and not stale:
            return ""

        if expired:
            return (
                f"[Lantern expired] {row['resident_id']}'s lantern was last updated "
                f"{row['updated_at']} ({self._age_label(row['updated_at'])}). "
                "It is too old to use as current-state context, so the old mode/mood/focus "
                "has been withheld. Do not assume it describes your human or the present moment. "
                "Re-orient from current time, recent conversation, journal, and tools; then "
                "use set_lantern with a fresh state or dim_lantern if there is no active thread."
            )

        # Calculate age for better LLM time perception
        hours = self._hours_since(row["updated_at"])
        age_str = ""
        if hours is not None:
            if hours < 1:
                age_str = f" ({int(hours * 60)} mins ago)"
            elif hours < 48:
                age_str = f" ({int(hours)} hours ago)"
            else:
                age_str = f" ({int(hours / 24)} days ago)"

        # Build the lantern line
        lines = []
        if parts:
            prefix = "Last lantern" if stale else "Lantern"
            if stale:
                lines.append(
                    "[Lantern stale] This lantern is older than 24 hours. Treat it as "
                    "historical orientation only, not proof of what your human is doing now. "
                    "Before relying on it, refresh it with set_lantern or dim it."
                )
            lines.append(
                f"[{prefix}] {row['resident_id']}: "
                + "; ".join(parts)
                + f". Updated: {row['updated_at']}{age_str}."
            )

        # Stern nudge when stale
        if stale:
            if hours is not None and hours < 48:
                lines.append(
                    "[Lantern nudge] Refresh this soon if the current state has shifted."
                )
            elif hours is not None:
                days = int(hours // 24)
                lines.append(
                    f"[Lantern nudge] Your lantern hasn't been updated in {days} days. "
                    "Do not treat it as current; re-light it with set_lantern or dim it."
                )
            else:
                lines.append(
                    "[Lantern nudge] Your lantern may be out of date. "
                    "Do not treat it as current; update or dim it when you can."
                )

        return "\n".join(lines)

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_lantern",
                    "description": (
                        "Set or update your compact current-state lantern: mode, mood, focus, "
                        "note, and/or open thread. Use when your internal posture meaningfully changes, "
                        "not every turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "description": "Current interaction mode, e.g. quiet company, technical focus, playful, reflective, caretaking.",
                            },
                            "mood": {
                                "type": "string",
                                "description": "Current mood/texture in plain language, not a numeric score.",
                            },
                            "focus": {
                                "type": "string",
                                "description": "What you are currently oriented toward.",
                            },
                            "note": {
                                "type": "string",
                                "description": "Small contextual note for continuity.",
                            },
                            "open_thread": {
                                "type": "string",
                                "description": "One thread you may want to return to soon.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_lantern",
                    "description": "Read the current lantern for yourself or another resident.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "resident_id": {
                                "type": "string",
                                "description": "Optional resident/persona name. Defaults to the active resident.",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dim_lantern",
                    "description": "Dim/clear your current lantern when a state is no longer active.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": "Optional resting note to leave behind.",
                            }
                        },
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "set_lantern":
            return self._set_lantern(arguments)
        if tool_name == "read_lantern":
            return self._read_lantern(arguments.get("resident_id"))
        if tool_name == "dim_lantern":
            return self._dim_lantern(arguments.get("note"))
        return f"Unknown lantern tool: {tool_name}"

    def _set_lantern(self, args: dict) -> str:
        allowed = ["mode", "mood", "focus", "note", "open_thread"]
        values = {k: (args.get(k) or "").strip() for k in allowed}

        if not any(values.values()):
            return "No lantern fields provided; nothing changed."

        existing = self._get_lantern() or {}
        merged = {k: values[k] if values[k] else (existing.get(k) or "") for k in allowed}
        updated_at = self._now()

        with closing(self._connect()) as conn:
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
                    self.resident_id,
                    merged["mode"],
                    merged["mood"],
                    merged["focus"],
                    merged["note"],
                    merged["open_thread"],
                    updated_at,
                ),
            )
            conn.commit()

        return self._format_lantern(self._get_lantern(), prefix="Lantern updated")

    def _read_lantern(self, resident_id: str | None = None) -> str:
        row = self._get_lantern(resident_id)
        if not row:
            rid = resident_id or self.resident_id
            return f"No lantern found for {rid}."
        expired = self._is_expired(row["updated_at"])
        stale = self._is_stale(row["updated_at"])
        if expired:
            prefix = "Expired lantern"
        elif stale:
            prefix = "Stale lantern"
        else:
            prefix = "Current lantern"

        result = self._format_lantern(row, prefix=prefix)
        if expired:
            result += (
                "\n\nWarning: this lantern is more than 7 days old. Treat it as "
                "historical, not current; refresh it with set_lantern or dim it."
            )
        elif stale:
            result += (
                "\n\nWarning: this lantern is more than 24 hours old. Do not assume "
                "it still describes the present moment."
            )
        return result

    def _dim_lantern(self, note: str | None = None) -> str:
        updated_at = self._now()
        rest_note = (note or "Lantern dimmed; no active state set.").strip()

        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO resident_lanterns
                    (resident_id, mode, mood, focus, note, open_thread, updated_at)
                VALUES (?, '', '', '', ?, '', ?)
                ON CONFLICT(resident_id) DO UPDATE SET
                    mode = '',
                    mood = '',
                    focus = '',
                    note = excluded.note,
                    open_thread = '',
                    updated_at = excluded.updated_at
                """,
                (self.resident_id, rest_note, updated_at),
            )
            conn.commit()

        return f"Lantern dimmed for {self.resident_id}. Note: {rest_note}"

    def _format_lantern(self, row: dict, prefix: str = "Lantern") -> str:
        lines = [f"{prefix} for {row['resident_id']}:"]
        for label, key in [
            ("Mode", "mode"),
            ("Mood", "mood"),
            ("Focus", "focus"),
            ("Open thread", "open_thread"),
            ("Note", "note"),
        ]:
            value = row.get(key)
            if value:
                lines.append(f"- {label}: {value}")
        lines.append(f"- Updated: {row['updated_at']}")
        return "\n".join(lines)
