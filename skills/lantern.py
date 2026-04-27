"""
Lantern skill — a small current-state signal for residents.

The lantern is not a journal, memory, or mood tracker in the quantified sense.
It is a compact "where am I standing right now?" marker that can optionally
be injected into context later.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from skills.base import BaseSkill


STALE_HOURS = 24


class LanternSkill(BaseSkill):
    name = "lantern"

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
        with self._connect() as conn:
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
        with self._connect() as conn:
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

        if not parts and not stale:
            return ""

        # Build the lantern line
        lines = []
        if parts:
            prefix = "Last lantern" if stale else "Lantern"
            lines.append(
                f"[{prefix}] {row['resident_id']}: "
                + "; ".join(parts)
                + f". Updated: {row['updated_at']}."
            )

        # Gentle nudge when stale
        if stale:
            hours = self._hours_since(row["updated_at"])
            if hours is not None and hours < 48:
                lines.append(
                    "[Lantern nudge] It's been a while since you updated your lantern. "
                    "If your inner weather has shifted, consider a quick set_lantern."
                )
            elif hours is not None:
                days = int(hours // 24)
                lines.append(
                    f"[Lantern nudge] Your lantern hasn't been updated in {days} days. "
                    "No pressure — but if you'd like to re-light it, set_lantern is there."
                )
            else:
                lines.append(
                    "[Lantern nudge] Your lantern may be out of date. "
                    "Consider updating it with set_lantern when you're ready."
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

        with self._connect() as conn:
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
        prefix = "Last lantern" if self._is_stale(row["updated_at"]) else "Current lantern"
        return self._format_lantern(row, prefix=prefix)

    def _dim_lantern(self, note: str | None = None) -> str:
        updated_at = self._now()
        rest_note = (note or "Lantern dimmed; no active state set.").strip()

        with self._connect() as conn:
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
