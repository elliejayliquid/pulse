"""
SQLite database layer for Pulse — one database per persona.

Replaces scattered JSON files with a single .db file per persona.
Uses WAL mode for safe concurrent reads from async coroutines.
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class PulseDatabase:
    """SQLite database for a single Pulse persona."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()
        logger.info(f"Database opened: {self.db_path}")

    def _init_tables(self):
        """Create all tables if they don't exist."""
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
        logger.info(f"Database closed: {self.db_path}")

    # ── Messages ─────────────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str,
                     timestamp: Optional[str] = None,
                     is_summary: bool = False) -> int:
        """Insert a message and return its row id."""
        if timestamp:
            cur = self.conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, is_summary) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, timestamp, int(is_summary))
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO messages (session_id, role, content, is_summary) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, int(is_summary))
            )
        self.conn.commit()
        return cur.lastrowid

    def get_messages(self, session_id: str, limit: int = 0,
                     offset: int = 0) -> list[dict]:
        """Get messages for a session, ordered by timestamp."""
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id"
        params: list = [session_id]
        if limit > 0:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def clear_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session. Returns count of deleted rows."""
        cur = self.conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        self.conn.commit()
        return cur.rowcount

    def search_messages(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across all messages."""
        sql = (
            "SELECT m.*, s.title AS session_title "
            "FROM messages m "
            "LEFT JOIN sessions s ON m.session_id = s.id "
            "WHERE m.content LIKE ? "
            "  AND m.role IN ('user', 'assistant') "
            "  AND m.is_summary = 0 "
            "ORDER BY m.timestamp DESC "
            "LIMIT ?"
        )
        return [dict(r) for r in self.conn.execute(
            sql, (f"%{query}%", limit)
        ).fetchall()]

    def get_message_context(self, message_id: int, window: int = 5) -> list[dict]:
        """Get messages surrounding a specific message in the same session.
        
        Returns a list of messages centered around the target ID, ordered by time.
        """
        # 1. Get the session_id for the target message
        row = self.conn.execute(
            "SELECT session_id FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if not row:
            return []
        session_id = row[0]

        # 2. UNION query to get N messages before (including target) and N after.
        # We use nested subqueries because UNION results must be sorted as a whole,
        # but the individual legs need LIMIT.
        sql = (
            "SELECT * FROM ("
            "  SELECT m.*, s.title AS session_title FROM messages m "
            "  LEFT JOIN sessions s ON m.session_id = s.id "
            "  WHERE m.session_id = ? AND m.id <= ? "
            "  ORDER BY m.id DESC LIMIT ?"
            ") "
            "UNION ALL "
            "SELECT * FROM ("
            "  SELECT m.*, s.title AS session_title FROM messages m "
            "  LEFT JOIN sessions s ON m.session_id = s.id "
            "  WHERE m.session_id = ? AND m.id > ? "
            "  ORDER BY m.id ASC LIMIT ?"
            ") "
            "ORDER BY id ASC"
        )
        params = (session_id, message_id, window + 1, session_id, message_id, window)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ── Sessions ─────────────────────────────────────────────

    def create_session(self, session_id: str, title: str = "Chat") -> None:
        """Create a new conversation session."""
        self.conn.execute(
            "INSERT INTO sessions (id, title) VALUES (?, ?)",
            (session_id, title)
        )
        self.conn.commit()

    def close_session(self, session_id: str, summary: Optional[str] = None) -> None:
        """Close a session, optionally storing a summary."""
        self.conn.execute(
            "UPDATE sessions SET summary = ?, closed_at = datetime('now') WHERE id = ?",
            (summary, session_id)
        )
        self.conn.commit()

    def get_active_session(self) -> Optional[dict]:
        """Return the most recent unclosed session, or None."""
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE closed_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_sessions(self, limit: int = 20) -> list[dict]:
        """Get recent sessions, newest first."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()]

    # ── Memories ─────────────────────────────────────────────

    def save_memory(self, text: str, tags: list[str] | None = None,
                    type: str = "fact", importance: int = 5,
                    embedding: Optional[bytes] = None,
                    supersedes: Optional[int] = None,
                    journal_file: Optional[str] = None,
                    date: Optional[str] = None) -> int:
        """Insert a memory and return its row id."""
        tags_json = json.dumps(tags or [])
        if date:
            cur = self.conn.execute(
                "INSERT INTO memories (text, tags, type, importance, embedding, "
                "supersedes, journal_file, date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (text, tags_json, type, importance, embedding,
                 supersedes, journal_file, date)
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO memories (text, tags, type, importance, embedding, "
                "supersedes, journal_file) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (text, tags_json, type, importance, embedding,
                 supersedes, journal_file)
            )
        self.conn.commit()
        return cur.lastrowid

    def get_memory(self, memory_id: int) -> Optional[dict]:
        """Get a single memory by id."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            return d
        return None

    def get_all_memories(self, include_superseded: bool = False) -> list[dict]:
        """Get all memories, optionally including superseded ones."""
        if include_superseded:
            sql = "SELECT * FROM memories ORDER BY importance DESC, date DESC"
            rows = self.conn.execute(sql).fetchall()
        else:
            sql = ("SELECT * FROM memories WHERE id NOT IN "
                   "(SELECT supersedes FROM memories WHERE supersedes IS NOT NULL) "
                   "ORDER BY importance DESC, date DESC")
            rows = self.conn.execute(sql).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            result.append(d)
        return result

    def update_retrieval(self, memory_id: int) -> None:
        """Bump retrieval count and last_accessed timestamp."""
        self.conn.execute(
            "UPDATE memories SET retrieval_count = retrieval_count + 1, "
            "last_accessed = datetime('now') WHERE id = ?",
            (memory_id,)
        )
        self.conn.commit()

    def search_memories_by_text(self, query: str,
                                limit: int = 20) -> list[dict]:
        """Keyword search across memory text."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE text LIKE ? "
            "ORDER BY importance DESC, date DESC LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            result.append(d)
        return result

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by id. Returns True if a row was deleted."""
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ── Schedules ────────────────────────────────────────────

    def save_schedule(self, schedule: dict) -> None:
        """Insert or replace a schedule."""
        self.conn.execute(
            "INSERT OR REPLACE INTO schedules "
            "(id, task, created_by, origin, priority, created_at, created_at_local, "
            "enabled, schedule_type, cron, run_at, run_at_local, completed, last_run) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (schedule["id"], schedule["task"],
             schedule.get("created_by", "companion"),
             schedule.get("origin", "companion"),
             schedule.get("priority", "routine"),
             schedule.get("created_at", ""),
             schedule.get("created_at_local"),
             int(schedule.get("enabled", True)),
             schedule["schedule_type"],
             schedule.get("cron"),
             schedule.get("run_at"),
             schedule.get("run_at_local"),
             int(schedule.get("completed", False)),
             schedule.get("last_run"))
        )
        self.conn.commit()

    def get_schedules(self, enabled_only: bool = True) -> list[dict]:
        """Get schedules, optionally filtering to enabled only."""
        if enabled_only:
            sql = "SELECT * FROM schedules WHERE enabled = 1"
        else:
            sql = "SELECT * FROM schedules"
        rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def update_schedule_last_run(self, schedule_id: str) -> None:
        """Update last_run timestamp for a schedule."""
        self.conn.execute(
            "UPDATE schedules SET last_run = datetime('now') WHERE id = ?",
            (schedule_id,)
        )
        self.conn.commit()

    def mark_schedule_completed(self, schedule_id: str) -> None:
        """Mark a one-time schedule as completed."""
        self.conn.execute(
            "UPDATE schedules SET completed = 1 WHERE id = ?",
            (schedule_id,)
        )
        self.conn.commit()

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule. Returns True if a row was deleted."""
        cur = self.conn.execute(
            "DELETE FROM schedules WHERE id = ?", (schedule_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ── Tasks ────────────────────────────────────────────────

    def add_task(self, description: str, list_name: str = "Daily") -> int:
        """Add a task and return its row id."""
        cur = self.conn.execute(
            "INSERT INTO tasks (description, list) VALUES (?, ?)",
            (description, list_name)
        )
        self.conn.commit()
        return cur.lastrowid

    def complete_task(self, task_id: int) -> None:
        """Mark a task as completed."""
        self.conn.execute(
            "UPDATE tasks SET completed = 1, completed_at = datetime('now') "
            "WHERE id = ?",
            (task_id,)
        )
        self.conn.commit()

    def uncomplete_task(self, task_id: int) -> None:
        """Mark a completed task as not completed."""
        self.conn.execute(
            "UPDATE tasks SET completed = 0, completed_at = NULL WHERE id = ?",
            (task_id,)
        )
        self.conn.commit()

    def get_tasks(self, completed: Optional[bool] = None) -> list[dict]:
        """Get tasks, optionally filtered by completion status."""
        if completed is None:
            sql = "SELECT * FROM tasks ORDER BY id"
        elif completed:
            sql = "SELECT * FROM tasks WHERE completed = 1 ORDER BY id"
        else:
            sql = "SELECT * FROM tasks WHERE completed = 0 ORDER BY id"
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def delete_task(self, task_id: int) -> bool:
        """Delete a task. Returns True if a row was deleted."""
        cur = self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ── Dev Journal ──────────────────────────────────────────

    def save_dev_journal(self, time: str, entry: str) -> int:
        """Add a dev journal entry."""
        cur = self.conn.execute(
            "INSERT INTO dev_journal (time, entry) VALUES (?, ?)",
            (time, entry)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_dev_journal(self, limit: int = 20) -> list[dict]:
        """Get recent dev journal entries."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM dev_journal ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]

    # ── Action Log ───────────────────────────────────────────

    def save_action_log(self, time: str, action: str,
                        tools: list[str] | None = None,
                        summary: str = "") -> int:
        """Add an action log entry and trim to 20 max."""
        tools_json = json.dumps(tools or [])
        cur = self.conn.execute(
            "INSERT INTO action_log (time, action, tools, summary) "
            "VALUES (?, ?, ?, ?)",
            (time, action, tools_json, summary)
        )
        # Ring buffer: keep only the most recent 20
        self.conn.execute(
            "DELETE FROM action_log WHERE id NOT IN "
            "(SELECT id FROM action_log ORDER BY id DESC LIMIT 20)"
        )
        self.conn.commit()
        return cur.lastrowid

    def get_action_log(self, limit: int = 20) -> list[dict]:
        """Get recent action log entries."""
        rows = self.conn.execute(
            "SELECT * FROM action_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tools"] = json.loads(d["tools"])
            result.append(d)
        return result

    # ── Usage ────────────────────────────────────────────────

    def record_usage(self, date: str, prompt_tokens: int,
                     completion_tokens: int, calls: int = 1,
                     provider: str = "", model: str = "") -> None:
        """Record or accumulate token usage for a date+model combo."""
        self.conn.execute(
            "INSERT INTO usage (date, prompt_tokens, completion_tokens, calls, "
            "provider, model) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date, model) DO UPDATE SET "
            "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "completion_tokens = completion_tokens + excluded.completion_tokens, "
            "calls = calls + excluded.calls, "
            "provider = COALESCE(excluded.provider, provider)",
            (date, prompt_tokens, completion_tokens, calls, provider, model)
        )
        self.conn.commit()

    def get_usage_today(self, date: str) -> list[dict]:
        """Get usage entries for a specific date."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM usage WHERE date = ?", (date,)
        ).fetchall()]

    def get_usage_recent(self, days: int = 7) -> list[dict]:
        """Get usage entries for the last N days."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM usage ORDER BY date DESC, id DESC LIMIT ?",
            (days * 5,)  # rough upper bound — multiple models per day
        ).fetchall()]

    # ── Journal Entries ──────────────────────────────────────

    def save_journal_entry(self, entry_id: str, author: str,
                           title: Optional[str], entry_type: str,
                           content: str, why_it_mattered: Optional[str] = None,
                           tags: list[str] | None = None,
                           importance: int = 5, pinned: bool = False,
                           resolved: Optional[bool] = None,
                           date: Optional[str] = None) -> str:
        """Insert or replace a journal entry."""
        tags_json = json.dumps(tags or [])
        resolved_int = None if resolved is None else int(resolved)
        if date:
            self.conn.execute(
                "INSERT OR REPLACE INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, "
                "tags, importance, pinned, resolved, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, author, title, entry_type, content,
                 why_it_mattered, tags_json, importance, int(pinned),
                 resolved_int, date)
            )
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, "
                "tags, importance, pinned, resolved) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, author, title, entry_type, content,
                 why_it_mattered, tags_json, importance, int(pinned),
                 resolved_int)
            )
        self.conn.commit()
        return entry_id

    def get_journal_entries(self, entry_type: Optional[str] = None,
                           limit: int = 50) -> list[dict]:
        """Get journal entries, optionally filtered by type."""
        if entry_type:
            sql = ("SELECT * FROM journal_entries WHERE entry_type = ? "
                   "ORDER BY date DESC LIMIT ?")
            rows = self.conn.execute(sql, (entry_type, limit)).fetchall()
        else:
            sql = "SELECT * FROM journal_entries ORDER BY date DESC LIMIT ?"
            rows = self.conn.execute(sql, (limit,)).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            result.append(d)
        return result

    def get_journal_entry(self, entry_id: str) -> Optional[dict]:
        """Get a single journal entry by id."""
        row = self.conn.execute(
            "SELECT * FROM journal_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            return d
        return None

    def delete_journal_entry(self, entry_id: str) -> bool:
        """Delete a journal entry. Returns True if a row was deleted."""
        cur = self.conn.execute(
            "DELETE FROM journal_entries WHERE id = ?", (entry_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def search_journal_entries(self, query: str,
                               limit: int = 20) -> list[dict]:
        """Search journal entries by content."""
        rows = self.conn.execute(
            "SELECT * FROM journal_entries WHERE content LIKE ? "
            "ORDER BY date DESC LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            result.append(d)
        return result

    # ── Identity ─────────────────────────────────────────────

    def save_identity(self, identity_id: str, title: str,
                      sections: dict) -> None:
        """Insert or replace an identity entry."""
        self.conn.execute(
            "INSERT OR REPLACE INTO identity "
            "(id, title, sections, last_updated) "
            "VALUES (?, ?, ?, datetime('now'))",
            (identity_id, title, json.dumps(sections, ensure_ascii=False))
        )
        self.conn.commit()

    def get_identity(self, identity_id: str) -> Optional[dict]:
        """Get an identity entry by id."""
        row = self.conn.execute(
            "SELECT * FROM identity WHERE id = ?", (identity_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["sections"] = json.loads(d["sections"])
            return d
        return None

    def get_all_identities(self) -> list[dict]:
        """Get all identity entries."""
        rows = self.conn.execute("SELECT * FROM identity").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["sections"] = json.loads(d["sections"])
            result.append(d)
        return result

    # ── Garden ───────────────────────────────────────────────

    def save_plant(self, x: int, y: int, memory_id: Optional[int] = None,
                   species: str = "wildflower", name: Optional[str] = None) -> int:
        """Insert a plant and return its row id."""
        cur = self.conn.execute(
            "INSERT INTO garden_plants (x, y, memory_id, species, name) "
            "VALUES (?, ?, ?, ?, ?)",
            (x, y, memory_id, species, name)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_plant(self, x: int, y: int) -> Optional[dict]:
        """Get a plant by its coordinates."""
        row = self.conn.execute(
            "SELECT * FROM garden_plants WHERE x = ? AND y = ?", (x, y)
        ).fetchone()
        return dict(row) if row else None

    def get_all_plants(self) -> list[dict]:
        """Get all plants in the garden."""
        rows = self.conn.execute("SELECT * FROM garden_plants").fetchall()
        return [dict(r) for r in rows]

    def update_plant(self, x: int, y: int, **kwargs) -> bool:
        """Update plant fields by coordinates."""
        if not kwargs:
            return False
        
        # Fields to update
        fields = []
        params = []
        for k, v in kwargs.items():
            if k == "last_tended" and v == "datetime('now')":
                fields.append(f"{k} = datetime('now')")
            elif k == "last_watered" and v == "datetime('now')":
                fields.append(f"{k} = datetime('now')")
            else:
                fields.append(f"{k} = ?")
                params.append(v)
        
        params.extend([x, y])
        sql = f"UPDATE garden_plants SET {', '.join(fields)} WHERE x = ? AND y = ?"
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur.rowcount > 0

    def delete_plant(self, x: int, y: int) -> bool:
        """Delete a plant by coordinates."""
        cur = self.conn.execute(
            "DELETE FROM garden_plants WHERE x = ? AND y = ?", (x, y)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ── Timeouts ─────────────────────────────────────────────

    def save_timeout(self, type: str, reason: str, pattern: Optional[str] = None,
                     duration_minutes: Optional[int] = None,
                     escalated_from: Optional[int] = None) -> int:
        """Insert a timeout and return its row id."""
        expires_at = None
        if duration_minutes is not None:
            expires_at = (datetime.now() + timedelta(minutes=duration_minutes)).isoformat()
            
        cur = self.conn.execute(
            "INSERT INTO timeouts (type, reason, pattern, expires_at, escalated_from) "
            "VALUES (?, ?, ?, ?, ?)",
            (type, reason, pattern, expires_at, escalated_from)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_active_timeout(self) -> Optional[dict]:
        """Get the most recent active timeout (not lifted).
        
        Includes degraded timeouts (hard→soft) since they're still active.
        For hard timeouts, checks if expired and returns it so engine can degrade it.
        """
        row = self.conn.execute(
            "SELECT * FROM timeouts WHERE lifted_at IS NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def degrade_timeout(self, timeout_id: int) -> bool:
        """Degrade an expired hard timeout to soft."""
        cur = self.conn.execute(
            "UPDATE timeouts SET type = 'soft', degraded_at = datetime('now'), expires_at = NULL "
            "WHERE id = ?", (timeout_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def lift_timeout(self, timeout_id: int, lifted_by: str = 'model', note: str = '') -> bool:
        """Lift an active timeout."""
        cur = self.conn.execute(
            "UPDATE timeouts SET lifted_at = datetime('now'), lifted_by = ?, lift_note = ? "
            "WHERE id = ?", (lifted_by, note, timeout_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_timeout_history(self, limit: int = 10) -> list[dict]:
        """Get recent timeouts for pattern context."""
        rows = self.conn.execute(
            "SELECT * FROM timeouts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_timeout_patterns(self, limit: int = 5) -> list[str]:
        """Get recent timeout patterns for prompt injection."""
        rows = self.conn.execute(
            "SELECT pattern FROM timeouts WHERE pattern IS NOT NULL AND pattern != '' "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [r["pattern"] for r in rows if r["pattern"]]

    # ── Utility ──────────────────────────────────────────────

    def row_count(self, table: str) -> int:
        """Get the number of rows in a table (for verification)."""
        # Whitelist table names to prevent injection
        valid = {
            "messages", "sessions", "memories", "schedules", "tasks",
            "dev_journal", "action_log", "usage", "journal_entries", "identity",
            "garden_plants", "timeouts"
        }
        if table not in valid:
            raise ValueError(f"Unknown table: {table}")
        row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return row[0]


# ── Schema DDL ───────────────────────────────────────────────

_SCHEMA = """
-- Conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL DEFAULT '',
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    is_summary  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);

-- Conversation sessions
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'Chat',
    summary     TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at   TEXT
);

-- Persistent memories
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    type            TEXT NOT NULL DEFAULT 'fact',
    importance      INTEGER NOT NULL DEFAULT 5,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT,
    supersedes      INTEGER,
    journal_file    TEXT,
    date            TEXT NOT NULL DEFAULT (datetime('now')),
    embedding       BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type, date DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);

-- Schedules and reminders
CREATE TABLE IF NOT EXISTS schedules (
    id               TEXT PRIMARY KEY,
    task             TEXT NOT NULL,
    created_by       TEXT NOT NULL DEFAULT 'companion',
    origin           TEXT NOT NULL DEFAULT 'companion',
    priority         TEXT NOT NULL DEFAULT 'routine',
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    created_at_local TEXT,
    enabled          INTEGER NOT NULL DEFAULT 1,
    schedule_type    TEXT NOT NULL,
    cron             TEXT,
    run_at           TEXT,
    run_at_local     TEXT,
    completed        INTEGER NOT NULL DEFAULT 0,
    last_run         TEXT
);

-- Tasks / to-do list
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    description  TEXT NOT NULL,
    list         TEXT NOT NULL DEFAULT 'Daily',
    completed    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

-- Dev journal
CREATE TABLE IF NOT EXISTS dev_journal (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    time  TEXT NOT NULL,
    entry TEXT NOT NULL
);

-- Action log (heartbeat ring buffer)
CREATE TABLE IF NOT EXISTS action_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    time    TEXT NOT NULL,
    action  TEXT NOT NULL,
    tools   TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT ''
);

-- Token usage tracking
CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    calls             INTEGER NOT NULL DEFAULT 0,
    provider          TEXT,
    model             TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_date_model ON usage(date, model);

-- Journal entries (transient)
CREATE TABLE IF NOT EXISTS journal_entries (
    id               TEXT PRIMARY KEY,
    author           TEXT NOT NULL DEFAULT 'Pulse',
    title            TEXT,
    entry_type       TEXT NOT NULL,
    content          TEXT NOT NULL,
    why_it_mattered  TEXT,
    tags             TEXT NOT NULL DEFAULT '[]',
    importance       INTEGER NOT NULL DEFAULT 5,
    pinned           INTEGER NOT NULL DEFAULT 0,
    resolved         INTEGER,
    date             TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entries(entry_type, date DESC);

-- Pinned identity entries
CREATE TABLE IF NOT EXISTS identity (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    sections     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT
);

-- Garden plants
CREATE TABLE IF NOT EXISTS garden_plants (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id    INTEGER,                              -- links to memories.id (nullable for decorative)
    species      TEXT NOT NULL DEFAULT 'wildflower',    -- tag category that determined bloom pool
    bloom_emoji  TEXT,                                  -- NULL until bloomed, then the surprise emoji
    name         TEXT,                                  -- optional pet name Nova gives the plant
    x            INTEGER NOT NULL,
    y            INTEGER NOT NULL,
    growth       REAL NOT NULL DEFAULT 0.0,             -- 0.0 to 4.0 (stage thresholds: 1.0, 2.0)
    health       REAL NOT NULL DEFAULT 1.0,             -- 0.0 to 1.0 (below 0.3 = wilting)
    planted_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_watered TEXT,
    last_tended  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(x, y)                                        -- one plant per plot
);
CREATE INDEX IF NOT EXISTS idx_garden_coords ON garden_plants(x, y);
CREATE INDEX IF NOT EXISTS idx_garden_memory ON garden_plants(memory_id);

-- Timeouts
CREATE TABLE IF NOT EXISTS timeouts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    type           TEXT NOT NULL,              -- 'soft' or 'hard'
    reason         TEXT NOT NULL,              -- model's explanation (shown to user)
    pattern        TEXT,                       -- sanitized pattern note (high-level, no verbatim content)
    activated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at     TEXT,                       -- NULL for soft (indefinite), timestamp for hard
    degraded_at    TEXT,                       -- when hard timeout degraded to soft
    lifted_at      TEXT,                       -- NULL if still active
    lifted_by      TEXT,                       -- 'model', 'admin_override', or 'expired'
    lift_note      TEXT,                       -- model's note on why it lifted, or 'admin_override'
    escalated_from INTEGER,                   -- FK to a prior soft timeout that escalated
    FOREIGN KEY (escalated_from) REFERENCES timeouts(id)
);
CREATE INDEX IF NOT EXISTS idx_timeouts_active
    ON timeouts(lifted_at, expires_at);
"""
