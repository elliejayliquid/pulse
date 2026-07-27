"""Sanitized conversation archive stored inside a persona database.

The archive is the middle layer between mutable import staging and live
conversation history:

    import_* staging -> legacy_* archive -> sessions/messages

This module implements only the first boundary.  It never writes live
sessions, messages, memories, journal entries, or prompt context.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARCHIVE_SCHEMA_VERSION = 1

_ARCHIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS legacy_sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'unknown',
    source_conversation_id TEXT NOT NULL DEFAULT '',
    import_batch_id TEXT NOT NULL DEFAULT '',
    source_version TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT NOT NULL DEFAULT '',
    live_session_id TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS legacy_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    source_message_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    source_role TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    attachment_placeholder INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES legacy_sessions(id) ON DELETE CASCADE,
    UNIQUE (session_id, position)
);
CREATE TABLE IF NOT EXISTS legacy_archive_migrations (
    source_path TEXT PRIMARY KEY,
    source_size INTEGER NOT NULL DEFAULT 0,
    source_mtime_ns INTEGER NOT NULL DEFAULT 0,
    migrated_at TEXT NOT NULL,
    session_count INTEGER NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0,
    summary_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS legacy_archive_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS legacy_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    summary_embedding BLOB,
    chunk_index INTEGER DEFAULT 0,
    first_msg_id INTEGER,
    last_msg_id INTEGER,
    is_topic_boundary INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ls_session ON legacy_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_ls_chunk ON legacy_summaries(session_id, chunk_index);
"""

_SESSION_COLUMNS = {
    "source": "TEXT NOT NULL DEFAULT 'unknown'",
    "source_conversation_id": "TEXT NOT NULL DEFAULT ''",
    "import_batch_id": "TEXT NOT NULL DEFAULT ''",
    "source_version": "TEXT NOT NULL DEFAULT ''",
    "title": "TEXT NOT NULL DEFAULT ''",
    "created_at": "TEXT NOT NULL DEFAULT ''",
    "updated_at": "TEXT NOT NULL DEFAULT ''",
    "model": "TEXT NOT NULL DEFAULT ''",
    "message_count": "INTEGER NOT NULL DEFAULT 0",
    "archived_at": "TEXT NOT NULL DEFAULT ''",
    "live_session_id": "TEXT NOT NULL DEFAULT ''",
}

_MESSAGE_COLUMNS = {
    "position": "INTEGER NOT NULL DEFAULT 0",
    "source_message_id": "TEXT NOT NULL DEFAULT ''",
    "source_role": "TEXT NOT NULL DEFAULT ''",
    "timestamp": "TEXT NOT NULL DEFAULT ''",
    "model": "TEXT NOT NULL DEFAULT ''",
    "attachment_placeholder": "INTEGER NOT NULL DEFAULT 0",
}


def ensure_legacy_archive_tables(conn: sqlite3.Connection) -> None:
    """Create or gently upgrade the persona-local archive schema."""
    conn.executescript(_ARCHIVE_SCHEMA)
    conn.execute(
        "INSERT INTO legacy_archive_meta(key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(ARCHIVE_SCHEMA_VERSION),),
    )
    _ensure_columns(conn, "legacy_sessions", _SESSION_COLUMNS)
    _ensure_columns(conn, "legacy_messages", _MESSAGE_COLUMNS)
    conn.execute(
        "UPDATE legacy_sessions "
        "SET source_conversation_id = id "
        "WHERE source_conversation_id = ''"
    )
    conn.execute(
        "UPDATE legacy_sessions "
        "SET archived_at = CASE WHEN created_at <> '' THEN created_at ELSE ? END "
        "WHERE archived_at = ''",
        (_now(),),
    )
    _backfill_message_positions(conn)
    conn.executescript("""
    CREATE INDEX IF NOT EXISTS idx_legacy_sessions_date
    ON legacy_sessions(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_legacy_messages_session
    ON legacy_messages(session_id, position);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_legacy_message_position
    ON legacy_messages(session_id, position);
    CREATE INDEX IF NOT EXISTS idx_legacy_messages_role
    ON legacy_messages(role);
    CREATE INDEX IF NOT EXISTS idx_legacy_messages_timestamp
    ON legacy_messages(timestamp DESC);
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_legacy_source_conversation "
        "ON legacy_sessions(source, source_conversation_id)"
    )
    _ensure_legacy_fts(conn)


def archive_batch_summary(conn: sqlite3.Connection, batch_id: str) -> dict[str, Any]:
    """Return a trusted staging-to-archive confirmation summary."""
    ensure_legacy_archive_tables(conn)
    batch = _archive_batch_row(conn, batch_id)
    candidates = _archive_candidates(conn, batch_id, str(batch[0] or "unknown"))
    return _summarize_archive_candidates(conn, batch_id, batch, candidates)


def _summarize_archive_candidates(
    conn: sqlite3.Connection,
    batch_id: str,
    batch: sqlite3.Row | tuple,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build confirmation counts from an already-sanitized candidate scan."""
    kept_total = conn.execute(
        "SELECT COUNT(*) FROM import_conversations "
        "WHERE batch_id = ? AND choice = 'keep'",
        (batch_id,),
    ).fetchone()[0]
    excluded_conversations = conn.execute(
        "SELECT COUNT(*) FROM import_conversations "
        "WHERE batch_id = ? AND choice = 'exclude'",
        (batch_id,),
    ).fetchone()[0]
    excluded_messages = conn.execute(
        "SELECT COUNT(*) FROM import_messages m "
        "JOIN import_conversations c "
        "ON c.batch_id = m.batch_id AND c.conversation_id = m.conversation_id "
        "WHERE m.batch_id = ? AND c.choice = 'keep' AND m.excluded = 1",
        (batch_id,),
    ).fetchone()[0]
    omitted_messages = conn.execute(
        "SELECT COUNT(*) FROM import_messages m "
        "JOIN import_conversations c "
        "ON c.batch_id = m.batch_id AND c.conversation_id = m.conversation_id "
        "WHERE m.batch_id = ? AND c.choice = 'keep' AND m.excluded = 0 "
        "AND (m.skipped = 1 OR m.role NOT IN ('user', 'assistant') OR TRIM(m.content) = '')",
        (batch_id,),
    ).fetchone()[0]

    already_archived = [item for item in candidates if item["already_archived"]]
    ready = [
        item for item in candidates
        if not item["already_archived"] and item["messages"]
    ]
    empty_after_sanitation = [
        item for item in candidates if not item["messages"]
    ]
    continuation_id = str(batch[2] or "")
    continuation = next(
        (item for item in candidates if item["conversation_id"] == continuation_id),
        None,
    )
    continuation_summary = None
    if continuation_id:
        if continuation:
            continuation_summary = {
                "conversation_id": continuation_id,
                "title": continuation["title"],
                "will_be_added": bool(
                    continuation["messages"] and not continuation["already_archived"]
                ),
                "already_archived": bool(continuation["already_archived"]),
                "message_count": len(continuation["messages"]),
            }
        else:
            continuation_summary = {
                "conversation_id": continuation_id,
                "title": "",
                "will_be_added": False,
                "already_archived": False,
                "message_count": 0,
            }

    return {
        "batch_id": batch_id,
        "source": str(batch[0] or "unknown"),
        "source_version": str(batch[1] or ""),
        "kept_conversations": kept_total,
        "conversations_to_add": len(ready),
        "messages_to_add": sum(len(item["messages"]) for item in ready),
        "excluded_conversations": excluded_conversations,
        "excluded_messages": excluded_messages,
        "omitted_messages": omitted_messages,
        "already_archived_conversations": len(already_archived),
        "empty_after_sanitation": len(empty_after_sanitation),
        "continuation": continuation_summary,
        "can_archive": bool(ready),
    }


def archive_staged_batch(conn: sqlite3.Connection, batch_id: str) -> dict[str, Any]:
    """Copy kept, sanitized staging rows into the write-once archive."""
    ensure_legacy_archive_tables(conn)
    batch = _archive_batch_row(conn, batch_id)
    candidates = _archive_candidates(conn, batch_id, str(batch[0] or "unknown"))
    summary = _summarize_archive_candidates(conn, batch_id, batch, candidates)
    archived_at = _now()
    added_conversations = 0
    added_messages = 0

    for candidate in candidates:
        if candidate["already_archived"] or not candidate["messages"]:
            continue
        archive_id = _archive_id(
            candidate["source"],
            candidate["source_conversation_id"],
        )
        try:
            conn.execute(
                "INSERT INTO legacy_sessions "
                "(id, source, source_conversation_id, import_batch_id, source_version, "
                " title, created_at, updated_at, model, message_count, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    archive_id,
                    candidate["source"],
                    candidate["source_conversation_id"],
                    batch_id,
                    str(batch[1] or ""),
                    candidate["title"],
                    candidate["created_at"],
                    candidate["updated_at"],
                    candidate["model"],
                    len(candidate["messages"]),
                    archived_at,
                ),
            )
        except sqlite3.IntegrityError:
            continue
        conn.executemany(
            "INSERT INTO legacy_messages "
            "(session_id, position, source_message_id, role, source_role, content, "
            " timestamp, model, attachment_placeholder) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    archive_id,
                    position,
                    message["source_message_id"],
                    message["role"],
                    message["source_role"],
                    message["content"],
                    message["timestamp"],
                    message["model"],
                    int(message["attachment_placeholder"]),
                )
                for position, message in enumerate(candidate["messages"])
            ],
        )
        added_conversations += 1
        added_messages += len(candidate["messages"])

    return {
        **summary,
        "added_conversations": added_conversations,
        "added_messages": added_messages,
    }


def _archive_batch_row(
    conn: sqlite3.Connection,
    batch_id: str,
) -> sqlite3.Row | tuple:
    batch = conn.execute(
        "SELECT source, source_version, continuation_conversation_id "
        "FROM import_batches WHERE id = ?",
        (batch_id,),
    ).fetchone()
    if not batch:
        raise ValueError("Staged import not found.")
    return batch


def archive_batch_status(conn: sqlite3.Connection, batch_id: str) -> dict[str, dict[str, Any]]:
    """Map staged conversation IDs to their current archive state."""
    ensure_legacy_archive_tables(conn)
    batch = conn.execute(
        "SELECT source FROM import_batches WHERE id = ?",
        (batch_id,),
    ).fetchone()
    if not batch:
        raise ValueError("Staged import not found.")
    rows = conn.execute(
        "SELECT c.conversation_id, c.source_id, c.title, "
        "s.id, s.message_count, s.archived_at, s.live_session_id "
        "FROM import_conversations c "
        "LEFT JOIN legacy_sessions s "
        "ON s.source = ? "
        "AND s.source_conversation_id = CASE "
        "WHEN c.source_id <> '' THEN c.source_id ELSE c.conversation_id END "
        "WHERE c.batch_id = ? ORDER BY c.position",
        (str(batch[0] or "unknown"), batch_id),
    ).fetchall()
    return {
        str(row[0]): {
            "archived": bool(row[3]),
            "archive_id": str(row[3] or ""),
            "title": str(row[2] or ""),
            "message_count": int(row[4] or 0),
            "archived_at": str(row[5] or ""),
            "live_session_id": str(row[6] or ""),
            "can_remove": bool(row[3]) and not bool(row[6]),
        }
        for row in rows
    }


def continuation_preflight(
    conn: sqlite3.Connection,
    batch_id: str,
    landing_budget_tokens: int,
    recent_tail_exchanges: int,
) -> dict[str, Any]:
    """Describe archive-to-live work without changing any database rows."""
    batch = conn.execute(
        "SELECT source, source_version, continuation_conversation_id "
        "FROM import_batches WHERE id = ?",
        (batch_id,),
    ).fetchone()
    if not batch:
        raise ValueError("Staged import not found.")

    source = str(batch[0] or "unknown")
    source_version = str(batch[1] or "")
    conversation_id = str(batch[2] or "")
    if not conversation_id:
        raise ValueError("Choose Continue on one conversation before reviewing continuation.")

    conversation = conn.execute(
        "SELECT source_id, title, created_at, updated_at, model, choice "
        "FROM import_conversations WHERE batch_id = ? AND conversation_id = ?",
        (batch_id, conversation_id),
    ).fetchone()
    if not conversation:
        raise ValueError("Selected continuation is not part of this staged import.")
    if str(conversation[5] or "") != "keep":
        raise ValueError("Selected continuation must be kept in the staged review.")

    source_conversation_id = str(conversation[0] or conversation_id)
    archived = conn.execute(
        "SELECT id, title, created_at, updated_at, model, message_count, "
        "archived_at, live_session_id "
        "FROM legacy_sessions WHERE source = ? AND source_conversation_id = ?",
        (source, source_conversation_id),
    ).fetchone()
    if not archived:
        raise ValueError(
            "Add the selected conversation to searchable history before continuing it."
        )

    archive_id = str(archived[0])
    message_rows = conn.execute(
        "SELECT role, content, timestamp FROM legacy_messages "
        "WHERE session_id = ? AND role IN ('user', 'assistant') "
        "ORDER BY position, id",
        (archive_id,),
    ).fetchall()
    if not message_rows:
        raise ValueError("Selected archived conversation has no usable dialogue.")

    landing_budget_tokens = max(1, int(landing_budget_tokens or 1))
    recent_tail_exchanges = max(1, int(recent_tail_exchanges or 1))
    estimated_tokens = _message_token_estimate(message_rows)
    requires_bridge = estimated_tokens > landing_budget_tokens
    if requires_bridge:
        tail = _recent_tail_plan(
            message_rows,
            exchange_limit=recent_tail_exchanges,
            token_budget=landing_budget_tokens,
        )
        strategy = "bridge_summary_recent_tail"
    else:
        tail = {
            "message_count": len(message_rows),
            "exchange_count": _exchange_count(message_rows),
            "estimated_tokens": estimated_tokens,
        }
        strategy = "full_conversation"

    summary_count = conn.execute(
        "SELECT COUNT(*) FROM legacy_summaries WHERE session_id = ?",
        (archive_id,),
    ).fetchone()[0]
    active = _active_session_preflight(conn)
    live_session_id = str(archived[7] or "")
    first_timestamp = next((str(row[2]) for row in message_rows if row[2]), "")
    last_timestamp = next(
        (str(row[2]) for row in reversed(message_rows) if row[2]),
        "",
    )

    return {
        "batch_id": batch_id,
        "conversation_id": conversation_id,
        "archive": {
            "id": archive_id,
            "source": source,
            "source_version": source_version,
            "source_conversation_id": source_conversation_id,
            "title": str(archived[1] or conversation[1] or ""),
            "created_at": str(archived[2] or conversation[2] or first_timestamp),
            "updated_at": str(archived[3] or conversation[3] or last_timestamp),
            "first_message_at": first_timestamp,
            "last_message_at": last_timestamp,
            "model": str(archived[4] or conversation[4] or ""),
            "message_count": len(message_rows),
            "estimated_tokens": estimated_tokens,
            "archived_at": str(archived[6] or ""),
            "summary_count": int(summary_count),
            "keyword_search_ready": True,
            "live_session_id": live_session_id,
        },
        "freshness": {
            "classification": "historical_import",
            "label": "Dated imported background",
            "note": (
                "Treat claims from this archived period as historical context "
                "until they are confirmed in the current relationship."
            ),
        },
        "current_session": active,
        "bridge_preparation": {
            "required": requires_bridge,
            "status": "preparation_required" if requires_bridge else "not_required",
            "method": "hierarchical_chunk_rollup" if requires_bridge else "none",
            "existing_archive_summary_rows": int(summary_count),
            "coverage_verified": False if requires_bridge else None,
            "reuse_policy": (
                "Existing summaries are reusable only after compatible type, "
                "generation lineage, and gap-free source-span coverage are verified."
            ),
        },
        "landing": {
            "strategy": strategy,
            "configured_budget_tokens": landing_budget_tokens,
            "requires_bridge_summary": requires_bridge,
            "seed_message_count": int(tail["message_count"]),
            "seed_exchange_count": int(tail["exchange_count"]),
            "seed_estimated_tokens": int(tail["estimated_tokens"]),
            "seed_within_budget": int(tail["estimated_tokens"]) <= landing_budget_tokens,
            "recent_tail_exchange_limit": recent_tail_exchanges,
        },
        "safety": {
            "read_only": True,
            "backup_required_before_apply": True,
            "archive_remains_intact": True,
            "idempotency_guard": "legacy_sessions.live_session_id",
            "already_continued": bool(live_session_id),
            "must_stop_persona_before_apply": True,
        },
        "can_apply_later": not bool(live_session_id),
        "execution_available": False,
    }


def _active_session_preflight(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return compact current-session state for a continuation preflight."""
    session = conn.execute(
        "SELECT id, title, created_at, closed_at FROM sessions "
        "WHERE closed_at IS NULL ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not session:
        return {
            "exists": False,
            "message_count": 0,
            "estimated_tokens": 0,
            "action": "start_landing_session",
        }

    message_rows = conn.execute(
        "SELECT role, content, timestamp FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (str(session[0]),),
    ).fetchall()
    first_timestamp = next((str(row[2]) for row in message_rows if row[2]), "")
    last_timestamp = next(
        (str(row[2]) for row in reversed(message_rows) if row[2]),
        "",
    )
    return {
        "exists": True,
        "id": str(session[0]),
        "title": str(session[1] or "Chat"),
        "created_at": str(session[2] or ""),
        "first_message_at": first_timestamp,
        "last_message_at": last_timestamp,
        "message_count": len(message_rows),
        "estimated_tokens": _message_token_estimate(message_rows),
        "action": (
            "summarize_and_close"
            if message_rows
            else "close_without_summary"
        ),
    }


def _message_token_estimate(rows: list[sqlite3.Row | tuple]) -> int:
    """Use the same deliberately rough four-chars-per-token import estimate."""
    characters = sum(len(str(row[1] or "")) for row in rows)
    return max(1, characters // 4) if rows else 0


def _exchange_count(rows: list[sqlite3.Row | tuple]) -> int:
    """Count user-led exchange groups while tolerating imperfect old transcripts."""
    exchanges = 0
    previous_role = ""
    for row in rows:
        role = str(row[0] or "")
        if role == "user" and previous_role != "user":
            exchanges += 1
        elif exchanges == 0:
            exchanges = 1
        previous_role = role
    return exchanges


def _recent_tail_plan(
    rows: list[sqlite3.Row | tuple],
    exchange_limit: int,
    token_budget: int,
) -> dict[str, int]:
    """Plan a recent raw tail using whole user-led exchange boundaries."""
    exchanges: list[list[sqlite3.Row | tuple]] = []
    current: list[sqlite3.Row | tuple] = []
    previous_role = ""
    for row in rows:
        role = str(row[0] or "")
        if current and role == "user" and previous_role != "user":
            exchanges.append(current)
            current = []
        current.append(row)
        previous_role = role
    if current:
        exchanges.append(current)

    selected = exchanges[-max(1, exchange_limit):]
    while len(selected) > 1:
        flattened = [row for exchange in selected for row in exchange]
        if _message_token_estimate(flattened) <= token_budget:
            break
        selected.pop(0)
    flattened = [row for exchange in selected for row in exchange]
    return {
        "message_count": len(flattened),
        "exchange_count": len(selected),
        "estimated_tokens": _message_token_estimate(flattened),
    }


def remove_archived_conversation(
    conn: sqlite3.Connection,
    batch_id: str,
    conversation_id: str,
) -> dict[str, Any]:
    """Remove one staged conversation's archive copy for correction/re-add."""
    status = archive_batch_status(conn, batch_id)
    item = status.get(str(conversation_id or ""))
    if not item:
        raise ValueError("Conversation not found in this staged import.")
    if not item["archived"]:
        raise ValueError("This conversation is not in the archive.")
    if item["live_session_id"]:
        raise ValueError(
            "This archived conversation has already been continued into a live session "
            "and cannot be removed here."
        )

    archive_id = item["archive_id"]
    message_count = conn.execute(
        "SELECT COUNT(*) FROM legacy_messages WHERE session_id = ?",
        (archive_id,),
    ).fetchone()[0]
    summary_count = conn.execute(
        "SELECT COUNT(*) FROM legacy_summaries WHERE session_id = ?",
        (archive_id,),
    ).fetchone()[0]
    enrichment_chunk_count = 0
    enrichment_artifact_count = 0
    enrichment_chunk_ids: list[str] = []
    if _table_exists(conn, "legacy_enrichment_chunks"):
        enrichment_chunk_ids = [
            str(row[0])
            for row in conn.execute(
                "SELECT id FROM legacy_enrichment_chunks WHERE session_id = ?",
                (archive_id,),
            ).fetchall()
        ]
        enrichment_chunk_count = len(enrichment_chunk_ids)
    if _table_exists(conn, "legacy_enrichment_artifacts"):
        enrichment_artifact_count = int(conn.execute(
            "SELECT COUNT(*) FROM legacy_enrichment_artifacts WHERE session_id = ?",
            (archive_id,),
        ).fetchone()[0])
    removed = conn.execute(
        "DELETE FROM legacy_sessions WHERE id = ? AND live_session_id = ''",
        (archive_id,),
    ).rowcount
    if not removed:
        raise ValueError("The archive changed before removal; reload and try again.")
    if enrichment_chunk_ids and _table_exists(conn, "legacy_enrichment_job_chunks"):
        placeholders = ",".join("?" for _ in enrichment_chunk_ids)
        conn.execute(
            f"DELETE FROM legacy_enrichment_job_chunks WHERE chunk_id IN ({placeholders})",
            tuple(enrichment_chunk_ids),
        )
    if _table_exists(conn, "legacy_enrichment_artifacts"):
        conn.execute(
            "DELETE FROM legacy_enrichment_artifacts WHERE session_id = ?",
            (archive_id,),
        )
    if _table_exists(conn, "legacy_enrichment_chunks"):
        conn.execute(
            "DELETE FROM legacy_enrichment_chunks WHERE session_id = ?",
            (archive_id,),
        )
    conn.execute("DELETE FROM legacy_summaries WHERE session_id = ?", (archive_id,))
    conn.execute("DELETE FROM legacy_messages WHERE session_id = ?", (archive_id,))
    return {
        "conversation_id": str(conversation_id),
        "archive_id": archive_id,
        "title": item["title"],
        "removed_messages": int(message_count),
        "removed_summaries": int(summary_count),
        "removed_enrichment_chunks": enrichment_chunk_count,
        "removed_enrichment_artifacts": enrichment_artifact_count,
    }


def legacy_migration_preview(
    conn: sqlite3.Connection,
    source_path: Path | None,
) -> dict[str, Any]:
    """Inspect an old standalone legacy.db without changing either database."""
    if not source_path or not source_path.is_file():
        return {"pending": False}
    ensure_legacy_archive_tables(conn)
    source_path = source_path.resolve()
    migrated = conn.execute(
        "SELECT session_count, message_count, summary_count "
        "FROM legacy_archive_migrations WHERE source_path = ?",
        (str(source_path),),
    ).fetchone()
    if migrated:
        return {
            "pending": False,
            "already_migrated": True,
            "source_path": str(source_path),
            "session_count": migrated[0],
            "message_count": migrated[1],
            "summary_count": migrated[2],
        }
    try:
        with closing(sqlite3.connect(str(source_path))) as source:
            if not _table_exists(source, "legacy_sessions"):
                return {"pending": False}
            return {
                "pending": True,
                "source_path": str(source_path),
                "session_count": _table_count(source, "legacy_sessions"),
                "message_count": _table_count(source, "legacy_messages"),
                "summary_count": _table_count(source, "legacy_summaries"),
            }
    except (OSError, sqlite3.Error) as exc:
        return {"pending": False, "error": str(exc)}


def migrate_legacy_database(
    conn: sqlite3.Connection,
    source_path: Path | None,
    *,
    ensure_schema: bool = True,
) -> dict[str, Any]:
    """One-time copy of a developer-era legacy.db into the persona database."""
    if ensure_schema:
        ensure_legacy_archive_tables(conn)
    if not source_path or not source_path.is_file():
        return {"migrated": False, "reason": "not_found"}

    source_path = source_path.resolve()
    destination_path = _connection_path(conn)
    if destination_path and source_path == destination_path:
        return {"migrated": False, "reason": "same_database"}
    migration_key = str(source_path)
    existing = conn.execute(
        "SELECT session_count, message_count, summary_count "
        "FROM legacy_archive_migrations WHERE source_path = ?",
        (migration_key,),
    ).fetchone()
    if existing:
        return {
            "migrated": False,
            "reason": "already_migrated",
            "session_count": existing[0],
            "message_count": existing[1],
            "summary_count": existing[2],
        }

    stat = source_path.stat()
    migrated_sessions = 0
    migrated_messages = 0
    migrated_summaries = 0
    skipped_sessions = 0
    with closing(sqlite3.connect(str(source_path))) as source:
        source.row_factory = sqlite3.Row
        if not _table_exists(source, "legacy_sessions"):
            return {"migrated": False, "reason": "missing_tables"}
        session_rows = source.execute(
            "SELECT * FROM legacy_sessions ORDER BY created_at, id"
        ).fetchall()
        for session in session_rows:
            old_session_id = str(_row_value(session, "id") or "")
            if not old_session_id:
                continue
            source_name = str(_row_value(session, "source") or "legacy")
            source_conversation_id = old_session_id
            archived_id = _archive_id(source_name, source_conversation_id)
            duplicate = conn.execute(
                "SELECT id FROM legacy_sessions "
                "WHERE source = ? AND source_conversation_id = ?",
                (source_name, source_conversation_id),
            ).fetchone()
            if duplicate:
                skipped_sessions += 1
                continue

            message_rows = []
            if _table_exists(source, "legacy_messages"):
                message_rows = source.execute(
                    "SELECT * FROM legacy_messages WHERE session_id = ? ORDER BY id",
                    (old_session_id,),
                ).fetchall()
            clean_messages = [
                row for row in message_rows
                if str(_row_value(row, "role") or "") in ("user", "assistant")
                and str(_row_value(row, "content") or "").strip()
            ]
            if not clean_messages:
                continue
            created_at = str(_row_value(session, "created_at") or "")
            conn.execute(
                "INSERT INTO legacy_sessions "
                "(id, source, source_conversation_id, import_batch_id, source_version, "
                " title, created_at, updated_at, model, message_count, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    archived_id,
                    source_name,
                    source_conversation_id,
                    f"legacy-db:{_path_digest(source_path)}",
                    "legacy.db",
                    str(_row_value(session, "title") or ""),
                    created_at,
                    created_at,
                    "",
                    len(clean_messages),
                    _now(),
                ),
            )
            message_id_map: dict[int, int] = {}
            for position, message in enumerate(clean_messages):
                cursor = conn.execute(
                    "INSERT INTO legacy_messages "
                    "(session_id, position, source_message_id, role, source_role, content, "
                    " timestamp, model, attachment_placeholder) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        archived_id,
                        position,
                        str(_row_value(message, "id") or ""),
                        str(_row_value(message, "role") or ""),
                        str(_row_value(message, "role") or ""),
                        str(_row_value(message, "content") or ""),
                        str(_row_value(message, "timestamp") or created_at),
                        "",
                    ),
                )
                old_message_id = _row_value(message, "id")
                if isinstance(old_message_id, int):
                    message_id_map[old_message_id] = int(cursor.lastrowid)
                migrated_messages += 1
            migrated_sessions += 1

            if _table_exists(source, "legacy_summaries"):
                summary_rows = source.execute(
                    "SELECT * FROM legacy_summaries WHERE session_id = ? ORDER BY chunk_index, id",
                    (old_session_id,),
                ).fetchall()
                for summary in summary_rows:
                    conn.execute(
                        "INSERT INTO legacy_summaries "
                        "(session_id, summary_text, summary_embedding, chunk_index, "
                        " first_msg_id, last_msg_id, is_topic_boundary) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            archived_id,
                            str(_row_value(summary, "summary_text") or ""),
                            _row_value(summary, "summary_embedding"),
                            int(_row_value(summary, "chunk_index") or 0),
                            message_id_map.get(_row_value(summary, "first_msg_id")),
                            message_id_map.get(_row_value(summary, "last_msg_id")),
                            int(_row_value(summary, "is_topic_boundary") or 0),
                        ),
                    )
                    migrated_summaries += 1

    conn.execute(
        "INSERT INTO legacy_archive_migrations "
        "(source_path, source_size, source_mtime_ns, migrated_at, "
        " session_count, message_count, summary_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            migration_key,
            stat.st_size,
            stat.st_mtime_ns,
            _now(),
            migrated_sessions,
            migrated_messages,
            migrated_summaries,
        ),
    )
    return {
        "migrated": True,
        "session_count": migrated_sessions,
        "message_count": migrated_messages,
        "summary_count": migrated_summaries,
        "skipped_sessions": skipped_sessions,
    }


def _archive_candidates(
    conn: sqlite3.Connection,
    batch_id: str,
    source: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT conversation_id, source_id, title, created_at, updated_at, model "
        "FROM import_conversations "
        "WHERE batch_id = ? AND choice = 'keep' ORDER BY position",
        (batch_id,),
    ).fetchall()
    candidates = []
    for row in rows:
        conversation_id = str(row[0])
        source_conversation_id = str(row[1] or conversation_id)
        existing = conn.execute(
            "SELECT id FROM legacy_sessions "
            "WHERE source = ? AND source_conversation_id = ?",
            (source, source_conversation_id),
        ).fetchone()
        message_rows = conn.execute(
            "SELECT source_message_id, role, source_role, content, created_at, "
            "model, attachment_placeholder "
            "FROM import_messages "
            "WHERE batch_id = ? AND conversation_id = ? "
            "AND excluded = 0 AND skipped = 0 "
            "AND role IN ('user', 'assistant') AND TRIM(content) <> '' "
            "ORDER BY position",
            (batch_id, conversation_id),
        ).fetchall()
        candidates.append({
            "conversation_id": conversation_id,
            "source": source,
            "source_conversation_id": source_conversation_id,
            "title": str(row[2] or ""),
            "created_at": str(row[3] or ""),
            "updated_at": str(row[4] or ""),
            "model": str(row[5] or ""),
            "already_archived": bool(existing),
            "messages": [
                {
                    "source_message_id": str(message[0] or ""),
                    "role": str(message[1] or ""),
                    "source_role": str(message[2] or ""),
                    "content": str(message[3] or ""),
                    "timestamp": str(message[4] or ""),
                    "model": str(message[5] or ""),
                    "attachment_placeholder": bool(message[6]),
                }
                for message in message_rows
            ],
        })
    return candidates


def _ensure_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _backfill_message_positions(conn: sqlite3.Connection) -> None:
    duplicates = conn.execute(
        "SELECT session_id FROM legacy_messages "
        "GROUP BY session_id HAVING COUNT(DISTINCT position) <> COUNT(*) "
        "OR MIN(position) < 0"
    ).fetchall()
    for row in duplicates:
        message_rows = conn.execute(
            "SELECT id FROM legacy_messages WHERE session_id = ? ORDER BY timestamp, id",
            (row[0],),
        ).fetchall()
        conn.executemany(
            "UPDATE legacy_messages SET position = ? WHERE id = ?",
            [(position, message[0]) for position, message in enumerate(message_rows)],
        )


def _ensure_legacy_fts(conn: sqlite3.Connection) -> None:
    try:
        had_insert_trigger = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'trigger' AND name = 'legacy_messages_ai'"
        ).fetchone() is not None
        conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS legacy_messages_fts
        USING fts5(content, content='legacy_messages', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS legacy_messages_ai AFTER INSERT ON legacy_messages BEGIN
          INSERT INTO legacy_messages_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS legacy_messages_ad AFTER DELETE ON legacy_messages BEGIN
          INSERT INTO legacy_messages_fts(legacy_messages_fts, rowid, content)
          VALUES ('delete', old.id, old.content);
        END;
        CREATE TRIGGER IF NOT EXISTS legacy_messages_au AFTER UPDATE ON legacy_messages BEGIN
          INSERT INTO legacy_messages_fts(legacy_messages_fts, rowid, content)
          VALUES ('delete', old.id, old.content);
          INSERT INTO legacy_messages_fts(rowid, content) VALUES (new.id, new.content);
        END;
        """)
        if not had_insert_trigger and conn.execute(
            "SELECT 1 FROM legacy_messages LIMIT 1"
        ).fetchone():
            conn.execute(
                "INSERT INTO legacy_messages_fts(legacy_messages_fts) VALUES ('rebuild')"
            )
    except sqlite3.OperationalError:
        # Keyword search still has its LIKE fallback on SQLite builds without FTS5.
        pass


def _archive_id(source: str, source_conversation_id: str) -> str:
    digest = hashlib.sha256(
        f"{source}\0{source_conversation_id}".encode("utf-8")
    ).hexdigest()
    return f"legacy_{digest[:24]}"


def _path_digest(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _row_value(row: sqlite3.Row, name: str) -> Any:
    return row[name] if name in row.keys() else None


def _connection_path(conn: sqlite3.Connection) -> Path | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    if not row or not row[2]:
        return None
    return Path(row[2]).resolve()
