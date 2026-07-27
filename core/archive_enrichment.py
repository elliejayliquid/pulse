"""Durable planning state for background enrichment of imported history.

This module deliberately does not run embedding or summary models.  It creates
deterministic, exchange-preserving chunk plans and persists the job lifecycle
that later workers will claim one chunk at a time.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


ENRICHMENT_SCHEMA_VERSION = 1
PLANNER_VERSION = "exchange-chunks-v1"
DEFAULT_CHUNK_TOKEN_BUDGET = 1600
RECOVERY_STALE_SECONDS = 300

_ENRICHMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS legacy_enrichment_jobs (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL DEFAULT 'archive',
    session_id TEXT NOT NULL DEFAULT '',
    tier INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    schedule_mode TEXT NOT NULL DEFAULT 'paused',
    planner_version TEXT NOT NULL,
    archive_fingerprint TEXT NOT NULL,
    chunk_token_budget INTEGER NOT NULL,
    total_sessions INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    total_messages INTEGER NOT NULL DEFAULT 0,
    estimated_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT '',
    completed_at TEXT NOT NULL DEFAULT '',
    heartbeat_at TEXT NOT NULL DEFAULT '',
    current_chunk_id TEXT NOT NULL DEFAULT '',
    pause_requested INTEGER NOT NULL DEFAULT 0,
    recovery_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_legacy_enrichment_jobs_status
ON legacy_enrichment_jobs(status, updated_at);

CREATE TABLE IF NOT EXISTS legacy_enrichment_chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    planner_version TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    first_message_id INTEGER NOT NULL,
    last_message_id INTEGER NOT NULL,
    first_position INTEGER NOT NULL,
    last_position INTEGER NOT NULL,
    first_timestamp TEXT NOT NULL DEFAULT '',
    last_timestamp TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL,
    exchange_count INTEGER NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    content_fingerprint TEXT NOT NULL,
    oversized_exchange INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, planner_version, chunk_index, content_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_legacy_enrichment_chunks_session
ON legacy_enrichment_chunks(session_id, chunk_index);

CREATE TABLE IF NOT EXISTS legacy_enrichment_job_chunks (
    job_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    claimed_at TEXT NOT NULL DEFAULT '',
    completed_at TEXT NOT NULL DEFAULT '',
    last_error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(job_id, chunk_id),
    FOREIGN KEY(job_id) REFERENCES legacy_enrichment_jobs(id) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id) REFERENCES legacy_enrichment_chunks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_legacy_enrichment_job_chunks_status
ON legacy_enrichment_job_chunks(job_id, status, ordinal);

CREATE TABLE IF NOT EXISTS legacy_enrichment_artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL DEFAULT '',
    artifact_type TEXT NOT NULL,
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    first_message_id INTEGER,
    last_message_id INTEGER,
    content_fingerprint TEXT NOT NULL,
    generation_fingerprint TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    artifact_text TEXT NOT NULL DEFAULT '',
    artifact_embedding BLOB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(artifact_type, content_fingerprint, generation_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_legacy_enrichment_artifacts_session
ON legacy_enrichment_artifacts(session_id, artifact_type, hierarchy_level);

CREATE TABLE IF NOT EXISTS legacy_enrichment_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def ensure_enrichment_tables(conn: sqlite3.Connection) -> None:
    """Create the durable enrichment planner/job schema."""
    conn.executescript(_ENRICHMENT_SCHEMA)
    conn.execute(
        "INSERT INTO legacy_enrichment_meta(key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(ENRICHMENT_SCHEMA_VERSION),),
    )


def plan_archive_enrichment(
    conn: sqlite3.Connection,
    *,
    schedule_mode: str = "paused",
    chunk_token_budget: int = DEFAULT_CHUNK_TOKEN_BUDGET,
    session_id: str = "",
) -> dict[str, Any]:
    """Persist an idempotent chunk plan without running enrichment models."""
    ensure_enrichment_tables(conn)
    schedule_mode = str(schedule_mode or "paused").strip().lower()
    if schedule_mode not in {"paused", "idle"}:
        raise ValueError("Enrichment schedule must be 'paused' or 'idle'.")
    chunk_token_budget = max(128, int(chunk_token_budget or DEFAULT_CHUNK_TOKEN_BUDGET))
    session_id = str(session_id or "")

    where = "WHERE id = ?" if session_id else ""
    params: tuple[Any, ...] = (session_id,) if session_id else ()
    sessions = conn.execute(
        "SELECT id, title, source, source_conversation_id "
        f"FROM legacy_sessions {where} ORDER BY created_at, id",
        params,
    ).fetchall()
    if not sessions:
        raise ValueError("Add at least one chat to searchable history before planning enrichment.")

    planned_chunks: list[dict[str, Any]] = []
    session_fingerprints: list[str] = []
    total_messages = 0
    total_tokens = 0
    now = _now()
    for session in sessions:
        archive_id = str(session[0])
        rows = conn.execute(
            "SELECT id, position, role, content, timestamp "
            "FROM legacy_messages WHERE session_id = ? "
            "AND role IN ('user', 'assistant') ORDER BY position, id",
            (archive_id,),
        ).fetchall()
        chunks = _plan_session_chunks(
            archive_id,
            rows,
            chunk_token_budget=chunk_token_budget,
            created_at=now,
        )
        if not chunks:
            continue
        planned_chunks.extend(chunks)
        total_messages += sum(int(chunk["message_count"]) for chunk in chunks)
        total_tokens += sum(int(chunk["estimated_tokens"]) for chunk in chunks)
        session_fingerprints.append(
            _digest(
                archive_id,
                str(session[2] or ""),
                str(session[3] or ""),
                *(str(chunk["content_fingerprint"]) for chunk in chunks),
            )
        )

    if not planned_chunks:
        raise ValueError("Searchable history has no usable dialogue to enrich.")

    archive_fingerprint = _digest(
        PLANNER_VERSION,
        str(chunk_token_budget),
        *(session_fingerprints),
    )
    scope = "session" if session_id else "archive"
    job_id = f"enrich_{_digest(scope, session_id, archive_fingerprint)[:24]}"
    initial_status = "paused" if schedule_mode == "paused" else "queued"
    inserted = conn.execute(
        "INSERT OR IGNORE INTO legacy_enrichment_jobs("
        "id, scope, session_id, tier, status, schedule_mode, planner_version, "
        "archive_fingerprint, chunk_token_budget, total_sessions, total_chunks, "
        "total_messages, estimated_tokens, created_at, updated_at"
        ") VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            job_id,
            scope,
            session_id,
            initial_status,
            schedule_mode,
            PLANNER_VERSION,
            archive_fingerprint,
            chunk_token_budget,
            len(session_fingerprints),
            len(planned_chunks),
            total_messages,
            total_tokens,
            now,
            now,
        ),
    ).rowcount

    for ordinal, chunk in enumerate(planned_chunks):
        conn.execute(
            "INSERT INTO legacy_enrichment_chunks("
            "id, session_id, planner_version, chunk_index, first_message_id, "
            "last_message_id, first_position, last_position, first_timestamp, "
            "last_timestamp, message_count, exchange_count, estimated_tokens, "
            "content_fingerprint, oversized_exchange, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO NOTHING",
            (
                chunk["id"],
                chunk["session_id"],
                PLANNER_VERSION,
                chunk["chunk_index"],
                chunk["first_message_id"],
                chunk["last_message_id"],
                chunk["first_position"],
                chunk["last_position"],
                chunk["first_timestamp"],
                chunk["last_timestamp"],
                chunk["message_count"],
                chunk["exchange_count"],
                chunk["estimated_tokens"],
                chunk["content_fingerprint"],
                chunk["oversized_exchange"],
                now,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO legacy_enrichment_job_chunks("
            "job_id, chunk_id, ordinal, status"
            ") VALUES (?, ?, ?, 'pending')",
            (job_id, chunk["id"], ordinal),
        )

    result = enrichment_job_status(conn, job_id)
    result["created"] = bool(inserted)
    result["idempotent_reuse"] = not bool(inserted)
    return result


def latest_enrichment_job(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Return the latest durable job, if any."""
    ensure_enrichment_tables(conn)
    row = conn.execute(
        "SELECT id FROM legacy_enrichment_jobs ORDER BY created_at DESC, id DESC LIMIT 1"
    ).fetchone()
    return enrichment_job_status(conn, str(row[0])) if row else None


def enrichment_job_status(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    """Return one job with counts derived from its durable chunk rows."""
    row = conn.execute(
        "SELECT id, scope, session_id, tier, status, schedule_mode, planner_version, "
        "archive_fingerprint, chunk_token_budget, total_sessions, total_chunks, "
        "total_messages, estimated_tokens, created_at, updated_at, started_at, "
        "completed_at, heartbeat_at, current_chunk_id, pause_requested, "
        "recovery_count, last_error "
        "FROM legacy_enrichment_jobs WHERE id = ?",
        (str(job_id or ""),),
    ).fetchone()
    if not row:
        raise ValueError("Enrichment job not found.")
    counts = {
        str(status): int(count)
        for status, count in conn.execute(
            "SELECT status, COUNT(*) FROM legacy_enrichment_job_chunks "
            "WHERE job_id = ? GROUP BY status",
            (str(row[0]),),
        ).fetchall()
    }
    total_chunks = int(row[10] or 0)
    completed = counts.get("completed", 0)
    if str(row[1]) == "session":
        current_counts = conn.execute(
            "SELECT COUNT(DISTINCT s.id), COUNT(m.id) "
            "FROM legacy_sessions s LEFT JOIN legacy_messages m ON m.session_id = s.id "
            "WHERE s.id = ?",
            (str(row[2]),),
        ).fetchone()
    else:
        current_counts = conn.execute(
            "SELECT COUNT(DISTINCT s.id), COUNT(m.id) "
            "FROM legacy_sessions s LEFT JOIN legacy_messages m ON m.session_id = s.id"
        ).fetchone()
    archive_changed = (
        int(current_counts[0] or 0) != int(row[9] or 0)
        or int(current_counts[1] or 0) != int(row[11] or 0)
    )
    return {
        "id": str(row[0]),
        "scope": str(row[1]),
        "session_id": str(row[2]),
        "tier": int(row[3]),
        "status": str(row[4]),
        "schedule_mode": str(row[5]),
        "planner_version": str(row[6]),
        "archive_fingerprint": str(row[7]),
        "chunk_token_budget": int(row[8]),
        "total_sessions": int(row[9]),
        "total_chunks": total_chunks,
        "total_messages": int(row[11]),
        "estimated_tokens": int(row[12]),
        "created_at": str(row[13]),
        "updated_at": str(row[14]),
        "started_at": str(row[15]),
        "completed_at": str(row[16]),
        "heartbeat_at": str(row[17]),
        "current_chunk_id": str(row[18]),
        "pause_requested": bool(row[19]),
        "recovery_count": int(row[20]),
        "last_error": str(row[21]),
        "chunk_counts": counts,
        "completed_chunks": completed,
        "progress_percent": round((completed / total_chunks) * 100, 1) if total_chunks else 0.0,
        "archive_changed": archive_changed,
        "worker_available": False,
    }


def pause_enrichment_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    """Pause queued work or cooperatively request a running worker to pause."""
    current = enrichment_job_status(conn, job_id)
    status = current["status"]
    now = _now()
    if status == "queued":
        conn.execute(
            "UPDATE legacy_enrichment_jobs SET status = 'paused', schedule_mode = 'paused', "
            "pause_requested = 0, updated_at = ? WHERE id = ?",
            (now, job_id),
        )
    elif status == "running":
        conn.execute(
            "UPDATE legacy_enrichment_jobs SET status = 'pausing', pause_requested = 1, "
            "updated_at = ? WHERE id = ?",
            (now, job_id),
        )
    elif status not in {"paused", "pausing", "completed"}:
        raise ValueError(f"Cannot pause an enrichment job in '{status}' state.")
    return enrichment_job_status(conn, job_id)


def resume_enrichment_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    """Queue paused or failed work without repeating completed chunks."""
    current = enrichment_job_status(conn, job_id)
    status = current["status"]
    if current["archive_changed"]:
        raise ValueError("Searchable history changed. Refresh the enrichment plan before queuing it.")
    if status not in {"paused", "failed"}:
        if status == "queued":
            return current
        raise ValueError(f"Cannot resume an enrichment job in '{status}' state.")
    conn.execute(
        "UPDATE legacy_enrichment_jobs SET status = 'queued', schedule_mode = 'idle', "
        "pause_requested = 0, completed_at = '', last_error = '', updated_at = ? "
        "WHERE id = ?",
        (_now(), job_id),
    )
    return enrichment_job_status(conn, job_id)


def recover_interrupted_enrichment_jobs(
    conn: sqlite3.Connection,
    *,
    stale_seconds: int = RECOVERY_STALE_SECONDS,
    now: datetime | None = None,
) -> list[str]:
    """Return stale running work to a safe durable state after a restart."""
    ensure_enrichment_tables(conn)
    now_dt = now or datetime.now(timezone.utc)
    cutoff = (now_dt - timedelta(seconds=max(1, int(stale_seconds)))).isoformat()
    rows = conn.execute(
        "SELECT id, status FROM legacy_enrichment_jobs "
        "WHERE status IN ('running', 'pausing') "
        "AND (heartbeat_at = '' OR heartbeat_at < ?)",
        (cutoff,),
    ).fetchall()
    recovered: list[str] = []
    for job_id, status in rows:
        job_id = str(job_id)
        conn.execute(
            "UPDATE legacy_enrichment_job_chunks SET status = 'pending', claimed_at = '', "
            "last_error = 'Recovered after an interrupted worker.' "
            "WHERE job_id = ? AND status = 'running'",
            (job_id,),
        )
        next_status = "paused" if str(status) == "pausing" else "queued"
        conn.execute(
            "UPDATE legacy_enrichment_jobs SET status = ?, current_chunk_id = '', "
            "pause_requested = 0, heartbeat_at = '', recovery_count = recovery_count + 1, "
            "last_error = 'Recovered safely after an interrupted worker.', updated_at = ? "
            "WHERE id = ?",
            (next_status, now_dt.isoformat(), job_id),
        )
        recovered.append(job_id)
    return recovered


def _plan_session_chunks(
    session_id: str,
    rows: list[sqlite3.Row | tuple],
    *,
    chunk_token_budget: int,
    created_at: str,
) -> list[dict[str, Any]]:
    usable = [row for row in rows if str(row[3] or "").strip()]
    if not usable:
        return []
    exchanges: list[list[sqlite3.Row | tuple]] = []
    current: list[sqlite3.Row | tuple] = []
    previous_role = ""
    for row in usable:
        role = str(row[2] or "")
        if current and role == "user" and previous_role != "user":
            exchanges.append(current)
            current = []
        current.append(row)
        previous_role = role
    if current:
        exchanges.append(current)

    groups: list[list[list[sqlite3.Row | tuple]]] = []
    current_group: list[list[sqlite3.Row | tuple]] = []
    current_tokens = 0
    for exchange in exchanges:
        exchange_tokens = _rows_token_estimate(exchange)
        if current_group and current_tokens + exchange_tokens > chunk_token_budget:
            groups.append(current_group)
            current_group = []
            current_tokens = 0
        current_group.append(exchange)
        current_tokens += exchange_tokens
    if current_group:
        groups.append(current_group)

    chunks: list[dict[str, Any]] = []
    for chunk_index, group in enumerate(groups):
        flattened = [row for exchange in group for row in exchange]
        fingerprint = _message_fingerprint(flattened)
        first = flattened[0]
        last = flattened[-1]
        chunk_id = f"chunk_{_digest(PLANNER_VERSION, session_id, str(chunk_index), fingerprint)[:28]}"
        estimated_tokens = _rows_token_estimate(flattened)
        chunks.append({
            "id": chunk_id,
            "session_id": session_id,
            "chunk_index": chunk_index,
            "first_message_id": int(first[0]),
            "last_message_id": int(last[0]),
            "first_position": int(first[1]),
            "last_position": int(last[1]),
            "first_timestamp": str(first[4] or ""),
            "last_timestamp": str(last[4] or ""),
            "message_count": len(flattened),
            "exchange_count": len(group),
            "estimated_tokens": estimated_tokens,
            "content_fingerprint": fingerprint,
            "oversized_exchange": int(len(group) == 1 and estimated_tokens > chunk_token_budget),
            "created_at": created_at,
        })
    return chunks


def _rows_token_estimate(rows: list[sqlite3.Row | tuple]) -> int:
    characters = sum(len(str(row[3] or "")) for row in rows)
    return max(1, characters // 4) if rows else 0


def _message_fingerprint(rows: list[sqlite3.Row | tuple]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for value in (row[0], row[1], row[2], row[3], row[4]):
            digest.update(str(value or "").encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
