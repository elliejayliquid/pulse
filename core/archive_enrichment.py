"""Durable planning and one-chunk execution for imported-history enrichment.

The planner creates deterministic, exchange-preserving chunks.  The Tier 1
runner claims exactly one durable chunk, embeds it locally, and records the
artifact.  Scheduling and Tier 2 summary generation remain separate concerns.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from core.embeddings import chunk_text, embedding_to_blob


ENRICHMENT_SCHEMA_VERSION = 2
PLANNER_VERSION = "exchange-chunks-v4-token-safe-segments"
DEFAULT_CHUNK_TOKEN_BUDGET = 220
MAX_TIER1_CHUNK_TOKEN_BUDGET = 220
RECOVERY_STALE_SECONDS = 300
TIER1_ARTIFACT_TYPE = "raw_chunk_embedding"
TIER1_SERIALIZATION_VERSION = "dialogue-segments-v4-token-safe"
DEFAULT_TIER1_MODEL = "all-MiniLM-L6-v2"
TIER1_MODEL_TOKEN_LIMIT = 240

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
    segment_index INTEGER NOT NULL DEFAULT 0,
    segment_count INTEGER NOT NULL DEFAULT 1,
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
    artifact_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(legacy_enrichment_artifacts)").fetchall()
    }
    if "segment_index" not in artifact_columns:
        conn.execute(
            "ALTER TABLE legacy_enrichment_artifacts "
            "ADD COLUMN segment_index INTEGER NOT NULL DEFAULT 0"
        )
    if "segment_count" not in artifact_columns:
        conn.execute(
            "ALTER TABLE legacy_enrichment_artifacts "
            "ADD COLUMN segment_count INTEGER NOT NULL DEFAULT 1"
        )
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
    chunk_token_budget = max(
        128,
        min(MAX_TIER1_CHUNK_TOKEN_BUDGET, int(chunk_token_budget or DEFAULT_CHUNK_TOKEN_BUDGET)),
    )
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
        "worker_available": True,
    }


def claim_next_enrichment_chunk(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Atomically claim the next pending chunk from one queued Tier 1 job.

    The caller must commit the claim before doing model work so another process
    can observe the heartbeat and recover it after an interruption.
    """
    ensure_enrichment_tables(conn)
    job_id = str(job_id or "")
    claimed_at = (now or datetime.now(timezone.utc)).isoformat()
    candidate = conn.execute(
        "SELECT jc.chunk_id FROM legacy_enrichment_job_chunks jc "
        "JOIN legacy_enrichment_jobs j ON j.id = jc.job_id "
        "WHERE jc.job_id = ? AND jc.status = 'pending' "
        "AND j.tier = 1 AND j.status = 'queued' AND j.pause_requested = 0 "
        "AND j.current_chunk_id = '' ORDER BY jc.ordinal LIMIT 1",
        (job_id,),
    ).fetchone()
    if not candidate:
        return None

    chunk_id = str(candidate[0])
    claimed = conn.execute(
        "UPDATE legacy_enrichment_job_chunks SET status = 'running', "
        "attempts = attempts + 1, claimed_at = ?, completed_at = '', last_error = '' "
        "WHERE job_id = ? AND chunk_id = ? AND status = 'pending' "
        "AND EXISTS (SELECT 1 FROM legacy_enrichment_jobs j WHERE j.id = ? "
        "AND j.tier = 1 AND j.status = 'queued' AND j.pause_requested = 0 "
        "AND j.current_chunk_id = '')",
        (claimed_at, job_id, chunk_id, job_id),
    ).rowcount
    if not claimed:
        return None

    started = conn.execute(
        "UPDATE legacy_enrichment_jobs SET status = 'running', current_chunk_id = ?, "
        "heartbeat_at = ?, started_at = CASE WHEN started_at = '' THEN ? ELSE started_at END, "
        "updated_at = ?, last_error = '' WHERE id = ? AND status = 'queued' "
        "AND pause_requested = 0 AND current_chunk_id = ''",
        (chunk_id, claimed_at, claimed_at, claimed_at, job_id),
    ).rowcount
    if not started:
        conn.execute(
            "UPDATE legacy_enrichment_job_chunks SET status = 'pending', claimed_at = '' "
            "WHERE job_id = ? AND chunk_id = ? AND status = 'running'",
            (job_id, chunk_id),
        )
        return None

    row = conn.execute(
        "SELECT c.id, c.session_id, c.chunk_index, c.first_message_id, "
        "c.last_message_id, c.first_position, c.last_position, c.message_count, "
        "c.exchange_count, c.estimated_tokens, c.content_fingerprint, jc.attempts "
        "FROM legacy_enrichment_chunks c "
        "JOIN legacy_enrichment_job_chunks jc ON jc.chunk_id = c.id "
        "WHERE jc.job_id = ? AND c.id = ?",
        (job_id, chunk_id),
    ).fetchone()
    if not row:
        raise RuntimeError("Claimed enrichment chunk disappeared before it could be loaded.")
    return {
        "job_id": job_id,
        "chunk_id": str(row[0]),
        "session_id": str(row[1]),
        "chunk_index": int(row[2]),
        "first_message_id": int(row[3]),
        "last_message_id": int(row[4]),
        "first_position": int(row[5]),
        "last_position": int(row[6]),
        "message_count": int(row[7]),
        "exchange_count": int(row[8]),
        "estimated_tokens": int(row[9]),
        "content_fingerprint": str(row[10]),
        "attempts": int(row[11]),
        "embedding_text": _chunk_embedding_text(conn, str(row[1]), int(row[5]), int(row[6])),
    }


def heartbeat_enrichment_chunk(
    conn: sqlite3.Connection,
    job_id: str,
    chunk_id: str,
    *,
    now: datetime | None = None,
) -> bool:
    """Refresh the heartbeat for the currently claimed chunk."""
    heartbeat_at = (now or datetime.now(timezone.utc)).isoformat()
    updated = conn.execute(
        "UPDATE legacy_enrichment_jobs SET heartbeat_at = ?, updated_at = ? "
        "WHERE id = ? AND current_chunk_id = ? AND status IN ('running', 'pausing') "
        "AND EXISTS (SELECT 1 FROM legacy_enrichment_job_chunks jc "
        "WHERE jc.job_id = ? AND jc.chunk_id = ? AND jc.status = 'running')",
        (heartbeat_at, heartbeat_at, str(job_id or ""), str(chunk_id or ""),
         str(job_id or ""), str(chunk_id or "")),
    ).rowcount
    return bool(updated)


def complete_enrichment_chunk(
    conn: sqlite3.Connection,
    job_id: str,
    chunk_id: str,
    embedding: bytes,
    *,
    model_name: str = DEFAULT_TIER1_MODEL,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for callers that produced one chunk embedding."""
    return complete_enrichment_chunk_segments(
        conn,
        job_id,
        chunk_id,
        [("", embedding)],
        model_name=model_name,
        now=now,
    )


def complete_enrichment_chunk_segments(
    conn: sqlite3.Connection,
    job_id: str,
    chunk_id: str,
    segments: list[tuple[str, bytes]],
    *,
    model_name: str = DEFAULT_TIER1_MODEL,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Persist every embedding segment and advance one parent chunk."""
    job_id = str(job_id or "")
    chunk_id = str(chunk_id or "")
    if not segments or any(not embedding for _, embedding in segments):
        raise ValueError("Tier 1 enrichment produced an empty embedding segment.")
    finished_at = (now or datetime.now(timezone.utc)).isoformat()
    row = conn.execute(
        "SELECT c.session_id, c.first_message_id, c.last_message_id, "
        "c.content_fingerprint, jc.status FROM legacy_enrichment_chunks c "
        "JOIN legacy_enrichment_job_chunks jc ON jc.chunk_id = c.id "
        "WHERE jc.job_id = ? AND c.id = ?",
        (job_id, chunk_id),
    ).fetchone()
    if not row:
        raise ValueError("Enrichment chunk not found.")
    if str(row[4]) == "completed":
        return enrichment_job_status(conn, job_id)
    if str(row[4]) != "running":
        raise ValueError("Enrichment chunk is not currently claimed.")

    parent_fingerprint = str(row[3])
    generation_fingerprint = _digest(
        TIER1_ARTIFACT_TYPE,
        TIER1_SERIALIZATION_VERSION,
        str(model_name or DEFAULT_TIER1_MODEL),
    )
    conn.execute(
        "DELETE FROM legacy_enrichment_artifacts WHERE session_id = ? "
        "AND artifact_type = ? AND first_message_id = ? AND last_message_id = ?",
        (str(row[0]), TIER1_ARTIFACT_TYPE, int(row[1]), int(row[2])),
    )
    segment_count = len(segments)
    for segment_index, (segment_text, embedding) in enumerate(segments):
        content_fingerprint = _digest(
            parent_fingerprint,
            TIER1_SERIALIZATION_VERSION,
            str(segment_index),
            segment_text,
        )
        artifact_id = (
            f"artifact_{_digest(TIER1_ARTIFACT_TYPE, content_fingerprint, generation_fingerprint)[:28]}"
        )
        conn.execute(
            "INSERT INTO legacy_enrichment_artifacts("
            "id, session_id, chunk_id, artifact_type, hierarchy_level, segment_index, "
            "segment_count, first_message_id, last_message_id, content_fingerprint, "
            "generation_fingerprint, provider, model, status, artifact_text, "
            "artifact_embedding, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, 'local', ?, 'completed', '', ?, ?, ?) "
            "ON CONFLICT(artifact_type, content_fingerprint, generation_fingerprint) "
            "DO UPDATE SET chunk_id = excluded.chunk_id, segment_index = excluded.segment_index, "
            "segment_count = excluded.segment_count, status = 'completed', "
            "artifact_embedding = excluded.artifact_embedding, updated_at = excluded.updated_at",
            (
                artifact_id,
                str(row[0]),
                chunk_id,
                TIER1_ARTIFACT_TYPE,
                segment_index,
                segment_count,
                int(row[1]),
                int(row[2]),
                content_fingerprint,
                generation_fingerprint,
                str(model_name or DEFAULT_TIER1_MODEL),
                embedding,
                finished_at,
                finished_at,
            ),
        )
    conn.execute(
        "UPDATE legacy_enrichment_job_chunks SET status = 'completed', "
        "completed_at = ?, last_error = '' WHERE job_id = ? AND chunk_id = ? "
        "AND status = 'running'",
        (finished_at, job_id, chunk_id),
    )
    pending = int(conn.execute(
        "SELECT COUNT(*) FROM legacy_enrichment_job_chunks "
        "WHERE job_id = ? AND status = 'pending'",
        (job_id,),
    ).fetchone()[0])
    job_row = conn.execute(
        "SELECT status, pause_requested FROM legacy_enrichment_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if not job_row:
        raise ValueError("Enrichment job not found.")
    pause_requested = bool(job_row[1]) or str(job_row[0]) == "pausing"
    next_status = "paused" if pause_requested else ("queued" if pending else "completed")
    schedule_mode = "paused" if pause_requested else "idle"
    completed_at = finished_at if next_status == "completed" else ""
    conn.execute(
        "UPDATE legacy_enrichment_jobs SET status = ?, schedule_mode = ?, "
        "current_chunk_id = '', pause_requested = 0, heartbeat_at = '', "
        "completed_at = ?, updated_at = ?, last_error = '' WHERE id = ?",
        (next_status, schedule_mode, completed_at, finished_at, job_id),
    )
    return enrichment_job_status(conn, job_id)


def fail_enrichment_chunk(
    conn: sqlite3.Connection,
    job_id: str,
    chunk_id: str,
    error: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record one failed attempt without discarding its retry history."""
    failed_at = (now or datetime.now(timezone.utc)).isoformat()
    message = str(error or "Tier 1 enrichment failed.")[:2000]
    conn.execute(
        "UPDATE legacy_enrichment_job_chunks SET status = 'failed', last_error = ? "
        "WHERE job_id = ? AND chunk_id = ? AND status = 'running'",
        (message, str(job_id or ""), str(chunk_id or "")),
    )
    conn.execute(
        "UPDATE legacy_enrichment_jobs SET status = 'failed', current_chunk_id = '', "
        "heartbeat_at = '', pause_requested = 0, updated_at = ?, last_error = ? "
        "WHERE id = ?",
        (failed_at, message, str(job_id or "")),
    )
    return enrichment_job_status(conn, str(job_id or ""))


def run_one_tier1_enrichment_chunk(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    encoder: Callable[[str], Any] | None = None,
    model_name: str = DEFAULT_TIER1_MODEL,
) -> dict[str, Any]:
    """Claim and locally embed at most one chunk.

    ``encoder`` is injectable for focused tests.  The default reuses Pulse's
    process-wide MiniLM instance and performs no network or LLM request.
    """
    claim = claim_next_enrichment_chunk(conn, job_id)
    if not claim:
        conn.commit()
        return {
            "claimed": False,
            "completed": False,
            "job": enrichment_job_status(conn, job_id),
        }
    conn.commit()
    try:
        if encoder is None:
            from core.context import _get_embedding_model

            model = _get_embedding_model()
            if model is None:
                raise RuntimeError("Local embedding model is not loaded.")
            segment_texts = _tier1_embedding_segments(
                claim["embedding_text"],
                model=model,
            )
            vectors = model.encode(segment_texts, normalize_embeddings=True)
        else:
            segment_texts = _tier1_embedding_segments(claim["embedding_text"])
            vectors = [encoder(segment_text) for segment_text in segment_texts]
        segment_embeddings = [
            (segment_text, embedding_to_blob(vector))
            for segment_text, vector in zip(segment_texts, vectors)
        ]
        if len(segment_embeddings) != len(segment_texts) or any(
            not embedding for _, embedding in segment_embeddings
        ):
            raise RuntimeError("Local embedding model returned an incomplete segment vector set.")
        job = complete_enrichment_chunk_segments(
            conn,
            job_id,
            claim["chunk_id"],
            segment_embeddings,
            model_name=model_name,
        )
        conn.commit()
        return {
            "claimed": True,
            "completed": True,
            "chunk": claim,
            "segment_count": len(segment_embeddings),
            "job": job,
        }
    except Exception as exc:
        conn.rollback()
        job = fail_enrichment_chunk(conn, job_id, claim["chunk_id"], str(exc))
        conn.commit()
        return {
            "claimed": True,
            "completed": False,
            "chunk": claim,
            "job": job,
            "error": str(exc),
        }


def run_one_idle_enrichment(
    db_path: str | Path,
    *,
    encoder: Callable[[str], Any] | None = None,
    model_name: str = DEFAULT_TIER1_MODEL,
) -> dict[str, Any]:
    """Recover stale state and consume at most one eligible idle chunk.

    A dedicated connection keeps background work off the engine's conversation
    connection and gives each scheduler tick a bounded database lifetime.
    """
    conn = sqlite3.connect(str(Path(db_path)), timeout=10)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_enrichment_tables(conn)
        recovered = recover_interrupted_enrichment_jobs(conn)
        job = latest_enrichment_job(conn)
        conn.commit()
        if not job:
            return {"claimed": False, "reason": "no_job", "job": None,
                    "recovered_jobs": recovered}
        if job["archive_changed"]:
            return {"claimed": False, "reason": "archive_changed", "job": job,
                    "recovered_jobs": recovered}
        if job["status"] != "queued" or job["schedule_mode"] != "idle":
            return {"claimed": False, "reason": "not_eligible", "job": job,
                    "recovered_jobs": recovered}
        result = run_one_tier1_enrichment_chunk(
            conn,
            job["id"],
            encoder=encoder,
            model_name=model_name,
        )
        result["reason"] = "processed" if result["claimed"] else "not_eligible"
        result["recovered_jobs"] = recovered
        return result
    finally:
        conn.close()


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
    if status == "failed":
        conn.execute(
            "UPDATE legacy_enrichment_job_chunks SET status = 'pending', claimed_at = '' "
            "WHERE job_id = ? AND status = 'failed'",
            (job_id,),
        )
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


def _chunk_embedding_text(
    conn: sqlite3.Connection,
    session_id: str,
    first_position: int,
    last_position: int,
) -> str:
    rows = conn.execute(
        "SELECT role, content FROM legacy_messages WHERE session_id = ? "
        "AND position BETWEEN ? AND ? AND role IN ('user', 'assistant') "
        "ORDER BY position, id",
        (session_id, first_position, last_position),
    ).fetchall()
    parts = [
        f"[{str(role)}]\n{str(content).strip()}"
        for role, content in rows
        if str(content or "").strip()
    ]
    if not parts:
        raise ValueError("Enrichment chunk has no usable dialogue.")
    return "\n\n".join(parts)


def _tier1_embedding_segments(text: str, *, model: Any | None = None) -> list[str]:
    """Split serialized dialogue without omitting text or exceeding MiniLM.

    Tests and injected encoders use the same complete character segmentation as
    passive recall, but without its recall-time count/minimum filters.  The real
    model path additionally measures every candidate with MiniLM's own tokenizer
    so dense Unicode cannot invalidate the character heuristic.
    """
    text = str(text or "")
    if not text:
        return []
    if model is None:
        return chunk_text(text, max_chars=600, max_chunks=0, min_chars=1) or [text]

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        raise RuntimeError("Local embedding model does not expose its tokenizer.")
    configured_limit = int(getattr(model, "max_seq_length", 256) or 256)
    token_limit = max(16, min(TIER1_MODEL_TOKEN_LIMIT, configured_limit))

    def token_count(candidate: str) -> int:
        try:
            token_ids = tokenizer.encode(
                candidate,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        except TypeError:
            # Small tokenizer doubles used by tests may not expose the
            # Hugging Face ``verbose`` switch.
            token_ids = tokenizer.encode(
                candidate,
                add_special_tokens=True,
                truncation=False,
            )
        return len(token_ids)

    segments: list[str] = []
    cursor = 0
    while cursor < len(text):
        remaining = text[cursor:]
        if token_count(remaining) <= token_limit:
            segments.append(remaining)
            break

        low, high, best = 1, len(remaining), 0
        while low <= high:
            midpoint = (low + high) // 2
            if token_count(remaining[:midpoint]) <= token_limit:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best <= 0:
            raise RuntimeError("MiniLM tokenizer could not fit one source character safely.")

        # Prefer a nearby dialogue/word boundary, without sacrificing more than
        # half the measured capacity. Whitespace stays in exactly one segment.
        prefix = remaining[:best]
        boundary = max(prefix.rfind("\n\n"), prefix.rfind("\n"), prefix.rfind(" "))
        if boundary >= best // 2:
            best = boundary + 1
        segments.append(remaining[:best])
        cursor += best

    if "".join(segments) != text:
        raise RuntimeError("Tier 1 segmentation did not preserve the complete source text.")
    if any(token_count(segment) > token_limit for segment in segments):
        raise RuntimeError("Tier 1 segmentation exceeded the embedding model token limit.")
    return segments


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
