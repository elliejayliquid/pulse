"""Persona-DB staging for conversation imports.

Staged imports live in quarantine tables (`import_batches`,
`import_conversations`, `import_messages`) inside the persona database.
The engine never reads these tables: nothing staged becomes live sessions,
messages, memories, journal entries, or prompt context. Only a future
explicit commit/landing step may promote staged content into live tables.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from gui.import_preview import PreviewBatch, PreviewConversation, PreviewMessage

IMPORT_SCHEMA_VERSION = 1

_IMPORT_TABLES = ("import_batches", "import_conversations", "import_messages")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS import_batches (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    source TEXT NOT NULL,
    source_version TEXT NOT NULL DEFAULT '',
    file_name TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL DEFAULT '',
    file_size INTEGER NOT NULL DEFAULT 0,
    staged_at TEXT NOT NULL,
    continuation_conversation_id TEXT NOT NULL DEFAULT '',
    warnings TEXT NOT NULL DEFAULT '[]',
    skipped_counts TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS import_conversations (
    batch_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    source_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    choice TEXT NOT NULL DEFAULT 'keep',
    warnings TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (batch_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS import_messages (
    batch_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    source_message_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    source_role TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    skipped INTEGER NOT NULL DEFAULT 0,
    skip_reason TEXT NOT NULL DEFAULT '',
    attachment_placeholder INTEGER NOT NULL DEFAULT 0,
    excluded INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (batch_id, conversation_id, position)
);
"""


def ensure_import_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def has_import_tables(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
        _IMPORT_TABLES,
    ).fetchall()
    return len(rows) == len(_IMPORT_TABLES)


def clean_curation(curation: Any) -> dict:
    """Validate the curation payload from the GUI into a trusted shape."""
    if curation is None:
        curation = {}
    if not isinstance(curation, dict):
        raise ValueError("Curation must be an object.")
    choices = {}
    for conversation_id, choice in (curation.get("choices") or {}).items():
        if choice not in ("keep", "exclude"):
            raise ValueError(f"Unknown curation choice: {choice}")
        choices[str(conversation_id)] = choice
    excluded = {}
    for conversation_id, positions in (curation.get("excluded_messages") or {}).items():
        cleaned = sorted({int(position) for position in positions or []})
        if any(position < 0 for position in cleaned):
            raise ValueError("Excluded message positions must be non-negative.")
        excluded[str(conversation_id)] = cleaned
    return {
        "choices": choices,
        "continuation_id": str(curation.get("continuation_id") or ""),
        "excluded_messages": excluded,
    }


def stage_batch(
    conn: sqlite3.Connection,
    batch_id: str,
    batch: PreviewBatch,
    file_path: str,
    curation: dict,
) -> dict:
    """Write a previewed batch into the staging tables (replace semantics).

    Re-staging the same batch id replaces the previous rows so a re-parsed
    file never duplicates its own archive. Empty conversations are staged
    with choice 'exclude' regardless of the requested choice.
    """
    ensure_import_tables(conn)
    delete_batch(conn, batch_id)
    now = datetime.now(timezone.utc).isoformat()
    choices = curation["choices"]
    excluded_messages = curation["excluded_messages"]
    continuation_id = curation["continuation_id"]
    known_ids = {conversation.id for conversation in batch.conversations}
    if continuation_id and continuation_id not in known_ids:
        continuation_id = ""
    conn.execute(
        "INSERT INTO import_batches "
        "(id, schema_version, source, source_version, file_name, file_path, file_size, "
        " staged_at, continuation_conversation_id, warnings, skipped_counts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            batch_id,
            IMPORT_SCHEMA_VERSION,
            batch.source,
            batch.source_version,
            batch.file_name,
            str(file_path or ""),
            batch.file_size,
            now,
            continuation_id,
            json.dumps(batch.warnings),
            json.dumps(batch.skipped_counts),
        ),
    )
    staged_conversations = 0
    staged_messages = 0
    for position, conversation in enumerate(batch.conversations):
        visible = [message for message in conversation.messages if not message.skipped]
        choice = choices.get(conversation.id, "keep")
        if not visible:
            choice = "exclude"
        conn.execute(
            "INSERT INTO import_conversations "
            "(batch_id, conversation_id, position, source_id, title, created_at, "
            " updated_at, model, choice, warnings) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                batch_id,
                conversation.id,
                position,
                conversation.source_id,
                conversation.title,
                conversation.created_at,
                conversation.updated_at,
                conversation.model,
                choice,
                json.dumps(conversation.warnings),
            ),
        )
        staged_conversations += 1
        excluded_positions = set(excluded_messages.get(conversation.id, []))
        conn.executemany(
            "INSERT INTO import_messages "
            "(batch_id, conversation_id, position, source_message_id, role, source_role, "
            " content, created_at, model, skipped, skip_reason, attachment_placeholder, excluded) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    batch_id,
                    conversation.id,
                    message_position,
                    message.id,
                    message.role,
                    message.source_role,
                    message.content,
                    message.created_at,
                    message.model,
                    int(message.skipped),
                    message.skip_reason,
                    int(message.attachment_placeholder),
                    int(message_position in excluded_positions),
                )
                for message_position, message in enumerate(conversation.messages)
            ],
        )
        staged_messages += len(conversation.messages)
    return {
        "staged_conversations": staged_conversations,
        "staged_messages": staged_messages,
        "continuation_id": continuation_id,
    }


def list_batches(conn: sqlite3.Connection) -> list[dict]:
    if not has_import_tables(conn):
        return []
    rows = conn.execute(
        """
        SELECT b.id, b.source, b.file_name, b.file_size, b.staged_at,
               b.continuation_conversation_id,
               COUNT(DISTINCT c.conversation_id) AS conversation_count,
               (SELECT COUNT(*) FROM import_messages m WHERE m.batch_id = b.id) AS message_count
        FROM import_batches b
        LEFT JOIN import_conversations c ON c.batch_id = b.id
        GROUP BY b.id
        ORDER BY b.staged_at DESC
        """
    ).fetchall()
    return [
        {
            "id": row[0],
            "source": row[1],
            "file_name": row[2],
            "file_size": row[3],
            "staged_at": row[4],
            "continuation_id": row[5],
            "conversation_count": row[6],
            "message_count": row[7],
        }
        for row in rows
    ]


def load_batch(conn: sqlite3.Connection, batch_id: str) -> dict | None:
    """Rebuild a staged batch into preview objects plus its curation state."""
    if not has_import_tables(conn):
        return None
    batch_row = conn.execute(
        "SELECT id, source, source_version, file_name, file_path, file_size, staged_at, "
        "continuation_conversation_id, warnings, skipped_counts "
        "FROM import_batches WHERE id = ?",
        (batch_id,),
    ).fetchone()
    if not batch_row:
        return None
    conversations: list[PreviewConversation] = []
    choices: dict[str, str] = {}
    conversation_rows = conn.execute(
        "SELECT conversation_id, source_id, title, created_at, updated_at, model, choice, warnings "
        "FROM import_conversations WHERE batch_id = ? ORDER BY position",
        (batch_id,),
    ).fetchall()
    for conv_row in conversation_rows:
        message_rows = conn.execute(
            "SELECT source_message_id, role, source_role, content, created_at, model, "
            "skipped, skip_reason, attachment_placeholder, excluded "
            "FROM import_messages WHERE batch_id = ? AND conversation_id = ? ORDER BY position",
            (batch_id, conv_row[0]),
        ).fetchall()
        messages = [
            PreviewMessage(
                id=row[0],
                role=row[1],
                source_role=row[2],
                content=row[3],
                created_at=row[4],
                model=row[5],
                skipped=bool(row[6]),
                skip_reason=row[7],
                attachment_placeholder=bool(row[8]),
                excluded=bool(row[9]),
            )
            for row in message_rows
        ]
        conversations.append(PreviewConversation(
            id=conv_row[0],
            source_id=conv_row[1],
            title=conv_row[2],
            created_at=conv_row[3],
            updated_at=conv_row[4],
            model=conv_row[5],
            messages=messages,
            warnings=_json_list(conv_row[7]),
        ))
        choices[conv_row[0]] = conv_row[6]
    batch = PreviewBatch(
        source=batch_row[1],
        source_version=batch_row[2],
        file_name=batch_row[3],
        file_size=batch_row[5],
        conversations=conversations,
        warnings=_json_list(batch_row[8]),
        skipped_counts=_json_dict(batch_row[9]),
    )
    return {
        "batch": batch,
        "file_path": batch_row[4],
        "staged_at": batch_row[6],
        "choices": choices,
        "continuation_id": batch_row[7],
    }


def update_curation(conn: sqlite3.Connection, batch_id: str, curation: dict) -> bool:
    """Persist curation changes for an already-staged batch.

    Conversations absent from the payload keep their staged values.
    """
    if not has_import_tables(conn):
        return False
    exists = conn.execute(
        "SELECT 1 FROM import_batches WHERE id = ?", (batch_id,)
    ).fetchone()
    if not exists:
        return False
    for conversation_id, choice in curation["choices"].items():
        conn.execute(
            "UPDATE import_conversations SET choice = ? WHERE batch_id = ? AND conversation_id = ?",
            (choice, batch_id, conversation_id),
        )
    conn.execute(
        "UPDATE import_batches SET continuation_conversation_id = ? WHERE id = ?",
        (curation["continuation_id"], batch_id),
    )
    for conversation_id, positions in curation["excluded_messages"].items():
        conn.execute(
            "UPDATE import_messages SET excluded = 0 WHERE batch_id = ? AND conversation_id = ?",
            (batch_id, conversation_id),
        )
        if positions:
            conn.executemany(
                "UPDATE import_messages SET excluded = 1 "
                "WHERE batch_id = ? AND conversation_id = ? AND position = ?",
                [(batch_id, conversation_id, position) for position in positions],
            )
    return True


def delete_batch(conn: sqlite3.Connection, batch_id: str) -> bool:
    if not has_import_tables(conn):
        return False
    removed = conn.execute(
        "DELETE FROM import_batches WHERE id = ?", (batch_id,)
    ).rowcount
    conn.execute("DELETE FROM import_conversations WHERE batch_id = ?", (batch_id,))
    conn.execute("DELETE FROM import_messages WHERE batch_id = ?", (batch_id,))
    return bool(removed)


def _json_list(value: Any) -> list:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _json_dict(value: Any) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
