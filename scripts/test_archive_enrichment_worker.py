"""Focused checks for the durable Tier 1 archive-enrichment worker core."""

import asyncio
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(os.getcwd())

from core.archive_enrichment import (
    claim_next_enrichment_chunk,
    complete_enrichment_chunk,
    ensure_enrichment_tables,
    heartbeat_enrichment_chunk,
    pause_enrichment_job,
    plan_archive_enrichment,
    resume_enrichment_job,
    run_one_idle_enrichment,
    run_one_tier1_enrichment_chunk,
    _tier1_embedding_segments,
)
from core.embeddings import embedding_to_blob
from core.legacy_archive import ensure_legacy_archive_tables


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=2)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _seed_job(db_path: Path) -> str:
    with closing(_connect(db_path)) as conn:
        ensure_legacy_archive_tables(conn)
        conn.execute(
            "INSERT INTO legacy_sessions("
            "id, source, source_conversation_id, title, message_count, archived_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                "legacy-worker-demo",
                "chatgpt",
                "worker-demo",
                "Worker demo",
                4,
                "2026-09-16T00:00:00+00:00",
            ),
        )
        conn.executemany(
            "INSERT INTO legacy_messages("
            "session_id, position, role, content, timestamp"
            ") VALUES ('legacy-worker-demo', ?, ?, ?, ?)",
            [
                (0, "user", "First question " + "A" * 300, "2026-09-16T00:00:00+00:00"),
                (1, "assistant", "First answer " + "B" * 300, "2026-09-16T00:01:00+00:00"),
                (2, "user", "Second question " + "C" * 300, "2026-09-16T00:02:00+00:00"),
                (3, "assistant", "Second answer " + "D" * 300, "2026-09-16T00:03:00+00:00"),
            ],
        )
        job = plan_archive_enrichment(
            conn,
            schedule_mode="idle",
            chunk_token_budget=128,
        )
        assert job["total_chunks"] == 2
        conn.commit()
        return job["id"]


def test_worker_completes_one_durable_chunk_at_a_time() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "worker.db"
        job_id = _seed_job(db_path)
        embedded_texts: list[str] = []

        def fake_encoder(text: str) -> list[float]:
            embedded_texts.append(text)
            return [0.25, 0.5, 0.75]

        with closing(_connect(db_path)) as conn:
            first = run_one_tier1_enrichment_chunk(
                conn,
                job_id,
                encoder=fake_encoder,
                model_name="test-minilm",
            )
            assert first["claimed"] is True
            assert first["completed"] is True
            assert first["job"]["status"] == "queued"
            assert first["job"]["chunk_counts"] == {"completed": 1, "pending": 1}
            assert first["segment_count"] == 2
            assert any("[user]\nFirst question" in text for text in embedded_texts)
            assert any("[assistant]\nFirst answer" in text for text in embedded_texts)

            second = run_one_tier1_enrichment_chunk(
                conn,
                job_id,
                encoder=fake_encoder,
                model_name="test-minilm",
            )
            assert second["completed"] is True
            assert second["job"]["status"] == "completed"
            assert second["job"]["progress_percent"] == 100.0

            exhausted = run_one_tier1_enrichment_chunk(
                conn,
                job_id,
                encoder=fake_encoder,
                model_name="test-minilm",
            )
            assert exhausted["claimed"] is False
            assert len(embedded_texts) == 4

            artifacts = conn.execute(
                "SELECT artifact_type, provider, model, status, segment_index, "
                "segment_count, artifact_embedding FROM legacy_enrichment_artifacts "
                "ORDER BY chunk_id, segment_index"
            ).fetchall()
            assert len(artifacts) == 4
            assert all(row[:4] == ("raw_chunk_embedding", "local", "test-minilm", "completed")
                       for row in artifacts)
            assert [row[4] for row in artifacts] == [0, 1, 0, 1]
            assert all(row[5] == 2 for row in artifacts)
            assert all(len(row[6]) == 12 for row in artifacts)


def test_claim_is_exclusive_and_pause_is_cooperative() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "worker.db"
        job_id = _seed_job(db_path)
        with closing(_connect(db_path)) as worker, closing(_connect(db_path)) as observer:
            claim = claim_next_enrichment_chunk(worker, job_id)
            assert claim is not None
            worker.commit()

            assert claim_next_enrichment_chunk(observer, job_id) is None
            paused = pause_enrichment_job(observer, job_id)
            assert paused["status"] == "pausing"
            observer.commit()

            assert heartbeat_enrichment_chunk(worker, job_id, claim["chunk_id"]) is True
            finished = complete_enrichment_chunk(
                worker,
                job_id,
                claim["chunk_id"],
                embedding_to_blob([1.0, 0.0]),
                model_name="test-minilm",
            )
            worker.commit()
            assert finished["status"] == "paused"
            assert finished["chunk_counts"] == {"completed": 1, "pending": 1}

            resumed = resume_enrichment_job(worker, job_id)
            assert resumed["status"] == "queued"
            worker.commit()
            final = run_one_tier1_enrichment_chunk(
                worker,
                job_id,
                encoder=lambda _: [0.0, 1.0],
                model_name="test-minilm",
            )
            assert final["job"]["status"] == "completed"


def test_failure_retry_and_fresh_connection_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "worker.db"
        job_id = _seed_job(db_path)
        with closing(_connect(db_path)) as conn:
            failed = run_one_tier1_enrichment_chunk(
                conn,
                job_id,
                encoder=lambda _: (_ for _ in ()).throw(RuntimeError("synthetic encoder failure")),
                model_name="test-minilm",
            )
            assert failed["completed"] is False
            assert failed["job"]["status"] == "failed"
            assert failed["job"]["chunk_counts"] == {"failed": 1, "pending": 1}
            assert "synthetic encoder failure" in failed["error"]

            retried = resume_enrichment_job(conn, job_id)
            assert retried["status"] == "queued"
            assert retried["chunk_counts"] == {"pending": 2}
            conn.commit()
            claim = claim_next_enrichment_chunk(
                conn,
                job_id,
                now=datetime.now(timezone.utc) - timedelta(hours=1),
            )
            assert claim is not None
            assert claim["attempts"] == 2
            conn.commit()

        # Simulate a stopped process: the bounded idle entry point opens a fresh
        # connection, recovers the stale claim, and processes only that chunk.
        result = run_one_idle_enrichment(
            db_path,
            encoder=lambda _: [0.1, 0.2],
            model_name="test-minilm",
        )
        assert result["completed"] is True
        assert result["chunk"]["attempts"] == 3
        assert result["job"]["recovery_count"] == 1
        assert result["recovered_jobs"] == [job_id]


def test_engine_runs_only_one_chunk_during_idle_window() -> None:
    from core.engine import PulseEngine

    engine = PulseEngine.__new__(PulseEngine)
    engine.db = SimpleNamespace(db_path=Path("test-persona.db"))
    engine._enrichment_idle_seconds = 60
    engine.interval = 3600
    engine._last_enrichment_status = None
    completed = {
        "claimed": True,
        "completed": True,
        "chunk": {"chunk_id": "chunk-one"},
        "job": {
            "id": "job-one",
            "status": "queued",
            "completed_chunks": 1,
            "total_chunks": 2,
            "updated_at": "2026-09-16T00:00:00+00:00",
        },
    }
    with patch("core.engine.run_one_idle_enrichment", return_value=completed) as process:
        assert asyncio.run(engine._run_idle_enrichment_once(30)) is None
        assert asyncio.run(engine._run_idle_enrichment_once(3600)) is None
        result = asyncio.run(engine._run_idle_enrichment_once(120))
        assert result == completed
        process.assert_called_once_with(Path("test-persona.db"))
    assert engine._last_enrichment_status == {
        "job_id": "job-one",
        "status": "queued",
        "completed_chunks": 1,
        "total_chunks": 2,
        "updated_at": "2026-09-16T00:00:00+00:00",
    }


def test_oversized_exchange_keeps_every_embedding_segment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oversized.db"
        with closing(_connect(db_path)) as conn:
            ensure_legacy_archive_tables(conn)
            conn.execute(
                "INSERT INTO legacy_sessions("
                "id, source, source_conversation_id, title, message_count, archived_at"
                ") VALUES ('legacy-oversized', 'generic', 'oversized', 'Oversized', 2, ?) ",
                ("2026-09-16T00:00:00+00:00",),
            )
            conn.executemany(
                "INSERT INTO legacy_messages(session_id, position, role, content, timestamp) "
                "VALUES ('legacy-oversized', ?, ?, ?, ?)",
                [
                    (0, "user", "Very long synthetic message " + "X" * 7000,
                     "2026-09-16T00:00:00+00:00"),
                    (1, "assistant", "OK",
                     "2026-09-16T00:01:00+00:00"),
                ],
            )
            job = plan_archive_enrichment(conn, schedule_mode="idle")
            assert job["total_chunks"] == 1
            oversized = conn.execute(
                "SELECT oversized_exchange FROM legacy_enrichment_chunks"
            ).fetchone()[0]
            assert oversized == 1
            conn.commit()

            embedded: list[str] = []
            result = run_one_tier1_enrichment_chunk(
                conn,
                job["id"],
                encoder=lambda text: embedded.append(text) or [0.1, 0.2],
                model_name="test-minilm",
            )
            assert result["completed"] is True
            assert result["segment_count"] > 8
            assert len(embedded) == result["segment_count"]
            assert any(text.endswith("OK") for text in embedded)
            rows = conn.execute(
                "SELECT segment_index, segment_count, length(artifact_embedding) "
                "FROM legacy_enrichment_artifacts ORDER BY segment_index"
            ).fetchall()
            assert len(rows) == result["segment_count"]
            assert [row[0] for row in rows] == list(range(len(rows)))
            assert all(row[1] == len(rows) and row[2] == 8 for row in rows)


def test_artifact_schema_migrates_existing_database() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "schema-v1.db"
        with closing(_connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE legacy_enrichment_artifacts("
                "id TEXT PRIMARY KEY, session_id TEXT NOT NULL, chunk_id TEXT NOT NULL DEFAULT '', "
                "artifact_type TEXT NOT NULL, hierarchy_level INTEGER NOT NULL DEFAULT 0, "
                "first_message_id INTEGER, last_message_id INTEGER, "
                "content_fingerprint TEXT NOT NULL, generation_fingerprint TEXT NOT NULL, "
                "provider TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '', "
                "status TEXT NOT NULL DEFAULT 'pending', artifact_text TEXT NOT NULL DEFAULT '', "
                "artifact_embedding BLOB, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "UNIQUE(artifact_type, content_fingerprint, generation_fingerprint))"
            )
            ensure_enrichment_tables(conn)
            columns = {
                row[1] for row in conn.execute(
                    "PRAGMA table_info(legacy_enrichment_artifacts)"
                ).fetchall()
            }
            assert {"segment_index", "segment_count"} <= columns
            version = conn.execute(
                "SELECT value FROM legacy_enrichment_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            assert version == "2"


def test_real_model_segmentation_uses_token_limit_and_preserves_unicode() -> None:
    class DenseTokenizer:
        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens is True
            assert truncation is False
            # Model dense Unicode as two tokens per source character plus its
            # pair of special tokens, making the 600-character heuristic unsafe.
            return [101] + [token for char in text for token in (ord(char), 1)] + [102]

    model = SimpleNamespace(tokenizer=DenseTokenizer(), max_seq_length=32)
    source = "[user]\n你好🙂" * 20 + "\n\n[assistant]\nOK"
    segments = _tier1_embedding_segments(source, model=model)
    assert len(segments) > 8
    assert "".join(segments) == source
    assert all(len(model.tokenizer.encode(
        segment,
        add_special_tokens=True,
        truncation=False,
    )) <= 32 for segment in segments)


if __name__ == "__main__":
    test_worker_completes_one_durable_chunk_at_a_time()
    test_claim_is_exclusive_and_pause_is_cooperative()
    test_failure_retry_and_fresh_connection_recovery()
    test_engine_runs_only_one_chunk_during_idle_window()
    test_oversized_exchange_keeps_every_embedding_segment()
    test_artifact_schema_migrates_existing_database()
    test_real_model_segmentation_uses_token_limit_and_preserves_unicode()
    print("[OK] archive enrichment worker")
