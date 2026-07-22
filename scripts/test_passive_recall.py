"""
Passive Recall test script — Phase 1 & Phase 2 acceptance criteria.

Runs standalone with a temp SQLite DB. No test framework needed.
Prints PASS/FAIL per case.

Usage:  python scripts/test_passive_recall.py

Requirements: sentence-transformers installed and model cached.
"""

import io
import sys
import struct
import tempfile
from pathlib import Path

# Windows stdout encoding safety
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ─────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0


def ok(label: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS  {label}")


def fail(label: str, detail: str = ""):
    global FAIL_COUNT
    FAIL_COUNT += 1
    msg = f"  FAIL  {label}"
    if detail:
        msg += f"\n        {detail}"
    print(msg)


def vec_to_blob(vec) -> bytes:
    import numpy as np
    return np.array(vec, dtype=np.float32).tobytes()


def make_config(db, tmpdir):
    return {
        "_db": db,
        "_shared_db": db,
        "paths": {
            "persona": str(Path(__file__).parent.parent / "persona.yaml"),
            "memories": str(Path(tmpdir) / "memories"),
            "conversation": str(Path(tmpdir) / "conversation.json"),
        },
        "model": {"max_context": 8192},
        "context": {
            "recall": {
                "enabled": True,
                "top_k": 6,
                "min_similarity": 0.30,
                "core_importance": 8,
                "heartbeat_query": False,
            }
        },
    }


def make_config_disabled(db, tmpdir):
    cfg = make_config(db, tmpdir)
    cfg["context"]["recall"]["enabled"] = False
    return cfg


def make_config_no_recall_block(db, tmpdir):
    cfg = make_config(db, tmpdir)
    del cfg["context"]["recall"]
    del cfg["context"]
    return cfg


# ── chunk_text unit tests ────────────────────────────────────────────────────

def test_chunk_text():
    print("\n--- chunk_text edge cases ---")
    from core.embeddings import chunk_text

    # Empty string
    result = chunk_text("")
    if result == []:
        ok("chunk_text('') returns []")
    else:
        fail("chunk_text('') returns []", f"got {result!r}")

    # Whitespace only
    result = chunk_text("   \n  ")
    if result == []:
        ok("chunk_text(whitespace) returns []")
    else:
        fail("chunk_text(whitespace) returns []", f"got {result!r}")

    # Single short line (under 20 chars threshold)
    result = chunk_text("ok")
    if result == []:
        ok("chunk_text(tiny text) returns [] (filtered)")
    else:
        fail("chunk_text(tiny text) filtered", f"got {result!r}")

    # Single normal line stays as one chunk
    result = chunk_text("This is a normal sentence that is long enough.")
    if len(result) == 1:
        ok("chunk_text(single line) returns 1 chunk")
    else:
        fail("chunk_text(single line) returns 1 chunk", f"got {len(result)} chunks")

    # Wall-of-text with no newlines — should hard-split
    long_text = "A" * 1300
    result = chunk_text(long_text, max_chars=600)
    if len(result) >= 2:
        ok(f"chunk_text(wall-of-text) hard-splits (got {len(result)} chunks)")
    else:
        fail("chunk_text(wall-of-text) should hard-split", f"got {len(result)}")

    # >8 paragraphs — cap applies, first and last kept
    paras = [f"Paragraph {i}: " + ("word " * 10) for i in range(12)]
    text = "\n\n".join(paras)
    result = chunk_text(text, max_chunks=8)
    if len(result) == 8:
        ok("chunk_text(12 paragraphs) caps at 8 chunks")
    else:
        fail("chunk_text(12 paragraphs) should cap at 8", f"got {len(result)}")

    # First and last chunks should be preserved when capping
    # (para 0 should be in first chunks, para 11 in last chunks)
    if result and "Paragraph 0:" in result[0]:
        ok("chunk_text cap preserves first chunk")
    else:
        fail("chunk_text cap preserves first chunk", f"first was: {result[0][:60] if result else 'empty'!r}")

    if result and "Paragraph 11:" in result[-1]:
        ok("chunk_text cap preserves last chunk")
    else:
        fail("chunk_text cap preserves last chunk", f"last was: {result[-1][:60] if result else 'empty'!r}")

    # Multi-paragraph (2 normal-sized paragraphs)
    two_para = "First paragraph talks about cats.\n\nSecond paragraph talks about dogs."
    result = chunk_text(two_para)
    if len(result) == 2:
        ok("chunk_text(2 paragraphs) returns 2 chunks")
    else:
        fail("chunk_text(2 paragraphs) returns 2 chunks", f"got {result!r}")


# ── max_similarity unit tests ────────────────────────────────────────────────

def test_max_similarity():
    print("\n--- max_similarity ---")
    import numpy as np
    from core.embeddings import max_similarity

    # Empty inputs
    if max_similarity([], []) == 0.0:
        ok("max_similarity([], []) = 0.0")
    else:
        fail("max_similarity([], []) = 0.0")

    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if max_similarity([v], []) == 0.0:
        ok("max_similarity([v], []) = 0.0")
    else:
        fail("max_similarity([v], []) = 0.0")

    # Identical vectors = 1.0
    sim = max_similarity([v], [v])
    if abs(sim - 1.0) < 1e-5:
        ok("max_similarity(v, v) ≈ 1.0")
    else:
        fail("max_similarity(v, v) ≈ 1.0", f"got {sim}")

    # Orthogonal = 0.0
    u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim = max_similarity([v], [u])
    if abs(sim) < 1e-5:
        ok("max_similarity(orthogonal) ≈ 0.0")
    else:
        fail("max_similarity(orthogonal) ≈ 0.0", f"got {sim}")

    # Multi-query: best of several query chunks
    w = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = max_similarity([u, w], [v])
    if abs(sim - 1.0) < 1e-5:
        ok("max_similarity picks best query chunk")
    else:
        fail("max_similarity picks best query chunk", f"got {sim}")

    # Zero vector handled gracefully
    zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sim = max_similarity([zero], [v])
    if sim == 0.0:
        ok("max_similarity(zero_vec, v) = 0.0 (no crash)")
    else:
        fail("max_similarity(zero_vec, v) should be 0.0", f"got {sim}")


# ── Integration tests ────────────────────────────────────────────────────────

def test_recall_disabled_byte_identical():
    """Criterion 1: disabled recall produces same output as old behavior."""
    print("\n--- Criterion 1: disabled recall = old behavior ---")
    import numpy as np
    from core.db import PulseDatabase
    from core.context import ContextManager

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            db.save_memory("Favorite color is blue.", tags=["prefs"], importance=5)
            db.save_memory("Has a dog named Max.", tags=["pets"], importance=6)

            # Enabled, no query
            ctx_enabled = ContextManager(make_config(db, tmpdir))
            out_no_query = ctx_enabled._load_memories(None)

            # Disabled config
            ctx_disabled = ContextManager(make_config_disabled(db, tmpdir))
            out_disabled = ctx_disabled._load_memories("anything")

            # No recall block config
            ctx_no_block = ContextManager(make_config_no_recall_block(db, tmpdir))
            out_no_block = ctx_no_block._load_memories("anything")

            if out_no_query == out_disabled:
                ok("disabled recall matches no-query output")
            else:
                fail("disabled recall matches no-query output",
                     f"\nEnabled/no-query:\n{out_no_query}\nDisabled:\n{out_disabled}")

            if out_no_block == out_disabled:
                ok("missing recall block also falls back identically")
            else:
                fail("missing recall block falls back identically",
                     f"\nNo block:\n{out_no_block}\nDisabled:\n{out_disabled}")

        finally:
            db.close()


def test_query_chunking_surfaces_low_importance():
    """Criterion 2: multi-paragraph message surfaces low-importance fact via last chunk."""
    print("\n--- Criterion 2: query chunking surfaces relevant fact ---")
    from core.db import PulseDatabase
    from core.context import ContextManager, load_embedding_model, _get_embedding_model
    from core.embeddings import embedding_to_blob

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            # Save a low-importance (importance=3, below core threshold of 8)
            # fact about "driveway"
            target_text = "The neighbor planted vines along the driveway in spring."
            emb = embedding_to_blob(model.encode(target_text))
            mem_id = db.save_memory(target_text, tags=["home"], importance=3, embedding=emb)

            # Query: first paragraph is off-topic, last paragraph mentions driveway
            query = (
                "I had a really busy day at work today. So many meetings.\n\n"
                "The project deadline is coming up fast.\n\n"
                "Oh, also — I walked past the driveway and noticed something odd."
            )
            ctx = ContextManager(make_config(db, tmpdir))
            out = ctx._load_memories(query)

            if "driveway" in out.lower() and "Possibly relevant" in out:
                ok("query chunking surfaced low-importance 'driveway' fact")
            else:
                fail("query chunking should surface 'driveway' fact",
                     f"Output:\n{out}")
        finally:
            db.close()


def test_unrelated_query_yields_empty_relevant():
    """Criterion 3: unrelated query yields no 'Possibly relevant' section."""
    print("\n--- Criterion 3: unrelated query yields no results ---")
    from core.db import PulseDatabase
    from core.context import ContextManager, _get_embedding_model
    from core.embeddings import embedding_to_blob

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            target_text = "Nova loves painting watercolor flowers."
            emb = embedding_to_blob(model.encode(target_text))
            db.save_memory(target_text, tags=["hobbies"], importance=4, embedding=emb)

            # Query about something totally unrelated
            query = "The quarterly earnings report for the tech sector looks positive."
            ctx = ContextManager(make_config(db, tmpdir))
            out = ctx._load_memories(query)

            if "Possibly relevant" not in out:
                ok("unrelated query yields no 'Possibly relevant' section")
            else:
                fail("unrelated query should yield no 'Possibly relevant' section",
                     f"Output:\n{out}")
        finally:
            db.close()


def test_high_importance_always_in_core():
    """Criterion 4: high-importance facts always appear regardless of query."""
    print("\n--- Criterion 4: high-importance facts always in core ---")
    from core.db import PulseDatabase
    from core.context import ContextManager, _get_embedding_model
    from core.embeddings import embedding_to_blob

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            core_text = "Has severe peanut allergy — never suggest peanuts."
            emb = embedding_to_blob(model.encode(core_text))
            db.save_memory(core_text, tags=["health"], importance=9, embedding=emb)

            # Totally unrelated query — should still see the core fact
            query = "What time is the bus to downtown?"
            ctx = ContextManager(make_config(db, tmpdir))
            out = ctx._load_memories(query)

            if "peanut" in out.lower() and "Key facts" in out:
                ok("high-importance fact appears in core regardless of query")
            else:
                fail("high-importance fact should appear in core",
                     f"Output:\n{out}")
        finally:
            db.close()


def test_null_embedding_no_crash():
    """Criterion 5: NULL embeddings don't crash recall."""
    print("\n--- Criterion 5: NULL embeddings don't crash ---")
    from core.db import PulseDatabase
    from core.context import ContextManager

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            # Save memory with no embedding
            db.save_memory("Memory with no embedding.", tags=[], importance=3, embedding=None)

            ctx = ContextManager(make_config(db, tmpdir))
            try:
                out = ctx._load_memories("some query about anything")
                ok("NULL embedding memory doesn't crash recall")
            except Exception as e:
                fail("NULL embedding memory should not crash", str(e))
        finally:
            db.close()


def test_old_session_log_surfaces_truncated():
    """Criterion 6: older session_log can surface in 'Possibly relevant', truncated to ~300 chars."""
    print("\n--- Criterion 6: old session_log surfaces truncated ---")
    from core.db import PulseDatabase
    from core.context import ContextManager, _get_embedding_model
    from core.embeddings import embedding_to_blob

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            # Old session log mentioning "garden" (longer than 300 chars)
            old_log_text = (
                "We talked about the garden today. The tomatoes are coming in nicely. "
                "Also discussed the new trellis design for the climbing roses. "
                "The neighbor asked if we'd share some seedlings this spring. "
                "It was a warm afternoon and the whole garden smelled amazing. "
                "We ended the session feeling happy about the outdoor plans."
            )
            emb = embedding_to_blob(model.encode(old_log_text))
            # Old session log — saved with an explicit old date so it sorts into pool
            db.save_memory(old_log_text, tags=["telegram_chat"], type="session_log",
                           importance=10, embedding=emb, status="historical",
                           date="2025-03-01T10:00:00")
            # Add a newest session log (will be "core" / "Last session")
            newest_log = "Quick check-in today. Nothing major."
            emb2 = embedding_to_blob(model.encode(newest_log))
            db.save_memory(newest_log, tags=["telegram_chat"], type="session_log",
                           importance=10, embedding=emb2, status="historical")

            query = "The garden is looking beautiful this spring!"
            ctx = ContextManager(make_config(db, tmpdir))
            out = ctx._load_memories(query)

            if "old session summary" in out.lower() or "from an old session" in out.lower():
                ok("old session_log surfaces in 'Possibly relevant'")
            else:
                fail("old session_log should surface in 'Possibly relevant'",
                     f"Output:\n{out}")

            # Check truncation (should have '…' if over 300 chars)
            if "…" in out or len(old_log_text) <= 300:
                ok("old session_log is truncated with '…'")
            else:
                fail("old session_log should be truncated", f"Output:\n{out}")
        finally:
            db.close()


def test_chunk_embedding_surfaces_late_paragraph():
    """Criterion 7: 3-paragraph summary whose third paragraph matches query
    is retrieved via chunk embedding (not just main embedding)."""
    print("\n--- Criterion 7: chunk embedding surfaces 3rd-paragraph match ---")
    from core.db import PulseDatabase
    from core.context import ContextManager, _get_embedding_model
    from core.embeddings import embedding_to_blob, chunk_text

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            # 3-para summary: para 1 and 2 totally off-topic from query,
            # para 3 is about "aquarium" which the query will ask about.
            summary_text = (
                "We had a long discussion about the weekend hike plans.\n\n"
                "Then we talked about a recipe for mushroom risotto that turned out well.\n\n"
                "At the very end, we noted that the aquarium fish tank filter broke and needs replacing."
            )
            # Full-text embedding (deliberately off-topic due to first two paragraphs)
            main_emb = embedding_to_blob(model.encode(summary_text))
            mem_id = db.save_memory(summary_text, tags=["session"], importance=4,
                                    embedding=main_emb, type="session_log", status="historical")

            # Also save as the latest session log so it doesn't become "core" automatically
            # — actually we need a newer session log to push this one to pool
            newest = "Today was quiet."
            emb2 = embedding_to_blob(model.encode(newest))
            db.save_memory(newest, tags=["session"], importance=10, embedding=emb2,
                           type="session_log", status="historical")

            # Save chunk embeddings — para 3 has "aquarium"
            chunks = chunk_text(summary_text)
            if len(chunks) > 1:
                chunk_vecs = model.encode(chunks)
                chunk_blobs = [embedding_to_blob(v) for v in chunk_vecs]
                db.save_memory_chunks(mem_id, chunk_blobs)

                # Query specifically about the aquarium topic
                query = "What happened with the aquarium fish tank?"
                ctx = ContextManager(make_config(db, tmpdir))
                out = ctx._load_memories(query)

                if "aquarium" in out.lower():
                    ok("chunk embedding surfaced 3rd-paragraph 'aquarium' match")
                else:
                    fail("chunk embedding should surface 3rd-paragraph match",
                         f"Chunks: {len(chunks)}, Output:\n{out}")
            else:
                print("  SKIP: summary text didn't produce multiple chunks (model may differ)")
        finally:
            db.close()


def test_backfill_idempotent():
    """Criterion 8: backfill script is idempotent — second run inserts 0 rows."""
    print("\n--- Criterion 8: backfill is idempotent ---")
    from core.db import PulseDatabase
    from core.context import _get_embedding_model
    from core.embeddings import embedding_to_blob, chunk_text

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            # Multi-chunk memory
            text = (
                "First paragraph about cats and their habits.\n\n"
                "Second paragraph about dogs and their loyalty."
            )
            emb = embedding_to_blob(model.encode(text))
            mem_id = db.save_memory(text, tags=[], importance=5, embedding=emb)

            # Run backfill manually
            chunks = chunk_text(text)
            if len(chunks) > 1:
                chunk_vecs = model.encode(chunks)
                chunk_blobs = [embedding_to_blob(v) for v in chunk_vecs]
                db.save_memory_chunks(mem_id, chunk_blobs)
                first_rows = db.get_all_memory_chunks()
                first_count = len(first_rows)

                # Run again — should be same count (save_memory_chunks is delete+reinsert per memory)
                db.save_memory_chunks(mem_id, chunk_blobs)
                second_rows = db.get_all_memory_chunks()
                second_count = len(second_rows)

                if first_count == second_count:
                    ok(f"backfill idempotent: {first_count} rows before = {second_count} rows after")
                else:
                    fail("backfill should be idempotent",
                         f"first={first_count}, second={second_count}")
            else:
                print("  SKIP: text didn't produce multiple chunks")
        finally:
            db.close()


def test_delete_memory_removes_chunks():
    """Criterion 9: deleting a memory removes its chunk rows."""
    print("\n--- Criterion 9: delete memory removes chunk rows ---")
    from core.db import PulseDatabase
    from core.context import _get_embedding_model
    from core.embeddings import embedding_to_blob, chunk_text

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            text = (
                "First paragraph about hiking in the mountains.\n\n"
                "Second paragraph about camping near a river."
            )
            emb = embedding_to_blob(model.encode(text))
            mem_id = db.save_memory(text, tags=[], importance=5, embedding=emb)

            chunks = chunk_text(text)
            if len(chunks) > 1:
                chunk_vecs = model.encode(chunks)
                chunk_blobs = [embedding_to_blob(v) for v in chunk_vecs]
                db.save_memory_chunks(mem_id, chunk_blobs)

                rows_before = db.get_all_memory_chunks()
                chunk_count = sum(1 for r in rows_before if r["memory_id"] == mem_id)
                assert chunk_count > 0, "No chunks saved!"

                db.delete_memory(mem_id)
                rows_after = db.get_all_memory_chunks()
                remaining = sum(1 for r in rows_after if r["memory_id"] == mem_id)

                if remaining == 0:
                    ok(f"delete_memory removed all {chunk_count} chunk rows")
                else:
                    fail("delete_memory should remove chunk rows",
                         f"{remaining} chunks remain for deleted memory #{mem_id}")
            else:
                print("  SKIP: text didn't produce multiple chunks")
        finally:
            db.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def test_recency_breaks_near_ties():
    """Recency: with equal cosine, the newer memory ranks ahead of the older one."""
    print("\n--- Recency: newer memory wins a near-tie ---")
    from core.db import PulseDatabase
    from core.context import ContextManager, _get_embedding_model
    from core.embeddings import embedding_to_blob

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            # Identical text -> identical cosine, so ranking is decided purely by
            # the importance + recency boosts. Give the OLDER memory a small
            # importance edge so it wins *without* recency; recency should then
            # flip the newer one to the top. (importance boost gap = 2*0.002 =
            # 0.004; the newer memory's recency bonus ≈ 0.018 >> that.)
            text = "Lena loves hiking in the Waitakere ranges on the weekend."
            emb = embedding_to_blob(model.encode(text))
            db.save_memory(text, tags=["oldmem"], type="fact", importance=7,
                           embedding=emb, status="current", date="2025-01-01T10:00:00")
            db.save_memory(text, tags=["newmem"], type="fact", importance=5,
                           embedding=emb, status="current", date="2026-07-01T10:00:00")

            query = "Did we ever talk about hiking?"

            # Recency OFF: the older, higher-importance memory should lead.
            cfg0 = make_config(db, tmpdir)
            cfg0["context"]["recall"]["recency_weight"] = 0
            out0 = ContextManager(cfg0)._load_memories(query)
            if out0.find("oldmem") != -1 and out0.find("oldmem") < out0.find("newmem"):
                ok("recency_weight=0: older higher-importance memory leads")
            else:
                fail("without recency the higher-importance memory should lead",
                     f"Output:\n{out0}")

            # Recency ON (default): the newer memory should overtake it.
            out = ContextManager(make_config(db, tmpdir))._load_memories(query)
            i_new, i_old = out.find("newmem"), out.find("oldmem")
            if i_new != -1 and i_old != -1 and i_new < i_old:
                ok("recency flips the newer memory ahead of the older one")
            else:
                fail("recency should lift the newer memory to the top",
                     f"i_new={i_new}, i_old={i_old}\nOutput:\n{out}")
        finally:
            db.close()


def test_expand_memory_returns_full_text():
    """expand_memory returns the untruncated memory text by ID."""
    print("\n--- expand_memory: full untruncated text by ID ---")
    from core.db import PulseDatabase
    from core.context import _get_embedding_model
    from core.embeddings import embedding_to_blob
    from skills.memory import MemorySkill

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            long_text = (
                "We talked about the garden today. The tomatoes are coming in nicely. "
                "Also discussed the new trellis design for the climbing roses. "
                "The neighbor asked if we'd share some seedlings this spring. "
                "It was a warm afternoon and the whole garden smelled amazing. "
                "The very last thing we agreed on was to repaint the back fence teal."
            )
            emb = embedding_to_blob(model.encode(long_text))
            mem_id = db.save_memory(long_text, tags=["telegram_chat"], type="session_log",
                                    importance=10, embedding=emb, status="historical")

            skill = MemorySkill(make_config(db, tmpdir))
            out = skill.execute("expand_memory", {"memory_id": mem_id})

            # The tail sentence is past the 300-char recall clip — it must appear.
            if "repaint the back fence teal" in out:
                ok("expand_memory returns the full text past the 300-char clip")
            else:
                fail("expand_memory should return untruncated text", f"Output:\n{out}")

            missing = skill.execute("expand_memory", {"memory_id": 999999})
            if "not found" in missing.lower():
                ok("expand_memory reports a missing ID cleanly")
            else:
                fail("expand_memory should report missing IDs", f"Output:\n{missing}")
        finally:
            db.close()


def test_recall_advertises_expand_for_clipped_summaries():
    """A clipped session summary in recall carries the expand_memory affordance."""
    print("\n--- Recall block advertises expand_memory on clipped summaries ---")
    from core.db import PulseDatabase
    from core.context import ContextManager, _get_embedding_model
    from core.embeddings import embedding_to_blob

    model = _get_embedding_model()
    if not model:
        print("  SKIP: embedding model not available")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        db = PulseDatabase(Path(tmpdir) / "test.db")
        try:
            old_log_text = (
                "We talked about the garden today. The tomatoes are coming in nicely. "
                "Also discussed the new trellis design for the climbing roses. "
                "The neighbor asked if we'd share some seedlings this spring. "
                "It was a warm afternoon and the whole garden smelled amazing. "
                "We ended the session feeling happy about the outdoor plans."
            )
            emb = embedding_to_blob(model.encode(old_log_text))
            clipped_id = db.save_memory(old_log_text, tags=["telegram_chat"], type="session_log",
                                        importance=10, embedding=emb, status="historical",
                                        date="2025-03-01T10:00:00")
            newest = "Quick check-in today. Nothing major."
            db.save_memory(newest, tags=["telegram_chat"], type="session_log",
                           importance=10, embedding=embedding_to_blob(model.encode(newest)),
                           status="historical")

            out = ContextManager(make_config(db, tmpdir))._load_memories(
                "The garden is looking beautiful this spring!"
            )
            if f"expand_memory({clipped_id})" in out:
                ok("clipped summary advertises expand_memory with its ID")
            else:
                fail("clipped summary should advertise expand_memory", f"Output:\n{out}")
        finally:
            db.close()


def main():
    print("=" * 60)
    print("  Passive Recall — Phase 1 & 2 Acceptance Tests")
    print("=" * 60)

    # Load embedding model for integration tests
    print("\nLoading embedding model (required for most tests)...")
    from core.context import load_embedding_model
    model_loaded = load_embedding_model()
    if not model_loaded:
        print("  WARNING: Embedding model not loaded — integration tests will be skipped.")

    # chunk_text and max_similarity (pure functions, no model needed)
    test_chunk_text()
    test_max_similarity()

    # Integration tests (require model + DB)
    test_recall_disabled_byte_identical()
    test_query_chunking_surfaces_low_importance()
    test_unrelated_query_yields_empty_relevant()
    test_high_importance_always_in_core()
    test_null_embedding_no_crash()
    test_old_session_log_surfaces_truncated()
    test_chunk_embedding_surfaces_late_paragraph()
    test_backfill_idempotent()
    test_delete_memory_removes_chunks()
    test_recency_breaks_near_ties()
    test_expand_memory_returns_full_text()
    test_recall_advertises_expand_for_clipped_summaries()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 60)

    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
