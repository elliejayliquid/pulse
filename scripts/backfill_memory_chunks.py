"""
Backfill chunk embeddings for long memories in the shared/persona DB.

This script finds all memories whose text produces more than one chunk
(via chunk_text) and inserts missing rows into the memory_chunks table.
It is idempotent: memories that already have chunk rows are skipped.

Run once per persona / shared DB after upgrading to passive recall:
    python scripts/backfill_memory_chunks.py [--db path/to/db.sqlite]

By default uses the shared DB path from config.yaml.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Backfill memory chunk embeddings.")
    parser.add_argument("--db", help="Path to the SQLite DB. Defaults to shared DB from config.yaml.")
    args = parser.parse_args()

    import yaml
    from core.context import load_embedding_model, _get_embedding_model
    from core.embeddings import chunk_text, embedding_to_blob
    from core.db import PulseDatabase

    print("Loading embedding model...")
    if not load_embedding_model():
        print("ERROR: Failed to load embedding model. Exiting.")
        sys.exit(1)
    model = _get_embedding_model()

    # Resolve DB path — mirrors pulse.py logic:
    # 1. --db flag
    # 2. paths.shared_database (Claude/shared personas with a shared.db)
    # 3. paths.database (local personas — shared_db == persona DB there)
    db_path = None
    if args.db:
        db_path = Path(args.db)
    else:
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            shared_db_path = config.get("paths", {}).get("shared_database")
            if shared_db_path:
                db_path = Path(shared_db_path)
            else:
                persona_db_path = config.get("paths", {}).get("database")
                if persona_db_path:
                    db_path = Path(persona_db_path)
        if not db_path:
            print("ERROR: Could not determine DB path. Use --db or set paths.database / paths.shared_database in config.yaml.")
            sys.exit(1)

    if not db_path.exists():
        print(f"ERROR: DB not found at: {db_path}")
        sys.exit(1)

    print(f"Using DB: {db_path}")
    db = PulseDatabase(db_path)

    # Load all memories
    memories = db.get_all_memories(include_superseded=True, include_archived=True)
    print(f"Found {len(memories)} memories (including superseded/archived).")

    # Load existing chunk memory IDs to skip
    existing_chunks = db.get_all_memory_chunks()
    already_chunked_ids = set(r["memory_id"] for r in existing_chunks)
    print(f"Memories already chunked: {len(already_chunked_ids)}")

    processed = 0
    skipped_single = 0
    skipped_already = 0
    errors = 0

    for mem in memories:
        mem_id = mem.get("id")
        text = mem.get("text", "")

        if not text.strip():
            continue

        chunks = chunk_text(text)
        if len(chunks) <= 1:
            skipped_single += 1
            continue

        if mem_id in already_chunked_ids:
            skipped_already += 1
            continue

        try:
            chunk_vecs = model.encode(chunks)
            chunk_blobs = [embedding_to_blob(vec) for vec in chunk_vecs]
            db.save_memory_chunks(mem_id, chunk_blobs)
            processed += 1
            print(f"  Chunked memory #{mem_id} -> {len(chunks)} chunks: {text[:60]}...")
        except Exception as e:
            errors += 1
            print(f"  ERROR on memory #{mem_id}: {e}")

    print()
    print(f"Done!")
    print(f"  Chunk rows inserted for: {processed} memories")
    print(f"  Skipped (single chunk):  {skipped_single} memories")
    print(f"  Skipped (already done):  {skipped_already} memories")
    if errors:
        print(f"  Errors:                  {errors}")


if __name__ == "__main__":
    main()
