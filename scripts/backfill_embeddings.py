"""
Backfill embeddings for journal entries and memories that have empty embeddings.

Run: python scripts/backfill_embeddings.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context import load_embedding_model, _get_embedding_model


def backfill_dir(directory: Path, pattern: str):
    """Backfill embeddings for all JSON files matching pattern in directory."""
    if not directory.exists():
        print(f"  Directory not found: {directory}")
        return 0

    model = _get_embedding_model()
    if not model:
        print("  ERROR: Embedding model not loaded!")
        return 0

    updated = 0
    files = sorted(directory.glob(pattern))

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"  SKIP (bad file): {filepath.name}")
            continue

        # Get the text to embed
        text = data.get("text") or data.get("content") or ""
        if not text:
            print(f"  SKIP (no text): {filepath.name}")
            continue

        # Check if embedding is missing or empty
        existing = data.get("embedding", [])
        if existing and len(existing) > 0:
            print(f"  OK (has embedding): {filepath.name}")
            continue

        # Generate embedding
        embedding = model.encode(text).tolist()
        data["embedding"] = embedding

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  UPDATED: {filepath.name} — {text[:60]}...")
        updated += 1

    return updated


def main():
    print("Loading embedding model...")
    if not load_embedding_model():
        print("Failed to load embedding model. Exiting.")
        sys.exit(1)

    print()

    # Journal entries
    journal_dir = Path("data/journal")
    print(f"Backfilling journal entries ({journal_dir}):")
    j_count = backfill_dir(journal_dir, "entry_*.json")

    # Memories
    import yaml
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        memory_dir = Path(config.get("paths", {}).get("nova_memory", ""))
    else:
        memory_dir = Path.home() / ".local-memory"

    print(f"\nBackfilling memories ({memory_dir}):")
    m_count = backfill_dir(memory_dir, "memory_*.json")

    print(f"\nDone! Updated {j_count} journal entries, {m_count} memories.")


if __name__ == "__main__":
    main()
