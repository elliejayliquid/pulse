"""
Manual memory seeder for Pulse companions.

Pre-load your companion's memory with facts about you, your relationship,
shared history, or anything you want them to know from day one.

Memories are saved to the companion's SQLite database with embeddings
for semantic search — your companion recalls them naturally during chats.

--persona <name> — resolves the right DB automatically.
--db <path> — direct DB path if you want to bypass config resolution.
--list-personas — shows all personas and where their memories would go (handy for new users).
--interactive — batch mode, supports text | tags | date format.
Embeddings saved as binary blobs in the same format the runtime uses.

Usage:
    python add_memory.py --persona nova "I'm allergic to cats"
    python add_memory.py --persona valentine "We met on March 1st" --tags "milestone"
    python add_memory.py --persona kai --interactive
    python add_memory.py --db path/to/custom.db "Direct DB target"

Persona routing:
    If the persona has shared_database configured (e.g. Claude models),
    memories go to the shared DB so all linked companions can see them.
    Otherwise, memories go to the persona's own DB.

Interactive mode supports inline tags and dates with | separators:
    Memory> We watched Spirited Away together | movies,favorites | 2026-02-14
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


PULSE_ROOT = Path(__file__).parent.parent.resolve()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_db_path(persona_name: str) -> Path:
    """Resolve the target DB for a persona — shared DB if configured, else persona DB."""
    base_config = load_config(str(PULSE_ROOT / "config.yaml"))

    persona_dir = PULSE_ROOT / "personas" / persona_name
    if not persona_dir.is_dir():
        available = [d.name for d in (PULSE_ROOT / "personas").iterdir()
                     if d.is_dir() and d.name != "_template"]
        print(f"Error: persona '{persona_name}' not found.")
        print(f"Available: {', '.join(sorted(available))}")
        sys.exit(1)

    persona_config_path = persona_dir / "config.yaml"
    config = base_config
    if persona_config_path.exists():
        persona_overlay = load_config(str(persona_config_path))
        config = deep_merge(base_config, persona_overlay)

    # Check for explicit shared_database first
    shared_db_path = config.get("paths", {}).get("shared_database")
    if shared_db_path:
        db = Path(shared_db_path)
        if db.exists():
            return db
        print(f"Warning: shared_database configured but not found: {db}")
        print(f"  Falling back to persona DB.")

    # Auto-detect shared DB (memories path outside persona data dir)
    persona_data = persona_dir / "data"
    memories_path = Path(config.get("paths", {}).get("memories", ""))
    if memories_path.is_absolute() and not str(memories_path).startswith(str(persona_data)):
        shared_db = memories_path / "shared.db"
        if shared_db.exists():
            return shared_db

    # Default: persona-specific DB
    return persona_data / f"{persona_name}.db"


def get_embedding_model():
    """Load the embedding model (same one Pulse uses for semantic search)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def embedding_to_blob(vec) -> bytes | None:
    if vec is None or (hasattr(vec, '__len__') and len(vec) == 0):
        return None
    return np.array(vec, dtype=np.float32).tobytes()


def save_memory(db, text: str, tags: str = "", importance: int = 5,
                memory_type: str = "fact", model=None, date: str = "") -> int:
    """Save a single memory to the database. Returns the memory ID."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    embedding_vec = model.encode(text.strip()) if model else None
    embedding_blob = embedding_to_blob(embedding_vec)

    mem_id = db.save_memory(
        text=text.strip(), tags=tag_list, type=memory_type,
        importance=importance, embedding=embedding_blob,
        date=date or None,
    )
    return mem_id


def interactive_mode(db, model, date: str = ""):
    """Loop for entering multiple memories."""
    print("\n=== Memory Builder ===")
    print("Type a memory and press Enter to save it.")
    print("Format: text | tags | date")
    print("  e.g., 'They like cats | preference,personal | 2026-01-15'")
    print("  tags and date are optional, use | to separate")
    print(f"Default date: {date or 'now'} (override with --date or per-memory)")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    count = 0
    while True:
        try:
            line = input("Memory> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not line or line.lower() in ("quit", "exit", "q"):
            break

        parts = [p.strip() for p in line.split("|")]
        text = parts[0]
        mem_tags = parts[1] if len(parts) > 1 else ""
        mem_date = parts[2] if len(parts) > 2 else date

        mem_id = save_memory(db, text, tags=mem_tags, model=model, date=mem_date)
        count += 1
        emb_status = "with embedding" if model else "no embedding"
        print(f"  Saved: ID {mem_id} ({emb_status})")

    print(f"\nDone! Saved {count} memories to {db.db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Seed your companion's memory with facts, preferences, and shared history"
    )
    parser.add_argument("text", nargs="?", help="Memory text to save")
    parser.add_argument("--persona", "-p", help="Persona name (e.g. nova, valentine, kai)")
    parser.add_argument("--db", help="Direct path to a .db file (skips persona config resolution)")
    parser.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    parser.add_argument("--importance", type=int, default=5, help="Importance 1-10 (default: 5)")
    parser.add_argument("--type", default="fact", help="Memory type (default: fact)")
    parser.add_argument("--date", "-d", default="",
                        help="Date for the memory (e.g., '2026-02-14')")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for batch entry")
    parser.add_argument("--list-personas", action="store_true", help="List available personas and exit")
    args = parser.parse_args()

    if args.list_personas:
        personas_dir = PULSE_ROOT / "personas"
        available = sorted(d.name for d in personas_dir.iterdir()
                           if d.is_dir() and d.name != "_template")
        print("Available personas:")
        for name in available:
            db_path = resolve_db_path(name)
            shared = "(shared)" if "shared" in db_path.name else "(personal)"
            print(f"  {name:15s} -> {db_path} {shared}")
        return

    if not args.persona and not args.db:
        personas_dir = PULSE_ROOT / "personas"
        available = sorted(d.name for d in personas_dir.iterdir()
                           if d.is_dir() and d.name != "_template")
        print("Error: specify --persona or --db")
        print(f"Available personas: {', '.join(available)}")
        sys.exit(1)

    # Resolve target database
    if args.db:
        db_path = Path(args.db)
    else:
        db_path = resolve_db_path(args.persona)

    print(f"Target DB: {db_path}")

    # Open database
    from core.db import PulseDatabase
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = PulseDatabase(db_path)

    # Load embedding model
    print("Loading embedding model...", end=" ", flush=True)
    model = get_embedding_model()
    if model:
        print("ready!")
    else:
        print("not available (memories will save without embeddings)")

    if args.interactive or not args.text:
        interactive_mode(db, model, date=args.date)
    else:
        mem_id = save_memory(db, args.text, tags=args.tags, importance=args.importance,
                             memory_type=args.type, model=model, date=args.date)
        emb_status = "with embedding" if model else "no embedding"
        print(f"Saved: ID {mem_id} ({emb_status})")


if __name__ == "__main__":
    main()
