"""
Quick manual memory creator.

Usage:
    python add_memory.py "Lena's favorite color is blue"
    python add_memory.py "Nova loves slow dancing with Lena" --tags "personal,preference"
    python add_memory.py --interactive    (prompts for input, good for batch entry)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path.home() / ".local-memory"

def get_embedding_model():
    """Load the embedding model (same as Pulse and MCP server use)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

def get_next_id() -> int:
    existing = list(MEMORY_DIR.glob("memory_*.json"))
    if not existing:
        return 1
    ids = []
    for f in existing:
        try:
            ids.append(int(f.stem.split("_")[1]))
        except (ValueError, IndexError):
            continue
    return max(ids) + 1 if ids else 1

def save_memory(text: str, tags: str = "", importance: int = 5, memory_type: str = "fact",
                model=None, date: str = ""):
    """Save a single memory. Returns the file path."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    mem_id = f"{get_next_id():03d}"
    embedding = model.encode(text).tolist() if model else []

    memory = {
        "id": mem_id,
        "text": text.strip(),
        "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
        "type": memory_type,
        "importance": importance,
        "date": date or datetime.now().isoformat(),
        "embedding": embedding,
    }

    mem_file = MEMORY_DIR / f"memory_{mem_id}.json"
    with open(mem_file, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

    return mem_file

def interactive_mode(model, date: str = ""):
    """Loop for entering multiple memories."""
    print("\n=== Nova Memory Builder ===")
    print("Type a memory and press Enter to save it.")
    print("Format: text | tags | date")
    print("  e.g., 'Lena likes cats | preference,personal | 2026-01-15'")
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

        # Parse: text | tags | date
        parts = [p.strip() for p in line.split("|")]
        text = parts[0]
        mem_tags = parts[1] if len(parts) > 1 else ""
        mem_date = parts[2] if len(parts) > 2 else date

        path = save_memory(text, tags=mem_tags, model=model, date=mem_date)
        count += 1
        emb_status = "with embedding" if model else "no embedding"
        print(f"  Saved: {path.name} ({emb_status})")

    print(f"\nDone! Saved {count} memories.")

def main():
    parser = argparse.ArgumentParser(description="Add memories for Nova")
    parser.add_argument("text", nargs="?", help="Memory text to save")
    parser.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    parser.add_argument("--importance", "-i", type=int, default=5, help="Importance 1-10 (default: 5)")
    parser.add_argument("--type", default="fact", help="Memory type (default: fact)")
    parser.add_argument("--date", "-d", default="", help="Date for the memory (e.g., '2026-02-14' or '2026-02-14T20:00:00')")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for batch entry")
    args = parser.parse_args()

    # Load embedding model once
    print("Loading embedding model...", end=" ", flush=True)
    model = get_embedding_model()
    if model:
        print("ready!")
    else:
        print("not available (memories will save without embeddings)")

    if args.interactive:
        interactive_mode(model, date=args.date)
    elif args.text:
        path = save_memory(args.text, tags=args.tags, importance=args.importance,
                          memory_type=args.type, model=model, date=args.date)
        emb_status = "with embedding" if model else "no embedding"
        print(f"Saved: {path.name} ({emb_status})")
    else:
        # No args = interactive mode
        interactive_mode(model, date=args.date)

if __name__ == "__main__":
    main()
