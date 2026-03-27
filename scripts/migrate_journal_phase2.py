#!/usr/bin/env python3
"""
Migrate journal from Phase 1 (JSON) to Phase 2 (markdown + companion memories).

What this does:
1. Moves pinned identity files (_self.json, _user.json, _relationship.json)
   from journal_dir/ to journal_dir/identity/
2. Converts entry_NNN.json files to entries/NNN.md (markdown + YAML frontmatter)
3. Creates companion memory_NNN.json in memory_dir for each entry
   (reuses existing embeddings — no re-encoding needed!)
4. Rebuilds memories.json aggregate index

Safe to run multiple times — skips already-migrated entries.

Usage:
    python scripts/migrate_journal_phase2.py [--config config.yaml]
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_next_memory_id(memory_dir: Path) -> int:
    existing = list(memory_dir.glob("memory_*.json"))
    if not existing:
        return 1
    ids = []
    for f in existing:
        try:
            ids.append(int(f.stem.split("_")[1]))
        except (ValueError, IndexError):
            continue
    return max(ids) + 1 if ids else 1


def migrate_pinned(journal_dir: Path, identity_dir: Path):
    """Move pinned identity files to identity/ subfolder."""
    pinned_ids = ("_self", "_user", "_relationship")
    moved = 0
    for pin_id in pinned_ids:
        src = journal_dir / f"{pin_id}.json"
        dst = identity_dir / f"{pin_id}.json"
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {pin_id}.json -> identity/{pin_id}.json")
            moved += 1
        elif dst.exists():
            print(f"  {pin_id}.json already in identity/ — skipped")
        else:
            print(f"  {pin_id}.json not found — skipped")
    return moved


def migrate_entries(journal_dir: Path, entries_dir: Path,
                    memory_dir: Path) -> int:
    """Convert entry_NNN.json to entries/NNN.md + companion memory."""
    json_entries = sorted(journal_dir.glob("entry_*.json"))
    if not json_entries:
        print("  No entry_*.json files found")
        return 0

    next_mem_id = get_next_memory_id(memory_dir)
    migrated = 0

    for json_path in json_entries:
        # Parse entry ID from filename
        try:
            entry_id = json_path.stem.split("_")[1]  # "entry_001" -> "001"
        except IndexError:
            print(f"  Skipping {json_path.name} — unexpected filename format")
            continue

        md_path = entries_dir / f"{entry_id}.md"
        if md_path.exists():
            print(f"  {entry_id}.md already exists — skipped")
            continue

        # Load JSON entry
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Failed to read {json_path.name}: {e}")
            continue

        content = entry.get("content", "")
        if not content.strip():
            print(f"  {json_path.name} is empty — skipped")
            continue

        # Build frontmatter
        # Use created_at as the date (original creation time)
        date = entry.get("created_at", datetime.now().isoformat())
        meta = {
            "date": date,
            "entry_type": entry.get("entry_type", "reflection"),
            "tags": entry.get("tags", []),
            "importance": 5,
            "resolved": entry.get("resolved"),
        }

        # Write markdown file
        frontmatter_str = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
        md_text = f"---\n{frontmatter_str}---\n\n{content.strip()}\n"
        try:
            md_path.write_text(md_text, encoding="utf-8")
        except IOError as e:
            print(f"  Failed to write {md_path.name}: {e}")
            continue

        # Create companion memory (reuse existing embedding!)
        embedding = entry.get("embedding", [])
        preview = content[:200] + ("..." if len(content) > 200 else "")
        mem_id = f"{next_mem_id:03d}"

        memory = {
            "id": mem_id,
            "text": f"Journal: {preview}",
            "tags": ["journal"] + entry.get("tags", []),
            "type": "journal",
            "importance": 5,
            "retrieval_count": 0,
            "last_accessed": None,
            "date": date,
            "embedding": embedding,
            "journal_file": f"entries/{entry_id}.md",
        }

        mem_file = memory_dir / f"memory_{mem_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
        except IOError as e:
            print(f"  Failed to write companion memory {mem_file.name}: {e}")
            continue

        next_mem_id += 1
        migrated += 1
        print(f"  {json_path.name} -> entries/{entry_id}.md + memory_{mem_id}.json")

    return migrated


def rebuild_aggregate(memory_dir: Path):
    """Rebuild memories.json aggregate index."""
    memories = []
    for filepath in memory_dir.glob("memory_*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                mem = json.load(f)
            memories.append({k: v for k, v in mem.items() if k != "embedding"})
        except (json.JSONDecodeError, IOError):
            continue
    memories.sort(key=lambda m: m.get("date", ""))
    agg_file = memory_dir / "memories.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2)
    print(f"  Rebuilt memories.json ({len(memories)} entries)")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not Path(config_path).exists():
        # Try from pulse root
        config_path = Path(__file__).parent.parent / "config.yaml"

    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    journal_dir = Path(config.get("paths", {}).get("journal", "data/journal"))
    memory_dir = Path(
        config.get("paths", {}).get("memories", str(Path.home() / ".local-memory"))
    )

    entries_dir = journal_dir / "entries"
    identity_dir = journal_dir / "identity"

    print(f"\nJournal dir: {journal_dir}")
    print(f"Memory dir:  {memory_dir}")
    print(f"Entries dir: {entries_dir}")
    print(f"Identity dir: {identity_dir}")

    # Create new directories
    entries_dir.mkdir(parents=True, exist_ok=True)
    identity_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Move pinned files
    print("\n--- Step 1: Migrate pinned identity files ---")
    moved = migrate_pinned(journal_dir, identity_dir)
    print(f"  Moved {moved} pinned file(s)")

    # Step 2: Convert entries
    print("\n--- Step 2: Convert journal entries to markdown ---")
    migrated = migrate_entries(journal_dir, entries_dir, memory_dir)
    print(f"  Migrated {migrated} entry/entries")

    # Step 3: Rebuild aggregate
    print("\n--- Step 3: Rebuild memory aggregate ---")
    rebuild_aggregate(memory_dir)

    # Step 4: Summary
    old_entries = list(journal_dir.glob("entry_*.json"))
    new_entries = list(entries_dir.glob("*.md"))
    print(f"\n--- Done! ---")
    print(f"  Old JSON entries remaining: {len(old_entries)} (safe to delete after verification)")
    print(f"  New markdown entries: {len(new_entries)}")
    print(f"  Pinned identity files in identity/: {len(list(identity_dir.glob('*.json')))}")

    if old_entries:
        print(f"\n  To clean up old entries after verifying everything works:")
        print(f"    del {journal_dir}\\entry_*.json")


if __name__ == "__main__":
    main()
