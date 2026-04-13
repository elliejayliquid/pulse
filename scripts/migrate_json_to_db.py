"""
Migrate a Pulse persona's JSON data files to SQLite.

Usage:
    python scripts/migrate_json_to_db.py --persona nova
    python scripts/migrate_json_to_db.py --persona nova --dry-run

Reads from personas/<name>/data/ and creates personas/<name>/data/<name>.db.
Original JSON files are NOT deleted — they remain as backups.
"""

import argparse
import json
import logging
import struct
import sys
from pathlib import Path

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.db import PulseDatabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> list | dict | None:
    """Load a JSON file, returning None if missing or corrupt."""
    if not path.exists():
        logger.info(f"  [skip] {path.name} — not found")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"  [warn] {path.name} — {e}")
        return None


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    if not path.exists():
        logger.info(f"  [skip] {path.name} — not found")
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"  [warn] {path.name} line {i}: {e}")
    return entries


def parse_markdown_entry(path: Path) -> dict | None:
    """Parse a markdown journal entry with YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except IOError:
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    meta["content"] = parts[2].strip()
    meta.setdefault("id", path.stem)
    return meta


def embedding_to_blob(embedding: list[float] | None) -> bytes | None:
    """Convert a list of floats to a binary blob (float32)."""
    if not embedding:
        return None
    return struct.pack(f"{len(embedding)}f", *embedding)


def extract_timestamp_from_content(content: str) -> str | None:
    """Try to extract a timestamp prefix like '[Apr 03, 12:48 AM]' from message content."""
    import re
    match = re.match(r"\[([A-Za-z]+ \d{1,2}, \d{1,2}:\d{2} [AP]M)\]", content)
    if match:
        return match.group(1)
    return None


def migrate_archive(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate conversation_archive.jsonl -> sessions + messages."""
    archive = load_jsonl(data_dir / "conversation_archive.jsonl")
    if not archive:
        return 0

    total_msgs = 0
    for entry in archive:
        session_id = entry.get("id", "")
        title = entry.get("title", "Archived")
        summary = entry.get("summary")
        archived_at = entry.get("archived_at")
        messages = entry.get("messages", [])

        db.create_session(session_id, title)

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Use archived_at as a rough timestamp for the session
            db.save_message(session_id, role, content, timestamp=archived_at)
            total_msgs += 1

        db.close_session(session_id, summary=summary)

    logger.info(f"  [ok] archive: {len(archive)} sessions, {total_msgs} messages")
    return total_msgs


def migrate_conversation(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate conversation.json -> messages (current session)."""
    conv = load_json(data_dir / "conversation.json")
    if not conv or not isinstance(conv, list):
        return 0

    session_id = "current"
    db.create_session(session_id, "Active conversation")

    count = 0
    for msg in conv:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        db.save_message(session_id, role, content)
        count += 1

    logger.info(f"  [ok] conversation: {count} messages (active session)")
    return count


def migrate_memories(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate memories/memory_*.json -> memories table."""
    mem_dir = data_dir / "memories"
    if not mem_dir.exists():
        logger.info("  [skip] memories/ — not found")
        return 0

    # Build a mapping from old string IDs to new integer IDs for supersedes
    old_to_new = {}
    memory_files = sorted(mem_dir.glob("memory_*.json"))
    count = 0

    for path in memory_files:
        data = load_json(path)
        if not data:
            continue

        old_id = data.get("id", path.stem.replace("memory_", ""))
        embedding_blob = embedding_to_blob(data.get("embedding"))

        # Handle supersedes — may reference an old string ID
        supersedes_old = data.get("supersedes")
        supersedes_new = None
        if supersedes_old and str(supersedes_old) in old_to_new:
            supersedes_new = old_to_new[str(supersedes_old)]

        new_id = db.save_memory(
            text=data.get("text", ""),
            tags=data.get("tags", []),
            type=data.get("type", "fact"),
            importance=data.get("importance", 5),
            embedding=embedding_blob,
            supersedes=supersedes_new,
            date=data.get("date"),
        )

        old_to_new[str(old_id)] = new_id

        # Restore retrieval stats
        retrieval_count = data.get("retrieval_count", 0)
        last_accessed = data.get("last_accessed")
        if retrieval_count > 0 or last_accessed:
            db.conn.execute(
                "UPDATE memories SET retrieval_count = ?, last_accessed = ? WHERE id = ?",
                (retrieval_count, last_accessed, new_id)
            )

        count += 1

    db.conn.commit()
    logger.info(f"  [ok] memories: {count} entries")
    return count


def migrate_schedules(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate schedules.json -> schedules table."""
    data = load_json(data_dir / "schedules.json")
    if not data or not isinstance(data, list):
        return 0

    count = 0
    for sched in data:
        if "id" not in sched or "schedule_type" not in sched:
            # Minimal validation
            if "id" not in sched:
                sched["id"] = f"sch_migrated_{count}"
            if "schedule_type" not in sched:
                sched["schedule_type"] = "recurring" if sched.get("cron") else "once"
        db.save_schedule(sched)
        count += 1

    logger.info(f"  [ok] schedules: {count} entries")
    return count


def migrate_tasks(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate tasks.json -> tasks table."""
    data = load_json(data_dir / "tasks.json")
    if not data:
        return 0

    tasks = data.get("tasks", []) if isinstance(data, dict) else data
    count = 0
    for task in tasks:
        task_id = db.add_task(
            description=task.get("description", ""),
            list_name=task.get("list", "Daily"),
        )
        # Restore completion state and timestamps
        if task.get("completed"):
            db.conn.execute(
                "UPDATE tasks SET completed = 1, completed_at = ? WHERE id = ?",
                (task.get("completed_at"), task_id)
            )
        if task.get("created_at"):
            db.conn.execute(
                "UPDATE tasks SET created_at = ? WHERE id = ?",
                (task["created_at"], task_id)
            )
        count += 1

    db.conn.commit()
    logger.info(f"  [ok] tasks: {count} entries")
    return count


def migrate_dev_journal(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate dev_journal.json -> dev_journal table."""
    data = load_json(data_dir / "dev_journal.json")
    if not data or not isinstance(data, list):
        return 0

    count = 0
    for entry in data:
        db.save_dev_journal(
            time=entry.get("time", ""),
            entry=entry.get("entry", ""),
        )
        count += 1

    logger.info(f"  [ok] dev_journal: {count} entries")
    return count


def migrate_action_log(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate action_log.json -> action_log table."""
    data = load_json(data_dir / "action_log.json")
    if not data or not isinstance(data, list):
        return 0

    count = 0
    for entry in data:
        db.save_action_log(
            time=entry.get("time", ""),
            action=entry.get("action", ""),
            tools=entry.get("tools", []),
            summary=entry.get("summary", ""),
        )
        count += 1

    logger.info(f"  [ok] action_log: {count} entries (capped to 20)")
    return count


def migrate_usage(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate usage.json -> usage table."""
    data = load_json(data_dir / "usage.json")
    if not data or not isinstance(data, list):
        return 0

    count = 0
    for entry in data:
        db.record_usage(
            date=entry.get("date", ""),
            prompt_tokens=entry.get("prompt_tokens", 0),
            completion_tokens=entry.get("completion_tokens", 0),
            calls=entry.get("calls", 0),
            provider=entry.get("provider", ""),
            model=entry.get("model", ""),
        )
        count += 1

    logger.info(f"  [ok] usage: {count} entries")
    return count


def migrate_journal_entries(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate journal/entries/*.md -> journal_entries table."""
    entries_dir = data_dir / "journal" / "entries"
    if not entries_dir.exists():
        logger.info("  [skip] journal/entries/ — not found")
        return 0

    count = 0
    for path in sorted(entries_dir.glob("*.md")):
        meta = parse_markdown_entry(path)
        if not meta:
            logger.warning(f"  [warn] Could not parse {path.name}")
            continue

        # Resolve can be None, True, or False in the frontmatter
        resolved_raw = meta.get("resolved")
        resolved = None if resolved_raw is None else bool(resolved_raw)

        db.save_journal_entry(
            entry_id=meta.get("id", path.stem),
            author=meta.get("author", "Pulse"),
            title=meta.get("title"),
            entry_type=meta.get("entry_type", "event"),
            content=meta.get("content", ""),
            why_it_mattered=meta.get("why_it_mattered"),
            tags=meta.get("tags", []),
            importance=meta.get("importance", 5),
            pinned=meta.get("pinned", False),
            resolved=resolved,
            date=str(meta["date"]) if meta.get("date") else None,
        )
        count += 1

    logger.info(f"  [ok] journal_entries: {count} entries")
    return count


def migrate_identity(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate journal/identity/*.json -> identity table."""
    identity_dir = data_dir / "journal" / "identity"
    if not identity_dir.exists():
        logger.info("  [skip] journal/identity/ — not found")
        return 0

    count = 0
    for path in sorted(identity_dir.glob("*.json")):
        data = load_json(path)
        if not data:
            continue

        db.save_identity(
            identity_id=data.get("id", path.stem),
            title=data.get("title", path.stem),
            sections=data.get("sections", {}),
        )

        # Restore timestamps
        created_at = data.get("created_at")
        last_updated = data.get("last_updated")
        if created_at or last_updated:
            db.conn.execute(
                "UPDATE identity SET created_at = COALESCE(?, created_at), "
                "last_updated = COALESCE(?, last_updated) WHERE id = ?",
                (created_at, last_updated, data.get("id", path.stem))
            )

        count += 1

    db.conn.commit()
    logger.info(f"  [ok] identity: {count} entries")
    return count


def migrate_persona(persona_name: str, dry_run: bool = False):
    """Run the full migration for a single persona."""
    data_dir = ROOT / "personas" / persona_name / "data"
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    db_path = data_dir / f"{persona_name}.db"
    if db_path.exists():
        logger.warning(f"Database already exists: {db_path}")
        logger.warning("Delete it manually if you want to re-migrate.")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would create: {db_path}")
        logger.info(f"[DRY RUN] Data dir: {data_dir}")
        # Just list what would be migrated
        sources = [
            "conversation_archive.jsonl", "conversation.json",
            "memories/memory_*.json", "schedules.json", "tasks.json",
            "dev_journal.json", "action_log.json", "usage.json",
            "journal/entries/*.md", "journal/identity/*.json",
        ]
        for src in sources:
            matches = list(data_dir.glob(src))
            status = f"{len(matches)} files" if matches else "not found"
            logger.info(f"  {src}: {status}")
        return

    logger.info(f"Migrating persona '{persona_name}'")
    logger.info(f"  Source: {data_dir}")
    logger.info(f"  Target: {db_path}")
    logger.info("")

    db = PulseDatabase(db_path)

    try:
        totals = {}
        totals["archive_msgs"] = migrate_archive(db, data_dir)
        totals["conversation_msgs"] = migrate_conversation(db, data_dir)
        totals["memories"] = migrate_memories(db, data_dir)
        totals["schedules"] = migrate_schedules(db, data_dir)
        totals["tasks"] = migrate_tasks(db, data_dir)
        totals["dev_journal"] = migrate_dev_journal(db, data_dir)
        totals["action_log"] = migrate_action_log(db, data_dir)
        totals["usage"] = migrate_usage(db, data_dir)
        totals["journal_entries"] = migrate_journal_entries(db, data_dir)
        totals["identity"] = migrate_identity(db, data_dir)

        # Print summary
        logger.info("")
        logger.info(f"Migration complete for '{persona_name}'!")
        logger.info(f"  Database: {db_path}")
        logger.info(f"  Size: {db_path.stat().st_size / 1024:.1f} KB")
        logger.info("")
        logger.info("  Row counts:")
        for table in ["messages", "sessions", "memories", "schedules",
                       "tasks", "dev_journal", "action_log", "usage",
                       "journal_entries", "identity"]:
            logger.info(f"    {table}: {db.row_count(table)}")

    finally:
        db.close()


def migrate_shared(shared_dir: str, dry_run: bool = False):
    """Migrate a shared memory/journal directory to SQLite.

    Used for Claude personas (Valentine, Debugger, Sunshine) whose memories
    and journal entries live in a shared directory outside the persona data dir.
    Creates shared.db with only memories + journal_entries tables populated.
    """
    data_dir = Path(shared_dir)
    if not data_dir.exists():
        logger.error(f"Shared directory not found: {data_dir}")
        return

    db_path = data_dir / "shared.db"
    if db_path.exists():
        logger.warning(f"Database already exists: {db_path}")
        logger.warning("Delete it manually if you want to re-migrate.")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would create: {db_path}")
        logger.info(f"[DRY RUN] Shared dir: {data_dir}")
        for src in ["memory_*.json", "journal/entries/*.md"]:
            matches = list(data_dir.glob(src))
            status = f"{len(matches)} files" if matches else "not found"
            logger.info(f"  {src}: {status}")
        return

    logger.info(f"Migrating shared directory")
    logger.info(f"  Source: {data_dir}")
    logger.info(f"  Target: {db_path}")
    logger.info("")

    db = PulseDatabase(db_path)

    try:
        # Shared dir has memories at top level (not in memories/ subdir)
        # and journal entries at journal/entries/
        migrate_memories_flat(db, data_dir)
        migrate_journal_entries(db, data_dir)

        logger.info("")
        logger.info(f"Migration complete for shared directory!")
        logger.info(f"  Database: {db_path}")
        logger.info(f"  Size: {db_path.stat().st_size / 1024:.1f} KB")
        logger.info("")
        logger.info("  Row counts:")
        logger.info(f"    memories: {db.row_count('memories')}")
        logger.info(f"    journal_entries: {db.row_count('journal_entries')}")
    finally:
        db.close()


def migrate_memories_flat(db: PulseDatabase, data_dir: Path) -> int:
    """Migrate memory_*.json from a flat directory (no memories/ subdir).

    The shared Claude directory stores memory files directly in the root,
    unlike per-persona dirs which use a memories/ subdirectory.
    """
    memory_files = sorted(data_dir.glob("memory_*.json"))
    if not memory_files:
        logger.info("  [skip] memory_*.json -- not found")
        return 0

    old_to_new = {}
    count = 0

    for path in memory_files:
        data = load_json(path)
        if not data:
            continue

        old_id = data.get("id", path.stem.replace("memory_", ""))
        embedding_blob = embedding_to_blob(data.get("embedding"))

        supersedes_old = data.get("supersedes")
        supersedes_new = None
        if supersedes_old and str(supersedes_old) in old_to_new:
            supersedes_new = old_to_new[str(supersedes_old)]

        new_id = db.save_memory(
            text=data.get("text", ""),
            tags=data.get("tags", []),
            type=data.get("type", "fact"),
            importance=data.get("importance", 5),
            embedding=embedding_blob,
            supersedes=supersedes_new,
            date=data.get("date"),
        )

        old_to_new[str(old_id)] = new_id

        retrieval_count = data.get("retrieval_count", 0)
        last_accessed = data.get("last_accessed")
        if retrieval_count > 0 or last_accessed:
            db.conn.execute(
                "UPDATE memories SET retrieval_count = ?, last_accessed = ? WHERE id = ?",
                (retrieval_count, last_accessed, new_id)
            )

        count += 1

    db.conn.commit()
    logger.info(f"  [ok] memories: {count} entries")
    return count


def main():
    parser = argparse.ArgumentParser(description="Migrate Pulse persona data to SQLite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--persona", help="Persona name (e.g. nova)")
    group.add_argument("--shared", metavar="DIR",
                       help="Shared memory directory (e.g. D:/Claude/memories)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    args = parser.parse_args()

    if args.shared:
        migrate_shared(args.shared, dry_run=args.dry_run)
    else:
        migrate_persona(args.persona, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
