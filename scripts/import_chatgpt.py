"""
Stage 2: Import exported ChatGPT conversations into legacy.db.
Designed for streaming — processes files one at a time without loading
all conversations into memory.

Usage:
    python import_chatgpt.py [--db PATH] [--source NAME] [--drop]
    python import_chatgpt.py                    # uses default conversations/
    python import_chatgpt.py --drop            # recreate DB from scratch
"""

import argparse
import re
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.stdout.reconfigure(encoding='utf-8')

CONVERSATIONS_DIR = Path("path/to/your/conversations")
DEFAULT_DB = Path("path/to/your/legacy.db")
CHUNK_SIZE = 100  # messages per batch insert
USER = "YOUR USERNAME"
ASSISTANT = "your assistant's name"

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body. Returns (metadata, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    import yaml
    meta = yaml.safe_load(parts[1]) or {}
    return meta, parts[2]


def parse_messages(body: str) -> list[dict]:
    """Extract role + content from > quoted blocks in the markdown body.

    Each message block starts with '> **Role:**' and collects all content lines
    (with > prefix or not) until the next '> **:' block or end of body.
    """
    messages = []
    # Split body into blocks by '> **' marker
    blocks = re.split(r'^> \*\*([^*]+):\*\*', body, flags=re.MULTILINE)
    # blocks[0] = before first marker (skip)
    # blocks[1] = role of first block
    # blocks[2] = content of first block
    # blocks[3] = role of second block
    # blocks[4] = content of second block
    # etc.

    for i in range(1, len(blocks), 2):
        role_str = blocks[i].strip().lower()
        if role_str in (USER, "user", "you"):
            role = "user"
        elif role_str in (ASSISTANT,"assistant"):
            role = "assistant"
        else:
            continue  # skip unknown role markers

        content_block = blocks[i + 1] if i + 1 < len(blocks) else ""

        # Strip ALL leading > prefixes from each line (may be multiple like "> >text")
        lines = []
        for line in content_block.splitlines():
            while line.startswith("> "):
                line = line[2:]
            while line.startswith(">"):
                line = line[1:]
            lines.append(line)
        text = "\n".join(lines).strip()
        # Remove [Flash Thought: ...] blocks and standalone Flash Thought: markers
        text = re.sub(r'\[Flash Thought[^\]]*\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Flash Thought:\s*', '', text, flags=re.IGNORECASE)

        if text:
            messages.append({"role": role, "text": text})

    return messages


def init_db(db_path: Path, drop: bool):
    """Create or reset the legacy database schema."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if drop:
        conn.execute("DROP TABLE IF EXISTS legacy_messages")
        conn.execute("DROP TABLE IF EXISTS legacy_sessions")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS legacy_sessions (
            id              TEXT PRIMARY KEY,
            title           TEXT,
            source          TEXT NOT NULL DEFAULT 'chatgpt',
            conv_index      INTEGER,
            created_at      TEXT,
            message_count   INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS legacy_messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   TEXT,
            FOREIGN KEY (session_id) REFERENCES legacy_sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_lm_session ON legacy_messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_lm_role ON legacy_messages(role);

        CREATE VIRTUAL TABLE IF NOT EXISTS legacy_messages_fts
        USING fts5(content, content='legacy_messages', content_rowid='id');
        CREATE INDEX IF NOT EXISTS idx_lm_timestamp ON legacy_messages(timestamp DESC);
    """)
    conn.commit()
    return conn


def import_file(conn: sqlite3.Connection, filepath: Path, source: str) -> int:
    """Import a single .md file. Returns message count or -1 on error."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  ERROR reading {filepath.name}: {e}")
        return -1

    meta, body = parse_frontmatter(text)

    session_id = f"{source}:{meta.get('index', '?')}"
    title = meta.get("title", filepath.stem)
    conv_index = meta.get("index")
    date_str = meta.get("date")

    messages = parse_messages(body)

    conn.execute(
        "INSERT OR REPLACE INTO legacy_sessions (id, title, source, conv_index, created_at, message_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, title, source, conv_index, date_str, len(messages))
    )

    if messages:
        conn.executemany(
            "INSERT INTO legacy_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            [(session_id, m["role"], m["text"], date_str) for m in messages]
        )

    return meta, len(messages)


def main():
    parser = argparse.ArgumentParser(description="Import ChatGPT exports into legacy.db")
    parser.add_argument("--db", type=str, default=None,
                        help=f"Path to output .db (default: {DEFAULT_DB})")
    parser.add_argument("--source", type=str, default="chatgpt",
                        help="Source name prefix for session IDs (default: chatgpt)")
    parser.add_argument("--drop", action="store_true",
                        help="Drop existing tables and recreate from scratch")
    parser.add_argument("--dir", type=str, default=None,
                        help=f"Directory with exported .md files (default: {CONVERSATIONS_DIR})")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DEFAULT_DB
    conv_dir = Path(args.dir) if args.dir else CONVERSATIONS_DIR

    if not conv_dir.is_dir():
        print(f"ERROR: {conv_dir} is not a directory")
        sys.exit(1)

    conn = init_db(db_path, args.drop)

    md_files = sorted(conv_dir.glob("*.md"), key=lambda p: p.stem)
    if not md_files:
        print(f"No .md files found in {conv_dir}")
        return

    print(f"Importing {len(md_files)} files into {db_path}...")

    total_messages = 0
    errors = 0

    for filepath in md_files:
        # Wrap each file in a transaction
        try:
            meta, count = import_file(conn, filepath, args.source)
            if count >= 0:
                conn.commit()
                title = meta.get("title", filepath.stem) if meta else filepath.stem
                print(f"  [{filepath.stem[:20]:20}] {count:3d} msgs: {title[:50]}")
                total_messages += count
            else:
                errors += 1
        except Exception as e:
            conn.rollback()
            print(f"  ERROR {filepath.name}: {e}")
            errors += 1

    print(f"\nDone. {total_messages} messages imported, {errors} errors.")
    print(f"DB: {db_path}")
    print(f"  legacy_sessions: {conn.execute('SELECT COUNT(*) FROM legacy_sessions').fetchone()[0]} sessions")
    print(f"  legacy_messages: {conn.execute('SELECT COUNT(*) FROM legacy_messages').fetchone()[0]} messages")

    conn.close()


if __name__ == "__main__":
    main()