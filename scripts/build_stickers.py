"""
Build stickers.db from YAML pack definitions in stickers/packs/.

Usage:
    python scripts/build_stickers.py                    # rebuild all packs
    python scripts/build_stickers.py cherry             # rebuild one pack
    python scripts/build_stickers.py --list             # list packs in DB

Reads YAML files from stickers/packs/, computes embeddings for each sticker's
keywords + description, and upserts into stickers/stickers.db.  The DB is
committed to Git so users without an embedding model can still use stickers.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PREVIEW_SIZE = 128

ROOT = Path(__file__).resolve().parent.parent
STICKERS_DIR = ROOT / "stickers"
PACKS_DIR = STICKERS_DIR / "packs"
DB_PATH = STICKERS_DIR / "stickers.db"


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS packs (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            title TEXT,
            added_at TEXT
        );
        CREATE TABLE IF NOT EXISTS stickers (
            id INTEGER PRIMARY KEY,
            pack_id INTEGER REFERENCES packs(id),
            file_id TEXT NOT NULL,
            keywords TEXT,
            description TEXT,
            embedding BLOB,
            image_path TEXT,
            added_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_stickers_pack ON stickers(pack_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_stickers_file_id ON stickers(file_id);
    """)


def _load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  Embedding model loaded: all-MiniLM-L6-v2")
        return model
    except ImportError:
        print("  WARNING: sentence-transformers not installed, skipping embeddings")
        return None


def _resize_preview(src: Path, previews_dir: Path) -> Path:
    """Resize a sticker image to PREVIEW_SIZE x PREVIEW_SIZE PNG for vision models.

    Skips resize if the image is already at or below PREVIEW_SIZE.
    """
    img = Image.open(src)
    if img.width <= PREVIEW_SIZE and img.height <= PREVIEW_SIZE:
        return src
    out = previews_dir / f"{src.stem}_preview.png"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    if img.mode not in ("RGBA",):
        img = img.convert("RGBA")
    img = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
    img.save(out, "PNG", optimize=True)
    print(f"    Resized {src.name} -> {out.name} ({PREVIEW_SIZE}x{PREVIEW_SIZE})")
    return out


def _embed(model, text: str) -> bytes | None:
    if model is None:
        return None
    vec = model.encode(text)
    return np.array(vec, dtype=np.float32).tobytes()


def _upsert_pack(conn: sqlite3.Connection, pack_data: dict, model) -> int:
    pack_name = pack_data["pack"]
    pack_title = pack_data.get("title", pack_name)
    now = datetime.now(timezone.utc).isoformat()

    row = conn.execute("SELECT id FROM packs WHERE name = ?", (pack_name,)).fetchone()
    if row:
        pack_id = row[0]
        conn.execute("UPDATE packs SET title = ? WHERE id = ?", (pack_title, pack_id))
    else:
        cur = conn.execute(
            "INSERT INTO packs (name, title, added_at) VALUES (?, ?, ?)",
            (pack_name, pack_title, now),
        )
        pack_id = cur.lastrowid

    previews_dir = STICKERS_DIR / "previews" / pack_name
    stickers = pack_data.get("stickers", [])
    added, updated = 0, 0

    for s in stickers:
        file_id = s["file_id"]
        keywords = s.get("keywords", "")
        description = s.get("description", "")
        embed_text = f"{keywords}. {description}" if description else keywords
        embedding = _embed(model, embed_text)

        image_path = None
        if s.get("image"):
            img = previews_dir / s["image"]
            if img.exists():
                resized = _resize_preview(img, previews_dir)
                image_path = str(resized.relative_to(ROOT))
            else:
                print(f"    WARNING: image not found: {img}")

        existing = conn.execute("SELECT id FROM stickers WHERE file_id = ?", (file_id,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE stickers SET pack_id=?, keywords=?, description=?, embedding=?, image_path=? WHERE id=?",
                (pack_id, keywords, description, embedding, image_path, existing[0]),
            )
            updated += 1
        else:
            conn.execute(
                "INSERT INTO stickers (pack_id, file_id, keywords, description, embedding, image_path, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pack_id, file_id, keywords, description, embedding, image_path, now),
            )
            added += 1

    print(f"  Pack '{pack_name}': {added} added, {updated} updated ({len(stickers)} total)")
    return pack_id


def build(pack_filter: str | None = None):
    if not PACKS_DIR.exists():
        print(f"No packs directory at {PACKS_DIR}")
        return

    yamls = sorted(PACKS_DIR.glob("*.yaml")) + sorted(PACKS_DIR.glob("*.yml"))
    if pack_filter:
        yamls = [y for y in yamls if y.stem == pack_filter]
        if not yamls:
            print(f"No YAML found for pack '{pack_filter}' in {PACKS_DIR}")
            return

    if not yamls:
        print(f"No YAML pack files found in {PACKS_DIR}")
        return

    print(f"Building sticker DB: {DB_PATH}")
    print(f"  Found {len(yamls)} pack file(s)")

    model = _load_embedding_model()

    conn = sqlite3.connect(str(DB_PATH))
    _init_db(conn)

    for yaml_path in yamls:
        print(f"\n  Processing {yaml_path.name}...")
        with open(yaml_path, "r", encoding="utf-8") as f:
            pack_data = yaml.safe_load(f)
        _upsert_pack(conn, pack_data, model)

    conn.commit()
    conn.close()

    total = sqlite3.connect(str(DB_PATH)).execute("SELECT COUNT(*) FROM stickers").fetchone()[0]
    print(f"\nDone! {total} stickers in DB.")


def list_packs():
    if not DB_PATH.exists():
        print("No stickers.db found. Run build first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT p.name, p.title, COUNT(s.id) "
        "FROM packs p LEFT JOIN stickers s ON s.pack_id = p.id "
        "GROUP BY p.id ORDER BY p.name"
    ).fetchall()
    conn.close()

    if not rows:
        print("No packs in DB.")
        return

    print(f"{'Pack':<20} {'Title':<30} {'Stickers':>8}")
    print("-" * 60)
    for name, title, count in rows:
        print(f"{name:<20} {title:<30} {count:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build stickers.db from YAML packs")
    parser.add_argument("pack", nargs="?", help="Rebuild a specific pack (by filename stem)")
    parser.add_argument("--list", action="store_true", help="List packs currently in DB")
    args = parser.parse_args()

    if args.list:
        list_packs()
    else:
        build(args.pack)
