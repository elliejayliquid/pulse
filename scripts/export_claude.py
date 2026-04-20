"""
Stage 1: Claude conversation export
Parses Claude's conversations.json export into per-conversation .md files
for user to review before Stage 2 import into legacy.db.

Usage:
    python scripts/export_claude.py [--count N] [--start N] [--force] [--list]
    python scripts/export_claude.py                    # exports all
    python scripts/export_claude.py --count 10        # first 10
    python scripts/export_claude.py --list            # show all titles without exporting
"""

import json
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

# Adjust for Windows Python installation path
sys.stdout.reconfigure(encoding='utf-8')

CONVERSATIONS_JSON = Path("path/to/your/conversations.json")
OUTPUT_DIR = Path("path/to/where/you/want/the/conversations")


def sanitize_text(text: str) -> str:
    """Remove Claude-internal 'unsupported block' noise and clean up artifacts."""
    # Remove the "unsupported block" pattern including its code block wrappers
    # We use a non-greedy dots to match everything between the backticks
    # if it contains the forbidden string.
    unsupported_pattern = r'```[\s\n]*This block is not supported on your current device yet\.?[\s\n]*```'
    text = re.sub(unsupported_pattern, '', text, flags=re.IGNORECASE)
    
    # Also clean up any loose instances of the text itself
    text = text.replace("This block is not supported on your current device yet.", "")
    
    # Strip any resulting triple-newlines or excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_message_text(msg: dict) -> str:
    """Extract readable text from a message, handling text and content blocks."""
    # Try the top-level 'text' field first
    text = msg.get('text', '').strip()
    
    # If text is empty/missing, fall back to joining 'content' blocks
    if not text:
        content_blocks = msg.get('content', [])
        parts = []
        for block in content_blocks:
            if block.get('type') == 'text' and block.get('text'):
                parts.append(block['text'].strip())
        text = "\n\n".join(parts)
        
    return sanitize_text(text)


def slugify(title: str) -> str:
    """Make a safe filename slug from a conversation title.
    Keeps Unicode (Cyrillic, etc.), strips only illegal filename chars."""
    # Replace illegal Windows filename chars with underscore
    illegal = '/\\:*?"<>|'
    for c in illegal:
        title = title.replace(c, "_")
    # Collapse multiple underscores and strip trailing dots/spaces
    title = re.sub(r'_+', '_', title)
    title = title.strip(". ")
    return title[:60] or "untitled"


def format_conversation(conv: dict, index: int) -> str:
    """Format a conversation as a markdown string with YAML frontmatter."""
    title = conv.get("name", "Untitled") or "Untitled"
    created_at = conv.get("created_at")
    
    if created_at:
        try:
            # Claude uses ISO 8601 strings like "2025-06-18T06:51:54.652762Z"
            # We strip the 'Z' or offset if present for fromisoformat
            dt_str = created_at.replace('Z', '+00:00')
            dt = datetime.fromisoformat(dt_str)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = "unknown"
    else:
        date_str = "unknown"

    messages = []
    for msg in conv.get('chat_messages', []):
        text = extract_message_text(msg)
        if text:
            sender = msg.get('sender', 'unknown')
            messages.append({"role": sender, "text": text})

    # Escape double quotes in title for YAML string safety
    title_escaped = title.replace('"', '\\"')
    lines = [
        "---",
        f"index: {index}",
        f"title: \"{title_escaped}\"",
        f"date: \"{date_str}\"",
        f"source: claude",
        f"conversation_id: \"{conv.get('uuid', '')}\"",
        "---",
        "",
        f"# {title}",
        "",
    ]

    for msg in messages:
        # Claude: human -> User, assistant -> Assistant
        role_label = "User" if msg["role"] == "human" else "Assistant"
        # Indent text for blockquote rendering
        text = msg["text"].replace("\n", "\n> ")
        lines.append(f"> **{role_label}:**")
        lines.append(f"> {text}")
        lines.append("")

    return "\n".join(lines)


def should_skip(conv: dict) -> bool:
    """Determine if a conversation should be skipped (e.g. empty/test chats)."""
    name = conv.get("name")
    messages = conv.get('chat_messages', [])
    
    # Skip if name is empty AND has < 2 messages
    if not name and len(messages) < 2:
        return True
        
    # Skip if all messages have empty text
    all_empty = True
    for msg in messages:
        if extract_message_text(msg):
            all_empty = False
            break
    if all_empty:
        return True
        
    return False


def main():
    parser = argparse.ArgumentParser(description="Export Claude conversations to markdown")
    parser.add_argument("--input", type=str, default=None,
                        help=f"Path to conversations.json (default: {CONVERSATIONS_JSON})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of conversations to export (default: all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (0-based, default: 0)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files without asking")
    parser.add_argument("--list", action="store_true",
                        help="List all conversation titles and exit")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only export titles containing this string (case-insensitive)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else CONVERSATIONS_JSON
    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total conversations: {total}")

    if args.list:
        for i, conv in enumerate(data):
            title = conv.get("name", "???") or "Untitled"
            created_at = conv.get("created_at")
            if created_at:
                try:
                    dt_str = created_at.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(dt_str).strftime("%Y-%m-%d")
                except Exception:
                    dt = "???"
            else:
                dt = "???"
            skip = " [SKIP]" if should_skip(conv) else ""
            print(f"  {i:3d} [{dt}] {title}{skip}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine range
    start = args.start
    end = start + args.count if args.count is not None else total
    end = min(end, total)

    exported = 0
    skipped = 0

    for i in range(start, end):
        conv = data[i]
        title = conv.get("name", "Untitled") or "Untitled"

        if should_skip(conv):
            print(f"  [{i}/{total}] SKIP: {title}")
            skipped += 1
            continue

        if args.filter and args.filter.lower() not in title.lower():
            print(f"  [{i}/{total}] SKIP (filter): {title}")
            skipped += 1
            continue

        slug = slugify(title)
        out_file = output_dir / f"{i:04d}_{slug}.md"

        if out_file.exists() and not args.force:
            response = input(f"  [{i}/{total}] {out_file.name} exists — overwrite? [y/n/q]: ").strip().lower()
            if response == "q":
                print("Quit.")
                break
            if response != "y":
                print(f"  [{i}/{total}] skipped")
                continue

        markdown = format_conversation(conv, i)

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"  [{i}/{total}] exported: {title[:60]}")
        exported += 1

    print(f"\nDone. Exported: {exported}, skipped: {skipped}")


if __name__ == "__main__":
    main()
