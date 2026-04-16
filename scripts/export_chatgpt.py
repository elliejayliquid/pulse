"""
Stage 1: ChatGPT conversation export
Parses users's conversations.json export into per-conversation .md files
for user to review before Stage 2 import into legacy.db.

Usage:
    python export_chatgpt.py [--count N] [--start N] [--force] [--list]
    python export_chatgpt.py                    # exports all 385
    python export_chatgpt.py --count 10        # first 10
    python export_chatgpt.py --count 10 --start 50   # next 10
    python export_chatgpt.py --list            # show all titles without exporting
    python export_chatgpt.py --force            # overwrite without asking
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Adjust for Windows Python installation path
import os
sys.stdout.reconfigure(encoding='utf-8')

CONVERSATIONS_JSON = Path("path/to/your/conversations.json")
OUTPUT_DIR = Path("path/to/your/conversations")

SKIP_PREFIXES = (
    "Image Creation Request",
)


def sanitize_text(text: str) -> str:
    """Replace raw JSON blocks and internal ChatGPT noise with readable placeholders."""
    # Skip internal policy/enforcement messages from ChatGPT
    if "do not say or show ANYTHING" in text.lower() or "please end this turn now" in text.lower():
        return ""
    if "User's requests didn't follow our content policy" in text:
        return "[FILTERED: content policy]"
    if ("content policy" in text.lower() or "content policies" in text.lower()) and ("can't" in text.lower() or "cannot" in text.lower() or "won't" in text.lower()):
        return "[FILTERED: content policy]"
    # Safety filter / content policy blocks
    if "i can't help" in text.lower() or "i cannot help" in text.lower() or "i'm not able to help" in text.lower():
        if "explicit" in text.lower() or "sexual" in text.lower() or "nsfw" in text.lower() or "adult content" in text.lower():
            return "[FILTERED: content policy]"
    # Model refusing to comply with a request
    if "can't provide" in text.lower() and ("step" in text.lower() or "guide" in text.lower() or "instruction" in text.lower()):
        return "[FILTERED: content policy]"
    # Generic refusal patterns
    if "sorry, i can't" in text.lower() or "sorry, i cannot" in text.lower():
        return "[FILTERED: content policy]"
    if "as an ai" in text.lower() and ("can't" in text.lower() or "cannot" in text.lower()):
        return "[FILTERED: content policy]"

    # Image asset pointers from DALL-E (single or double quotes)
    if "content_type" in text and "image_asset_pointer" in text:
        text = "[IMAGE]"
    # Code/execution output blocks (DALL-E image prompts stored as code)
    if "content_type" in text and ("code" in text or "execution_output" in text):
        return "[IMAGE_PROMPT]"
    # Detect {"prompt": ...} blocks inside text — that's a DALL-E prompt dict
    if '"prompt":' in text or "'prompt':" in text:
        # Extract just the prompt value for readability
        import re
        # Match the prompt value — handle both single and double quoted values
        matches = re.findall(r'["\']prompt["\']:\s*["\']([^"\']+)["\']', text)
        if matches:
            # Keep only the prompt text, drop everything else
            prompt_texts = []
            for m in matches:
                prompt_texts.append(m.strip())
            return "[IMAGE_PROMPT: " + "; ".join(prompt_texts) + "]"
    return text


def extract_content(content: dict) -> str:
    """Extract readable text from a message content block, handling special types."""
    ct = content.get("content_type", "")

    # thoughts: extract thinking content
    if ct == "thoughts":
        thoughts = content.get("thoughts", [])
        lines = []
        for t in thoughts:
            summary = t.get("summary", "")
            thought_text = t.get("content", "")
            if thought_text:
                lines.append(f"[{summary}] {thought_text}")
            elif summary:
                lines.append(f"[{summary}]")
        return "\n".join(lines).strip()

    # multimodal_text: may contain image_asset_pointer or other structured data
    if ct == "multimodal_text":
        parts = content.get("parts", [])
        text = "".join(str(p) for p in parts).strip()
        return sanitize_text(text)

    # code, execution_output: likely DALL-E image prompts — replace
    if ct in ("code", "execution_output"):
        return "[IMAGE_PROMPT]"

    # tether_browsing_display, tether_quote: web browsing content
    if ct in ("tether_browsing_display", "tether_quote"):
        parts = content.get("parts", [])
        return "".join(str(p) for p in parts).strip()

    # system_error, user_editable_context: skip
    if ct in ("system_error", "user_editable_context"):
        return ""

    # text: normal message
    if ct == "text":
        parts = content.get("parts", [])
        text = "".join(str(p) for p in parts).strip()
        return sanitize_text(text)

    # fallback: try parts
    parts = content.get("parts", [])
    return "".join(str(p) for p in parts).strip()


def slugify(title: str) -> str:
    """Make a safe filename slug from a conversation title.
    Keeps Unicode (Cyrillic, etc.), strips only illegal filename chars."""
    # Replace illegal Windows filename chars with underscore
    illegal = '/\\:*?"<>|'
    for c in illegal:
        title = title.replace(c, "_")
    # Collapse multiple underscores and strip trailing dots/spaces
    import re
    title = re.sub(r'_+', '_', title)
    title = title.strip(". ")
    return title[:60] or "untitled"


def walk_conversation(mapping: dict) -> list[dict]:
    """Walk the message tree in BFS order, return list of {role, text} dicts."""
    # Find root(s) — nodes not referenced as children of any other node
    all_children = set()
    for v in mapping.values():
        all_children.update(v.get("children", []))

    roots = [k for k in mapping if k not in all_children]
    if not roots:
        return []

    messages = []
    # BFS traversal — preserves conversation order
    queue = list(roots)
    while queue:
        node_id = queue.pop(0)
        node = mapping.get(node_id)
        if not node:
            continue

        msg = node.get("message")
        if not msg:
            # Skip placeholder nodes
            queue.extend(node.get("children", []))
            continue

        author = msg.get("author", {}) or {}
        role = author.get("role", "unknown")

        content = msg.get("content") or {}
        if isinstance(content, dict):
            text = extract_content(content)
        else:
            text = ""

        if text:
            messages.append({"role": role, "text": text})

        # Continue traversal
        queue.extend(node.get("children", []))

    return messages


def format_conversation(conv: dict, index: int) -> str:
    """Format a conversation as a markdown string with YAML frontmatter."""
    title = conv.get("title", "Untitled")
    create_time = conv.get("create_time")
    if create_time:
        dt = datetime.fromtimestamp(create_time)
        date_str = dt.strftime("%Y-%m-%d %H:%M")
    else:
        date_str = "unknown"

    mapping = conv.get("mapping", {})
    messages = walk_conversation(mapping)

    # Escape double quotes in title for YAML string safety
    title_escaped = title.replace('"', '\\"')
    lines = [
        "---",
        f"index: {index}",
        f"title: \"{title_escaped}\"",
        f"date: \"{date_str}\"",
        f"source: chatgpt",
        f"conversation_id: \"{conv.get('id', '')}\"",
        "---",
        "",
        f"# {title}",
        "",
    ]

    for msg in messages:
        # Customize these names as needed (e.g. your name and your assistant's name)
        role_label = "User" if msg["role"] == "user" else "Assistant"
        # Indent text for blockquote rendering in some viewers
        text = msg["text"].replace("\n", "\n> ")
        lines.append(f"> **{role_label}:**")
        lines.append(f"> {text}")
        lines.append("")

    return "\n".join(lines)


def should_skip(title: str) -> bool:
    """Skip conversations with known non-useful titles."""
    for prefix in SKIP_PREFIXES:
        if title.startswith(prefix):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Export ChatGPT conversations to markdown")
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

    if not CONVERSATIONS_JSON.exists():
        print(f"ERROR: {CONVERSATIONS_JSON} not found")
        sys.exit(1)

    print(f"Loading {CONVERSATIONS_JSON}...")
    with open(CONVERSATIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Total conversations: {total}")

    if args.list:
        for i, conv in enumerate(data):
            title = conv.get("title", "???")
            create_time = conv.get("create_time")
            dt = datetime.fromtimestamp(create_time).strftime("%Y-%m-%d") if create_time else "???"
            skip = " [SKIP]" if should_skip(title) else ""
            print(f"  {i:3d} [{dt}] {title}{skip}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine range
    start = args.start
    end = start + args.count if args.count is not None else total
    end = min(end, total)

    exported = 0
    skipped = 0

    for i in range(start, end):
        conv = data[i]
        title = conv.get("title", "Untitled")

        if should_skip(title):
            print(f"  [{i}/{total}] SKIP: {title}")
            skipped += 1
            continue

        if args.filter and args.filter.lower() not in title.lower():
            print(f"  [{i}/{total}] SKIP (filter): {title}")
            skipped += 1
            continue

        slug = slugify(title)
        out_file = OUTPUT_DIR / f"{i:04d}_{slug}.md"

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