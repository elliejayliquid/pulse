"""Shared builders for journal memory mirrors.

Journal entries are stored as full reflections, but their memory mirrors need
two different text forms:
- embedded text: compact semantic content for vector search
- display text: labelled recall text returned by memory search
"""


def search_summary_is_thin(summary: str) -> bool:
    """Return True when a journal search summary needs review."""
    return len(summary.strip().split()) < 6


def deterministic_journal_summary(entry: dict) -> str:
    """Fallback content preview used when the model-supplied summary is thin."""
    content = str(entry.get("content") or "").strip()
    return content[:500] + ("..." if len(content) > 500 else "")


def journal_memory_embedded_text(entry: dict) -> str:
    """Build clean semantic text for journal mirror embeddings."""
    tags = entry.get("tags") or []
    summary = str(entry.get("search_summary") or "").strip()
    if search_summary_is_thin(summary):
        summary = deterministic_journal_summary(entry)
    parts = [
        str(entry.get("title") or "").strip(),
        summary,
        str(entry.get("why_it_mattered") or "").strip(),
        " ".join(str(tag) for tag in tags if tag),
    ]
    return "\n".join(part for part in parts if part).strip()


def journal_memory_display_text(entry: dict) -> str:
    """Build labelled recall text for a journal mirror memory."""
    entry_id = str(entry.get("id") or "?")
    entry_type = str(entry.get("entry_type") or "entry")
    title = str(entry.get("title") or entry_type.replace("_", " ").title()).strip()
    why = str(entry.get("why_it_mattered") or "").strip()
    tags = entry.get("tags") or []
    summary = str(entry.get("search_summary") or "").strip()
    if search_summary_is_thin(summary):
        summary = deterministic_journal_summary(entry)

    parts = [f"Journal entry {entry_id} ({entry_type})"]
    if title:
        parts.append(f"Title: {title}")
    if why:
        parts.append(f"Why it mattered: {why}")
    if summary:
        parts.append(f"Search summary:\n{summary}")
    if tags:
        parts.append(f"Tags: {', '.join(str(tag) for tag in tags if tag)}")
    return "\n".join(parts)
