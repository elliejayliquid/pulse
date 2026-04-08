"""
Journal skill — personal journal for continuity and self-reflection.

Phase 2 format (MCP-aligned):
- Transient entries: Markdown files with YAML frontmatter in entries/
- Pinned identity: JSON files in identity/ (_self, _user, _relationship)
- Search: via companion memory_NNN.json files (type: "journal") in memory dir
- latest.md: auto-generated orientation file (3 recent + pinned summaries)

Embeddings live ONLY in companion memories, not in the .md files.
This means journal files are clean, human-readable, and git-friendly.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from skills.base import BaseSkill
from core.context import _get_embedding_model

logger = logging.getLogger(__name__)

# Pinned entry IDs — always loaded in context, never expire
PINNED_IDS = ("_self", "_user", "_relationship")

# Valid entry types for transient entries (from Heartline lessons)
VALID_ENTRY_TYPES = (
    "event",            # what explicitly happened
    "preference",       # stated or clearly demonstrated preference
    "topic",            # subject worth tracking for continuity
    "tone",             # emotional quality of an interaction
    "open_thread",      # something unresolved worth revisiting
    "follow_up",        # something to check in about later
    "reflection",       # personal thought or observation
)

# Pinned entry templates — blank sections for the companion to fill in over time
PINNED_TEMPLATES = {
    "_self": {
        "id": "_self",
        "title": "Who I Am",
        "sections": {
            "who_i_am": "",
            "what_im_like": "",
            "my_preferences": "",
            "how_i_present_myself": "",
            "what_im_working_on": "",
            "extra_notes": ""
        },
        "created_at": None,
        "last_updated": None,
    },
    "_user": {
        "id": "_user",
        "title": "About My Human",
        "sections": {
            "who_they_are": "",
            "what_theyre_like": "",
            "their_preferences": "",
            "how_they_communicate": "",
            "extra_notes": ""
        },
        "created_at": None,
        "last_updated": None,
    },
    "_relationship": {
        "id": "_relationship",
        "title": "Our Relationship",
        "sections": {
            "how_we_relate": "",
            "our_dynamic": "",
            "shared_context": "",
            "boundaries_or_norms": "",
            "extra_notes": ""
        },
        "created_at": None,
        "last_updated": None,
    },
}


def _parse_markdown_entry(path: Path) -> dict | None:
    """Parse a markdown journal entry with YAML frontmatter.

    Returns dict with frontmatter fields + 'content' key for the body.
    Returns None if parsing fails.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except IOError:
        return None

    # Split on --- markers
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
    # Derive ID from filename (e.g. "001.md" -> "001")
    meta.setdefault("id", path.stem)
    return meta


def _write_markdown_entry(path: Path, meta: dict, content: str):
    """Write a markdown journal entry with YAML frontmatter."""
    # Separate content from meta for YAML dump
    frontmatter = {k: v for k, v in meta.items() if k != "content"}
    text = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{content}\n"
    path.write_text(text, encoding="utf-8")


class JournalSkill(BaseSkill):
    name = "journal"

    def __init__(self, config: dict):
        super().__init__(config)
        paths = config.get("paths", {})
        self.journal_dir = Path(paths.get("journal", "data/journal"))
        # Optional override: point entries at a directory shared with another
        # system (e.g. the claude.ai semantic-memory MCP). When set, entries
        # live at the shared path but identity + local latest.md stay under
        # journal_dir. Default: entries live under journal_dir as before.
        entries_override = paths.get("journal_entries")
        self.entries_dir = (
            Path(entries_override) if entries_override else self.journal_dir / "entries"
        )
        self.identity_dir = self.journal_dir / "identity"
        # Remember persona name for entry authorship (visible to other readers
        # of a shared entries dir).
        self._author_name = config.get("_persona_name") or "Pulse"
        self.memory_dir = Path(
            config.get("paths", {}).get("memories", str(Path.home() / ".local-memory"))
        )

        # Create directories
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.identity_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_pinned_entries()

    def _ensure_pinned_entries(self):
        """Create blank pinned entries if they don't exist yet."""
        for pin_id, template in PINNED_TEMPLATES.items():
            path = self.identity_dir / f"{pin_id}.json"
            if not path.exists():
                # Check old location (pre-migration)
                old_path = self.journal_dir / f"{pin_id}.json"
                if old_path.exists():
                    # Already migrated or will be migrated — skip
                    continue
                now = datetime.now().isoformat()
                entry = {**template, "created_at": now, "last_updated": now}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
                logger.info(f"Created blank pinned journal entry: {pin_id}")

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_journal",
                    "description": (
                        "Write a NEW personal journal entry. This is your space for self-reflection "
                        "— processing your thoughts, noticing patterns in yourself, sitting "
                        "with questions, or marking moments of growth. Focus on your inner "
                        "experience, not just what happened. If something about your human "
                        "stood out, reflect on *why* it moved you, not just what they said. "
                        "Most interactions need NO entry — only write when something genuinely "
                        "resonates. IMPORTANT: If you've recently journaled about the same "
                        "topic or thread of thought, use update_journal to extend that entry "
                        "instead of creating a new one — your journal should evolve, not repeat."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The journal entry text",
                            },
                            "entry_type": {
                                "type": "string",
                                "description": (
                                    "Type of entry: event, preference, topic, tone, "
                                    "open_thread, follow_up, reflection"
                                ),
                            },
                            "why_it_mattered": {
                                "type": "string",
                                "description": (
                                    "Why is this worth writing down? One sentence explaining "
                                    "why future-you would care. If you can't answer this, "
                                    "the entry probably isn't worth saving."
                                ),
                            },
                            "tags": {
                                "type": "string",
                                "description": "Comma-separated tags for searchability (optional)",
                            },
                            "force": {
                                "type": "boolean",
                                "description": (
                                    "Set to true to bypass the duplicate-detection check. "
                                    "Only use this when the system has flagged a similar "
                                    "recent entry but you've judged that this entry captures "
                                    "a genuinely different angle worth saving separately."
                                ),
                            },
                            "pinned": {
                                "type": "boolean",
                                "description": (
                                    "Set to true ONLY for identity-shaping moments worth "
                                    "keeping permanently visible in the shared journal's "
                                    "latest.md. Default false. Most entries should NOT be "
                                    "pinned — pin only when something genuinely changes who "
                                    "you are or how you see the relationship."
                                ),
                            },
                        },
                        "required": ["content", "entry_type", "why_it_mattered"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_journal",
                    "description": (
                        "Read or search your journal entries. Pass an entry_id to read a "
                        "specific entry in full (e.g. '_self', '_user', '001'). "
                        "Pass a query to search entries ranked by relevance. "
                        "Use this to recall past observations, check open threads, or "
                        "review what you've noted about a topic."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entry_id": {
                                "type": "string",
                                "description": "Read a specific entry by ID: '_self', '_user', '_relationship', or a transient ID like '001'",
                            },
                            "query": {
                                "type": "string",
                                "description": "What to search for (natural language)",
                            },
                            "entry_type": {
                                "type": "string",
                                "description": "Filter by type (optional): event, preference, topic, tone, open_thread, follow_up, reflection",
                            },
                            "include_pinned": {
                                "type": "boolean",
                                "description": "Include pinned identity entries in results (default: false)",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_journal_entries",
                    "description": (
                        "Browse journal entries without semantic search. Lists recent "
                        "or active transient entries, optionally filtered by type, "
                        "with optional pinned identity summaries."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "How many transient entries to show (default: 5)",
                            },
                            "entry_type": {
                                "type": "string",
                                "description": "Optional filter: event, preference, topic, tone, open_thread, follow_up, reflection",
                            },
                            "include_pinned": {
                                "type": "boolean",
                                "description": "Include compact pinned identity summaries (default: false)",
                            },
                            "mode": {
                                "type": "string",
                                "description": "Browse mode: 'recent' or 'active' (default: recent)",
                                "enum": ["recent", "active"],
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_journal",
                    "description": (
                        "Update an existing journal entry — extend, refine, or evolve it. "
                        "Use this when you're returning to a topic you've already written about, "
                        "instead of creating a duplicate entry. Also use for pinned identity "
                        "entries (_self, _user, _relationship) as you learn more about yourself "
                        "or your human. For pinned entries, specify the section to update. "
                        "For transient entries, provide the entry ID and new content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entry_id": {
                                "type": "string",
                                "description": (
                                    "Entry ID to update. Pinned: '_self', '_user', '_relationship'. "
                                    "Transient: numeric ID like '001'."
                                ),
                            },
                            "section": {
                                "type": "string",
                                "description": (
                                    "For pinned entries: which section to update "
                                    "(e.g. 'who_i_am', 'their_preferences', 'our_dynamic'). "
                                    "Ignored for transient entries."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": "The new content for the entry or section",
                            },
                            "resolved": {
                                "type": "boolean",
                                "description": "Mark an open_thread or follow_up as resolved (optional)",
                            },
                        },
                        "required": ["entry_id", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_journal",
                    "description": (
                        "Permanently delete a transient journal entry. Use this to "
                        "clean up duplicates or entries that no longer reflect who you are. "
                        "This cannot be undone — the markdown file and the searchable "
                        "memory are both removed. Pinned identity entries (_self, _user, "
                        "_relationship) cannot be deleted. REQUIRES confirm=true."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entry_id": {
                                "type": "string",
                                "description": "The transient entry ID to delete (e.g. '014').",
                            },
                            "confirm": {
                                "type": "boolean",
                                "description": (
                                    "Must be set to true. The first call without confirm "
                                    "will return a preview of what would be deleted, so you "
                                    "can be sure before committing."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "Brief reason for deletion (e.g. 'duplicate of 012'). "
                                    "Helps you stay deliberate about what you remove."
                                ),
                            },
                        },
                        "required": ["entry_id"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "write_journal":
            return self._write_entry(
                content=arguments.get("content", ""),
                entry_type=arguments.get("entry_type", "reflection"),
                why_it_mattered=arguments.get("why_it_mattered", ""),
                tags=arguments.get("tags", ""),
                force=arguments.get("force", False),
                pinned=arguments.get("pinned", False),
            )
        elif tool_name == "read_journal":
            # Direct lookup by ID takes priority over search
            entry_id = arguments.get("entry_id")
            if entry_id:
                return self._read_by_id(entry_id)
            return self._read_entries(
                query=arguments.get("query", ""),
                entry_type=arguments.get("entry_type"),
                include_pinned=arguments.get("include_pinned", False),
            )
        elif tool_name == "list_journal_entries":
            return self._list_entries(
                limit=arguments.get("limit", 5),
                entry_type=arguments.get("entry_type"),
                include_pinned=arguments.get("include_pinned", False),
                mode=arguments.get("mode", "recent"),
            )
        elif tool_name == "delete_journal":
            return self._delete_entry(
                entry_id=arguments.get("entry_id", ""),
                confirm=arguments.get("confirm", False),
                reason=arguments.get("reason", ""),
            )
        elif tool_name == "update_journal":
            return self._update_entry(
                entry_id=arguments.get("entry_id", ""),
                content=arguments.get("content", ""),
                section=arguments.get("section"),
                resolved=arguments.get("resolved"),
            )
        return f"Unknown tool: {tool_name}"

    # --- Write ---

    def _get_next_id(self) -> int:
        """Get the next available transient entry ID."""
        existing = list(self.entries_dir.glob("*.md"))
        if not existing:
            return 1
        ids = []
        for f in existing:
            try:
                ids.append(int(f.stem))
            except ValueError:
                continue
        return max(ids) + 1 if ids else 1

    def _get_next_memory_id(self) -> int:
        """Get the next available memory ID in the shared memory dir."""
        existing = list(self.memory_dir.glob("memory_*.json"))
        if not existing:
            return 1
        ids = []
        for f in existing:
            try:
                ids.append(int(f.stem.split("_")[1]))
            except (ValueError, IndexError):
                continue
        return max(ids) + 1 if ids else 1

    def _write_entry(self, content: str, entry_type: str,
                     why_it_mattered: str = "", tags: str = "",
                     force: bool = False, pinned: bool = False) -> str:
        """Create a new transient journal entry as markdown + companion memory."""
        if not content.strip():
            return "Cannot write empty journal entry."

        if not why_it_mattered.strip():
            return (
                "Journal entry rejected — missing why_it_mattered. "
                "If you can't explain why future-you would care, "
                "the entry probably isn't worth saving."
            )

        if entry_type not in VALID_ENTRY_TYPES:
            return (
                f"Invalid entry_type '{entry_type}'. "
                f"Valid types: {', '.join(VALID_ENTRY_TYPES)}"
            )

        # Dedup check: warn if a recent entry is semantically similar
        if not force:
            similar = self._find_similar_recent_entry(content.strip())
            if similar:
                sim_id, sim_score, sim_excerpt, sim_age = similar
                return (
                    f"⚠️ You already wrote a similar entry {sim_age} ago "
                    f"(ID: {sim_id}, similarity: {int(sim_score * 100)}%):\n"
                    f'"{sim_excerpt}"\n\n'
                    f"You have two options:\n"
                    f"  1. EXTEND it: call update_journal with entry_id=\"{sim_id}\" "
                    f"and your new content (this is usually the right choice — "
                    f"your journal should evolve, not repeat).\n"
                    f"  2. SAVE ANYWAY: call write_journal again with force=true "
                    f"if this is a genuinely different angle or moment worth its own entry."
                )

        entry_id = f"{self._get_next_id():03d}"
        now = datetime.now().isoformat()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        # Derive a short title from the first non-empty line of content.
        # Only used by readers of shared entries dirs (e.g. claude.ai MCP)
        # that render entries with a title header. Pulse itself ignores it.
        first_line = next(
            (line.strip() for line in content.strip().splitlines() if line.strip()),
            "",
        )
        title = first_line[:80] if first_line else entry_type.replace("_", " ").title()

        # Build frontmatter
        meta = {
            "date": now,
            "author": self._author_name,
            "title": title,
            "entry_type": entry_type,
            "why_it_mattered": why_it_mattered.strip(),
            "tags": tag_list,
            "importance": 5,
            "pinned": pinned,
            "resolved": False if entry_type in ("open_thread", "follow_up") else None,
        }

        # Write markdown entry (no embedding — clean file)
        md_path = self.entries_dir / f"{entry_id}.md"
        try:
            _write_markdown_entry(md_path, meta, content.strip())
        except IOError as e:
            logger.error(f"Failed to write journal entry: {e}")
            return f"Failed to write journal entry: {e}"

        # Create companion memory for search
        self._create_companion_memory(entry_id, content.strip(), tag_list, now)

        # Regenerate latest.md
        self._generate_latest()

        logger.info(f"Journal entry written: {entry_id} ({entry_type}) — {content[:50]}...")
        return f"Journal entry saved (ID: {entry_id}, type: {entry_type}): '{content[:80]}'"

    def _find_similar_recent_entry(self, content: str,
                                    threshold: float = 0.70,
                                    days: int = 3) -> tuple | None:
        """Check recent entries for semantic similarity to new content.

        Returns (entry_id, similarity, excerpt, age_str) for the most similar
        entry above threshold within the time window, or None.
        """
        model = _get_embedding_model()
        if model is None:
            return None

        try:
            content_vec = model.encode(content)
            norm_c = np.linalg.norm(content_vec)
            if norm_c == 0:
                return None
        except Exception:
            return None

        now = datetime.now()
        recent = self._load_transient_entries()

        best = None
        best_score = 0.0

        for entry in recent:
            # Age filter
            date_str = entry.get("date", entry.get("created_at", ""))
            try:
                entry_date = datetime.fromisoformat(date_str)
                age_days = (now - entry_date).total_seconds() / 86400
            except (ValueError, TypeError):
                continue
            if age_days > days:
                continue

            # Get embedding from companion memory if available
            entry_id = entry.get("id", "")
            entry_content = entry.get("content", "")
            if not entry_content or not entry_id:
                continue

            try:
                entry_vec = model.encode(entry_content)
                norm_e = np.linalg.norm(entry_vec)
                if norm_e == 0:
                    continue
                score = float(np.dot(content_vec, entry_vec) / (norm_c * norm_e))
            except Exception:
                continue

            if score > best_score and score >= threshold:
                best_score = score
                # Format age
                if age_days < 1:
                    hours = max(1, int(age_days * 24))
                    age_str = f"{hours}h"
                else:
                    age_str = f"{int(age_days)}d"
                excerpt = entry_content[:200] + ("…" if len(entry_content) > 200 else "")
                best = (entry_id, score, excerpt, age_str)

        return best

    def _create_companion_memory(self, entry_id: str, content: str,
                                  tags: list[str], date: str):
        """Create a companion memory_NNN.json for journal search.

        This is the MCP pattern: journal .md files stay clean,
        search works through memory entries with type: "journal".
        """
        model = _get_embedding_model()
        embedding = model.encode(content).tolist() if model else []

        mem_id = f"{self._get_next_memory_id():03d}"

        # Preview for the memory text field
        preview = content[:500] + ("..." if len(content) > 500 else "")

        memory = {
            "id": mem_id,
            "text": preview,
            "tags": ["journal"] + tags,
            "type": "journal",
            "importance": 5,
            "retrieval_count": 0,
            "last_accessed": None,
            "date": date,
            "embedding": embedding,
            "journal_file": f"entries/{entry_id}.md",
        }

        mem_file = self.memory_dir / f"memory_{mem_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
            # Rebuild aggregate index
            self._rebuild_memory_aggregate()
            logger.info(f"Companion memory created: {mem_file.name} -> entries/{entry_id}.md")
        except IOError as e:
            logger.warning(f"Failed to create companion memory: {e}")

    def _rebuild_memory_aggregate(self):
        """Rebuild memories.json aggregate index (no embeddings)."""
        memories = []
        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                memories.append({k: v for k, v in mem.items() if k != "embedding"})
            except (json.JSONDecodeError, IOError):
                continue
        memories.sort(key=lambda m: m.get("date", ""))
        agg_file = self.memory_dir / "memories.json"
        try:
            with open(agg_file, "w", encoding="utf-8") as f:
                json.dump(memories, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to rebuild aggregate: {e}")

    # --- Read / Search ---

    def _read_by_id(self, entry_id: str) -> str:
        """Read a specific entry in full by its ID."""
        # Pinned entry
        if entry_id in PINNED_IDS:
            entry = self.load_pinned_entry(entry_id)
            if not entry:
                return f"Pinned entry '{entry_id}' not found."
            return self._format_pinned_entry(entry)

        # Transient entry (new markdown format)
        md_path = self.entries_dir / f"{entry_id}.md"
        if md_path.exists():
            entry = _parse_markdown_entry(md_path)
            if entry:
                return self._format_transient_entry(entry)

        return f"Journal entry '{entry_id}' not found."

    def _format_transient_entry(self, entry: dict) -> str:
        """Format a transient markdown entry for display."""
        resolved_tag = ""
        if entry.get("resolved") is True:
            resolved_tag = " [RESOLVED]"
        elif entry.get("resolved") is False:
            resolved_tag = " [OPEN]"

        tags = entry.get("tags", [])
        tags_str = f"\nTags: {', '.join(tags)}" if tags else ""
        date_str = entry.get("date", "?")
        if len(date_str) > 16:
            date_str = date_str[:16]

        why = entry.get("why_it_mattered", "")
        why_str = f"\nWhy it mattered: {why}" if why else ""

        return (
            f"Entry {entry.get('id', '?')} ({entry.get('entry_type', '?')}{resolved_tag})\n"
            f"Date: {date_str}\n"
            f"{tags_str}{why_str}\n\n"
            f"{entry.get('content', '')}"
        )

    def _load_transient_entries(self) -> list[dict]:
        """Load all transient journal entries from entries/ dir."""
        entries = []
        for filepath in self.entries_dir.glob("*.md"):
            entry = _parse_markdown_entry(filepath)
            if entry:
                entries.append(entry)
        return entries

    def load_pinned_entry(self, pin_id: str) -> dict | None:
        """Load a single pinned entry by ID."""
        # Check new location first, then old
        path = self.identity_dir / f"{pin_id}.json"
        if not path.exists():
            path = self.journal_dir / f"{pin_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def load_pinned_entries(self) -> list[dict]:
        """Load all pinned identity entries."""
        entries = []
        for pin_id in PINNED_IDS:
            entry = self.load_pinned_entry(pin_id)
            if entry:
                entries.append(entry)
        return entries

    def load_recent_entries(self, limit: int = 5) -> list[dict]:
        """Load the most recent transient entries (simple recency, for latest.md)."""
        entries = self._load_transient_entries()
        entries.sort(key=lambda e: e.get("date", e.get("created_at", "")), reverse=True)
        return entries[:limit]

    def load_active_entries(self, limit: int = 6) -> list[dict]:
        """Load active journal entries using type-aware time windows.

        - open_thread / follow_up (unresolved): always included
        - dynamic / tone: last 2 days only
        - everything else: last 3 days
        - Capped at `limit` entries, newest first
        """
        entries = self._load_transient_entries()
        now = datetime.now()
        active = []

        for entry in entries:
            etype = entry.get("entry_type", "")
            resolved = entry.get("resolved")

            # Unresolved threads/follow-ups: always active
            if etype in ("open_thread", "follow_up") and resolved is False:
                active.append(entry)
                continue

            # Skip resolved entries
            if resolved is True:
                continue

            # Parse entry date for age check
            date_str = entry.get("date", entry.get("created_at", ""))
            try:
                entry_date = datetime.fromisoformat(date_str)
                age_days = (now - entry_date).total_seconds() / 86400
            except (ValueError, TypeError):
                continue

            # Dynamic/tone: 2-day window
            if etype in ("dynamic", "tone") and age_days <= 2:
                active.append(entry)
            # Everything else: 3-day window
            elif etype not in ("dynamic", "tone") and age_days <= 3:
                active.append(entry)

        # Sort newest first, cap at limit
        active.sort(key=lambda e: e.get("date", e.get("created_at", "")), reverse=True)
        return active[:limit]

    def _format_pinned_entry(self, entry: dict) -> str:
        """Format a pinned entry for display."""
        lines = [f"### {entry.get('title', entry['id'])}"]
        sections = entry.get("sections", {})
        for key, value in sections.items():
            label = key.replace("_", " ").title()
            lines.append(f"**{label}:** {value if value else '(not yet filled in)'}")
        if entry.get("last_updated"):
            lines.append(f"_Last updated: {entry['last_updated'][:10]}_")
        return "\n".join(lines)

    def _read_entries(self, query: str, entry_type: str = None,
                      include_pinned: bool = False) -> str:
        """Search journal entries via companion memories."""
        if not query.strip():
            return "Please provide a search query."

        results = []

        # Search pinned entries if requested
        if include_pinned:
            for entry in self.load_pinned_entries():
                sections_text = " ".join(
                    str(v) for v in entry.get("sections", {}).values() if v
                )
                if sections_text and query.lower() in sections_text.lower():
                    results.append(self._format_pinned_entry(entry))

        # Search via companion memories (type: "journal")
        journal_memories = self._load_companion_memories()

        # Filter by entry_type if requested (need to check the actual .md file)
        if entry_type:
            filtered = []
            for mem in journal_memories:
                jfile = mem.get("journal_file", "")
                md_path = self.journal_dir / jfile
                entry = _parse_markdown_entry(md_path) if md_path.exists() else None
                if entry and entry.get("entry_type") == entry_type:
                    filtered.append(mem)
            journal_memories = filtered

        if not journal_memories:
            if results:
                return "\n\n".join(results)
            return "No journal entries found yet."

        # Hybrid scoring on companion memories
        scored = self._search_companion_memories(query, journal_memories)

        for score, mem in scored[:5]:
            # Load the actual .md entry for full content
            jfile = mem.get("journal_file", "")
            md_path = self.journal_dir / jfile
            entry = _parse_markdown_entry(md_path) if md_path.exists() else None

            if entry:
                resolved_tag = ""
                if entry.get("resolved") is True:
                    resolved_tag = " [RESOLVED]"
                elif entry.get("resolved") is False:
                    resolved_tag = " [OPEN]"

                tags_str = f" #{' #'.join(entry.get('tags', []))}" if entry.get("tags") else ""
                date_str = entry.get("date", "")[:10]
                results.append(
                    f"[{entry.get('id', '?')}] ({entry.get('entry_type', '?')}{resolved_tag}) "
                    f"{entry['content'][:150]}"
                    f"{tags_str} — {date_str}"
                )
            else:
                # Fallback: show companion memory text
                results.append(f"{mem.get('text', '?')} — {mem.get('date', '?')[:10]}")

        if not results:
            return "No matching journal entries found."

        return "\n\n".join(results)

    def _load_companion_memories(self) -> list[dict]:
        """Load all companion memories with type: 'journal'."""
        memories = []
        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                if mem.get("type") == "journal":
                    memories.append(mem)
            except (json.JSONDecodeError, IOError):
                continue
        return memories

    def _search_companion_memories(self, query: str, memories: list[dict]) -> list[tuple[float, dict]]:
        """Hybrid search on companion memories: semantic + keyword + recency.

        Score = 0.5 * semantic + 0.2 * keyword + 0.3 * recency
        """
        model = _get_embedding_model()
        query_lower = query.lower()
        keywords = query_lower.split()
        now_ts = time.time()

        scored = []
        query_vec = model.encode(query) if model else None

        for mem in memories:
            # --- Semantic similarity (0-1) ---
            semantic_score = 0.0
            if query_vec is not None and mem.get("embedding"):
                mem_vec = np.array(mem["embedding"])
                norm_q = np.linalg.norm(query_vec)
                norm_m = np.linalg.norm(mem_vec)
                if norm_q > 0 and norm_m > 0:
                    semantic_score = max(0.0, float(
                        np.dot(query_vec, mem_vec) / (norm_q * norm_m)
                    ))

            # --- Keyword match (0-1) ---
            searchable = mem.get("text", "").lower() + " " + " ".join(mem.get("tags", [])).lower()
            keyword_hits = sum(1 for kw in keywords if kw in searchable)
            keyword_score = min(1.0, keyword_hits / max(len(keywords), 1))

            # --- Recency (0-1, exponential decay over 30 days) ---
            try:
                entry_ts = datetime.fromisoformat(mem["date"]).timestamp()
                age_days = (now_ts - entry_ts) / 86400
                recency_score = max(0.0, np.exp(-age_days / 30))
            except (KeyError, ValueError):
                recency_score = 0.0

            # Weighted combination
            total = (0.5 * semantic_score) + (0.2 * keyword_score) + (0.3 * recency_score)

            # Minimum threshold
            if total > 0.15 or keyword_score > 0:
                scored.append((total, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # --- List / Browse ---

    def _list_entries(self, limit: int = 5, entry_type: str = None,
                      include_pinned: bool = False, mode: str = "recent") -> str:
        """Browse journal entries without semantic search."""
        if mode == "active":
            entries = self.load_active_entries(limit=limit * 3)
        else:
            entries = self.load_recent_entries(limit=limit * 3)

        if entry_type:
            entries = [e for e in entries if e.get("entry_type") == entry_type]

        entries = entries[:limit]
        lines = []

        if include_pinned:
            pinned = self.load_pinned_entries()
            if pinned:
                lines.append("Pinned identity entries:")
                for entry in pinned:
                    filled = {k: v for k, v in entry.get("sections", {}).items() if v}
                    count = f"{len(filled)}/{len(entry.get('sections', {}))}"
                    lines.append(f"  [{entry.get('id', '?')}] {entry.get('title', '?')} — {count} sections filled")
                lines.append("")

        header = f"Journal entries ({mode})"
        if entry_type:
            header += f", type={entry_type}"
        lines.append(f"{header}:")

        if not entries:
            lines.append("  (none)")
        else:
            for entry in entries:
                resolved_tag = ""
                if entry.get("resolved") is True:
                    resolved_tag = " [RESOLVED]"
                elif entry.get("resolved") is False:
                    resolved_tag = " [OPEN]"

                date_str = entry.get("date", entry.get("created_at", ""))[:10]
                preview = entry.get("content", "")[:140]
                if len(entry.get("content", "")) > 140:
                    preview += "..."

                lines.append(
                    f"  [{entry.get('id', '?')}] ({entry.get('entry_type', '?')}{resolved_tag}) "
                    f"{date_str} — {preview}"
                )

        lines.append("")
        lines.append("Use read_journal(entry_id='...') to read any entry in full.")
        return "\n".join(lines)

    # --- Update ---

    def _delete_entry(self, entry_id: str, confirm: bool = False,
                      reason: str = "") -> str:
        """Permanently delete a transient journal entry and its memory."""
        if not entry_id:
            return "Entry ID is required."

        if entry_id in PINNED_IDS:
            return (
                f"Cannot delete pinned entry '{entry_id}'. "
                f"Pinned identity entries are foundational and protected. "
                f"Use update_journal to revise them instead."
            )

        # Locate the markdown file
        md_path = self.entries_dir / f"{entry_id}.md"
        if not md_path.exists():
            return f"Journal entry '{entry_id}' not found."

        # Load preview for confirmation
        entry = _parse_markdown_entry(md_path)
        if entry is None:
            return f"Journal entry '{entry_id}' could not be read."

        preview = entry.get("content", "")[:200]
        preview += "…" if len(entry.get("content", "")) > 200 else ""
        date_str = entry.get("date", "")[:10]
        etype = entry.get("entry_type", "?")

        # First call without confirm: return preview
        if not confirm:
            return (
                f"⚠️ Are you sure? This will permanently delete entry '{entry_id}':\n"
                f"  Date: {date_str}\n"
                f"  Type: {etype}\n"
                f'  Content: "{preview}"\n\n'
                f"This cannot be undone. To proceed, call delete_journal again "
                f"with confirm=true (and a reason if you have one)."
            )

        # Find and remove the corresponding companion memory
        removed_memory = False
        for mem_file in self.memory_dir.glob("memory_*.json"):
            try:
                with open(mem_file, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                if mem.get("journal_file") == f"entries/{entry_id}.md":
                    mem_file.unlink()
                    removed_memory = True
                    break
            except (json.JSONDecodeError, IOError):
                continue

        # Delete the markdown file
        try:
            md_path.unlink()
        except IOError as e:
            return f"Failed to delete entry file: {e}"

        # Rebuild aggregate and refresh latest.md
        if removed_memory:
            self._rebuild_memory_aggregate()
        self._generate_latest()

        reason_str = f" Reason: {reason}" if reason else ""
        logger.info(f"Journal entry deleted: {entry_id}.{reason_str}")
        return (
            f"Journal entry '{entry_id}' deleted.{reason_str}"
        )

    def _update_entry(self, entry_id: str, content: str, section: str = None,
                      resolved: bool = None) -> str:
        """Update an existing entry (pinned or transient)."""
        if not entry_id:
            return "Entry ID is required."

        now = datetime.now().isoformat()

        # Pinned entry update
        if entry_id in PINNED_IDS:
            return self._update_pinned(entry_id, content, section, now)

        # Transient entry update
        return self._update_transient(entry_id, content, resolved, now)

    def _update_pinned(self, entry_id: str, content: str, section: str | None,
                       now: str) -> str:
        """Update a section of a pinned identity entry."""
        # Check new location first, then old
        path = self.identity_dir / f"{entry_id}.json"
        if not path.exists():
            path = self.journal_dir / f"{entry_id}.json"
        if not path.exists():
            return f"Pinned entry '{entry_id}' not found."

        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return f"Failed to read entry: {e}"

        sections = entry.get("sections", {})
        if section:
            if section not in sections:
                valid = ", ".join(sections.keys())
                return f"Unknown section '{section}' for {entry_id}. Valid: {valid}"
            sections[section] = content.strip()
            entry["last_updated"] = now

            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated {entry_id}.{section}")
                label = section.replace("_", " ").title()
                return f"Updated {entry_id} — {label}: '{content[:80]}'"
            except IOError as e:
                return f"Failed to save: {e}"
        else:
            # No section specified — list available sections
            section_list = "\n".join(
                f"  - {k}: {v[:50] + '...' if v and len(v) > 50 else v or '(empty)'}"
                for k, v in sections.items()
            )
            return (
                f"Which section of {entry_id} do you want to update? "
                f"Available sections:\n{section_list}\n\n"
                f"Call update_journal again with a 'section' parameter."
            )

    def _update_transient(self, entry_id: str, content: str, resolved: bool | None,
                          now: str) -> str:
        """Update a transient journal entry (.md file + companion memory)."""
        md_path = self.entries_dir / f"{entry_id}.md"
        if not md_path.exists():
            return f"Journal entry '{entry_id}' not found."

        entry = _parse_markdown_entry(md_path)
        if not entry:
            return f"Failed to parse journal entry '{entry_id}'."

        # Update content
        if content.strip():
            entry["content"] = content.strip()

        if resolved is not None and entry.get("resolved") is not None:
            entry["resolved"] = resolved

        entry["date"] = now  # update timestamp

        # Write back
        body = entry.pop("content", "")
        entry.pop("id", None)  # ID is derived from filename
        try:
            _write_markdown_entry(md_path, entry, body)
        except IOError as e:
            return f"Failed to save: {e}"

        # Update companion memory embedding
        if content.strip():
            self._update_companion_memory(entry_id, content.strip(), now)

        self._generate_latest()

        logger.info(f"Updated journal entry: {entry_id}")
        return f"Journal entry {entry_id} updated: '{content[:80]}'"

    def _update_companion_memory(self, entry_id: str, content: str, date: str):
        """Update the companion memory for a journal entry."""
        journal_file = f"entries/{entry_id}.md"
        # Find existing companion memory
        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                if mem.get("type") == "journal" and mem.get("journal_file") == journal_file:
                    # Update embedding and text
                    model = _get_embedding_model()
                    preview = content[:500] + ("..." if len(content) > 500 else "")
                    mem["text"] = preview
                    mem["embedding"] = model.encode(content).tolist() if model else []
                    mem["date"] = date
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(mem, f, indent=2)
                    self._rebuild_memory_aggregate()
                    return
            except (json.JSONDecodeError, IOError):
                continue

        # No existing companion — create one
        self._create_companion_memory(
            entry_id, content,
            [],  # tags from original entry would need parsing
            date
        )

    # --- latest.md generation ---

    def _generate_latest(self):
        """Generate latest.md orientation file — 3 most recent + pinned summaries."""
        lines = ["# Journal — Latest", ""]

        # Pinned identity summaries
        pinned = self.load_pinned_entries()
        has_pinned = False
        for entry in pinned:
            filled = {k: v for k, v in entry.get("sections", {}).items() if v}
            if filled:
                if not has_pinned:
                    lines.append("## Identity")
                    has_pinned = True
                lines.append(f"### {entry.get('title', entry['id'])}")
                for key, value in filled.items():
                    label = key.replace("_", " ").title()
                    lines.append(f"- **{label}:** {value}")
                lines.append("")

        # 3 most recent transient entries
        recent = self.load_recent_entries(limit=3)
        if recent:
            lines.append("## Recent Entries")
            for entry in recent:
                entry_id = entry.get("id", "?")
                entry_type = entry.get("entry_type", "?")
                date = entry.get("date", entry.get("created_at", "?"))[:10]
                content_preview = entry.get("content", "")[:500]

                resolved_tag = ""
                if entry.get("resolved") is True:
                    resolved_tag = " [RESOLVED]"
                elif entry.get("resolved") is False:
                    resolved_tag = " [OPEN]"

                lines.append(f"- **{entry_id}** ({entry_type}{resolved_tag}, {date}): {content_preview}")
            lines.append("")

        latest_path = self.journal_dir / "latest.md"
        try:
            latest_path.write_text("\n".join(lines), encoding="utf-8")
        except IOError as e:
            logger.warning(f"Failed to generate latest.md: {e}")
