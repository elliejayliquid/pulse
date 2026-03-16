"""
Journal skill — Nova's personal journal for continuity and self-reflection.

Three types of entries:
- Pinned: _self, _user, _relationship — identity entries, always in context
- Transient: observations, topics, follow-ups — searchable, recent ones auto-loaded

Three tools:
- write_journal: create a new transient entry
- read_journal: search entries (hybrid: semantic + keyword + recency)
- update_journal: update any entry (especially pinned identity ones)

Adapted from HeartlineChat's reformed journal system. Key guard: no speculative
psychology or hidden-motive inference. Grounded categories only.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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

# Pinned entry templates — blank sections for Nova to fill in over time
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


class JournalSkill(BaseSkill):
    name = "journal"

    def __init__(self, config: dict):
        super().__init__(config)
        self.journal_dir = Path(
            config.get("paths", {}).get("journal", "data/journal")
        )
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_pinned_entries()

    def _ensure_pinned_entries(self):
        """Create blank pinned entries if they don't exist yet."""
        for pin_id, template in PINNED_TEMPLATES.items():
            path = self.journal_dir / f"{pin_id}.json"
            if not path.exists():
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
                        "Write a new journal entry. Use this to note events, preferences, "
                        "topics, open threads, or reflections worth preserving for future "
                        "conversations. Most interactions need NO entry — only write when "
                        "something is genuinely worth remembering."
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
                            "tags": {
                                "type": "string",
                                "description": "Comma-separated tags for searchability (optional)",
                            },
                        },
                        "required": ["content", "entry_type"],
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
                    "name": "update_journal",
                    "description": (
                        "Update an existing journal entry. Use this especially for pinned "
                        "identity entries (_self, _user, _relationship) as you learn more. "
                        "For pinned entries, specify the section to update. "
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
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "write_journal":
            return self._write_entry(
                content=arguments.get("content", ""),
                entry_type=arguments.get("entry_type", "reflection"),
                tags=arguments.get("tags", ""),
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
        existing = list(self.journal_dir.glob("entry_*.json"))
        if not existing:
            return 1
        ids = []
        for f in existing:
            try:
                ids.append(int(f.stem.split("_")[1]))
            except (ValueError, IndexError):
                continue
        return max(ids) + 1 if ids else 1

    def _write_entry(self, content: str, entry_type: str, tags: str = "") -> str:
        """Create a new transient journal entry."""
        if not content.strip():
            return "Cannot write empty journal entry."

        if entry_type not in VALID_ENTRY_TYPES:
            return (
                f"Invalid entry_type '{entry_type}'. "
                f"Valid types: {', '.join(VALID_ENTRY_TYPES)}"
            )

        entry_id = f"{self._get_next_id():03d}"
        now = datetime.now().isoformat()

        # Generate embedding for search
        model = _get_embedding_model()
        embedding = model.encode(content).tolist() if model else []

        entry = {
            "id": entry_id,
            "entry_type": entry_type,
            "content": content.strip(),
            "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
            "embedding": embedding,
            "created_at": now,
            "last_updated": now,
            "resolved": False if entry_type in ("open_thread", "follow_up") else None,
        }

        path = self.journal_dir / f"entry_{entry_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            logger.info(f"Journal entry written: {entry_id} ({entry_type}) — {content[:50]}...")
            return f"Journal entry saved (ID: {entry_id}, type: {entry_type}): '{content[:80]}'"
        except IOError as e:
            logger.error(f"Failed to write journal entry: {e}")
            return f"Failed to write journal entry: {e}"

    # --- Read / Search ---

    def _read_by_id(self, entry_id: str) -> str:
        """Read a specific entry in full by its ID."""
        # Pinned entry
        if entry_id in PINNED_IDS:
            entry = self.load_pinned_entry(entry_id)
            if not entry:
                return f"Pinned entry '{entry_id}' not found."
            return self._format_pinned_entry(entry)

        # Transient entry
        path = self.journal_dir / f"entry_{entry_id}.json"
        if not path.exists():
            return f"Journal entry '{entry_id}' not found."
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return f"Failed to read entry: {e}"

        resolved_tag = ""
        if entry.get("resolved") is True:
            resolved_tag = " [RESOLVED]"
        elif entry.get("resolved") is False:
            resolved_tag = " [OPEN]"

        tags_str = f"\nTags: {', '.join(entry['tags'])}" if entry.get("tags") else ""
        return (
            f"Entry {entry['id']} ({entry.get('entry_type', '?')}{resolved_tag})\n"
            f"Created: {entry.get('created_at', '?')[:16]}\n"
            f"Updated: {entry.get('last_updated', '?')[:16]}\n"
            f"{tags_str}\n\n"
            f"{entry.get('content', '')}"
        )

    def _load_transient_entries(self) -> list[dict]:
        """Load all transient journal entries."""
        entries = []
        for filepath in self.journal_dir.glob("entry_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    entries.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        return entries

    def load_pinned_entry(self, pin_id: str) -> dict | None:
        """Load a single pinned entry by ID."""
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
        """Load the most recent transient entries (for heartbeat context)."""
        entries = self._load_transient_entries()
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries[:limit]

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
        """Search journal entries with hybrid ranking."""
        if not query.strip():
            return "Please provide a search query."

        results = []

        # Search pinned entries if requested
        if include_pinned:
            for entry in self.load_pinned_entries():
                # Check if any section content matches
                sections_text = " ".join(
                    str(v) for v in entry.get("sections", {}).values() if v
                )
                if sections_text and query.lower() in sections_text.lower():
                    results.append(self._format_pinned_entry(entry))

        # Search transient entries
        entries = self._load_transient_entries()
        if entry_type:
            entries = [e for e in entries if e.get("entry_type") == entry_type]

        if not entries:
            if results:
                return "\n\n".join(results)
            return "No journal entries found yet."

        # Hybrid scoring
        scored = self._hybrid_search(query, entries)

        for score, entry in scored[:5]:
            resolved_tag = ""
            if entry.get("resolved") is True:
                resolved_tag = " [RESOLVED]"
            elif entry.get("resolved") is False:
                resolved_tag = " [OPEN]"

            tags_str = f" #{' #'.join(entry['tags'])}" if entry.get("tags") else ""
            results.append(
                f"[{entry['id']}] ({entry['entry_type']}{resolved_tag}) "
                f"{entry['content'][:150]}"
                f"{tags_str} — {entry['created_at'][:10]}"
            )

        if not results:
            return "No matching journal entries found."

        return "\n\n".join(results)

    def _hybrid_search(self, query: str, entries: list[dict]) -> list[tuple[float, dict]]:
        """Rank entries by semantic similarity + keyword match + recency.

        Score = 0.5 * semantic + 0.2 * keyword + 0.3 * recency
        """
        model = _get_embedding_model()
        query_lower = query.lower()
        keywords = query_lower.split()
        now_ts = time.time()

        scored = []
        query_vec = model.encode(query) if model else None

        for entry in entries:
            # --- Semantic similarity (0-1) ---
            semantic_score = 0.0
            if query_vec is not None and entry.get("embedding"):
                mem_vec = np.array(entry["embedding"])
                norm_q = np.linalg.norm(query_vec)
                norm_m = np.linalg.norm(mem_vec)
                if norm_q > 0 and norm_m > 0:
                    semantic_score = max(0.0, float(
                        np.dot(query_vec, mem_vec) / (norm_q * norm_m)
                    ))

            # --- Keyword match (0-1) ---
            searchable = (
                entry.get("content", "").lower() + " " +
                " ".join(entry.get("tags", [])).lower() + " " +
                entry.get("entry_type", "").lower()
            )
            keyword_hits = sum(1 for kw in keywords if kw in searchable)
            keyword_score = min(1.0, keyword_hits / max(len(keywords), 1))

            # --- Recency (0-1, exponential decay over 30 days) ---
            try:
                entry_ts = datetime.fromisoformat(entry["created_at"]).timestamp()
                age_days = (now_ts - entry_ts) / 86400
                recency_score = max(0.0, np.exp(-age_days / 30))
            except (KeyError, ValueError):
                recency_score = 0.0

            # Weighted combination
            total = (0.5 * semantic_score) + (0.2 * keyword_score) + (0.3 * recency_score)

            # Minimum threshold — don't return completely irrelevant entries
            if total > 0.15 or keyword_score > 0:
                scored.append((total, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    # --- Update ---

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
            old_value = sections[section]
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
        """Update a transient journal entry."""
        path = self.journal_dir / f"entry_{entry_id}.json"
        if not path.exists():
            return f"Journal entry '{entry_id}' not found."

        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return f"Failed to read entry: {e}"

        if content.strip():
            entry["content"] = content.strip()
            # Re-generate embedding for updated content
            model = _get_embedding_model()
            if model:
                entry["embedding"] = model.encode(content.strip()).tolist()

        if resolved is not None and entry.get("resolved") is not None:
            entry["resolved"] = resolved

        entry["last_updated"] = now

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated journal entry: {entry_id}")
            return f"Journal entry {entry_id} updated: '{content[:80]}'"
        except IOError as e:
            return f"Failed to save: {e}"
