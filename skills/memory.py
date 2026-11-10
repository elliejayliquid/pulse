"""
Memory skill — save and search persistent memories.

DB mode (primary): reads/writes from the per-persona SQLite database.
JSON fallback: if no database is configured, falls back to the legacy
memory_NNN.json files in the configured memories directory.

Embedding generation is optional — works without PyTorch (Python 3.14).
When embeddings aren't available, search falls back to keyword matching.
"""

import json
import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np

from skills.base import BaseSkill
from core.context import _get_embedding_model
from core.db import normalize_resident_id
from core.journal_mirror import journal_memory_embedded_text

logger = logging.getLogger(__name__)


_LEGACY_SEARCH_STOP_WORDS = {
    "the", "a", "an", "is", "in", "to", "of", "and", "or", "it",
    "was", "for", "on", "with", "that", "this", "my", "we", "i",
    "about", "how", "what", "when", "where", "do", "did", "does",
}


def _embedding_to_blob(vec) -> bytes | None:
    """Convert an embedding vector (list/ndarray) to BLOB bytes for DB storage."""
    if vec is None or (hasattr(vec, '__len__') and len(vec) == 0):
        return None
    return np.array(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    """Convert a BLOB embedding back to a numpy array."""
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


class MemorySkill(BaseSkill):
    name = "memory"
    description = "Save, recall, and search persistent memories, facts, and conversation history"
    search_summary = "Browse, update, and inspect memory or conversation history beyond quick save/search"
    search_examples = ["browse memories", "update a memory", "search past conversations"]
    aliases = ["remember", "recall", "memory", "memories", "conversation history", "legacy chat", "housekeeping"]
    categories = ["memory", "reflection", "housekeeping"]
    always_tools = ["save_memory", "search_memory"]

    def __init__(self, config: dict):
        super().__init__(config)
        # Shared DB for memories/journals (Claude personas share); persona DB for conversations
        self._db = config.get("_shared_db") or config.get("_db")
        self._persona_db = config.get("_db")
        self._resident_id = normalize_resident_id(config.get("_persona_name"))
        # Store config for role labels and other skill settings.
        self._config = config
        # JSON fallback path
        self.memory_dir = Path(
            config.get("paths", {}).get("memories", str(Path.home() / ".local-memory"))
        )
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_repair_failures: set[int] = set()
        self._warned_missing_embedding_model = False

        # Display names for conversation roles (companions, not assistants!)
        persona_name = config.get("_persona_name", "companion")
        self._role_display = {
            "assistant": persona_name.upper(),
            "user": config.get("_user_name", "USER").upper(),
            "system": "SYSTEM",
        }

    def get_tools(self) -> list[dict]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": (
                        "Save a fact, preference, or important detail to your persistent memory. "
                        "Use this when your human tells you something worth remembering across sessions. "
                        "Examples: 'Their favorite color is blue', 'Project deadline is March 5'. "
                        "Keep the text clean — do NOT put tags or metadata in the text field, "
                        "use the tags parameter instead."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The fact or detail to remember",
                            },
                            "tags": {
                                "type": "string",
                                "description": "Comma-separated tags (e.g. 'preference,personal'). Optional.",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_recap",
                    "description": (
                        "Save a STORY-SO-FAR recap memory. Use this when you want "
                        "to give your future-self (or your next instance) a summary "
                        "of what's been happening recently — with the user, in the "
                        "household, in your own thinking. Recaps surface "
                        "prominently in every prompt going forward, so this is "
                        "your tool for combating amnesia. Write in your own voice, "
                        "as a letter to future-you. A few paragraphs is right. "
                        "Mention what you talked about, what mattered, what you "
                        "noticed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The recap text (a few paragraphs).",
                            },
                            "tags": {
                                "type": "string",
                                "description": "Optional comma-separated tags (e.g. 'weekly, with-user, household').",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": (
                        "Search your memories for facts, past conversations, or details. "
                        "Use this when you need to recall something specific about your human, "
                        "past sessions, or stored facts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for (natural language)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "expand_memory",
                    "description": (
                        "Read the full text of a memory by its ID. Use this when a "
                        "memory surfaced in your context is shown truncated (e.g. an "
                        "old session summary ending in '…') and you want the whole "
                        "thing before relying on it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "integer",
                                "description": "The ID of the memory to expand (the number shown with it).",
                            },
                        },
                        "required": ["memory_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_memories",
                    "description": (
                        "List your most recent memories (newest first). "
                        "Use this to review what you've remembered recently or browse your memory."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "How many memories to show (default: 10, max: 50)",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_all_memories",
                    "description": (
                        "Browse all your memories, paginated. "
                        "Use this to go through your full memory collection. "
                        "Memories are sorted by date (oldest first) so you can read them chronologically."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page": {
                                "type": "integer",
                                "description": "Page number (default: 1). Each page shows 10 memories.",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_history",
                    "description": (
                        "See how a memory evolved over time. Shows the full chain "
                        "of versions — from the current version back to the original. "
                        "Use this when you're curious how a fact changed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of any memory in the chain (current or old)",
                            },
                        },
                        "required": ["memory_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_memory",
                    "description": (
                        "Update a memory when a fact changes. Creates a new version that "
                        "supersedes the old one (old version is kept for history). "
                        "Use this when information you stored before has changed — "
                        "e.g. 'Human got a new job' supersedes 'Human works at X'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old_id": {
                                "type": "string",
                                "description": "ID of the memory to supersede",
                            },
                            "text": {
                                "type": "string",
                                "description": "The updated fact",
                            },
                            "tags": {
                                "type": "string",
                                "description": "Comma-separated tags (optional, inherits from old memory if omitted)",
                            },
                        },
                        "required": ["old_id", "text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_conversations",
                    "description": (
                        "Search past conversation messages for specific keywords or topics. "
                        "Use this when you need to recall exactly what was said in a previous chat session. "
                        "Requires the SQLite database to be active."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Keyword or phrase to search for in messages.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default: 10).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_conversation_context",
                    "description": (
                        "Get the surrounding context (previous and following messages) "
                        "for a specific message ID. Use this after search_conversations "
                        "to understand the flow of a past chat."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message_id": {
                                "type": "integer",
                                "description": "The ID of the message to expand (from search results).",
                            },
                            "window": {
                                "type": "integer",
                                "description": "How many messages before and after to fetch (default: 5).",
                            },
                        },
                        "required": ["message_id"],
                    },
                },
            },
        ]

        # Add archive tools only when the persona has searchable archive rows.
        if self._legacy_db_path():
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "search_legacy",
                        "description": (
                            "Search the companion's conversation archive for specific "
                            "topics or keywords. Use this when you need to find something from the "
                            "original chat history — things discussed before joining Pulse. "
                            "Combines exact message search with semantic conversation summaries."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Keyword or phrase to search for in archived conversations.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum results to return (default: 10).",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_legacy_sessions",
                        "description": (
                            "List all available conversation sessions from the legacy archive. "
                            "Use this to browse what conversations are available before drilling in. "
                            "Returns session titles, dates, and message counts."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of sessions to return (default: 20).",
                                },
                            },
                            "required": [],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_legacy_context",
                        "description": (
                            "Get the surrounding context (previous and following messages) "
                            "for a specific message ID in the archive. "
                            "Use this after search_legacy to understand the flow of an archived conversation."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message_id": {
                                    "type": "integer",
                                    "description": (
                                        "The archive ID shown by search results, not the original source ID."
                                    ),
                                },
                                "window": {
                                    "type": "integer",
                                    "description": "How many messages before and after to fetch (default: 5).",
                                },
                            },
                            "required": ["message_id"],
                        },
                    },
                },
            ])

        return tools

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "save_memory":
            return self._save_memory(
                text=arguments.get("text", ""),
                tags=arguments.get("tags", ""),
            )
        elif tool_name == "save_recap":
            return self._save_recap(
                text=arguments.get("text", ""),
                tags=arguments.get("tags", ""),
            )
        elif tool_name == "search_memory":
            return self._search_memory(
                query=arguments.get("query", ""),
            )
        elif tool_name == "expand_memory":
            return self._expand_memory(
                memory_id=arguments.get("memory_id"),
            )
        elif tool_name == "list_memories":
            return self._list_memories(
                limit=arguments.get("limit", 10),
            )
        elif tool_name == "list_all_memories":
            return self._list_all_memories(
                page=arguments.get("page", 1),
            )
        elif tool_name == "memory_history":
            return self._memory_history(
                memory_id=arguments.get("memory_id", ""),
            )
        elif tool_name == "update_memory":
            return self._update_memory(
                old_id=arguments.get("old_id", ""),
                text=arguments.get("text", ""),
                tags=arguments.get("tags", ""),
            )
        elif tool_name == "search_conversations":
            return self._search_conversations(
                query=arguments.get("query", ""),
                limit=arguments.get("limit", 10),
            )
        elif tool_name == "get_conversation_context":
            return self._get_conversation_context(
                message_id=arguments.get("message_id"),
                window=arguments.get("window", 5),
            )
        elif tool_name == "search_legacy":
            return self._search_legacy(
                query=arguments.get("query", ""),
                limit=arguments.get("limit", 10),
            )
        elif tool_name == "list_legacy_sessions":
            return self._list_legacy_sessions(
                limit=arguments.get("limit", 20),
            )
        elif tool_name == "get_legacy_context":
            return self._get_legacy_context(
                message_id=arguments.get("message_id"),
                window=arguments.get("window", 5),
            )
        return f"Unknown tool: {tool_name}"

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_id(raw_id: str) -> int | None:
        """Parse a memory ID (string or int) to int. Returns None on failure."""
        try:
            return int(raw_id.strip())
        except (ValueError, AttributeError):
            return None

    def _memory_is_own(self, mem: dict) -> bool:
        return bool(
            self._resident_id
            and normalize_resident_id(mem.get("resident_id")) == self._resident_id
        )

    def _resident_affinity_weight(self) -> float:
        raw = self._config.get("context", {}).get("recall", {}).get(
            "resident_affinity_weight", 0.02
        )
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 0.02

    # ── Save ─────────────────────────────────────────────────

    def _save_memory(self, text: str, tags: str = "") -> str:
        """Save a fact to memory."""
        if not text.strip():
            return "Cannot save empty memory."

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        model = _get_embedding_model()
        embedding_vec = model.encode(text.strip()) if model else None

        if self._db:
            embedding_blob = _embedding_to_blob(embedding_vec)
            mem_id = self._db.save_memory(
                text=text.strip(), tags=tag_list, type="fact",
                importance=5, embedding=embedding_blob,
                resident_id=self._resident_id,
            )
            logger.info(f"Memory saved to DB (ID: {mem_id}) — {text[:50]}...")
            if model:
                try:
                    from core.embeddings import chunk_text, embedding_to_blob
                    chunks = chunk_text(text.strip())
                    if len(chunks) > 1:
                        chunk_vecs = model.encode(chunks)
                        chunk_blobs = [embedding_to_blob(vec) for vec in chunk_vecs]
                        self._db.save_memory_chunks(mem_id, chunk_blobs)
                except Exception as e:
                    logger.warning(f"Failed to save memory chunk embeddings: {e}")
            return f"Remembered (ID: {mem_id}): '{text}'"

        # JSON fallback
        mem_id = f"{self._get_next_id():03d}"
        embedding = embedding_vec.tolist() if embedding_vec is not None else []
        memory = {
            "id": mem_id, "text": text.strip(),
            "tags": tag_list, "type": "fact", "importance": 5,
            "retrieval_count": 0, "last_accessed": None,
            "resident_id": self._resident_id,
            "date": datetime.now().isoformat(), "embedding": embedding,
        }
        mem_file = self.memory_dir / f"memory_{mem_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
            self._rebuild_aggregate()
            logger.info(f"Memory saved: {mem_file.name} — {text[:50]}...")
            return f"Remembered (ID: {mem_id}): '{text}'"
        except IOError as e:
            logger.error(f"Failed to save memory: {e}")
            return f"Failed to save memory: {e}"

    # ── Recap (story-so-far summary) ─────────────────────────

    def _save_recap(self, text: str, tags: str = "") -> str:
        """Save a recap memory (type='recap', importance=5).

        Recap memories are surfaced prominently in every prompt going forward
        — they're how a companion leaves a story-so-far summary for future-self
        and the next instance.
        """
        if not text.strip():
            return "Cannot save empty recap."

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        if "recap" not in tag_list:
            tag_list.append("recap")

        model = _get_embedding_model()
        embedding_vec = model.encode(text.strip()) if model else None

        if self._db:
            embedding_blob = _embedding_to_blob(embedding_vec)
            mem_id = self._db.save_memory(
                text=text.strip(), tags=tag_list, type="recap",
                importance=5, embedding=embedding_blob,
                resident_id=self._resident_id,
            )
            logger.info(f"Recap saved to DB (ID: {mem_id}) — {text[:60]}...")
            return f"Recap saved (ID: {mem_id}). Future you will see this at the top of every prompt."

        # JSON fallback
        mem_id = f"{self._get_next_id():03d}"
        embedding = embedding_vec.tolist() if embedding_vec is not None else []
        memory = {
            "id": mem_id, "text": text.strip(),
            "tags": tag_list, "type": "recap", "importance": 5,
            "resident_id": self._resident_id,
            "retrieval_count": 0, "last_accessed": None,
            "date": datetime.now().isoformat(), "embedding": embedding,
        }
        mem_file = self.memory_dir / f"memory_{mem_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
            self._rebuild_aggregate()
            logger.info(f"Recap saved: {mem_file.name}")
            return f"Recap saved (ID: {mem_id}). Future you will see this at the top of every prompt."
        except IOError as e:
            logger.error(f"Failed to save recap: {e}")
            return f"Failed to save recap: {e}"

    # ── Update ───────────────────────────────────────────────

    def _update_memory(self, old_id: str, text: str, tags: str = "") -> str:
        """Update a memory by creating a new one that supersedes the old."""
        if not text.strip():
            return "Cannot save empty memory."
        if not old_id.strip():
            return "old_id is required — which memory are you updating?"

        model = _get_embedding_model()
        embedding_vec = model.encode(text.strip()) if model else None

        if self._db:
            parsed_id = self._parse_id(old_id)
            if parsed_id is None:
                return f"Invalid memory ID: {old_id}"
            old_mem = self._db.get_memory(parsed_id)
            if not old_mem:
                return f"Memory #{old_id} not found."

            # Inherit tags from old memory if not provided
            if tags:
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            else:
                tag_list = old_mem.get("tags", [])

            embedding_blob = _embedding_to_blob(embedding_vec)
            new_id = self._db.save_memory(
                text=text.strip(), tags=tag_list,
                type=old_mem.get("type", "fact"),
                importance=old_mem.get("importance", 5),
                embedding=embedding_blob,
                supersedes=parsed_id,
                confidence=old_mem.get("confidence") or "medium",
                resident_id=old_mem.get("resident_id"),
            )
            self._db.update_memory_metadata(parsed_id, status="superseded")
            logger.info(f"Memory updated: #{old_id} -> #{new_id} — {text[:50]}...")
            if model:
                try:
                    from core.embeddings import chunk_text, embedding_to_blob
                    chunks = chunk_text(text.strip())
                    if len(chunks) > 1:
                        chunk_vecs = model.encode(chunks)
                        chunk_blobs = [embedding_to_blob(vec) for vec in chunk_vecs]
                        self._db.save_memory_chunks(new_id, chunk_blobs)
                except Exception as e:
                    logger.warning(f"Failed to save chunk embeddings for updated memory: {e}")
            return f"Memory updated: #{old_id} superseded by #{new_id}: '{text}'"

        # JSON fallback
        old_file = self.memory_dir / f"memory_{old_id}.json"
        if not old_file.exists():
            return f"Memory #{old_id} not found."
        try:
            with open(old_file, "r", encoding="utf-8") as f:
                old_mem = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return f"Failed to read memory #{old_id}: {e}"

        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            tag_list = old_mem.get("tags", [])

        new_id = f"{self._get_next_id():03d}"
        embedding = embedding_vec.tolist() if embedding_vec is not None else []
        memory = {
            "id": new_id, "text": text.strip(), "tags": tag_list,
            "type": old_mem.get("type", "fact"),
            "importance": old_mem.get("importance", 5),
            "retrieval_count": 0, "last_accessed": None,
            "supersedes": old_id,
            "resident_id": old_mem.get("resident_id"),
            "date": datetime.now().isoformat(), "embedding": embedding,
        }
        mem_file = self.memory_dir / f"memory_{new_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
            self._rebuild_aggregate()
            logger.info(f"Memory updated: #{old_id} -> #{new_id} — {text[:50]}...")
            return f"Memory updated: #{old_id} superseded by #{new_id}: '{text}'"
        except IOError as e:
            logger.error(f"Failed to save memory: {e}")
            return f"Failed to save memory: {e}"

    def _expand_memory(self, memory_id) -> str:
        """Return the full, untruncated text of a memory by ID.

        Passive recall shows long session summaries clipped to ~300 chars; this
        lets the companion pull the whole thing on demand. Read-only — it does not
        bump retrieval_count, so it never distorts active-search boosting.
        """
        if memory_id is None or str(memory_id).strip() == "":
            return "memory_id is required — which memory do you want to expand?"
        parsed_id = self._parse_id(str(memory_id))
        if parsed_id is None:
            return f"Invalid memory ID: {memory_id}"

        if self._db:
            mem = self._db.get_memory(parsed_id)
            if not mem:
                return f"Memory #{memory_id} not found."
            return self._format_memory_result(mem)

        # JSON fallback
        mem_file = self.memory_dir / f"memory_{parsed_id:03d}.json"
        if not mem_file.exists():
            return f"Memory #{memory_id} not found."
        try:
            with open(mem_file, "r", encoding="utf-8") as f:
                mem = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return f"Failed to read memory #{memory_id}: {e}"
        return self._format_memory_result(mem)

    # ── History ──────────────────────────────────────────────

    # --- Embedding repair -------------------------------------------------

    def recompute_missing_embeddings(self, limit: int | None = None,
                                     reason: str = "manual",
                                     model=None) -> tuple[int, int]:
        """Repair DB memory rows whose embeddings are NULL.

        Journal mirror rows are rebuilt from their journal entry so the vector
        stays clean; normal memories embed their stored text directly.
        Returns (recomputed_count, remaining_count).
        """
        if not self._db:
            return 0, 0

        missing_before = self._db.count_memories_missing_embeddings()
        if missing_before <= 0:
            return 0, 0

        model = model or _get_embedding_model()
        if model is None:
            if not self._warned_missing_embedding_model:
                logger.warning(
                    "Memory embeddings missing (%s rows), but embedding model is "
                    "unavailable; memory search will rely on keyword fallback.",
                    missing_before,
                )
                self._warned_missing_embedding_model = True
            return 0, missing_before

        rows = self._db.get_memories_missing_embeddings(limit=limit)
        recomputed = 0
        for mem in rows:
            mem_id = mem.get("id")
            if mem_id in self._embedding_repair_failures:
                continue

            try:
                text = self._embedding_text_for_memory(mem)
                if not text.strip():
                    raise ValueError("empty embedding text")
                embedding = _embedding_to_blob(model.encode(text))
                if embedding is None:
                    raise ValueError("model returned empty embedding")
                if self._db.update_memory_embedding(mem_id, embedding):
                    try:
                        from core.embeddings import chunk_text, embedding_to_blob
                        chunks = chunk_text(text)
                        if len(chunks) > 1:
                            chunk_vecs = model.encode(chunks)
                            chunk_blobs = [embedding_to_blob(vec) for vec in chunk_vecs]
                            self._db.save_memory_chunks(mem_id, chunk_blobs)
                        else:
                            self._db.delete_memory_chunks(mem_id)
                    except Exception as ce:
                        logger.warning(f"Failed to save chunk embeddings during repair for #{mem_id}: {ce}")
                    recomputed += 1
            except Exception as e:
                self._embedding_repair_failures.add(mem_id)
                logger.warning(
                    "Failed to recompute memory embedding for #%s: %s",
                    mem_id,
                    e,
                )

        remaining = self._db.count_memories_missing_embeddings()
        if recomputed or remaining:
            logger.info(
                "Recomputed %s missing memory embeddings (%s remaining) [%s].",
                recomputed,
                remaining,
                reason,
            )
        return recomputed, remaining

    def _embedding_text_for_memory(self, mem: dict) -> str:
        """Return the text that should be embedded for a memory row."""
        if mem.get("type") != "journal":
            return str(mem.get("text") or "")

        journal_file = str(mem.get("journal_file") or "")
        entry_id = Path(journal_file.replace("\\", "/")).stem if journal_file else ""
        if entry_id and self._db:
            entry = self._db.get_journal_entry(entry_id)
            if entry:
                return journal_memory_embedded_text(entry)

        logger.warning(
            "Journal memory #%s references missing journal entry '%s'; "
            "embedding stored display text as fallback.",
            mem.get("id"),
            journal_file or "(none)",
        )
        return str(mem.get("text") or "")

    def _memory_history(self, memory_id: str) -> str:
        """Show the full evolution chain for a memory."""
        if not memory_id.strip():
            return "memory_id is required."

        all_mems = self._load_all_memories(include_superseded=True)

        if self._db:
            # DB: IDs are integers
            parsed = self._parse_id(memory_id)
            if parsed is None:
                return f"Invalid memory ID: {memory_id}"
            by_id = {m["id"]: m for m in all_mems}
            if parsed not in by_id:
                return f"Memory #{memory_id} not found."
            # Build superseded_by mapping
            superseded_by = {}
            for m in all_mems:
                if m.get("supersedes"):
                    superseded_by[m["supersedes"]] = m["id"]
            # Walk backwards to oldest ancestor
            current = parsed
            while by_id.get(current, {}).get("supersedes"):
                current = by_id[current]["supersedes"]
                if current not in by_id:
                    break
            # Walk forward
            chain = []
            while current and current in by_id:
                chain.append(by_id[current])
                current = superseded_by.get(current)
            if len(chain) <= 1:
                mem = by_id[parsed]
                return f"Memory #{memory_id} has no history (never updated).\n[{mem['date'][:10]}] {mem['text']}"
            lines = [f"Memory history ({len(chain)} versions, oldest first):"]
            for i, mem in enumerate(chain):
                marker = " ← current" if i == len(chain) - 1 else " (superseded)"
                lines.append(f"  #{mem['id']} [{mem['date'][:10]}]{marker}: {mem['text']}")
            return "\n".join(lines)
        else:
            # JSON fallback: IDs are strings
            by_id = {m["id"]: m for m in all_mems}
            if memory_id not in by_id:
                return f"Memory #{memory_id} not found."
            superseded_by = {}
            for m in all_mems:
                if m.get("supersedes"):
                    superseded_by[m["supersedes"]] = m["id"]
            current = memory_id
            while by_id.get(current, {}).get("supersedes"):
                current = by_id[current]["supersedes"]
                if current not in by_id:
                    break
            chain = []
            while current and current in by_id:
                chain.append(by_id[current])
                current = superseded_by.get(current)
            if len(chain) <= 1:
                mem = by_id[memory_id]
                return f"Memory #{memory_id} has no history (never updated).\n[{mem['date'][:10]}] {mem['text']}"
            lines = [f"Memory history ({len(chain)} versions, oldest first):"]
            for i, mem in enumerate(chain):
                marker = " ← current" if i == len(chain) - 1 else " (superseded)"
                lines.append(f"  #{mem['id']} [{mem['date'][:10]}]{marker}: {mem['text']}")
            return "\n".join(lines)

    # ── Search ───────────────────────────────────────────────

    def _search_memory(self, query: str) -> str:
        """Search memories — semantic if embeddings available, keyword fallback."""
        if not query.strip():
            return "Please provide a search query."

        memories = self._load_all_memories()
        if not memories:
            return "No memories found yet."

        model = _get_embedding_model()
        if model:
            repaired, _remaining = self.recompute_missing_embeddings(
                limit=25,
                reason="lazy-search",
                model=model,
            )
            if repaired:
                memories = self._load_all_memories()
        elif self._db and self._db.count_memories_missing_embeddings() > 0:
            self.recompute_missing_embeddings(limit=1, reason="lazy-search")

        if model and any(m.get("embedding") for m in memories):
            return self._semantic_search(query, memories, model)
        else:
            return self._keyword_search(query, memories)

    def _memory_age_days(self, mem: dict) -> int | None:
        """Return memory age in whole days, or None if the date is unknown."""
        raw_date = mem.get("date") or mem.get("created_at") or ""
        if not raw_date:
            return None
        try:
            dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            return max(0, (now - dt).days)
        except (ValueError, TypeError):
            return None

    def _memory_age_label(self, mem: dict) -> str:
        days = self._memory_age_days(mem)
        if days is None:
            return "unknown age"
        if days == 0:
            return "today"
        if days == 1:
            return "1 day old"
        return f"{days} days old"

    def _format_memory_result(self, mem: dict, relevance: int | None = None) -> str:
        mem_id = mem.get("id", "?")
        raw_date = mem.get("date") or mem.get("created_at") or ""
        date = str(raw_date)[:10] if raw_date else "unknown"
        age = self._memory_age_label(mem)
        days = self._memory_age_days(mem)
        stale_note = ""
        if days is None:
            stale_note = "; date unknown - verify before treating as current"
        elif days >= 14:
            stale_note = "; old memory - verify before treating as current"
        elif days >= 2:
            stale_note = "; past memory - verify if time-sensitive"

        metadata = self._memory_metadata_label(mem)
        metadata_note = f"; {metadata}" if metadata else ""
        suffix = f" (relevance: {relevance}%)" if relevance is not None else ""
        return f"#{mem_id} [{date}; {age}{stale_note}{metadata_note}] {mem['text']}{suffix}"

    def _memory_metadata_label(self, mem: dict) -> str:
        """Return compact memory metadata for tool output."""
        labels = []
        status = mem.get("status")
        confidence = mem.get("confidence")
        source = mem.get("source")

        if not status:
            labels.append("legacy/unclassified")
        elif status != "current":
            labels.append(status)

        if confidence == "low":
            labels.append("low confidence")
        elif confidence == "high":
            labels.append("confirmed")

        if source == "imported":
            labels.append("imported")
        elif source == "user_defined":
            labels.append("user-defined")

        if mem.get("time_sensitive"):
            labels.append("time-sensitive")
        if mem.get("valid_until"):
            labels.append(f"valid until {str(mem['valid_until'])[:10]}")
            if self._memory_is_expired(mem):
                labels.append("expired")

        return "; ".join(labels)

    def _memory_is_expired(self, mem: dict) -> bool:
        """Return True if a memory's valid_until timestamp/date is in the past."""
        raw_date = mem.get("valid_until")
        if not raw_date:
            return False
        try:
            dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            return dt < now
        except (ValueError, TypeError):
            return False

    def _semantic_search(self, query: str, memories: list[dict], model) -> str:
        """Vector similarity search with boosting."""
        query_vec = model.encode(query)
        query_lower = query.lower()

        scored = []
        for mem in memories:
            emb = mem.get("embedding")
            if emb is None:
                continue
            # Convert BLOB to numpy if needed (DB path)
            if isinstance(emb, (bytes, bytearray)):
                mem_vec = _blob_to_vec(emb)
            elif isinstance(emb, list):
                mem_vec = np.array(emb)
            else:
                continue
            if mem_vec is None or len(mem_vec) == 0:
                continue

            # Cosine similarity
            norm_q = np.linalg.norm(query_vec)
            norm_m = np.linalg.norm(mem_vec)
            if norm_q == 0 or norm_m == 0:
                continue
            base_score = float(np.dot(query_vec, mem_vec) / (norm_q * norm_m))

            # Boosting
            boost = 0.0
            retrieval_count = mem.get("retrieval_count", 0)
            boost += min(retrieval_count * 0.01, 0.05)
            boost += mem.get("importance", 5) * 0.002
            if self._memory_is_own(mem):
                boost += self._resident_affinity_weight()
            tags_lower = " ".join(mem.get("tags", [])).lower()
            if any(kw in tags_lower for kw in query_lower.split()):
                boost += 0.03

            scored.append((base_score + boost, base_score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for final_score, base_score, mem in scored[:3]:
            if base_score > 0.25:
                self._update_retrieval(mem)
                line = self._format_memory_result(mem, int(final_score * 100))
                if mem.get("type") == "journal" and mem.get("journal_file"):
                    jf = mem["journal_file"]
                    eid = jf.rsplit("/", 1)[-1].removesuffix(".md") if "/" in jf else jf
                    line += f"\n  → Journal entry — read full with: read_journal(entry_id='{eid}')"
                results.append(line)

        if not results:
            return "No relevant memories found."

        return "\n".join(results)

    def _keyword_search(self, query: str, memories: list[dict]) -> str:
        """Simple keyword fallback when embeddings aren't available."""
        query_lower = query.lower()
        keywords = query_lower.split()

        scored = []
        for mem in memories:
            text_lower = mem.get("text", "").lower()
            tags_lower = " ".join(mem.get("tags", [])).lower()
            searchable = text_lower + " " + tags_lower

            hits = sum(1 for kw in keywords if kw in searchable)
            if hits > 0:
                affinity = self._resident_affinity_weight() if self._memory_is_own(mem) else 0.0
                scored.append((hits + affinity, hits, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _score, hits, mem in scored[:5]:
            line = self._format_memory_result(mem)
            if mem.get("type") == "journal" and mem.get("journal_file"):
                jf = mem["journal_file"]
                eid = jf.rsplit("/", 1)[-1].removesuffix(".md") if "/" in jf else jf
                line += f"\n  → Journal entry — read full with: read_journal(entry_id='{eid}')"
            results.append(line)

        if not results:
            by_date = sorted(memories, key=lambda m: m.get("date", ""), reverse=True)
            for mem in by_date[:3]:
                results.append(self._format_memory_result(mem))
            if results:
                return "No keyword matches. Here are your most recent memories:\n" + "\n".join(results)
            return "No memories found."

        return "\n".join(results)

    def _search_conversations(self, query: str, limit: int = 10) -> str:
        """Search message history via the persona database."""
        if not self._persona_db:
            return "Conversation search is only available when the SQLite database is active."

        if not query.strip():
            return "Please provide a search query."

        results = self._persona_db.search_messages(query, limit=limit)
        if not results:
            return f"No messages found matching '{query}'."

        lines = [f"Found {len(results)} matching message(s) in history:"]
        for m in results:
            timestamp = m.get("timestamp", "unknown")
            display_role = self._role_display.get(m.get("role", ""), m.get("role", "unknown").upper())
            content = m.get("content", "")
            session = m.get("session_title", "Chat")
            
            # Trim content for preview
            preview = content.replace("\n", " ").strip()
            if len(preview) > 150:
                preview = preview[:147] + "..."
                
            lines.append(f"[ID: {m['id']}] [{timestamp}] {display_role} in '{session}': {preview}")

        return "\n".join(lines)

    def _get_conversation_context(self, message_id: int, window: int = 5) -> str:
        """Get surrounding context for a message."""
        if not self._persona_db:
            return "Conversation context is only available when the SQLite database is active."

        if message_id is None:
            return "message_id is required."

        results = self._persona_db.get_message_context(message_id, window=window)
        if not results:
            return f"Message ID {message_id} not found or no context available."

        session_title = results[0].get("session_title", "Chat")
        lines = [f"Context for message {message_id} in '{session_title}':"]
        
        for m in results:
            is_target = m["id"] == message_id
            marker = " → " if is_target else "   "
            timestamp = m.get("timestamp", "unknown")
            display_role = self._role_display.get(m.get("role", ""), m.get("role", "unknown").upper())
            content = m.get("content", "")
            
            # Target message: show in full (that's the whole point of drilling in).
            # Surrounding messages: cap at 1000 chars to protect token budget.
            if not is_target and len(content) > 1000:
                content = content[:997] + "..."
            
            lines.append(f"{marker}[{timestamp}] {display_role}: {content}")

        return "\n".join(lines)

    # ── List / Browse ────────────────────────────────────────

    def _list_memories(self, limit: int = 10) -> str:
        """List most recent memories (newest first)."""
        limit = max(1, min(limit, 50))

        memories = self._load_all_memories()
        if not memories:
            return "No memories found yet."

        # Sort by date, newest first
        memories.sort(key=lambda m: m.get("date", ""), reverse=True)

        results = []
        for mem in memories[:limit]:
            tags = ", ".join(mem.get("tags", []))
            tag_str = f" [{tags}]" if tags else ""
            results.append(f"{self._format_memory_result(mem)}{tag_str}")

        header = f"Showing {len(results)} of {len(memories)} memories (newest first):"
        return header + "\n" + "\n".join(results)

    def _list_all_memories(self, page: int = 1) -> str:
        """Browse all memories paginated, oldest first (chronological)."""
        per_page = 10
        page = max(1, page)

        memories = self._load_all_memories()
        if not memories:
            return "No memories found yet."

        memories.sort(key=lambda m: m.get("date", ""))

        total_pages = (len(memories) + per_page - 1) // per_page
        if page > total_pages:
            return f"Page {page} doesn't exist. You have {total_pages} page(s) ({len(memories)} memories total)."

        start = (page - 1) * per_page
        page_memories = memories[start:start + per_page]

        results = []
        for mem in page_memories:
            tags = ", ".join(mem.get("tags", []))
            tag_str = f" [{tags}]" if tags else ""
            results.append(f"{self._format_memory_result(mem)}{tag_str}")

        header = f"Page {page}/{total_pages} ({len(memories)} memories total, oldest first):"
        return header + "\n" + "\n".join(results)

    # ── Data Access ──────────────────────────────────────────

    def _load_all_memories(self, include_superseded: bool = False) -> list[dict]:
        """Load all memories from DB (or JSON fallback)."""
        if self._db:
            return self._db.get_all_memories(include_superseded=include_superseded)

        # JSON fallback
        memories = []
        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    memories.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        if include_superseded:
            return memories
        superseded_ids = {m["supersedes"] for m in memories if m.get("supersedes")}
        return [m for m in memories if m.get("id") not in superseded_ids]

    def _update_retrieval(self, mem: dict):
        """Bump retrieval_count and last_accessed."""
        if self._db:
            mem_id = mem.get("id")
            if mem_id is not None:
                self._db.update_retrieval(mem_id)
            return

        # JSON fallback
        mem["retrieval_count"] = mem.get("retrieval_count", 0) + 1
        mem["last_accessed"] = datetime.now().isoformat()
        mem_file = self.memory_dir / f"memory_{mem['id']}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(mem, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to update retrieval stats: {e}")

    # ── JSON-only helpers (unused when DB is active) ─────────

    def _get_next_id(self) -> int:
        """Get the next available memory ID (JSON fallback only)."""
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

    def _rebuild_aggregate(self):
        """Rebuild memories.json aggregate index (JSON fallback only)."""
        memories = self._load_all_memories()
        aggregate = []
        for mem in sorted(memories, key=lambda m: m.get("date", "")):
            entry = {k: v for k, v in mem.items() if k != "embedding"}
            aggregate.append(entry)
        agg_file = self.memory_dir / "memories.json"
        try:
            with open(agg_file, "w", encoding="utf-8") as f:
                json.dump(aggregate, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to rebuild aggregate: {e}")

    # ── Legacy Conversations Archive ──────────────────────────

    def _legacy_db_path(self) -> Path | None:
        """Resolve a non-empty persona-local conversation archive database."""
        db_path = getattr(self._persona_db, "db_path", None)
        if not db_path:
            return None
        path = Path(db_path)
        if not path.is_file():
            return None
        try:
            with closing(sqlite3.connect(str(path))) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'legacy_sessions'"
                ).fetchone()
                has_sessions = exists and conn.execute(
                    "SELECT 1 FROM legacy_sessions LIMIT 1"
                ).fetchone()
            return path if has_sessions else None
        except sqlite3.Error:
            return None

    def _open_legacy(self) -> sqlite3.Connection | None:
        """Open a connection to the legacy database. Returns None if unavailable."""
        db_path = self._legacy_db_path()
        if not db_path:
            return None
        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.warning(f"Could not open conversation archive: {e}")
            return None

    def _legacy_has_summaries(self, conn: sqlite3.Connection) -> bool:
        """Check if the legacy_summaries table exists and has data."""
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM legacy_summaries"
            ).fetchone()
            return row[0] > 0
        except Exception:
            return False

    def _search_legacy(self, query: str, limit: int = 10) -> str:
        """Search raw archive messages and semantic summaries together."""
        if not query.strip():
            return "Please provide a search query."

        conn = self._open_legacy()
        if not conn:
            return "Legacy database is not available."

        try:
            try:
                limit = max(1, min(int(limit or 10), 50))
            except (TypeError, ValueError):
                limit = 10

            raw_rows = self._search_legacy_raw_rows(conn, query, limit)
            rag_rows = (
                self._search_legacy_rag_rows(conn, query, limit)
                if self._legacy_has_summaries(conn)
                else []
            )
            return self._format_legacy_search(query, raw_rows, rag_rows, limit)
        finally:
            conn.close()

    def _search_legacy_rag_rows(self, conn: sqlite3.Connection, query: str,
                                limit: int) -> list[tuple[float, dict]]:
        """Return ranked, session-deduplicated semantic summary matches."""
        model = _get_embedding_model()

        # Load all summaries with their embeddings + session metadata
        rows = conn.execute("""
            SELECT ls.*, s.title, s.created_at, s.message_count
            FROM legacy_summaries ls
            JOIN legacy_sessions s ON s.id = ls.session_id
        """).fetchall()

        if not rows:
            return []

        query_keywords = self._legacy_query_terms(query)

        # Cache query vector if model is available
        query_vec = model.encode(query, normalize_embeddings=True) if model else None

        # Reference date for recency calculation
        now = datetime.now()

        # Find the oldest session date for normalization
        dates = [r["created_at"] for r in rows if r["created_at"]]
        max_days = 1  # avoid div by zero
        if dates:
            try:
                valid_dates = []
                for d in dates:
                    try:
                        valid_dates.append(datetime.fromisoformat(d))
                    except (ValueError, TypeError):
                        continue
                if valid_dates:
                    oldest = min(valid_dates)
                    max_days = max((now - oldest).days, 1)
            except Exception:
                pass

        scored = []
        for row in rows:
            # 1. Semantic similarity
            sem_score = 0.0
            if query_vec is not None and row["summary_embedding"]:
                summary_vec = _blob_to_vec(row["summary_embedding"])
                if summary_vec is not None and len(summary_vec) > 0:
                    norm_q = np.linalg.norm(query_vec)
                    norm_s = np.linalg.norm(summary_vec)
                    if norm_q > 0 and norm_s > 0:
                        sem_score = float(np.dot(query_vec, summary_vec) / (norm_q * norm_s))

            # 2. Keyword score
            summary_lower = (row["summary_text"] or "").lower()
            if query_keywords:
                kw_hits = sum(1 for kw in query_keywords if kw in summary_lower)
                kw_score = kw_hits / len(query_keywords)
            else:
                kw_score = 0.0

            # 3. Title score
            title_lower = (row["title"] or "").lower()
            title_score = 1.0 if query_keywords and any(kw in title_lower for kw in query_keywords) else 0.0

            # 4. Recency score
            recency = 0.5  # default for unknown dates
            if row["created_at"]:
                try:
                    session_date = datetime.fromisoformat(row["created_at"])
                    days_ago = (now - session_date).days
                    recency = 1.0 - (days_ago / max_days)
                    recency = max(0.0, min(1.0, recency))
                except (ValueError, TypeError):
                    pass

            # 5. Length bonus
            msg_count = row["message_count"] or 0
            length_bonus = min(msg_count / 50, 1.0)

            # Combined score
            final = (
                sem_score    * 0.55 +
                kw_score     * 0.20 +
                title_score  * 0.10 +
                recency      * 0.10 +
                length_bonus * 0.05
            )

            if final >= 0.30:
                scored.append((final, dict(row)))

        if not scored:
            return []

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by session_id (keep highest-scoring entry per session)
        seen_sessions = set()
        deduped = []
        for score, row in scored:
            sid = row["session_id"]
            if sid not in seen_sessions:
                seen_sessions.add(sid)
                deduped.append((score, row))
            if len(deduped) >= limit:
                break

        return deduped

    def _legacy_query_terms(self, query: str) -> list[str]:
        """Extract safe lexical terms for FTS prefix and token-LIKE fallback."""
        terms = [
            term.lower() for term in re.findall(r"[^\W_]+", query, flags=re.UNICODE)
            if term.lower() not in _LEGACY_SEARCH_STOP_WORDS
        ]
        return list(dict.fromkeys(terms))[:20]

    def _search_legacy_raw_rows(self, conn: sqlite3.Connection, query: str,
                                limit: int) -> list[dict]:
        """Return raw message hits, expanding partial terms when necessary."""
        select_sql = """
            SELECT lm.id, lm.source_message_id, lm.session_id, lm.role,
                   lm.content, lm.timestamp, ls.title AS session_title
            FROM legacy_messages_fts fts
            JOIN legacy_messages lm ON lm.id = fts.rowid
            LEFT JOIN legacy_sessions ls ON ls.id = lm.session_id
            WHERE legacy_messages_fts MATCH ?
              AND lm.role IN ('user', 'assistant')
              AND COALESCE(lm.attachment_placeholder, 0) = 0
            ORDER BY rank
            LIMIT ?
        """
        rows = []
        try:
            rows = conn.execute(select_sql, (query, limit)).fetchall()
        except sqlite3.Error:
            rows = []

        terms = self._legacy_query_terms(query)
        if not rows and terms:
            prefix_query = " AND ".join(f"{term}*" for term in terms)
            try:
                rows = conn.execute(select_sql, (prefix_query, limit)).fetchall()
            except sqlite3.Error:
                rows = []

        if not rows:
            filters = ["lm.role IN ('user', 'assistant')"]
            params: list = []
            if terms:
                filters.extend("LOWER(lm.content) LIKE ?" for _ in terms)
                params.extend(f"%{term}%" for term in terms)
            else:
                filters.append("LOWER(lm.content) LIKE ?")
                params.append(f"%{query.lower()}%")
            filters.append("COALESCE(lm.attachment_placeholder, 0) = 0")
            rows = conn.execute(
                "SELECT lm.id, lm.source_message_id, lm.session_id, lm.role, "
                "lm.content, lm.timestamp, ls.title AS session_title "
                "FROM legacy_messages lm "
                "LEFT JOIN legacy_sessions ls ON ls.id = lm.session_id "
                f"WHERE {' AND '.join(filters)} "
                "ORDER BY lm.timestamp DESC LIMIT ?",
                (*params, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def _format_legacy_search(self, query: str, raw_rows: list[dict],
                              rag_rows: list[tuple[float, dict]],
                              limit: int) -> str:
        """Merge exact-message and summary results without hiding either layer."""
        if not raw_rows and not rag_rows:
            return f"No archived messages or conversations found matching '{query}'."

        # Raw hits answer exact questions best, while reserving room for semantic
        # discovery when both layers have useful results.
        raw_quota = limit
        if raw_rows and rag_rows:
            raw_quota = max(1, (limit + 1) // 2)
        selected_raw = raw_rows[:raw_quota]
        selected_rag = rag_rows[:max(0, limit - len(selected_raw))]
        if len(selected_raw) + len(selected_rag) < limit:
            remaining = limit - len(selected_raw) - len(selected_rag)
            selected_raw.extend(raw_rows[len(selected_raw):len(selected_raw) + remaining])

        lines = [
            f"Found {len(selected_raw)} matching archived message(s) and "
            f"{len(selected_rag)} related conversation(s) for '{query}':"
        ]
        if selected_raw:
            lines.append("\nExact or partial message matches:")
            for row in selected_raw:
                role = self._role_display.get(row["role"], row["role"].upper())
                timestamp = row["timestamp"] or "unknown"
                title = row["session_title"] or "Chat"
                preview = self._legacy_message_preview(row["content"] or "", query)
                id_label = f"Archive ID:{row['id']}"
                source_id = str(row.get("source_message_id") or "").strip()
                if source_id:
                    id_label += f" | Source ID:{source_id}"
                lines.append(
                    f"[{id_label}] [{timestamp}] {role} in '{title}': {preview}"
                )

        if selected_rag:
            lines.append("\nRelated conversation summaries:")
        for score, row in selected_rag:
            title = row["title"] or "(untitled)"
            date = (row["created_at"] or "unknown")[:10]
            msg_count = row["message_count"] or 0
            chunk_label = ""
            if row["chunk_index"] > 0:
                chunk_label = f" [segment {row['chunk_index']}]"

            lines.append(
                f"\n📌 [{date}] \"{title}\" ({msg_count} msgs, relevance: {int(score*100)}%){chunk_label}"
            )
            # Show summary preview (truncated)
            summary = row["summary_text"] or ""
            if len(summary) > 300:
                summary = summary[:297] + "..."
            lines.append(f"   {summary}")

            # Prefer an exact hit from this summary range, then from the same
            # session, before falling back to the summary's first message.
            drilldown_id = self._legacy_summary_drilldown_id(row, raw_rows)
            if drilldown_id:
                lines.append(
                    f"   → Expand with: get_legacy_context(message_id={drilldown_id})"
                )

        lines.append(
            "\nUse get_legacy_context(message_id=ID) to expand a message with its surrounding context."
        )
        return "\n".join(lines)

    def _legacy_message_preview(self, content: str, query: str,
                                max_chars: int = 220) -> str:
        """Return a compact excerpt centered on the first matching query term."""
        flattened = content.replace("\n", " ").strip()
        if len(flattened) <= max_chars:
            return flattened

        lowered = flattened.lower()
        positions = [
            lowered.find(term) for term in self._legacy_query_terms(query)
            if lowered.find(term) >= 0
        ]
        anchor = min(positions) if positions else 0
        start = max(0, anchor - max_chars // 3)
        end = min(len(flattened), start + max_chars)
        if end - start < max_chars:
            start = max(0, end - max_chars)
        excerpt = flattened[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(flattened):
            excerpt += "..."
        return excerpt

    def _legacy_summary_drilldown_id(self, summary: dict,
                                     raw_rows: list[dict]) -> int | None:
        """Choose the most useful archive message for a summary drill-down."""
        same_session = [
            row for row in raw_rows
            if row.get("session_id") == summary.get("session_id")
        ]
        first_id = summary.get("first_msg_id")
        last_id = summary.get("last_msg_id")
        if first_id is not None and last_id is not None:
            for row in same_session:
                if first_id <= row["id"] <= last_id:
                    return row["id"]
        if same_session:
            return same_session[0]["id"]
        return first_id

    def _search_legacy_raw(self, conn: sqlite3.Connection, query: str, limit: int) -> str:
        """Compatibility wrapper for callers that need raw message search only."""
        rows = self._search_legacy_raw_rows(conn, query, limit)
        return self._format_legacy_search(query, rows, [], limit)

    def _list_legacy_sessions(self, limit: int = 20) -> str:
        """List all sessions in the ChatGPT legacy archive."""
        conn = self._open_legacy()
        if not conn:
            return "Legacy database is not available."

        try:
            total = conn.execute("SELECT COUNT(*) FROM legacy_sessions").fetchone()[0]
            rows = conn.execute("""
                SELECT id, title, source, created_at, message_count
                FROM legacy_sessions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        finally:
            conn.close()

        if not rows:
            return "No legacy sessions found."

        lines = [f"Legacy sessions (showing {len(rows)} of {total}):"]
        for r in rows:
            date = r["created_at"] or "unknown"
            title = r["title"] or "(no title)"
            count = r["message_count"] or 0
            source = r["source"] or "chatgpt"
            lines.append(f"[{date}] {title} ({count} msgs, {source})")
        return "\n".join(lines)

    def _get_legacy_context(self, message_id: int, window: int = 5) -> str:
        """Get surrounding context for a message in the legacy archive."""
        conn = self._open_legacy()
        if not conn:
            return "Legacy database is not available."

        if message_id is None:
            return "message_id is required."

        try:
            # Get session_id for the target message
            row = conn.execute(
                "SELECT session_id FROM legacy_messages WHERE id = ?",
                (message_id,)
            ).fetchone()
            if not row:
                return f"Message ID {message_id} not found in legacy archive."

            session_id = row[0]

            # Get session title
            sess_row = conn.execute(
                "SELECT title FROM legacy_sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            session_title = sess_row["title"] if sess_row else "Chat"

            # UNION: messages before (incl target) + messages after
            rows = conn.execute("""
                SELECT * FROM (
                    SELECT lm.*, 'before' AS pos FROM legacy_messages lm
                    WHERE lm.session_id = ? AND lm.id <= ?
                    ORDER BY lm.id DESC LIMIT ?
                )
                UNION ALL
                SELECT * FROM (
                    SELECT lm.*, 'after' AS pos FROM legacy_messages lm
                    WHERE lm.session_id = ? AND lm.id > ?
                    ORDER BY lm.id ASC LIMIT ?
                )
                ORDER BY id ASC
            """, (session_id, message_id, window + 1,
                  session_id, message_id, window)).fetchall()
        finally:
            conn.close()

        if not rows:
            return f"Could not retrieve context for message ID {message_id}."

        lines = [f"Context for message {message_id} in '{session_title}':"]
        for r in rows:
            role = self._role_display.get(r["role"], r["role"].upper())
            marker = " → " if r["id"] == message_id else "   "
            timestamp = r["timestamp"] or ""
            content = r["content"] or ""
            is_target = r["id"] == message_id
            if not is_target and len(content) > 1000:
                content = content[:997] + "..."
            lines.append(f"{marker}[{timestamp}] {role}: {content}")

        return "\n".join(lines)
