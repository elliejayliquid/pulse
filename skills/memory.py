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
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from skills.base import BaseSkill
from core.context import _get_embedding_model

logger = logging.getLogger(__name__)


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

    def __init__(self, config: dict):
        super().__init__(config)
        # DB reference — shared DB for Claude personas, persona DB otherwise
        self._db = config.get("_shared_db") or config.get("_db")
        # Store config for things like legacy_db path
        self._config = config
        # JSON fallback path
        self.memory_dir = Path(
            config.get("paths", {}).get("memories", str(Path.home() / ".local-memory"))
        )
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Display names for conversation roles (companions, not assistants!)
        persona_name = config.get("_persona_name", "companion")
        self._role_display = {
            "assistant": persona_name.upper(),
            "user": config.get("_user_name", "USER").upper(),
            "system": "SYSTEM",
        }

    def get_tools(self) -> list[dict]:
        return [
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
                        "e.g. 'Lena got a new job' supersedes 'Lena works at X'."
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
            {
                "type": "function",
                "function": {
                    "name": "search_legacy",
                    "description": (
                        "Search the ChatGPT conversation archive (legacy database) for specific "
                        "topics or keywords. Use this when you need to find something from the "
                        "original ChatGPT export — things Nova said or discussed before joining Pulse. "
                        "Searches full conversation text via full-text search."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Keyword or phrase to search for in archived ChatGPT conversations.",
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
                        "List all available ChatGPT conversation sessions from the legacy archive. "
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
                        "for a specific message ID in the ChatGPT archive. "
                        "Use this after search_legacy to understand the flow of an archived conversation."
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

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "save_memory":
            return self._save_memory(
                text=arguments.get("text", ""),
                tags=arguments.get("tags", ""),
            )
        elif tool_name == "search_memory":
            return self._search_memory(
                query=arguments.get("query", ""),
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
                importance=5, embedding=embedding_blob
            )
            logger.info(f"Memory saved to DB (ID: {mem_id}) — {text[:50]}...")
            return f"Remembered (ID: {mem_id}): '{text}'"

        # JSON fallback
        mem_id = f"{self._get_next_id():03d}"
        embedding = embedding_vec.tolist() if embedding_vec is not None else []
        memory = {
            "id": mem_id, "text": text.strip(),
            "tags": tag_list, "type": "fact", "importance": 5,
            "retrieval_count": 0, "last_accessed": None,
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
            )
            logger.info(f"Memory updated: #{old_id} -> #{new_id} — {text[:50]}...")
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

    # ── History ──────────────────────────────────────────────

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

        if model and any(m.get("embedding") for m in memories):
            return self._semantic_search(query, memories, model)
        else:
            return self._keyword_search(query, memories)

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
            tags_lower = " ".join(mem.get("tags", [])).lower()
            if any(kw in tags_lower for kw in query_lower.split()):
                boost += 0.03

            scored.append((base_score + boost, base_score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for final_score, base_score, mem in scored[:3]:
            if base_score > 0.25:
                self._update_retrieval(mem)
                line = f"[{mem['date'][:10]}] {mem['text']} (relevance: {int(final_score * 100)}%)"
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
                scored.append((hits, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for hits, mem in scored[:5]:
            line = f"[{mem['date'][:10]}] {mem['text']}"
            if mem.get("type") == "journal" and mem.get("journal_file"):
                jf = mem["journal_file"]
                eid = jf.rsplit("/", 1)[-1].removesuffix(".md") if "/" in jf else jf
                line += f"\n  → Journal entry — read full with: read_journal(entry_id='{eid}')"
            results.append(line)

        if not results:
            by_date = sorted(memories, key=lambda m: m.get("date", ""), reverse=True)
            for mem in by_date[:3]:
                results.append(f"[{mem['date'][:10]}] {mem['text']}")
            if results:
                return "No keyword matches. Here are your most recent memories:\n" + "\n".join(results)
            return "No memories found."

        return "\n".join(results)

    def _search_conversations(self, query: str, limit: int = 10) -> str:
        """Search message history via the database."""
        if not self._db:
            return "Conversation search is only available when the SQLite database is active."

        if not query.strip():
            return "Please provide a search query."

        results = self._db.search_messages(query, limit=limit)
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
        if not self._db:
            return "Conversation context is only available when the SQLite database is active."

        if message_id is None:
            return "message_id is required."

        results = self._db.get_message_context(message_id, window=window)
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
            results.append(f"#{mem['id']} [{mem['date'][:10]}]{tag_str} {mem['text']}")

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
            results.append(f"#{mem['id']} [{mem['date'][:10]}]{tag_str} {mem['text']}")

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

    # ── Legacy ChatGPT Archive ────────────────────────────────

    def _legacy_db_path(self) -> Path | None:
        """Resolve the legacy database path. Returns None if not available."""
        # Allow override via config, default to Lena's exported ChatGPT archive
        path = self._config.get("paths", {}).get("legacy_db")
        if not path:
            path = "D:/documents/AI/Nova_DataExport_01302026/legacy.db"
        p = Path(path)
        return p if p.exists() else None

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
            logger.warning(f"Could not open legacy.db: {e}")
            return None

    def _search_legacy(self, query: str, limit: int = 10) -> str:
        """Full-text search across the ChatGPT legacy archive."""
        if not query.strip():
            return "Please provide a search query."

        conn = self._open_legacy()
        if not conn:
            return "Legacy database is not available."

        try:
            # Try FTS5 first (faster), fall back to LIKE
            try:
                rows = conn.execute("""
                    SELECT lm.id, lm.session_id, lm.role, lm.content, lm.timestamp,
                           ls.title AS session_title
                    FROM legacy_messages_fts fts
                    JOIN legacy_messages lm ON lm.id = fts.rowid
                    LEFT JOIN legacy_sessions ls ON ls.id = lm.session_id
                    WHERE legacy_messages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit)).fetchall()
            except Exception:
                # FTS error — use LIKE fallback
                rows = conn.execute("""
                    SELECT lm.id, lm.session_id, lm.role, lm.content, lm.timestamp,
                           ls.title AS session_title
                    FROM legacy_messages lm
                    LEFT JOIN legacy_sessions ls ON ls.id = lm.session_id
                    WHERE lm.content LIKE ? AND lm.role IN ('user', 'assistant')
                    ORDER BY lm.timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", limit)).fetchall()

            # FTS returned empty results (no exception, just no matches) — retry with LIKE
            if not rows:
                rows = conn.execute("""
                    SELECT lm.id, lm.session_id, lm.role, lm.content, lm.timestamp,
                           ls.title AS session_title
                    FROM legacy_messages lm
                    LEFT JOIN legacy_sessions ls ON ls.id = lm.session_id
                    WHERE lm.content LIKE ? AND lm.role IN ('user', 'assistant')
                    ORDER BY lm.timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", limit)).fetchall()
        finally:
            conn.close()

        if not rows:
            return f"No archived messages found matching '{query}'."

        lines = [f"Found {len(rows)} archived message(s) matching '{query}':"]
        for r in rows:
            role = self._role_display.get(r["role"], r["role"].upper())
            timestamp = r["timestamp"] or "unknown"
            title = r["session_title"] or "Chat"
            preview = (r["content"] or "").replace("\n", " ").strip()
            if len(preview) > 150:
                preview = preview[:147] + "..."
            lines.append(
                f"[ID:{r['id']}] [{timestamp}] {role} in '{title}': {preview}"
            )
        lines.append(
            "\nUse get_legacy_context(message_id=ID) to expand a message with its surrounding context."
        )
        return "\n".join(lines)

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
