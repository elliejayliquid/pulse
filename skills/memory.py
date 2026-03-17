"""
Memory skill — save and search persistent memories.

Writes to the SAME directory (~/.local-memory/) in the SAME JSON format
as the MCP memory server. This means memories saved here are visible
to Claude sessions via MCP, and vice versa.

Embedding generation is optional — works without PyTorch (Python 3.14).
When embeddings aren't available, search falls back to keyword matching.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from skills.base import BaseSkill
from core.context import _get_embedding_model

logger = logging.getLogger(__name__)


class MemorySkill(BaseSkill):
    name = "memory"

    def __init__(self, config: dict):
        super().__init__(config)
        self.memory_dir = Path(
            config.get("paths", {}).get("nova_memory", str(Path.home() / ".local-memory"))
        )
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": (
                        "Save a fact, preference, or important detail to your persistent memory. "
                        "Use this when your human tells you something worth remembering across sessions. "
                        "Examples: 'Their favorite color is blue', 'Project deadline is March 5'."
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
        return f"Unknown tool: {tool_name}"

    # --- Implementation ---

    def _get_next_id(self) -> int:
        """Get the next available memory ID (same logic as MCP server)."""
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

    def _save_memory(self, text: str, tags: str = "") -> str:
        """Save a fact to memory — mirrors MCP server's store_fact()."""
        if not text.strip():
            return "Cannot save empty memory."

        mem_id = f"{self._get_next_id():03d}"

        # Generate embedding if available (graceful on Python 3.14)
        model = _get_embedding_model()
        embedding = model.encode(text).tolist() if model else []

        memory = {
            "id": mem_id,
            "text": text.strip(),
            "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
            "type": "fact",
            "importance": 5,
            "date": datetime.now().isoformat(),
            "embedding": embedding,
        }

        mem_file = self.memory_dir / f"memory_{mem_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
            logger.info(f"Memory saved: {mem_file.name} — {text[:50]}...")
            return f"Remembered (ID: {mem_id}): '{text}'"
        except IOError as e:
            logger.error(f"Failed to save memory: {e}")
            return f"Failed to save memory: {e}"

    def _search_memory(self, query: str) -> str:
        """Search memories — semantic if embeddings available, keyword fallback."""
        if not query.strip():
            return "Please provide a search query."

        memories = self._load_all_memories()
        if not memories:
            return "No memories found yet."

        model = _get_embedding_model()

        if model and any(m.get("embedding") for m in memories):
            # Semantic search (same as MCP server)
            return self._semantic_search(query, memories, model)
        else:
            # Keyword fallback
            return self._keyword_search(query, memories)

    def _load_all_memories(self) -> list[dict]:
        """Load all memory files."""
        memories = []
        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    memories.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        return memories

    def _semantic_search(self, query: str, memories: list[dict], model) -> str:
        """Vector similarity search — mirrors MCP server's search_memory()."""
        query_vec = model.encode(query)

        scored = []
        for mem in memories:
            emb = mem.get("embedding", [])
            if not emb:
                continue
            mem_vec = np.array(emb)
            # Cosine similarity
            norm_q = np.linalg.norm(query_vec)
            norm_m = np.linalg.norm(mem_vec)
            if norm_q == 0 or norm_m == 0:
                continue
            score = float(np.dot(query_vec, mem_vec) / (norm_q * norm_m))
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, mem in scored[:3]:
            if score > 0.25:
                results.append(f"[{mem['date'][:10]}] {mem['text']} (relevance: {int(score * 100)}%)")

        if not results:
            return "No relevant memories found."

        return "\n".join(results)

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

        # Sort by date, oldest first (chronological reading order)
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

    def _keyword_search(self, query: str, memories: list[dict]) -> str:
        """Simple keyword fallback when embeddings aren't available."""
        query_lower = query.lower()
        keywords = query_lower.split()

        scored = []
        for mem in memories:
            text_lower = mem.get("text", "").lower()
            tags_lower = " ".join(mem.get("tags", [])).lower()
            searchable = text_lower + " " + tags_lower

            # Count keyword hits
            hits = sum(1 for kw in keywords if kw in searchable)
            if hits > 0:
                scored.append((hits, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for hits, mem in scored[:5]:
            results.append(f"[{mem['date'][:10]}] {mem['text']}")

        if not results:
            # Fall back to most recent memories
            by_date = sorted(memories, key=lambda m: m.get("date", ""), reverse=True)
            for mem in by_date[:3]:
                results.append(f"[{mem['date'][:10]}] {mem['text']}")
            if results:
                return "No keyword matches. Here are your most recent memories:\n" + "\n".join(results)
            return "No memories found."

        return "\n".join(results)
