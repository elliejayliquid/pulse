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
            config.get("paths", {}).get("memories", str(Path.home() / ".local-memory"))
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
                                "description": "ID of the memory to supersede (e.g. '005')",
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
            "retrieval_count": 0,
            "last_accessed": None,
            "date": datetime.now().isoformat(),
            "embedding": embedding,
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

    def _memory_history(self, memory_id: str) -> str:
        """Show the full evolution chain for a memory."""
        if not memory_id.strip():
            return "memory_id is required."

        all_mems = self._load_all_memories(include_superseded=True)
        by_id = {m["id"]: m for m in all_mems}

        if memory_id not in by_id:
            return f"Memory #{memory_id} not found."

        # Build a lookup: old_id -> new_id
        superseded_by = {}
        for m in all_mems:
            if m.get("supersedes"):
                superseded_by[m["supersedes"]] = m["id"]

        # Walk backwards to find the oldest ancestor
        current = memory_id
        while by_id.get(current, {}).get("supersedes"):
            current = by_id[current]["supersedes"]
            if current not in by_id:
                break

        # Walk forward from oldest to newest
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

    def _update_memory(self, old_id: str, text: str, tags: str = "") -> str:
        """Update a memory by creating a new one that supersedes the old."""
        if not text.strip():
            return "Cannot save empty memory."
        if not old_id.strip():
            return "old_id is required — which memory are you updating?"

        # Load the old memory to verify it exists and inherit tags
        old_file = self.memory_dir / f"memory_{old_id}.json"
        if not old_file.exists():
            return f"Memory #{old_id} not found."

        try:
            with open(old_file, "r", encoding="utf-8") as f:
                old_mem = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return f"Failed to read memory #{old_id}: {e}"

        # Inherit tags from old memory if not provided
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            tag_list = old_mem.get("tags", [])

        # Create new memory that supersedes the old one
        new_id = f"{self._get_next_id():03d}"
        model = _get_embedding_model()
        embedding = model.encode(text).tolist() if model else []

        memory = {
            "id": new_id,
            "text": text.strip(),
            "tags": tag_list,
            "type": old_mem.get("type", "fact"),
            "importance": old_mem.get("importance", 5),
            "retrieval_count": 0,
            "last_accessed": None,
            "supersedes": old_id,
            "date": datetime.now().isoformat(),
            "embedding": embedding,
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

    def _load_all_memories(self, include_superseded: bool = False) -> list[dict]:
        """Load all memory files, filtering out superseded ones by default."""
        memories = []
        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    memories.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue

        if include_superseded:
            return memories

        # Build set of superseded IDs and filter them out
        superseded_ids = {m["supersedes"] for m in memories if m.get("supersedes")}
        return [m for m in memories if m.get("id") not in superseded_ids]

    def _semantic_search(self, query: str, memories: list[dict], model) -> str:
        """Vector similarity search with boosting — mirrors MCP server's search_memory()."""
        query_vec = model.encode(query)
        query_lower = query.lower()

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
            base_score = float(np.dot(query_vec, mem_vec) / (norm_q * norm_m))

            # Boosting (aligned with MCP server)
            boost = 0.0
            retrieval_count = mem.get("retrieval_count", 0)
            boost += min(retrieval_count * 0.01, 0.05)  # frequently accessed
            boost += mem.get("importance", 5) * 0.002     # importance
            # Tag match
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
                    jf = mem["journal_file"]  # e.g. "entries/001.md"
                    eid = jf.rsplit("/", 1)[-1].removesuffix(".md") if "/" in jf else jf
                    line += f"\n  → Journal entry — read full with: read_journal(entry_id='{eid}')"
                results.append(line)

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

    def _update_retrieval(self, mem: dict):
        """Bump retrieval_count and last_accessed, persist to disk."""
        mem["retrieval_count"] = mem.get("retrieval_count", 0) + 1
        mem["last_accessed"] = datetime.now().isoformat()
        mem_file = self.memory_dir / f"memory_{mem['id']}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(mem, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to update retrieval stats: {e}")

    def _rebuild_aggregate(self):
        """Rebuild memories.json aggregate index (no embeddings, for quick browse)."""
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
            line = f"[{mem['date'][:10]}] {mem['text']}"
            if mem.get("type") == "journal" and mem.get("journal_file"):
                jf = mem["journal_file"]
                eid = jf.rsplit("/", 1)[-1].removesuffix(".md") if "/" in jf else jf
                line += f"\n  → Journal entry — read full with: read_journal(entry_id='{eid}')"
            results.append(line)

        if not results:
            # Fall back to most recent memories
            by_date = sorted(memories, key=lambda m: m.get("date", ""), reverse=True)
            for mem in by_date[:3]:
                results.append(f"[{mem['date'][:10]}] {mem['text']}")
            if results:
                return "No keyword matches. Here are your most recent memories:\n" + "\n".join(results)
            return "No memories found."

        return "\n".join(results)
