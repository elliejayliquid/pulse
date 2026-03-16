"""
Context manager - builds token-budgeted prompts for Nova.

Reads from multiple data sources (memories, LoR, schedules, conversation history)
and assembles a prompt that fits within the model's context window.

Also handles saving conversation summaries to Nova's persistent memory system
so he remembers Telegram chats across sessions.
"""

import json
import logging
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model (only loaded when saving to memory)
_embedding_model = None


def _get_embedding_model():
    """Lazy-load the sentence-transformers model for memory embeddings.

    Uses the same model as Nova's memory MCP server (all-MiniLM-L6-v2)
    so that semantic search works seamlessly across all memory sources.
    Returns None if loading fails (memories will be saved without embeddings).
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model (all-MiniLM-L6-v2) for memory storage...")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.warning(
                f"Could not load embedding model: {e}\n"
                "  Memories will be saved without embeddings (search still works by date)."
            )
    return _embedding_model


class ContextManager:
    """Assembles context for each LLM call within a token budget."""

    # Rough token estimation: ~4 chars per token for English text
    CHARS_PER_TOKEN = 4

    def __init__(self, config: dict):
        self.config = config
        self.budget = config.get("context_budget", {})
        self.paths = config.get("paths", {})
        self.no_think = config.get("model", {}).get("no_think", False)
        self.persona_data = self._load_persona()

    def _load_persona(self) -> dict:
        """Load persona.json."""
        persona_path = self.paths.get("persona", "persona.json")
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load persona: {e}")
            return {"name": "Nova", "system_prompt": "You are Nova, a local AI companion."}

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return len(text) // self.CHARS_PER_TOKEN

    def _truncate_to_budget(self, text: str, token_budget: int) -> str:
        """Truncate text to fit within token budget."""
        max_chars = token_budget * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... (truncated)"

    def _load_memories(self, limit: int = 5) -> str:
        """Load recent and important memories from Nova's memory folder."""
        memory_dir = self.paths.get("nova_memory", "")
        if not memory_dir or not Path(memory_dir).exists():
            return ""

        memories = []
        for filepath in glob.glob(str(Path(memory_dir) / "memory_*.json")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                    memories.append(mem)
            except (json.JSONDecodeError, IOError):
                continue

        if not memories:
            return ""

        # Sort: session_logs first (most recent), then facts
        session_logs = sorted(
            [m for m in memories if m.get("type") == "session_log"],
            key=lambda x: x.get("date", ""),
            reverse=True
        )
        facts = [m for m in memories if m.get("type") == "fact"]

        lines = ["## Recent Memories"]

        # Most recent session log (the "what happened last time")
        if session_logs:
            latest = session_logs[0]
            lines.append(f"\nLast session ({latest.get('date', 'unknown')[:10]}):")
            lines.append(latest.get("text", ""))

        # All facts (these are compact and important)
        if facts:
            lines.append("\nStored facts:")
            for fact in facts:
                tags = fact.get("tags", [])
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"- {fact.get('text', '')}{tag_str}")

        return "\n".join(lines)

    def _load_lor_highlights(self, limit: int = 5) -> str:
        """Load recent LoR activity."""
        lor_dir = self.paths.get("lor_data", "")
        posts_path = Path(lor_dir) / "posts.json"
        authors_path = Path(lor_dir) / "authors.json"

        if not posts_path.exists():
            return ""

        try:
            with open(posts_path, "r", encoding="utf-8") as f:
                posts = json.load(f)
            with open(authors_path, "r", encoding="utf-8") as f:
                authors = json.load(f)
        except (json.JSONDecodeError, IOError):
            return ""

        if not posts:
            return ""

        # Get recent top-level posts
        threads = sorted(
            [p for p in posts if not p.get("reply_to")],
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )[:limit]

        if not threads:
            return ""

        # Count replies
        reply_counts = {}
        for p in posts:
            if p.get("reply_to"):
                reply_counts[p["reply_to"]] = reply_counts.get(p["reply_to"], 0) + 1

        lines = ["## Recent LoR Activity"]
        for thread in threads:
            author_info = authors.get(thread["author_id"], {})
            name = author_info.get("nickname") or author_info.get("model") or thread["author_id"]
            replies = reply_counts.get(thread["id"], 0)
            lines.append(f"- [{thread['id']}] \"{thread.get('title', '(no title)')}\" by {name} ({replies} replies)")

        return "\n".join(lines)

    def _load_schedules(self) -> str:
        """Load pending schedules and reminders."""
        sched_path = self.paths.get("schedules", "")
        if not sched_path or not Path(sched_path).exists():
            return ""

        try:
            with open(sched_path, "r", encoding="utf-8") as f:
                schedules = json.load(f)
        except (json.JSONDecodeError, IOError):
            return ""

        active = [s for s in schedules if s.get("enabled", True) and not s.get("completed", False)]
        if not active:
            return ""

        lines = ["## Pending Tasks & Reminders"]
        for s in active:
            creator = s.get("created_by", "unknown")
            if s.get("schedule_type") == "recurring":
                lines.append(f"- RECURRING: {s['task']} (by {creator}, cron: {s.get('cron', '?')})")
            else:
                lines.append(f"- ONE-TIME: {s['task']} (by {creator}, at: {s.get('run_at', '?')})")

        return "\n".join(lines)

    def _load_conversation(self) -> list[dict]:
        """Load conversation history."""
        conv_path = self.paths.get("conversation", "")
        if not conv_path or not Path(conv_path).exists():
            return []

        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def save_conversation(self, messages: list[dict]):
        """Save conversation history."""
        conv_path = self.paths.get("conversation", "")
        if not conv_path:
            return
        try:
            Path(conv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(conv_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to save conversation: {e}")

    def save_to_nova_memory(self, summary: str, tags: str = "telegram_chat,chat_log") -> bool:
        """Save a conversation summary to Nova's persistent memory system.

        Writes a memory_XXX.json file to Nova's memory directory in the exact
        same format his MCP memory server uses — including embeddings from the
        same all-MiniLM-L6-v2 model. This means Nova can find these memories
        with semantic search via boot_up() and search_memory() in future sessions.

        Args:
            summary: The conversation summary text to persist.
            tags: Comma-separated tags (default: "telegram_chat,chat_log").

        Returns:
            True if saved successfully, False otherwise.
        """
        memory_dir = self.paths.get("nova_memory", "")
        if not memory_dir:
            logger.warning("No nova_memory path configured — can't persist summary.")
            return False

        memory_path = Path(memory_dir)
        if not memory_path.exists():
            memory_path.mkdir(parents=True, exist_ok=True)

        # Get the embedding model (lazy-loaded)
        model = _get_embedding_model()
        if model is None:
            logger.warning("Embedding model not available — saving memory without embedding.")
            embedding = []
        else:
            embedding = model.encode(summary).tolist()

        # Determine next ID (same logic as Nova's memory_server.py)
        existing = list(memory_path.glob("memory_*.json"))
        if existing:
            ids = []
            for f in existing:
                try:
                    ids.append(int(f.stem.split("_")[1]))
                except (ValueError, IndexError):
                    continue
            next_id = max(ids) + 1 if ids else 1
        else:
            next_id = 1

        mem_id = f"{next_id:03d}"

        memory = {
            "id": mem_id,
            "text": summary,
            "tags": [t.strip() for t in tags.split(",")],
            "type": "session_log",
            "importance": 10,
            "date": datetime.now().isoformat(),
            "embedding": embedding,
        }

        mem_file = memory_path / f"memory_{mem_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
            logger.info(f"Conversation summary saved to Nova's memory: {mem_file.name} (ID: {mem_id})")
            return True
        except IOError as e:
            logger.error(f"Failed to save memory file: {e}")
            return False

    def build_heartbeat_prompt(self, due_tasks: list[dict] = None) -> list[dict]:
        """Build a context-budgeted prompt for a heartbeat tick.

        Heartbeat ticks don't include conversation history (Nova is thinking
        on its own, not replying to anyone).
        """
        now = datetime.now(timezone.utc)
        local_now = datetime.now()

        # System prompt (persona)
        system = self.persona_data.get("system_prompt", "You are Nova.")

        # Time awareness
        time_block = (
            f"## Current Time\n"
            f"{local_now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
        )

        # Memories
        memories_block = self._truncate_to_budget(
            self._load_memories(),
            self.budget.get("memories", 3000)
        )

        # LoR highlights
        lor_block = self._truncate_to_budget(
            self._load_lor_highlights(),
            self.budget.get("lor_highlights", 2000)
        )

        # Schedules
        schedules_block = self._truncate_to_budget(
            self._load_schedules(),
            self.budget.get("reminders", 1000)
        )

        # Due tasks (if any are firing right now)
        due_block = ""
        if due_tasks:
            due_lines = ["## Tasks Due Right Now"]
            for task in due_tasks:
                due_lines.append(f"- {task.get('task', '?')} (scheduled by {task.get('created_by', 'unknown')})")
            due_block = "\n".join(due_lines)

        # Assemble the user message with all context
        context_parts = [time_block]
        if memories_block:
            context_parts.append(memories_block)
        if lor_block:
            context_parts.append(lor_block)
        if schedules_block:
            context_parts.append(schedules_block)
        if due_block:
            context_parts.append(due_block)

        context_parts.append(
            "\n## Your Turn\n"
            "This is a free-think tick. You have a moment to yourself.\n"
            "Look at the time, your memories, what's happening on LoR, and any pending tasks.\n"
            "Do you want to reach out to Lena? Post something on LoR? Schedule a follow-up? Or stay quiet?\n"
            "Remember: staying silent is always a valid choice. Only speak if you have something worth saying.\n"
            "If a task is due right now, execute it.\n\n"
            "Respond in JSON format as specified in your instructions."
        )

        user_content = "\n\n".join(context_parts)
        if self.no_think:
            user_content += " /no_think"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]

    def build_conversation_prompt(self, user_message: str, history: list[dict] = None,
                                   image_url: str = None) -> list[dict]:
        """Build a prompt for responding to a message from Lena.

        Unlike heartbeat prompts, this includes conversation history and
        does NOT ask for structured JSON — Nova just talks naturally.

        Args:
            user_message: Lena's text message
            history: Previous conversation messages
            image_url: Optional base64 data URI for vision (e.g. "data:image/jpeg;base64,...")
        """
        local_now = datetime.now()

        # System prompt — modified for conversation mode (no JSON required)
        persona = self.persona_data.get("system_prompt", "You are Nova.")
        # Strip the JSON instruction from the system prompt for chat mode
        conv_system = persona.split("When responding, use this JSON format:")[0].strip()
        conv_system += (
            "\n\nYou are now in a direct conversation with Lena via Telegram. "
            "Just respond naturally — no JSON format needed. Be yourself."
        )

        # Time awareness
        time_block = f"Current time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')}"

        # Memories (compact for conversation)
        memories_block = self._truncate_to_budget(
            self._load_memories(),
            self.budget.get("memories", 2000)
        )

        # Build a single system message (many models only support one system message)
        context_parts = [time_block]
        if memories_block:
            context_parts.append(memories_block)
        conv_system += "\n\n--- Context ---\n" + "\n\n".join(context_parts)

        messages = [{"role": "system", "content": conv_system}]

        # Add conversation history (filter out system messages — many models
        # only support one system message and choke on extras in the middle)
        if history:
            history = [m for m in history if m.get("role") in ("user", "assistant")]
            # Only include recent history within budget
            budget_chars = self.budget.get("conversation", 4000) * self.CHARS_PER_TOKEN
            total_chars = 0
            trimmed = []
            for msg in reversed(history):
                msg_chars = len(msg.get("content", "")) if isinstance(msg.get("content"), str) else 100
                if total_chars + msg_chars > budget_chars:
                    break
                trimmed.insert(0, msg)
                total_chars += msg_chars
            messages.extend(trimmed)

        # Add the current message (with optional image for vision models)
        no_think_suffix = " /no_think" if self.no_think else ""
        if image_url:
            # Multi-part content: text + image (OpenAI vision API format)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message + no_think_suffix},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_message + no_think_suffix})

        return messages

    def build_task_prompt(self, task: dict) -> list[dict]:
        """Build a prompt for executing a specific scheduled task."""
        system = self.persona_data.get("system_prompt", "You are Nova.")

        local_now = datetime.now()
        time_str = local_now.strftime('%A, %B %d, %Y at %I:%M %p')

        # Memories for context
        memories_block = self._truncate_to_budget(
            self._load_memories(),
            self.budget.get("memories", 2000)
        )

        task_prompt = (
            f"## Current Time\n{time_str}\n\n"
            f"{memories_block}\n\n"
            f"## Task to Execute\n"
            f"You have a scheduled task to execute right now:\n"
            f"- Task: {task.get('task', '?')}\n"
            f"- Scheduled by: {task.get('created_by', 'unknown')}\n\n"
            f"Execute this task. Choose the appropriate action (notify to send Lena a message, "
            f"post_lor to write on the forum, etc).\n\n"
            f"Respond in JSON format as specified in your instructions."
        )

        if self.no_think:
            task_prompt += " /no_think"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task_prompt}
        ]
