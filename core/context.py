"""
Context manager - builds token-budgeted prompts for Nova.

Reads from multiple data sources (memories, LoR, schedules, conversation history)
and assembles a prompt that fits within the model's context window.
"""

import json
import logging
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContextManager:
    """Assembles context for each LLM call within a token budget."""

    # Rough token estimation: ~4 chars per token for English text
    CHARS_PER_TOKEN = 4

    def __init__(self, config: dict):
        self.config = config
        self.budget = config.get("context_budget", {})
        self.paths = config.get("paths", {})
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

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(context_parts)}
        ]

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

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task_prompt}
        ]
