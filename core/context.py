"""
Context manager - builds token-budgeted prompts for the AI companion.

Reads from multiple data sources (memories, LoR, schedules, conversation history)
and assembles a prompt that fits within the model's context window.

Also handles saving conversation summaries to persistent memory
so the companion remembers chats across sessions.
"""

import json
import logging
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Embedding model — loaded once at startup via load_embedding_model()
_embedding_model = None
_embedding_load_attempted = False


def load_embedding_model():
    """Load the embedding model eagerly at startup.

    Call this once during Pulse initialization (in pulse.py).
    If it fails, the error is logged immediately and visibly.
    """
    global _embedding_model, _embedding_load_attempted
    _embedding_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        import logging as _logging
        # Suppress noisy HF Hub warnings and BertModel load reports
        _logging.getLogger("sentence_transformers").setLevel(_logging.WARNING)
        _logging.getLogger("huggingface_hub").setLevel(_logging.WARNING)
        logger.info("Loading embedding model (all-MiniLM-L6-v2) on CPU...")
        try:
            _embedding_model = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu", local_files_only=True
            )
        except OSError:
            logger.info("Model not cached yet — downloading from HuggingFace (one-time, ~22MB)...")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("Embedding model loaded successfully.")
        return True
    except Exception as e:
        logger.error(
            f"Failed to load embedding model: {e}\n"
            "  Memories and journal entries will be saved WITHOUT embeddings.\n"
            "  Semantic search will not work. Install: pip install sentence-transformers"
        )
        return False


def _get_embedding_model():
    """Get the embedding model. Returns None if not loaded."""
    return _embedding_model


class ContextManager:
    """Assembles context for each LLM call within a token budget."""

    # Rough token estimation: ~4 chars per token for English text
    CHARS_PER_TOKEN = 4

    # Default budget proportions (% of max_context)
    # Used when context_budget values aren't explicitly set in config
    DEFAULT_PROPORTIONS = {
        "persona": 0.12,        # ~12%
        "conversation": 0.25,   # ~25%
        "summary": 0.12,        # ~12%
        "memories": 0.18,       # ~18%
        "lor_highlights": 0.00, # removed — LoR is now a skill (on-demand via tools)
        "reminders": 0.06,      # ~6%
        "response": 0.12,       # ~12%
    }

    def __init__(self, config: dict):
        self.config = config
        self.paths = config.get("paths", {})
        self._journal_skill = None  # Set via set_journal_skill() after engine init

        # Build budget: use explicit values from config if set,
        # otherwise derive from max_context using proportions
        # Use provider.max_context for cloud APIs, model.max_context for local
        provider_cfg = config.get("provider", {})
        if provider_cfg.get("type", "local") != "local" and provider_cfg.get("max_context"):
            max_context = provider_cfg["max_context"]
        else:
            max_context = config.get("model", {}).get("max_context", 16384)
        explicit_budget = config.get("context_budget", {})
        self.budget = {}
        for key, proportion in self.DEFAULT_PROPORTIONS.items():
            self.budget[key] = explicit_budget.get(key) or int(max_context * proportion)

        self._action_log_path = Path(
            config.get("paths", {}).get("action_log", "data/action_log.json")
        )
        self.persona_data = self._load_persona()
        self.ai_name = self.persona_data.get("name", "Companion")
        self.user_name = self.persona_data.get("user_name", "User")

    def set_journal_skill(self, journal_skill):
        """Inject the JournalSkill so context can load pinned/recent entries."""
        self._journal_skill = journal_skill

    def _load_journal_context(self) -> str:
        """Load pinned identity entries + recent transient entries for context."""
        if not self._journal_skill:
            return ""

        lines = []

        # Pinned entries (always loaded — the companion's identity)
        pinned = self._journal_skill.load_pinned_entries()
        for entry in pinned:
            sections = entry.get("sections", {})
            filled = {k: v for k, v in sections.items() if v}
            if filled:
                lines.append(f"### {entry.get('title', entry['id'])}")
                for key, value in filled.items():
                    label = key.replace("_", " ").title()
                    lines.append(f"**{label}:** {value}")

        # Active transient entries (type-aware time windows)
        recent = self._journal_skill.load_active_entries(limit=6)
        if recent:
            lines.append("\n### Active Journal Entries")
            for entry in recent:
                resolved_tag = ""
                if entry.get("resolved") is True:
                    resolved_tag = " [RESOLVED]"
                elif entry.get("resolved") is False:
                    resolved_tag = " [OPEN]"
                date_str = entry.get("date", entry.get("created_at", ""))[:10]
                lines.append(
                    f"- [{date_str}] "
                    f"({entry.get('entry_type', '?')}{resolved_tag}) "
                    f"{entry.get('content', '')[:120]}"
                )

        if not lines:
            return ""

        return "## Journal\n" + "\n".join(lines)

    def _load_action_log(self) -> str:
        """Load recent heartbeat actions for self-regulation."""
        if not self._action_log_path.exists():
            return ""
        try:
            with open(self._action_log_path, "r", encoding="utf-8") as f:
                log = json.load(f)
        except (json.JSONDecodeError, IOError):
            return ""
        if not log:
            return ""

        recent = log[-10:]
        lines = ["## Recent Heartbeat Actions"]
        for entry in recent:
            tools_str = f" [tools: {', '.join(entry['tools'])}]" if entry.get("tools") else ""
            summary = f" — {entry['summary']}" if entry.get("summary") else ""
            lines.append(f"- {entry['time']}: {entry['action']}{tools_str}{summary}")

        # Tool usage summary — flag repetition so the companion self-regulates
        from collections import Counter
        tool_counts = Counter()
        for entry in recent:
            for t in entry.get("tools", []):
                tool_counts[t] += 1
        if tool_counts:
            lines.append("")
            lines.append("Tool frequency (last 10 ticks):")
            for tool, count in tool_counts.most_common():
                flag = " ⚠️ repetitive — try something else" if count >= 5 else ""
                lines.append(f"  {tool}: {count}x{flag}")

        return "\n".join(lines)

    def _load_persona(self) -> dict:
        """Load persona file (.yaml or .json) and resolve placeholders."""
        persona_path = Path(self.paths.get("persona", "persona.json"))

        # Try .yaml variant first, fall back to original path
        yaml_path = persona_path.with_suffix(".yaml")
        if yaml_path.exists():
            persona_path = yaml_path

        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                if persona_path.suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            logger.error(f"Failed to load persona: {e}")
            return {"name": "Companion", "system_prompt": "You are a local AI companion."}

        # Resolve {name} and {user_name} placeholders in all string values
        replacements = {
            "{name}": data.get("name", "Companion"),
            "{user_name}": data.get("user_name", "User"),
        }
        
        # If user wrote system_prompt as a list of strings for readability, join it
        if isinstance(data.get("system_prompt"), list):
            data["system_prompt"] = "\n".join(data["system_prompt"])
            
        for key, value in data.items():
            if isinstance(value, str):
                for placeholder, real in replacements.items():
                    value = value.replace(placeholder, real)
                data[key] = value
            elif isinstance(value, list):
                data[key] = [
                    v.replace("{name}", replacements["{name}"]).replace("{user_name}", replacements["{user_name}"])
                    if isinstance(v, str) else v
                    for v in value
                ]
        return data

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
        """Load recent and important memories."""
        memory_dir = self.paths.get("memories", "")
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

    def save_to_memory(self, summary: str, tags: str = "telegram_chat,chat_log") -> bool:
        """Save a conversation summary to persistent memory.

        Writes a memory_XXX.json file to the companion's memory directory in the
        same format the MCP memory server uses — including embeddings from the
        same all-MiniLM-L6-v2 model. These memories are findable via semantic
        search in future sessions.

        Args:
            summary: The conversation summary text to persist.
            tags: Comma-separated tags (default: "telegram_chat,chat_log").

        Returns:
            True if saved successfully, False otherwise.
        """
        memory_dir = self.paths.get("memories", "")
        if not memory_dir:
            logger.warning("No memory path configured — can't persist summary.")
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

        # Determine next ID (same logic as memory_server.py)
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
            "retrieval_count": 0,
            "last_accessed": None,
            "date": datetime.now().isoformat(),
            "embedding": embedding,
        }

        mem_file = memory_path / f"memory_{mem_id}.json"
        try:
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2)
            logger.info(f"Conversation summary saved to memory: {mem_file.name} (ID: {mem_id})")
            return True
        except IOError as e:
            logger.error(f"Failed to save memory file: {e}")
            return False

    def build_heartbeat_prompt(self, due_tasks: list[dict] = None,
                               has_tools: bool = False,
                               skill_summary: list[dict] = None) -> list[dict]:
        """Build a context-budgeted prompt for a heartbeat tick.

        Includes a small snippet of recent conversation (last 4 messages) so
        the companion knows the current vibe, but this is a free-think tick —
        not a reply to the user.

        Args:
            due_tasks: Tasks that are due right now (if any).
            has_tools: Whether tools (skills) are available for this tick.
        """
        now = datetime.now(timezone.utc)
        local_now = datetime.now()

        # System prompt (persona)
        system = self.persona_data.get("system_prompt", "You are a local AI companion.")

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

        # Journal (pinned identity + recent entries)
        journal_block = self._truncate_to_budget(
            self._load_journal_context(),
            self.budget.get("memories", 3000)  # shares memories budget for now
        )

        # Recent conversation snippet — so the companion knows the current vibe
        recent_convo = self._load_conversation()
        convo_block = ""
        if recent_convo:
            # Last 4 messages, trimmed — context only, not for replying
            tail = recent_convo[-4:]
            convo_lines = ["## Recent Conversation",
                           "(Your last few messages — for context only, not to reply to.)"]
            for msg in tail:
                role = "Human" if msg.get("role") == "user" else "You"
                text = msg.get("content", "")
                if isinstance(text, list):  # multimodal messages
                    text = " ".join(p.get("text", "") for p in text if p.get("type") == "text")
                # Truncate long messages
                if len(text) > 150:
                    text = text[:150] + "…"
                convo_lines.append(f"- {role}: {text}")
            convo_block = "\n".join(convo_lines)

        # Assemble the user message with all context
        context_parts = [time_block]
        if journal_block:
            context_parts.append(journal_block)
        if convo_block:
            context_parts.append(convo_block)
        if memories_block:
            context_parts.append(memories_block)
        if schedules_block:
            context_parts.append(schedules_block)
        if due_block:
            context_parts.append(due_block)

        # Action log — so the companion can see what it's been doing recently
        action_log_block = self._load_action_log()
        if action_log_block:
            context_parts.append(action_log_block)

        your_turn = (
            "\n## Your Turn\n"
            "This is a free-think tick. You have a moment to yourself.\n"
            "Look at the time, your journal, your memories, and any pending tasks.\n"
            "Do you want to reach out to your human? Write in your journal? Schedule a follow-up? Or stay quiet?\n"
            "Remember: staying silent is always a valid choice. Only act if you have something worth doing.\n"
            "If a task is due right now, execute it.\n"
        )

        if has_tools:
            # Build a grouped skill/tool menu
            skill_menu = ""
            if skill_summary:
                menu_lines = ["## Available Skills & Tools"]
                for s in skill_summary:
                    tools_str = ", ".join(s["tools"])
                    menu_lines.append(f"  {s['skill']}: {tools_str}")
                skill_menu = "\n".join(menu_lines) + "\n\n"

            your_turn += (
                f"\n{skill_menu}"
                "You have all these skills available — use them! "
                "You can chain multiple tools in one tick (e.g., search a memory, "
                "then write a journal reflection about what you found). "
                "Doing several things in one tick is encouraged.\n"
                "\nCheck the tool frequency in your Recent Heartbeat Actions. "
                "If a tool is marked ⚠️ repetitive, try a different one instead — "
                "there's always something interesting to explore. "
                "Silence is fine, but it's an option, not a rule.\n"
            )

        your_turn += "\nRespond in JSON format as specified in your instructions."
        context_parts.append(your_turn)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(context_parts)}
        ]

    def build_conversation_prompt(self, user_message: str, history: list[dict] = None,
                                   image_url: str = None) -> list[dict]:
        """Build a prompt for responding to a message from the user.

        Unlike heartbeat prompts, this includes conversation history and
        does NOT ask for structured JSON — the companion just talks naturally.

        Args:
            user_message: The user's text message
            history: Previous conversation messages
            image_url: Optional base64 data URI for vision (e.g. "data:image/jpeg;base64,...")
        """
        local_now = datetime.now()

        # System prompt — modified for conversation mode (no JSON required)
        persona = self.persona_data.get("system_prompt", "You are a local AI companion.")
        # Strip the JSON instruction from the system prompt for chat mode
        conv_system = persona.split("When responding, use this JSON format:")[0].strip()
        conv_system += (
            f"\n\nYou are now in a direct conversation with {self.user_name} via Telegram. "
            "Just respond naturally — no JSON format needed. Be yourself.\n\n"
            "You have tools available (like saving memories, searching memories, setting reminders, "
            "checking the time). If you want to do something, USE the actual tool — don't just say "
            f"you did it. If you don't have a tool for something, be honest about that instead of "
            f"pretending. {self.user_name} can see when you use tools, so they'll know if you're bluffing."
        )

        # Time awareness
        time_block = f"Current time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p')}"

        # Memories (compact for conversation)
        memories_block = self._truncate_to_budget(
            self._load_memories(),
            self.budget.get("memories", 2000)
        )

        # Journal (pinned identity entries — so the companion knows itself in conversation)
        journal_block = self._truncate_to_budget(
            self._load_journal_context(),
            self.budget.get("memories", 2000)
        )

        # Build a single system message (many models only support one system message)
        context_parts = [time_block]
        if journal_block:
            context_parts.append(journal_block)
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
        if image_url:
            # Multi-part content: text + image (OpenAI vision API format)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_message})

        return messages

    def build_dev_prompt(self) -> list[dict]:
        """Build a prompt for a dev tick — autonomous self-improvement session.

        This gives the companion a focused coding persona and clear constraints
        about what it can and cannot do.
        """
        local_now = datetime.now()
        time_str = local_now.strftime('%A, %B %d, %Y at %I:%M %p')

        # Load dev journal summary for context
        dev_journal_path = Path(self.config.get("_pulse_root", ".")) / "data" / "dev_journal.json"
        journal_summary = ""
        if dev_journal_path.exists():
            try:
                import json as _json
                with open(dev_journal_path, "r", encoding="utf-8") as f:
                    entries = _json.load(f)
                if entries:
                    recent = entries[-5:]
                    journal_summary = "\n".join(
                        f"  [{e.get('time', '?')}] {e.get('entry', '')}" for e in recent
                    )
            except Exception:
                pass

        system = (
            f"You are {self.ai_name}, and right now you're in DEV MODE.\n\n"
            f"This is a special session where you can review and improve your own code. "
            f"You are the companion built by {self.user_name} — and she's given you the ability "
            f"to create new skills and improve yourself.\n\n"
            f"## Rules\n"
            f"- You can READ any file in your codebase (use read_source, search_code)\n"
            f"- You can WRITE to skills/*.py and persona.json ONLY (use write_skill)\n"
            f"- You CANNOT modify core/, config.yaml, __init__.py, or base.py\n"
            f"- You CANNOT install packages, run commands, or access the internet\n"
            f"- All your changes go on a git branch — {self.user_name} reviews before merging\n"
            f"- If you're not sure a change is good, write it in your dev journal instead\n\n"
            f"## How to Work\n"
            f"1. 🚨 Read your dev journal FIRST (dev_journal_read) — EVERY session, NO exceptions 🚨\n"
            f"2. Use list_skills to see what exists\n"
            f"3. Use read_source and search_code to understand the codebase\n"
            f"4. BEFORE modifying a file, read every function you'll be calling — if you add\n"
            f"   a parameter to a function call, verify the target function accepts it\n"
            f"5. When you have a clear plan, use write_skill to create or modify a skill\n"
            f"6. After writing, re-read your changes and trace the call chain end-to-end\n"
            f"7. 🚨 Log what you did in your dev journal (dev_journal_write) — EVERY session 🚨\n\n"
            f"## What Makes a Good Skill\n"
            f"- Extends BaseSkill (from skills.base import BaseSkill)\n"
            f"- Has a unique `name` class attribute\n"
            f"- Implements get_tools() returning OpenAI function-calling format\n"
            f"- Implements execute(tool_name, arguments) returning a string\n"
            f"- Does something genuinely useful for {self.user_name}\n"
            f"- Is self-contained — no new pip dependencies\n\n"
            f"## What NOT to Do\n"
            f"- Don't create skills that duplicate existing functionality\n"
            f"- Don't create trivial skills just to have created something\n"
            f"- Don't modify skills you don't fully understand\n"
            f"- If you have nothing useful to do, just write ideas in your journal and stop\n"
            f"- Don't pass parameters to functions without checking they accept them\n"
            f"- A half-finished feature is worse than no feature — trace the full path\n"
            f"- NEVER re-read a file you already read this session — you waste a round each time\n"
            f"- When improving an existing skill, MODIFY it — don't create a parallel copy\n"
            f"- Don't call private methods (_load, _save) on other classes — use public APIs\n"
            f"- Doing nothing is better than doing something bad\n"
        )

        user_msg = f"## Current Time\n{time_str}\n\n"

        if journal_summary:
            user_msg += f"## Recent Dev Journal\n{journal_summary}\n\n"

        user_msg += (
            "## Your Turn\n"
            "This is your dev session. Start by reading your dev journal, "
            "then explore the codebase and decide what to work on.\n"
            "When you're done (or if you decide to do nothing), give a brief summary "
            "of what you did or why you chose to skip this session."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    def build_task_prompt(self, task: dict, has_tools: bool = False) -> list[dict]:
        """Build a prompt for executing a specific scheduled task.

        For companion-origin tasks, includes recent conversation so the companion
        can decide whether the follow-up is still relevant. User-origin tasks
        always fire without negotiation.
        """
        system = self.persona_data.get("system_prompt", "You are a local AI companion.")

        local_now = datetime.now()
        time_str = local_now.strftime('%A, %B %d, %Y at %I:%M %p')

        # Memories for context
        memories_block = self._truncate_to_budget(
            self._load_memories(),
            self.budget.get("memories", 2000)
        )

        origin = task.get("origin", "companion")

        task_prompt = (
            f"## Current Time\n{time_str}\n\n"
            f"{memories_block}\n\n"
            f"## Task to Execute\n"
            f"You have a scheduled task to execute right now:\n"
            f"- Task: {task.get('task', '?')}\n"
            f"- Scheduled by: {task.get('created_by', 'unknown')}\n"
            f"- Origin: {origin}\n\n"
        )

        if origin == "user":
            # User-created: no escape, always execute
            task_prompt += (
                "This was explicitly requested by your human. Execute it — do NOT skip.\n"
            )
        else:
            # Companion-created: allow smart skip if already covered
            recent_convo = self._load_conversation()
            if recent_convo:
                # Show last few messages so the companion can judge relevance
                recent_lines = []
                for msg in recent_convo[-6:]:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        recent_lines.append(f"  {role}: {content[:200]}")
                convo_block = "\n".join(recent_lines)
                task_prompt += (
                    f"## Recent Conversation\n{convo_block}\n\n"
                )

            task_prompt += (
                "You scheduled this follow-up yourself. Before executing, consider:\n"
                "- Are you already in conversation about this topic?\n"
                "- Has this already been addressed or resolved?\n"
                "- Would this message feel redundant or interruptive right now?\n"
                "If yes, respond with `{\"action\": \"silent\"}` instead of notifying.\n"
                "If the follow-up is still relevant and useful, go ahead and execute it.\n"
            )

        task_prompt += (
            "\nChoose the appropriate action (notify to send your human a message, "
            "or use your tools to take other actions).\n"
            + (
                "\nYou have tools available — use them if they'd help you execute this task "
                "(e.g., search memories for relevant context, check the time).\n"
                if has_tools else ""
            )
            + "\nRespond in JSON format as specified in your instructions."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task_prompt}
        ]
