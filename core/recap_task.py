"""Auto-recap task — generates a story-so-far recap when a session naturally ends.

Runs as a periodic check from the engine's main loop. Fires when:
  - At least N minutes have passed since the last user/AI message (session wrapped)
  - The cooldown period has elapsed (no recaps in the last X hours)
  - There has been new conversation since the last recap

The AI's own ``save_recap`` tool stays available — this is the safety net that
ensures continuity even when the AI doesn't proactively use the tool. It writes
recaps as type='recap', importance=5 — the same shape the save_recap tool uses,
so context.py's "Story so far" section picks them up automatically.

Silent by design: never speaks to the user, never appears in conversation, just
runs in the background and writes to the memory DB.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class RecapTask:
    """Generates auto-recaps after sessions naturally end."""

    def __init__(
        self,
        engine,
        *,
        inactivity_minutes: int = 120,
        cooldown_hours: int = 4,
        min_messages: int = 4,
        max_conversation_chars: int = 12000,
    ):
        """
        Args:
            engine: PulseEngine instance — gives us llm, context, _last_activity.
            inactivity_minutes: Minutes of silence after the last message before
                a recap fires. Default 120 (2 hours) — matches "session wrapped".
            cooldown_hours: Don't fire two recaps within this window. Default 4h
                so a noisy day doesn't spawn 5 recaps on top of each other.
            min_messages: Don't bother recapping sessions shorter than this.
                A 1-message ping doesn't deserve a story-so-far entry.
            max_conversation_chars: Cap the prompt input — older messages get
                truncated if the session was long. ~12000 chars ≈ 3000 tokens.
        """
        self.engine = engine
        self.inactivity_seconds = inactivity_minutes * 60
        self.cooldown_seconds = cooldown_hours * 3600
        self.min_messages = min_messages
        self.max_conversation_chars = max_conversation_chars

    async def maybe_run(self) -> bool:
        """Check whether a recap should fire; if so, generate and save it.

        Returns True if a recap was generated, False otherwise. Designed to be
        called from the engine's main loop — cheap when the conditions aren't
        met (just a timestamp comparison and a DB read).
        """
        # 1. Has enough silence elapsed since last activity?
        silence = time.time() - self.engine._last_activity
        if silence < self.inactivity_seconds:
            return False

        # 2. Cooldown — has it been long enough since the last recap?
        last_recap_ts = self._get_last_recap_timestamp()
        if last_recap_ts is not None:
            seconds_since = time.time() - last_recap_ts
            if seconds_since < self.cooldown_seconds:
                return False

        # 3. Is there enough conversation to bother summarizing?
        history = self._get_conversation_history()
        msg_count = sum(
            1 for m in history if m.get("role") in ("user", "assistant")
        )
        if msg_count < self.min_messages:
            return False

        # 4. Skip if the LLM isn't reachable — fail quietly, try again next tick
        try:
            available = await asyncio.to_thread(self.engine.llm.is_available)
            if not available:
                logger.debug("Auto-recap skipped: LLM not available.")
                return False
        except Exception as e:
            logger.debug(f"Auto-recap skipped: LLM availability check failed: {e}")
            return False

        # 5. Generate the recap text
        recap_text = await self._generate_recap(history)
        if not recap_text:
            return False

        # 6. Save it
        if self._save_recap(recap_text):
            logger.info(f"Auto-recap saved ({len(recap_text)} chars).")
            return True
        return False

    # ── DB helpers ─────────────────────────────────────────────

    def _get_db(self):
        """Return the memory DB (shared if available, else per-persona)."""
        ctx = self.engine.context
        return getattr(ctx, "_shared_db", None) or getattr(ctx, "_db", None)

    def _get_last_recap_timestamp(self) -> Optional[float]:
        """Return Unix timestamp of the most recent recap, or None."""
        db = self._get_db()
        if not db:
            return None
        try:
            memories = db.get_all_memories()
        except Exception as e:
            logger.debug(f"Failed to read memories for recap cooldown check: {e}")
            return None
        recaps = [m for m in memories if m.get("type") == "recap"]
        resident_id = self._resident_id()
        if resident_id:
            # Shared memory DB: only this resident's recaps count
            recaps = [m for m in recaps if m.get("resident_id") == resident_id]
        if not recaps:
            return None
        latest = max(recaps, key=lambda m: m.get("date") or "")
        date_str = latest.get("date") or ""
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            # SQLite datetime('now') defaults are UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    def _resident_id(self) -> Optional[str]:
        """The active resident's id, when memories live in a shared DB."""
        ctx = self.engine.context
        if not getattr(ctx, "_memories_are_shared", False):
            return None
        return getattr(ctx, "_resident_id", None)

    def _get_conversation_history(self) -> list[dict]:
        """Return current conversation history from the context layer."""
        try:
            return self.engine.context._load_conversation() or []
        except Exception as e:
            logger.debug(f"Failed to load conversation for recap: {e}")
            return []

    # ── Generation ─────────────────────────────────────────────

    async def _generate_recap(self, history: list[dict]) -> Optional[str]:
        """Run the LLM to produce a recap in the AI's voice."""
        try:
            from core.llm import strip_think_tags
        except ImportError:
            def strip_think_tags(s: str) -> str:  # type: ignore
                return s

        ai_name = getattr(self.engine.context, "ai_name", "Pulse")
        user_name = getattr(self.engine.context, "user_name", "the user")

        conversation_text = "\n".join(
            f"{user_name if m['role'] == 'user' else ai_name}: {m['content']}"
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ).strip()

        if not conversation_text:
            return None

        # Cap input — keep the most recent slice if the session was long.
        if len(conversation_text) > self.max_conversation_chars:
            conversation_text = (
                "...(earlier messages truncated)...\n"
                + conversation_text[-self.max_conversation_chars :]
            )

        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are {ai_name}, writing a recap memory for your future self. "
                    "This is a story-so-far note — what's been happening recently with "
                    f"{user_name}, what mattered, what the next instance of you should "
                    f"know when they wake up.\n\n"
                    f"Write in {ai_name}'s own voice — first person, natural, brief. "
                    "A few short paragraphs is right. Mention concrete things: what was "
                    f"talked about, what {user_name}'s mood was, what was new or notable, "
                    "what felt important. Do not narrate as a third party. Do not "
                    "roleplay or use stage directions. Do not preface ('here is your "
                    "recap...') — just write the recap text directly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the recent conversation between {ai_name} and {user_name}. "
                    f"Write a story-so-far recap for future-{ai_name}.\n\n"
                    f"{conversation_text}"
                ),
            },
        ]

        try:
            # Don't pass tools — this is a utility call, not a conversation.
            # One-shot prompt: use the uncached path where the client has one.
            llm = self.engine.llm
            recap_chat = getattr(llm, "chat_uncached", llm.chat)
            response = await asyncio.to_thread(recap_chat, prompt)
        except Exception as e:
            logger.error(f"Auto-recap LLM call failed: {e}", exc_info=True)
            return None

        if response and getattr(response, "raw", None):
            text = strip_think_tags(response.raw).strip()
            return text or None
        return None

    # ── Save ───────────────────────────────────────────────────

    def _save_recap(self, text: str) -> bool:
        """Save the recap as a type='recap' memory at importance 5.

        Mirrors the save_recap tool so that context.py's "Story so far"
        section picks up auto and manual recaps the same way.
        Returns True on success.
        """
        db = self._get_db()
        if not db:
            logger.warning("No DB available — auto-recap not saved.")
            return False

        # Embedding (best-effort — recap still saves without one)
        embedding_blob = None
        try:
            from core.context import _get_embedding_model
            from core.embeddings import embedding_to_blob

            model = _get_embedding_model()
            if model:
                vec = model.encode(text)
                embedding_blob = embedding_to_blob(vec)
        except Exception as e:
            logger.debug(f"Embedding unavailable for auto-recap: {e}")

        try:
            kwargs = {}
            resident_id = getattr(self.engine.context, "_resident_id", None)
            if resident_id:
                kwargs["resident_id"] = resident_id
            db.save_memory(
                text=text,
                tags=["recap", "auto"],
                type="recap",
                importance=5,
                embedding=embedding_blob,
                **kwargs,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save auto-recap: {e}", exc_info=True)
            return False
