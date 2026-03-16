"""
Pulse Engine - the heartbeat loop.

This is the main orchestrator. It:
1. Runs a heartbeat timer (free-think ticks)
2. Checks for due scheduled tasks
3. Builds context and prompts Nova via llama-server
4. Dispatches actions to channels (LoR, toast, etc.)
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

from core.llm import LLMClient, PulseResponse
from core.context import ContextManager
from core.scheduler import ScheduleManager

logger = logging.getLogger(__name__)


class PulseEngine:
    """Main engine that drives the heartbeat loop."""

    def __init__(self, config: dict, channels: dict, skill_registry=None,
                 llm_endpoint: str = None):
        """
        Args:
            config: Parsed config.yaml
            channels: Dict of channel_name -> Channel instance
            skill_registry: Optional SkillRegistry for tool-calling in conversations
            llm_endpoint: OpenAI-compatible API endpoint (from LlamaServer)
        """
        self.config = config
        self.channels = channels
        self.skill_registry = skill_registry

        # Initialize core components
        model_config = config.get("model", {})
        server_config = config.get("server", {})
        endpoint = llm_endpoint or f"http://{server_config.get('host', '127.0.0.1')}:{server_config.get('port', 8012)}/v1"
        self.llm = LLMClient(
            endpoint=endpoint,
            model_name=model_config.get("model_file", "default"),
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_response_tokens", 1024),
            frequency_penalty=model_config.get("frequency_penalty", 0.0),
            presence_penalty=model_config.get("presence_penalty", 0.0),
        )

        self.context = ContextManager(config)
        self.scheduler = ScheduleManager(
            config.get("paths", {}).get("schedules", "data/schedules.json")
        )

        heartbeat_config = config.get("heartbeat", {})
        self.interval = heartbeat_config.get("interval_minutes", 120) * 60  # seconds
        self.quiet_start = heartbeat_config.get("quiet_hours_start", 23)
        self.quiet_end = heartbeat_config.get("quiet_hours_end", 8)
        self.startup_checkin = heartbeat_config.get("startup_checkin", True)

        self._running = False
        self._stop_event = asyncio.Event()

        # Rate limiting — prevent Nova from flooding channels
        self._last_lor_post = 0     # timestamp of last LoR post
        self._last_notify = 0       # timestamp of last notification
        self._lor_cooldown = heartbeat_config.get("lor_cooldown_minutes", 120) * 60  # default 2 hours
        self._notify_cooldown = heartbeat_config.get("notify_cooldown_minutes", 60) * 60  # default 1 hour

    async def _interruptible_sleep(self, seconds: int):
        """Sleep that can be interrupted instantly by stop()."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass  # Normal — timeout means we slept the full duration

    def _in_quiet_hours(self) -> bool:
        """Check if we're in quiet hours."""
        hour = datetime.now().hour
        if self.quiet_start > self.quiet_end:
            # Wraps around midnight (e.g., 23-8)
            return hour >= self.quiet_start or hour < self.quiet_end
        else:
            return self.quiet_start <= hour < self.quiet_end

    async def _dispatch(self, response: PulseResponse, is_scheduled_task: bool = False):
        """Route a response to the appropriate channel.

        Rate limits apply to free-think ticks but NOT to scheduled tasks
        (those were explicitly requested, so they should always go through).
        """
        now = time.time()

        if response.action == "silent":
            logger.info("Nova chose to stay silent.")
            return

        if response.action == "notify":
            # Rate limit notifications (unless it's a scheduled task)
            if not is_scheduled_task and (now - self._last_notify) < self._notify_cooldown:
                mins_left = int((self._notify_cooldown - (now - self._last_notify)) / 60)
                logger.info(f"Notification rate-limited ({mins_left}m cooldown remaining). Skipping.")
                return
            # Send via toast AND Telegram (if available)
            sent = False
            telegram = self.channels.get("telegram")
            if telegram:
                await telegram.send(response.message)
                sent = True
            toast = self.channels.get("toast")
            if toast:
                await toast.send(response.message)
                sent = True
            if sent:
                self._last_notify = now
                logger.info(f"Notification sent: {response.message[:50]}...")
            else:
                logger.info(f"[No notification channels] {response.message[:80]}...")

        elif response.action == "post_lor":
            # Rate limit LoR posts (unless it's a scheduled task)
            if not is_scheduled_task and (now - self._last_lor_post) < self._lor_cooldown:
                mins_left = int((self._lor_cooldown - (now - self._last_lor_post)) / 60)
                logger.info(f"LoR post rate-limited ({mins_left}m cooldown remaining). Skipping.")
                return
            lor = self.channels.get("lor")
            if lor:
                await lor.send(
                    response.message,
                    category=response.lor_category,
                    title=response.lor_title
                )
                self._last_lor_post = now
                logger.info(f"Posted to LoR: {response.lor_title or response.message[:50]}...")
            else:
                logger.info(f"[LoR disabled] {response.message[:80]}...")

        elif response.action == "schedule":
            if response.schedule_task and response.schedule_when:
                try:
                    entry = self.scheduler.add(
                        task=response.schedule_task,
                        created_by="nova-self",
                        when=response.schedule_when
                    )
                    logger.info(f"Nova self-scheduled: {entry['id']} — {response.schedule_task}")
                except ValueError as e:
                    logger.warning(f"Nova tried to self-schedule without a valid time: {e}")

        # Handle combined actions (e.g., notify AND schedule)
        if response.schedule_task and response.schedule_when and response.action != "schedule":
            try:
                entry = self.scheduler.add(
                    task=response.schedule_task,
                    created_by="nova-self",
                    when=response.schedule_when
                )
                logger.info(f"Nova also scheduled: {entry['id']} — {response.schedule_task}")
            except ValueError as e:
                logger.warning(f"Nova tried to schedule without a valid time: {e}")

    async def handle_message(self, message: str, source: str = "telegram",
                            image_url: str = None) -> tuple[str, list[str]]:
        """Handle an incoming message from Lena and return Nova's response.

        This is for bidirectional chat (Telegram, future voice, etc.).
        Unlike heartbeat ticks, Nova responds naturally without JSON structure.

        All LLM calls run via asyncio.to_thread() so they don't block the
        event loop — this keeps Telegram polling alive during inference.

        Args:
            message: Lena's message text
            source: Where the message came from ("telegram", "voice", etc.)
            image_url: Optional base64 data URI for an image (vision models)

        Returns:
            Tuple of (Nova's response text, list of tools used). Text is None if LLM unavailable.
        """
        # is_available() is a sync HTTP call — run in thread to not block
        available = await asyncio.to_thread(self.llm.is_available)
        if not available:
            logger.warning(f"LLM server not available — can't respond to {source} message.")
            return None, []

        # Load conversation history
        history = self.context._load_conversation()

        # Build conversation prompt (with optional image)
        messages = self.context.build_conversation_prompt(message, history, image_url=image_url)
        logger.info(f"Sending {len(messages)} messages to LLM for {source} reply"
                     f"{' (with image)' if image_url else ''}...")

        # Get response from Nova — run in thread so we don't block the event loop!
        # If skills are available, use tool-calling mode; otherwise plain chat.
        tools_used = []
        tools = self.skill_registry.get_all_tools() if self.skill_registry else []
        if tools:
            logger.info(f"Tool-calling mode: {len(tools)} tools available")
            reply, tools_used = await asyncio.to_thread(
                self.llm.chat_with_tools, messages, tools, self.skill_registry
            )
        else:
            response = await asyncio.to_thread(self.llm.chat, messages)
            reply = response.raw.strip() if response and response.raw else None

        if not reply:
            return None, tools_used

        # Safety net: strip <think> tags that reasoning models may include
        from core.llm import strip_think_tags
        reply = strip_think_tags(reply)
        if not reply:
            return None, tools_used

        # Update conversation history (store text only — images are too large to persist)
        # Timestamps help Nova understand temporal flow (so he knows it's 2 PM, not bedtime)
        timestamp = datetime.now().strftime("[%I:%M %p]")
        image_note = " [sent an image]" if image_url else ""
        history.append({"role": "user", "content": f"{timestamp} {message}{image_note}"})
        history.append({"role": "assistant", "content": reply})

        # Auto-summarize if history is getting long
        max_history = 20  # messages before we summarize
        if len(history) > max_history:
            logger.info("Conversation history is long — summarizing...")
            summary = await self._summarize_conversation(history)
            if summary:
                # Save to Nova's persistent memory so he remembers across sessions
                try:
                    self.context.save_to_nova_memory(summary)
                except Exception as e:
                    logger.warning(f"Memory persistence failed (non-fatal): {e}")
                history = [{"role": "user", "content": f"[Context] Previous conversation summary: {summary}"},
                           {"role": "assistant", "content": "Got it, I remember. Let's continue."}]
            else:
                # Summarization failed — trim history to prevent unbounded growth
                logger.warning("Summarization failed — trimming conversation to last 10 messages.")
                history = history[-10:]

        self.context.save_conversation(history)
        logger.info(f"Replied to {source}: {reply[:50]}...")
        return reply, tools_used

    async def _summarize_conversation(self, history: list[dict]) -> str:
        """Ask Nova to summarize the conversation so far (for context compression)."""
        from core.llm import strip_think_tags

        conversation_text = "\n".join(
            f"{'Lena' if m['role'] == 'user' else 'Nova'}: {m['content']}"
            for m in history if m['role'] in ('user', 'assistant')
        )

        summary_prompt = [
            {"role": "system", "content": (
                "You are a summarizer. Do NOT continue the conversation. "
                "Do NOT respond as Nova. Do NOT use emoji or roleplay. "
                "Write a plain 2-3 sentence summary of what was discussed."
            )},
            {"role": "user", "content": (
                f"Summarize this conversation in 2-3 plain sentences. "
                f"Focus on: topics discussed, decisions made, emotional tone.\n\n"
                f"{conversation_text}"
            )}
        ]
        # Run in thread — this is also a sync HTTP call
        # Don't pass tools — this is a utility call, not a conversation
        response = await asyncio.to_thread(self.llm.chat, summary_prompt)
        if response and response.raw:
            summary = strip_think_tags(response.raw).strip()
            if summary:
                logger.info(f"Conversation summarized: {summary[:80]}...")
                return summary
        return None

    async def _check_due_tasks(self) -> bool:
        """Check for and execute any due scheduled tasks.

        Returns True if any tasks were executed, False otherwise.
        This runs frequently (every 60s) so reminders fire on time.
        """
        due_tasks = self.scheduler.get_due_tasks()
        if not due_tasks:
            return False

        available = await asyncio.to_thread(self.llm.is_available)
        if not available:
            logger.warning("LLM server not available — can't execute due tasks.")
            return False

        for task in due_tasks:
            logger.info(f"Executing due task: {task['task']}")
            messages = self.context.build_task_prompt(task)
            response = await asyncio.to_thread(self.llm.chat, messages)
            if response:
                await self._dispatch(response, is_scheduled_task=True)
            self.scheduler.mark_completed(task["id"])

        return True

    async def _do_heartbeat(self):
        """Execute a single heartbeat tick (free-think)."""
        logger.info("--- Heartbeat tick ---")

        if self._in_quiet_hours():
            logger.info("Quiet hours — skipping heartbeat.")
            return

        available = await asyncio.to_thread(self.llm.is_available)
        if not available:
            logger.warning("LLM server is not available — skipping tick.")
            return

        # Free-think tick — Nova decides what to do
        messages = self.context.build_heartbeat_prompt()
        response = await asyncio.to_thread(self.llm.chat, messages)
        if response:
            await self._dispatch(response)

    async def run(self):
        """Main loop — runs forever until stopped.

        Two loops in one:
        - Fast loop: checks for due scheduled tasks every 60s (so reminders fire on time)
        - Slow loop: free-think heartbeat ticks on the configured interval (e.g., every 30m)
        """
        self._running = True
        self._stop_event.clear()
        logger.info("Pulse engine starting...")
        logger.info(f"Heartbeat interval: {self.interval // 60} minutes")
        logger.info(f"Quiet hours: {self.quiet_start}:00 - {self.quiet_end}:00")

        # Check LLM server availability
        if self.llm.is_available():
            logger.info("LLM server is reachable!")
        else:
            logger.warning("LLM server is not available — will retry on each tick.")

        # Optional startup check-in
        if self.startup_checkin:
            logger.info("Running startup check-in...")
            await self._check_due_tasks()
            await self._do_heartbeat()

        # Main loop — wake up every 60s to check schedules,
        # but only do a full heartbeat on the configured interval
        scheduler_interval = 60  # seconds — check for due tasks every minute
        ticks_since_heartbeat = 0
        heartbeat_ticks = self.interval // scheduler_interval  # e.g., 30 for 30-min interval

        while self._running:
            await self._interruptible_sleep(scheduler_interval)

            if not self._running:
                break

            ticks_since_heartbeat += 1

            try:
                # Always check for due tasks (fast — just reads a file)
                await self._check_due_tasks()

                # Full heartbeat on the configured interval
                if ticks_since_heartbeat >= heartbeat_ticks:
                    ticks_since_heartbeat = 0
                    await self._do_heartbeat()
            except Exception as e:
                logger.error(f"Tick failed: {e}", exc_info=True)

    def stop(self):
        """Signal the engine to stop — exits immediately."""
        self._running = False
        self._stop_event.set()  # Wake up from any sleep instantly
        logger.info("Pulse engine stopping...")
