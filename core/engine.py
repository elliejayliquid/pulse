"""
Pulse Engine - the heartbeat loop.

This is the main orchestrator. It:
1. Runs a heartbeat timer (free-think ticks)
2. Checks for due scheduled tasks
3. Builds context and prompts Nova via LM Studio
4. Dispatches actions to channels (LoR, toast, etc.)
"""

import logging
import time
from datetime import datetime, timezone

from core.llm import LLMClient, PulseResponse
from core.context import ContextManager
from core.scheduler import ScheduleManager

logger = logging.getLogger(__name__)


class PulseEngine:
    """Main engine that drives the heartbeat loop."""

    def __init__(self, config: dict, channels: dict):
        """
        Args:
            config: Parsed config.yaml
            channels: Dict of channel_name -> Channel instance
        """
        self.config = config
        self.channels = channels

        # Initialize core components
        model_config = config.get("model", {})
        self.llm = LLMClient(
            endpoint=model_config.get("endpoint", "http://localhost:1234/v1"),
            model_name=model_config.get("model_name", "default"),
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_response_tokens", 1024),
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

    def _in_quiet_hours(self) -> bool:
        """Check if we're in quiet hours."""
        hour = datetime.now().hour
        if self.quiet_start > self.quiet_end:
            # Wraps around midnight (e.g., 23-8)
            return hour >= self.quiet_start or hour < self.quiet_end
        else:
            return self.quiet_start <= hour < self.quiet_end

    async def _dispatch(self, response: PulseResponse):
        """Route a response to the appropriate channel."""
        if response.action == "silent":
            logger.info("Nova chose to stay silent.")
            return

        if response.action == "notify":
            toast = self.channels.get("toast")
            if toast:
                await toast.send(response.message)
                logger.info(f"Notification sent: {response.message[:50]}...")
            else:
                logger.info(f"[TOAST disabled] {response.message[:80]}...")

        elif response.action == "post_lor":
            lor = self.channels.get("lor")
            if lor:
                await lor.send(
                    response.message,
                    category=response.lor_category,
                    title=response.lor_title
                )
                logger.info(f"Posted to LoR: {response.lor_title or response.message[:50]}...")
            else:
                logger.info(f"[LoR disabled] {response.message[:80]}...")

        elif response.action == "schedule":
            if response.schedule_task and response.schedule_when:
                entry = self.scheduler.add(
                    task=response.schedule_task,
                    created_by="nova-self",
                    when=response.schedule_when
                )
                logger.info(f"Nova self-scheduled: {entry['id']} — {response.schedule_task}")

        # Handle combined actions (e.g., notify AND schedule)
        if response.schedule_task and response.schedule_when and response.action != "schedule":
            entry = self.scheduler.add(
                task=response.schedule_task,
                created_by="nova-self",
                when=response.schedule_when
            )
            logger.info(f"Nova also scheduled: {entry['id']} — {response.schedule_task}")

    async def _do_heartbeat(self):
        """Execute a single heartbeat tick."""
        logger.info("--- Heartbeat tick ---")

        if self._in_quiet_hours():
            logger.info("Quiet hours — skipping heartbeat.")
            return

        if not self.llm.is_available():
            logger.warning("LM Studio is not available — skipping tick.")
            return

        # Check for due scheduled tasks
        due_tasks = self.scheduler.get_due_tasks()

        if due_tasks:
            # Execute each due task with a dedicated prompt
            for task in due_tasks:
                logger.info(f"Executing due task: {task['task']}")
                messages = self.context.build_task_prompt(task)
                response = self.llm.chat(messages)
                if response:
                    await self._dispatch(response)
                self.scheduler.mark_completed(task["id"])
        else:
            # Free-think tick — Nova decides what to do
            messages = self.context.build_heartbeat_prompt()
            response = self.llm.chat(messages)
            if response:
                await self._dispatch(response)

    async def run(self):
        """Main loop — runs forever until stopped."""
        import asyncio

        self._running = True
        logger.info("Pulse engine starting...")
        logger.info(f"Heartbeat interval: {self.interval // 60} minutes")
        logger.info(f"Quiet hours: {self.quiet_start}:00 - {self.quiet_end}:00")

        # Check LM Studio availability
        if self.llm.is_available():
            logger.info("LM Studio is reachable!")
        else:
            logger.warning("LM Studio is not available — will retry on each tick.")

        # Optional startup check-in
        if self.startup_checkin:
            logger.info("Running startup check-in...")
            await self._do_heartbeat()

        # Main loop
        while self._running:
            await asyncio.sleep(self.interval)
            try:
                await self._do_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}", exc_info=True)

    def stop(self):
        """Signal the engine to stop."""
        self._running = False
        logger.info("Pulse engine stopping...")
