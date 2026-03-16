"""
Schedule skill — lets Nova set reminders during conversations.

Wraps the existing ScheduleManager (core/scheduler.py) so Nova can
self-schedule tasks via tool calling instead of relying on JSON actions.
"""

import logging
from skills.base import BaseSkill

logger = logging.getLogger(__name__)


class ScheduleSkill(BaseSkill):
    name = "schedule"

    def __init__(self, config: dict, scheduler=None):
        super().__init__(config)
        self._scheduler = scheduler

    def set_scheduler(self, scheduler):
        """Inject the ScheduleManager after init (engine owns it)."""
        self._scheduler = scheduler

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_reminder",
                    "description": (
                        "Set a reminder or schedule a task for later. "
                        "Use this when Lena asks you to remind her of something, "
                        "or when you want to schedule a follow-up for yourself. "
                        "Supports: 'in 2 hours', 'in 30 minutes', 'in 3 days', "
                        "or ISO datetime like '2026-03-01T15:00:00'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "What to do or remind about",
                            },
                            "when": {
                                "type": "string",
                                "description": "When to trigger (e.g. 'in 2 hours', 'in 30 minutes', 'tomorrow 3pm')",
                            },
                        },
                        "required": ["task", "when"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "set_reminder":
            return self._set_reminder(
                task=arguments.get("task", ""),
                when=arguments.get("when", ""),
            )
        return f"Unknown tool: {tool_name}"

    def _set_reminder(self, task: str, when: str) -> str:
        """Create a one-time reminder via the ScheduleManager."""
        if not self._scheduler:
            return "Scheduler not available."
        if not task.strip():
            return "Task description cannot be empty."
        if not when.strip():
            return "Please specify when (e.g. 'in 2 hours', 'in 30 minutes')."

        try:
            entry = self._scheduler.add(
                task=task.strip(),
                created_by="nova-self",
                when=when.strip(),
            )
            logger.info(f"Reminder set via skill: {entry['id']} — {task}")

            # Show the actual resolved time so the model reports it accurately
            actual_time = entry.get("run_at", when)
            if isinstance(actual_time, str) and "T" in actual_time:
                # Parse ISO and show local time
                from datetime import datetime, timezone
                try:
                    dt = datetime.fromisoformat(actual_time)
                    actual_time = dt.astimezone().strftime("%I:%M %p")
                except ValueError:
                    pass

            return f"Reminder set (ID: {entry['id']}): '{task}' — will fire at {actual_time}"
        except Exception as e:
            logger.error(f"Failed to set reminder: {e}")
            return f"Failed to set reminder: {e}"
