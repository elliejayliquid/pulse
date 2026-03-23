"""
Schedule skill — set reminders during conversations.

Wraps the existing ScheduleManager (core/scheduler.py) so the companion can
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
                        "Use this when your human asks you to remind them of something, "
                        "or when you want to schedule a follow-up for yourself. "
                        "Supports one-time: 'in 2 hours', 'in 30 minutes', 'in 3 days', "
                        "ISO datetime like '2026-03-01T15:00:00'. "
                        "Supports recurring: 'daily 8:00', 'daily 14:30'. "
                        "IMPORTANT: Always report the exact time and type from the tool result — "
                        "do not guess or paraphrase the scheduled time."
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
                origin="user",
                when=when.strip(),
            )
            logger.info(f"Reminder set via skill: {entry['id']} — {task}")

            # Format the actual resolved time clearly — the model MUST report this
            # accurately to the user (not guess or hallucinate a different time).
            schedule_type = entry.get("schedule_type", "once")

            if schedule_type == "recurring":
                cron = entry.get("cron", "")
                return (
                    f"RECURRING REMINDER CONFIRMED — report these details exactly to the user:\n"
                    f"  Task: {task}\n"
                    f"  Schedule: {cron} (daily recurring)\n"
                    f"  Type: recurring\n"
                    f"  ID: {entry['id']}\n"
                    f"Do NOT say a different schedule. This repeats every day."
                )
            else:
                actual_time = entry.get("run_at", when)
                local_time = entry.get("run_at_local", "")
                if not local_time and isinstance(actual_time, str) and "T" in actual_time:
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(actual_time)
                        local_time = dt.astimezone().strftime("%A %b %d, %I:%M %p")
                    except ValueError:
                        local_time = actual_time

                return (
                    f"REMINDER CONFIRMED — report these details exactly to the user:\n"
                    f"  Task: {task}\n"
                    f"  Fires at: {local_time}\n"
                    f"  Type: one-time (NOT recurring)\n"
                    f"  ID: {entry['id']}\n"
                    f"Do NOT say a different time or claim it's recurring."
                )
        except Exception as e:
            logger.error(f"Failed to set reminder: {e}")
            return f"Failed to set reminder: {e}"
