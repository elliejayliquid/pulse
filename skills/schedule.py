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
                        "ISO datetime like '2026-03-01T15:00:00' (interpreted as LOCAL time). "
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
                            "priority": {
                                "type": "string",
                                "description": "Priority level for display: urgent, routine, or creative (default: routine)",
                                "enum": ["urgent", "routine", "creative"],
                            },
                        },
                        "required": ["task", "when"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_reminders",
                    "description": (
                        "List all active reminders and scheduled tasks. "
                        "Use this to check what's scheduled, find reminder IDs "
                        "for updating or deleting, or answer when the user asks "
                        "'what reminders do I have?'"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_completed": {
                                "type": "boolean",
                                "description": "Include completed one-time reminders (default: false)",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_reminder",
                    "description": (
                        "Update an existing reminder's time, task text, or priority. "
                        "Use this when the user wants to change when a reminder fires, "
                        "fix a wrong time, or update the task description. "
                        "You need the reminder's ID (from set_reminder or list output). "
                        "IMPORTANT: Always report the updated details from the tool result."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The schedule ID to update (e.g. 'sch_87517db9')",
                            },
                            "task": {
                                "type": "string",
                                "description": "New task description (leave empty to keep current)",
                            },
                            "when": {
                                "type": "string",
                                "description": "New time (e.g. 'in 2 hours', '2026-04-01T15:00:00', 'daily 9:00')",
                            },
                            "priority": {
                                "type": "string",
                                "description": "New priority level",
                                "enum": ["urgent", "routine", "creative"],
                            },
                        },
                        "required": ["id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_reminder",
                    "description": (
                        "Delete a reminder entirely. Use this when the user no longer "
                        "needs a reminder or wants to cancel a scheduled task. "
                        "You need the reminder's ID."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The schedule ID to delete (e.g. 'sch_87517db9')",
                            },
                        },
                        "required": ["id"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "set_reminder":
            return self._set_reminder(
                task=arguments.get("task", ""),
                when=arguments.get("when", ""),
                priority=arguments.get("priority", "routine"),
            )
        if tool_name == "list_reminders":
            return self._list_reminders(
                include_completed=arguments.get("include_completed", False),
            )
        if tool_name == "update_reminder":
            return self._update_reminder(
                schedule_id=arguments.get("id", ""),
                task=arguments.get("task", ""),
                when=arguments.get("when", ""),
                priority=arguments.get("priority", ""),
            )
        if tool_name == "delete_reminder":
            return self._delete_reminder(
                schedule_id=arguments.get("id", ""),
            )
        return f"Unknown tool: {tool_name}"

    def _list_reminders(self, include_completed: bool = False) -> str:
        """List reminders."""
        if not self._scheduler:
            return "Scheduler not available."

        if include_completed:
            entries = self._scheduler.list_all()
        else:
            entries = self._scheduler.list_active()

        if not entries:
            return "No active reminders."

        lines = [f"{'All' if include_completed else 'Active'} reminders ({len(entries)}):"]
        for e in entries:
            sid = e.get("id", "?")
            task = e.get("task", "(no description)")
            priority = e.get("priority", "routine")
            stype = e.get("schedule_type", "once")

            if stype == "recurring":
                when = f"recurring — {e.get('cron', '?')}"
            else:
                when = e.get("run_at_local", e.get("run_at", "?"))
                if e.get("completed"):
                    when += " (completed)"

            lines.append(f"  [{sid}] {task} | {when} | {priority}")

        return "\n".join(lines)

    def _set_reminder(self, task: str, when: str, priority: str = "routine") -> str:
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
                created_by="companion",
                origin="user",
                when=when.strip(),
                priority=priority,
            )

            # Was this deduped against an existing reminder? If so, the model
            # needs to KNOW it didn't actually create a new one — otherwise it
            # will narrate "I set it!" and confuse the user.
            was_deduped = entry.pop("_was_deduped", False)
            if was_deduped:
                logger.info(f"Reminder dedup hit via skill: {entry['id']} — '{task}' matched existing")
                schedule_type = entry.get("schedule_type", "once")
                if schedule_type == "recurring":
                    when_str = f"{entry.get('cron', '?')} (daily recurring)"
                else:
                    when_str = entry.get("run_at_local", entry.get("run_at", "?"))
                created_at = entry.get("created_at_local", entry.get("created_at", "earlier"))
                return (
                    f"⚠️ ALREADY EXISTS — no new reminder was created.\n"
                    f"You (or someone) already set a similar reminder previously:\n"
                    f"  Existing task: {entry.get('task', '')}\n"
                    f"  Fires at: {when_str}\n"
                    f"  Priority: {entry.get('priority', 'routine')}\n"
                    f"  ID: {entry['id']}\n"
                    f"  Originally created: {created_at}\n\n"
                    f"DO NOT tell the user you just set a new reminder — that would be wrong.\n"
                    f"Instead, acknowledge that the reminder is already in place from earlier. "
                    f"If they actually want a *different* reminder (different time or task), "
                    f"ask them to clarify, or use update_reminder to change the existing one."
                )

            logger.info(f"Reminder set via skill: {entry['id']} — {task} (priority: {priority})")

            # Format the actual resolved time clearly — the model MUST report this
            # accurately to the user (not guess or hallucinate a different time).
            schedule_type = entry.get("schedule_type", "once")

            if schedule_type == "recurring":
                cron = entry.get("cron", "")
                return (
                    f"RECURRING REMINDER CONFIRMED — report these details exactly to the user:\n"
                    f"  Task: {task}\n"
                    f"  Schedule: {cron} (daily recurring)\n"
                    f"  Priority: {priority}\n"
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
                    f"  Priority: {priority}\n"
                    f"  Type: one-time (NOT recurring)\n"
                    f"  ID: {entry['id']}\n"
                    f"Do NOT say a different time or claim it's recurring."
                )
        except Exception as e:
            logger.error(f"Failed to set reminder: {e}")
            return f"Failed to set reminder: {e}"

    def _update_reminder(self, schedule_id: str, task: str = "",
                         when: str = "", priority: str = "") -> str:
        """Update an existing reminder."""
        if not self._scheduler:
            return "Scheduler not available."
        if not schedule_id.strip():
            return "Please provide the reminder ID to update."

        kwargs = {}
        if task.strip():
            kwargs["task"] = task.strip()
        if when.strip():
            kwargs["when"] = when.strip()
        if priority.strip():
            kwargs["priority"] = priority.strip()

        if not kwargs:
            return "Nothing to update — provide at least one of: task, when, priority."

        try:
            entry = self._scheduler.update(schedule_id.strip(), **kwargs)
            if entry is None:
                return f"Reminder {schedule_id} not found. Check the ID and try again."

            logger.info(f"Reminder updated via skill: {entry['id']}")

            schedule_type = entry.get("schedule_type", "once")
            if schedule_type == "recurring":
                cron = entry.get("cron", "")
                return (
                    f"REMINDER UPDATED — report these details exactly to the user:\n"
                    f"  Task: {entry.get('task', '')}\n"
                    f"  Schedule: {cron} (daily recurring)\n"
                    f"  Priority: {entry.get('priority', 'routine')}\n"
                    f"  Type: recurring\n"
                    f"  ID: {entry['id']}\n"
                    f"Do NOT say a different schedule."
                )
            else:
                actual_time = entry.get("run_at", "")
                local_time = entry.get("run_at_local", "")
                if not local_time and isinstance(actual_time, str) and "T" in actual_time:
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(actual_time)
                        local_time = dt.astimezone().strftime("%A %b %d, %I:%M %p")
                    except ValueError:
                        local_time = actual_time

                return (
                    f"REMINDER UPDATED — report these details exactly to the user:\n"
                    f"  Task: {entry.get('task', '')}\n"
                    f"  Fires at: {local_time}\n"
                    f"  Priority: {entry.get('priority', 'routine')}\n"
                    f"  Type: one-time\n"
                    f"  ID: {entry['id']}\n"
                    f"Do NOT say a different time."
                )
        except Exception as e:
            logger.error(f"Failed to update reminder: {e}")
            return f"Failed to update reminder: {e}"

    def _delete_reminder(self, schedule_id: str) -> str:
        """Delete a reminder entirely."""
        if not self._scheduler:
            return "Scheduler not available."
        if not schedule_id.strip():
            return "Please provide the reminder ID to delete."

        try:
            removed = self._scheduler.remove(schedule_id.strip())
            if removed:
                logger.info(f"Reminder deleted via skill: {schedule_id}")
                return (
                    f"REMINDER DELETED — ID: {schedule_id}\n"
                    f"Confirm to the user that the reminder has been removed."
                )
            else:
                return f"Reminder {schedule_id} not found. Check the ID and try again."
        except Exception as e:
            logger.error(f"Failed to delete reminder: {e}")
            return f"Failed to delete reminder: {e}"
