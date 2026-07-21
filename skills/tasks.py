"""
Tasks skill — persistent to-do lists with completion tracking.
"""

import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from skills.base import BaseSkill


class TasksSkill(BaseSkill):
    name = "tasks"
    description = "Manage persistent to-do lists with completion tracking"
    search_summary = "Add, list, complete, delete, and clean up persistent to-do items"
    search_examples = ["manage tasks", "add a todo", "check off a task"]
    aliases = ["todo", "to do", "task list", "checklist", "track tasks", "housekeeping"]
    categories = ["planning", "housekeeping"]

    def __init__(self, config: dict):
        super().__init__(config)
        self.storage_path = config.get("paths", {}).get("tasks", os.path.join("data", "tasks.json"))
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w") as f:
                json.dump({"tasks": [], "next_id": 1}, f)

    def _read_tasks(self):
        with open(self.storage_path, "r") as f:
            return json.load(f)

    def _write_tasks(self, data):
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=4)

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "add_task",
                    "description": "Add a new task to the to-do list.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "The task description."},
                            "list_name": {"type": "string", "description": "Optional list name (default: 'Daily')."}
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_task",
                    "description": "Mark a pending task as completed using the Pending task number shown by list_tasks. When your human says they finished or done something that matches a task, call this tool to check it off!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "integer", "description": "The Pending task number shown by list_tasks."}
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_task",
                    "description": "Delete a single pending task by the Pending task number shown by list_tasks without marking it as completed. Use this to remove stale, mistaken, or unwanted tasks cleanly.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "integer", "description": "The Pending task number shown by list_tasks."}
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_tasks",
                    "description": "List current tasks. Pending tasks are numbered for complete_task/delete_task; completed tasks are shown separately for reference. IMPORTANT: Always show the full task list to the user - do not summarize or omit tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "show_completed": {"type": "boolean", "description": "Whether to include completed tasks. Defaults to true."}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_tasks",
                    "description": "Remove all completed tasks from the list.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def get_context(self) -> str:
        """Surface pending tasks when this skill is listed in inject_skills."""
        try:
            pending = self._get_pending_tasks()
        except Exception:
            return ""

        if not pending:
            return ""

        max_show = 8
        lines = [
            f"Tasks: {len(pending)} pending.",
            "Use these Pending task numbers with complete_task/delete_task.",
        ]

        shown = 0
        for index, task in enumerate(pending[:max_show], start=1):
            desc = (task.get("description") or "").strip()
            if not desc:
                continue
            list_name = (task.get("list") or "Daily").strip() or "Daily"
            lines.append(f"{index}. [ ] {desc} ({list_name})")
            shown += 1

        if not shown:
            return ""

        if len(pending) > max_show:
            lines.append(
                f"...and {len(pending) - max_show} more. "
                "Call list_tasks before acting on hidden tasks."
            )
        else:
            lines.append(
                "If the target is unclear or the list may have changed, "
                "call list_tasks before acting."
            )

        return "\n".join(lines)

    def _get_pending_tasks(self) -> list[dict]:
        """Return pending tasks from the DB or legacy JSON store."""
        # Tasks are persona-local — always the persona DB, never _shared_db.
        db = self.config.get("_db")
        if db:
            return db.get_tasks(completed=False)
        data = self._read_tasks()
        return [task for task in data.get("tasks", []) if not task.get("completed")]

    def execute(self, tool_name: str, arguments: dict) -> str:
        db = self.config.get("_db")

        if db:
            if tool_name == "add_task":
                description = arguments["task"]
                list_name = arguments.get("list_name", "Daily")

                # Dedup: check if a similar pending task already exists in DB
                pending = db.get_tasks(completed=False)
                for existing in pending:
                    if self._tasks_similar(description, existing["description"]):
                        return (
                            f"Similar task already exists: "
                            f"{existing['description']} — not adding duplicate."
                        )

                db.add_task(description, list_name)
                return f"Task added: {description}"

            elif tool_name == "complete_task":
                task_number = arguments["task_id"]
                pending = db.get_tasks(completed=False)
                target = self._resolve_visible_task(pending, task_number)
                if not target:
                    tasks = db.get_tasks()
                    target = self._resolve_visible_task(tasks, task_number)
                if not target:
                    return f"Task number {task_number} not found. Use list_tasks to see current task numbers."

                if target["completed"]:
                    return f"Task is already completed: {target['description']}"

                db.complete_task(target["id"])
                return f"Task completed: {target['description']}."

            elif tool_name == "delete_task":
                task_number = arguments["task_id"]
                pending = db.get_tasks(completed=False)
                target = self._resolve_visible_task(pending, task_number)
                if not target:
                    tasks = db.get_tasks()
                    target = self._resolve_visible_task(tasks, task_number)
                if not target:
                    return f"Task number {task_number} not found. Use list_tasks to see current task numbers."

                db.delete_task(target["id"])
                return f"Task deleted: {target['description']}."

            elif tool_name == "list_tasks":
                show_completed = arguments.get("show_completed", True)
                pending = db.get_tasks(completed=False)
                completed = db.get_tasks(completed=True) if show_completed else []

                if not pending and not completed:
                    return "The task list is empty."

                return self._format_task_list(pending, completed)

            elif tool_name == "clear_tasks":
                completed = db.get_tasks(completed=True)
                for t in completed:
                    db.delete_task(t["id"])
                return f"Cleared {len(completed)} completed task(s)."

        # --- JSON Fallback ---
        data = self._read_tasks()

        # Ensure migration for old storage format if needed
        if "next_id" not in data:
            data["next_id"] = max([t["id"] for t in data["tasks"]] + [0]) + 1

        if tool_name == "add_task":
            # Dedup: check if a similar pending task already exists
            description = arguments["task"]
            for existing in data["tasks"]:
                if existing["completed"]:
                    continue
                if self._tasks_similar(description, existing["description"]):
                    return (
                        f"Similar task already exists: "
                        f"{existing['description']} — not adding duplicate."
                    )

            new_task = {
                "id": data["next_id"],
                "description": description,
                "list": arguments.get("list_name", "Daily"),
                "completed": False,
                "created_at": datetime.now().isoformat()
            }
            data["tasks"].append(new_task)
            data["next_id"] += 1
            self._write_tasks(data)
            return f"Task added: {new_task['description']}"

        elif tool_name == "complete_task":
            task_number = arguments["task_id"]
            pending = [t for t in data["tasks"] if not t["completed"]]
            task = self._resolve_visible_task(pending, task_number)
            if not task:
                task = self._resolve_visible_task(data["tasks"], task_number)
            if not task:
                return f"Task number {task_number} not found. Use list_tasks to see current task numbers."
            if task["completed"]:
                return f"Task is already completed: {task['description']}"
            task["completed"] = True
            task["completed_at"] = datetime.now().isoformat()
            self._write_tasks(data)
            return f"Task completed: {task['description']}."

        elif tool_name == "delete_task":
            task_number = arguments["task_id"]
            pending = [t for t in data["tasks"] if not t["completed"]]
            target = self._resolve_visible_task(pending, task_number)
            if not target:
                target = self._resolve_visible_task(data["tasks"], task_number)
            if not target:
                return f"Task number {task_number} not found. Use list_tasks to see current task numbers."
            data["tasks"] = [task for task in data["tasks"] if task["id"] != target["id"]]
            self._write_tasks(data)
            return f"Task deleted: {target['description']}."

        elif tool_name == "list_tasks":
            show_completed = arguments.get("show_completed", True)
            pending = [t for t in data["tasks"] if not t["completed"]]
            completed = [t for t in data["tasks"] if t["completed"]] if show_completed else []

            if not pending and not completed:
                return "The task list is empty."

            return self._format_task_list(pending, completed)

        elif tool_name == "clear_tasks":
            before = len(data["tasks"])
            data["tasks"] = [t for t in data["tasks"] if not t["completed"]]
            cleared = before - len(data["tasks"])
            self._write_tasks(data)
            return f"Cleared {cleared} completed task(s)."

        return f"Unknown tool: {tool_name}"

    @staticmethod
    def _resolve_visible_task(tasks: list[dict], task_number: int) -> dict | None:
        """Resolve the 1-based display number from list_tasks.

        If an old caller passes a raw DB/JSON ID that is outside the current
        display range, fall back to that ID for compatibility.
        """
        try:
            number = int(task_number)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        if number <= len(tasks):
            return tasks[number - 1]
        return next((task for task in tasks if task.get("id") == number), None)

    @staticmethod
    def _format_task_list(pending: list[dict], completed: list[dict]) -> str:
        output = (
            "Current Tasks (use Pending task numbers with complete_task/delete_task):\n"
            "Pending:\n"
        )
        if pending:
            for index, task in enumerate(pending, start=1):
                output += f"{index}. [ ] {task['description']} ({task['list']})\n"
        else:
            output += "- none\n"

        if completed:
            output += "\nCompleted (for reference; not numbered for task actions):\n"
            for task in completed:
                output += f"- [x] {task['description']} ({task['list']})\n"

        return output

    @staticmethod
    def _tasks_similar(a: str, b: str) -> bool:
        """Return True if two task descriptions are close enough to be likely duplicates."""
        a_norm = " ".join(a.lower().strip().split())
        b_norm = " ".join(b.lower().strip().split())

        if not a_norm or not b_norm:
            return False
        if a_norm == b_norm:
            return True
        if a_norm in b_norm or b_norm in a_norm:
            return min(len(a_norm), len(b_norm)) >= 12

        return SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.88
