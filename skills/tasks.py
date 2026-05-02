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
                    "description": "Mark a task as completed using its ID. When your human says they finished or done something that matches a task, call this tool to check it off!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "integer", "description": "The ID of the task to complete."}
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_task",
                    "description": "Delete a single task by ID without marking it as completed. Use this to remove stale, mistaken, or unwanted tasks cleanly.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "integer", "description": "The ID of the task to delete."}
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_tasks",
                    "description": "List all current tasks, showing their status and IDs. IMPORTANT: Always show the full task list to the user — do not summarize or omit tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "show_completed": {"type": "boolean", "description": "Whether to include completed tasks."}
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

    def execute(self, tool_name: str, arguments: dict) -> str:
        db = self.config.get("_shared_db") or self.config.get("_db")

        if db:
            if tool_name == "add_task":
                description = arguments["task"]
                list_name = arguments.get("list_name", "Daily")

                # Dedup: check if a similar pending task already exists in DB
                pending = db.get_tasks(completed=False)
                for existing in pending:
                    if self._tasks_similar(description, existing["description"]):
                        return (
                            f"Similar task already exists: [{existing['id']}] "
                            f"{existing['description']} — not adding duplicate."
                        )

                task_id = db.add_task(description, list_name)
                return f"Task added: [{task_id}] {description}"

            elif tool_name == "complete_task":
                task_id = arguments["task_id"]
                tasks = db.get_tasks()
                target = next((t for t in tasks if t["id"] == task_id), None)
                if not target:
                    return f"Task with ID {task_id} not found."

                if target["completed"]:
                    return f"Task {task_id} is already completed."

                db.complete_task(task_id)
                return f"Task completed: {target['description']}."

            elif tool_name == "delete_task":
                task_id = arguments["task_id"]
                tasks = db.get_tasks()
                target = next((t for t in tasks if t["id"] == task_id), None)
                if not target:
                    return f"Task with ID {task_id} not found."

                db.delete_task(task_id)
                return f"Task deleted: [{task_id}] {target['description']}."

            elif tool_name == "list_tasks":
                show_completed = arguments.get("show_completed", False)
                tasks = db.get_tasks(completed=None if show_completed else False)

                if not tasks:
                    return "The task list is empty."

                output = "Current Tasks:\n"
                for t in tasks:
                    status = "[x]" if t["completed"] else "[ ]"
                    output += f"{t['id']}. {status} {t['description']} ({t['list']})\n"
                return output

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
                        f"Similar task already exists: [{existing['id']}] "
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
            return f"Task added: [{new_task['id']}] {new_task['description']}"

        elif tool_name == "complete_task":
            task_id = arguments["task_id"]
            for task in data["tasks"]:
                if task["id"] == task_id:
                    if task["completed"]:
                        return f"Task {task_id} is already completed."
                    task["completed"] = True
                    task["completed_at"] = datetime.now().isoformat()
                    self._write_tasks(data)
                    return f"Task completed: {task['description']}."
            return f"Task with ID {task_id} not found."

        elif tool_name == "delete_task":
            task_id = arguments["task_id"]
            for index, task in enumerate(data["tasks"]):
                if task["id"] == task_id:
                    removed = data["tasks"].pop(index)
                    self._write_tasks(data)
                    return f"Task deleted: [{task_id}] {removed['description']}."
            return f"Task with ID {task_id} not found."

        elif tool_name == "list_tasks":
            show_completed = arguments.get("show_completed", False)
            tasks = data["tasks"]
            if not show_completed:
                tasks = [t for t in tasks if not t["completed"]]

            if not tasks:
                return "The task list is empty."

            output = "Current Tasks:\n"
            for t in tasks:
                status = "[x]" if t["completed"] else "[ ]"
                output += f"{t['id']}. {status} {t['description']} ({t['list']})\n"
            return output

        elif tool_name == "clear_tasks":
            before = len(data["tasks"])
            data["tasks"] = [t for t in data["tasks"] if not t["completed"]]
            cleared = before - len(data["tasks"])
            self._write_tasks(data)
            return f"Cleared {cleared} completed task(s)."

        return f"Unknown tool: {tool_name}"

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
