"""
Tasks skill — persistent to-do lists with completion tracking.
"""

import json
import os
from datetime import datetime
from skills.base import BaseSkill

class TasksSkill(BaseSkill):
    name = "tasks"

    def __init__(self, config: dict):
        super().__init__(config)
        # Store tasks in the data directory to persist across reboots
        self.storage_path = os.path.join("data", "tasks.json")
        self.pending_display = None  # Set by list_tasks for Telegram to show
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs("data", exist_ok=True)
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
        data = self._read_tasks()

        # Ensure migration for old storage format if needed
        if "next_id" not in data:
            data["next_id"] = max([t["id"] for t in data["tasks"]] + [0]) + 1

        if tool_name == "add_task":
            new_task = {
                "id": data["next_id"],
                "description": arguments["task"],
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

        elif tool_name == "list_tasks":
            show_completed = arguments.get("show_completed", False)
            tasks = data["tasks"]
            if not show_completed:
                tasks = [t for t in tasks if not t["completed"]]

            if not tasks:
                self.pending_display = None
                return "The task list is empty."

            # Pretty version for Telegram display
            display = "📋 Tasks:\n"
            for t in tasks:
                check = "✅" if t["completed"] else "☐"
                display += f"  {check} {t['id']}. {t['description']}\n"
            self.pending_display = display.strip()

            # Plain version for the LLM
            output = "Current Tasks:\n"
            for t in tasks:
                status = "[x]" if t["completed"] else "[ ]"
                output += f"{t['id']}. {status} {t['description']} ({t['list']})\n"
            return output

        elif tool_name == "clear_tasks":
            data["tasks"] = [t for t in data["tasks"] if not t["completed"]]
            self._write_tasks(data)
            return "Cleared all completed tasks."

        return f"Unknown tool: {tool_name}"
