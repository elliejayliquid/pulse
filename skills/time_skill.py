"""
Time skill — gives Nova awareness of the current date and time.

Simple but essential: without this, the model can't orient itself
temporally during conversations. Mirrors the MCP memory server's
get_current_time() tool.
"""

from datetime import datetime
from skills.base import BaseSkill


class TimeSkill(BaseSkill):
    name = "time"

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current date and time. Use this when you need to know what day or time it is.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "get_current_time":
            now = datetime.now()
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
        return f"Unknown tool: {tool_name}"
