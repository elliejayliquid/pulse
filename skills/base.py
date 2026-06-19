"""
Base skill interface for Pulse.

Every skill is a self-contained plugin that:
- Declares its tools in OpenAI function-calling format
- Executes tool calls and returns results
- Can be enabled/disabled via config.yaml
"""

from abc import ABC, abstractmethod


class BaseSkill(ABC):
    """Abstract base class for all Pulse skills.

    Subclasses must set `name` and implement `get_tools()` + `execute()`.

    Example:
        class TimeSkill(BaseSkill):
            name = "time"
            def get_tools(self): ...
            def execute(self, tool_name, arguments): ...
    """

    name: str = ""
    description: str = ""          # one-line summary for search + prompt display
    search_summary: str = ""       # optional friendlier summary for on-demand menus
    search_examples: list[str] = [] # example phrases that should discover this skill
    aliases: list[str] = []        # search_tools hints, e.g. "reach out", "creative"
    categories: list[str] = []     # broad search groups, e.g. "social", "creative"
    always_loaded: bool = False    # True = all tools always in API calls
    always_tools: list[str] = []   # specific tool names always loaded even if skill is on-demand
    workflow: str = ""             # usage hints injected when skill is loaded via search_tools
    tool_loop_mode: str = ""       # "" = use engine default, "capped", "unlimited"
    tool_loop_budget: int = 0      # 0 = use engine default. Only used when mode is "capped"

    def __init__(self, config: dict):
        """
        Args:
            config: Full Pulse config dict (from config.yaml).
        """
        self.config = config

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return tool definitions in OpenAI function-calling format.

        Each entry should look like:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "What this tool does",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": [ ... ]
                }
            }
        }
        """

    def get_context(self) -> str:
        """Return context to inject into every prompt (optional).

        Override this to provide always-on context — identity, journal,
        knowledge base, etc. — that should be present in every heartbeat
        and conversation prompt without requiring a tool call.

        Used when the skill is listed in config: context.inject_skills.
        Return empty string to inject nothing.
        """
        return ""

    @abstractmethod
    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string.

        Args:
            tool_name: The function name (e.g. "save_memory")
            arguments: Parsed arguments dict from the model's tool call

        Returns:
            Result text that gets sent back to the model.
        """
