"""
Skill Registry — discovers, loads, and manages Pulse skills.

Each skill is a Python file in this directory with a BaseSkill subclass.
Skills are enabled/disabled via config.yaml's `skills:` section.

Usage:
    registry = SkillRegistry(config)
    tools = registry.get_all_tools()       # For the OpenAI API `tools` param
    result = registry.execute("save_memory", {"text": "..."})  # Run a tool call
"""

import logging
from skills.base import BaseSkill
from skills.time_skill import TimeSkill
from skills.memory import MemorySkill
from skills.schedule import ScheduleSkill
from skills.journal import JournalSkill
from skills.lor import LoRSkill
from skills.web_search import WebSearchSkill

logger = logging.getLogger(__name__)

# All available skill classes — add new skills here
SKILL_CLASSES = [
    TimeSkill,
    MemorySkill,
    ScheduleSkill,
    JournalSkill,
    LoRSkill,
    WebSearchSkill,
]


class SkillRegistry:
    """Loads enabled skills and routes tool calls to the right skill."""

    def __init__(self, config: dict):
        self.config = config
        self.skills: dict[str, BaseSkill] = {}
        self._tool_map: dict[str, BaseSkill] = {}  # tool_name -> skill instance

        skills_config = config.get("skills", {})

        for cls in SKILL_CLASSES:
            skill_name = cls.name
            skill_conf = skills_config.get(skill_name, {})

            # Default to enabled if not specified
            if not skill_conf.get("enabled", True):
                logger.info(f"Skill '{skill_name}' is disabled in config.")
                continue

            try:
                skill = cls(config)
                self.skills[skill_name] = skill

                # Map each tool name to its skill
                for tool_def in skill.get_tools():
                    tool_name = tool_def["function"]["name"]
                    self._tool_map[tool_name] = skill
                    logger.debug(f"Registered tool: {tool_name} (from {skill_name})")

                logger.info(f"Skill loaded: {skill_name} ({len(skill.get_tools())} tools)")
            except Exception as e:
                logger.error(f"Failed to load skill '{skill_name}': {e}")

        logger.info(
            f"SkillRegistry ready: {len(self.skills)} skills, "
            f"{len(self._tool_map)} tools"
        )

    def get_all_tools(self) -> list[dict]:
        """Collect tool definitions from all enabled skills.

        Returns a flat list suitable for the OpenAI API `tools` parameter.
        """
        tools = []
        for skill in self.skills.values():
            tools.extend(skill.get_tools())
        return tools

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call by routing to the appropriate skill.

        Args:
            tool_name: The function name from the model's tool call
            arguments: Parsed arguments dict

        Returns:
            Result string from the skill.
        """
        skill = self._tool_map.get(tool_name)
        if not skill:
            logger.warning(f"No skill found for tool: {tool_name}")
            return f"Unknown tool: {tool_name}"

        try:
            result = skill.execute(tool_name, arguments)
            logger.info(f"Tool executed: {tool_name} -> {result[:80]}...")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed ({tool_name}): {e}")
            return f"Tool error: {e}"

    def get_skill(self, name: str) -> BaseSkill | None:
        """Get a specific skill instance by name."""
        return self.skills.get(name)
