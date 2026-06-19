"""
Skill Registry — discovers, loads, and manages Pulse skills.

Skills are auto-discovered: any .py file in this directory that contains
a BaseSkill subclass with a `name` attribute will be found and loaded.
Skills are enabled/disabled via config.yaml's `skills:` section.

Tools are split into always-loaded (every API call) and on-demand (loaded
via the search_tools meta-tool). This reduces per-call token cost by ~60-70%.

Usage:
    registry = SkillRegistry(config)
    tools = registry.get_always_tools()    # Core tools + search_tools meta-tool
    tools = registry.get_all_tools()       # Everything (for scheduled tasks)
    result = registry.execute("save_memory", {"text": "..."})
"""

import importlib
import inspect
import logging
import pkgutil
import re

import numpy as np

from skills.base import BaseSkill

logger = logging.getLogger(__name__)


def _terms(text: str) -> list[str]:
    """Normalize text into lightweight search terms."""
    return [term for term in re.split(r"[^a-z0-9]+", text.lower()) if term]


def _discover_skills() -> list[type[BaseSkill]]:
    """Scan the skills package for all BaseSkill subclasses."""
    import skills as skills_pkg

    found = []
    for finder, module_name, is_pkg in pkgutil.iter_modules(skills_pkg.__path__):
        if module_name in ("base", "__init__"):
            continue
        try:
            module = importlib.import_module(f"skills.{module_name}")
        except Exception as e:
            logger.error(f"Failed to import skills.{module_name}: {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseSkill)
                and obj is not BaseSkill
                and getattr(obj, "name", "")
            ):
                found.append(obj)

    return found


class SkillRegistry:
    """Loads enabled skills and routes tool calls to the right skill.

    Tools are categorized into:
    - Always-loaded: present in every API call (core tools like memory, schedule)
    - On-demand: discovered and loaded via the search_tools meta-tool
    """

    def __init__(self, config: dict):
        self.config = config
        self.skills: dict[str, BaseSkill] = {}
        self._tool_map: dict[str, BaseSkill] = {}  # tool_name -> skill instance

        # Always vs on-demand tool separation
        self._always_tools: list[dict] = []     # tool defs always in API calls
        self._on_demand_tools: dict[str, list[dict]] = {}  # skill_name -> tool defs
        self._on_demand_skills: dict[str, BaseSkill] = {}  # skill_name -> instance

        skills_config = config.get("skills", {})
        discovered = _discover_skills()

        for cls in discovered:
            skill_name = cls.name
            skill_conf = skills_config.get(skill_name, {})

            if not skill_conf.get("enabled", True):
                logger.info(f"Skill '{skill_name}' is disabled in config.")
                continue

            try:
                skill = cls(config)
                self.skills[skill_name] = skill

                all_tools = skill.get_tools()
                always_tool_names = set(cls.always_tools) if cls.always_tools else set()

                for tool_def in all_tools:
                    tool_name = tool_def["function"]["name"]
                    self._tool_map[tool_name] = skill

                if cls.always_loaded:
                    self._always_tools.extend(all_tools)
                elif always_tool_names:
                    for td in all_tools:
                        tn = td["function"]["name"]
                        if tn in always_tool_names:
                            self._always_tools.append(td)
                        else:
                            self._on_demand_tools.setdefault(skill_name, []).append(td)
                    self._on_demand_skills[skill_name] = skill
                else:
                    self._on_demand_tools[skill_name] = all_tools
                    self._on_demand_skills[skill_name] = skill

                logger.info(f"Skill loaded: {skill_name} ({len(all_tools)} tools)")
            except Exception as e:
                logger.error(f"Failed to load skill '{skill_name}': {e}")

        always_count = len(self._always_tools)
        on_demand_count = sum(len(t) for t in self._on_demand_tools.values())
        logger.info(
            f"SkillRegistry ready: {len(self.skills)} skills, "
            f"{always_count} always-loaded + {on_demand_count} on-demand tools"
        )

    # ── Tool retrieval ──────────────────────────────────────────

    def get_always_tools(self) -> list[dict]:
        """Tools for always-loaded skills + the search_tools meta-tool."""
        tools = list(self._always_tools)
        if self._on_demand_skills:
            tools.append(self._search_tools_definition())
        return tools

    def get_all_tools(self) -> list[dict]:
        """All tool definitions (for scheduled tasks that might need anything)."""
        tools = []
        for skill in self.skills.values():
            tools.extend(skill.get_tools())
        return tools

    def get_skill_summary(self) -> list[dict]:
        """Compact summary of always-loaded skills and their tool names."""
        summary = []
        for name, skill in self.skills.items():
            tool_names = [
                t.get("function", {}).get("name", "?")
                for t in skill.get_tools()
                if t.get("function", {}).get("name", "?") in
                {td["function"]["name"] for td in self._always_tools}
            ]
            if tool_names:
                summary.append({"skill": name, "tools": tool_names})
        return summary

    def _skill_display_summary(self, name: str, skill: BaseSkill) -> str:
        """Return a compact, model-facing summary for a skill."""
        summary = (getattr(skill, "search_summary", "") or skill.description or "").strip()
        if summary:
            return summary

        tool_descriptions = [
            (td.get("function", {}).get("description") or "").strip()
            for td in self._on_demand_tools.get(name, [])
        ]
        tool_descriptions = [desc for desc in tool_descriptions if desc]
        if tool_descriptions:
            return tool_descriptions[0].split(".")[0].strip()
        return f"Load tools from the {name} skill"

    def _skill_search_examples(self, skill: BaseSkill) -> list[str]:
        """Return short example queries for menus and search scoring."""
        examples = list(getattr(skill, "search_examples", []) or [])
        if not examples:
            examples = list(getattr(skill, "aliases", []) or [])[:3]
        return [str(example).strip() for example in examples if str(example).strip()]

    def get_on_demand_manifest(self) -> list[dict]:
        """Compact summaries for prompt injection."""
        manifest = []
        for name, skill in self._on_demand_skills.items():
            tool_count = len(self._on_demand_tools.get(name, []))
            tool_names = [
                td["function"]["name"]
                for td in self._on_demand_tools.get(name, [])
            ]
            manifest.append({
                "skill": name,
                "description": self._skill_display_summary(name, skill),
                "categories": list(getattr(skill, "categories", []) or []),
                "examples": self._skill_search_examples(skill),
                "tool_count": tool_count,
                "tools": tool_names,
            })
        return manifest

    # ── search_tools meta-tool ──────────────────────────────────

    def _skill_search_text(self, name: str, skill: BaseSkill) -> str:
        summary = getattr(skill, "search_summary", "") or ""
        examples = " ".join(self._skill_search_examples(skill))
        aliases = " ".join(getattr(skill, "aliases", []) or [])
        categories = " ".join(getattr(skill, "categories", []) or [])
        tool_text = " ".join(
            " ".join([
                td["function"]["name"].replace("_", " "),
                td["function"].get("description", ""),
            ])
            for td in self._on_demand_tools.get(name, [])
        )
        return f"{name} {skill.description or ''} {summary} {examples} {aliases} {categories} {tool_text}".lower()

    def _keyword_score(self, query: str, name: str, skill: BaseSkill) -> float:
        query_lower = query.lower()
        query_terms = set(_terms(query_lower))
        if not query_terms:
            return 0.0

        searchable = self._skill_search_text(name, skill)
        searchable_terms = set(_terms(searchable))
        score = 0.0

        if name in query_terms or name in query_lower:
            score += 5.0

        for category in getattr(skill, "categories", []) or []:
            category_lower = category.lower()
            category_terms = set(_terms(category_lower))
            if category_lower and category_lower in query_lower:
                score += 8.0
            elif category_terms and category_terms & query_terms:
                score += 4.0

        for alias in getattr(skill, "aliases", []) or []:
            alias_lower = alias.lower()
            alias_terms = set(_terms(alias_lower))
            if alias_lower and alias_lower in query_lower:
                score += 10.0
            elif alias_terms and alias_terms <= query_terms:
                score += 7.0
            elif alias_terms and alias_terms & query_terms:
                score += 2.0 * len(alias_terms & query_terms)

        for example in self._skill_search_examples(skill):
            example_lower = example.lower()
            example_terms = set(_terms(example_lower))
            if example_lower and example_lower in query_lower:
                score += 10.0
            elif example_terms and example_terms <= query_terms:
                score += 7.0
            elif example_terms and example_terms & query_terms:
                score += 2.0 * len(example_terms & query_terms)

        score += float(len(query_terms & searchable_terms))
        return score

    def _search_tools_definition(self) -> dict:
        manifest = self.get_on_demand_manifest()
        skill_list = "; ".join(
            f"{s['skill']}: {s['description']} ({s['tool_count']} tools"
            + (f"; try: {', '.join(s.get('examples', [])[:2])}" if s.get("examples") else "")
            + ")"
            for s in manifest
        )
        examples = []
        for skill in manifest:
            examples.extend(skill.get("examples", [])[:1])
        example_text = ", ".join(examples[:8]) or "search the web, send a sticker, write in my journal"
        return {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": (
                    "Load additional tools that are available but not shown in full yet. "
                    "Use this whenever a task may need an on-demand skill. "
                    "Available skills: " + skill_list + ". "
                    "After this returns, matching tools are available immediately "
                    "in this same turn. search_tools opens the toolbox; it does "
                    "not perform the requested action. Next, call the loaded tool "
                    "that actually does the work before describing any result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Name a skill or describe what you want to do. "
                                "Examples: " + example_text
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def search_tools(self, query: str, top_k: int = 3) -> tuple[list[dict], str]:
        """Find on-demand tools matching a query.

        Returns (tool_defs, result_text) — tool_defs to extend the tools list,
        result_text to feed back to the model.
        """
        if not query.strip():
            return [], (
                "search_tools is ready to help. Describe the capability you need "
                "or name one of the available skills."
            )

        scored: dict[str, float] = {}

        for name, skill in self._on_demand_skills.items():
            keyword_score = self._keyword_score(query, name, skill)
            if keyword_score > 0:
                scored[name] = max(scored.get(name, 0.0), keyword_score)

        # Try semantic search first
        model = self._get_embedding_model()
        if model:
            query_vec = model.encode(query)
            norm_q = np.linalg.norm(query_vec)
            if norm_q > 0:
                for name, skill in self._on_demand_skills.items():
                    embed_text = self._skill_search_text(name, skill)
                    vec = model.encode(embed_text)
                    norm_v = np.linalg.norm(vec)
                    if norm_v == 0:
                        continue
                    cosine = float(np.dot(query_vec, vec) / (norm_q * norm_v))
                    if cosine > 0.2:
                        scored[name] = scored.get(name, 0.0) + cosine

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        matched_skills = [name for name, _ in ranked[:top_k]]

        if not matched_skills:
            available = ", ".join(self._on_demand_skills.keys())
            return [], (
                "No matching tools found yet. Try a skill name or a plainer "
                f"description of the task. Available on-demand skills: {available}."
            )

        tools = []
        result_parts = []
        for skill_name in matched_skills:
            skill_tools = self._on_demand_tools.get(skill_name, [])
            tools.extend(skill_tools)
            skill = self._on_demand_skills[skill_name]
            tool_names = [td["function"]["name"] for td in skill_tools]
            summary = self._skill_display_summary(skill_name, skill)
            result_parts.append(f"- {skill_name}: {summary}")
            result_parts.append(f"  Tools: {', '.join(tool_names)}")
            if skill.workflow:
                result_parts.append(f"  Workflow: {skill.workflow}")

        result_text = (
            f"Loaded {len(tools)} tools. They are available now in this same turn. "
            "No requested action has been performed yet; search_tools only opens "
            "access to tools. If you intended to act, call the appropriate loaded "
            "tool now before describing any result.\n" + "\n".join(result_parts)
        )
        return tools, result_text

    def _get_embedding_model(self):
        try:
            from core.context import _get_embedding_model
            return _get_embedding_model()
        except Exception:
            return None

    # ── Execution ───────────────────────────────────────────────

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call by routing to the appropriate skill."""
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

    def get_all_skills(self) -> list[BaseSkill]:
        """Return all registered skills."""
        return list(self.skills.values())
