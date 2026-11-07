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

import hashlib
import importlib
import inspect
import json
import logging
import pkgutil
import re
from datetime import date, datetime
from pathlib import Path

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

        # Failure log — when a tool call crashes, it's recorded here as well
        # as logged, so failures become visible (scripts/health_check.py and
        # scripts/watchdog.py read it) instead of dying silently in a
        # per-persona logfile.
        persona_dir = config.get("_persona_dir")
        self._persona_name = config.get("_persona_name", "?")
        if persona_dir:
            self._failure_log = Path(persona_dir) / "data" / "failures.jsonl"
        else:
            self._failure_log = Path("data") / "failures.jsonl"

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

        # ── tool-cap guard ──────────────────────────────────────────
        # OpenAI hard-caps the `tools` array at 128 per request. Pulse does no
        # automatic trimming, so an over-cap OpenAI persona 400s on EVERY call —
        # SILENTLY. Surface it loudly at startup (log + failure log, which
        # health_check.py reads) so it's caught here, not in dead silence later.
        # Anthropic has no such cap; "custom" endpoints vary — check those by hand.
        provider_type = (config.get("provider", {}) or {}).get("type", "")
        tool_count = len(self.get_all_tools())
        if provider_type == "openai" and tool_count > 128:
            msg = (
                f"persona '{self._persona_name}' loaded {tool_count} tools on an "
                f"OpenAI provider — OpenAI rejects >128 per request, so EVERY API "
                f"call will 400. Disable unused skills in "
                f"personas/{self._persona_name}/config.yaml (e.g. games=13, "
                f"paint=6, dev=6) to get back under 128."
            )
            logger.warning(f"⚠️ TOOL CAP EXCEEDED: {msg}")
            try:
                self._failure_log.parent.mkdir(parents=True, exist_ok=True)
                with self._failure_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "persona": self._persona_name,
                        "tool": "<startup: tool-cap>",
                        "error": ("OpenAI 128-tool cap exceeded — " + msg)[:300],
                        "arg_keys": [],
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass

    # ── Tool retrieval ──────────────────────────────────────────

    def get_always_tools(self) -> list[dict]:
        """Tools for always-loaded skills + the search_tools meta-tool."""
        tools = list(self._always_tools)
        if self._on_demand_skills:
            tools.append(self._search_tools_definition())
        return tools

    def get_heartbeat_shelf_names(self, day_key: str | None = None) -> list[str]:
        """Return the small, daily-stable set of on-demand skills kept in reach.

        A heartbeat is open-ended, so a few concrete creative/social affordances
        matter more there than in an ordinary user-directed conversation.  The
        rotating slot changes only once per local calendar day: enough novelty
        to prevent a permanent chore-shaped toolbox without invalidating the
        provider tool cache on every tick.
        """
        heartbeat = self.config.get("heartbeat", {}) or {}

        configured_favorites = heartbeat.get(
            "tool_shelf_skills", ["tts", "sticker"]
        )
        if isinstance(configured_favorites, str):
            configured_favorites = [configured_favorites]
        if not isinstance(configured_favorites, list):
            configured_favorites = ["tts", "sticker"]

        names = []
        for raw_name in configured_favorites:
            name = str(raw_name).strip()
            if name in self._on_demand_skills and name not in names:
                names.append(name)

        rotation = heartbeat.get("tool_discovery_rotation", {}) or {}
        if not isinstance(rotation, dict) or not rotation.get("enabled", True):
            return names

        try:
            slots = max(0, min(2, int(rotation.get("skills_per_day", 1))))
        except (TypeError, ValueError):
            slots = 1
        try:
            max_tools = max(1, min(12, int(rotation.get("max_tools_per_skill", 8))))
        except (TypeError, ValueError):
            max_tools = 8

        raw_categories = rotation.get(
            "categories", ["creative", "social", "research"]
        )
        if isinstance(raw_categories, str):
            raw_categories = [raw_categories]
        categories = {
            str(category).strip().lower()
            for category in (raw_categories or [])
            if str(category).strip()
        }

        candidates = []
        for name, skill in self._on_demand_skills.items():
            if name in names:
                continue
            skill_categories = {
                str(category).strip().lower()
                for category in (getattr(skill, "categories", []) or [])
            }
            if categories and not (categories & skill_categories):
                continue
            if len(self._on_demand_tools.get(name, [])) > max_tools:
                continue
            candidates.append(name)

        if not candidates or slots == 0:
            return names

        identity = str(self.config.get("_persona_name") or "pulse")
        rotation_day = day_key or date.today().isoformat()
        seed = f"{identity}:{rotation_day}"
        candidates.sort(
            key=lambda name: hashlib.sha256(
                f"{seed}:{name}".encode("utf-8")
            ).digest()
        )
        names.extend(candidates[:slots])
        return names

    def get_heartbeat_tools(self, shelf_names: list[str] | None = None) -> list[dict]:
        """Always-loaded tools plus the heartbeat's direct-access shelf."""
        names = (
            self.get_heartbeat_shelf_names()
            if shelf_names is None else shelf_names
        )
        tools = self.get_always_tools()
        seen = {
            tool.get("function", {}).get("name")
            for tool in tools
        }
        for name in names:
            for tool in self._on_demand_tools.get(name, []):
                tool_name = tool.get("function", {}).get("name")
                if tool_name and tool_name not in seen:
                    tools.append(tool)
                    seen.add(tool_name)
        return tools

    def get_all_tools(self) -> list[dict]:
        """All tool definitions (for scheduled tasks that might need anything)."""
        tools = []
        for skill in self.skills.values():
            tools.extend(skill.get_tools())
        return tools

    def get_skill_summary(self, tools: list[dict] | None = None) -> list[dict]:
        """Compact summary of the supplied (or always-loaded) tool set."""
        visible_tools = self._always_tools if tools is None else tools
        visible_names = {
            td.get("function", {}).get("name", "?") for td in visible_tools
        }
        summary = []
        for name, skill in self.skills.items():
            tool_names = [
                t.get("function", {}).get("name", "?")
                for t in skill.get_tools()
                if t.get("function", {}).get("name", "?") in visible_names
            ]
            if tool_names:
                summary.append({"skill": name, "tools": tool_names})
        return summary

    @staticmethod
    def _unwrap_arguments(tool_name: str, arguments):
        """Defensively flatten a nested-argument envelope from the model.

        Some providers — notably Gemini via Google's OpenAI-compatible
        endpoint — intermittently emit tool calls whose real arguments are
        nested one level down inside a single-key {"parameters": {...}} (or
        {"args": {...}}) object instead of being passed flat. When that
        happens the skill sees no arguments at all and reports the required
        field as missing/empty — e.g. lighthouse_read -> "entry_id
        required.", write_journal -> "Cannot write empty journal entry." —
        which silently locks the companion out of their own tools.

        No tool in this codebase declares a lone dict-valued parameter named
        "parameters" or "args", so collapsing exactly that shape is a no-op
        for well-formed calls and only rescues the malformed ones. (Seen
        with gemini-2.5-pro.)
        """
        if isinstance(arguments, dict) and len(arguments) == 1:
            for wrap_key in ("parameters", "args"):
                inner = arguments.get(wrap_key)
                if isinstance(inner, dict):
                    logger.info(
                        f"Unwrapped '{wrap_key}' envelope for {tool_name} "
                        f"(model nested its arguments one level deep)"
                    )
                    return inner
        return arguments

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

    def get_on_demand_manifest(
        self, exclude_skills: set[str] | None = None
    ) -> list[dict]:
        """Compact summaries for prompt injection."""
        excluded = exclude_skills or set()
        manifest = []
        for name, skill in self._on_demand_skills.items():
            if name in excluded:
                continue
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
                    "Browse and load additional tools whose full definitions are not "
                    "currently in reach. Use this freely whenever you are curious about "
                    "what you can do, want a different kind of interaction, or need a "
                    "capability from the catalogue. Searching is ordinary toolbox "
                    "browsing and never needs an apology or special justification. "
                    "Available skills: " + skill_list + ". "
                    "Matching tool definitions become callable immediately in this same "
                    "turn. Browsing does not itself perform an action; if you decide to "
                    "do something, call the loaded tool before claiming its result."
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
            f"Loaded {len(tools)} tools; they are callable now in this same turn. "
            "Explore them freely, browse again, or decide not to act. Browsing the "
            "toolbox has not used any of these capabilities yet, so call the tool you "
            "choose before describing its result.\n" + "\n".join(result_parts)
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
        arguments = self._unwrap_arguments(tool_name, arguments)
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
            self._record_failure(tool_name, e, arguments)
            return f"Tool error: {e}"

    def _record_failure(self, tool_name: str, error: Exception, arguments: dict) -> None:
        """Append a tool-execution crash to the failure log for health_check
        to surface. Best-effort: logging a failure must never itself crash
        the turn, so any error here is swallowed."""
        try:
            # Argument keys only — never values, which could carry private
            # content (letter bodies, journal text, etc.).
            arg_keys = sorted(arguments.keys()) if isinstance(arguments, dict) else []
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "persona": self._persona_name,
                "tool": tool_name,
                "error": f"{type(error).__name__}: {error}"[:300],
                "arg_keys": arg_keys,
            }
            self._failure_log.parent.mkdir(parents=True, exist_ok=True)
            with self._failure_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as log_err:
            logger.debug(f"Could not record failure: {log_err}")

    def get_skill(self, name: str) -> BaseSkill | None:
        """Get a specific skill instance by name."""
        return self.skills.get(name)

    def get_all_skills(self) -> list[BaseSkill]:
        """Return all registered skills."""
        return list(self.skills.values())
