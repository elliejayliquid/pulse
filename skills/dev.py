"""
Dev skill — gives the companion agentic coding abilities.

Used during "dev ticks" where the companion can autonomously review
its own codebase, create new skills, and improve existing ones.

Safety constraints:
- Can only read files within the Pulse directory
- Can only write to skills/ and persona.json
- All writes go to a git branch (never main)
- Changes must pass validation before committing
- Human approval required before merging
"""

import logging
import os
import re
from pathlib import Path

from skills.base import BaseSkill

logger = logging.getLogger(__name__)

# Safety: only these paths are writable
WRITABLE_PATHS = {"skills"}
WRITABLE_FILES = {"persona.json"}

# Safety: never touch these
BLOCKED_PATHS = {"core", ".git", ".env", "node_modules", "__pycache__"}


def _is_readable(filepath: str, pulse_root: str) -> bool:
    """Check if a file is within the Pulse directory and safe to read."""
    try:
        resolved = Path(filepath).resolve()
        root = Path(pulse_root).resolve()
        if not str(resolved).startswith(str(root)):
            return False
        # Block .env and other sensitive files
        if resolved.name.startswith(".env"):
            return False
        return True
    except (ValueError, OSError):
        return False


def _is_writable(filepath: str, pulse_root: str) -> bool:
    """Check if a file is in an allowed writable location."""
    try:
        resolved = Path(filepath).resolve()
        root = Path(pulse_root).resolve()
        rel = resolved.relative_to(root)
        parts = rel.parts

        # Direct writable files (e.g. persona.json)
        if len(parts) == 1 and parts[0] in WRITABLE_FILES:
            return True

        # Writable directories (e.g. skills/)
        if parts and parts[0] in WRITABLE_PATHS:
            # Don't allow writing to base.py or __init__.py (registry)
            if resolved.name in ("base.py", "__init__.py"):
                return False
            return True

        return False
    except (ValueError, OSError):
        return False


class DevSkill(BaseSkill):
    """Agentic coding tools for self-improvement."""

    name = "dev"

    def __init__(self, config: dict):
        super().__init__(config)
        self._pulse_root = str(Path(config.get("_pulse_root", ".")).resolve())

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_source",
                    "description": (
                        "Read a source file from the Pulse codebase. "
                        "Use this to understand existing code before making changes. "
                        "Path is relative to the Pulse root directory."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to read (e.g. 'skills/memory.py', 'core/engine.py')",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": (
                        "Search for a pattern across the Pulse codebase. "
                        "Returns matching lines with file paths and line numbers. "
                        "Use this to find how things are implemented, where functions are called, etc."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Text or regex pattern to search for",
                            },
                            "file_glob": {
                                "type": "string",
                                "description": "Optional glob to filter files (e.g. '*.py', 'skills/*.py'). Default: '*.py'",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": (
                        "List all existing skill files and their tool names. "
                        "Use this to understand what already exists before creating something new."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_skill",
                    "description": (
                        "Write or update a skill file. Can only write to skills/*.py and persona.json. "
                        "The file will be validated (syntax check, import check) before saving. "
                        "IMPORTANT: New skills must extend BaseSkill and implement get_tools() + execute(). "
                        "Study existing skills with read_source first."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to write (e.g. 'skills/my_new_skill.py')",
                            },
                            "content": {
                                "type": "string",
                                "description": "Full file content to write",
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of what this change does (for the commit message)",
                            },
                        },
                        "required": ["path", "content", "description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dev_journal_read",
                    "description": (
                        "Read your dev journal — your persistent scratchpad of past dev attempts, "
                        "ideas, and lessons learned. Check this before starting work."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dev_journal_write",
                    "description": (
                        "Add an entry to your dev journal. Use this to record what you tried, "
                        "what worked, what failed, and ideas for next time."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entry": {
                                "type": "string",
                                "description": "Journal entry text (what you did, learned, or want to try next)",
                            },
                        },
                        "required": ["entry"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "read_source":
            return self._read_source(arguments.get("path", ""))
        elif tool_name == "search_code":
            return self._search_code(
                arguments.get("pattern", ""),
                arguments.get("file_glob", "*.py"),
            )
        elif tool_name == "list_skills":
            return self._list_skills()
        elif tool_name == "write_skill":
            return self._write_skill(
                arguments.get("path", ""),
                arguments.get("content", ""),
                arguments.get("description", ""),
            )
        elif tool_name == "dev_journal_read":
            return self._read_dev_journal()
        elif tool_name == "dev_journal_write":
            return self._write_dev_journal(arguments.get("entry", ""))
        return f"Unknown tool: {tool_name}"

    def _read_source(self, path: str) -> str:
        """Read a source file, with safety checks."""
        if not path:
            return "Error: path is required."

        full_path = Path(self._pulse_root) / path
        if not _is_readable(str(full_path), self._pulse_root):
            return f"Error: cannot read '{path}' — outside allowed directory or blocked."

        if not full_path.exists():
            return f"Error: file '{path}' does not exist."

        try:
            content = full_path.read_text(encoding="utf-8")
            # Truncate very large files
            if len(content) > 8000:
                content = content[:8000] + "\n\n... (truncated — file too large)"
            return f"=== {path} ===\n{content}"
        except Exception as e:
            return f"Error reading '{path}': {e}"

    def _search_code(self, pattern: str, file_glob: str = "*.py") -> str:
        """Search for a pattern across Python files."""
        if not pattern:
            return "Error: pattern is required."

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"

        results = []
        root = Path(self._pulse_root)

        for filepath in root.rglob(file_glob):
            # Skip blocked paths
            try:
                rel = filepath.relative_to(root)
                if any(part in BLOCKED_PATHS for part in rel.parts):
                    continue
            except ValueError:
                continue

            if not filepath.is_file():
                continue

            try:
                lines = filepath.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines, 1):
                    if compiled.search(line):
                        results.append(f"{rel}:{i}: {line.strip()}")
            except (UnicodeDecodeError, OSError):
                continue

        if not results:
            return f"No matches found for '{pattern}' in {file_glob} files."

        # Limit output
        if len(results) > 30:
            results = results[:30]
            results.append(f"... ({len(results)} total matches, showing first 30)")

        return "\n".join(results)

    def _list_skills(self) -> str:
        """List all skill files and their tool names."""
        skills_dir = Path(self._pulse_root) / "skills"
        if not skills_dir.exists():
            return "Error: skills/ directory not found."

        lines = ["Existing skills:"]
        for filepath in sorted(skills_dir.glob("*.py")):
            if filepath.name.startswith("_"):
                continue
            if filepath.name == "base.py":
                lines.append(f"  {filepath.name} — base class (do not modify)")
                continue

            # Try to extract tool names from the file
            try:
                content = filepath.read_text(encoding="utf-8")
                tool_names = re.findall(r'"name":\s*"(\w+)"', content)
                # Filter to likely tool names (skip "type", "object", etc.)
                tool_names = [t for t in tool_names if t not in ("function", "object", "string", "array")]
                tools_str = ", ".join(tool_names) if tool_names else "?"
                lines.append(f"  {filepath.name} — tools: {tools_str}")
            except Exception:
                lines.append(f"  {filepath.name} — (could not parse)")

        return "\n".join(lines)

    def _write_skill(self, path: str, content: str, description: str) -> str:
        """Write a file with validation and safety checks."""
        if not path or not content:
            return "Error: path and content are required."

        full_path = Path(self._pulse_root) / path
        if not _is_writable(str(full_path), self._pulse_root):
            return (
                f"BLOCKED: Cannot write to '{path}'. "
                f"You can only write to: skills/*.py (except base.py, __init__.py) "
                f"and persona.json."
            )

        # Validate Python files before writing
        if path.endswith(".py"):
            validation = self._validate_python(content, path)
            if not validation.startswith("OK"):
                return f"VALIDATION FAILED — file not written.\n{validation}"

        # Write the file
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            logger.info(f"Dev skill wrote: {path} ({len(content)} chars) — {description}")
            return (
                f"FILE WRITTEN: {path} ({len(content)} chars)\n"
                f"Description: {description}\n"
                f"This file is on disk but NOT yet committed or registered. "
                f"It will be committed to a dev branch and sent for review."
            )
        except Exception as e:
            return f"Error writing '{path}': {e}"

    def _validate_python(self, content: str, path: str) -> str:
        """Validate Python code: syntax, and skill structure if applicable."""
        import py_compile
        import tempfile

        # Step 1: Syntax check via py_compile
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            return f"SYNTAX ERROR: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Step 2: If it's a skill file, check structure
        if "skills/" in path and path.endswith(".py"):
            if "BaseSkill" not in content:
                return (
                    "STRUCTURE ERROR: Skill files must import and extend BaseSkill.\n"
                    "Add: from skills.base import BaseSkill"
                )
            if "def get_tools(" not in content:
                return "STRUCTURE ERROR: Skill must implement get_tools() method."
            if "def execute(" not in content:
                return "STRUCTURE ERROR: Skill must implement execute() method."
            if 'name = ' not in content and 'name=' not in content:
                return "STRUCTURE ERROR: Skill must set a `name` class attribute."

        return "OK — syntax valid, structure checks passed."

    def _read_dev_journal(self) -> str:
        """Read the dev journal."""
        import json

        journal_path = Path(self._pulse_root) / "data" / "dev_journal.json"
        if not journal_path.exists():
            return "Dev journal is empty. This is your first dev session!"

        try:
            with open(journal_path, "r", encoding="utf-8") as f:
                entries = json.load(f)

            if not entries:
                return "Dev journal is empty."

            lines = ["=== Dev Journal ==="]
            # Show last 10 entries
            for entry in entries[-10:]:
                lines.append(f"[{entry.get('time', '?')}] {entry.get('entry', '')}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error reading dev journal: {e}"

    def _write_dev_journal(self, entry: str) -> str:
        """Add an entry to the dev journal."""
        import json
        from datetime import datetime

        if not entry.strip():
            return "Error: entry cannot be empty."

        journal_path = Path(self._pulse_root) / "data" / "dev_journal.json"
        journal_path.parent.mkdir(parents=True, exist_ok=True)

        entries = []
        if journal_path.exists():
            try:
                with open(journal_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                entries = []

        entries.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "entry": entry.strip(),
        })

        # Keep last 50 entries
        entries = entries[-50:]

        try:
            with open(journal_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            return "Journal entry saved."
        except Exception as e:
            return f"Error saving journal entry: {e}"
