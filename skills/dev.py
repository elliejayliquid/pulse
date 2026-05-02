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

# Paths to skip during search/listing (dependencies, caches, VCS)
SKIP_PATHS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".vs"}

# Files that can never be deleted even within writable paths
PROTECTED_FILES = {"base.py", "__init__.py"}


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
            # Don't allow writing to base.py or __init__.py (core infrastructure)
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
        self._dev_journal_path = Path(
            config.get("paths", {}).get(
                "dev_journal",
                str(Path(self._pulse_root) / "data" / "dev_journal.json"),
            )
        )

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
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": (
                        "List files and directories in the Pulse project. "
                        "Use this to explore the project structure and discover files. "
                        "Path is relative to the Pulse root (use '.' for root)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative directory path (e.g. '.', 'skills', 'core', 'channels'). Default: '.'",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": (
                        "Edit a file by finding and replacing a specific block of text. "
                        "More efficient than write_skill for small changes. "
                        "The target text must match EXACTLY (including whitespace/indentation). "
                        "Can only edit files in skills/*.py and persona.json."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to edit (e.g. 'skills/memory.py')",
                            },
                            "target": {
                                "type": "string",
                                "description": "Exact text block to find (must match exactly once)",
                            },
                            "replacement": {
                                "type": "string",
                                "description": "Text to replace the target with",
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of the edit",
                            },
                        },
                        "required": ["path", "target", "replacement", "description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_file",
                    "description": (
                        "Check a Python file for syntax errors and skill structure issues "
                        "WITHOUT writing anything. Use this to verify code before editing, "
                        "or to diagnose problems with an existing file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to check (e.g. 'skills/memory.py')",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_skill",
                    "description": (
                        "Dry-run a skill file: import it, instantiate the class, "
                        "and verify get_tools() returns valid definitions. "
                        "Catches import errors, missing dependencies, and broken constructors. "
                        "Use this after writing or editing a skill to verify it will load."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the skill file (e.g. 'skills/my_skill.py')",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": (
                        "Delete a file from the project. Can only delete files in "
                        "skills/*.py (except base.py and __init__.py). "
                        "Use this to clean up broken or abandoned skill files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to delete (e.g. 'skills/broken_skill.py')",
                            },
                        },
                        "required": ["path"],
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
        elif tool_name == "list_dir":
            return self._list_dir(arguments.get("path", "."))
        elif tool_name == "edit_file":
            return self._edit_file(
                arguments.get("path", ""),
                arguments.get("target", ""),
                arguments.get("replacement", ""),
                arguments.get("description", ""),
            )
        elif tool_name == "validate_file":
            return self._validate_file(arguments.get("path", ""))
        elif tool_name == "test_skill":
            return self._test_skill(arguments.get("path", ""))
        elif tool_name == "delete_file":
            return self._delete_file(arguments.get("path", ""))
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
                if any(part in SKIP_PATHS for part in rel.parts):
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
                f"This file is on disk and will be auto-discovered on next restart. "
                f"New skills are loaded automatically — no manual registration needed. "
                f"Just make sure your class extends BaseSkill and sets a `name` attribute."
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

        journal_path = self._dev_journal_path
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

        journal_path = self._dev_journal_path
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

    def _list_dir(self, path: str = ".") -> str:
        """List directory contents within the Pulse project."""
        target = (Path(self._pulse_root) / path).resolve()
        root = Path(self._pulse_root).resolve()

        if not str(target).startswith(str(root)):
            return f"Error: '{path}' is outside the project directory."
        if not target.exists():
            return f"Error: '{path}' does not exist."
        if not target.is_dir():
            return f"Error: '{path}' is not a directory. Use read_source to view files."

        lines = [f"Contents of {path}/"]
        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for entry in entries:
                if entry.name in SKIP_PATHS:
                    continue
                if entry.is_dir():
                    try:
                        count = sum(1 for _ in entry.iterdir())
                    except PermissionError:
                        count = "?"
                    lines.append(f"  {entry.name}/ ({count} items)")
                else:
                    size = entry.stat().st_size
                    if size >= 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    lines.append(f"  {entry.name} ({size_str})")
        except PermissionError:
            return f"Error: permission denied for '{path}'."

        if len(lines) == 1:
            lines.append("  (empty)")

        return "\n".join(lines)

    def _edit_file(self, path: str, target: str, replacement: str, description: str) -> str:
        """Edit a file by finding and replacing a specific block of text."""
        if not path or not target:
            return "Error: path and target are required."

        full_path = Path(self._pulse_root) / path
        if not _is_writable(str(full_path), self._pulse_root):
            return (
                f"BLOCKED: Cannot edit '{path}'. "
                f"You can only edit: skills/*.py (except base.py, __init__.py) "
                f"and persona.json."
            )

        if not full_path.exists():
            return f"Error: file '{path}' does not exist. Use write_skill to create new files."

        try:
            content = full_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading '{path}': {e}"

        count = content.count(target)
        if count == 0:
            return (
                f"Error: target text not found in '{path}'. "
                f"Make sure you match the exact text including whitespace."
            )
        if count > 1:
            return (
                f"Error: target text found {count} times in '{path}'. "
                f"Provide a longer/more specific target to match exactly once."
            )

        new_content = content.replace(target, replacement, 1)

        if path.endswith(".py"):
            validation = self._validate_python(new_content, path)
            if not validation.startswith("OK"):
                return f"VALIDATION FAILED — file not modified.\n{validation}"

        try:
            full_path.write_text(new_content, encoding="utf-8")
            logger.info(f"Dev skill edited: {path} — {description}")
            return (
                f"FILE EDITED: {path}\n"
                f"Replaced {len(target)} chars with {len(replacement)} chars.\n"
                f"Description: {description}"
            )
        except Exception as e:
            return f"Error writing '{path}': {e}"

    def _delete_file(self, path: str) -> str:
        """Delete a file with safety checks."""
        if not path:
            return "Error: path is required."

        full_path = Path(self._pulse_root) / path
        if not _is_writable(str(full_path), self._pulse_root):
            return (
                f"BLOCKED: Cannot delete '{path}'. "
                f"You can only delete files in: skills/*.py "
                f"(except base.py and __init__.py)."
            )

        if not full_path.exists():
            return f"Error: file '{path}' does not exist."

        if full_path.name in PROTECTED_FILES:
            return f"BLOCKED: '{full_path.name}' is a protected core file and cannot be deleted."

        try:
            full_path.unlink()
            logger.info(f"Dev skill deleted: {path}")
            return f"FILE DELETED: {path}\nThe skill will be unloaded on next restart."
        except Exception as e:
            return f"Error deleting '{path}': {e}"

    def _validate_file(self, path: str) -> str:
        """Standalone syntax and structure check on a Python file."""
        if not path:
            return "Error: path is required."

        full_path = Path(self._pulse_root) / path
        if not _is_readable(str(full_path), self._pulse_root):
            return f"Error: cannot read '{path}' — outside allowed directory or blocked."
        if not full_path.exists():
            return f"Error: file '{path}' does not exist."
        if not path.endswith(".py"):
            return f"Error: validate_file only works on Python files."

        try:
            content = full_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading '{path}': {e}"

        return self._validate_python(content, path)

    def _test_skill(self, path: str) -> str:
        """Dry-run import and instantiation of a skill file."""
        import subprocess
        import sys

        if not path:
            return "Error: path is required."
        if not path.startswith("skills/") or not path.endswith(".py"):
            return "Error: test_skill only works on skills/*.py files."

        full_path = Path(self._pulse_root) / path
        if not full_path.exists():
            return f"Error: file '{path}' does not exist."

        # Build a test script that imports and instantiates the skill
        module_name = path.replace("/", ".").replace(".py", "")
        test_script = (
            "import sys, json\n"
            "sys.path.insert(0, {root!r})\n"
            "try:\n"
            "    import importlib, inspect\n"
            "    from skills.base import BaseSkill\n"
            "    mod = importlib.import_module({mod!r})\n"
            "    classes = [\n"
            "        (name, cls) for name, cls in inspect.getmembers(mod, inspect.isclass)\n"
            "        if issubclass(cls, BaseSkill) and cls is not BaseSkill\n"
            "    ]\n"
            "    if not classes:\n"
            "        print('FAIL: No BaseSkill subclass found in module.')\n"
            "        sys.exit(1)\n"
            "    for name, cls in classes:\n"
            "        skill_name = getattr(cls, 'name', None)\n"
            "        if not skill_name:\n"
            "            print(f'FAIL: {{name}} has no `name` class attribute.')\n"
            "            sys.exit(1)\n"
            "        instance = cls({{}})\n"
            "        tools = instance.get_tools()\n"
            "        tool_names = [t['function']['name'] for t in tools]\n"
            "        print(f'OK: {{name}} (skill={{skill_name!r}}) — {{len(tools)}} tools: {{tool_names}}')\n"
            "except Exception as e:\n"
            "    print(f'FAIL: {{type(e).__name__}}: {{e}}')\n"
            "    sys.exit(1)\n"
        ).format(root=self._pulse_root, mod=module_name)

        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self._pulse_root,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                return f"TEST PASSED:\n{output}"
            else:
                return f"TEST FAILED:\n{output}"
        except subprocess.TimeoutExpired:
            return "TEST FAILED: Skill took too long to load (>10s). Possible infinite loop?"
        except Exception as e:
            return f"TEST ERROR: {e}"
