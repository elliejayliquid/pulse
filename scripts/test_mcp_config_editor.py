"""Tests for the ConfigEditor mcp_servers change group.

Runs against a throwaway persona in a temp directory:

    .venv/Scripts/python.exe scripts/test_mcp_config_editor.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from gui.backup import BackupManager  # noqa: E402
from gui.config_editor import ConfigEditor  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}  {detail}")


BASE_CONFIG = """# Test persona config — comments must survive edits.
provider:
  type: "anthropic"   # keep me
  model: "claude-opus-4-6"

skills:
  memory:
    enabled: true
  mcp:
    enabled: true

heartbeat:
  interval_minutes: 30
"""

SERVERS = [
    {"name": "memory", "command": "python", "args": ["srv.py"], "tool_timeout": 45},
    {"name": "docs", "url": "http://127.0.0.1:8765/mcp"},
]


def expect_error(editor, changes, label):
    try:
        editor.preview("testp", changes)
    except ValueError as e:
        check(label, True, str(e))
        return
    check(label, False, "no error raised")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "testp"
        persona_dir.mkdir(parents=True)
        (persona_dir / "persona.yaml").write_text("name: Test\n", encoding="utf-8")
        config_path = persona_dir / "config.yaml"
        config_path.write_text(BASE_CONFIG, encoding="utf-8")

        editor = ConfigEditor(root, BackupManager(root))

        # 1. Save a server list into an existing skills.mcp block
        result = editor.save("testp", {"mcp_servers": SERVERS})
        check("save reports changed", result["ok"] and result["changed"])
        text = config_path.read_text(encoding="utf-8")
        check("comments survive", "# keep me" in text and "comments must survive" in text)
        parsed = yaml.safe_load(text)
        servers = parsed["skills"]["mcp"]["servers"]
        check("round-trips through yaml", len(servers) == 2, repr(servers))
        check("stdio entry intact", servers[0] == SERVERS[0], repr(servers[0]))
        check("url entry intact", servers[1] == SERVERS[1], repr(servers[1]))
        check("sibling keys untouched",
              parsed["skills"]["memory"]["enabled"] is True
              and parsed["skills"]["mcp"]["enabled"] is True
              and parsed["heartbeat"]["interval_minutes"] == 30)

        # 2. Replace with a single server (full-replace semantics)
        editor.save("testp", {"mcp_servers": [SERVERS[1]]})
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        check("list fully replaced", parsed["skills"]["mcp"]["servers"] == [SERVERS[1]])

        # 3. Clear all servers with an empty list
        editor.save("testp", {"mcp_servers": []})
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        check("empty list clears servers", parsed["skills"]["mcp"]["servers"] == [])

        # 4. None / absent leaves the file alone
        result = editor.save("testp", {"skills": {"memory": True}})
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        check("absent group leaves servers alone", parsed["skills"]["mcp"]["servers"] == [])

        # 5. Creating the block when skills.mcp doesn't exist at all
        config_path.write_text("provider:\n  type: \"local\"\n", encoding="utf-8")
        editor.save("testp", {"mcp_servers": SERVERS})
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        check("creates skills.mcp.servers from scratch",
              parsed["skills"]["mcp"]["servers"] == SERVERS)

        # 6. Validation errors
        expect_error(editor, {"mcp_servers": [{"name": "x", "command": "a", "url": "http://b"}]},
                     "rejects url+command together")
        expect_error(editor, {"mcp_servers": [{"name": "x"}]},
                     "rejects neither url nor command")
        expect_error(editor, {"mcp_servers": [{"name": "bad name!", "command": "a"}]},
                     "rejects invalid name")
        expect_error(editor, {"mcp_servers": [
            {"name": "dup", "command": "a"}, {"name": "DUP", "command": "b"}]},
                     "rejects duplicate names")
        expect_error(editor, {"mcp_servers": [{"name": "x", "url": "ftp://nope"}]},
                     "rejects non-http url")
        expect_error(editor, {"mcp_servers": [{"name": "x", "command": "a", "evil": 1}]},
                     "rejects unknown fields")
        expect_error(editor, {"mcp_servers": [{"name": "x", "command": "a", "tool_timeout": 0}]},
                     "rejects out-of-range timeout")
        expect_error(editor, {"mcp_servers": "nope"}, "rejects non-list")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
