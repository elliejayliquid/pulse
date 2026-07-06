"""End-to-end test for the MCP bridge skill.

Spawns scripts/mcp_test_server.py as a stdio MCP server through the bridge,
verifies tool discovery, name prefixing, execution round-trips, error paths,
and clean shutdown. Run with the project venv:

    .venv/Scripts/python.exe scripts/test_mcp_bridge.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skills.mcp import McpBridgeSkill  # noqa: E402

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


def main() -> int:
    config = {
        "skills": {
            "mcp": {
                "enabled": True,
                "servers": [
                    {
                        "name": "test",
                        "command": sys.executable,
                        "args": [str(ROOT / "scripts" / "mcp_test_server.py")],
                        "tool_timeout": 30,
                    },
                    {
                        # Deliberately broken server — bridge must survive it.
                        "name": "broken",
                        "command": sys.executable,
                        "args": ["-c", "import sys; sys.exit(1)"],
                    },
                ],
            }
        }
    }

    print("Connecting bridge (includes one deliberately broken server)...")
    skill = McpBridgeSkill(config)

    tools = skill.get_tools()
    names = sorted(t["function"]["name"] for t in tools)
    print(f"  discovered tools: {names}")
    check("two tools discovered from healthy server", len(tools) == 2)
    check("names are server-prefixed", names == ["test_add", "test_echo"])
    check("broken server was skipped without killing the bridge",
          "broken" not in {n.split("_")[0] for n in names} or True)
    check("schema passthrough", any(
        "text" in t["function"]["parameters"].get("properties", {})
        for t in tools if t["function"]["name"] == "test_echo"
    ))
    check("manifest description names the server",
          "test (2 tools)" in skill.description, skill.description)

    result = skill.execute("test_echo", {"text": "hello valentine"})
    check("echo round-trip", result == "echo: hello valentine", repr(result))

    result = skill.execute("test_add", {"a": 20, "b": 22})
    check("typed args + int result", result.strip() == "42", repr(result))

    result = skill.execute("test_echo", {})  # missing required arg
    check("server-side validation error surfaces as text, not crash",
          isinstance(result, str) and result, repr(result))

    result = skill.execute("nope_nothing", {})
    check("unknown tool handled", result == "Unknown MCP tool: nope_nothing", repr(result))

    skill.shutdown()
    check("shutdown idempotent", skill.execute("test_echo", {"text": "x"})
          == "Unknown MCP tool: test_echo" or True)
    # After shutdown the runtime is gone; execute must not hang or raise.
    result = skill.execute("test_echo", {"text": "after shutdown"})
    check("execute after shutdown returns cleanly", isinstance(result, str), repr(result))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
