"""Tiny stdio MCP server used by scripts/test_mcp_bridge.py.

Run standalone: python scripts/mcp_test_server.py (speaks MCP over stdio).
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pulse-test")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back, prefixed so the round-trip is verifiable."""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
