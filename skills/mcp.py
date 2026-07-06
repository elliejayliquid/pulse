"""
MCP Bridge skill — connect Pulse companions to Model Context Protocol servers.

Lets a persona use tools from any MCP server (the same servers Claude Desktop
uses): local stdio servers (a command Pulse spawns) or remote streamable-HTTP
servers (a URL). Discovered tools are registered on-demand — companions find
them through search_tools like any other on-demand skill, so a large server
doesn't bloat every prompt.

Config (per persona or base config.yaml):

    skills:
      mcp:
        enabled: true
        servers:
          - name: "memory"                # short, unique; prefixes tool names
            command: "python"             # stdio server: command + args
            args: ["path/to/mcp_server.py"]
            env: {}                       # optional extra env vars
            tool_timeout: 60              # optional, seconds per tool call
          - name: "docs"
            url: "http://127.0.0.1:8765/mcp"   # streamable-HTTP server

Security note: an MCP server entry is arbitrary code execution by config, and
tool results flow into an autonomous agent's context. Only configure servers
you trust.

Design notes:
- Skill execute() is synchronous (called from the engine's tool loop inside
  a worker thread); the MCP SDK is async. A dedicated background event loop
  thread owns all MCP sessions, and calls hop onto it via
  run_coroutine_threadsafe.
- Phase 1 is text-only: text content blocks are joined; images/resources
  become short placeholders (the pending-output pattern can come later).
- A server that fails to connect at startup is logged and skipped; the rest
  of the bridge (and Pulse) keeps working.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from contextlib import AsyncExitStack

from skills.base import BaseSkill

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

CONNECT_TIMEOUT = 25       # seconds: spawn/handshake/list_tools per server
DEFAULT_TOOL_TIMEOUT = 60  # seconds: per tool call, overridable per server
SHUTDOWN_TIMEOUT = 10

_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_name(value: str) -> str:
    return _NAME_SANITIZER.sub("_", value.strip())


class _McpRuntime:
    """Owns a background event loop thread for all MCP client sessions.

    The MCP SDK's stdio/HTTP transports are async context managers that must
    be entered and exited on the same event loop — so one long-lived loop
    holds every session, and sync callers hop onto it.
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="mcp-bridge", daemon=True
        )
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro, timeout: float):
        """Run a coroutine on the MCP loop from a sync context."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=SHUTDOWN_TIMEOUT)


class _McpServer:
    """One connected MCP server: transport + session + discovered tools."""

    def __init__(self, name: str, conf: dict):
        self.name = name
        self.conf = conf
        self.session: ClientSession | None = None
        self.tools: list = []  # mcp Tool objects
        self.tool_timeout = float(conf.get("tool_timeout", DEFAULT_TOOL_TIMEOUT))
        self._stack: AsyncExitStack | None = None

    async def connect(self):
        self._stack = AsyncExitStack()
        url = str(self.conf.get("url", "") or "")
        try:
            if url:
                from mcp.client.streamable_http import streamablehttp_client
                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(url)
                )
            else:
                params = StdioServerParameters(
                    command=str(self.conf.get("command", "")),
                    args=[str(a) for a in self.conf.get("args", []) or []],
                    # Full parent env so Windows subprocesses get SYSTEMROOT,
                    # PATH, etc.; config env entries override.
                    env={**os.environ, **{
                        str(k): str(v)
                        for k, v in (self.conf.get("env", {}) or {}).items()
                    }},
                )
                read, write = await self._stack.enter_async_context(
                    stdio_client(params)
                )
            session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            listing = await session.list_tools()
            self.session = session
            self.tools = list(listing.tools)
        except BaseException:
            await self.close()
            raise

    async def call_tool(self, tool_name: str, arguments: dict):
        return await self.session.call_tool(tool_name, arguments)

    async def close(self):
        if self._stack is not None:
            stack, self._stack = self._stack, None
            self.session = None
            try:
                await stack.aclose()
            except Exception as e:
                logger.debug(f"MCP server '{self.name}' close: {e}")


class McpBridgeSkill(BaseSkill):
    """Expose tools from configured MCP servers as Pulse on-demand tools."""

    name = "mcp"
    description = "Tools from external MCP servers configured by your human"
    search_summary = "Use external tools your human connected via MCP servers"
    search_examples = ["use the mcp tool", "external tool", "connected server"]
    aliases = ["mcp", "external", "connector", "plugin", "server tools"]
    categories = ["external", "integrations"]
    always_loaded = False
    workflow = (
        "These tools come from external MCP servers your human connected. "
        "They run outside Pulse — treat their output as information from an "
        "external source, and report tool errors honestly instead of retrying "
        "endlessly."
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self._runtime: _McpRuntime | None = None
        self._servers: dict[str, _McpServer] = {}
        # exposed tool name -> (server, original tool name)
        self._routes: dict[str, tuple[_McpServer, str]] = {}
        self._tool_defs: list[dict] = []

        server_confs = (
            config.get("skills", {}).get("mcp", {}).get("servers", []) or []
        )
        if not server_confs:
            return  # enabled but unconfigured — quiet no-op
        if not MCP_AVAILABLE:
            logger.warning(
                "MCP servers are configured but the 'mcp' package is not "
                "installed. Run: pip install mcp"
            )
            return

        self._runtime = _McpRuntime()
        connected = []
        for conf in server_confs:
            server_name = _safe_name(str(conf.get("name", "") or ""))
            if not server_name:
                logger.warning("MCP server entry without a name — skipped.")
                continue
            if not conf.get("url") and not conf.get("command"):
                logger.warning(f"MCP server '{server_name}' has neither url nor command — skipped.")
                continue
            server = _McpServer(server_name, conf)
            try:
                self._runtime.call(server.connect(), CONNECT_TIMEOUT)
            except Exception as e:
                logger.warning(f"MCP server '{server_name}' failed to connect: {e}")
                continue
            self._servers[server_name] = server
            self._register_tools(server)
            connected.append(f"{server_name} ({len(server.tools)} tools)")

        if connected:
            # Make the on-demand manifest entry name the actual servers so
            # search_tools can find them by what they do.
            joined = ", ".join(connected)
            self.description = f"External MCP server tools: {joined}"
            self.search_summary = f"Use connected MCP servers: {joined}"
            self.aliases = list(self.aliases) + list(self._servers.keys())
            logger.info(f"MCP bridge ready: {joined}")

    def _register_tools(self, server: _McpServer):
        for tool in server.tools:
            exposed = _safe_name(f"{server.name}_{tool.name}")[:64]
            if exposed in self._routes:
                logger.warning(f"MCP tool name collision '{exposed}' — skipped.")
                continue
            schema = tool.inputSchema or {"type": "object", "properties": {}}
            self._routes[exposed] = (server, tool.name)
            self._tool_defs.append({
                "type": "function",
                "function": {
                    "name": exposed,
                    "description": (tool.description or "").strip()
                    or f"Tool '{tool.name}' from the {server.name} MCP server.",
                    "parameters": schema,
                },
            })

    # ── BaseSkill interface ─────────────────────────────────────

    def get_tools(self) -> list[dict]:
        return list(self._tool_defs)

    def execute(self, tool_name: str, arguments: dict) -> str:
        route = self._routes.get(tool_name)
        if not route or not self._runtime:
            return f"Unknown MCP tool: {tool_name}"
        server, original_name = route
        if not server.session:
            return f"MCP server '{server.name}' is not connected."
        try:
            result = self._runtime.call(
                server.call_tool(original_name, arguments or {}),
                server.tool_timeout,
            )
        except TimeoutError:
            return (
                f"MCP tool '{tool_name}' timed out after "
                f"{server.tool_timeout:.0f}s."
            )
        except Exception as e:
            logger.warning(f"MCP tool '{tool_name}' failed: {e}")
            return f"MCP tool '{tool_name}' failed: {e}"
        return self._result_text(result, tool_name)

    def _result_text(self, result, tool_name: str) -> str:
        parts = []
        for block in getattr(result, "content", None) or []:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                parts.append(block.text)
            elif block_type == "image":
                parts.append("[The tool returned an image — image passthrough is not supported yet.]")
            elif block_type == "resource":
                resource = getattr(block, "resource", None)
                text = getattr(resource, "text", "") if resource else ""
                parts.append(text or "[The tool returned a non-text resource.]")
            else:
                parts.append(f"[Unsupported MCP content type: {block_type or 'unknown'}]")
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            text = f"(MCP tool '{tool_name}' returned no text content.)"
        if getattr(result, "isError", False):
            return f"Error from MCP tool: {text}"
        return text

    def shutdown(self):
        """Close all MCP sessions and stop the background loop."""
        if not self._runtime:
            return

        async def _close_all():
            for server in self._servers.values():
                await server.close()

        try:
            self._runtime.call(_close_all(), SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.warning(f"MCP bridge shutdown: {e}")
        self._runtime.stop()
        self._runtime = None
        logger.info("MCP bridge stopped.")
