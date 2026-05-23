"""Manages a single MCP server connection over stdio.

stdio (standard input/output) is MCP's process-based transport: the client spawns the MCP
server as a subprocess and exchanges JSON-RPC messages through its stdin and stdout pipes.
This is the simplest transport — no network port, no auth — suitable for local servers.
The alternative transports (SSE, WebSocket) are used for remote/network-hosted servers.
"""
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool


class MCPClient:
    """Manages a single MCP server connection over stdio.

    In MCP, a "tool" is a named function registered by the server that the LLM can call.
    Each tool has a name, a description the LLM reads to decide when to use it, and a JSON
    Schema describing its input parameters. This class exposes two operations on tools:
    list_tools() to discover what the server offers, and call_tool() to invoke one and get
    its result back as text.

    Supports both direct use (connect/aclose) and the async context manager protocol
    (async with MCPClient(...) as client), which calls connect() on entry and aclose()
    on exit — even if an exception is raised inside the block.
    """

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        """Store server launch parameters without connecting."""
        self.name = name
        self._command = command
        self._args = args
        self._env = env
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        """Spawn the server process, complete the MCP handshake, and log available tools.

        Uses AsyncExitStack to track the subprocess and session so that aclose() can
        tear down both in the correct order.
        """
        params = StdioServerParameters(command=self._command, args=self._args, env=self._env or {})
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read, write = stdio_transport
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        tools = await self.list_tools()
        print(f"[{self.name}] connected — {len(tools)} tool(s): {[t.name for t in tools]}")

    async def list_tools(self) -> list[Tool]:
        """Return the tools registered by this server.

        Each Tool carries a name, a human-readable description, and a JSON Schema for its
        inputs — the host uses these to build the tools list sent to the LLM.
        """
        assert self._session, "Not connected"
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a server tool by name and return its text output as a string.

        MCP responses are lists of typed content blocks; this method concatenates all
        text blocks into a single string for the LLM to read.
        """
        assert self._session, "Not connected"
        result = await self._session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(str(result.content))
        texts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(texts) if texts else repr(result.content)

    async def aclose(self) -> None:
        """Close the server connection and release all resources.

        Unwinds the AsyncExitStack, which closes the ClientSession and terminates the
        subprocess in the correct order. Safe to call multiple times.
        """
        await self._exit_stack.aclose()

    async def __aenter__(self) -> "MCPClient":
        """Connect on entry — enables `async with MCPClient(...) as client:`."""
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Call aclose() on exit, whether the block succeeded or raised an exception."""
        await self.aclose()
