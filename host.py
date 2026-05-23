"""Orchestrates multiple MCP clients and an LLM into an agentic REPL."""
import asyncio
import json
import os
import sys

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from mcp_client import MCPClient
from llm_provider import LLMProvider, ToolCall, build_provider

load_dotenv()

CONFIG_FILE = Path(__file__).parent / "mcp_servers.json"

_SYSTEM_ENV_KEYS = frozenset({})

MAX_TOOL_ITERATIONS = 20

HELP_TEXT = """
Commands:
  /tools   — list all available tools
  /history — print conversation history
  /reset   — clear conversation history
  /quit    — exit
"""


class MCPHost:
    """Central coordinator: loads config, connects MCP servers, and drives the agent loop."""

    def __init__(self) -> None:
        """Initialise empty state; call run() to connect servers and start the REPL."""
        self._clients: dict[str, MCPClient] = {}
        self._provider: LLMProvider | None = None
        # name visible to LLM → (client, original tool name)
        self._tool_index: dict[str, tuple[MCPClient, str]] = {}
        self._tools_schema: list[dict] = []
        self._history: list[dict] = []
        self._exit_stack = AsyncExitStack()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def _load(self, config_path: Path) -> None:
        """Read mcp_servers.json, instantiate the LLM provider, and connect all MCP clients."""
        with config_path.open() as f:
            config: dict = json.load(f)

        self._provider = build_provider(config.get("llm", {}))

        servers: dict[str, dict] = config.get("mcpServers", {})
        for server_name, server_cfg in servers.items():
            command = server_cfg["command"]
            args = server_cfg.get("args", [])
            allowed_keys = _SYSTEM_ENV_KEYS | set(server_cfg.get("allowedEnv", []))
            env = {k: v for k, v in os.environ.items() if k in allowed_keys}
            env.update(server_cfg.get("env", {}))
            client = MCPClient(name=server_name, command=command, args=args, env=env)
            await self._exit_stack.enter_async_context(client)
            self._clients[server_name] = client

        await self._build_tool_index()

    async def _build_tool_index(self) -> None:
        """Build a namespaced tool registry (servername__toolname) across all connected clients."""
        self._tool_index = {}
        self._tools_schema = []

        for server_name, client in self._clients.items():
            for tool in await client.list_tools():
                namespaced = f"{server_name}__{tool.name}"
                self._tool_index[namespaced] = (client, tool.name)
                schema: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": namespaced,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
                    },
                }
                self._tools_schema.append(schema)

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    async def _dispatch_tool(self, call: ToolCall) -> str:
        """Route a tool call to the correct MCPClient and return the result as a string."""
        if call.name not in self._tool_index:
            return f"Error: unknown tool '{call.name}'"
        client, original_name = self._tool_index[call.name]
        print(f"  → {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        try:
            result = str(await client.call_tool(original_name, call.arguments))
            print(f"  ← {result[:200]}")
            return result
        except Exception as exc:
            error = f"Error: {exc}"
            print(f"  ← {error}")
            return error

    async def _turn(self, user_input: str) -> None:
        """Run one conversation turn: call the LLM, execute any tool calls, loop until a final answer."""
        self._history.append({"role": "user", "content": user_input})

        assert self._provider is not None
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3
        for _ in range(MAX_TOOL_ITERATIONS):
            response = await self._provider.chat(self._history, self._tools_schema)

            if response.tool_calls:
                # Append the assistant's tool-calling message
                self._history.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in response.tool_calls
                    ],
                })
                # Run each tool and append results
                all_errors = True
                for tc in response.tool_calls:
                    result = await self._dispatch_tool(tc)
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    if not result.startswith("Error:"):
                        all_errors = False

                if all_errors:
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        last_error = self._history[-1]["content"]
                        print(f"\n[Stopped] Tool kept failing after {consecutive_errors} attempts: {last_error}\n")
                        break
                else:
                    consecutive_errors = 0
            else:
                # Final answer
                self._history.append({"role": "assistant", "content": response.content or ""})
                print(f"\nAssistant: {response.content}\n")
                break
        else:
            print(f"\n[Warning] Reached {MAX_TOOL_ITERATIONS} tool iterations without a final answer.\n")

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def _print_history(self) -> None:
        """Print the conversation history."""
        if not self._history:
            print("No history yet.")
            return
        print()
        for msg in self._history:
            role = msg["role"].capitalize()
            if msg["role"] == "tool":
                print(f"  [Tool result id={msg.get('tool_call_id', '')}] {msg['content'][:200]}")
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                calls = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
                print(f"  {role}: <tool calls: {calls}>")
            else:
                print(f"  {role}: {msg.get('content', '')}")
        print()

    def _print_tools(self) -> None:
        """Print all available tools with their descriptions."""
        if not self._tool_index:
            print("No tools available.")
            return
        print("\nAvailable tools:")
        for namespaced, (_, original) in self._tool_index.items():
            server = namespaced.split("__")[0]
            schema = next(
                (s["function"] for s in self._tools_schema if s["function"]["name"] == namespaced), {}
            )
            desc = schema.get("description", "")
            print(f"  {namespaced:<40} {desc}")
        print()

    async def _read_input(self, prompt: str) -> str:
        """Read a line of user input without blocking the event loop."""
        return await asyncio.to_thread(input, prompt)

    async def run(self, config_path: Path = CONFIG_FILE) -> None:
        """Load config, connect servers, then run the interactive REPL until the user quits."""
        print(f"Loading config from {config_path} ...")
        await self._load(config_path)
        print(f"\nReady. {len(self._tool_index)} tool(s) loaded. Type /help for commands.\n")

        try:
            while True:
                try:
                    line = await self._read_input("> ")
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break

                line = line.strip()
                if not line:
                    continue
                if line in ("/quit", "/exit", "quit", "exit"):
                    print("Goodbye.")
                    break
                if line == "/history":
                    self._print_history()
                    continue
                if line == "/reset":
                    self._history.clear()
                    print("History cleared.")
                    continue
                if line == "/tools":
                    self._print_tools()
                    continue
                if line in ("/help", "help"):
                    print(HELP_TEXT)
                    continue

                await self._turn(line)
        finally:
            await self._exit_stack.aclose()


async def main() -> None:
    """Parse an optional config path argument and run the MCPHost."""
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_FILE
    host = MCPHost()
    await host.run(config_path)


if __name__ == "__main__":
    asyncio.run(main())
