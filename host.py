"""Orchestrates multiple MCP clients and an LLM into an agentic REPL."""
import argparse
import asyncio
import json
import os

from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from mcp_client import MCPClient
from agent_provider import AgentProvider, ToolCall, build_provider
from enums import EventType

load_dotenv()

CONFIG_FILE = Path(__file__).parent / "mcp_servers_config.json"
CUSTOM_COMMANDS_FILE = Path(__file__).parent / "custom_commands.json"

_SYSTEM_ENV_KEYS = frozenset({})

MAX_TOOL_ITERATIONS = 20

_BUILTIN_HELP = """\
Commands:
  /tools              — list all available MCP tools
  /commands           — list custom commands loaded from custom_commands.json
  /history            — print conversation history
  /reset              — clear conversation history
  /confirm [on|off]   — toggle (or set) tool-call confirmation gate
  /quit               — exit
"""


class MCPHost:
    """Central coordinator: loads config, connects MCP servers, and drives the agent loop."""

    def __init__(self) -> None:
        """Initialise empty state; call run() to connect servers and start the REPL."""
        self._clients: dict[str, MCPClient] = {}
        self._provider: AgentProvider | None = None
        # name visible to LLM → (client, original tool name)
        self._tool_index: dict[str, tuple[MCPClient, str]] = {}
        self._tools_schema: list[dict] = []
        self._history: list[dict] = []
        self._custom_commands: dict[str, dict] = {}
        self._exit_stack = AsyncExitStack()
        self._confirm_tool_calls: bool = False
        self._prompt_fn = None  # stored on host so _dispatch_tool can reach it

    def set_interaction_fns(self, notify_fn, prompt_fn) -> None:
        """Set (or clear) the notify and prompt callbacks on the active provider."""
        self._prompt_fn = prompt_fn
        if self._provider is not None:
            if hasattr(self._provider, "notify_fn"):
                self._provider.notify_fn = notify_fn
            if hasattr(self._provider, "prompt_fn"):
                self._provider.prompt_fn = prompt_fn

    def set_confirm(self, enabled: bool) -> None:
        """Enable or disable the tool-call confirmation gate."""
        self._confirm_tool_calls = enabled

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def _load(self, config_path: Path) -> None:
        """Read mcp_servers_config.json, instantiate the LLM provider, and connect all MCP clients."""
        with config_path.open() as f:
            config: dict = json.load(f)

        self._provider = build_provider(config.get("agent", {}))
        self._confirm_tool_calls = config.get("agent", {}).get("confirm_tool_calls", False)

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
        self._custom_commands = self._load_custom_commands()

    def _load_custom_commands(self) -> dict:
        """Load custom commands from custom_commands.json if it exists."""
        if not CUSTOM_COMMANDS_FILE.exists():
            return {}
        with CUSTOM_COMMANDS_FILE.open() as f:
            return json.load(f)

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

    async def _confirm_tool_call(
        self,
        call: ToolCall,
        emit: Callable[[dict], Coroutine] | None,
    ) -> str | None:
        """Ask the user to confirm a tool call. Returns None to proceed, or a cancellation string."""
        args_str = json.dumps(call.arguments, indent=2, ensure_ascii=False)
        prompt = (
            f"Confirm call to '{call.name}' with:\n{args_str}\n"
            "Proceed? [Y/n]: "
        )
        answer = await self._prompt_fn(prompt) if self._prompt_fn else await self._read_input(prompt)
        if answer.strip().lower() in ("n", "no"):
            cancelled = "Tool call cancelled by user."
            if emit:
                await emit({"type": EventType.TOOL_RESULT, "name": call.name, "result": cancelled})
            else:
                print(f"  ← {cancelled}")
            return cancelled
        return None

    async def _dispatch_tool(
        self,
        call: ToolCall,
        emit: Callable[[dict], Coroutine] | None = None,
    ) -> str:
        """Route a tool call to the correct MCPClient and return the result as a string."""
        if call.name not in self._tool_index:
            return f"Error: unknown tool '{call.name}'"
        client, original_name = self._tool_index[call.name]
        if emit:
            await emit({"type": EventType.TOOL_CALL, "name": call.name, "args": call.arguments})
        else:
            print(f"  → {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")

        if self._confirm_tool_calls:
            cancelled = await self._confirm_tool_call(call, emit)
            if cancelled is not None:
                return cancelled

        try:
            result = str(await client.call_tool(original_name, call.arguments))
            if emit:
                await emit({"type": EventType.TOOL_RESULT, "name": call.name, "result": result})
            else:
                print(f"  ← {result[:200]}")
            return result
        except Exception as exc:
            error = f"Error: {exc}"
            if emit:
                await emit({"type": EventType.TOOL_RESULT, "name": call.name, "result": error})
            else:
                print(f"  ← {error}")
            return error

    async def _turn(
        self,
        user_input: str,
        emit: Callable[[dict], Coroutine] | None = None,
    ) -> None:
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
                    result = await self._dispatch_tool(tc, emit=emit)
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
                        if emit:
                            await emit({"type": EventType.ERROR, "message": f"Stopped after {consecutive_errors} consecutive tool errors: {last_error}"})
                        else:
                            print(f"\n[Stopped] Tool kept failing after {consecutive_errors} attempts: {last_error}\n")
                        break
                else:
                    consecutive_errors = 0
            else:
                # Final answer
                self._history.append({"role": "assistant", "content": response.content or ""})
                if emit:
                    await emit({"type": EventType.ASSISTANT, "content": response.content or ""})
                else:
                    print(f"\nAssistant: {response.content}\n")
                break
        else:
            if emit:
                await emit({"type": EventType.ERROR, "message": f"Reached {MAX_TOOL_ITERATIONS} tool iterations without a final answer."})
            else:
                print(f"\n[Warning] Reached {MAX_TOOL_ITERATIONS} tool iterations without a final answer.\n")

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def _format_history(self) -> str:
        if not self._history:
            return "No history yet."
        lines = []
        for msg in self._history:
            role = msg["role"].capitalize()
            if msg["role"] == "tool":
                lines.append(f"  [Tool result id={msg.get('tool_call_id', '')}] {msg['content'][:200]}")
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                calls = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
                lines.append(f"  {role}: <tool calls: {calls}>")
            else:
                lines.append(f"  {role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _format_tools(self) -> str:
        if not self._tool_index:
            return "No tools available."
        lines = ["Available tools:"]
        for namespaced in self._tool_index:
            schema = next(
                (s["function"] for s in self._tools_schema if s["function"]["name"] == namespaced), {}
            )
            desc = schema.get("description", "")
            lines.append(f"  {namespaced:<40} {desc}")
        return "\n".join(lines)

    def _format_custom_commands(self) -> str:
        if not self._custom_commands:
            return "No custom commands loaded. Add them to custom_commands.json."
        lines = ["Custom commands (from custom_commands.json):"]
        for name, meta in self._custom_commands.items():
            lines.append(f"  /{name:<20} — {meta.get('description', '(no description)')}")
        return "\n".join(lines)

    def _format_help(self) -> str:
        return _BUILTIN_HELP + "\n" + self._format_custom_commands()

    async def _handle_confirm_command(
        self,
        content: str,
        emit: Callable[[dict], Coroutine] | None,
    ) -> None:
        """Toggle or set the tool-call confirmation gate and notify the user."""
        async def send_system(msg: str) -> None:
            if emit:
                await emit({"type": EventType.SYSTEM, "message": msg})
                await emit({"type": EventType.SET_CONFIRM, "value": self._confirm_tool_calls})
            else:
                print(msg)

        parts = content.split()
        if len(parts) > 1 and parts[1] == "off":
            self._confirm_tool_calls = False
        elif len(parts) > 1 and parts[1] == "on":
            self._confirm_tool_calls = True
        else:
            self._confirm_tool_calls = not self._confirm_tool_calls

        state = "enabled" if self._confirm_tool_calls else "disabled"
        await send_system(f"Tool-call confirmation {state}.")

    async def handle_input(
        self,
        content: str,
        emit: Callable[[dict], Coroutine] | None = None,
    ) -> None:
        """Dispatch user input: handle built-in/custom commands or run an agent turn."""

        async def send_text(text: str) -> None:
            if emit:
                await emit({"type": EventType.ASSISTANT, "content": text})
            else:
                print(text)

        async def send_system(msg: str) -> None:
            if emit:
                await emit({"type": EventType.SYSTEM, "message": msg})
            else:
                print(msg)

        async def send_error(msg: str) -> None:
            if emit:
                await emit({"type": EventType.ERROR, "message": msg})
            else:
                print(msg)

        if content == "/history":
            await send_text(self._format_history())
            return
        if content == "/reset":
            self._history.clear()
            await send_system("History cleared.")
            return
        if content == "/tools":
            await send_text(self._format_tools())
            return
        if content in ("/help", "help"):
            await send_text(self._format_help())
            return
        if content == "/commands":
            await send_text(self._format_custom_commands())
            return
        if content == "/confirm" or content.startswith("/confirm "):
            await self._handle_confirm_command(content, emit)
            return
        if content.startswith("/"):
            cmd_name = content[1:]
            if cmd_name in self._custom_commands:
                await self._turn(self._custom_commands[cmd_name]["prompt"], emit=emit)
            else:
                await send_error(f"Unknown command: {content}. Type /help for available commands.")
            return

        await self._turn(content, emit=emit)

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
                await self.handle_input(line)
        finally:
            await self._exit_stack.aclose()


    async def run_web(self, config_path: Path = CONFIG_FILE, port: int = 8000) -> None:
        """Load config, connect servers, then start the web UI on the given port."""
        import uvicorn
        from server import make_app

        print(f"Loading config from {config_path} ...")
        await self._load(config_path)
        print(f"\nReady. {len(self._tool_index)} tool(s) loaded.")
        print(f"Web UI starting at http://localhost:{port}\n")

        app = make_app(self)
        cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(cfg)
        try:
            await server.serve()
        finally:
            await self._exit_stack.aclose()


async def main() -> None:
    """Parse arguments and run the MCPHost in terminal or web mode."""
    parser = argparse.ArgumentParser(description="MCP Agent REPL")
    parser.add_argument("config", nargs="?", type=Path, default=CONFIG_FILE, help="Path to config JSON")
    parser.add_argument("--web", action="store_true", help="Start web UI instead of terminal REPL")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web UI (default: 8000)")
    args = parser.parse_args()

    host = MCPHost()
    if args.web:
        await host.run_web(args.config, port=args.port)
    else:
        await host.run(args.config)


if __name__ == "__main__":
    asyncio.run(main())
