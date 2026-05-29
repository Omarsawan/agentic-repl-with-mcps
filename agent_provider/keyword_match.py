import asyncio
import sys
import uuid
from typing import Callable, Coroutine

from .base import AgentProvider, ChatResponse, ToolCall


class KeywordMatchProvider(AgentProvider):
    """No-LLM provider: matches the user's input against tool names and descriptions.

    Scores each tool by counting how many query tokens appear in its name + description.
    Returns the highest-scoring tool as a ToolCall with arguments collected interactively,
    or a text response if nothing matches. Useful for testing the MCP plumbing without an LLM.
    """

    def __init__(self) -> None:
        # Set by the host when running in web mode; each value is called with a
        # prompt string and must return the user's input as a string.
        self.prompt_fn: Callable[[str], Coroutine] | None = None

    async def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        if messages and messages[-1].get("role") == "tool":
            result = messages[-1].get("content", "")
            return ChatResponse(content=f"Result: {result}")

        query = self._last_user_message(messages)
        if not query or not tools:
            return ChatResponse(content="No matching tool found for your input.")

        tokens = query.lower().split()
        best_tool: dict | None = None
        best_score = 0

        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            haystack = f"{name} {desc}".lower()
            score = sum(1 for t in tokens if t in haystack)
            if score > best_score:
                best_score = score
                best_tool = fn

        if best_tool is None:
            return ChatResponse(content="No matching tool found for your input.")

        return ChatResponse(
            content=None,
            tool_calls=[ToolCall(
                id=str(uuid.uuid4()),
                name=best_tool["name"],
                arguments=await self._extract_arguments(best_tool, query),
            )],
        )

    @staticmethod
    def _last_user_message(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    async def _extract_arguments(self, tool_fn: dict, query: str) -> dict:
        params = tool_fn.get("parameters", {})
        required: list[str] = params.get("required", [])
        properties: dict = params.get("properties", {})

        if not properties:
            return {}

        interactive = self.prompt_fn is not None or sys.stdin.isatty()
        if not interactive:
            # Non-interactive fallback: use query for a single required string param
            if len(required) == 1:
                param_name = required[0]
                if properties.get(param_name, {}).get("type") == "string":
                    return {param_name: query}
            return {}

        tool_name = tool_fn.get("name", "unknown")
        announcement = self._build_announcement(tool_name, required, properties)

        if self.prompt_fn is not None:
            await self.prompt_fn(announcement, is_announcement=True)
        else:
            print(announcement)

        optional = [p for p in properties if p not in required]
        arguments: dict = {}

        for param in required:
            prop = properties.get(param, {})
            ptype = prop.get("type", "string")
            prompt = f"Enter '{param}' ({ptype}): "
            arguments[param] = await self._ask(prompt)

        for param in optional:
            prop = properties.get(param, {})
            ptype = prop.get("type", "string")
            prompt = f"Enter '{param}' ({ptype}, optional — press Enter to skip): "
            value = await self._ask(prompt)
            if value:
                arguments[param] = value

        return arguments

    async def _ask(self, prompt: str) -> str:
        if self.prompt_fn is not None:
            return await self.prompt_fn(prompt, is_announcement=False)
        if sys.stdin.isatty():
            return await asyncio.to_thread(input, prompt)
        # Non-interactive web context with no prompt_fn — shouldn't reach here,
        # but return empty string to avoid blocking on stdin.
        return ""

    @staticmethod
    def _build_announcement(tool_name: str, required: list[str], properties: dict) -> str:
        lines = [f"Agent will use tool: **{tool_name}**", "This tool expects the following parameters:"]
        for name, prop in properties.items():
            ptype = prop.get("type", "any")
            desc = prop.get("description", "")
            req_tag = "" if name in required else " (optional)"
            line = f"  • {name} ({ptype}{req_tag})"
            if desc:
                line += f" — {desc}"
            lines.append(line)
        return "\n".join(lines)
