import uuid

from .base import ChatResponse, LLMProvider, ToolCall


class KeywordMatchProvider(LLMProvider):
    """No-LLM provider: matches the user's input against tool names and descriptions.

    Scores each tool by counting how many query tokens appear in its name + description.
    Returns the highest-scoring tool as a ToolCall (with empty arguments), or a text
    response if nothing matches. Useful for testing the MCP plumbing without an LLM.
    """

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
                arguments=self._extract_arguments(best_tool, query),
            )],
        )

    @staticmethod
    def _last_user_message(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    @staticmethod
    def _extract_arguments(tool_fn: dict, query: str) -> dict:
        params = tool_fn.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})
        if len(required) == 1:
            param_name = required[0]
            if properties.get(param_name, {}).get("type") == "string":
                return {param_name: query}

        return {}
