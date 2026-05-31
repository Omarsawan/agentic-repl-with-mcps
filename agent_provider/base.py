from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """The LLM's response: optional text content plus zero or more tool calls."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class AgentProvider(ABC):
    """Abstract base class for LLM backends.

    Subclass this to add a custom provider. The only method to implement is chat().
    Instances are created by build_provider() based on the mcp_servers_config.json config;
    any extra config keys are forwarded as constructor keyword arguments.
    """

    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        """Send conversation history and available tools; return the model's response.

        ``messages`` is a list of dicts in OpenAI chat format:

            # user turn
            {"role": "user", "content": "<text>"}

            # assistant turn (with tool calls)
            {
                "role": "assistant",
                "content": "<text or None>",
                "tool_calls": [
                    {
                        "id": "<call-id>",
                        "type": "function",
                        "function": {"name": "<tool-name>", "arguments": "<json-string>"},
                    }
                ],
            }

            # tool result turn
            {"role": "tool", "tool_call_id": "<call-id>", "content": "<result-text>"}

            # assistant turn (no tool calls)
            {"role": "assistant", "content": "<text>"}

        ``tools`` is a list of dicts in OpenAI function-calling format:

            {
                "type": "function",
                "function": {
                    "name": "<server>__<tool>",   # e.g. "mysql__execute_query"
                    "description": "<text>",
                    "parameters": {               # JSON Schema (mirrors MCP inputSchema)
                        "type": "object",
                        "properties": {...},
                        "required": [...],
                    },
                },
            }
        """
        ...
