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
        """Send conversation history and tools schema; return the model's response."""
        ...
