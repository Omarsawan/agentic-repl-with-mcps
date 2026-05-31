"""Agent provider abstraction for the agentic REPL framework.

AgentProvider is the single extension point for adding LLM backends. Subclass it,
implement chat(), and register it via build_provider() in loader.py.

Files in this package
---------------------
base.py              AgentProvider ABC, ToolCall and ChatResponse dataclasses (this file)
openai_compatible.py Works with any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenRouter, etc.)
keyword_match.py     No-LLM test provider: matches user input against tool names without calling an LLM
generic_routing.py   Classifies each prompt and delegates to the matching PromptHandler (e.g. SQLHandler).
                     Add new handlers here without touching the router.
loader.py            build_provider() factory; supports dynamic loading of custom provider classes

Provider contract
-----------------
chat(messages, tools) follows the OpenAI chat completions format for both arguments.

messages shapes:
  {"role": "user", "content": "..."}
  {"role": "assistant", "content": null, "tool_calls": [{"id": "...", "type": "function",
      "function": {"name": "server__tool", "arguments": "{...}"}}]}
  {"role": "tool", "tool_call_id": "...", "content": "..."}
  {"role": "assistant", "content": "..."}

Invariant: content is always null when tool_calls is non-empty; when tool_calls is empty,
content must be a non-null string (the final answer).

tools entries follow OpenAI function-calling format; names are namespaced as server__toolname.
Tools with no input schema receive {"type": "object", "properties": {}} as their parameters value.

Edge cases enforced by MCPHost (not the provider):
- content: None in a final answer is converted to "". Return a non-empty string when possible.
- Three consecutive turns where every tool result starts with "Error:" abort the turn.
- chat() is called at most MAX_TOOL_ITERATIONS = 20 times per user turn.
"""
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
