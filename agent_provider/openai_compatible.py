import json
from typing import Any

from openai import AsyncOpenAI

from .base import AgentProvider, ChatResponse, ToolCall


class OpenAICompatibleProvider(AgentProvider):
    """Works with any OpenAI-compatible endpoint: Ollama, vLLM, LM Studio, OpenRouter, etc."""

    def __init__(self, base_url: str, model: str, api_key: str = "ollama") -> None:
        self._model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return ChatResponse(content=message.content, tool_calls=tool_calls)
