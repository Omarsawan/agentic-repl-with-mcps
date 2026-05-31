import logging
from abc import abstractmethod

from .base import AgentProvider, ChatResponse
from .utils import last_user_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler protocol
# ---------------------------------------------------------------------------


class PromptHandler(AgentProvider):
    """Base class for prompt handlers registered with GenericRoutingProvider.

    Implement can_handle() to claim a message; implement chat() to respond to it.
    """

    @abstractmethod
    def can_handle(self, user_text: str) -> bool:
        """Return True if this handler should process the given user message."""

    def reset(self) -> None:
        """Called when a new user query is detected. Override to clear per-turn state."""


# ---------------------------------------------------------------------------
# Generic routing provider
# ---------------------------------------------------------------------------


class GenericRoutingProvider(AgentProvider):
    """Routes each user prompt to the first matching handler, or falls back to fallback_provider.

    To add a new prompt type, implement PromptHandler and pass an instance in the handlers list.
    The router checks handlers in order and delegates to the first whose can_handle() returns True.
    """

    def __init__(
        self,
        handlers: list[PromptHandler],
        fallback_provider: AgentProvider,
    ) -> None:
        self.handlers = handlers
        self.fallback_provider = fallback_provider

    async def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        user_text = last_user_message(messages)
        for handler in self.handlers:
            if handler.can_handle(user_text):
                return await handler.chat(messages, tools)
        return await self.fallback_provider.chat(messages, tools)


