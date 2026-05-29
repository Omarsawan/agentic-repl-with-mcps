from enum import StrEnum


class EventType(StrEnum):
    AGENT_NOTIFICATION = "agent_notification"
    """Server → client. One-way notification from the agent (e.g. announcing which tool it
    will call and what parameters it expects). No user response is required."""

    INPUT_PROMPT = "input_prompt"
    """Server → client. The agent needs a value from the user. Renders an inline form
    in the UI. Expects an INPUT_RESPONSE to be sent back before the agent can continue."""

    INPUT_RESPONSE = "input_response"
    """Client → server. The user's answer to an INPUT_PROMPT. Contains the value the
    agent was waiting for; unblocks the suspended prompt_fn coroutine."""

    TOOL_CALL = "tool_call"
    """Server → client. The agent is invoking an MCP tool. Carries the tool name and
    arguments so the UI can display an in-progress tool card."""

    TOOL_RESULT = "tool_result"
    """Server → client. The MCP tool has returned. Carries the tool name and result
    string; the UI updates the matching tool card to show the outcome."""

    ASSISTANT = "assistant"
    """Server → client. A final text response from the LLM. Rendered as a markdown
    assistant message and re-enables the main chat input."""

    ERROR = "error"
    """Server → client. An unrecoverable error occurred (tool failure, LLM error, etc.).
    Displayed as an error banner and re-enables the chat input."""

    SYSTEM = "system"
    """Server → client. An informational message from the host (e.g. 'History cleared').
    Currently not rendered in the UI but reserved for status feedback."""
