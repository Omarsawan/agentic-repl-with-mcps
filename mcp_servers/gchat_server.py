import json
import os

import google.auth.transport.requests
import requests
from dotenv import load_dotenv
from google.oauth2 import service_account
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("gchat")

_CHAT_API = "https://chat.googleapis.com/v1"
_SCOPES = [
    "https://www.googleapis.com/auth/chat.messages",
    "https://www.googleapis.com/auth/chat.spaces.readonly",
]

_creds = None


def _get_token() -> str | None:
    global _creds
    sa_path = os.environ.get("GCHAT_SERVICE_ACCOUNT_JSON")
    if not sa_path:
        return None
    if _creds is None:
        _creds = service_account.Credentials.from_service_account_file(sa_path, scopes=_SCOPES)
    if not _creds.valid:
        _creds.refresh(google.auth.transport.requests.Request())
    return _creds.token


def _require_token() -> tuple[str | None, str | None]:
    token = _get_token()
    if token is None:
        return None, (
            "Error: GCHAT_SERVICE_ACCOUNT_JSON is not configured. "
            "See the 'Google Chat setup' section in README.md."
        )
    return token, None


def _api_post(path: str, body: dict) -> tuple[dict | None, str | None]:
    token, err = _require_token()
    if err:
        return None, err
    resp = requests.post(
        f"{_CHAT_API}/{path}",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=15,
    )
    if not resp.ok:
        return None, f"Chat API error {resp.status_code}: {resp.text}"
    return resp.json(), None


def _api_get(path: str, params: dict | None = None) -> tuple[dict | None, str | None]:
    token, err = _require_token()
    if err:
        return None, err
    resp = requests.get(
        f"{_CHAT_API}/{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=15,
    )
    if not resp.ok:
        return None, f"Chat API error {resp.status_code}: {resp.text}"
    return resp.json(), None


def _get_thread_messages(space: str, thread_name: str) -> tuple[list | None, str | None]:
    data, err = _api_get(
        f"{space}/messages",
        params={"filter": f'thread.name = "{thread_name}"', "pageSize": 100},
    )
    if err:
        return None, err
    messages = []
    for msg in data.get("messages", []):
        ts = msg.get("createTime", "")[:19].replace("T", " ")
        messages.append(
            {
                "sender": msg.get("sender", {}).get("displayName", "Unknown"),
                "text": msg.get("text", ""),
                "timestamp": ts,
            }
        )
    return messages, None


def _format_thread(thread_name: str, messages: list) -> str:
    lines = [f"Thread: {thread_name}", f"Messages: {len(messages)}", ""]
    for msg in messages:
        lines.append(f"[{msg['timestamp']}] {msg['sender']}: {msg['text']}")
    return "\n".join(lines)


@mcp.tool()
def send_message(space_or_webhook: str, text: str) -> str:
    """Send a message to a Google Chat space or webhook.

    Pass 'default' to use the GCHAT_WEBHOOK_URL env var (no Service Account needed).
    Pass a full webhook URL (https://...) to send to a specific webhook.
    Pass a space name like 'spaces/SPACE_ID' to send via the REST API (requires Service Account).
    """
    if space_or_webhook == "default" or space_or_webhook.startswith("https://"):
        url = (
            os.environ.get("GCHAT_WEBHOOK_URL")
            if space_or_webhook == "default"
            else space_or_webhook
        )
        if not url:
            return "Error: GCHAT_WEBHOOK_URL is not set. Add it to your .env file."
        resp = requests.post(url, json={"text": text}, timeout=15)
        if not resp.ok:
            return f"Webhook error {resp.status_code}: {resp.text}"
        return f"Message sent. Name: {resp.json().get('name', 'unknown')}"

    data, err = _api_post(f"{space_or_webhook}/messages", {"text": text})
    if err:
        return err
    return f"Message sent. Name: {data.get('name', 'unknown')}"


@mcp.tool()
def send_thread_reply(space: str, thread_name: str, text: str) -> str:
    """Reply to a thread in a Google Chat space (requires Service Account auth).

    space: 'spaces/SPACE_ID'
    thread_name: 'spaces/SPACE_ID/threads/THREAD_ID'
    """
    body = {
        "text": text,
        "thread": {"name": thread_name},
        "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
    }
    data, err = _api_post(f"{space}/messages", body)
    if err:
        return err
    sent_thread = data.get("thread", {}).get("name", "unknown")
    return f"Reply sent. Message: {data.get('name', 'unknown')}, Thread: {sent_thread}"


@mcp.tool()
def list_spaces() -> str:
    """List Google Chat spaces the service account bot belongs to (requires Service Account auth)."""
    data, err = _api_get("spaces", params={"pageSize": 100})
    if err:
        return err
    spaces = data.get("spaces", [])
    if not spaces:
        return "No spaces found. Make sure the bot has been added to at least one space."
    lines = [f"Found {len(spaces)} space(s):\n"]
    for s in spaces:
        lines.append(
            f"  {s.get('name', '')}  |  {s.get('displayName', '(no name)')}  |  {s.get('spaceType', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
def summarize_thread(space: str, thread_name: str) -> str:
    """Fetch all messages in a thread so the LLM can summarize them (requires Service Account auth).

    space: 'spaces/SPACE_ID'
    thread_name: 'spaces/SPACE_ID/threads/THREAD_ID'
    """
    messages, err = _get_thread_messages(space, thread_name)
    if err:
        return err
    if not messages:
        return "No messages found in this thread."
    return _format_thread(thread_name, messages) + "\n\nPlease summarize this thread."


@mcp.tool()
def suggest_reply(space: str, thread_name: str) -> str:
    """Fetch all messages in a thread so the LLM can suggest an appropriate reply (requires Service Account auth).

    space: 'spaces/SPACE_ID'
    thread_name: 'spaces/SPACE_ID/threads/THREAD_ID'
    """
    messages, err = _get_thread_messages(space, thread_name)
    if err:
        return err
    if not messages:
        return "No messages found in this thread."
    return _format_thread(thread_name, messages) + "\n\nPlease suggest an appropriate reply to this thread."


if __name__ == "__main__":
    mcp.run(transport="stdio")
