"""Example MCP server exposing three toy tools: echo, get_time, and add."""

from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sample")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the provided text back."""
    return text


@mcp.tool()
def get_time() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    mcp.run(transport="stdio")
