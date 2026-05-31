# agentic-repl-with-mcps

A lightweight Python framework that connects one or more [MCP](https://modelcontextprotocol.io) servers to an LLM, running an agentic tool-use loop inside an interactive REPL (Read, Eval, Print, Loop). LLM backends are swappable via a pluggable provider interface; the built-in provider works with any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, and similar). The LLM decides which tools to call, the framework dispatches those calls to the right server, and the results are fed back until the model produces a final answer.

## Architecture

```
User input (REPL)
      │
      ▼
MCPHost._turn()          ← drives the agentic loop
      │
      ▼
AgentProvider.chat()     ← sends history + tools schema to the LLM
      │
      ├─ tool calls? ──► MCPHost._dispatch_tool()
      │                        │
      │                        ▼
      │                  MCPClient.call_tool()   ← talks to the MCP server
      │                        │
      │                  result appended to history, loop continues
      │
      └─ final answer ──► printed to the user
```

Configuration is read from `mcp_servers_config.json` at startup; no code changes are needed to add a new server or switch the LLM backend.

### Web UI architecture

When `--web` is passed, `MCPHost.run_web()` creates a FastAPI app via `server.make_app()` and hands it to uvicorn, which drives the asyncio event loop directly — no separate thread is needed.

```
Browser
   │  HTTP GET /          → serves static/index.html
   │  POST /reset         → clears conversation history
   │  WebSocket /ws
   │       │
   │       │  { "content": "..." }          (client → server)
   │       │
   │       ▼
   │  server.ws_endpoint()
   │       │
   │       ▼
   │  MCPHost._turn(content, emit=ws.send_json)
   │       │                                 │
   │       ▼                                 │
   │  (same agentic loop as REPL)            │
   │       │                                 │
   │       └─ tool call / final answer ──────►  { "type": "...", ... }  (server → client)
   │
uvicorn (asyncio event loop)
```

Each browser message triggers one full agentic turn. The `emit` callback streams JSON events back over the same WebSocket connection as the turn progresses — tool call arguments, tool results, and the final assistant message are all sent incrementally as separate events. The WebSocket stays open between turns so the conversation history is maintained on the server.

**Key files:**

| File | Role |
|------|------|
| [host.py:303](host.py#L303) `MCPHost.run_web()` | Builds the FastAPI app, configures uvicorn, and starts the server |
| [server.py](server.py) `make_app()` | Defines the three routes (`GET /`, `POST /reset`, `WS /ws`) and wires them to the host |
| [static/index.html](static/index.html) | Single-page chat UI served at `/` |

## Quick start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create the virtual environment
uv sync

# Copy and fill in environment variables
cp .env.example .env

# Pull and serve a model with Ollama
ollama pull qwen2.5:7b
ollama serve          # starts the API at http://localhost:11434
```

If you prefer plain pip (Python 3.12+ required):

```bash
pip install -e .
```

> Provider contract (messages/tools format, ChatResponse, edge cases): see [agent_provider/base.py](agent_provider/base.py)

## Running

### Terminal REPL

```bash
uv run python host.py

# Optional: pass a custom config file
uv run python host.py path/to/my_config.json
```

### Web UI

```bash
uv run python host.py --web

# Custom port
uv run python host.py --web --port 8080

# Custom config + custom port
uv run python host.py --web --port 8080 path/to/my_config.json
```

Then open `http://localhost:8000` (or whichever port you chose) in your browser. The web UI shows chat bubbles for user and assistant messages, collapsible tool call blocks with arguments and results, and a **Reset** button to clear the conversation history.

### Switching LLM backends

Edit the `agent` block in `mcp_servers_config.json` — no code changes required.

**Ollama** (default, local):
```json
"agent": { "provider": "openai_compatible", "base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b" }
```

**OpenRouter** (cloud, many models):
```json
"agent": { "provider": "openai_compatible", "base_url": "https://openrouter.ai/api/v1", "model": "mistralai/mistral-7b-instruct", "api_key_env": "OPENROUTER_API_KEY" }
```

**vLLM** (self-hosted, OpenAI-compatible):
```json
"agent": { "provider": "openai_compatible", "base_url": "http://localhost:8000/v1", "model": "mistralai/Mistral-7B-Instruct-v0.2", "api_key_env": "VLLM_API_KEY" }
```

**LM Studio** (local GUI):
```json
"agent": { "provider": "openai_compatible", "base_url": "http://localhost:1234/v1", "model": "local-model" }
```

**Keyword Match** (no LLM, for testing):
```json
"agent": { "provider": "keyword_match" }
```
Selects a tool by matching tokens from your query against tool names and descriptions — no API key or running model needed. Useful for testing tool wiring without an LLM. If no tool matches, it returns a plain text fallback message.

**Generic routing** (SQL + extensible handlers):
```json
"agent": { "provider": "generic", "checkpoint_path": "sql_model/checkpoints/best_model.pt", "intent_confidence_threshold": 0.75 }
```
Classifies each prompt and routes it to the appropriate handler. SQL queries are detected automatically, translated to SQL via intent classification + template engine (with optional neural fallback), executed against MySQL, and the results are returned. Non-SQL prompts fall back to `keyword_match`. Add new handlers by implementing `PromptHandler` in `agent_provider/generic_routing.py`.

**Tool-call confirmation** (works with any provider):
Add `"confirm_tool_calls": true` to the `agent` block to make the agent pause and ask for confirmation before every MCP tool call. Toggle at runtime with `/confirm` in the terminal REPL or the **Confirm: ON/OFF** button in the web UI.

Available REPL commands:

| Command               | Effect                                      |
|-----------------------|---------------------------------------------|
| `/tools`              | List all available tools                    |
| `/history`            | Print conversation history                  |
| `/reset`              | Clear conversation history                  |
| `/confirm [on\|off]`  | Toggle (or set) tool-call confirmation gate |
| `/quit`               | Exit                                        |

## Temporal Cloud setup

> See [mcp_servers/temporal_server.py](mcp_servers/temporal_server.py) for env vars, WQL reference, and usage examples.

## MySQL setup

> See [mcp_servers/mysql_server.py](mcp_servers/mysql_server.py) for env vars and tool reference.

## Google Chat setup

> See [mcp_servers/gchat_server.py](mcp_servers/gchat_server.py) for webhook and Service Account setup.

## Adding a new MCP server

1. Write a FastMCP server (use [mcp_servers/sample_server.py](mcp_servers/sample_server.py) as a template).
2. Add an entry to `mcp_servers_config.json`:

```json
"mcpServers": {
  "myserver": {
    "command": "python",
    "args": ["my_server.py"],
    "env": {}
  }
}
```

> **Do not put real secrets in `mcp_servers_config.json`** — this file is committed to version control. Set secrets as environment variables (e.g. in `.env`) and read them in your server with `os.environ`.

No other code changes are needed. Tools are automatically namespaced as `myserver__toolname`.
