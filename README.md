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

## Files

### [mcp_client.py](mcp_client.py)

Manages a single MCP server connection over stdio. `MCPClient` spawns the server subprocess, completes the MCP handshake, and exposes two operations: `list_tools()` to discover what the server offers, and `call_tool()` to invoke a tool and get its text result back. The class is an async context manager — resources are cleaned up automatically on exit.

### [host.py](host.py)

The central coordinator. `MCPHost` reads `mcp_servers_config.json`, instantiates the LLM provider and one `MCPClient` per configured server, and builds a namespaced tool registry (`servername__toolname`) so tool names stay unique across servers. `_turn()` implements the core agentic loop, and `run()` wraps it in an interactive REPL that also handles `/tools`, `/history`, `/reset`, and `/quit` commands.

### [agent_provider/](agent_provider/)

Agent provider abstraction layer.

| File | Purpose |
|------|---------|
| `base.py` | `AgentProvider` ABC, `ToolCall` and `ChatResponse` dataclasses |
| `openai_compatible.py` | Works with any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenRouter, etc.) |
| `keyword_match.py` | No-LLM test provider: matches user input against tool names without calling an LLM |
| `loader.py` | `build_provider()` factory; supports dynamic loading of custom provider classes |

### [mcp_servers/sample_server.py](mcp_servers/sample_server.py)

A minimal example MCP server built with FastMCP. Exposes three toy tools — `echo`, `get_time`, and `add` — to demonstrate how to build tools that `MCPHost` can discover and call.

### [mcp_servers/temporal_server.py](mcp_servers/temporal_server.py)

MCP server for querying **Temporal Cloud** workflows. Connects via mTLS and exposes two tools:

| Tool | Description |
|------|-------------|
| `list_workflows` | Search workflows using [Temporal's Workflow Query Language (WQL)](#temporal-wql-reference). Returns ID, run ID, type, status, start/close times, and task queue for each match. |
| `describe_workflow` | Get full details for a specific workflow. Accepts a `workflow_id`, a `run_id`, or both. If only a `run_id` is given, the workflow ID is resolved automatically. |

### [mcp_servers/gchat_server.py](mcp_servers/gchat_server.py)

MCP server for sending and reading messages in **Google Chat**. Supports two auth modes simultaneously — configure just a webhook URL for simple sending, or add a Service Account key to unlock all tools.

| Tool | Auth required | Description |
|------|---------------|-------------|
| `send_message` | Webhook URL or Service Account | Send a message to a space or webhook. Pass `"default"` to use `GCHAT_WEBHOOK_URL`. |
| `send_thread_reply` | Service Account | Reply to a specific message thread. |
| `list_spaces` | Service Account | List spaces the bot belongs to. |
| `summarize_thread` | Service Account | Fetch thread messages so the LLM can summarize them. |
| `suggest_reply` | Service Account | Fetch thread messages so the LLM can suggest an appropriate reply. |

### [mcp_servers/mysql_server.py](mcp_servers/mysql_server.py)

MCP server for read-only MySQL access over an SSH tunnel. Automatically opens an SSH port-forward on startup and exposes three tools:

| Tool | Description |
|------|-------------|
| `execute_query` | Run a read-only SQL statement (`SELECT`, `SHOW`, `EXPLAIN`, `DESCRIBE`, `WITH`) and return results as JSON. |
| `list_tables` | List all tables in a database. |
| `describe_table` | Return column definitions (name, type, nullable, key, default, extra) for a table. |

### [mcp_servers_config.json](mcp_servers_config.json)

Runtime configuration file. The `agent` section sets the provider type, endpoint URL, model name, and the environment variable that holds the API key. The `mcpServers` section maps server names to their launch commands. Add a new entry here to connect any additional MCP-compatible server without touching Python code.

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

Available REPL commands:

| Command    | Effect                          |
|------------|---------------------------------|
| `/tools`   | List all available tools        |
| `/history` | Print conversation history      |
| `/reset`   | Clear conversation history      |
| `/quit`    | Exit                            |

## Temporal Cloud setup

Set the following environment variables before running (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `TEMPORAL_ADDRESS` | Cloud endpoint, e.g. `myns.abc123.tmprl.cloud:7233` |
| `TEMPORAL_NAMESPACE` | Namespace, e.g. `myns.abc123` |
| `TEMPORAL_TLS_CERT` | Path to your mTLS client certificate (PEM) |
| `TEMPORAL_TLS_KEY` | Path to your mTLS private key (PEM) |

Once set, the `temporal` server connects automatically on startup:

```
[temporal] connected — 2 tool(s): ['list_workflows', 'describe_workflow']
```

You can then query workflows in plain language — the LLM translates your intent into WQL:

```
> show me all running OrderWorkflow executions
> describe workflow id my-order-12345
> list failed workflows from last week
> which workflow has run id abc-def-456
```

## Temporal WQL reference

The `list_workflows` tool accepts any valid [Workflow Query Language](https://docs.temporal.io/visibility) expression via its `query` parameter.

```
# By type
WorkflowType = 'OrderWorkflow'

# By status
ExecutionStatus = 'Running'
ExecutionStatus = 'Failed'

# By ID
WorkflowId = 'my-workflow-123'
RunId = 'abc-def-456'

# By time range
StartTime BETWEEN '2024-01-01T00:00:00Z' AND '2024-12-31T23:59:59Z'
CloseTime > '2024-06-01T00:00:00Z'

# Compound
WorkflowType = 'OrderWorkflow' AND ExecutionStatus = 'Completed'
ExecutionStatus = 'Running' AND TaskQueue = 'my-queue'
```

Valid `ExecutionStatus` values: `Running`, `Completed`, `Failed`, `Canceled`, `Terminated`, `ContinuedAsNew`, `TimedOut`.

## MySQL setup

Set the following environment variables before running (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `MYSQL_SSH_HOST` | Bastion / jump host hostname |
| `MYSQL_SSH_PORT` | SSH port (default `22`) |
| `MYSQL_SSH_USER` | SSH username |
| `MYSQL_SSH_KEY_PATH` | Path to your SSH private key (e.g. `~/.ssh/id_rsa`) |
| `MYSQL_HOST` | MySQL host as seen from the SSH server |
| `MYSQL_PORT` | MySQL port as seen from the SSH server |
| `MYSQL_USER` | MySQL username |
| `MYSQL_PASSWORD` | MySQL password |
| `MYSQL_DATABASE` | Default database (optional) |

The server opens an SSH tunnel automatically on startup and only allows read-only queries (`SELECT`, `SHOW`, `EXPLAIN`, `DESCRIBE`, `WITH`).

## Google Chat setup

### Webhook-only (quick start — enables `send_message` only)

1. Open a Google Chat space → **Apps & integrations** → **Webhooks** → **Add webhook**.
2. Copy the generated webhook URL.
3. Add it to your `.env`:

```
GCHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=...
```

Then call `gchat__send_message` with `space_or_webhook="default"`.

### Service Account (enables all tools)

To use `send_thread_reply`, `list_spaces`, `summarize_thread`, and `suggest_reply`:

1. In [Google Cloud Console](https://console.cloud.google.com), create or select a project and enable the **Google Chat API**.
2. Go to **IAM & Admin → Service Accounts** → create a service account → download the JSON key.
3. In the [Google Chat API configuration](https://console.cloud.google.com/apis/api/chat.googleapis.com/hangouts-chat), create a **Chat app** linked to the service account email.
4. Invite the bot to each target space (type `@<bot-name>` in the space and confirm).
5. Set the key path in your `.env`:

```
GCHAT_SERVICE_ACCOUNT_JSON=/path/to/service-account-key.json
```

No code changes needed — the server detects the env var at startup and unlocks all tools automatically.

| Variable | Description |
|----------|-------------|
| `GCHAT_WEBHOOK_URL` | Incoming webhook URL from a Chat space (webhook mode) |
| `GCHAT_SERVICE_ACCOUNT_JSON` | Path to the GCP Service Account JSON key file (full REST API access) |

Space and thread name formats used by the REST API tools:

| Identifier | Format |
|------------|--------|
| `space` | `spaces/SPACE_ID` |
| `thread_name` | `spaces/SPACE_ID/threads/THREAD_ID` |

Run `gchat__list_spaces` to discover `SPACE_ID` values once the bot is configured.

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
