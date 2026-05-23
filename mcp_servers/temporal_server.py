"""MCP server exposing Temporal Cloud workflow tools."""

import os

from mcp.server.fastmcp import FastMCP
from temporalio.client import Client, TLSConfig

mcp = FastMCP("temporal")


async def _get_client() -> Client:
    address = os.environ.get("TEMPORAL_ADDRESS")
    if not address:
        raise RuntimeError("TEMPORAL_ADDRESS environment variable is not set")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    cert_path = os.environ.get("TEMPORAL_TLS_CERT")
    key_path = os.environ.get("TEMPORAL_TLS_KEY")

    tls: bool | TLSConfig = False
    if cert_path and key_path:
        with open(cert_path, "rb") as f:
            cert = f.read()
        with open(key_path, "rb") as f:
            key = f.read()
        tls = TLSConfig(client_cert=cert, client_private_key=key)

    return await Client.connect(address, namespace=namespace, tls=tls)


@mcp.tool()
async def list_workflows(query: str = "ExecutionStatus = 'Running'", max_results: int = 20) -> str:
    """List Temporal workflows matching a Workflow Query Language (WQL) filter.

    Use WQL syntax for the query parameter. Examples:
      WorkflowType = 'OrderWorkflow'
      ExecutionStatus = 'Running'
      WorkflowId = 'my-workflow-123'
      RunId = 'abc-def-123'
      StartTime > '2024-01-01T00:00:00Z'
      StartTime >= '2024-01-01T00:00:00Z' AND StartTime <= '2024-12-31T23:59:59Z'
      StartTime BETWEEN '2024-01-01T00:00:00Z' AND '2024-12-31T23:59:59Z'
      WorkflowType = 'OrderWorkflow' AND ExecutionStatus = 'Completed'

    Time comparison operators: use > >= < <= = or BETWEEN ... AND ...
    Do NOT use AFTER, BEFORE, or SINCE — those are not valid WQL keywords.
    Timestamps must be ISO-8601 format, e.g. '2024-01-01T00:00:00Z'.

    ExecutionStatus values: Running, Completed, Failed, Canceled, Terminated,
    ContinuedAsNew, TimedOut

    Leave query empty to list recent workflows.
    max_results is capped at 100.
    """
    max_results = min(max(1, max_results), 100)

    try:
        client = await _get_client()
    except Exception:
        return "Error: could not connect to Temporal service"
    try:
        rows = []
        async for wf in client.list_workflows(query=query):
            close_time = wf.close_time.isoformat() if wf.close_time else "-"
            rows.append(
                f"ID:         {wf.id}\n"
                f"  RunID:    {wf.run_id}\n"
                f"  Type:     {wf.workflow_type}\n"
                f"  Status:   {wf.status.name}\n"
                f"  Started:  {wf.start_time.isoformat()}\n"
                f"  Closed:   {close_time}\n"
                f"  Queue:    {wf.task_queue}"
            )
            if len(rows) >= max_results:
                break
        if not rows:
            return "No workflows found matching the query."
        return f"Found {len(rows)} workflow(s):\n\n" + "\n\n".join(rows)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
async def describe_workflow(workflow_id: str = "", run_id: str = "") -> str:
    """Get detailed information about a specific Temporal workflow.

    Provide workflow_id, run_id, or both:
    - workflow_id only: describes the latest run of that workflow.
    - run_id only: resolves the workflow automatically then describes it.
    - Both: describes that exact run.
    """
    if not workflow_id and not run_id:
        return "Error: provide at least one of workflow_id or run_id."

    try:
        client = await _get_client()
    except Exception:
        return "Error: could not connect to Temporal service"
    try:
        if not workflow_id:
            async for wf in client.list_workflows(query=f'RunId = "{run_id}"'):
                workflow_id = wf.id
                break
            if not workflow_id:
                return f"Error: no workflow found with run_id '{run_id}'."

        handle = client.get_workflow_handle(workflow_id, run_id=run_id or None)
        desc = await handle.describe()

        close_time = desc.close_time.isoformat() if desc.close_time else "-"
        exec_timeout = str(desc.execution_time) if desc.execution_time else "-"
        parent = f"{desc.parent_id} / {desc.parent_run_id}" if desc.parent_id else "-"

        namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
        ui_base = os.environ.get("TEMPORAL_UI_BASE_URL", "https://cloud.temporal.io")
        ui_url = f"{ui_base}/namespaces/{namespace}/workflows/{desc.id}/{desc.run_id}"

        return "\n".join([
            f"WorkflowID:       {desc.id}",
            f"RunID:            {desc.run_id}",
            f"Type:             {desc.workflow_type}",
            f"Status:           {desc.status.name}",
            f"TaskQueue:        {desc.task_queue}",
            f"Started:          {desc.start_time.isoformat()}",
            f"Closed:           {close_time}",
            f"ExecTimeout:      {exec_timeout}",
            f"Parent:           {parent}",
            f"UI:               {ui_url}",
        ])
    except Exception as exc:
        return f"Error: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
