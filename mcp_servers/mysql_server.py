"""MCP server for MySQL access via an SSH tunnel."""

import atexit
import os
import socket
import subprocess
import time
from pathlib import Path

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

mcp = FastMCP("mysql")

# SSH tunnel config
_SSH_HOST = os.environ["MYSQL_SSH_HOST"]
_SSH_PORT = int(os.getenv("MYSQL_SSH_PORT", "22"))
_SSH_USER = os.environ["MYSQL_SSH_USER"]
_SSH_KEY_PATH = os.path.expanduser(os.environ["MYSQL_SSH_KEY_PATH"])

# MySQL config (as seen from the SSH server)
_MYSQL_HOST = os.environ["MYSQL_HOST"]
_MYSQL_PORT = int(os.environ["MYSQL_PORT"])
_MYSQL_USER = os.environ["MYSQL_USER"]
_MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
_MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

_ssh_proc: subprocess.Popen | None = None
_local_port: int | None = None


def _get_local_port() -> int:
    global _ssh_proc, _local_port
    if _ssh_proc is not None and _ssh_proc.poll() is not None:
        _ssh_proc = None  # process died, restart

    if _ssh_proc is None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            _local_port = s.getsockname()[1]

        cmd = [
            "ssh", "-N", "-q",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes",
            "-i", _SSH_KEY_PATH,
            "-p", str(_SSH_PORT),
            "-L", f"{_local_port}:{_MYSQL_HOST}:{_MYSQL_PORT}",
            f"{_SSH_USER}@{_SSH_HOST}",
        ]
        _ssh_proc = subprocess.Popen(cmd)
        atexit.register(_ssh_proc.terminate)
        time.sleep(1)  # wait for tunnel to be ready

    return _local_port  # type: ignore[return-value]


def _connect() -> pymysql.Connection:
    # Connect to the local end of the SSH tunnel, which forwards to _MYSQL_HOST:_MYSQL_PORT
    return pymysql.connect(
        host="127.0.0.1",
        port=_get_local_port(),
        user=_MYSQL_USER,
        password=_MYSQL_PASSWORD,
        database=_MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def _format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows returned)"
    columns = list(rows[0].keys())
    cell_rows = [
        [str(row[col]) if row[col] is not None else "NULL" for col in columns]
        for row in rows
    ]
    col_widths = [max(len(col), max(len(row[col_idx]) for row in cell_rows)) for col_idx, col in enumerate(columns)]
    separator = "| " + " | ".join("-" * width for width in col_widths) + " |"
    header = "| " + " | ".join(col.ljust(col_widths[col_idx]) for col_idx, col in enumerate(columns)) + " |"
    lines = [header, separator]
    for row in cell_rows:
        lines.append("| " + " | ".join(cell.ljust(col_widths[col_idx]) for col_idx, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def _is_select(sql: str) -> bool:
    first = sql.strip().split()[0].upper()
    return first in {"SELECT", "SHOW", "EXPLAIN", "DESCRIBE", "DESC", "WITH"}


@mcp.tool()
def execute_query(sql: str, database: str | None = None, limit: int = 100) -> str:
    """Execute a read-only SQL query and return results as a formatted table.

    Only SELECT, SHOW, EXPLAIN, DESCRIBE, and WITH statements are allowed.
    At most `limit` rows are returned (default 100, max 1000).
    """
    if not _is_select(sql):
        return "error: Only read-only queries (SELECT/SHOW/EXPLAIN/DESCRIBE/WITH) are allowed."

    limit = min(max(1, limit), 1000)

    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                if database:
                    cur.execute("USE `%s`" % database.replace("`", ""))
                cur.execute(sql)
                rows = cur.fetchmany(limit)
        return f"{len(rows)} row(s)\n\n{_format_table(rows)}"
    except pymysql.Error as exc:
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else 0
        if code >= 2000:
            return f"error: Connection error (code {code})"
        return f"error: {exc}"
    except Exception:
        return "error: Server error"


@mcp.tool()
def list_tables(database: str | None = None) -> str:
    """List all tables in the given database (or the default database if omitted)."""
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                if database:
                    cur.execute("USE `%s`" % database.replace("`", ""))
                cur.execute("SHOW TABLES")
                rows = cur.fetchall()
        tables = [list(row.values())[0] for row in rows]
        return "\n".join(tables)
    except pymysql.Error as exc:
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else 0
        if code >= 2000:
            return f"error: Connection error (code {code})"
        return f"error: {exc}"
    except Exception:
        return "error: Server error"


@mcp.tool()
def describe_table(table: str, database: str | None = None) -> str:
    """Describe the columns of a table (name, type, nullable, key, default, extra)."""
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                if database:
                    cur.execute("USE `%s`" % database.replace("`", ""))
                cur.execute("DESCRIBE `%s`" % table.replace("`", ""))
                rows = cur.fetchall()
        return _format_table(rows)
    except pymysql.Error as exc:
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else 0
        if code >= 2000:
            return f"error: Connection error (code {code})"
        return f"error: {exc}"
    except Exception:
        return "error: Server error"


if __name__ == "__main__":
    mcp.run(transport="stdio")
