_MAX_CELL = 60


def last_user_message(messages: list[dict]) -> str:
    """Return the content of the most recent user message, or an empty string if none exists."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _fmt_cell(val: object) -> str:
    s = str(val) if val is not None else "NULL"
    s = s.replace("\r", "").replace("\n", " ")
    return s if len(s) <= _MAX_CELL else s[: _MAX_CELL - 3] + "..."


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows returned)"
    columns = list(rows[0].keys())
    cell_rows = [[_fmt_cell(row[col]) for col in columns] for row in rows]
    col_widths = [
        max(3, len(col), max(len(r[i]) for r in cell_rows))
        for i, col in enumerate(columns)
    ]
    divider = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header = (
        "| "
        + " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
        + " |"
    )
    lines = [divider, header, divider]
    for row in cell_rows:
        lines.append(
            "| "
            + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
            + " |"
        )
    lines.append(divider)
    return "```\n" + "\n".join(lines) + "\n```"
