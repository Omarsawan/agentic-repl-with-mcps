import re

_SQL_SIGNALS = [
    "how many", "count", "total", "sum", "average", "avg", "mean",
    "maximum", "minimum", "max", "min", "top", "bottom", "rank",
    "show me", "list", "find", "get", "fetch", "retrieve",
    "group by", "grouped by", "per", "by each",
    "trend", "over time", "last month", "last week", "last year",
    "filter", "where", "having", "between", "greater than", "less than",
    "select", "query", "table", "row", "record", "column",
    "revenue", "sales", "orders", "customers", "products",
    "breakdown", "distribution", "report", "analytics",
]

_NON_SQL_SIGNALS = [
    "send", "message", "notify", "alert", "email", "chat",
    "create ticket", "open ticket", "file a", "schedule",
    "remind", "post", "publish", "deploy", "restart", "stop",
    "what time", "current time", "weather", "news",
    "search the web", "google", "browse",
]


def is_sql_query(text: str) -> bool:
    low = text.lower()

    non_sql_hits = sum(1 for sig in _NON_SQL_SIGNALS if sig in low)
    if non_sql_hits >= 1:
        return False

    sql_hits = sum(1 for sig in _SQL_SIGNALS if sig in low)
    return sql_hits >= 1
