TEMPLATES = {
    "COUNT": "SELECT COUNT(*) FROM {table}{where_clause}",
    "SUM": "SELECT SUM({column}) FROM {table}{where_clause}",
    "AVG": "SELECT AVG({column}) FROM {table}{where_clause}",
    "GROUP_BY": "SELECT {group_column}, COUNT(*) FROM {table} GROUP BY {group_column}{order_clause}",
    "TOP_N": "SELECT * FROM {table} ORDER BY {column} DESC LIMIT {limit}",
    "FILTER": "SELECT * FROM {table} WHERE {condition}",
    "TIME_SERIES": (
        "SELECT DATE({time_column}), COUNT(*) FROM {table} "
        "GROUP BY DATE({time_column}) ORDER BY DATE({time_column})"
    ),
    "DISTINCT": "SELECT COUNT(DISTINCT {column}) FROM {table}",
    "JOIN": "SELECT * FROM {table1} JOIN {table2} ON {table1}.id = {table2}.{table1}_id LIMIT 100",
}

NUMERIC_COLUMNS = {"amount", "price", "revenue", "value", "cost", "total", "count", "qty", "quantity"}
TIME_KEYWORDS = {"date", "time", "created", "updated", "at"}


class TemplateEngine:
    def __init__(self, schema: dict[str, list[str]]):
        self.schema = schema

    def _find_table(self, text: str) -> str | None:
        lower = text.lower()
        for table in self.schema:
            if table.lower() in lower:
                return table
        return None

    def _find_column(self, text: str, table: str | None) -> str | None:
        lower = text.lower()
        if table and table in self.schema:
            columns = self.schema[table]
        else:
            columns = [col for cols in self.schema.values() for col in cols]

        # Prefer numeric-sounding columns
        for col in columns:
            if col.lower() in NUMERIC_COLUMNS and col.lower() in lower:
                return col

        # Fall back to any column found in text
        for col in columns:
            if col.lower() in lower:
                return col

        return None

    def _find_time_column(self, table: str | None) -> str | None:
        if table and table in self.schema:
            columns = self.schema[table]
        else:
            columns = [col for cols in self.schema.values() for col in cols]

        for col in columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in TIME_KEYWORDS):
                return col
        return None

    def _build_where_clause(self, entities: dict) -> str:
        time_filter = entities.get("time_filter")
        if time_filter:
            n = time_filter.get("n", 1)
            unit = time_filter.get("unit", "DAY")
            return f" WHERE created_at >= DATE_SUB(NOW(), INTERVAL {n} {unit})"
        return ""

    def generate(self, intent: str, entities: dict, user_text: str) -> str | None:
        intent = intent.upper()

        if intent == "COUNT":
            table = self._find_table(user_text)
            if not table:
                return None
            where_clause = self._build_where_clause(entities)
            return TEMPLATES["COUNT"].format(table=table, where_clause=where_clause)

        elif intent == "SUM":
            table = self._find_table(user_text)
            column = self._find_column(user_text, table)
            if not table or not column:
                return None
            where_clause = self._build_where_clause(entities)
            return TEMPLATES["SUM"].format(column=column, table=table, where_clause=where_clause)

        elif intent == "AVG":
            table = self._find_table(user_text)
            column = self._find_column(user_text, table)
            if not table or not column:
                return None
            where_clause = self._build_where_clause(entities)
            return TEMPLATES["AVG"].format(column=column, table=table, where_clause=where_clause)

        elif intent == "GROUP_BY":
            table = self._find_table(user_text)
            group_column = entities.get("group_column") or self._find_column(user_text, table)
            if not table or not group_column:
                return None
            order_clause = entities.get("order_clause", "")
            return TEMPLATES["GROUP_BY"].format(
                group_column=group_column,
                table=table,
                order_clause=order_clause,
            )

        elif intent == "TOP_N":
            table = self._find_table(user_text)
            column = self._find_column(user_text, table)
            if not table or not column:
                return None
            limit = entities.get("limit", 10)
            return TEMPLATES["TOP_N"].format(table=table, column=column, limit=limit)

        elif intent == "FILTER":
            table = self._find_table(user_text)
            condition = entities.get("condition")
            if not table or not condition:
                return None
            return TEMPLATES["FILTER"].format(table=table, condition=condition)

        elif intent == "TIME_SERIES":
            table = self._find_table(user_text)
            time_column = self._find_time_column(table)
            if not table or not time_column:
                return None
            return TEMPLATES["TIME_SERIES"].format(time_column=time_column, table=table)

        elif intent == "DISTINCT":
            table = self._find_table(user_text)
            column = self._find_column(user_text, table)
            if not table or not column:
                return None
            return TEMPLATES["DISTINCT"].format(column=column, table=table)

        elif intent == "JOIN":
            table1 = entities.get("table1") or self._find_table(user_text)
            table2 = entities.get("table2")
            if not table1 or not table2:
                return None
            return TEMPLATES["JOIN"].format(table1=table1, table2=table2)

        return None
