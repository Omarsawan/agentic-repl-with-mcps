import json
import logging
from pathlib import Path

import torch

from .base import ChatResponse, ToolCall
from .generic_routing import PromptHandler
from .utils import last_user_message, format_table
from enums import SqlStep
from sql_model.intent_classifier import IntentClassifier
from sql_model.template_engine import TemplateEngine
from sql_model.query_detector import is_sql_query
from sql_model.tokenizer import BOS_ID, EOS_ID

logger = logging.getLogger(__name__)

_MAX_SCHEMA_RETRIES = 2  # TODO: implement exponential backoff with jitter for advanced retry
_MAX_SRC_TOKENS = 512

_SCHEMA_FETCH_QUERY = (
    "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE "
    "FROM INFORMATION_SCHEMA.COLUMNS "
    "WHERE TABLE_SCHEMA NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys') "
    "ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION"
)


class SQLHandler(PromptHandler):
    """Handles SQL-related prompts: generates SQL and executes it via MySQL MCP."""

    def __init__(
        self,
        checkpoint_path: str = "sql_model/checkpoints/best_model.pt",
        intent_confidence_threshold: float = 0.75,
        execute_query_tool: str = "mysql__execute_query",
    ) -> None:
        self.intent_confidence_threshold = intent_confidence_threshold
        self._configured_tool_name: str = execute_query_tool
        self._sql_tool_name: str | None = None
        self._sql_param_name: str = "query"

        self._model = None
        self._tokenizer = None
        ckpt = Path(checkpoint_path)
        if ckpt.is_file():
            try:
                from sql_model.trainer import Trainer  # type: ignore[import]

                self._model, self._tokenizer = Trainer.load(checkpoint_path)
                logger.info("SQLHandler: loaded model from %s", checkpoint_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLHandler: could not load model from %s: %s", checkpoint_path, exc)
        else:
            logger.info("SQLHandler: checkpoint not found at %s — template-only mode", checkpoint_path)

        self._classifier = IntentClassifier()
        self._template_engine = TemplateEngine({})

        self._schema: dict[str, list[str]] = {}
        self._schema_fetch_retries: int = 0

        self._step: SqlStep = SqlStep.FETCH_SCHEMA
        self._dispatched_sql: str | None = None
        self._last_user_text: str = ""
        self._last_sql_meta: dict = {}

    # ------------------------------------------------------------------
    # PromptHandler interface
    # ------------------------------------------------------------------

    def can_handle(self, user_text: str) -> bool:
        return is_sql_query(user_text)

    def reset(self) -> None:
        if self._step == SqlStep.COLLECT_RESULT:
            self._step = SqlStep.DISPATCH_SQL
        self._dispatched_sql = None
        self._last_user_text = ""
        self._last_sql_meta = {}

    # ------------------------------------------------------------------
    # Core chat logic — 3-step state machine
    # ------------------------------------------------------------------

    async def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        resolved = self._resolve_sql_tool(tools)
        if resolved is None:
            logger.error(
                "SQLHandler: no SQL execution tool found in tools list: %s",
                [t.get("function", {}).get("name") for t in tools],
            )
            return ChatResponse(content="No SQL execution tool is available.", tool_calls=[])
        self._sql_tool_name = resolved

        user_text = last_user_message(messages)

        # New question detected — reset SQL dispatch state but keep schema cache.
        if user_text != self._last_user_text:
            self._last_user_text = user_text
            self._dispatched_sql = None
            if self._step == SqlStep.COLLECT_RESULT:
                self._step = SqlStep.DISPATCH_SQL

        # Step 3: SQL has been dispatched; collect the execution result.
        if self._step == SqlStep.COLLECT_RESULT:
            return self._step3_collect_result(messages)

        # Step 1: Schema not yet cached — fetch it before generating SQL.
        if self._step == SqlStep.FETCH_SCHEMA:
            response = self._step1_fetch_schema(messages)
            if response is not None:
                return response

        # Step 2: Schema is ready — generate and dispatch SQL for execution.
        return self._step2_dispatch_sql(user_text)

    def _step1_fetch_schema(self, messages: list[dict]) -> ChatResponse | None:
        """Emit schema-fetch tool call, or parse result and advance to DISPATCH_SQL.

        Returns None when schema is ready so the caller falls through to step 2.
        """
        call_id = f"schema_fetch_{self._schema_fetch_retries}"
        schema_content = self._find_tool_result(messages, call_id)
        if schema_content is None:
            return ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name=self._sql_tool_name,
                        arguments={self._sql_param_name: _SCHEMA_FETCH_QUERY, 'limit': 1000},
                    )
                ],
            )

        if self._is_error_response(schema_content):
            logger.warning(
                "SQLHandler: schema fetch error (attempt %d): %s",
                self._schema_fetch_retries + 1,
                schema_content[:200],
            )
            if self._schema_fetch_retries < _MAX_SCHEMA_RETRIES:
                self._schema_fetch_retries += 1
                next_id = f"schema_fetch_{self._schema_fetch_retries}"
                # TODO: implement exponential backoff for more advanced retry strategies
                return ChatResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=next_id,
                            name=self._sql_tool_name,
                            arguments={self._sql_param_name: _SCHEMA_FETCH_QUERY},
                        )
                    ],
                )
            logger.error("SQLHandler: schema fetch failed after %d retries", _MAX_SCHEMA_RETRIES)
            self._schema_fetch_retries = 0
            return ChatResponse(
                content="Could not fetch the database schema after multiple attempts. Please check the database connection and try again.",
                tool_calls=[],
            )

        self._schema = self._parse_schema(schema_content)
        if not self._schema:
            logger.warning("SQLHandler: schema fetch returned 0 rows (empty schema)")
            self._schema_fetch_retries = 0
            return ChatResponse(
                content=(
                    "Could not fetch the database schema: the query returned no rows. "
                    "The database may be empty, the wrong database may be selected, "
                    "or the user may lack permission to read INFORMATION_SCHEMA."
                ),
                tool_calls=[],
            )
        self._schema_fetch_retries = 0
        self._template_engine = TemplateEngine(self._schema)
        self._step = SqlStep.DISPATCH_SQL
        return None

    def _step2_dispatch_sql(self, user_text: str) -> ChatResponse:
        """Generate SQL and emit tool call to execute it; advance to COLLECT_RESULT."""
        sql = self._generate_sql(user_text)
        if sql.startswith("--"):
            return ChatResponse(content="Could not generate SQL for this query.", tool_calls=[])
        self._dispatched_sql = sql
        self._step = SqlStep.COLLECT_RESULT
        return ChatResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="sql_exec_0",
                    name=self._sql_tool_name,
                    arguments={self._sql_param_name: sql},
                )
            ],
        )

    def _step3_collect_result(self, messages: list[dict]) -> ChatResponse:
        """Read the SQL execution result from messages and return formatted output."""
        sql_result = self._find_tool_result(messages, "sql_exec_0")
        if sql_result is None:
            return ChatResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="sql_exec_0",
                        name=self._sql_tool_name,
                        arguments={self._sql_param_name: self._dispatched_sql},
                    )
                ],
            )

        if self._is_error_response(sql_result):
            logger.warning("SQLHandler: SQL execution returned error: %s", sql_result[:200])
            self._dispatched_sql = None
            self._step = SqlStep.DISPATCH_SQL
            return ChatResponse(content=f"SQL execution failed:\n```\n{sql_result}\n```", tool_calls=[])

        executed_sql = self._dispatched_sql
        self._dispatched_sql = None
        self._step = SqlStep.DISPATCH_SQL
        try:
            rows = json.loads(sql_result)
            formatted_sql = format_table(rows) if isinstance(rows, list) else f"```\n{sql_result}\n```"
        except (json.JSONDecodeError, TypeError):
            formatted_sql = f"```\n{sql_result}\n```"
        content = f"```sql\n{executed_sql}\n```\n\n**Results:**\n{formatted_sql}"
        meta = self._last_sql_meta
        if meta:
            source_labels = {"template": "template engine", "neural": "neural model", None: "no match"}
            parts = [
                f"Intent: `{meta['intent']}`",
                f"Confidence: `{meta['confidence']:.0%}`",
                f"Source: {source_labels.get(meta['source'], meta['source'])}",
            ]
            if meta.get("entities"):
                entities_str = ", ".join(f"{k}={v}" for k, v in meta["entities"].items())
                parts.append(f"Entities: `{entities_str}`")
            content += "\n\n---\n*" + " · ".join(parts) + "*"
        return ChatResponse(content=content, tool_calls=[])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_sql_tool(self, tools: list[dict]) -> str | None:
        """Return the name of the SQL execution tool from the live tools list.

        Also discovers and stores the correct SQL parameter name in ``_sql_param_name``
        so tool calls use the actual parameter name from the schema rather than a
        hardcoded guess.

        Three-tier resolution:
        1. Exact match on the configured tool name.
        2. Name pattern — any tool whose name contains "execute_query".
        3. Schema inspection — any tool with a recognised SQL parameter name.
        """
        tool_map = {t.get("function", {}).get("name", ""): t.get("function", {}) for t in tools}

        def _find_sql_param(fn: dict) -> str:
            props = (fn.get("parameters") or {}).get("properties", {})
            for candidate in ("query", "sql", "statement", "q"):
                if candidate in props:
                    return candidate
            for name, spec in props.items():
                if spec.get("type") == "string":
                    return name
            return "query"

        if self._configured_tool_name in tool_map:
            self._sql_param_name = _find_sql_param(tool_map[self._configured_tool_name])
            return self._configured_tool_name

        for name in tool_map:
            if "execute_query" in name:
                self._sql_param_name = _find_sql_param(tool_map[name])
                return name

        for name, fn in tool_map.items():
            props = (fn.get("parameters") or {}).get("properties", {})
            for candidate in ("query", "sql", "statement"):
                if candidate in props:
                    self._sql_param_name = candidate
                    return name

        return None

    def _generate_sql(self, user_text: str) -> str:
        """Classify intent and generate SQL via template or neural model."""
        result = self._classifier.classify(user_text)
        self._last_sql_meta = {
            "intent": result.intent,
            "confidence": result.confidence,
            "entities": result.entities,
            "source": None,
        }

        sql: str | None = None
        if result.confidence >= self.intent_confidence_threshold:
            sql = self._template_engine.generate(result.intent, result.entities, user_text)
            if sql is not None:
                self._last_sql_meta["source"] = "template"

        if sql is None and self._model is not None:
            sql = self._generate_neural(user_text)
            if sql is not None:
                self._last_sql_meta["source"] = "neural"

        return sql if sql is not None else "-- Could not generate SQL for this query"

    def _parse_schema(self, tool_content: str) -> dict[str, list[str]]:
        try:
            data = json.loads(tool_content)
        except json.JSONDecodeError:
            logger.warning("SQLHandler: could not parse schema from tool content")
            return {}

        schema: dict[str, list[str]] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            db = row.get("TABLE_SCHEMA", "")
            table_name = row.get("TABLE_NAME", "")
            table = f"{db}.{table_name}" if db and table_name else table_name
            column = row.get("COLUMN_NAME", "")
            if table and column:
                schema.setdefault(table, []).append(column)

        return schema

    def _generate_neural(self, user_text: str) -> str | None:
        try:
            schema = self._relevant_schema(user_text)
            schema_str = "; ".join(
                f"{table}({', '.join(cols)})" for table, cols in schema.items()
            )
            src_text = f"{user_text} [SEP] {schema_str}"
            src_ids = self._tokenizer.encode(src_text)
            if len(src_ids) > _MAX_SRC_TOKENS:
                logger.warning(
                    "SQLHandler: input truncated from %d to %d tokens",
                    len(src_ids),
                    _MAX_SRC_TOKENS,
                )
                src_ids = src_ids[:_MAX_SRC_TOKENS]
            src = torch.tensor([src_ids])
            generated_ids = self._model.generate(src, bos_id=BOS_ID, eos_id=EOS_ID, max_len=256)
            sql = self._tokenizer.decode(generated_ids)
            return sql if sql.strip() else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQLHandler: neural generation failed: %s", exc)
            return None

    def _relevant_schema(self, user_text: str) -> dict[str, list[str]]:
        lower = user_text.lower()
        matched = {
            table: cols
            for table, cols in self._schema.items()
            if table.split(".")[-1].lower() in lower
        }
        return matched if matched else self._schema

    @staticmethod
    def _is_error_response(content: str) -> bool:
        stripped = content.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and ("error" in data or data.get("isError")):
                return True
        except json.JSONDecodeError:
            pass
        return stripped.lower().startswith("error")

    @staticmethod
    def _find_tool_result(messages: list[dict], tool_call_id: str) -> str | None:
        """Return the content of the most recent tool message matching tool_call_id, or None.

        Iterates in reverse because tool_call_id is reused across turns (e.g. "sql_exec_0"
        appears once per SQL query in the session), so the latest result must be returned.
        """
        for msg in reversed(messages):
            if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
                return str(msg.get("content", ""))
        return None
