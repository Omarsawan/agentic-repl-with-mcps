"""Synthetic data generation for fine-tuning on a production schema.

Schema file format: JSON array with TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME fields —
identical to the output of an INFORMATION_SCHEMA.COLUMNS query.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import SQLTokenizer, PAD_ID

logger = logging.getLogger(__name__)

_NL_QUESTIONS: dict[str, list[str]] = {
    "COUNT": [
        "how many {table} are there",
        "count all {table}",
        "what is the total number of {table}",
        "what is the total number of rows in {table}",
        "show me the count of {table}",
    ],
    "SUM": [
        "what is the total {column} in {table}",
        "sum of {column} from {table}",
        "total {column} across all {table}",
    ],
    "AVG": [
        "what is the average {column} in {table}",
        "average {column} of {table}",
        "mean {column} for {table}",
    ],
    "GROUP_BY": [
        "count {table} grouped by {group_col}",
        "how many {table} per {group_col}",
        "show {table} broken down by {group_col}",
        "{table} count for each {group_col}",
    ],
    "TOP_N": [
        "top 10 {table} by {column}",
        "show 5 {table} with highest {column}",
        "top 10 {table} ordered by {column} descending",
    ],
    "TIME_SERIES": [
        "show {table} trend over time",
        "count {table} by day",
        "{table} count over time by {time_col}",
    ],
    "DISTINCT": [
        "how many unique {column} in {table}",
        "distinct {column} values in {table}",
        "count distinct {column} from {table}",
    ],
}

_SQL: dict[str, str] = {
    "COUNT": "SELECT COUNT(*) FROM {table}",
    "SUM": "SELECT SUM({column}) FROM {table}",
    "AVG": "SELECT AVG({column}) FROM {table}",
    "GROUP_BY": "SELECT {group_col}, COUNT(*) FROM {table} GROUP BY {group_col} ORDER BY COUNT(*) DESC",
    "TOP_N": "SELECT * FROM {table} ORDER BY {column} DESC LIMIT 10",
    "TIME_SERIES": "SELECT DATE({time_col}), COUNT(*) FROM {table} GROUP BY DATE({time_col}) ORDER BY DATE({time_col})",
    "DISTINCT": "SELECT COUNT(DISTINCT {column}) FROM {table}",
}

_NUMERIC_HINTS = {"amount", "price", "revenue", "value", "cost", "total", "count", "qty", "quantity", "score", "rate", "fee", "salary"}
_TIME_HINTS = {"date", "time", "created", "updated", "at", "timestamp", "day", "month", "year"}
_NON_GROUP_HINTS = {"id", "uuid", "key", "hash", "token", "password", "email", "phone"}


def _numeric_col(cols: list[str]) -> Optional[str]:
    for c in cols:
        if any(h in c.lower() for h in _NUMERIC_HINTS):
            return c
    return None


def _time_col(cols: list[str]) -> Optional[str]:
    for c in cols:
        if any(h in c.lower() for h in _TIME_HINTS):
            return c
    return None


def _group_col(cols: list[str]) -> Optional[str]:
    """Pick a non-id, non-numeric column suitable for GROUP BY."""
    for c in cols:
        low = c.lower()
        if any(h in low for h in _NON_GROUP_HINTS):
            continue
        if any(h in low for h in _NUMERIC_HINTS):
            continue
        return c
    return cols[0] if cols else None


def _format_schema(schema: dict[str, list[str]]) -> str:
    return "; ".join(f"{tbl}({', '.join(cols)})" for tbl, cols in schema.items() if cols)


def parse_schema_file(path: str) -> dict[str, list[str]]:
    """Parse INFORMATION_SCHEMA JSON output into {table: [columns]} dict."""
    with open(path) as f:
        rows = json.load(f)
    schema: dict[str, list[str]] = {}
    for row in rows:
        db = row.get("TABLE_SCHEMA", "")
        tbl = row.get("TABLE_NAME", "")
        col = row.get("COLUMN_NAME", "")
        full_table = f"{db}.{tbl}" if db and tbl else tbl
        if full_table and col:
            schema.setdefault(full_table, []).append(col)
    return schema


def _relevant_schema_for_pair(question: str, schema: dict[str, list[str]]) -> str:
    """Order schema tables using the same 3-tier logic as sql_handler._relevant_schema.

    Tier 1: both db prefix AND short table name appear in the question.
    Tier 2: either db prefix OR short table name appears.
    Tier 3: neither.

    This mirrors what the inference handler does so training and inference see
    the same schema ordering, keeping the target table at the top of the input.
    """
    lower = question.lower()
    tier1: dict[str, list[str]] = {}
    tier2: dict[str, list[str]] = {}
    tier3: dict[str, list[str]] = {}
    for tbl, cols in schema.items():
        parts = tbl.lower().split(".")
        db_hit = len(parts) > 1 and parts[0] in lower
        tbl_hit = parts[-1] in lower
        if db_hit and tbl_hit:
            tier1[tbl] = cols
        elif db_hit or tbl_hit:
            tier2[tbl] = cols
        else:
            tier3[tbl] = cols
    return _format_schema({**tier1, **tier2, **tier3})


def generate_pairs(schema: dict[str, list[str]]) -> list[dict]:
    """Generate synthetic (src_text, tgt_text) training pairs from a production schema."""
    pairs: list[dict] = []

    for table, cols in schema.items():
        if not cols:
            continue
        short_table = table.split(".")[-1]

        num_col = _numeric_col(cols)
        t_col = _time_col(cols)
        g_col = _group_col(cols)

        for intent, questions in _NL_QUESTIONS.items():
            if intent in ("SUM", "AVG", "TOP_N") and num_col is None:
                continue
            if intent == "TIME_SERIES" and t_col is None:
                continue
            if intent == "GROUP_BY" and g_col is None:
                continue

            col = num_col or cols[0]
            sql_slots = {
                "table": table,
                "column": col,
                "group_col": g_col or col,
                "time_col": t_col or col,
            }
            nl_slots = {**sql_slots, "table": short_table}
            sql = _SQL[intent].format(**sql_slots)

            for q_template in questions:
                question = q_template.format(**nl_slots)
                schema_str = _relevant_schema_for_pair(question, schema)
                pairs.append({"src_text": f"{question} [SEP] {schema_str}", "tgt_text": sql})

    logger.info("Generated %d synthetic training pairs from %d tables", len(pairs), len(schema))
    return pairs


class SyntheticDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer: SQLTokenizer, max_src_len: int = 512, max_tgt_len: int = 256):
        self.examples = pairs
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        bos_id = self.tokenizer.token2id.get("<bos>", 1)
        eos_id = self.tokenizer.token2id.get("<eos>", 2)

        src_ids = self.tokenizer.encode(ex["src_text"])[: self.max_src_len]
        src_ids += [PAD_ID] * (self.max_src_len - len(src_ids))

        tgt_ids = self.tokenizer.encode(ex["tgt_text"])
        tgt_in = [bos_id] + tgt_ids[: self.max_tgt_len - 1]
        tgt_in += [PAD_ID] * (self.max_tgt_len - len(tgt_in))
        tgt_core = tgt_ids[: self.max_tgt_len - 1]
        tgt_out = tgt_core + [eos_id]
        tgt_out += [PAD_ID] * (self.max_tgt_len - len(tgt_out))

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in": torch.tensor(tgt_in, dtype=torch.long),
            "tgt_out": torch.tensor(tgt_out, dtype=torch.long),
        }
