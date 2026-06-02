import json
import logging
import random
import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .tokenizer import SQLTokenizer, PAD_ID

logger = logging.getLogger(__name__)


def _parse_schema_string(schema_str: str) -> dict[str, list[str]]:
    """Parse 'table : col1 (type), col2 (type) | table2 : ...' into {table: [cols]}."""
    schema: dict[str, list[str]] = {}
    for table_entry in schema_str.split(" | "):
        parts = table_entry.split(" : ", 1)
        if len(parts) != 2:
            continue
        table_name = parts[0].strip()
        col_entries = [c.strip() for c in parts[1].split(",")]
        cols = [c.split(" (")[0].strip() for c in col_entries if c]
        schema[table_name] = cols
    return schema


def _load_spider_schemas() -> dict[str, dict[str, list[str]]]:
    """Download Spider schemas from richardr1126/spider-schema on HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="richardr1126/spider-schema",
            filename="spider_schema_rows_v2.json",
            repo_type="dataset",
        )
        with open(path) as f:
            rows = json.load(f)
    except Exception as exc:
        logger.warning("Could not load Spider schemas: %s — falling back to db_id format", exc)
        return {}

    schemas: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        db_id = row.get("db_id", "")
        raw = row.get("Schema (values (type))", "")
        if db_id and raw:
            schemas[db_id] = _parse_schema_string(raw)
    return schemas


def _format_schema(db_id: str, schema: dict[str, list[str]]) -> str:
    """Format schema as 'db_id.table1(col1, col2); db_id.table2(col3)' — matches inference format."""
    return "; ".join(f"{db_id}.{tbl}({', '.join(cols)})" for tbl, cols in schema.items() if cols)


def _qualify_sql(sql: str, db_id: str, table_names: list[str]) -> str:
    """Prefix each bare table name in a Spider SQL query with db_id.

    Processes longest names first to avoid partial matches (e.g. 'singer' inside
    'singer_in_concert'). Only replaces standalone occurrences — not those already
    preceded or followed by a word char or dot.
    """
    for table in sorted(table_names, key=len, reverse=True):
        pattern = rf"(?<![.\w]){re.escape(table)}(?![.\w])"
        sql = re.sub(pattern, f"{db_id}.{table}", sql, flags=re.IGNORECASE)
    return sql


class SpiderDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: "SQLTokenizer",
        max_src_len: int = 512,
        max_tgt_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        logger.info("Loading Spider schemas from HuggingFace...")
        db_schemas = _load_spider_schemas()
        using_schema = bool(db_schemas)
        if not using_schema:
            logger.warning("Schema unavailable — training with db_id format (degraded mode)")

        raw = load_dataset("xlangai/spider", split=split)
        self.examples: list[dict] = []
        for item in raw:
            question = item["question"]
            db_id = item["db_id"]
            if using_schema:
                if db_id not in db_schemas:
                    logger.warning("Skipping example: no schema for db_id=%s", db_id)
                    continue
                schema = db_schemas[db_id]
                lower_question = question.lower()
                db_hit = db_id.lower() in lower_question
                both, one, neither = {}, {}, {}
                for t, c in schema.items():
                    tbl_hit = t.lower() in lower_question
                    if tbl_hit and db_hit:
                        both[t] = c
                    elif tbl_hit or db_hit:
                        one[t] = c
                    else:
                        neither[t] = c
                schema = {**both, **one, **neither}
                schema_str = _format_schema(db_id, schema)
                query = _qualify_sql(item["query"], db_id, list(schema.keys()))
                src_text = f"{question} [SEP] {schema_str}"
            else:
                query = item["query"]
                src_text = f"{question} [SEP] db:{db_id}"
            self.examples.append({"src_text": src_text, "tgt_text": query})

    def sample_preview(self, n: int = 3) -> list[tuple[str, str]]:
        """Return n random (src_text, tgt_text) pairs for a quick visual sanity check."""
        samples = random.sample(self.examples, min(n, len(self.examples)))
        return [(s["src_text"], s["tgt_text"]) for s in samples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        src_ids = self.tokenizer.encode(ex["src_text"])
        # Pad / truncate src
        src_ids = src_ids[: self.max_src_len]
        src_ids = src_ids + [PAD_ID] * (self.max_src_len - len(src_ids))

        tgt_ids = self.tokenizer.encode(ex["tgt_text"])

        bos_id = self.tokenizer.token2id.get("<bos>", 1)
        eos_id = self.tokenizer.token2id.get("<eos>", 2)

        # tgt_in: [BOS] + tgt tokens, truncated to max_tgt_len - 1, then pad to max_tgt_len
        tgt_in = [bos_id] + tgt_ids[: self.max_tgt_len - 1]
        tgt_in = tgt_in + [PAD_ID] * (self.max_tgt_len - len(tgt_in))

        # tgt_out: tgt tokens truncated to max_tgt_len - 1, then [EOS], then pad to max_tgt_len
        tgt_core = tgt_ids[: self.max_tgt_len - 1]
        tgt_out = tgt_core + [eos_id]
        tgt_out = tgt_out + [PAD_ID] * (self.max_tgt_len - len(tgt_out))

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in": torch.tensor(tgt_in, dtype=torch.long),
            "tgt_out": torch.tensor(tgt_out, dtype=torch.long),
        }


def build_tokenizer_from_spider() -> "SQLTokenizer":
    tokenizer = SQLTokenizer()
    raw = load_dataset("xlangai/spider", split="train")
    db_schemas = _load_spider_schemas()
    all_texts: list[str] = []
    for item in raw:
        all_texts.append(item["question"])
        all_texts.append(item["query"])
    # Include db-qualified schema strings so the vocabulary covers all db_id.table references
    for db_id, schema in db_schemas.items():
        schema_str = _format_schema(db_id, schema)
        if schema_str:
            all_texts.append(schema_str)
    tokenizer.build_vocab(all_texts)
    return tokenizer


def collate_fn(batch: list[dict]) -> dict:
    result = {}
    for key in ("src", "tgt_in", "tgt_out"):
        tensors = [item[key] for item in batch]
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            pad_size = max_len - t.size(0)
            if pad_size > 0:
                t = torch.cat([t, torch.full((pad_size,), PAD_ID, dtype=torch.long)])
            padded.append(t)
        result[key] = torch.stack(padded, dim=0)
    return result
