import sys
import os
import json
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sql_model.tokenizer import SQLTokenizer, BOS_ID, EOS_ID, PAD_ID, UNK_ID
from sql_model.query_detector import is_sql_query
from sql_model.intent_classifier import IntentClassifier
from sql_model.template_engine import TemplateEngine


# --- Tokenizer ---

def test_tokenizer_roundtrip():
    tok = SQLTokenizer()
    tok.build_vocab(["how many orders", "select count from orders"])
    ids = tok.encode("how many orders")
    decoded = tok.decode(ids)
    assert "how" in decoded and "many" in decoded and "orders" in decoded


def test_tokenizer_unknown_token():
    tok = SQLTokenizer()
    tok.build_vocab(["hello world"])
    ids = tok.encode("zzz_unknown_xyz")
    assert UNK_ID in ids


def test_tokenizer_special_tokens():
    tok = SQLTokenizer()
    assert tok.token2id["<pad>"] == PAD_ID
    assert tok.token2id["<bos>"] == BOS_ID
    assert tok.token2id["<eos>"] == EOS_ID


# --- Query Detector ---

def test_query_detector_sql_queries():
    assert is_sql_query("how many orders were placed last month") is True
    assert is_sql_query("show me top 10 customers by revenue") is True
    assert is_sql_query("average order value grouped by region") is True


def test_query_detector_non_sql():
    assert is_sql_query("send a message to the team") is False
    assert is_sql_query("notify the on-call engineer") is False


# --- Intent Classifier ---

def test_intent_classifier_count():
    clf = IntentClassifier()
    result = clf.classify("how many orders were placed")
    assert result.intent == "COUNT"
    assert result.confidence > 0.0


def test_intent_classifier_top_n_extracts_limit():
    clf = IntentClassifier()
    result = clf.classify("show me top 5 customers")
    assert result.intent == "TOP_N"
    assert result.entities.get("limit") == 5


def test_intent_classifier_avg():
    clf = IntentClassifier()
    result = clf.classify("average revenue per customer")
    assert result.intent == "AVG"


# --- Template Engine ---

SCHEMA = {
    "orders": ["id", "customer_id", "amount", "created_at"],
    "customers": ["id", "name", "region"],
}


def test_template_engine_count():
    engine = TemplateEngine(SCHEMA)
    sql = engine.generate("COUNT", {}, "how many orders")
    assert sql is not None
    assert "COUNT(*)" in sql
    assert "orders" in sql


def test_template_engine_top_n():
    engine = TemplateEngine(SCHEMA)
    sql = engine.generate("TOP_N", {"limit": 5}, "top 5 orders by amount")
    assert sql is not None
    assert "LIMIT 5" in sql
    assert "orders" in sql


def test_template_engine_unknown_table_returns_none():
    engine = TemplateEngine(SCHEMA)
    sql = engine.generate("COUNT", {}, "how many unicorns flew today")
    assert sql is None


# --- Model smoke test ---

def test_model_forward_pass():
    from sql_model.model import Seq2SeqTransformer
    vocab_size = 100
    model = Seq2SeqTransformer(vocab_size=vocab_size, d_model=32, nhead=2,
                                num_encoder_layers=1, num_decoder_layers=1,
                                dim_feedforward=64)
    src = torch.randint(1, vocab_size, (2, 10))
    tgt = torch.randint(1, vocab_size, (2, 8))
    out = model(src, tgt)
    assert out.shape == (2, 8, vocab_size)


def test_model_generate():
    from sql_model.model import Seq2SeqTransformer
    vocab_size = 100
    model = Seq2SeqTransformer(vocab_size=vocab_size, d_model=32, nhead=2,
                                num_encoder_layers=1, num_decoder_layers=1,
                                dim_feedforward=64)
    src = torch.randint(1, vocab_size, (1, 10))
    result = model.generate(src, bos_id=BOS_ID, eos_id=EOS_ID, max_len=20)
    assert isinstance(result, list)


# --- Provider: schema fetch tool call ---

@pytest.mark.anyio
async def test_provider_requests_schema_when_missing():
    from agent_provider.keyword_match import KeywordMatchProvider
    from agent_provider.text_to_sql import TextToSQLProvider

    provider = TextToSQLProvider(
        fallback_provider=KeywordMatchProvider(),
        checkpoint_path="sql_model/checkpoints/nonexistent.pt",
    )

    messages = [{"role": "user", "content": "how many orders last month"}]
    tools = [{"type": "function", "function": {"name": "mysql__execute_query",
              "description": "run sql", "parameters": {}}}]

    response = await provider.chat(messages, tools)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "mysql__execute_query"
    assert "INFORMATION_SCHEMA" in response.tool_calls[0].arguments["query"]
