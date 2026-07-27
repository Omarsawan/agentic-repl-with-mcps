# sql_model

A hybrid text-to-SQL model that combines intent-based template matching with a small seq2seq transformer trained from scratch.

## Architecture

```
User Query
   │
   ├─ query_detector.py   — is this a SQL/data query?
   │
   ├─ intent_classifier.py — TF-IDF cosine similarity → intent + confidence
   │
   ├─ confidence ≥ threshold → template_engine.py — fill SQL template
   │
   └─ confidence < threshold → model.py (Seq2SeqTransformer) — generate SQL
```

**Why hybrid?** Questions like "how many orders last month" or "top 10
customers by revenue" follow a small number of shapes. `template_engine.py`
handles those deterministically and cheaply. The neural model only has to
cover the long tail of phrasing templates can't match, so it can stay tiny
(4 layers, 256 dim) and train from scratch on a laptop instead of needing a
large pretrained LLM.

### Components

| Module | Role |
|--------|------|
| `model.py` | Encoder-decoder transformer (4 layers, 256 dim, 4 heads) |
| `tokenizer.py` | Word-level tokenizer with SQL keyword vocab |
| `intent_classifier.py` | TF-IDF + cosine similarity intent classifier |
| `template_engine.py` | Schema-aware SQL template filler |
| `query_detector.py` | Heuristic classifier: SQL query vs general action |
| `dataset.py` | HuggingFace Spider dataset loader |
| `finetune_data.py` | Synthetic data generation from a production schema |
| `trainer.py` | AdamW training loop with checkpoint save/load |
| `train.py` | Unified CLI entry point for pretraining and fine-tuning |

---

## How it works

### The neural model ([model.py](model.py))

`Seq2SeqTransformer` is a standard encoder-decoder transformer — the same
architecture family behind machine translation models — wrapping
`torch.nn.Transformer`:

- Separate embeddings for source and target tokens (`src_embed`,
  `tgt_embed`), each scaled by `sqrt(d_model)` before adding positional
  encoding, so embedding magnitude doesn't dwarf the position signal.
- `PositionalEncoding` ([model.py:7-20](model.py#L7)) is the classic
  sinusoidal kind (sin/cos of position) — no learned position embeddings.
- `forward()` ([model.py:58-76](model.py#L58)) builds three masks before
  calling the transformer:
  - a **padding mask** on `src` and `tgt` so attention ignores `<pad>`
    tokens
  - a **causal mask** on `tgt`
    (`generate_square_subsequent_mask`) so during training the decoder
    can't attend to future tokens — this is what makes teacher-forced
    training work: the whole target sequence is fed in at once, but each
    position can only "see" what came before it.
  - `proj`, a linear layer mapping the decoder's output back to
    vocab-sized logits, trained with cross-entropy loss
    (`ignore_index` on the pad token).

**Input framing:** the model never sees just the question. Both training
and inference encode `"{question} [SEP] {schema_str}"`, e.g.
`"how many orders last month [SEP] orders(id, customer_id, created_at)"`.
This conditions generation on the actual schema, so the model learns to
copy real table/column names rather than memorizing them.

### Beam search inference ([model.py:78-133](model.py#L78))

Greedy decoding (always taking the single most likely next token) can lock
in an early mistake it can never undo. Beam search hedges by tracking
several candidate sequences at once:

1. **Encode once**: run the encoder on `src`, cache the result as `memory`
   — every decode step reuses it instead of re-encoding.
2. **Start with one beam**: `[BOS]` with cumulative log-probability `0.0`.
3. **At each step**, for every active beam, run the decoder forward, take
   `log_softmax` over the vocab for the last position, and keep the
   top `beam_size` (default 4) next-token candidates.
4. **Expand and prune**: every beam is extended by every one of its
   candidates (cumulative log-prob = old + new), all resulting hypotheses
   across *all* beams are pooled, globally sorted by score, and only the
   top `beam_size` survive to the next step. This global re-ranking is
   what distinguishes beam search from just running greedy decoding 4
   times in parallel.
5. Any beam whose top token is `EOS` is moved to a `completed` list and
   stops expanding. Decoding halts at `max_len` or once beams run out.
6. Return the highest-scoring completed sequence (or the best incomplete
   beam if none finished).

**Known limitation to explore:** scores are cumulative log-probabilities,
not length-normalized, so beam search here is subtly biased toward shorter
outputs (each extra token can only make the cumulative log-prob more
negative). Adding a length penalty is a good first hands-on exercise.

### 3-tier schema matching

The same relevance-ordering logic appears in three places — training data
generation ([finetune_data.py:122-146](finetune_data.py#L122)), inference
([sql_handler.py:360-382](../agent_provider/sql_handler.py#L360)), and
template filling ([template_engine.py:24-36](template_engine.py#L24)) —
so training and serving see schema formatted the same way.

For each `db.table` in the schema, check whether the question mentions the
db name and/or the bare table name:

| Tier | Condition | Priority |
|------|-----------|----------|
| 1 | both db prefix **and** table name appear in the question | highest |
| 2 | either one appears | medium |
| 3 | neither appears | lowest |

Tables are concatenated tier-1-first. This matters because encoder input
is truncated to `_MAX_SRC_TOKENS` (512) tokens — with a large schema, not
every table fits, so ordering by relevance ensures the tables the question
actually needs survive truncation instead of being cut arbitrarily.

### Training pipeline

Two distinct data regimes, both driven by [train.py](train.py):

- **Pretrain**: learns general text-to-SQL mapping from the
  [Spider](https://huggingface.co/datasets/xlangai/spider) benchmark
  (~8k question/SQL pairs across many databases) via
  [dataset.py](dataset.py). The tokenizer vocabulary is built from
  scratch off Spider's questions, queries, and schemas
  ([tokenizer.py](tokenizer.py)).
- **Fine-tune**: adapts a pretrained checkpoint to *your* actual database.
  [finetune_data.py](finetune_data.py) synthesizes `(question, sql)` pairs
  per table/intent directly from an `INFORMATION_SCHEMA` dump — no real
  user queries required, since the synthetic questions are templated from
  schema shape alone. If fine-tuning introduces new vocabulary (real
  table/column names), embeddings and the output projection are grown in
  place (`_extend_vocab`, [train.py:20-35](train.py#L20)) rather than
  retraining a new tokenizer from scratch.

Both modes share the same `Trainer` loop
([trainer.py:13-68](trainer.py#L13)): AdamW, linear LR warmup, gradient
clipping at norm 1.0, and early stopping on loss plateau. See the CLI flag
tables below for the exact hyperparameters of each mode.

### Request lifecycle (where the model lives at runtime)

One query traces through
[agent_provider/sql_handler.py](../agent_provider/sql_handler.py)'s 3-step
state machine:

1. **FETCH_SCHEMA** — emits an `INFORMATION_SCHEMA.COLUMNS` query via the
   `mysql__execute_query` MCP tool and parses it into `{table: [columns]}`.
2. **DISPATCH_SQL** — classifies intent; if confidence clears the
   threshold, fills a template, otherwise falls back to
   `Seq2SeqTransformer.generate()` with the 3-tier-ordered schema; emits
   the resulting SQL as another `mysql__execute_query` call.
3. **COLLECT_RESULT** — reads the query result, formats it as a markdown
   table, and returns both the generated SQL and the results to the user.

This is also why the model itself never touches a database connection —
it only ever sees text (question + schema string) and produces text
(a SQL string); execution is entirely the MCP tool's job.

---

## Training

`train.py` is the single entry point for both modes.

### Pretrain from scratch

Train on the Spider text-to-SQL dataset:

```bash
uv run python -m sql_model.train pretrain --epochs 10 --batch-size 32
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 3e-4 | Peak learning rate (linear warmup then constant) |
| `--checkpoint-dir` | `sql_model/checkpoints` | Where to save checkpoints |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--save-every` | 5 | Save a periodic snapshot every N epochs |
| `--resume` | — | Path to a checkpoint to load weights from before training |
| `--patience` | 3 | Stop early after this many epochs with no loss improvement |
| `--no-warmup` | — | Disable LR warmup, train at constant LR from step 0 |

### Fine-tune on a production schema

Adapt a pretrained checkpoint to a specific database by generating synthetic training data from its schema:

```bash
uv run python -m sql_model.train finetune \
    --checkpoint sql_model/checkpoints/best_model.pt \
    --schema-file path/to/schema.json
```

The schema file must be a JSON array with `TABLE_SCHEMA`, `TABLE_NAME`, and `COLUMN_NAME` fields — the output of an `INFORMATION_SCHEMA.COLUMNS` query. You can generate it by asking the agent to run:

```sql
SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA NOT IN ('information_schema','mysql','performance_schema','sys')
ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Pretrained checkpoint to start from |
| `--schema-file` | (required) | JSON file with INFORMATION_SCHEMA output |
| `--output` | `sql_model/checkpoints/finetuned.pt` | Output checkpoint path |
| `--epochs` | 10 | Fine-tuning epochs |
| `--lr` | 3e-5 | Learning rate (lower than pretraining) |
| `--batch-size` | 16 | Batch size |
| `--device` | `cpu` | `cpu` or `cuda` |

### Checkpoints

`best_model.pt` is overwritten whenever training loss improves. Periodic snapshots are saved as new files with a run timestamp so multiple runs don't collide:

```
sql_model/checkpoints/
  best_model.pt                          ← always the best pretrain loss seen
  checkpoint_20260531_143012_epoch_5.pt  ← periodic snapshot
  finetuned.pt                           ← best fine-tuned checkpoint
```

### Resuming pretraining

Load weights from any checkpoint before starting a new run:

```bash
uv run python -m sql_model.train pretrain --epochs 10 --resume sql_model/checkpoints/best_model.pt
```

The epoch counter always starts at 1; `--epochs` controls how many epochs the new run trains for.

### Early stopping

Pretraining stops automatically when loss does not improve for `--patience` consecutive epochs (default: 3). To disable, set `--patience` higher than `--epochs`:

```bash
uv run python -m sql_model.train pretrain --epochs 10 --patience 20
```

---

## Inference

### Via the agentic REPL (recommended)

Set `"provider": "generic"` in `mcp_servers_config.json` and run:

```bash
uv run host.py
```

The provider will:
1. Auto-fetch your MySQL schema via the `mysql__execute_query` MCP tool
2. Classify intent and route to template engine or neural model
3. Execute the generated SQL via MySQL MCP and return the results

### Direct inference from Python

```python
import torch
from sql_model.trainer import Trainer
from sql_model.tokenizer import BOS_ID, EOS_ID

# Load checkpoint
model, tokenizer = Trainer.load("sql_model/checkpoints/best_model.pt")
model.eval()

# Encode input
question = "how many orders were placed last month"
schema = "orders(id, customer_id, amount, created_at)"
src_text = f"{question} [SEP] {schema}"
src_ids = tokenizer.encode(src_text)
src = torch.tensor([src_ids])

# Generate SQL
generated_ids = model.generate(src, bos_id=BOS_ID, eos_id=EOS_ID, max_len=128)
sql = tokenizer.decode(generated_ids)
print(sql)
```

---

## Adding New Intents

1. Add the intent name to `INTENTS` list in `intent_classifier.py`
2. Add seed phrases to `_INTENT_SEEDS` dict in `intent_classifier.py`
3. Add a SQL template to the `_TEMPLATES` dict in `template_engine.py`
4. Add the slot-filling logic to `TemplateEngine.generate()` in `template_engine.py`

---

## Checkpoint Format

Checkpoints are `.pt` files (PyTorch save format) containing:

```python
{
    "epoch": int,                      # epoch this checkpoint was saved at
    "loss": float,                     # training loss at save time
    "model": model.state_dict(),       # model weights
    "optimizer": optimizer.state_dict(), # optimizer state (for resume)
    "tokenizer": tokenizer.token2id,   # vocabulary mapping
}
```
