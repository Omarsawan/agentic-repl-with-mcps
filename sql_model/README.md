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
