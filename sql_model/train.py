import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SpiderDataset, build_tokenizer_from_spider, collate_fn
from .finetune_data import SyntheticDataset, generate_pairs, parse_schema_file
from .model import Seq2SeqTransformer
from .trainer import Trainer

logger = logging.getLogger(__name__)


def _extend_vocab(model: Seq2SeqTransformer, old_size: int, new_size: int) -> None:
    """Grow embedding and projection layers in-place when vocab is extended."""
    if new_size <= old_size:
        return
    d = model.d_model
    device = next(model.parameters()).device
    for attr in ("src_embed", "tgt_embed"):
        old_emb = getattr(model, attr)
        new_emb = nn.Embedding(new_size, d, padding_idx=old_emb.padding_idx)
        new_emb.weight.data[:old_size] = old_emb.weight.data
        setattr(model, attr, new_emb.to(device))
    old_proj = model.proj
    new_proj = nn.Linear(d, new_size)
    new_proj.weight.data[:old_size] = old_proj.weight.data
    new_proj.bias.data[:old_size] = old_proj.bias.data
    model.proj = new_proj.to(device)


def run_pretrain(args) -> None:
    print("Loading Spider dataset and building tokenizer...")
    tokenizer = build_tokenizer_from_spider()

    train_dataset = SpiderDataset(split="train", tokenizer=tokenizer)

    print("\nDataset sample preview:")
    for src, tgt in train_dataset.sample_preview():
        print(f"  Q: {src[:512]}")
        print(f"  SQL: {tgt}")
        print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    total_steps = args.epochs * len(train_loader)
    model = Seq2SeqTransformer(vocab_size=tokenizer.vocab_size)
    model.to(args.device)

    trainer = Trainer(
        model,
        tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        total_steps=0 if args.no_warmup else total_steps,
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume:
        trainer.resume(args.resume)
        print(f"Loaded weights from {args.resume}")

    best_loss = float("inf")
    best_path = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch}/{args.epochs} — loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            trainer.save(name="best_model.pt", epoch=epoch, loss=avg_loss)
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            print(f"  -> New best checkpoint saved (loss={best_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} consecutive epochs.")
                break

        if epoch % args.save_every == 0:
            snap_name = f"checkpoint_{run_ts}_epoch_{epoch}.pt"
            trainer.save(name=snap_name, epoch=epoch, loss=avg_loss)
            print(f"  Saved periodic checkpoint: {snap_name}")

    print(f"\nTraining complete. Best checkpoint: {best_path}")


def run_finetune(args) -> None:
    print(f"Loading checkpoint from {args.checkpoint}")
    model, tokenizer, _ = Trainer.load(args.checkpoint, device=args.device)
    model.train()

    print(f"Parsing schema from {args.schema_file}")
    schema = parse_schema_file(args.schema_file)
    print(f"Schema: {len(schema)} tables")

    pairs = generate_pairs(schema)
    if not pairs:
        print("ERROR: No training pairs generated — check that schema file is non-empty")
        return

    print("Sample fine-tuning pairs:")
    for pair in random.sample(pairs, min(3, len(pairs))):
        print(f"  src: {pair['src_text'][:120]}")
        print(f"  tgt: {pair['tgt_text']}")

    old_vocab_size = tokenizer.vocab_size
    all_texts = [p["src_text"] for p in pairs] + [p["tgt_text"] for p in pairs]
    tokenizer.build_vocab(all_texts)
    new_vocab_size = tokenizer.vocab_size
    if new_vocab_size > old_vocab_size:
        print(f"Vocabulary extended: {old_vocab_size} → {new_vocab_size} tokens (+{new_vocab_size - old_vocab_size} new)")
        _extend_vocab(model, old_vocab_size, new_vocab_size)
    else:
        print(f"Vocabulary unchanged: {old_vocab_size} tokens (all schema tokens already in vocab)")

    print("Vocabulary coverage check on sample pairs (UNK=3 means OOV):")
    from .tokenizer import UNK_ID
    for pair in random.sample(pairs, min(5, len(pairs))):
        src_ids = tokenizer.encode(pair["src_text"])
        unk = sum(1 for t in src_ids if t == UNK_ID)
        print(f"  OOV {unk}/{len(src_ids)} ({100*unk//max(1,len(src_ids))}%)  src: {pair['src_text'][:100]}")

    dataset = SyntheticDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    output_path = Path(args.output)
    total_steps = args.epochs * len(loader)
    trainer = Trainer(model, tokenizer, checkpoint_dir=str(output_path.parent), lr=args.lr, total_steps=total_steps)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch(loader)
        print(f"Epoch {epoch}/{args.epochs} — loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save(name=output_path.name, epoch=epoch, loss=avg_loss)

    print(f"\nFine-tuning complete. Output: {args.output}  (best loss: {best_loss:.4f})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="SQL seq2seq model training")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── pretrain ──────────────────────────────────────────────────────────────
    pre = sub.add_parser("pretrain", help="Pretrain from scratch on the Spider dataset")
    pre.add_argument("--epochs", type=int, default=5)
    pre.add_argument("--batch-size", type=int, default=32)
    pre.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (with warmup)")
    pre.add_argument("--checkpoint-dir", type=str, default="sql_model/checkpoints")
    pre.add_argument("--device", type=str, default="cpu")
    pre.add_argument("--save-every", type=int, default=5, help="Save a periodic checkpoint every N epochs")
    pre.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    pre.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    pre.add_argument("--no-warmup", action="store_true", help="Disable LR warmup, train at constant LR")

    # ── finetune ──────────────────────────────────────────────────────────────
    ft = sub.add_parser("finetune", help="Fine-tune a checkpoint on a production schema")
    ft.add_argument("--checkpoint", required=True, help="Pretrained checkpoint to start from")
    ft.add_argument("--schema-file", required=True, help="JSON file with INFORMATION_SCHEMA output")
    ft.add_argument("--output", default="sql_model/checkpoints/finetuned.pt")
    ft.add_argument("--epochs", type=int, default=10)
    ft.add_argument("--lr", type=float, default=3e-5)
    ft.add_argument("--batch-size", type=int, default=16)
    ft.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    {"pretrain": run_pretrain, "finetune": run_finetune}[args.command](args)


if __name__ == "__main__":
    main()
