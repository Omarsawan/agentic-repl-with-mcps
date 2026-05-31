import argparse
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader

from .dataset import SpiderDataset, build_tokenizer_from_spider, collate_fn
from .model import Seq2SeqTransformer
from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the text-to-SQL seq2seq transformer.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="sql_model/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on")
    parser.add_argument("--save-every", type=int, default=5, help="Save a periodic checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume weights from")
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs with no loss improvement")
    args = parser.parse_args()

    print("Loading Spider dataset and building tokenizer...")
    tokenizer = build_tokenizer_from_spider()

    train_dataset = SpiderDataset(split="train", tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Seq2SeqTransformer(vocab_size=tokenizer.vocab_size)
    model.to(args.device)

    trainer = Trainer(model, tokenizer, checkpoint_dir=args.checkpoint_dir)

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
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} consecutive epochs.")
                break

        if epoch % args.save_every == 0:
            snap_name = f"checkpoint_{run_ts}_epoch_{epoch}.pt"
            trainer.save(name=snap_name, epoch=epoch, loss=avg_loss)
            print(f"  Saved periodic checkpoint: {snap_name}")

    print(f"Training complete. Best checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
