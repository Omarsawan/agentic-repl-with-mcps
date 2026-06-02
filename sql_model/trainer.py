import logging
import os

import torch
import torch.nn as nn

from .model import Seq2SeqTransformer
from .tokenizer import SQLTokenizer, PAD_ID

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: Seq2SeqTransformer,
        tokenizer: SQLTokenizer,
        checkpoint_dir: str = "sql_model/checkpoints",
        lr: float = 3e-4,
        total_steps: int = 0,
        warmup_ratio: float = 0.1,
        max_src_len: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        # Linear warmup for first warmup_ratio of steps, then constant LR
        if total_steps > 0:
            warmup_steps = max(1, int(total_steps * warmup_ratio))
            def _lr_lambda(step: int) -> float:
                return min(1.0, step / warmup_steps)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        else:
            self.scheduler = None

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            src = batch["src"]
            tgt_in = batch["tgt_in"]
            tgt_out = batch["tgt_out"]

            self.optimizer.zero_grad()
            logits = self.model(src, tgt_in)  # (B, T, V)

            B, T, V = logits.shape
            loss = self.criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_batches += 1

        return total_loss / total_batches if total_batches > 0 else 0.0

    def save(self, name: str = "best_model.pt", epoch: int = 0, loss: float = float("inf")) -> None:
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "tokenizer": self.tokenizer.token2id,
                "max_src_len": self.max_src_len,
            },
            path,
        )

    def resume(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=next(self.model.parameters()).device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> tuple["Seq2SeqTransformer", "SQLTokenizer", int]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        token2id: dict = checkpoint["tokenizer"]
        tokenizer = SQLTokenizer()
        tokenizer.token2id = token2id
        tokenizer.id2token = {v: k for k, v in token2id.items()}

        vocab_size = len(token2id)
        model = Seq2SeqTransformer(vocab_size=vocab_size)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        max_src_len: int = checkpoint.get("max_src_len", 256)
        logger.info(
            "Loaded checkpoint: vocab_size=%d, max_src_len=%d, epoch=%s, loss=%s",
            vocab_size,
            max_src_len,
            checkpoint.get("epoch", "?"),
            f"{checkpoint['loss']:.4f}" if "loss" in checkpoint else "?",
        )
        return model, tokenizer, max_src_len
