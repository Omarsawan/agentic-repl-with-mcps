import os
import torch
import torch.nn as nn

from .model import Seq2SeqTransformer
from .tokenizer import SQLTokenizer, PAD_ID


class Trainer:
    def __init__(
        self,
        model: Seq2SeqTransformer,
        tokenizer: SQLTokenizer,
        checkpoint_dir: str = "sql_model/checkpoints",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

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
            self.optimizer.step()

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
            },
            path,
        )

    def resume(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=next(self.model.parameters()).device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> tuple["Seq2SeqTransformer", "SQLTokenizer"]:
        checkpoint = torch.load(path, map_location=device)

        token2id: dict = checkpoint["tokenizer"]
        tokenizer = SQLTokenizer()
        tokenizer.token2id = token2id
        tokenizer.id2token = {v: k for k, v in token2id.items()}

        vocab_size = len(token2id)
        model = Seq2SeqTransformer(vocab_size=vocab_size)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        return model, tokenizer
