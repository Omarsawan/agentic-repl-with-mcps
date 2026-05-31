import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(d_model, vocab_size)

    def _src_key_padding_mask(self, src: Tensor) -> Tensor:
        return src == self.pad_id  # (B, S)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_pad_mask = self._src_key_padding_mask(src)
        tgt_pad_mask = self._src_key_padding_mask(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=src.device
        )

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.proj(out)  # (B, T, vocab_size)

    @torch.inference_mode()
    def generate(
        self,
        src: Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> list[int]:
        self.eval()
        src_pad_mask = self._src_key_padding_mask(src)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        tgt = torch.tensor([[bos_id]], device=src.device)
        generated: list[int] = []

        for _ in range(max_len):
            tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=src.device
            )
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.proj(out[:, -1, :])
            next_id = logits.argmax(-1).item()
            if next_id == eos_id:
                break
            generated.append(next_id)
            tgt = torch.cat([tgt, torch.tensor([[next_id]], device=src.device)], dim=1)

        return generated
