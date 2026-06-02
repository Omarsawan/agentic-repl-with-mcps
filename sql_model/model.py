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
        beam_size: int = 4,
    ) -> list[int]:
        self.eval()
        src_pad_mask = self._src_key_padding_mask(src)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # Each beam: (cumulative_log_prob, token_ids, tgt_tensor)
        beams: list[tuple[float, list[int], Tensor]] = [
            (0.0, [], torch.tensor([[bos_id]], device=src.device))
        ]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            if not beams:
                break
            candidates: list[tuple[float, list[int], Tensor]] = []
            for log_prob, tokens, tgt in beams:
                tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt.size(1), device=src.device
                )
                out = self.transformer.decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_pad_mask,
                )
                log_probs = torch.log_softmax(self.proj(out[:, -1, :]), dim=-1)[0]
                top_lp, top_ids = log_probs.topk(beam_size)
                for lp, idx in zip(top_lp.tolist(), top_ids.tolist()):
                    new_lp = log_prob + lp
                    new_tokens = tokens + [idx]
                    new_tgt = torch.cat(
                        [tgt, torch.tensor([[idx]], device=src.device)], dim=1
                    )
                    if idx == eos_id:
                        completed.append((new_lp, tokens))  # exclude eos token
                    else:
                        candidates.append((new_lp, new_tokens, new_tgt))
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]
        if beams:
            beams.sort(key=lambda x: x[0], reverse=True)
            return beams[0][1]
        return []
