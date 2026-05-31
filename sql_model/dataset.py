import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .tokenizer import SQLTokenizer, PAD_ID


class SpiderDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: "SQLTokenizer",
        max_src_len: int = 256,
        max_tgt_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        raw = load_dataset("xlangai/spider", split=split)
        self.examples = []
        for item in raw:
            question = item["question"]
            query = item["query"]
            db_id = item["db_id"]
            src_text = f"{question} [SEP] db:{db_id}"
            self.examples.append({"src_text": src_text, "tgt_text": query})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        src_ids = self.tokenizer.encode(ex["src_text"])
        # Pad / truncate src
        src_ids = src_ids[: self.max_src_len]
        src_ids = src_ids + [PAD_ID] * (self.max_src_len - len(src_ids))

        tgt_ids = self.tokenizer.encode(ex["tgt_text"])

        bos_id = self.tokenizer.token2id.get("<BOS>", 1)
        eos_id = self.tokenizer.token2id.get("<EOS>", 2)

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
    all_texts = []
    for item in raw:
        all_texts.append(item["question"])
        all_texts.append(item["query"])
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
