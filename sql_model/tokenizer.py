import re
from pathlib import Path
import json

SQL_KEYWORDS = [
    "select", "from", "where", "group", "by", "order", "limit", "having",
    "join", "inner", "left", "right", "outer", "on", "as", "distinct",
    "count", "sum", "avg", "max", "min", "and", "or", "not", "in", "like",
    "is", "null", "between", "case", "when", "then", "else", "end",
    "insert", "update", "delete", "into", "values", "set", "create", "drop",
    "asc", "desc", "union", "all", "exists", "with",
    "(", ")", ",", ".", "*", "=", "!=", "<", ">", "<=", ">=", "'", "\"",
]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "[SEP]"]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SEP_ID = 4


class SQLTokenizer:
    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token2id[tok] = i
            self.id2token[i] = tok

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+|[^\w\s]", text)
        return tokens

    def build_vocab(self, sentences: list[str]) -> None:
        freq: dict[str, int] = {}
        for sent in sentences:
            for tok in self._tokenize(sent):
                freq[tok] = freq.get(tok, 0) + 1

        for kw in SQL_KEYWORDS:
            if kw not in self.token2id:
                idx = len(self.token2id)
                self.token2id[kw] = idx
                self.id2token[idx] = kw

        for tok, cnt in sorted(freq.items(), key=lambda x: -x[1]):
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

    def encode(self, text: str) -> list[int]:
        return [self.token2id.get(t, UNK_ID) for t in self._tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        # Suppress spaces around punctuation so count(*) and schema.table render correctly.
        NO_SPACE_BEFORE = {")", ".", ",", ";", "("}
        NO_SPACE_AFTER = {"(", "."}

        tokens = []
        for i in ids:
            tok = self.id2token.get(i, "<unk>")
            if tok in ("<pad>", "<bos>", "<eos>"):
                continue
            tokens.append(tok)

        if not tokens:
            return ""

        parts = [tokens[0]]
        for tok in tokens[1:]:
            if tok in NO_SPACE_BEFORE or parts[-1] in NO_SPACE_AFTER:
                parts.append(tok)
            else:
                parts.append(" ")
                parts.append(tok)
        return "".join(parts)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"token2id": self.token2id}))

    @classmethod
    def load(cls, path: str) -> "SQLTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls()
        tok.token2id = data["token2id"]
        tok.id2token = {int(v): k for k, v in data["token2id"].items()}
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)
