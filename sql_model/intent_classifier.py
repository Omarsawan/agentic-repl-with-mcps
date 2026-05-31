from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

INTENTS = ["COUNT", "SUM", "AVG", "GROUP_BY", "TOP_N", "FILTER", "JOIN", "TIME_SERIES", "DISTINCT"]

_INTENT_SEEDS: dict[str, list[str]] = {
    "COUNT": [
        "how many", "count the number", "total number of", "number of records",
        "how many rows", "count of", "tally", "occurrences",
    ],
    "SUM": [
        "total amount", "sum of", "total revenue", "total sales", "add up",
        "cumulative", "aggregate total", "total value",
    ],
    "AVG": [
        "average", "mean", "avg", "average value", "typical",
        "arithmetic mean", "average amount", "average price",
    ],
    "GROUP_BY": [
        "group by", "grouped by", "per category", "by each", "broken down by",
        "per region", "by department", "for each", "segmented by",
    ],
    "TOP_N": [
        "top 10", "top 5", "highest", "best", "top performers",
        "most", "largest", "biggest", "rank", "leading",
    ],
    "FILTER": [
        "where", "filter", "only", "having", "condition", "greater than",
        "less than", "between", "exclude", "include only", "that match",
    ],
    "JOIN": [
        "join", "combined with", "related", "linked", "associated",
        "from multiple tables", "along with", "together with",
    ],
    "TIME_SERIES": [
        "over time", "trend", "by month", "by day", "by year",
        "last month", "last week", "last year", "daily", "monthly", "weekly",
        "time series", "historical", "over the past",
    ],
    "DISTINCT": [
        "unique", "distinct", "different", "how many unique",
        "deduplicated", "no duplicates", "distinct values",
    ],
}


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    entities: dict = field(default_factory=dict)


class IntentClassifier:
    def __init__(self):
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        self._centroids: np.ndarray | None = None
        self._intent_order: list[str] = []
        self._fit()

    def _fit(self) -> None:
        docs = []
        labels = []
        for intent, seeds in _INTENT_SEEDS.items():
            for seed in seeds:
                docs.append(seed)
                labels.append(intent)

        X = self._vectorizer.fit_transform(docs)
        self._intent_order = INTENTS
        centroids = []
        for intent in INTENTS:
            idxs = [i for i, l in enumerate(labels) if l == intent]
            centroid = X[idxs].mean(axis=0)
            centroids.append(np.asarray(centroid))
        self._centroids = np.vstack(centroids)

    def _extract_entities(self, text: str) -> dict:
        entities: dict = {}
        low = text.lower()

        m = re.search(r"\btop\s+(\d+)\b", low)
        if m:
            entities["limit"] = int(m.group(1))

        m = re.search(r"\blast\s+(\d+)\s+(day|week|month|year)s?\b", low)
        if m:
            entities["time_filter"] = f"{m.group(1)} {m.group(2)}s"

        return entities

    def classify(self, text: str) -> ClassificationResult:
        vec = self._vectorizer.transform([text.lower()])
        sims = cosine_similarity(vec, self._centroids)[0]
        best_idx = int(np.argmax(sims))
        confidence = float(sims[best_idx])
        intent = self._intent_order[best_idx]
        entities = self._extract_entities(text)
        return ClassificationResult(intent=intent, confidence=confidence, entities=entities)
