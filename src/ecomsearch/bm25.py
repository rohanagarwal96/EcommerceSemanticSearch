"""BM25 keyword search index over product search_text."""
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self):
        self._bm25 = None
        self._item_ids: np.ndarray = np.empty((0,), dtype="int64")

    @property
    def item_ids(self) -> np.ndarray:
        return self._item_ids

    def build(self, texts: list[str], item_ids: np.ndarray) -> None:
        if len(texts) != len(item_ids):
            raise ValueError("texts and item_ids must have the same length")
        tokenized_corpus = [_tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._item_ids = item_ids.astype("int64")

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [(int(self._item_ids[i]), float(scores[i])) for i in top_indices]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "item_ids": self._item_ids}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._bm25 = data["bm25"]
        instance._item_ids = data["item_ids"]
        return instance
