"""FAISS-backed nearest neighbor index over product embeddings."""

from pathlib import Path

import faiss
import numpy as np


class ProductIndex:
    def __init__(self, dim: int):
        self._index = faiss.IndexFlatIP(dim)
        self._item_ids: np.ndarray = np.empty((0,), dtype="int64")

    @property
    def item_ids(self) -> np.ndarray:
        return self._item_ids

    def add(self, vectors: np.ndarray, item_ids: np.ndarray) -> None:
        if vectors.shape[0] != item_ids.shape[0]:
            raise ValueError("vectors and item_ids must have the same length")
        self._index.add(vectors.astype("float32"))
        self._item_ids = np.concatenate([self._item_ids, item_ids.astype("int64")])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        query = np.expand_dims(query_vector.astype("float32"), axis=0)
        scores, positions = self._index.search(query, top_k)
        results = []
        for position, score in zip(positions[0], scores[0]):
            if position == -1:
                continue
            results.append((int(self._item_ids[position]), float(score)))
        return results

    def save(self, index_path: Path, item_ids_path: Path) -> None:
        faiss.write_index(self._index, str(index_path))
        np.save(item_ids_path, self._item_ids)

    @classmethod
    def load(cls, index_path: Path, item_ids_path: Path) -> "ProductIndex":
        index = faiss.read_index(str(index_path))
        instance = cls(dim=index.d)
        instance._index = index
        instance._item_ids = np.load(item_ids_path)
        return instance
