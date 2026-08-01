import numpy as np
import pytest

from ecomsearch.index import ProductIndex


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_search_returns_nearest_neighbor_first():
    vectors = _normalize(np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ], dtype="float32"))
    item_ids = np.array([101, 202, 303])

    index = ProductIndex(dim=2)
    index.add(vectors, item_ids)

    query = _normalize(np.array([[1.0, 0.0]], dtype="float32"))[0]
    results = index.search(query, top_k=2)

    assert results[0][0] == 101
    assert results[1][0] == 303


def test_add_rejects_mismatched_lengths():
    index = ProductIndex(dim=2)
    with pytest.raises(ValueError):
        index.add(np.zeros((2, 2), dtype="float32"), np.array([1]))


def test_save_and_load_round_trip(tmp_path):
    vectors = _normalize(np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype="float32"))
    item_ids = np.array([11, 22])

    index = ProductIndex(dim=2)
    index.add(vectors, item_ids)

    index_path = tmp_path / "catalog.faiss"
    item_ids_path = tmp_path / "item_ids.npy"
    index.save(index_path, item_ids_path)

    loaded = ProductIndex.load(index_path, item_ids_path)
    results = loaded.search(vectors[0], top_k=1)

    assert results[0][0] == 11
