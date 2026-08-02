import numpy as np
import pytest

from ecomsearch.bm25 import BM25Index


def test_search_returns_best_keyword_match_first():
    texts = [
        "organic almond milk unsweetened",
        "wireless bluetooth headphones noise cancelling",
        "store brand paper towels six rolls",
    ]
    item_ids = np.array([101, 202, 303])

    index = BM25Index()
    index.build(texts, item_ids)

    results = index.search("almond milk", top_k=2)

    assert results[0][0] == 101


def test_save_and_load_round_trip(tmp_path):
    texts = ["organic almond milk", "wireless bluetooth headphones"]
    item_ids = np.array([11, 22])

    index = BM25Index()
    index.build(texts, item_ids)

    path = tmp_path / "bm25.pkl"
    index.save(path)

    loaded = BM25Index.load(path)
    results = loaded.search("almond milk", top_k=1)

    assert results[0][0] == 11


def test_build_rejects_mismatched_lengths():
    index = BM25Index()
    with pytest.raises(ValueError):
        index.build(["one text", "two text"], np.array([1]))
