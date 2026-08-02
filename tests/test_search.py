import numpy as np
import pandas as pd
import pytest

from ecomsearch import search
from ecomsearch.bm25 import BM25Index
from ecomsearch.index import ProductIndex


@pytest.fixture
def synthetic_catalog(tmp_path, monkeypatch, embedder):
    texts = [
        "Organic almond milk unsweetened dairy free beverage",
        "Wireless bluetooth headphones noise cancelling",
        "Store brand paper towels six rolls",
    ]
    item_ids = np.array([101, 202, 303])

    vectors = embedder.embed_documents(texts)
    dense_index = ProductIndex(dim=vectors.shape[1])
    dense_index.add(vectors, item_ids)
    index_path = tmp_path / "catalog.faiss"
    item_ids_path = tmp_path / "item_ids.npy"
    dense_index.save(index_path, item_ids_path)

    bm25_index = BM25Index()
    bm25_index.build(texts, item_ids)
    bm25_path = tmp_path / "bm25.pkl"
    bm25_index.save(bm25_path)

    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame({"item_id": item_ids, "search_text": texts}).to_csv(catalog_path, index=False)

    monkeypatch.setattr(search, "INDEX_PATH", index_path)
    monkeypatch.setattr(search, "ITEM_IDS_PATH", item_ids_path)
    monkeypatch.setattr(search, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(search, "CATALOG_PATH", catalog_path)

    return item_ids


def test_dense_search_returns_best_semantic_match(synthetic_catalog):
    results = search.dense_search("organic dairy free milk", top_k=1)
    assert results[0][0] == 101


def test_bm25_search_returns_best_keyword_match(synthetic_catalog):
    results = search.bm25_search("almond milk", top_k=1)
    assert results[0][0] == 101


def test_hybrid_search_without_rerank_returns_fused_top_result(synthetic_catalog):
    results = search.hybrid_search("almond milk", top_k=1, use_rerank=False)
    assert results[0][0] == 101


def test_hybrid_search_with_rerank_returns_relevant_top_result(synthetic_catalog):
    results = search.hybrid_search("almond milk", top_k=1, use_rerank=True)
    assert results[0][0] == 101


def test_dense_search_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "INDEX_PATH", tmp_path / "catalog.faiss")
    monkeypatch.setattr(search, "ITEM_IDS_PATH", tmp_path / "item_ids.npy")

    with pytest.raises(SystemExit) as excinfo:
        search.dense_search("anything", top_k=1)

    assert "build_index.py" in str(excinfo.value)


def test_bm25_search_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "BM25_INDEX_PATH", tmp_path / "bm25.pkl")

    with pytest.raises(SystemExit) as excinfo:
        search.bm25_search("anything", top_k=1)

    assert "build_bm25_index.py" in str(excinfo.value)
