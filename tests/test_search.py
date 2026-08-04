import numpy as np
import pandas as pd
import pytest

from ecomsearch import search
from ecomsearch.bm25 import BM25Index
from ecomsearch.index import ProductIndex
from ecomsearch.embeddings import Embedder
from ecomsearch.reranker import CrossEncoderReranker


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


@pytest.fixture(autouse=True)
def reset_search_caches(monkeypatch):
    monkeypatch.setattr(search, "_dense_index", None, raising=False)
    monkeypatch.setattr(search, "_bm25_index", None, raising=False)
    monkeypatch.setattr(search, "_embedder", None, raising=False)
    monkeypatch.setattr(search, "_reranker", None, raising=False)
    monkeypatch.setattr(search, "_catalog", None, raising=False)


def test_dense_search_loads_index_and_embedder_only_once_across_calls(
    synthetic_catalog, monkeypatch
):
    load_calls = []
    original_load = ProductIndex.load.__func__

    def counting_load(cls, *args, **kwargs):
        load_calls.append(1)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(ProductIndex, "load", classmethod(counting_load))

    init_calls = []
    original_init = Embedder.__init__

    def counting_init(self, *args, **kwargs):
        init_calls.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Embedder, "__init__", counting_init)

    search.dense_search("almond milk", top_k=1)
    search.dense_search("paper towels", top_k=1)

    assert len(load_calls) == 1
    assert len(init_calls) == 1


def test_bm25_search_loads_index_only_once_across_calls(synthetic_catalog, monkeypatch):
    load_calls = []
    original_load = BM25Index.load.__func__

    def counting_load(cls, *args, **kwargs):
        load_calls.append(1)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(BM25Index, "load", classmethod(counting_load))

    search.bm25_search("almond milk", top_k=1)
    search.bm25_search("paper towels", top_k=1)

    assert len(load_calls) == 1


def test_load_dense_index_returns_qdrant_index_when_backend_is_qdrant(monkeypatch):
    monkeypatch.setattr(search, "VECTOR_BACKEND", "qdrant")

    class FakeQdrantIndex:
        def __init__(self, collection_name):
            self.collection_name = collection_name

    monkeypatch.setattr(search, "QdrantIndex", FakeQdrantIndex)

    index = search.load_dense_index()

    assert isinstance(index, FakeQdrantIndex)
    assert index.collection_name == search.QDRANT_COLLECTION_NAME


def test_hybrid_search_with_rerank_loads_reranker_and_catalog_only_once_across_calls(
    synthetic_catalog, monkeypatch
):
    init_calls = []
    original_init = CrossEncoderReranker.__init__

    def counting_init(self, *args, **kwargs):
        init_calls.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(CrossEncoderReranker, "__init__", counting_init)

    read_csv_calls = []
    original_read_csv = search.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        read_csv_calls.append(1)
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(search.pd, "read_csv", counting_read_csv)

    search.hybrid_search("almond milk", top_k=1, use_rerank=True)
    search.hybrid_search("paper towels", top_k=1, use_rerank=True)

    assert len(init_calls) == 1
    assert len(read_csv_calls) == 1
