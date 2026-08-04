import numpy as np
import pytest

from ecomsearch.index import ProductIndex
from ecomsearch.multimodal import search
from ecomsearch.multimodal.clip_embedder import ClipEmbedder


@pytest.fixture(autouse=True)
def reset_image_search_caches(monkeypatch):
    monkeypatch.setattr(search, "_index", None, raising=False)
    monkeypatch.setattr(search, "_embedder", None, raising=False)


@pytest.fixture
def synthetic_image_index(tmp_path, monkeypatch, clip_embedder):
    texts = [
        "a photo of a red bicycle",
        "a photo of a laptop computer",
        "a photo of a wooden chair",
    ]
    item_ids = np.array([501, 502, 503])

    vectors = clip_embedder.embed_text(texts)
    index = ProductIndex(dim=vectors.shape[1])
    index.add(vectors, item_ids)
    index_path = tmp_path / "catalog.faiss"
    item_ids_path = tmp_path / "item_ids.npy"
    index.save(index_path, item_ids_path)

    monkeypatch.setattr(search, "INDEX_PATH", index_path)
    monkeypatch.setattr(search, "ITEM_IDS_PATH", item_ids_path)

    return item_ids


def test_image_search_returns_best_semantic_match(synthetic_image_index):
    results = search.image_search("red bicycle", top_k=1)
    assert results[0][0] == 501


def test_image_search_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "INDEX_PATH", tmp_path / "catalog.faiss")
    monkeypatch.setattr(search, "ITEM_IDS_PATH", tmp_path / "item_ids.npy")

    with pytest.raises(SystemExit) as excinfo:
        search.image_search("anything", top_k=1)

    assert "build_multimodal_index.py" in str(excinfo.value)


def test_image_search_loads_index_and_embedder_only_once_across_calls(
    synthetic_image_index, monkeypatch
):
    load_calls = []
    original_load = ProductIndex.load.__func__

    def counting_load(cls, *args, **kwargs):
        load_calls.append(1)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(ProductIndex, "load", classmethod(counting_load))

    init_calls = []
    original_init = ClipEmbedder.__init__

    def counting_init(self, *args, **kwargs):
        init_calls.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(ClipEmbedder, "__init__", counting_init)

    search.image_search("red bicycle", top_k=1)
    search.image_search("wooden chair", top_k=1)

    assert len(load_calls) == 1
    assert len(init_calls) == 1


def test_load_index_returns_qdrant_index_when_backend_is_qdrant(monkeypatch):
    monkeypatch.setattr(search, "VECTOR_BACKEND", "qdrant")

    class FakeQdrantIndex:
        def __init__(self, collection_name):
            self.collection_name = collection_name

    monkeypatch.setattr(search, "QdrantIndex", FakeQdrantIndex)

    index = search.load_index()

    assert isinstance(index, FakeQdrantIndex)
    assert index.collection_name == search.QDRANT_IMAGE_COLLECTION_NAME
