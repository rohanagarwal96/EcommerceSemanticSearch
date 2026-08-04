"""Real end-to-end verification that dense_search/image_search work against the
actual Qdrant Cloud cluster with production-scale data (populated by
scripts/upload_index_to_qdrant.py and scripts/upload_multimodal_index_to_qdrant.py).
Skipped if Qdrant Cloud credentials aren't configured.
"""
import pytest

from ecomsearch import search
from ecomsearch.config import QDRANT_URL
from ecomsearch.multimodal import search as multimodal_search

pytestmark = pytest.mark.skipif(not QDRANT_URL, reason="QDRANT_URL not configured")

KNOWN_RELEVANT_ITEM_IDS_FOR_ALMOND_MILK = {
    92137, 92144, 92585, 92641, 92671, 92700, 93002, 98504, 98505,
    952903, 954673, 954690, 1163175, 1859122, 2026646,
}


@pytest.fixture(autouse=True)
def qdrant_backend(monkeypatch):
    monkeypatch.setattr(search, "VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(multimodal_search, "VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(search, "_dense_index", None, raising=False)
    monkeypatch.setattr(multimodal_search, "_index", None, raising=False)


def test_dense_search_returns_relevant_result_from_qdrant():
    results = search.dense_search("organic almond milk", top_k=5)

    result_ids = {item_id for item_id, _ in results}
    assert result_ids & KNOWN_RELEVANT_ITEM_IDS_FOR_ALMOND_MILK


def test_image_search_returns_results_from_qdrant():
    results = multimodal_search.image_search("shoes", top_k=5)

    assert len(results) > 0
