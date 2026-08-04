import numpy as np
import pytest

from ecomsearch import qdrant_index


class FakeScoredPoint:
    def __init__(self, id, score):
        self.id = id
        self.score = score


class FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    def __init__(self, url, api_key):
        self.url = url
        self.api_key = api_key
        self.collections = {}
        self.upserted_points = []

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = vectors_config

    def upsert(self, collection_name, points):
        self.upserted_points.extend(points)

    def query_points(self, collection_name, query, limit):
        canned = [FakeScoredPoint(101, 0.9), FakeScoredPoint(202, 0.5)]
        return FakeQueryResponse(canned[:limit])


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    monkeypatch.setattr(qdrant_index, "QdrantClient", FakeQdrantClient)


def test_create_collection_creates_with_correct_dimension():
    index = qdrant_index.QdrantIndex("test_collection")
    index.create_collection(dim=4)

    assert "test_collection" in index._client.collections
    assert index._client.collections["test_collection"].size == 4


def test_create_collection_replaces_an_existing_collection():
    index = qdrant_index.QdrantIndex("test_collection")
    index.create_collection(dim=4)
    index.create_collection(dim=8)

    assert index._client.collections["test_collection"].size == 8


def test_upsert_sends_points_with_item_id_as_point_id():
    index = qdrant_index.QdrantIndex("test_collection")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    item_ids = np.array([101, 202])

    index.upsert(vectors, item_ids)

    sent_ids = [p.id for p in index._client.upserted_points]
    assert sent_ids == [101, 202]


def test_search_returns_item_id_score_tuples():
    index = qdrant_index.QdrantIndex("test_collection")

    results = index.search(np.array([1.0, 0.0], dtype="float32"), top_k=2)

    assert results == [(101, 0.9), (202, 0.5)]
