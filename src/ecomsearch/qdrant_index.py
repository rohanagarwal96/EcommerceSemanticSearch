"""Qdrant Cloud-backed nearest neighbor index over product embeddings."""

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ecomsearch.config import QDRANT_API_KEY, QDRANT_URL


class QdrantIndex:
    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    def create_collection(self, dim: int) -> None:
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, vectors: np.ndarray, item_ids: np.ndarray) -> None:
        if vectors.shape[0] != item_ids.shape[0]:
            raise ValueError("vectors and item_ids must have the same length")
        points = [
            PointStruct(id=int(item_id), vector=vector.astype("float32").tolist())
            for vector, item_id in zip(vectors, item_ids)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector.astype("float32").tolist(),
            limit=top_k,
        )
        return [(int(point.id), float(point.score)) for point in response.points]
