"""Real end-to-end round trip against the actual Qdrant Cloud cluster (no mocking)."""

import numpy as np
import pytest

from ecomsearch.config import QDRANT_URL
from ecomsearch.qdrant_index import QdrantIndex

pytestmark = pytest.mark.skipif(not QDRANT_URL, reason="QDRANT_URL not configured")

TEST_COLLECTION_NAME = "ecomsearch_qdrant_index_test"


def test_create_upsert_search_round_trip_against_real_cluster():
    index = QdrantIndex(TEST_COLLECTION_NAME)
    index.create_collection(dim=4)

    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="float32")
    item_ids = np.array([101, 202])
    index.upsert(vectors, item_ids)

    # A free-tier Qdrant Cloud cluster auto-suspends after inactivity (see the
    # README's Known Limitations) -- the first request after a long idle period
    # can be slow to wake it. Retry once before failing if the first attempt
    # doesn't return the expected top result.
    results = index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), top_k=2)
    if not results or results[0][0] != 101:
        results = index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), top_k=2)

    assert results[0][0] == 101
    assert results[0][1] > results[1][1]

    index._client.delete_collection(TEST_COLLECTION_NAME)
