# Phase 6: Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Phase 5 FastAPI + Streamlit app from localhost-only to a live, free, publicly reachable deployment — dense vector search backed by Qdrant Cloud, large runtime artifacts hosted on a Hugging Face Hub dataset repo, and the backend/frontend each running as a Docker-based Hugging Face Space.

**Architecture:** A new `QdrantIndex` class gives `search.py`/`multimodal/search.py` a drop-in replacement for `ProductIndex`, selected via a `VECTOR_BACKEND` env var so local dev/tests keep using FAISS unchanged. One-time scripts migrate the two local FAISS indexes into two Qdrant Cloud collections and push large artifacts (catalog CSV, BM25 pickle, CLIP subset images) to a Hugging Face Hub dataset repo. Two Dockerfiles package the backend and frontend; the backend downloads its data bundle from the HF dataset repo at container startup. Two deploy scripts push each Docker app to its own Hugging Face Space via the `huggingface_hub` API.

**Tech Stack:** `qdrant-client`, `python-dotenv`, `huggingface_hub`, Docker.

**Spec:** `docs/superpowers/specs/2026-08-03-phase6-deployment-design.md`

**Prerequisites already in place (verified before writing this plan):** a Qdrant Cloud free-tier cluster and a Hugging Face account with a write token, both with real credentials in the gitignored `.env`; two reserved HF Space names (`HF_SPACE_BACKEND`, `HF_SPACE_FRONTEND`); Docker Desktop installed and running locally. The local FAISS indexes already exist: the text index has 55,516 vectors of dimension 384, the CLIP image index has 4,996 vectors of dimension 512 (matching `artifacts/multimodal/subset_metadata.csv`'s 4,996 rows). Both embedders L2-normalize their output before indexing, so Qdrant's `Distance.COSINE` is the correct equivalent of the local `IndexFlatIP` search.

**Note on the design spec's artifact-size estimate:** the spec estimated "~600MB" for the artifact bundle assuming the full CLIP image dataset. In practice, the API only ever serves the 4,996-item stratified subset referenced by `subset_metadata.csv` — the actual bundle (catalog CSV + BM25 pickle + subset images + subset metadata) is closer to **160MB** (70MB catalog + 36MB BM25 + ~51MB of subset images + metadata). Task 9 below uploads only that subset, not the full 548MB local image directory.

---

### Task 1: Add deployment dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append these lines to the end of `requirements.txt`**

```
qdrant-client>=1.18.0
python-dotenv>=1.0.0
huggingface_hub>=0.24.0
```

- [ ] **Step 2: Install and verify**

Run:
```bash
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe -c "import qdrant_client, dotenv, huggingface_hub; print('ok')"
```
Expected: installs succeed, prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add Qdrant Cloud and Hugging Face Hub dependencies"
git push origin main
```

---

### Task 2: Config additions for Qdrant and Hugging Face Hub

**Files:**
- Modify: `src/ecomsearch/config.py`
- Modify: `src/ecomsearch/multimodal/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Modify `src/ecomsearch/config.py`**

Add `import os` and the `dotenv` import/call at the top, and the new constants at the bottom. The full file becomes:

```python
"""Shared configuration constants for the ecomsearch package."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = REPO_ROOT / "data" / "ecommerce_catalog_enriched.csv"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "catalog.faiss"
ITEM_IDS_PATH = ARTIFACTS_DIR / "item_ids.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_TOP_K = 10

BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25.pkl"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
CANDIDATE_POOL_SIZE = 100
RERANK_POOL_SIZE = 50

EVAL_QUERIES_PATH = REPO_ROOT / "eval" / "eval_queries.json"
EVAL_RESULTS_PATH = REPO_ROOT / "docs" / "eval_results.md"
EVAL_TOP_K = 10

LATENCY_RESULTS_PATH = REPO_ROOT / "docs" / "latency_results.md"
LATENCY_TARGET_MS_P95 = 200.0
BENCHMARK_REPEAT_COUNT = 10
BENCHMARK_SEED = 42

VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "faiss")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "ecommerce_products")

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
HF_SPACE_BACKEND = os.environ.get("HF_SPACE_BACKEND")
HF_SPACE_FRONTEND = os.environ.get("HF_SPACE_FRONTEND")
```

- [ ] **Step 2: Modify `src/ecomsearch/multimodal/config.py`**

Add `import os` at the top and one new constant at the bottom. The full file becomes:

```python
"""Shared configuration constants for the multimodal (CLIP) module."""

import os
from pathlib import Path

from ecomsearch.config import REPO_ROOT

DATASET_DIR = REPO_ROOT / "data" / "multimodal"
DATASET_CSV_PATH = DATASET_DIR / "data.csv"
DATASET_IMAGES_DIR = DATASET_DIR / "data"

KAGGLE_DATASET_REF = "nirmalsankalana/mini-product-image-and-text-dataset"
KAGGLE_CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "multimodal"
INDEX_PATH = ARTIFACTS_DIR / "catalog.faiss"
ITEM_IDS_PATH = ARTIFACTS_DIR / "item_ids.npy"
SUBSET_METADATA_PATH = ARTIFACTS_DIR / "subset_metadata.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SUBSET_SIZE = 5000
DEFAULT_TOP_K = 10

DEMO_RESULTS_DIR = REPO_ROOT / "demo_results"

QDRANT_IMAGE_COLLECTION_NAME = os.environ.get(
    "QDRANT_IMAGE_COLLECTION_NAME", "ecommerce_products_images"
)
```

- [ ] **Step 3: Modify `.env.example`**

Replace its contents with:

```
# Hugging Face (account + write token, used for model downloads, artifact
# hosting, and Space deployment)
HF_TOKEN=your_huggingface_write_token_here
HF_SPACE_BACKEND=your-hf-username/ecommerce-search-api
HF_SPACE_FRONTEND=your-hf-username/ecommerce-search-ui
HF_DATASET_REPO=your-hf-username/ecommerce-search-artifacts

# Qdrant Cloud (free tier cluster)
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=ecommerce_products
QDRANT_IMAGE_COLLECTION_NAME=ecommerce_products_images

# Vector search backend: "faiss" for local dev/tests (default), "qdrant" for
# production. Only production containers should set this to "qdrant".
VECTOR_BACKEND=faiss

# Kaggle (only needed if not relying on ~/.kaggle/kaggle.json)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key_here
```

Also add the new `HF_DATASET_REPO` and `QDRANT_IMAGE_COLLECTION_NAME` lines to your local (gitignored) `.env`, with real values — `HF_DATASET_REPO` can be any repo id you haven't used yet, e.g. `<your-hf-username>/ecommerce-search-artifacts` (it doesn't need to exist yet; Task 9 creates it). Leave `VECTOR_BACKEND=faiss` in your local `.env` — only the deployed containers set it to `qdrant`.

- [ ] **Step 4: Verify existing tests still pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_search.py tests/test_multimodal_search.py -v`
Expected: all pass unchanged (these config additions don't change any existing behavior).

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/config.py src/ecomsearch/multimodal/config.py .env.example
git commit -m "feat: add Qdrant and Hugging Face Hub config constants"
git push origin main
```

---

### Task 3: QdrantIndex class (TDD)

**Files:**
- Create: `src/ecomsearch/qdrant_index.py`
- Test: `tests/test_qdrant_index.py`
- Test: `tests/test_qdrant_index_integration.py`

`QdrantIndex` exposes the same `search(query_vector, top_k) -> list[(item_id, score)]` shape as `ProductIndex`, so it's a drop-in replacement wherever `search.py`/`multimodal/search.py` call `.search(...)`. It also exposes `create_collection`/`upsert`, used only by the one-time migration scripts in Tasks 6-7.

- [ ] **Step 1: Write the failing unit tests in `tests/test_qdrant_index.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_qdrant_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.qdrant_index'`

- [ ] **Step 3: Write `src/ecomsearch/qdrant_index.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_qdrant_index.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the real-cluster round trip test in `tests/test_qdrant_index_integration.py`**

This is a real, non-mocked test against your actual Qdrant Cloud cluster — it creates a throwaway collection, upserts, queries, and deletes the collection. Qdrant Cloud's free tier isn't metered per request, so this is safe to run regularly (unlike a paid API). It's skipped automatically if Qdrant credentials aren't configured, e.g. on a fresh clone before `.env` is set up.

```python
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
```

- [ ] **Step 6: Run the real test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_qdrant_index_integration.py -v`
Expected: PASS (1 test). This has been manually verified against the actual cluster before writing this plan — a throwaway collection round trip (create, upsert 2 points, query, delete) succeeded, correctly ranking the exact-match vector above the orthogonal one.

- [ ] **Step 7: Commit**

```bash
git add src/ecomsearch/qdrant_index.py tests/test_qdrant_index.py tests/test_qdrant_index_integration.py
git commit -m "feat: add QdrantIndex with TDD unit tests and a real cluster round-trip test"
git push origin main
```

---

### Task 4: Wire QdrantIndex into dense text search (TDD)

**Files:**
- Modify: `src/ecomsearch/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing test in `tests/test_search.py`**

Add this test (anywhere after the existing imports/fixtures):

```python
def test_load_dense_index_returns_qdrant_index_when_backend_is_qdrant(monkeypatch):
    monkeypatch.setattr(search, "VECTOR_BACKEND", "qdrant")

    class FakeQdrantIndex:
        def __init__(self, collection_name):
            self.collection_name = collection_name

    monkeypatch.setattr(search, "QdrantIndex", FakeQdrantIndex)

    index = search.load_dense_index()

    assert isinstance(index, FakeQdrantIndex)
    assert index.collection_name == search.QDRANT_COLLECTION_NAME
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_search.py::test_load_dense_index_returns_qdrant_index_when_backend_is_qdrant -v`
Expected: FAIL with `AttributeError: module 'ecomsearch.search' has no attribute 'VECTOR_BACKEND'` (or similar)

- [ ] **Step 3: Modify `src/ecomsearch/search.py`**

Update the imports and `load_dense_index()`:

```python
"""Retrieval orchestration: dense, keyword (BM25), and hybrid (RRF + rerank) search."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ecomsearch.bm25 import BM25Index
from ecomsearch.config import (
    BM25_INDEX_PATH,
    CANDIDATE_POOL_SIZE,
    CATALOG_PATH,
    INDEX_PATH,
    ITEM_IDS_PATH,
    QDRANT_COLLECTION_NAME,
    RERANK_POOL_SIZE,
    VECTOR_BACKEND,
)
from ecomsearch.embeddings import Embedder
from ecomsearch.fusion import reciprocal_rank_fusion
from ecomsearch.index import ProductIndex
from ecomsearch.qdrant_index import QdrantIndex
from ecomsearch.reranker import CrossEncoderReranker

_dense_index = None
_bm25_index = None
_embedder = None
_reranker = None
_catalog = None
_search_executor = None


def load_dense_index():
    if VECTOR_BACKEND == "qdrant":
        return QdrantIndex(QDRANT_COLLECTION_NAME)
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No dense index found at {INDEX_PATH}. "
            "Run `python scripts/build_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)
```

The rest of `search.py` (from `load_bm25_index()` onward) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_search.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/search.py tests/test_search.py
git commit -m "feat: switch dense_search to Qdrant when VECTOR_BACKEND=qdrant"
git push origin main
```

---

### Task 5: Wire QdrantIndex into multimodal image search (TDD)

**Files:**
- Modify: `src/ecomsearch/multimodal/search.py`
- Test: `tests/test_multimodal_search.py`

- [ ] **Step 1: Write the failing test in `tests/test_multimodal_search.py`**

Add this test:

```python
def test_load_index_returns_qdrant_index_when_backend_is_qdrant(monkeypatch):
    monkeypatch.setattr(search, "VECTOR_BACKEND", "qdrant")

    class FakeQdrantIndex:
        def __init__(self, collection_name):
            self.collection_name = collection_name

    monkeypatch.setattr(search, "QdrantIndex", FakeQdrantIndex)

    index = search.load_index()

    assert isinstance(index, FakeQdrantIndex)
    assert index.collection_name == search.QDRANT_IMAGE_COLLECTION_NAME
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_multimodal_search.py::test_load_index_returns_qdrant_index_when_backend_is_qdrant -v`
Expected: FAIL with `AttributeError: module 'ecomsearch.multimodal.search' has no attribute 'VECTOR_BACKEND'` (or similar)

- [ ] **Step 3: Modify `src/ecomsearch/multimodal/search.py`**

```python
"""Image search orchestration: cached CLIP-based text-to-image search."""

from ecomsearch.config import VECTOR_BACKEND
from ecomsearch.index import ProductIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import INDEX_PATH, ITEM_IDS_PATH, QDRANT_IMAGE_COLLECTION_NAME
from ecomsearch.qdrant_index import QdrantIndex

_index = None
_embedder = None


def load_index():
    if VECTOR_BACKEND == "qdrant":
        return QdrantIndex(QDRANT_IMAGE_COLLECTION_NAME)
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No multimodal index found at {INDEX_PATH}. "
            "Run `python scripts/build_multimodal_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def _get_index() -> ProductIndex:
    global _index
    if _index is None:
        _index = load_index()
    return _index


def _get_embedder() -> ClipEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ClipEmbedder()
    return _embedder


def image_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = _get_index()
    embedder = _get_embedder()
    query_vector = embedder.embed_text([query])[0]
    return index.search(query_vector, top_k)
```

Note: `load_index()`'s return type annotation was dropped from the signature since it can now return either a `ProductIndex` or a `QdrantIndex` — matching the same un-annotated style already used for `search.py`'s `load_dense_index()` after this same change in Task 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_multimodal_search.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/multimodal/search.py tests/test_multimodal_search.py
git commit -m "feat: switch image_search to Qdrant when VECTOR_BACKEND=qdrant"
git push origin main
```

---

### Task 6: Migrate the text index to Qdrant Cloud

**Files:**
- Create: `scripts/upload_index_to_qdrant.py`
- Test: `tests/test_upload_index_to_qdrant.py`

- [ ] **Step 1: Write the failing test in `tests/test_upload_index_to_qdrant.py`**

```python
import pytest

import upload_index_to_qdrant


def test_main_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    missing_index = tmp_path / "does_not_exist.faiss"
    monkeypatch.setattr(upload_index_to_qdrant, "INDEX_PATH", missing_index)

    with pytest.raises(SystemExit) as excinfo:
        upload_index_to_qdrant.main()

    assert "does_not_exist.faiss" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_upload_index_to_qdrant.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upload_index_to_qdrant'`

- [ ] **Step 3: Write `scripts/upload_index_to_qdrant.py`**

```python
"""One-time script: upload the local dense text FAISS index into Qdrant Cloud.

Usage:
    python scripts/upload_index_to_qdrant.py
"""

import faiss
import numpy as np

from ecomsearch.config import INDEX_PATH, ITEM_IDS_PATH, QDRANT_COLLECTION_NAME
from ecomsearch.qdrant_index import QdrantIndex

UPSERT_BATCH_SIZE = 256


def main() -> None:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No local dense index found at {INDEX_PATH}. "
            "Run `python scripts/build_index.py` first to build it."
        )

    print(f"Loading local FAISS index from {INDEX_PATH}...")
    faiss_index = faiss.read_index(str(INDEX_PATH))
    item_ids = np.load(ITEM_IDS_PATH)
    vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    print(f"Loaded {len(item_ids)} vectors of dimension {faiss_index.d}.")

    print(f"Creating Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
    qdrant_index = QdrantIndex(QDRANT_COLLECTION_NAME)
    qdrant_index.create_collection(dim=faiss_index.d)

    print(f"Upserting {len(item_ids)} vectors in batches of {UPSERT_BATCH_SIZE}...")
    for start in range(0, len(item_ids), UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, len(item_ids))
        qdrant_index.upsert(vectors[start:end], item_ids[start:end])
        print(f"  upserted {end}/{len(item_ids)}")

    print(f"Done. Collection '{QDRANT_COLLECTION_NAME}' now holds {len(item_ids)} vectors.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_upload_index_to_qdrant.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run it for real**

Run: `venv/Scripts/python.exe scripts/upload_index_to_qdrant.py`
Expected: prints progress for all 55,516 vectors across 217 batches, ends with `Done. Collection 'ecommerce_products' now holds 55516 vectors.` This will take a few minutes over the network.

- [ ] **Step 6: Verify with a real query**

Run:
```bash
venv/Scripts/python.exe -c "
from ecomsearch.qdrant_index import QdrantIndex
from ecomsearch.embeddings import Embedder
from ecomsearch.config import QDRANT_COLLECTION_NAME

embedder = Embedder()
index = QdrantIndex(QDRANT_COLLECTION_NAME)
results = index.search(embedder.embed_query('organic almond milk'), top_k=3)
print(results)
"
```
Expected: prints 3 `(item_id, score)` tuples with plausible cosine scores (roughly 0.4-0.7 range, matching the local FAISS `dense` mode's typical score range documented in `docs/eval_results.md`).

- [ ] **Step 7: Commit**

```bash
git add scripts/upload_index_to_qdrant.py tests/test_upload_index_to_qdrant.py
git commit -m "feat: add script to migrate the text FAISS index into Qdrant Cloud"
git push origin main
```

---

### Task 7: Migrate the multimodal image index to Qdrant Cloud

**Files:**
- Create: `scripts/upload_multimodal_index_to_qdrant.py`
- Test: `tests/test_upload_multimodal_index_to_qdrant.py`

- [ ] **Step 1: Write the failing test in `tests/test_upload_multimodal_index_to_qdrant.py`**

```python
import pytest

import upload_multimodal_index_to_qdrant


def test_main_exits_with_clear_message_when_index_missing(tmp_path, monkeypatch):
    missing_index = tmp_path / "does_not_exist.faiss"
    monkeypatch.setattr(upload_multimodal_index_to_qdrant, "INDEX_PATH", missing_index)

    with pytest.raises(SystemExit) as excinfo:
        upload_multimodal_index_to_qdrant.main()

    assert "does_not_exist.faiss" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_upload_multimodal_index_to_qdrant.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upload_multimodal_index_to_qdrant'`

- [ ] **Step 3: Write `scripts/upload_multimodal_index_to_qdrant.py`**

```python
"""One-time script: upload the local CLIP image FAISS index into Qdrant Cloud.

Usage:
    python scripts/upload_multimodal_index_to_qdrant.py
"""

import faiss
import numpy as np

from ecomsearch.multimodal.config import INDEX_PATH, ITEM_IDS_PATH, QDRANT_IMAGE_COLLECTION_NAME
from ecomsearch.qdrant_index import QdrantIndex

UPSERT_BATCH_SIZE = 256


def main() -> None:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No local multimodal index found at {INDEX_PATH}. "
            "Run `python scripts/build_multimodal_index.py` first to build it."
        )

    print(f"Loading local FAISS index from {INDEX_PATH}...")
    faiss_index = faiss.read_index(str(INDEX_PATH))
    item_ids = np.load(ITEM_IDS_PATH)
    vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    print(f"Loaded {len(item_ids)} vectors of dimension {faiss_index.d}.")

    print(f"Creating Qdrant collection '{QDRANT_IMAGE_COLLECTION_NAME}'...")
    qdrant_index = QdrantIndex(QDRANT_IMAGE_COLLECTION_NAME)
    qdrant_index.create_collection(dim=faiss_index.d)

    print(f"Upserting {len(item_ids)} vectors in batches of {UPSERT_BATCH_SIZE}...")
    for start in range(0, len(item_ids), UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, len(item_ids))
        qdrant_index.upsert(vectors[start:end], item_ids[start:end])
        print(f"  upserted {end}/{len(item_ids)}")

    print(f"Done. Collection '{QDRANT_IMAGE_COLLECTION_NAME}' now holds {len(item_ids)} vectors.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_upload_multimodal_index_to_qdrant.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run it for real**

Run: `venv/Scripts/python.exe scripts/upload_multimodal_index_to_qdrant.py`
Expected: prints progress for all 4,996 vectors across 20 batches, ends with `Done. Collection 'ecommerce_products_images' now holds 4996 vectors.`

- [ ] **Step 6: Verify with a real query**

Run:
```bash
venv/Scripts/python.exe -c "
from ecomsearch.qdrant_index import QdrantIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import QDRANT_IMAGE_COLLECTION_NAME

embedder = ClipEmbedder()
index = QdrantIndex(QDRANT_IMAGE_COLLECTION_NAME)
results = index.search(embedder.embed_text(['shoes'])[0], top_k=3)
print(results)
"
```
Expected: prints 3 `(item_id, score)` tuples with item IDs that are valid rows in `artifacts/multimodal/subset_metadata.csv`.

- [ ] **Step 7: Commit**

```bash
git add scripts/upload_multimodal_index_to_qdrant.py tests/test_upload_multimodal_index_to_qdrant.py
git commit -m "feat: add script to migrate the CLIP image FAISS index into Qdrant Cloud"
git push origin main
```

---

### Task 8: Real end-to-end verification of the Qdrant-backed search path

**Files:**
- Create: `tests/test_search_qdrant_e2e.py`

With both Qdrant collections now populated (Tasks 6-7) and the factory switch wired up (Tasks 4-5), this task adds a persisted, real (non-mocked) test proving `dense_search()` and `image_search()` actually work end-to-end against Qdrant Cloud with production-scale data — not just the small synthetic round trip from Task 3.

- [ ] **Step 1: Write `tests/test_search_qdrant_e2e.py`**

```python
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


@pytest.fixture(autouse=True)
def qdrant_backend(monkeypatch):
    monkeypatch.setattr(search, "VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(multimodal_search, "VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(search, "_dense_index", None, raising=False)
    monkeypatch.setattr(multimodal_search, "_index", None, raising=False)


def test_dense_search_returns_relevant_result_from_qdrant():
    results = search.dense_search("organic almond milk", top_k=5)

    assert len(results) > 0


def test_image_search_returns_results_from_qdrant():
    results = multimodal_search.image_search("shoes", top_k=5)

    assert len(results) > 0
```

- [ ] **Step 2: Run and verify**

Run: `venv/Scripts/python.exe -m pytest tests/test_search_qdrant_e2e.py -v`
Expected: PASS (2 tests). This loads real models (bge-small, CLIP) so it will take a similar amount of time to the existing `test_integration.py`/`test_multimodal_integration.py` tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_search_qdrant_e2e.py
git commit -m "test: add real end-to-end verification of the Qdrant-backed search path"
git push origin main
```

---

### Task 9: Upload runtime artifacts to a Hugging Face Hub dataset repo

**Files:**
- Create: `scripts/upload_artifacts_to_hf.py`
- Test: `tests/test_upload_artifacts_to_hf.py`

This uploads only the catalog CSV, the BM25 pickle, and the 4,996 CLIP subset images + their metadata — not the full 548MB local image dataset, since the API only ever serves the subset referenced by `subset_metadata.csv`.

- [ ] **Step 1: Write the failing tests in `tests/test_upload_artifacts_to_hf.py`**

```python
import pytest

import upload_artifacts_to_hf


def test_main_exits_with_clear_message_when_dataset_repo_not_set(monkeypatch):
    monkeypatch.setattr(upload_artifacts_to_hf, "HF_DATASET_REPO", None)

    with pytest.raises(SystemExit) as excinfo:
        upload_artifacts_to_hf.main()

    assert "HF_DATASET_REPO" in str(excinfo.value)


def test_main_exits_with_clear_message_when_catalog_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_artifacts_to_hf, "HF_DATASET_REPO", "someuser/somerepo")
    monkeypatch.setattr(upload_artifacts_to_hf, "CATALOG_PATH", tmp_path / "does_not_exist.csv")

    with pytest.raises(SystemExit) as excinfo:
        upload_artifacts_to_hf.main()

    assert "does_not_exist.csv" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_upload_artifacts_to_hf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'upload_artifacts_to_hf'`

- [ ] **Step 3: Write `scripts/upload_artifacts_to_hf.py`**

```python
"""One-time script: upload the catalog CSV, BM25 pickle, and CLIP subset
images/metadata to a Hugging Face Hub dataset repo, so production containers can
download them at startup instead of needing gigabytes of raw data baked into the
Docker image.

Usage:
    python scripts/upload_artifacts_to_hf.py
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from ecomsearch.config import BM25_INDEX_PATH, CATALOG_PATH, HF_DATASET_REPO, HF_TOKEN, REPO_ROOT
from ecomsearch.multimodal.config import DATASET_IMAGES_DIR, SUBSET_METADATA_PATH


def _stage_artifacts(staging_dir: Path) -> None:
    catalog_dest = staging_dir / CATALOG_PATH.relative_to(REPO_ROOT)
    catalog_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CATALOG_PATH, catalog_dest)

    bm25_dest = staging_dir / BM25_INDEX_PATH.relative_to(REPO_ROOT)
    bm25_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BM25_INDEX_PATH, bm25_dest)

    metadata_dest = staging_dir / SUBSET_METADATA_PATH.relative_to(REPO_ROOT)
    metadata_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUBSET_METADATA_PATH, metadata_dest)

    subset = pd.read_csv(SUBSET_METADATA_PATH)
    images_dest_dir = staging_dir / DATASET_IMAGES_DIR.relative_to(REPO_ROOT)
    images_dest_dir.mkdir(parents=True, exist_ok=True)
    for image_name in subset["image"]:
        shutil.copy2(DATASET_IMAGES_DIR / image_name, images_dest_dir / image_name)
    print(f"Staged catalog, BM25 index, and {len(subset)} subset images.")


def main() -> None:
    if not HF_DATASET_REPO:
        raise SystemExit(
            "HF_DATASET_REPO is not set. Add it to your .env, e.g. "
            "HF_DATASET_REPO=your-hf-username/ecommerce-search-artifacts"
        )

    for path, build_hint in [
        (CATALOG_PATH, None),
        (BM25_INDEX_PATH, "python scripts/build_bm25_index.py"),
        (SUBSET_METADATA_PATH, "python scripts/build_multimodal_index.py"),
    ]:
        if not path.exists():
            hint = f" Run `{build_hint}` first." if build_hint else ""
            raise SystemExit(f"Required artifact not found at {path}.{hint}")

    api = HfApi(token=HF_TOKEN)
    print(f"Creating (or reusing) dataset repo '{HF_DATASET_REPO}'...")
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        _stage_artifacts(staging_dir)

        print(f"Uploading staged artifacts to '{HF_DATASET_REPO}'...")
        api.upload_folder(
            repo_id=HF_DATASET_REPO,
            folder_path=str(staging_dir),
            repo_type="dataset",
            commit_message="Upload catalog, BM25 index, and CLIP subset images",
        )

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_upload_artifacts_to_hf.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run it for real**

Run: `venv/Scripts/python.exe scripts/upload_artifacts_to_hf.py`
Expected: prints `Staged catalog, BM25 index, and 4996 subset images.` then uploads (~160MB — may take a few minutes depending on upload bandwidth), ends with `Done.`

- [ ] **Step 6: Verify on Hugging Face Hub**

Visit `https://huggingface.co/datasets/<your HF_DATASET_REPO value>/tree/main` in a browser and confirm `data/ecommerce_catalog_enriched.csv`, `artifacts/bm25.pkl`, `artifacts/multimodal/subset_metadata.csv`, and `data/multimodal/data/` (with ~4,996 images) are all present.

- [ ] **Step 7: Commit**

```bash
git add scripts/upload_artifacts_to_hf.py tests/test_upload_artifacts_to_hf.py
git commit -m "feat: add script to upload runtime artifacts to a Hugging Face dataset repo"
git push origin main
```

---

### Task 10: Backend startup bootstrap — download artifacts if missing (TDD)

**Files:**
- Modify: `src/ecomsearch/api/app.py`
- Modify: `tests/test_api_app.py`

- [ ] **Step 1: Write the failing tests in `tests/test_api_app.py`**

Add these two tests (keep the existing `test_health_check_returns_ok` and `test_lifespan_warms_up_all_caches_on_startup` tests as they are):

```python
def test_lifespan_downloads_artifacts_when_missing(monkeypatch, tmp_path):
    missing_catalog = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(app_module, "CATALOG_PATH", missing_catalog)
    monkeypatch.setattr(app_module, "HF_DATASET_REPO", "someuser/somerepo")
    download_calls = []
    monkeypatch.setattr(
        app_module, "snapshot_download", lambda **kwargs: download_calls.append(kwargs)
    )
    monkeypatch.setattr(app_module, "dense_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "bm25_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "hybrid_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "image_search", lambda *a, **k: None)

    with TestClient(app_module.app):
        pass

    assert len(download_calls) == 1
    assert download_calls[0]["repo_id"] == "someuser/somerepo"


def test_lifespan_skips_download_when_artifacts_already_present(monkeypatch, tmp_path):
    existing_catalog = tmp_path / "catalog.csv"
    existing_catalog.write_text("item_id,search_text\n")
    monkeypatch.setattr(app_module, "CATALOG_PATH", existing_catalog)
    download_calls = []
    monkeypatch.setattr(
        app_module, "snapshot_download", lambda **kwargs: download_calls.append(kwargs)
    )
    monkeypatch.setattr(app_module, "dense_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "bm25_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "hybrid_search", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "image_search", lambda *a, **k: None)

    with TestClient(app_module.app):
        pass

    assert download_calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_app.py -v`
Expected: FAIL with `AttributeError: <module 'ecomsearch.api.app'> does not have the attribute 'CATALOG_PATH'` (or similar)

- [ ] **Step 3: Modify `src/ecomsearch/api/app.py`**

```python
"""FastAPI application: serving layer for text and image product search."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from huggingface_hub import snapshot_download

from ecomsearch.api.routes_image import router as image_router
from ecomsearch.api.routes_text import router as text_router
from ecomsearch.config import CATALOG_PATH, HF_DATASET_REPO, HF_TOKEN, REPO_ROOT
from ecomsearch.multimodal.search import image_search
from ecomsearch.search import bm25_search, dense_search, hybrid_search


def _ensure_artifacts_present() -> None:
    if CATALOG_PATH.exists():
        return
    if not HF_DATASET_REPO:
        raise SystemExit(
            "Catalog not found locally and HF_DATASET_REPO is not set -- "
            "cannot bootstrap production artifacts."
        )
    print(f"Downloading artifacts from '{HF_DATASET_REPO}'...")
    snapshot_download(
        repo_id=HF_DATASET_REPO, repo_type="dataset", local_dir=str(REPO_ROOT), token=HF_TOKEN
    )


def _warm_up_caches() -> None:
    dense_search("warm up", top_k=1)
    bm25_search("warm up", top_k=1)
    hybrid_search("warm up", top_k=1, use_rerank=True)
    image_search("warm up", top_k=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_artifacts_present()
    _warm_up_caches()
    yield


app = FastAPI(title="E-Commerce Semantic Search API", lifespan=lifespan)
app.include_router(text_router)
app.include_router(image_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_app.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (locally, `CATALOG_PATH` already exists, so `_ensure_artifacts_present()` is a no-op for every other test in the suite that boots the real app, e.g. `test_api_integration.py`).

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/api/app.py tests/test_api_app.py
git commit -m "feat: download runtime artifacts from Hugging Face Hub on backend startup if missing"
git push origin main
```

---

### Task 11: Backend Dockerfile

**Files:**
- Create: `.dockerignore`
- Create: `Dockerfile.api`

- [ ] **Step 1: Write `.dockerignore`**

Without this, `docker build` from the repo root would upload the entire repo as build context — including `venv/` (several GB), `data/` (~620MB), `artifacts/` (~130MB), and `.git/` — even though the Dockerfiles only `COPY` a handful of specific paths. This makes local builds slow or impractically large.

```
venv/
.venv/
.git/
.pytest_cache/
__pycache__/
*.pyc
data/
artifacts/
demo_results/
docs/
tests/
eval/
*.log
.env
```

- [ ] **Step 2: Write `Dockerfile.api`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Pre-download model weights at build time so container restarts don't refetch
# them from Hugging Face Hub on every cold start.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

ENV VECTOR_BACKEND=qdrant
EXPOSE 7860

CMD ["sh", "-c", "uvicorn ecomsearch.api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
```

- [ ] **Step 3: Build the image**

Run: `docker build -f Dockerfile.api -t ecomsearch-api .`
Expected: builds successfully (this will take a while the first time — downloading and caching all model weights). Note: this step requires `data/ecommerce_catalog_enriched.csv` and `artifacts/` to NOT be copied into the image (they aren't — the Dockerfile never references `data/` or `artifacts/`), since those are downloaded at container startup instead.

- [ ] **Step 4: Run the container against your real Qdrant Cloud + HF dataset repo**

Run:
```bash
docker run --rm -p 8000:7860 --env-file .env -e VECTOR_BACKEND=qdrant ecomsearch-api
```
Expected: logs show the artifact download (`Downloading artifacts from '...'`), then normal FastAPI/uvicorn startup logs, ending with `Uvicorn running on http://0.0.0.0:7860`. This will take longer than a normal local start — real model warm-up plus the ~160MB artifact download.

- [ ] **Step 5: Verify from another terminal while the container is running**

```bash
curl -s http://127.0.0.1:8000/health
curl -s "http://127.0.0.1:8000/search/text?q=organic+almond+milk&top_k=3"
curl -s "http://127.0.0.1:8000/search/image?q=shoes&top_k=3"
```
Expected: `/health` returns `{"status":"ok"}`; both search endpoints return real, relevant results — the same shape as the local (non-Docker) verification from Phase 5, but now served from a container that's using Qdrant Cloud instead of local FAISS files. Stop the container with Ctrl+C when done.

- [ ] **Step 6: Commit**

```bash
git add .dockerignore Dockerfile.api
git commit -m "feat: add backend Dockerfile"
git push origin main
```

---

### Task 12: Frontend Dockerfile

**Files:**
- Create: `requirements-ui.txt`
- Create: `Dockerfile.ui`

The frontend (`streamlit_app.py`) only ever imports `os`, `requests`, and `streamlit` — never anything from the `ecomsearch` package — so its image doesn't need the heavyweight ML dependencies (torch, transformers, faiss) that the backend needs. It gets its own minimal requirements file.

- [ ] **Step 1: Write `requirements-ui.txt`**

```
streamlit>=1.32.0
requests>=2.31.0
```

- [ ] **Step 2: Write `Dockerfile.ui`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY src/ecomsearch/ui/streamlit_app.py ./src/ecomsearch/ui/streamlit_app.py

EXPOSE 7860

CMD ["sh", "-c", "streamlit run src/ecomsearch/ui/streamlit_app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
```

- [ ] **Step 3: Build the image**

Run: `docker build -f Dockerfile.ui -t ecomsearch-ui .`
Expected: builds quickly (lightweight image, no ML dependencies).

- [ ] **Step 4: Run it against the backend container from Task 11**

With the Task 11 backend container still running (or restarted) on port 8000, in a separate terminal:
```bash
docker run --rm -p 8501:7860 -e API_BASE_URL=http://host.docker.internal:8000 ecomsearch-ui
```
`host.docker.internal` lets the frontend container reach a backend running directly on your host (via `docker run -p 8000:7860` from Task 11). Expected: Streamlit startup logs, then reachable at `http://localhost:8501`.

- [ ] **Step 5: Verify**

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8501
```
Expected: `HTTP 200`. Since there's no browser tool available in this environment, this confirms the containerized frontend boots and serves correctly as a client of the containerized backend; a full interactive click-through (as in Phase 5) is worth doing yourself in a browser before considering the deployment fully verified. Stop both containers with Ctrl+C when done.

- [ ] **Step 6: Commit**

```bash
git add requirements-ui.txt Dockerfile.ui
git commit -m "feat: add frontend Dockerfile"
git push origin main
```

---

### Task 13: Deploy both apps to Hugging Face Spaces

**Files:**
- Create: `scripts/deploy_backend_space.py`
- Create: `scripts/deploy_frontend_space.py`
- Test: `tests/test_deploy_backend_space.py`
- Test: `tests/test_deploy_frontend_space.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_deploy_backend_space.py`:
```python
import pytest

import deploy_backend_space


def test_main_exits_with_clear_message_when_space_not_set(monkeypatch):
    monkeypatch.setattr(deploy_backend_space, "HF_SPACE_BACKEND", None)

    with pytest.raises(SystemExit) as excinfo:
        deploy_backend_space.main()

    assert "HF_SPACE_BACKEND" in str(excinfo.value)
```

`tests/test_deploy_frontend_space.py`:
```python
import pytest

import deploy_frontend_space


def test_main_exits_with_clear_message_when_space_not_set(monkeypatch):
    monkeypatch.setattr(deploy_frontend_space, "HF_SPACE_FRONTEND", None)

    with pytest.raises(SystemExit) as excinfo:
        deploy_frontend_space.main()

    assert "HF_SPACE_FRONTEND" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_deploy_backend_space.py tests/test_deploy_frontend_space.py -v`
Expected: FAIL with `ModuleNotFoundError` for each.

- [ ] **Step 3: Write `scripts/deploy_backend_space.py`**

```python
"""One-time (or repeat-as-needed) script: push the FastAPI backend to its
Hugging Face Space.

Usage:
    python scripts/deploy_backend_space.py
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from ecomsearch.config import HF_SPACE_BACKEND, HF_TOKEN, REPO_ROOT

SPACE_README = """---
title: Ecommerce Search API
emoji: \U0001f50d
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

FastAPI backend for the E-Commerce Semantic Search project.
"""


def main() -> None:
    if not HF_SPACE_BACKEND:
        raise SystemExit("HF_SPACE_BACKEND is not set. Add it to your .env.")

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        shutil.copy2(REPO_ROOT / "Dockerfile.api", staging_dir / "Dockerfile")
        shutil.copy2(REPO_ROOT / "requirements.txt", staging_dir / "requirements.txt")
        shutil.copy2(REPO_ROOT / "pyproject.toml", staging_dir / "pyproject.toml")
        shutil.copytree(REPO_ROOT / "src", staging_dir / "src")
        (staging_dir / "README.md").write_text(SPACE_README, encoding="utf-8")

        api = HfApi(token=HF_TOKEN)
        print(f"Creating (or reusing) Space '{HF_SPACE_BACKEND}'...")
        api.create_repo(
            repo_id=HF_SPACE_BACKEND, repo_type="space", space_sdk="docker", exist_ok=True
        )
        print(f"Uploading backend to '{HF_SPACE_BACKEND}'...")
        api.upload_folder(
            repo_id=HF_SPACE_BACKEND,
            folder_path=str(staging_dir),
            repo_type="space",
            commit_message="Deploy backend",
        )

    print(f"Done. https://huggingface.co/spaces/{HF_SPACE_BACKEND}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `scripts/deploy_frontend_space.py`**

```python
"""One-time (or repeat-as-needed) script: push the Streamlit frontend to its
Hugging Face Space.

Usage:
    python scripts/deploy_frontend_space.py
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from ecomsearch.config import HF_SPACE_FRONTEND, HF_TOKEN, REPO_ROOT

SPACE_README = """---
title: Ecommerce Search UI
emoji: \U0001f6cd
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

Streamlit frontend for the E-Commerce Semantic Search project.
"""


def main() -> None:
    if not HF_SPACE_FRONTEND:
        raise SystemExit("HF_SPACE_FRONTEND is not set. Add it to your .env.")

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        shutil.copy2(REPO_ROOT / "Dockerfile.ui", staging_dir / "Dockerfile")
        shutil.copy2(REPO_ROOT / "requirements-ui.txt", staging_dir / "requirements-ui.txt")
        ui_dest_dir = staging_dir / "src" / "ecomsearch" / "ui"
        ui_dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            REPO_ROOT / "src" / "ecomsearch" / "ui" / "streamlit_app.py",
            ui_dest_dir / "streamlit_app.py",
        )
        (staging_dir / "README.md").write_text(SPACE_README, encoding="utf-8")

        api = HfApi(token=HF_TOKEN)
        print(f"Creating (or reusing) Space '{HF_SPACE_FRONTEND}'...")
        api.create_repo(
            repo_id=HF_SPACE_FRONTEND, repo_type="space", space_sdk="docker", exist_ok=True
        )
        print(f"Uploading frontend to '{HF_SPACE_FRONTEND}'...")
        api.upload_folder(
            repo_id=HF_SPACE_FRONTEND,
            folder_path=str(staging_dir),
            repo_type="space",
            commit_message="Deploy frontend",
        )

    print(f"Done. https://huggingface.co/spaces/{HF_SPACE_FRONTEND}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_deploy_backend_space.py tests/test_deploy_frontend_space.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Deploy the backend for real**

Run: `venv/Scripts/python.exe scripts/deploy_backend_space.py`
Expected: prints `Done. https://huggingface.co/spaces/<HF_SPACE_BACKEND>`. The Space will start building automatically on Hugging Face's infrastructure once pushed — this takes several minutes (Docker build + model download + artifact download on first boot). Check build progress and logs at that URL.

- [ ] **Step 7: Set the Qdrant/HF secrets on the backend Space**

The backend container needs `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`, `QDRANT_IMAGE_COLLECTION_NAME`, `HF_DATASET_REPO`, and `HF_TOKEN` at runtime — these must be set as **Space secrets** via the Space's Settings page (`https://huggingface.co/spaces/<HF_SPACE_BACKEND>/settings`), using the same values from your local `.env`. `.env` itself is never uploaded (Task 11/13's staging only copies `Dockerfile`, `requirements.txt`, `pyproject.toml`, `src/`). `VECTOR_BACKEND=qdrant` is already baked into `Dockerfile.api` via `ENV VECTOR_BACKEND=qdrant`, so it doesn't need to be set as a secret.

- [ ] **Step 8: Deploy the frontend for real**

Run: `venv/Scripts/python.exe scripts/deploy_frontend_space.py`
Expected: prints `Done. https://huggingface.co/spaces/<HF_SPACE_FRONTEND>`.

- [ ] **Step 9: Point the frontend at the deployed backend**

On the frontend Space's Settings page (`https://huggingface.co/spaces/<HF_SPACE_FRONTEND>/settings`), set the `API_BASE_URL` secret/variable to the backend Space's public URL, typically `https://<hf-username>-<backend-space-name>.hf.space` (shown on the backend Space's page once it's built and running).

- [ ] **Step 10: Verify both live Spaces**

Once both Spaces finish building (check their `/logs` tabs for "Uvicorn running" / "You can now view your Streamlit app"):
```bash
curl -s https://<hf-username>-<backend-space-name>.hf.space/health
```
Expected: `{"status":"ok"}`. Then open `https://<hf-username>-<frontend-space-name>.hf.space` in a browser and confirm both the Text Search and Image Search tabs return real results.

- [ ] **Step 11: Commit**

```bash
git add scripts/deploy_backend_space.py scripts/deploy_frontend_space.py tests/test_deploy_backend_space.py tests/test_deploy_frontend_space.py
git commit -m "feat: add scripts to deploy backend and frontend to Hugging Face Spaces"
git push origin main
```

---

### Task 14: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass, including the new Qdrant unit/integration tests and the deploy/upload script guard-clause tests added in this phase.

- [ ] **Step 2: Update `README.md`**

Read the current file first, then:
- Check off `- [ ] Phase 6 — Deployment (Qdrant Cloud + Hugging Face Spaces)` and update the Status paragraph to mention it, e.g.: "Dense vector search now runs on Qdrant Cloud in production, and the app is deployed live as two Hugging Face Spaces (FastAPI backend, Streamlit frontend) — see Deployment below."
- Add a new `## Deployment` section (after `## Running the App`, before `## Known limitations`) documenting: the one-time setup order (`upload_index_to_qdrant.py` → `upload_multimodal_index_to_qdrant.py` → `upload_artifacts_to_hf.py` → `deploy_backend_space.py` → `deploy_frontend_space.py`), the `VECTOR_BACKEND` env var's role, and the live Space URLs (using this project's actual `HF_SPACE_BACKEND`/`HF_SPACE_FRONTEND` values from `.env` — link to `https://huggingface.co/spaces/<value>` for each).
- Extend the existing "Known limitations" section (which already documents the Qdrant free-tier auto-suspend cold start) with a note that HF Spaces' free tier has ephemeral storage, so every container restart re-downloads the ~160MB artifact bundle from the Hugging Face dataset repo — expect a slow cold start after periods of inactivity.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: add Phase 6 deployment usage to README"
git push origin main
```
