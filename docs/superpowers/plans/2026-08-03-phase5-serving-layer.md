# Phase 5: Serving Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI backend serving both text search (all 4 retrieval modes) and multimodal image search over HTTP, plus a Streamlit frontend that consumes it — the first phase that turns `ecomsearch` into a running service instead of a CLI/scripts-only package.

**Architecture:** A new `src/ecomsearch/multimodal/search.py` module (mirroring `ecomsearch/search.py`'s Phase 4b caching pattern) gives image search a pure, cacheable function. A FastAPI app (`src/ecomsearch/api/`) wraps both text and multimodal search behind HTTP routes, pre-warming all caches at startup via a `lifespan` hook. A Streamlit app (`src/ecomsearch/ui/streamlit_app.py`) is a pure HTTP client of the backend.

**Tech Stack:** FastAPI, Uvicorn, Streamlit, `requests` (frontend→backend HTTP calls), `httpx` (required by FastAPI's `TestClient`).

**Spec:** `docs/superpowers/specs/2026-08-03-phase5-serving-layer-design.md`

**Amendment to the spec's file list:** the spec says dependencies go in `pyproject.toml`, but this repo actually manages dependencies in `requirements.txt` (`pyproject.toml` only has `[project.scripts]` and packaging config, no `[project.dependencies]`). Task 1 below adds the new packages to `requirements.txt` instead.

---

### Task 1: Add serving-layer dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append these lines to the end of `requirements.txt`**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
streamlit>=1.32.0
requests>=2.31.0
httpx>=0.27.0
```

- [ ] **Step 2: Install and verify**

Run (using the project venv):
```bash
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe -c "import fastapi, uvicorn, streamlit, requests, httpx; print('ok')"
```
Expected: installs succeed, prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add FastAPI/Streamlit serving layer dependencies"
git push origin main
```

---

### Task 2: Multimodal search module (TDD)

**Files:**
- Create: `src/ecomsearch/multimodal/search.py`
- Test: `tests/test_multimodal_search.py`

`multimodal/cli.py`'s `search()` function inlines index-loading, embedding, and CLI-only side effects (copying image files to `demo_results/`) together with no caching. This task extracts a pure, cached `image_search(query, top_k)` function that the API can call repeatedly — mirroring `ecomsearch/search.py`'s exact lazy-singleton pattern from Phase 4b. `multimodal/cli.py` is not modified in this task.

- [ ] **Step 1: Write the failing tests in `tests/test_multimodal_search.py`**

```python
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
```

This uses the existing session-scoped `clip_embedder` fixture already defined in `tests/conftest.py`. `raising=False` on the reset fixture matters for the same reason it did in Phase 4b's `search.py` caching task: at this point `search.py` (the new module you're about to create) doesn't have `_index`/`_embedder` yet.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multimodal_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.multimodal.search'`

- [ ] **Step 3: Write `src/ecomsearch/multimodal/search.py`**

```python
"""Image search orchestration: cached CLIP-based text-to-image search."""
from ecomsearch.index import ProductIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import INDEX_PATH, ITEM_IDS_PATH

_index = None
_embedder = None


def load_index() -> ProductIndex:
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multimodal_search.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/multimodal/search.py tests/test_multimodal_search.py
git commit -m "feat: add cached image_search function for multimodal module"
git push origin main
```

---

### Task 3: FastAPI app skeleton — schemas, lifespan warm-up, health check (TDD)

**Files:**
- Create: `src/ecomsearch/api/__init__.py`
- Create: `src/ecomsearch/api/schemas.py`
- Create: `src/ecomsearch/api/app.py`
- Test: `tests/test_api_app.py`

- [ ] **Step 1: Create the empty package marker**

Create `src/ecomsearch/api/__init__.py` with empty content (0 bytes).

- [ ] **Step 2: Write `src/ecomsearch/api/schemas.py`**

```python
"""Pydantic request/response models for the FastAPI app."""
from pydantic import BaseModel


class TextSearchResult(BaseModel):
    item_id: int
    name: str
    brand: str
    category_path: str
    score: float


class TextSearchResponse(BaseModel):
    query: str
    mode: str
    results: list[TextSearchResult]


class ImageSearchResult(BaseModel):
    item_id: int
    display_name: str
    category: str
    score: float
    image_url: str


class ImageSearchResponse(BaseModel):
    query: str
    results: list[ImageSearchResult]
```

No test needed for this file — it's pure data declarations with no logic, exercised indirectly by the route tests in later tasks.

- [ ] **Step 3: Write the failing tests in `tests/test_api_app.py`**

```python
from fastapi.testclient import TestClient

from ecomsearch.api import app as app_module


def test_health_check_returns_ok():
    client = TestClient(app_module.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_warms_up_all_caches_on_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "dense_search", lambda *a, **k: calls.append("dense"))
    monkeypatch.setattr(app_module, "bm25_search", lambda *a, **k: calls.append("bm25"))
    monkeypatch.setattr(app_module, "hybrid_search", lambda *a, **k: calls.append("hybrid"))
    monkeypatch.setattr(app_module, "image_search", lambda *a, **k: calls.append("image"))

    with TestClient(app_module.app):
        pass

    assert calls == ["dense", "bm25", "hybrid", "image"]
```

`test_health_check_returns_ok` uses `TestClient(app)` **without** the `with` block, so the real `lifespan` (real model loading) does not run — this keeps the test fast. `test_lifespan_warms_up_all_caches_on_startup` uses `with TestClient(app_module.app):` specifically to trigger `lifespan`, with every underlying search function mocked so no real models load.

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_api_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.api.app'`

- [ ] **Step 5: Write `src/ecomsearch/api/app.py`**

```python
"""FastAPI application: serving layer for text and image product search."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ecomsearch.multimodal.search import image_search
from ecomsearch.search import bm25_search, dense_search, hybrid_search


def _warm_up_caches() -> None:
    dense_search("warm up", top_k=1)
    bm25_search("warm up", top_k=1)
    hybrid_search("warm up", top_k=1, use_rerank=True)
    image_search("warm up", top_k=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_up_caches()
    yield


app = FastAPI(title="E-Commerce Semantic Search API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_api_app.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add src/ecomsearch/api/__init__.py src/ecomsearch/api/schemas.py src/ecomsearch/api/app.py tests/test_api_app.py
git commit -m "feat: add FastAPI app skeleton with cache warm-up and health check"
git push origin main
```

---

### Task 4: Text search route (TDD)

**Files:**
- Create: `src/ecomsearch/api/routes_text.py`
- Modify: `src/ecomsearch/api/app.py`
- Test: `tests/test_api_text.py`

- [ ] **Step 1: Write the failing tests in `tests/test_api_text.py`**

```python
import pandas as pd
from fastapi.testclient import TestClient

from ecomsearch.api import routes_text
from ecomsearch.api.app import app


def test_search_text_returns_results_from_default_hybrid_mode(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "item_id": [101],
            "name": ["Organic Almond Milk"],
            "brand": ["Test Brand"],
            "category_path": ["Dairy > Milk Alternatives"],
        }
    ).to_csv(catalog_path, index=False)
    monkeypatch.setattr(routes_text, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(routes_text, "_catalog", None, raising=False)
    monkeypatch.setattr(
        routes_text, "hybrid_search", lambda query, top_k, use_rerank: [(101, 0.87)]
    )

    client = TestClient(app)
    response = client.get("/search/text", params={"q": "almond milk"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert data["results"][0]["item_id"] == 101
    assert data["results"][0]["name"] == "Organic Almond Milk"


def test_search_text_dispatches_to_requested_mode(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "item_id": [202],
            "name": ["Wireless Headphones"],
            "brand": ["AudioCo"],
            "category_path": ["Electronics > Audio"],
        }
    ).to_csv(catalog_path, index=False)
    monkeypatch.setattr(routes_text, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(routes_text, "_catalog", None, raising=False)
    monkeypatch.setattr(routes_text, "dense_search", lambda query, top_k: [(202, 0.5)])

    client = TestClient(app)
    response = client.get("/search/text", params={"q": "headphones", "mode": "dense"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "dense"
    assert data["results"][0]["item_id"] == 202


def test_search_text_rejects_invalid_mode():
    client = TestClient(app)
    response = client.get("/search/text", params={"q": "anything", "mode": "not-a-real-mode"})

    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.api.routes_text'` (or a 404, since `/search/text` doesn't exist yet)

- [ ] **Step 3: Write `src/ecomsearch/api/routes_text.py`**

```python
"""FastAPI routes for text (catalog) search."""
from typing import Literal

import pandas as pd
from fastapi import APIRouter

from ecomsearch.api.schemas import TextSearchResponse, TextSearchResult
from ecomsearch.config import CATALOG_PATH, DEFAULT_TOP_K
from ecomsearch.search import bm25_search, dense_search, hybrid_search

router = APIRouter()

_catalog = None

MODES = {
    "dense": lambda query, top_k: dense_search(query, top_k),
    "bm25": lambda query, top_k: bm25_search(query, top_k),
    "hybrid": lambda query, top_k: hybrid_search(query, top_k, use_rerank=False),
    "hybrid-rerank": lambda query, top_k: hybrid_search(query, top_k, use_rerank=True),
}


def _get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        _catalog = pd.read_csv(
            CATALOG_PATH, usecols=["item_id", "name", "brand", "category_path"]
        ).set_index("item_id")
    return _catalog


@router.get("/search/text", response_model=TextSearchResponse)
def search_text(
    q: str,
    mode: Literal["dense", "bm25", "hybrid", "hybrid-rerank"] = "hybrid",
    top_k: int = DEFAULT_TOP_K,
) -> TextSearchResponse:
    search_fn = MODES[mode]
    results = search_fn(q, top_k)
    catalog = _get_catalog()

    items = []
    for item_id, score in results:
        row = catalog.loc[item_id]
        items.append(
            TextSearchResult(
                item_id=item_id,
                name=str(row["name"]),
                brand=str(row["brand"]),
                category_path=str(row["category_path"]),
                score=score,
            )
        )

    return TextSearchResponse(query=q, mode=mode, results=items)
```

Note: in `test_search_text_dispatches_to_requested_mode`, monkeypatching `routes_text.dense_search` does not change what `MODES["dense"]` calls, because `MODES` is a dict of lambdas built once at import time that closes over the name `dense_search` looked up **at call time** from the module's global namespace — so replacing `routes_text.dense_search` via `monkeypatch.setattr` does correctly redirect the lambda's call. This is the same mechanism `test_cli.py` and `test_run_eval.py` already rely on elsewhere in this codebase.

- [ ] **Step 4: Wire the router into the app**

Modify `src/ecomsearch/api/app.py`: add this import near the top (after the existing `from ecomsearch.search import ...` line):
```python
from ecomsearch.api.routes_text import router as text_router
```
And add this line right after `app = FastAPI(...)`:
```python
app.include_router(text_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_api_text.py tests/test_api_app.py -v`
Expected: PASS (5 tests total)

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/api/routes_text.py src/ecomsearch/api/app.py tests/test_api_text.py
git commit -m "feat: add /search/text API route"
git push origin main
```

---

### Task 5: Image search + image-serving routes (TDD)

**Files:**
- Create: `src/ecomsearch/api/routes_image.py`
- Modify: `src/ecomsearch/api/app.py`
- Test: `tests/test_api_image.py`

- [ ] **Step 1: Write the failing tests in `tests/test_api_image.py`**

```python
import pandas as pd
from fastapi.testclient import TestClient

from ecomsearch.api import routes_image
from ecomsearch.api.app import app

METADATA_COLUMNS = ["item_id", "display name", "category", "image"]


def test_search_image_returns_results(monkeypatch, tmp_path):
    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)
    monkeypatch.setattr(routes_image, "image_search", lambda query, top_k: [(501, 0.91)])

    client = TestClient(app)
    response = client.get("/search/image", params={"q": "red bicycle"})

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["item_id"] == 501
    assert data["results"][0]["display_name"] == "Red Bicycle"
    assert data["results"][0]["image_url"] == "/images/501"


def test_get_image_returns_404_for_unknown_item(monkeypatch, tmp_path):
    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)

    client = TestClient(app)
    response = client.get("/images/99999")

    assert response.status_code == 404


def test_get_image_returns_file_for_known_item(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "501.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)
    monkeypatch.setattr(routes_image, "DATASET_IMAGES_DIR", image_dir)

    client = TestClient(app)
    response = client.get("/images/501")

    assert response.status_code == 200
    assert response.content == b"\xff\xd8\xff\xe0fake-jpeg-bytes"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_image.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.api.routes_image'`

- [ ] **Step 3: Write `src/ecomsearch/api/routes_image.py`**

```python
"""FastAPI routes for multimodal (image) search."""
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ecomsearch.api.schemas import ImageSearchResponse, ImageSearchResult
from ecomsearch.multimodal.config import DATASET_IMAGES_DIR, DEFAULT_TOP_K, SUBSET_METADATA_PATH
from ecomsearch.multimodal.search import image_search

router = APIRouter()

_metadata = None


def _get_metadata() -> pd.DataFrame:
    global _metadata
    if _metadata is None:
        _metadata = pd.read_csv(SUBSET_METADATA_PATH).set_index("item_id")
    return _metadata


@router.get("/search/image", response_model=ImageSearchResponse)
def search_image(q: str, top_k: int = DEFAULT_TOP_K) -> ImageSearchResponse:
    results = image_search(q, top_k)
    metadata = _get_metadata()

    items = []
    for item_id, score in results:
        row = metadata.loc[item_id]
        items.append(
            ImageSearchResult(
                item_id=item_id,
                display_name=str(row["display name"]),
                category=str(row["category"]),
                score=score,
                image_url=f"/images/{item_id}",
            )
        )

    return ImageSearchResponse(query=q, results=items)


@router.get("/images/{item_id}")
def get_image(item_id: int) -> FileResponse:
    metadata = _get_metadata()
    if item_id not in metadata.index:
        raise HTTPException(status_code=404, detail=f"No image found for item_id {item_id}")

    image_filename = metadata.loc[item_id, "image"]
    image_path = DATASET_IMAGES_DIR / image_filename
    return FileResponse(image_path)
```

- [ ] **Step 4: Wire the router into the app**

Modify `src/ecomsearch/api/app.py`: add this import next to the text router import:
```python
from ecomsearch.api.routes_image import router as image_router
```
And add this line right after `app.include_router(text_router)`:
```python
app.include_router(image_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_api_image.py tests/test_api_app.py tests/test_api_text.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/api/routes_image.py src/ecomsearch/api/app.py tests/test_api_image.py
git commit -m "feat: add /search/image and /images/{item_id} API routes"
git push origin main
```

---

### Task 6: End-to-end integration tests (real models)

**Files:**
- Create: `tests/test_api_integration.py`

These use real models and real indexes (no mocking), mirroring how `tests/test_integration.py` and `tests/test_multimodal_integration.py` already work. They will be slow (loading real models on first `with TestClient(app):` entry) but prove the full wiring — routes, schemas, and real `search.py`/`multimodal/search.py` — actually works end to end.

- [ ] **Step 1: Write `tests/test_api_integration.py`**

```python
from fastapi.testclient import TestClient

from ecomsearch.api.app import app


def test_search_text_end_to_end_returns_relevant_result():
    with TestClient(app) as client:
        response = client.get("/search/text", params={"q": "organic almond milk", "top_k": 5})

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert any("almond" in r["name"].lower() for r in data["results"])


def test_search_image_end_to_end_returns_results():
    with TestClient(app) as client:
        response = client.get("/search/image", params={"q": "shoes", "top_k": 5})

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
```

- [ ] **Step 2: Run and verify**

Run: `pytest tests/test_api_integration.py -v`
Expected: PASS (2 tests). This will take longer than the mocked tests (real model loading via the lifespan warm-up) — if it takes more than a couple of minutes, that's unusual given Phase 4b's caching work; check `Get-Process python | Select-Object Id,CPU,StartTime` to confirm it's still actively computing rather than stuck.

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass — Phase 1-4b's 60 plus this phase's new tests (multimodal search x3, api app x2, api text x3, api image x3, api integration x2 = 13) = 73 total.

- [ ] **Step 4: Commit**

```bash
git add tests/test_api_integration.py
git commit -m "test: add end-to-end integration tests for the search API"
git push origin main
```

---

### Task 7: Streamlit frontend

**Files:**
- Create: `src/ecomsearch/ui/__init__.py`
- Create: `src/ecomsearch/ui/streamlit_app.py`

This is a UI-layer task without automated tests, consistent with this project's approach to frontend work — verify manually by actually running both apps (see Step 3).

- [ ] **Step 1: Create the empty package marker**

Create `src/ecomsearch/ui/__init__.py` with empty content (0 bytes).

- [ ] **Step 2: Write `src/ecomsearch/ui/streamlit_app.py`**

```python
"""Streamlit frontend for the E-Commerce Semantic Search API."""
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="E-Commerce Semantic Search", layout="wide")
st.title("E-Commerce Semantic Search")

text_tab, image_tab = st.tabs(["Text Search", "Image Search"])

with text_tab:
    query = st.text_input("Search the catalog", key="text_query")
    mode = st.selectbox(
        "Mode",
        ["hybrid", "hybrid-rerank", "dense", "bm25"],
        index=0,
        help="hybrid-rerank is slower (cross-encoder reranking) but can be more precise.",
    )
    top_k = st.number_input("Results", min_value=1, max_value=50, value=10, key="text_top_k")

    if st.button("Search", key="text_search_button") and query:
        spinner_text = (
            "Searching (reranking, this takes a few seconds)..."
            if mode == "hybrid-rerank"
            else "Searching..."
        )
        with st.spinner(spinner_text):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/search/text",
                    params={"q": query, "mode": mode, "top_k": top_k},
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                st.error(f"Could not reach the search API: {e}")
            else:
                results = response.json()["results"]
                if not results:
                    st.info("No results found.")
                else:
                    st.table(
                        [
                            {
                                "Rank": i + 1,
                                "Name": r["name"],
                                "Brand": r["brand"],
                                "Category": r["category_path"],
                                "Score": round(r["score"], 4),
                            }
                            for i, r in enumerate(results)
                        ]
                    )

with image_tab:
    image_query = st.text_input("Search product images", key="image_query")
    image_top_k = st.number_input("Results", min_value=1, max_value=50, value=10, key="image_top_k")

    if st.button("Search", key="image_search_button") and image_query:
        with st.spinner("Searching..."):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/search/image",
                    params={"q": image_query, "top_k": image_top_k},
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                st.error(f"Could not reach the search API: {e}")
            else:
                results = response.json()["results"]
                if not results:
                    st.info("No results found.")
                else:
                    columns = st.columns(5)
                    for i, r in enumerate(results):
                        with columns[i % 5]:
                            st.image(
                                f"{API_BASE_URL}{r['image_url']}",
                                caption=f"{r['display_name']} ({r['category']}) — {r['score']:.4f}",
                            )
```

- [ ] **Step 3: Manually verify both tabs work end to end**

In one terminal, start the backend:
```bash
venv/Scripts/python.exe -m uvicorn ecomsearch.api.app:app --reload
```
Wait for it to log that startup is complete (the lifespan warm-up will take a while the first time — real models loading). In a second terminal, start the frontend:
```bash
venv/Scripts/streamlit.exe run src/ecomsearch/ui/streamlit_app.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`). Verify:
- Text Search tab: search for `"almond milk"` with mode `hybrid` — confirm results appear in a table with plausible names/brands/scores. Try `mode = hybrid-rerank` and confirm the slower spinner message appears and results still return (allow several seconds).
- Image Search tab: search for a term like `"shoes"` or `"bicycle"` — confirm an image grid renders with real images, captions, and scores.
- Stop the backend (Ctrl+C) and try a search in either tab — confirm `st.error(...)` shows a clear "could not reach the API" message instead of crashing.

- [ ] **Step 4: Commit**

```bash
git add src/ecomsearch/ui/__init__.py src/ecomsearch/ui/streamlit_app.py
git commit -m "feat: add Streamlit frontend for text and image search"
git push origin main
```

---

### Task 8: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass — 73 total (per Task 6).

- [ ] **Step 2: Update `README.md`**

Read the current file first, then:
- Check off `- [ ] Phase 5 — Serving layer (FastAPI + Streamlit)` in the Status section, and update the status paragraph to mention it, e.g.: "A FastAPI backend and Streamlit frontend now serve both text and image search over HTTP — see Setup below for how to run them locally."
- Add a new `## Running the App` section (after `## Setup`, before any existing usage/CLI documentation) with the two commands from Task 7 Step 3 (`uvicorn ecomsearch.api.app:app --reload` and `streamlit run src/ecomsearch/ui/streamlit_app.py`), a one-line description of what each serves, and a note that `API_BASE_URL` (default `http://localhost:8000`) controls where the Streamlit app looks for the backend.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: add Phase 5 serving layer usage to README"
git push origin main
```
