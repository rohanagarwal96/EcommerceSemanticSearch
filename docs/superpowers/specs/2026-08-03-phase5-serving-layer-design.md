# Phase 5: Serving Layer — Design

## Context

Phases 1-4 are complete and merged: text embedding baseline (bge-small-en-v1.5 + FAISS), multimodal CLIP demo, hybrid retrieval (BM25 + dense + RRF fusion + cross-encoder rerank), evaluation (35 hand-labeled queries, results in `docs/eval_results.md`), and latency engineering (per-process caching + parallelized dense/bm25 search, results in `docs/latency_results.md`). Everything so far is consumed via CLI (`ecomsearch search ...`, `ecomsearch-images search ...`) or directly in scripts/tests — there is no running service.

Per the original project brief, Phase 5 builds the serving layer: FastAPI (backend) and Streamlit (frontend), per the `$0` infra stack table. This is the first phase where `search.py`'s per-process caching (module-level lazy singletons, built in Phase 4b specifically for a long-lived process) actually pays off — the CLI's one-shot-process-per-invocation usage never benefited from it.

Phase 6 (Qdrant Cloud + Hugging Face Spaces deployment) and Phase 7 (CI, logging, rate limiting) come after this and are explicitly out of scope here.

## Goals

1. A FastAPI backend exposing both text search (the main catalog, all 4 retrieval modes) and multimodal image search (Phase 2's CLIP demo dataset), with caches pre-warmed at startup.
2. A Streamlit frontend that talks to the backend exclusively over HTTP (never imports `ecomsearch.search`/`ecomsearch.multimodal` directly) — matching the eventual two-service Phase 6 deployment shape.
3. Both apps runnable locally (`uvicorn`, `streamlit run`) with no code changes needed later to point at a deployed backend.

## Non-Goals

- Dockerfiles, containerization, or any deployment configuration — Phase 6.
- CI, structured logging, rate limiting, auth — Phase 7.
- Changing `search.py`'s or the multimodal module's core retrieval logic — this phase wraps and serves existing functionality, it doesn't change it.
- Production-grade concurrency tuning beyond what's already in place (see the `ThreadPoolExecutor` note below) — acceptable for a demo-scale service now, revisited later if needed.

## Design

### 1. Architecture

**Gap found while writing this spec:** `ecomsearch.multimodal` has no equivalent of `ecomsearch/search.py` — `multimodal/cli.py`'s `search()` function inlines index-loading, embedding, and result-formatting together with CLI-only side effects (copying matched image files to `demo_results/`), and none of it is cached. The API needs a pure, cacheable function it can call repeatedly without side effects, so this phase adds `src/ecomsearch/multimodal/search.py`, mirroring `search.py`'s exact pattern from Phase 4b: module-level lazy singletons for the CLIP index and `ClipEmbedder`, and a pure `image_search(query: str, top_k: int) -> list[tuple[int, float]]` function. `multimodal/cli.py` is left as-is (still works standalone) — refactoring it to reuse this new module is a nice-to-have, not required for this phase, and shouldn't be done as unplanned scope creep.

Two independent processes:
- **Backend**: `src/ecomsearch/api/app.py`, a FastAPI app wrapping `ecomsearch.search` (text) and `ecomsearch.multimodal` (images).
- **Frontend**: `src/ecomsearch/ui/streamlit_app.py`, a Streamlit app that calls the backend via `requests`, using an `API_BASE_URL` environment variable (default `http://localhost:8000`) so the same code points at a deployed backend later without changes.

The FastAPI app uses a `lifespan` startup hook to pre-warm all caches (dense index, BM25 index, embedder, reranker, catalog from `search.py`; the CLIP index/embedder from the multimodal module) before accepting traffic, so the first real request isn't slow.

All search route handlers are plain `def` (not `async def`). `search.py` and the multimodal module are fully synchronous/CPU-bound (FAISS, BM25, transformer inference); Starlette automatically dispatches sync route handlers to its own internal thread pool, keeping the event loop unblocked with zero changes to existing code. `hybrid_search`'s own internal `ThreadPoolExecutor(max_workers=2)` (from Phase 4b) is a separate, independent pool, so nesting is safe — no deadlock risk. Known, accepted limitation: since that executor is a shared 2-worker singleton, concurrent requests queue for it rather than scaling independently. Fine for a demo; revisit only if real production load becomes a goal (Phase 7 territory at the earliest, not this phase).

### 2. FastAPI design

**Endpoints:**
- `GET /search/text?q=...&mode=hybrid&top_k=10` — `mode` defaults to `hybrid` (fast, no rerank — `hybrid-rerank`'s ~6.2s p95 is too slow for a good default interactive experience per `docs/latency_results.md`), validated against `dense`/`bm25`/`hybrid`/`hybrid-rerank` via a `Literal` type (invalid mode → automatic 422).
- `GET /search/image?q=...&top_k=10` — text-to-image search over the multimodal CLIP index.
- `GET /images/{item_id}` — streams the image file for a multimodal result (`FileResponse`, content-type inferred from the file). 404 if `item_id` isn't in the multimodal subset.
- `GET /health` — trivial liveness check.

**Response schemas** (Pydantic):
```python
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
    image_url: str  # points at /images/{item_id}

class ImageSearchResponse(BaseModel):
    query: str
    results: list[ImageSearchResult]
```

**Testing:** FastAPI's `TestClient` against the real app, with `search.py`'s functions (and the multimodal equivalents) monkeypatched at the route-module level for fast unit tests — the same pattern `tests/test_cli.py` already uses for `cli.dense_search`/etc. A small number of real end-to-end tests (actual models, actual indexes) verify the wiring, mirroring `tests/test_integration.py`/`tests/test_multimodal_integration.py`.

### 3. Streamlit UI

Single app, two tabs via `st.tabs(["Text Search", "Image Search"])`.

**Text Search tab:** search box, mode selector (default `hybrid`; `hybrid-rerank` available, visibly labeled as slower), top_k input, results as a table (rank, name, brand, category, score). `st.spinner(...)` while the request is in flight.

**Image Search tab:** search box, top_k input, results as an image grid — `st.image()` pointed directly at each result's `/images/{item_id}` URL (Streamlit fetches it, no manual download needed) — with display name/category/score as captions.

Both tabs handle "no results" and "API unreachable" with a plain `st.error(...)` rather than crashing.

## File Summary

| File | Change |
|---|---|
| `src/ecomsearch/multimodal/search.py` | New — cached `image_search()`, mirroring `ecomsearch/search.py`'s pattern |
| `src/ecomsearch/api/__init__.py` | New — package marker |
| `src/ecomsearch/api/app.py` | New — FastAPI app, lifespan startup warm-up, route registration |
| `src/ecomsearch/api/schemas.py` | New — Pydantic request/response models |
| `src/ecomsearch/api/routes_text.py` | New — `/search/text` route |
| `src/ecomsearch/api/routes_image.py` | New — `/search/image`, `/images/{item_id}` routes |
| `src/ecomsearch/ui/streamlit_app.py` | New — Streamlit frontend |
| `tests/test_api_text.py` | New — text search route tests (mocked search functions) |
| `tests/test_api_image.py` | New — image search + image-serving route tests (mocked) |
| `tests/test_api_integration.py` | New — small number of real end-to-end tests |
| `pyproject.toml` | Modify — add `fastapi`, `uvicorn`, `streamlit`, `requests`, `httpx` (for `TestClient`) dependencies |
| `README.md` | Modify — document how to run both apps locally |
