# Phase 7 Production Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CI (lint + full test suite via GitHub Actions), structured logging, and rate limiting to the FastAPI backend, without changing Phase 6's local-only deployment model.

**Architecture:** Adopt Ruff for lint/format (config + a one-time cleanup of the small number of pre-existing violations), add a GitHub Actions workflow that runs it plus the full pytest suite on every push/PR, add structlog-based JSON logging to the backend (request middleware + per-route search events + exception logging), and add slowapi-based per-IP rate limiting to the two search endpoints.

**Tech Stack:** Ruff, GitHub Actions, structlog, slowapi.

---

### Task 1: Adopt Ruff for linting and formatting

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `data/genericize_catalog.py` (2 lines, manual wrap)
- Modify: up to ~47 files reformatted by `ruff format` (mechanical, no logic changes)

- [ ] **Step 1: Add the `[tool.ruff]` config to `pyproject.toml`**

Add this block anywhere in `pyproject.toml` (e.g. after `[tool.pytest.ini_options]`):
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```

- [ ] **Step 2: Add `ruff` to `requirements.txt`**

Add this line (anywhere, e.g. near `pytest>=8.0.0`):
```
ruff>=0.6.0
```

- [ ] **Step 3: Install ruff and confirm the current violation count**

Run:
```bash
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe -m ruff check .
venv/Scripts/python.exe -m ruff format --check .
```
Expected (as of this plan being written): `ruff check .` reports 15 errors (13 `I001` unsorted-imports, auto-fixable; 2 `E501` line-too-long in `data/genericize_catalog.py`, not auto-fixable). `ruff format --check .` reports 47 files would be reformatted. If the actual numbers differ because other code changed since this plan was written, that's fine — just work through whatever `ruff check`/`ruff format --check` actually report using the same fix approach below.

- [ ] **Step 4: Apply `ruff format` across the codebase**

Run:
```bash
venv/Scripts/python.exe -m ruff format .
```
This is a purely mechanical reformat (whitespace, quote style, line wrapping) — no logic changes. Expect a large diff across many files; that's expected for a one-time formatter adoption.

- [ ] **Step 5: Auto-fix the import-sort violations**

Run:
```bash
venv/Scripts/python.exe -m ruff check --fix .
```
This should resolve the `I001` (unsorted-imports) violations automatically.

- [ ] **Step 6: Manually fix the two `E501` (line-too-long) violations in `data/genericize_catalog.py`**

Replace:
```python
text = re.sub(
    r"Food\s+You\s+Feel\s+Good\s+About", GENERIC_QUALITY_BANNER, text, flags=re.IGNORECASE
)
```
with:
```python
    text = re.sub(
        r"Food\s+You\s+Feel\s+Good\s+About", GENERIC_QUALITY_BANNER, text, flags=re.IGNORECASE
    )
```

Replace:
```python
tags_str = re.sub(
    r"Food\s+You\s+Feel\s+Good\s+About", GENERIC_QUALITY_BANNER, tags_str, flags=re.IGNORECASE
)
```
with:
```python
    tags_str = re.sub(
        r"Food\s+You\s+Feel\s+Good\s+About", GENERIC_QUALITY_BANNER, tags_str, flags=re.IGNORECASE
    )
```

- [ ] **Step 7: Confirm both ruff checks are now clean**

Run:
```bash
venv/Scripts/python.exe -m ruff check .
venv/Scripts/python.exe -m ruff format --check .
```
Expected: both report no issues (`All checks passed!` / no files needing reformatting).

- [ ] **Step 8: Run the full test suite to confirm nothing broke**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: same pass/skip counts as before this task (83 passed, 3 skipped) — formatting and import-sorting are behavior-preserving.

- [ ] **Step 9: Commit**

```bash
git add -u
git commit -m "chore: adopt Ruff for linting and formatting, fix pre-existing violations"
git push origin main
```
(`git add -u` stages every change to already-tracked files — `pyproject.toml`, `requirements.txt`, every file `ruff format`/`ruff check --fix` touched, and the manual `data/genericize_catalog.py` edit. No new untracked files are created by this task, so `-u` alone covers everything.)

---

### Task 2: Add GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Ruff check
        run: ruff check .
      - name: Ruff format check
        run: ruff format --check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      - name: Cache Hugging Face models
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: ${{ runner.os }}-hf-models-v1
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run tests
        run: pytest -v
```

Notes for whoever implements this: no `QDRANT_URL` is set in this workflow, so the 3 real-Qdrant tests will skip cleanly (per Phase 6's `.env.example` fix making empty/unset the correct default). No `.env` file exists in CI at all — every test that needs config falls back to `os.environ.get(...)`'s `None`/default, which is the same behavior already relied on for the Qdrant-test skip.

- [ ] **Step 2: Validate the YAML parses**

Run:
```bash
venv/Scripts/python.exe -c "
import yaml
with open('.github/workflows/ci.yml') as f:
    doc = yaml.safe_load(f)
assert set(doc['jobs']) == {'lint', 'test'}
print('ci.yml OK')
"
```
Expected: `ci.yml OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "feat: add GitHub Actions CI workflow (lint + full test suite)"
git push origin main
```

(Real verification that the workflow actually succeeds on GitHub is Task 6, after logging and rate limiting are added — no need to block on a green run here since more commits are coming in this same plan.)

---

### Task 3: Add structlog request-logging middleware and exception logging

**Files:**
- Modify: `requirements.txt`
- Modify: `src/ecomsearch/api/app.py`
- Test: `tests/test_api_app.py`

- [ ] **Step 1: Add `structlog` to `requirements.txt`**

```
structlog>=24.1.0
```
Then: `venv/Scripts/python.exe -m pip install -r requirements.txt`

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_api_app.py`:
```python
import structlog
from fastapi.testclient import TestClient

from ecomsearch.api import app as app_module


def test_request_middleware_logs_method_path_status_and_duration():
    client = TestClient(app_module.app)

    with structlog.testing.capture_logs() as captured:
        client.get("/health")

    request_logs = [e for e in captured if e.get("event") == "request_completed"]
    assert len(request_logs) == 1
    assert request_logs[0]["method"] == "GET"
    assert request_logs[0]["path"] == "/health"
    assert request_logs[0]["status_code"] == 200
    assert "duration_ms" in request_logs[0]


def test_unhandled_exception_is_logged_before_500_response(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "dense_search",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    client = TestClient(app_module.app, raise_server_exceptions=False)

    with structlog.testing.capture_logs() as captured:
        response = client.get("/search/text", params={"q": "anything", "mode": "dense"})

    assert response.status_code == 500
    error_logs = [e for e in captured if e.get("event") == "unhandled_exception"]
    assert len(error_logs) == 1
    assert error_logs[0]["path"] == "/search/text"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_app.py -v`
Expected: FAIL — no `request_completed`/`unhandled_exception` events exist yet (no logging configured at all).

- [ ] **Step 4: Configure structlog and add the middleware + exception handler to `src/ecomsearch/api/app.py`**

Replace the full file with:
```python
"""FastAPI application: serving layer for text and image product search."""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ecomsearch.api.routes_image import router as image_router
from ecomsearch.api.routes_text import router as text_router
from ecomsearch.multimodal.search import image_search
from ecomsearch.search import bm25_search, dense_search, hybrid_search

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def _warm_up_caches() -> None:
    dense_search("warm up", top_k=1)
    bm25_search("warm up", top_k=1)
    hybrid_search("warm up", top_k=1, use_rerank=True)
    image_search("warm up", top_k=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_up_caches()
    logger.info("startup_complete")
    yield


app = FastAPI(title="E-Commerce Semantic Search API", lifespan=lifespan)
app.include_router(text_router)
app.include_router(image_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


@app.exception_handler(Exception)
async def log_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        method=request.method,
        path=request.url.path,
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_app.py -v`
Expected: PASS (4 tests: the 2 new ones plus the 2 pre-existing ones).

- [ ] **Step 6: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (83 + 2 new = 85 passed, 3 skipped). If any existing test unexpectedly starts failing, check whether the new `@app.exception_handler(Exception)` is now catching an exception that a test previously expected to propagate as a raw Python exception (FastAPI's `TestClient` normally re-raises server exceptions by default unless `raise_server_exceptions=False`) — this is expected to be fine since no other test triggers an unhandled exception path, but verify.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/ecomsearch/api/app.py tests/test_api_app.py
git commit -m "feat: add structlog request-logging middleware and exception logging"
git push origin main
```

---

### Task 4: Add per-route structured search-event logging

**Files:**
- Modify: `src/ecomsearch/api/routes_text.py`
- Modify: `src/ecomsearch/api/routes_image.py`
- Test: `tests/test_api_text.py`
- Test: `tests/test_api_image.py`

- [ ] **Step 1: Write the failing test for text search**

Add to `tests/test_api_text.py`:
```python
def test_search_text_logs_a_structured_search_event(monkeypatch, tmp_path):
    import structlog

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
    with structlog.testing.capture_logs() as captured:
        client.get("/search/text", params={"q": "almond milk"})

    search_logs = [e for e in captured if e.get("event") == "text_search_completed"]
    assert len(search_logs) == 1
    assert search_logs[0]["query"] == "almond milk"
    assert search_logs[0]["mode"] == "hybrid"
    assert search_logs[0]["result_count"] == 1
    assert "duration_ms" in search_logs[0]
```

- [ ] **Step 2: Write the failing test for image search**

Add to `tests/test_api_image.py` (it already defines `METADATA_COLUMNS = ["item_id", "display name", "category", "image"]` at module level — reuse it):
```python
def test_search_image_logs_a_structured_search_event(monkeypatch, tmp_path):
    import structlog

    metadata_path = tmp_path / "subset_metadata.csv"
    pd.DataFrame(
        [[501, "Red Bicycle", "Sporting Goods", "501.jpg"]], columns=METADATA_COLUMNS
    ).to_csv(metadata_path, index=False)
    monkeypatch.setattr(routes_image, "SUBSET_METADATA_PATH", metadata_path)
    monkeypatch.setattr(routes_image, "_metadata", None, raising=False)
    monkeypatch.setattr(routes_image, "image_search", lambda query, top_k: [(501, 0.91)])

    client = TestClient(app)
    with structlog.testing.capture_logs() as captured:
        client.get("/search/image", params={"q": "red bicycle"})

    search_logs = [e for e in captured if e.get("event") == "image_search_completed"]
    assert len(search_logs) == 1
    assert search_logs[0]["query"] == "red bicycle"
    assert search_logs[0]["result_count"] == 1
    assert "duration_ms" in search_logs[0]
```

- [ ] **Step 3: Run both tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_text.py tests/test_api_image.py -v`
Expected: the two new tests FAIL (no search-event logging exists yet); existing tests in both files still pass.

- [ ] **Step 4: Add logging to `search_text` in `src/ecomsearch/api/routes_text.py`**

Add `import time` and `import structlog` to the top of the file, add `logger = structlog.get_logger()` near the top-level `router = APIRouter()` line, then wrap the route body:
```python
@router.get("/search/text", response_model=TextSearchResponse)
def search_text(
    q: str,
    mode: Literal["dense", "bm25", "hybrid", "hybrid-rerank"] = "hybrid",
    top_k: int = DEFAULT_TOP_K,
) -> TextSearchResponse:
    start = time.perf_counter()
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

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "text_search_completed",
        query=q,
        mode=mode,
        top_k=top_k,
        result_count=len(items),
        duration_ms=round(duration_ms, 2),
    )
    return TextSearchResponse(query=q, mode=mode, results=items)
```

- [ ] **Step 5: Add logging to `search_image` in `src/ecomsearch/api/routes_image.py`**

Add `import time` and `import structlog` to the top of the file, add `logger = structlog.get_logger()` near the top-level `router = APIRouter()` line, then wrap the route body:
```python
@router.get("/search/image", response_model=ImageSearchResponse)
def search_image(q: str, top_k: int = DEFAULT_TOP_K) -> ImageSearchResponse:
    start = time.perf_counter()
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

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "image_search_completed",
        query=q,
        result_count=len(items),
        duration_ms=round(duration_ms, 2),
    )
    return ImageSearchResponse(query=q, results=items)
```

- [ ] **Step 6: Run both test files to verify the new tests pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_text.py tests/test_api_image.py -v`
Expected: PASS (all tests in both files).

- [ ] **Step 7: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (85 + 2 new = 87 passed, 3 skipped).

- [ ] **Step 8: Commit**

```bash
git add src/ecomsearch/api/routes_text.py src/ecomsearch/api/routes_image.py tests/test_api_text.py tests/test_api_image.py
git commit -m "feat: log structured search events for text and image search routes"
git push origin main
```

---

### Task 5: Add rate limiting to the search endpoints

**Files:**
- Create: `src/ecomsearch/api/limiter.py`
- Modify: `requirements.txt`
- Modify: `src/ecomsearch/api/app.py`
- Modify: `src/ecomsearch/api/routes_text.py`
- Modify: `src/ecomsearch/api/routes_image.py`
- Test: `tests/test_api_rate_limiting.py`

- [ ] **Step 1: Add `slowapi` to `requirements.txt`**

```
slowapi>=0.1.9
```
Then: `venv/Scripts/python.exe -m pip install -r requirements.txt`

- [ ] **Step 2: Create `src/ecomsearch/api/limiter.py`**

A separate module for the shared `Limiter` instance, so both `app.py` and the route modules can import it without a circular import (`app.py` imports the routers, which would otherwise need to import `app.py` for the limiter):
```python
"""Shared rate limiter instance for the FastAPI backend."""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_api_rate_limiting.py`:
```python
import pandas as pd
from fastapi.testclient import TestClient

from ecomsearch.api import routes_text
from ecomsearch.api.app import app
from ecomsearch.api.limiter import limiter


def test_search_text_returns_429_after_30_requests_per_minute(monkeypatch, tmp_path):
    limiter.reset()
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
    for _ in range(30):
        response = client.get("/search/text", params={"q": "almond milk"})
        assert response.status_code == 200

    response = client.get("/search/text", params={"q": "almond milk"})
    assert response.status_code == 429
    limiter.reset()
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_rate_limiting.py -v`
Expected: FAIL — the 31st request currently still returns 200 (no rate limiting configured yet).

- [ ] **Step 5: Wire the limiter into `src/ecomsearch/api/app.py`**

Add these imports:
```python
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ecomsearch.api.limiter import limiter
```
Add after the `app = FastAPI(...)` line and router includes:
```python
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

- [ ] **Step 6: Add the rate-limit decorator to `search_text` in `src/ecomsearch/api/routes_text.py`**

Add `from fastapi import APIRouter, Request` (adding `Request` to the existing `fastapi` import) and `from ecomsearch.api.limiter import limiter`, then decorate and add the `request: Request` parameter:
```python
@router.get("/search/text", response_model=TextSearchResponse)
@limiter.limit("30/minute")
def search_text(
    request: Request,
    q: str,
    mode: Literal["dense", "bm25", "hybrid", "hybrid-rerank"] = "hybrid",
    top_k: int = DEFAULT_TOP_K,
) -> TextSearchResponse:
```
(body unchanged — only the decorator and the new `request: Request` first parameter are added; slowapi's `Limiter.limit` decorator requires a `Request`-typed parameter on the decorated function to read the client's IP.)

- [ ] **Step 7: Add the same rate-limit decorator to `search_image` in `src/ecomsearch/api/routes_image.py`**

Add `from fastapi import APIRouter, HTTPException, Request` (adding `Request`) and `from ecomsearch.api.limiter import limiter`, then:
```python
@router.get("/search/image", response_model=ImageSearchResponse)
@limiter.limit("30/minute")
def search_image(request: Request, q: str, top_k: int = DEFAULT_TOP_K) -> ImageSearchResponse:
```
(body unchanged. Do NOT add rate limiting to `get_image` / `/images/{item_id}` — per the spec, thumbnails aren't throttled.)

- [ ] **Step 8: Run the new test to verify it passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_api_rate_limiting.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (87 + 1 new = 88 passed, 3 skipped). If any *other* existing test now fails with a 429 (because it makes many rapid requests to the same endpoint within one test session and trips the new limit), add a `limiter.reset()` call at the start of that test — the rate limiter's in-memory storage persists across tests within the same pytest process.

- [ ] **Step 10: Commit**

```bash
git add src/ecomsearch/api/limiter.py requirements.txt src/ecomsearch/api/app.py src/ecomsearch/api/routes_text.py src/ecomsearch/api/routes_image.py tests/test_api_rate_limiting.py
git commit -m "feat: add per-IP rate limiting to search endpoints via slowapi"
git push origin main
```

---

### Task 6: Verify the CI workflow succeeds for real

**Files:** none (verification only)

The `gh` CLI is not installed in this environment — do not attempt to install new CLI tools for this task. Use the GitHub REST API directly instead (works unauthenticated for a public repo, subject to GitHub's unauthenticated rate limit).

- [ ] **Step 1: Confirm the workflow ran on the most recent push**

Get the current HEAD commit SHA (`git rev-parse HEAD`), then check its status via the REST API:
```bash
curl -s "https://api.github.com/repos/rohanagarwal96/EcommerceSemanticSearch/commits/$(git rev-parse HEAD)/check-runs"
```
Expected: JSON listing check runs named `lint` and `test` for this commit. If the response is empty or the runs aren't there yet, GitHub Actions may not have picked up the push yet — wait ~15-30 seconds and retry.

- [ ] **Step 2: Poll until both jobs complete**

Repeat the same request every 15-20 seconds until every check run's `"status"` field is `"completed"` (not `"in_progress"` or `"queued"`). Do not sleep for more than a couple of minutes total without checking — if a job seems stuck well beyond the normal build time (heavy `torch`/`transformers`/`faiss-cpu` installs mean a first, uncached run could reasonably take several minutes), that's fine to keep waiting on, but report back if it exceeds roughly 15 minutes.

- [ ] **Step 3: Check the conclusion**

Once completed, check each check run's `"conclusion"` field. Expected: `"success"` for both `lint` and `test`. If either is `"failure"`, fetch its logs via the URL in the JSON response's `"details_url"` field (a browser-viewable Actions run page — if you can't fetch it directly, report the run URL and the failing job name/step so the failure can be diagnosed and fixed) before re-pushing and re-verifying. Don't disable or skip the failing check as a workaround.

- [ ] **Step 4: Report final status**

Summarize: the commit SHA checked, both jobs' conclusions, and the run URL (constructable as `https://github.com/rohanagarwal96/EcommerceSemanticSearch/actions/runs/<run_id>`, where `run_id` is in the check-runs response). No commit needed for this task unless a fix was required (in which case follow the same commit conventions as prior tasks).

---

### Task 7: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (88 passed, 3 skipped).

- [ ] **Step 2: Update the Status section**

Check off Phase 7:
```
- [x] Phase 7 — Production hygiene (CI, logging, rate limiting)
```
Update the Status paragraph's last sentence (currently "Phases 7-8 in progress; this section will be updated as each phase lands.") to:
```
Phase 8 in progress; this section will be updated as it lands.
```

- [ ] **Step 3: Update the Stack table's CI/CD row**

Replace:
```
| CI/CD | Planned (Phase 7) |
```
with:
```
| CI/CD | GitHub Actions (lint + full test suite on every push) |
```

- [ ] **Step 4: Add a short "Production hygiene" note**

Add a new subsection after "Running the App" and before "Known limitations":
```
## Production hygiene

- **CI**: every push/PR runs Ruff (lint + format check) and the full pytest
  suite via GitHub Actions (`.github/workflows/ci.yml`).
- **Logging**: the FastAPI backend emits structured JSON logs to stdout
  (via `structlog`) for every request and every search, plus stack traces
  for unhandled exceptions — viewable with `docker compose logs backend`.
- **Rate limiting**: `/search/text` and `/search/image` are limited to 30
  requests/minute per client IP (via `slowapi`); exceeding it returns
  HTTP 429. `/health` and `/images/{item_id}` are unaffected.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README for Phase 7 production hygiene"
git push origin main
```
