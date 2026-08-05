# Phase 6 Local Docker Compose Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all remaining cloud-deployment work (HF Spaces, Render) with a local Docker Compose stack — Qdrant, FastAPI backend, Streamlit frontend — brought up with a single `docker compose up`.

**Architecture:** Strip out everything tied to the abandoned cloud paths (HF dataset-artifact hosting, the backend's HF-download bootstrap, the Render Blueprint, the HF-Space frontend deploy script), then add `docker-compose.yml` wiring the three services together with local bind-mounts replacing what used to be a remote download. `Dockerfile.api`/`Dockerfile.ui` are reused unchanged.

**Tech Stack:** Docker Compose, official `qdrant/qdrant` image, existing FastAPI/Streamlit Dockerfiles, pytest.

---

### Task 1: Remove HF dataset-artifact-hosting machinery

**Files:**
- Delete: `scripts/upload_artifacts_to_hf.py`
- Delete: `tests/test_upload_artifacts_to_hf.py`
- Modify: `src/ecomsearch/api/app.py`
- Modify: `tests/test_api_app.py`
- Modify: `src/ecomsearch/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Delete the HF dataset-upload script and its test**

```bash
rm -f scripts/upload_artifacts_to_hf.py tests/test_upload_artifacts_to_hf.py
```

- [ ] **Step 2: Rewrite `src/ecomsearch/api/app.py` to remove the artifact-bootstrap step**

Replace the entire file with:

```python
"""FastAPI application: serving layer for text and image product search."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ecomsearch.api.routes_image import router as image_router
from ecomsearch.api.routes_text import router as text_router
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
app.include_router(text_router)
app.include_router(image_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 3: Remove the now-obsolete tests from `tests/test_api_app.py`**

Replace the entire file with:

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

- [ ] **Step 4: Remove `HF_TOKEN`, `HF_DATASET_REPO`, `HF_SPACE_FRONTEND` from `src/ecomsearch/config.py`**

Delete these three lines (currently the last three lines of the file):
```python
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
HF_SPACE_FRONTEND = os.environ.get("HF_SPACE_FRONTEND")
```
So the file now ends with:
```python
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "faiss")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "ecommerce_products")
```

- [ ] **Step 5: Remove the Hugging Face block from `.env.example`**

Delete this entire block (currently the first 5 lines):
```
# Hugging Face (account + write token, used for model downloads, artifact
# hosting, and Space deployment)
HF_TOKEN=your_huggingface_write_token_here
HF_SPACE_FRONTEND=your-hf-username/ecommerce-search-ui
HF_DATASET_REPO=your-hf-username/ecommerce-search-artifacts

```
So the file now starts directly with the `# Qdrant Cloud (free tier cluster)` section (this section gets rewritten in Task 3 — just remove the HF block here, don't touch the Qdrant section yet).

- [ ] **Step 6: Confirm nothing else references the removed names**

Run: `grep -rn "HF_TOKEN\|HF_DATASET_REPO\|HF_SPACE_FRONTEND\|snapshot_download\|upload_artifacts_to_hf" src/ tests/ scripts/ .env.example`
Expected: no output.

- [ ] **Step 7: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add -A -- scripts/upload_artifacts_to_hf.py tests/test_upload_artifacts_to_hf.py src/ecomsearch/api/app.py tests/test_api_app.py src/ecomsearch/config.py .env.example
git commit -m "chore: remove HF dataset-artifact hosting in favor of local bind mounts"
git push origin main
```

---

### Task 2: Remove the Render Blueprint and HF-Space frontend deploy script

**Files:**
- Delete: `render.yaml`

> **Note (updated after Task 1 executed):** `scripts/deploy_frontend_space.py`
> and `tests/test_deploy_frontend_space.py` were already deleted as part of
> Task 1 (they imported `HF_TOKEN`/`HF_SPACE_FRONTEND`, which Task 1 removed
> from `config.py`, so their deletion had to be pulled forward to keep the
> test suite green). This task now only needs to remove `render.yaml`.

- [ ] **Step 1: Delete `render.yaml`**

```bash
git rm render.yaml
```

- [ ] **Step 2: Confirm nothing else references it**

Run: `grep -rln "render.yaml" src/ tests/ scripts/ | grep -v docs/`
Expected: no output.

- [ ] **Step 3: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove Render Blueprint in favor of local Docker Compose"
git push origin main
```

---

### Task 3: Point Qdrant config at a local container by default

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Rewrite the Qdrant section of `.env.example`**

Replace:
```
# Qdrant Cloud (free tier cluster)
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=ecommerce_products
QDRANT_IMAGE_COLLECTION_NAME=ecommerce_products_images
```
with:
```
# Qdrant. Defaults point at the local container started by `docker compose
# up` (see docker-compose.yml) -- no API key needed for a local, unauthenticated
# instance. Point QDRANT_URL/QDRANT_API_KEY at a cloud cluster instead if you
# have one.
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=ecommerce_products
QDRANT_IMAGE_COLLECTION_NAME=ecommerce_products_images
```

- [ ] **Step 2: Also update the `VECTOR_BACKEND` comment for accuracy**

Replace:
```
# Vector search backend: "faiss" for local dev/tests (default), "qdrant" for
# production. Only production containers should set this to "qdrant".
VECTOR_BACKEND=faiss
```
with:
```
# Vector search backend: "faiss" for local dev/tests (default), "qdrant" to
# use the Qdrant container from docker-compose.yml instead.
VECTOR_BACKEND=faiss
```

- [ ] **Step 3: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (this is a docs/example-config-only change, no code touched).

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "docs: point .env.example's Qdrant defaults at a local container"
git push origin main
```

---

### Task 4: Write `docker-compose.yml`

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Write `docker-compose.yml` at the repo root**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  backend:
    build:
      context: .
      dockerfile: Dockerfile.api
    depends_on:
      - qdrant
    environment:
      VECTOR_BACKEND: qdrant
      QDRANT_URL: http://qdrant:6333
      QDRANT_COLLECTION_NAME: ${QDRANT_COLLECTION_NAME:-ecommerce_products}
      QDRANT_IMAGE_COLLECTION_NAME: ${QDRANT_IMAGE_COLLECTION_NAME:-ecommerce_products_images}
    ports:
      - "8000:7860"
    volumes:
      - ./data:/app/data:ro
      - ./artifacts:/app/artifacts:ro

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.ui
    depends_on:
      - backend
    environment:
      API_BASE_URL: http://backend:7860
    ports:
      - "8501:7860"

volumes:
  qdrant_storage:
```

Notes for whoever implements this: `backend`'s `VECTOR_BACKEND`/`QDRANT_URL` are hardcoded to `qdrant`/`http://qdrant:6333` (not read from `.env`) because the container must always talk to the Compose-network Qdrant service, regardless of what a host-side `.env` might say for running scripts directly. `QDRANT_COLLECTION_NAME`/`QDRANT_IMAGE_COLLECTION_NAME` use `${VAR:-default}` interpolation so Compose pulls them from a root `.env` file if present, else falls back to the same defaults as `.env.example`. No `QDRANT_API_KEY` is set for `backend` — a local unauthenticated Qdrant container doesn't need one, and `QdrantClient(url=..., api_key=None)` already handles that (confirmed in `src/ecomsearch/qdrant_index.py`).

- [ ] **Step 2: Validate the YAML parses**

Run:
```bash
venv/Scripts/python.exe -c "
import yaml
with open('docker-compose.yml') as f:
    doc = yaml.safe_load(f)
assert set(doc['services']) == {'qdrant', 'backend', 'frontend'}
assert doc['services']['backend']['environment']['VECTOR_BACKEND'] == 'qdrant'
assert doc['services']['frontend']['ports'] == ['8501:7860']
print('docker-compose.yml OK')
"
```
Expected: `docker-compose.yml OK`.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add docker-compose.yml for local Qdrant + backend + frontend stack"
git push origin main
```

---

### Task 5: Populate local Qdrant and verify the full stack for real

**Files:** none (infrastructure verification only)

**Depends on:** Tasks 1-4 (code/config in their final state) and local artifacts already existing (`data/ecommerce_catalog_enriched.csv`, `artifacts/catalog.faiss`, `artifacts/item_ids.npy`, `artifacts/bm25.pkl`, `artifacts/multimodal/...` — already present in this repo from earlier phases, confirmed via `ls`).

- [ ] **Step 1: Confirm Docker is available**

Run: `docker info`
If this fails (e.g. `docker Desktop` not running), STOP and report back that Docker Desktop needs to be started before this task can proceed. Do not attempt to work around a missing Docker daemon.

- [ ] **Step 2: Point the local `.env` at the local Qdrant container**

The real `.env` file (gitignored, contains other secrets like Kaggle credentials — do NOT read, cat, or print its full contents) currently has `QDRANT_URL` and `QDRANT_API_KEY` pointing at the old cloud cluster from earlier phases. Update just those two lines with targeted, non-printing edits:
```bash
sed -i 's|^QDRANT_URL=.*|QDRANT_URL=http://localhost:6333|' .env
sed -i 's|^QDRANT_API_KEY=.*|QDRANT_API_KEY=|' .env
```

- [ ] **Step 3: Build the backend and frontend images**

Run: `docker compose build`
Expected: both images build successfully (likely fast — should hit Docker's build cache from this project's earlier `Dockerfile.api`/`Dockerfile.ui` builds this session, since neither Dockerfile changed).

- [ ] **Step 4: Start Qdrant on its own first**

Run: `docker compose up -d qdrant`
Then poll until it's ready (Qdrant's REST API responds once up):
```bash
for i in $(seq 1 30); do
  curl -s http://localhost:6333/ >/dev/null 2>&1 && echo "qdrant ready" && break
  sleep 2
done
```
Expected: prints `qdrant ready` within the retry loop.

- [ ] **Step 5: Populate the two Qdrant collections from the local FAISS indexes**

Run:
```bash
venv/Scripts/python.exe scripts/upload_index_to_qdrant.py
venv/Scripts/python.exe scripts/upload_multimodal_index_to_qdrant.py
```
Expected: both scripts complete successfully, reporting the same vector counts as the original cloud migration (55,516 text vectors, 4,996 image vectors) — these scripts are unchanged from Phase 6's earlier cloud work, just now pointed at `http://localhost:6333` via the `.env` update in Step 2.

- [ ] **Step 6: Bring up the full stack**

Run: `docker compose up -d`
This starts (or leaves running) `qdrant`, and starts `backend` + `frontend`.

- [ ] **Step 7: Wait for the backend to finish warming up, then verify it**

Poll (model loading takes real time — allow a couple of minutes):
```bash
for i in $(seq 1 60); do
  curl -s http://localhost:8000/health 2>/dev/null | grep -q '"status":"ok"' && echo "backend ready" && break
  sleep 5
done
```
Then:
```bash
curl -s "http://localhost:8000/search/text?q=organic+almond+milk&top_k=2"
curl -s "http://localhost:8000/search/image?q=shoes&top_k=2"
```
Expected: `/health` prints `backend ready`; the text query returns real almond milk products; the image query returns real shoe products (same kind of results seen in this project's earlier local Docker verification).

- [ ] **Step 8: Verify the frontend responds**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501`
Expected: `200`.

- [ ] **Step 9: Bring the stack down, leaving data persisted**

Run: `docker compose down`
(The named `qdrant_storage` volume persists the populated collections — `docker compose up` again later brings everything back without re-running Step 5.)

- [ ] **Step 10: Report final status**

Summarize: Docker availability confirmed, both collections populated with expected counts, backend `/health` + both search endpoints verified with real results, frontend returned HTTP 200, stack brought down cleanly. No commit needed for this task (verification only, `.env` changes are gitignored).

---

### Task 6: Rewrite README for local-only Docker Compose

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (confirms Tasks 1-4's removals left nothing broken).

- [ ] **Step 2: Update the opening tagline (currently line 3-5)**

Replace:
```
A semantic product search engine over a real e-commerce catalog: text and
image (multimodal/CLIP) search that goes beyond exact keyword matching,
targeting sub-200ms latency, deployed live at $0 infrastructure cost.
```
with:
```
A semantic product search engine over a real e-commerce catalog: text and
image (multimodal/CLIP) search that goes beyond exact keyword matching,
targeting sub-200ms latency, fully containerized and runnable locally with
a single `docker compose up`.
```

- [ ] **Step 3: Update the Status section**

Replace the Phase 6 checklist line:
```
- [ ] Phase 6 — Deployment (Qdrant Cloud + Hugging Face Spaces)
```
with:
```
- [x] Phase 6 — Deployment (local Docker Compose: Qdrant + FastAPI + Streamlit)
```
And update the Status paragraph's last two sentences (currently: "A FastAPI backend and Streamlit frontend now serve both text and image search over HTTP — see [Running the App](#running-the-app) below for how to run them locally. Phases 6-8 in progress; this section will be updated as each phase lands.") to:
```
A FastAPI backend and Streamlit frontend now serve both text and image
search over HTTP, either directly via `venv` or as a 3-container Docker
Compose stack (Qdrant + backend + frontend) — see
[Running the App](#running-the-app) below for both options. Phases 7-8
in progress; this section will be updated as each phase lands.
```

- [ ] **Step 4: Update the Stack table**

Replace:
```
| Vector index (dev) | FAISS |
| Vector index (deployed) | Qdrant Cloud (free tier) |
| Keyword search | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend | FastAPI (Docker) |
| Frontend | Streamlit (Docker) |
| Hosting | Hugging Face Spaces (free) |
| CI/CD | GitHub Actions |
```
with:
```
| Vector index (dev) | FAISS |
| Vector index (containerized) | Qdrant (self-hosted via Docker Compose) |
| Keyword search | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend | FastAPI (Docker) |
| Frontend | Streamlit (Docker) |
| Deployment | Local Docker Compose (see [Running the App](#running-the-app)) |
| CI/CD | Planned (Phase 7) |
```

- [ ] **Step 5: Restructure "Running the App" into two subsections**

Replace the entire `## Running the App` section (currently everything from `## Running the App` through the line before `## Known limitations`) with:

```
## Running the App

A FastAPI backend serves all 4 text search modes plus multimodal image
search over HTTP; a Streamlit frontend consumes it. Both ways of running
the app below require the dense, BM25, and multimodal indexes built above.

This project intentionally runs locally rather than as a public cloud
deployment. During Phase 6 we evaluated free-tier cloud hosting options —
Hugging Face Spaces (Docker-SDK Spaces now require a paid PRO plan), Render
(its cheapest tiers with enough RAM for this backend's three-model
footprint start at $85/month), and Google Cloud Run (more setup overhead
than a portfolio demo justifies) — and concluded a small, reproducible
local stack was the better choice for demonstrating the system end-to-end
without ongoing cost.

### Option A: directly via `venv` (FAISS backend)

Start the backend (in one terminal):

```bash
uvicorn ecomsearch.api.app:app --reload
```

Serves the API at `http://localhost:8000` (interactive docs at
`/docs`). Startup pre-warms all search caches, so the first request is
fast — but this makes startup itself slow (real model loads).

Start the frontend (in a second terminal):

```bash
streamlit run src/ecomsearch/ui/streamlit_app.py
```

Serves the UI at `http://localhost:8501`, with tabs for text search and
image search. Set the `API_BASE_URL` environment variable (default
`http://localhost:8000`) to point the frontend at a different backend.

### Option B: Docker Compose (Qdrant + backend + frontend)

Runs the same app as three containers — a local Qdrant instance instead
of FAISS files, plus the backend and frontend. One-time setup to populate
Qdrant from your already-built local indexes:

```bash
docker compose up -d qdrant
python scripts/upload_index_to_qdrant.py
python scripts/upload_multimodal_index_to_qdrant.py
```

Then bring up the full stack:

```bash
docker compose up
```

Frontend at `http://localhost:8501`, backend at `http://localhost:8000`.
Qdrant's data persists in a named Docker volume, so the one-time setup
above only needs to run once per machine — future `docker compose up`
runs reuse the already-populated collections.

<!-- Demo: a short GIF/video of a few example searches (with the latency
number visible) goes here once recorded. -->

```

- [ ] **Step 6: Update "Known limitations"**

Replace:
```
## Known limitations

To be documented as they arise. Note in advance: the Qdrant free-tier
cluster auto-suspends after about a week of inactivity, so a demo visitor
may see a cold-start delay on first query.
```
with:
```
## Known limitations

To be documented as they arise. Note in advance: the backend's startup
pre-warms three ML models (the bge-small embedder, the MiniLM
cross-encoder reranker, and CLIP) plus the BM25 index — the first
`docker compose up` after an image rebuild takes noticeably longer while
these load into memory before the backend reports healthy.
```

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for local-only Docker Compose deployment"
git push origin main
```

---

## Notes for whoever picks up the demo GIF/video

Step 5 of Task 6 leaves an HTML comment placeholder in the README
(`<!-- Demo: ... goes here once recorded -->`) instead of an actual embed,
because recording a screen capture requires a human to run the local stack
and record it — not something automatable in this session. Once the human
provides the file (e.g. `docs/demo.gif`), replace that comment with a real
Markdown image embed and commit as its own small `docs:` commit.
