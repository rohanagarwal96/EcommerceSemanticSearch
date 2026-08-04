# Phase 6: Deployment — Design Spec

## Overview & scope

Phase 6 takes the Phase 5 FastAPI + Streamlit app from "runs on localhost"
to "runs live, free, on the internet." Three things move:

1. The two dense vector indexes (main text catalog + CLIP image subset)
   migrate from local FAISS files to Qdrant Cloud collections.
2. Large runtime artifacts that aren't in git (catalog CSV, BM25 pickle,
   CLIP image dataset + metadata) get hosted in a Hugging Face Hub
   dataset repo and downloaded by the backend container at startup.
3. The backend and frontend each get a Dockerfile and get pushed to their
   own Hugging Face Space via a scripted `huggingface_hub` upload.

Both text search and multimodal (image) search are included in the live
deployment.

CI/CD (auto-deploy on merge) is explicitly out of scope — that belongs to
Phase 7 ("Production hygiene: CI, logging, rate limiting"). Phase 6
deployment is manual/scripted: you run a script by hand when you want to
(re)deploy.

Local dev is unaffected. An env var (`VECTOR_BACKEND=faiss|qdrant`)
selects which backend `search.py` / `multimodal/search.py` use, defaulting
to `faiss`, so all existing tests and the local workflow keep working
exactly as they do today, offline and without touching Qdrant Cloud.

Prerequisite accounts (already provisioned before this phase started, with
real credentials in the gitignored `.env`): a Qdrant Cloud free-tier
cluster, and a Hugging Face account with a write token and two reserved
Space names (`HF_SPACE_BACKEND`, `HF_SPACE_FRONTEND`).

## Architecture & components

### Vector backend abstraction

The only change to existing search orchestration code is at the index
factory functions — everything downstream (embedding, fusion, reranking,
CLI, API routes) is untouched.

- `src/ecomsearch/qdrant_index.py`: a new `QdrantIndex` class exposing the
  same shape as the existing `ProductIndex.search(query_vector, top_k) ->
  list[(item_id, score)]`, backed by a Qdrant Cloud collection instead of
  a local FAISS file. One class is reused for both the text and image
  collections (constructor takes `collection_name` + `dim`), mirroring
  how `ProductIndex` is already reused across Phase 1 (text, 384-dim) and
  Phase 2 (CLIP, 512-dim).
- `search.py`'s `load_dense_index()` and `multimodal/search.py`'s
  `load_index()` each become a small factory: if
  `config.VECTOR_BACKEND == "qdrant"`, return a `QdrantIndex`; otherwise,
  existing FAISS-loading behavior, unchanged.

### Config additions

- `src/ecomsearch/config.py` gains: `VECTOR_BACKEND` (default `"faiss"`),
  `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`,
  `HF_DATASET_REPO` — all read from environment variables.
- `src/ecomsearch/multimodal/config.py` gains: `QDRANT_IMAGE_COLLECTION_NAME`.
- `python-dotenv` is added as a new dependency so local dev picks up
  `.env` automatically. `.env` already exists with real values but
  nothing currently loads it. `.env` is never shipped to production — the
  deployed Spaces set the equivalent variables as HF Space **secrets**
  via the HF UI/API.

### Qdrant migration scripts (one-time, run by hand)

- `scripts/upload_index_to_qdrant.py` — reads the existing local FAISS
  text index + `item_ids.npy` (no re-embedding needed), creates the
  Qdrant collection if it doesn't exist (correct dimension, Cosine
  distance — equivalent to `IndexFlatIP` over normalized vectors), and
  upserts all vectors with `item_id` as the point ID.
- `scripts/upload_multimodal_index_to_qdrant.py` — same, for the CLIP
  image index into a second collection.

### Artifact hosting

- `scripts/upload_artifacts_to_hf.py` — one-time script that pushes the
  catalog CSV, the BM25 pickle, and the CLIP subset metadata + images to
  a new Hugging Face Hub **dataset** repo (`HF_DATASET_REPO`).
- The backend's FastAPI `lifespan` gains a bootstrap step: if these files
  aren't present locally, download them via
  `huggingface_hub.snapshot_download` before `_warm_up_caches()` runs.

### Docker

- `Dockerfile.api` — installs dependencies, copies `src/`, pre-downloads
  the embedding and reranker model weights at build time (so container
  restarts don't refetch them from HF Hub on every cold start), runs
  uvicorn bound to `$PORT`.
- `Dockerfile.ui` — much lighter: just `streamlit` + `requests` + the one
  `streamlit_app.py` file.
- Each Space gets a minimal `README.md` with the HF Spaces YAML metadata
  header (`sdk: docker`, `app_port`, title, etc.).

### Deploy scripts

- `scripts/deploy_backend_space.py` / `scripts/deploy_frontend_space.py`
  — each assembles its Space's minimal file set into a staging directory
  and pushes it via `HfApi().upload_folder(...)` using `HF_TOKEN`. Run by
  hand whenever you want to (re)deploy; no CI trigger.

## Data flow

**One-time setup, run by hand, in order:**
1. Local FAISS indexes already exist (built in earlier phases).
2. `upload_index_to_qdrant.py` + `upload_multimodal_index_to_qdrant.py`
   populate the two Qdrant Cloud collections.
3. `upload_artifacts_to_hf.py` pushes the catalog CSV, BM25 pickle, and
   CLIP images/metadata to the HF dataset repo.
4. `deploy_backend_space.py` + `deploy_frontend_space.py` push the two
   Docker apps to their HF Spaces.

**Container startup (backend, every cold start):**
FastAPI `lifespan` runs → bootstrap step downloads catalog/BM25/CLIP
artifacts from the HF dataset repo into local paths if missing →
`_warm_up_caches()` runs as it did in Phase 5, calling `dense_search`,
`bm25_search`, `hybrid_search`, `image_search` once each → those
functions see `VECTOR_BACKEND=qdrant` and connect to Qdrant Cloud instead
of loading a local FAISS file → BM25 and the reranker work exactly as
before (local pickle file, local model weights baked into the image).

**Request time:** identical to Phase 5. The only production difference is
which object the index factory functions return; query embedding,
fusion, reranking, and response formatting are unchanged code paths.

## Error handling & operational notes

- If Qdrant Cloud is unreachable or the collection doesn't exist,
  `QdrantIndex.search()` lets the underlying `qdrant-client` exception
  propagate — the same "fail loud, don't guess" pattern as
  `load_dense_index()`'s existing `SystemExit` for a missing local FAISS
  file, adapted to the network case.
- If the HF dataset download fails at startup, the app fails fast
  (raises during `lifespan`, container doesn't come up) rather than
  silently serving with missing data.
- HF Spaces' free tier has ephemeral storage, so every restart
  re-downloads the ~600MB of artifacts from the HF dataset repo — expect
  a slow cold start after periods of inactivity. This should be added to
  the README's existing "Known limitations" section, alongside the
  already-documented Qdrant free-tier auto-suspend cold-start caveat.

## Testing strategy

- `QdrantIndex` gets unit tests with a mocked `qdrant-client` (same
  monkeypatching style used throughout this codebase, e.g. `test_cli.py`),
  covering collection creation, upsert, and search/score-mapping logic.
- The `load_dense_index()` / `load_index()` factory functions get a small
  unit test confirming the env var correctly selects FAISS vs. Qdrant.
- One real end-to-end integration test runs against the actual Qdrant
  Cloud cluster, mirroring how `test_integration.py` and
  `test_api_integration.py` already hit real models/indexes rather than
  mocks. Qdrant Cloud's free tier isn't metered per-request, so this is
  safe to run regularly, unlike a paid API.
- Docker builds and the HF Space deploys themselves are verified
  manually (build + run locally, then curl the live Space URLs after
  deploying) — there's no automated way to test a live deployment in this
  test suite, consistent with how Phase 5's Streamlit UI was verified.

## Out of scope for this phase

- Automated CI/CD (GitHub Actions auto-deploy on merge) — Phase 7.
- Rate limiting, structured logging, monitoring — Phase 7.
- Making Qdrant the only backend everywhere (including local dev/tests) —
  not planned; FAISS remains the local/dev/test backend indefinitely.
