# Phase 6: Local Docker Compose Deployment — Design Spec

## Overview & scope

This spec supersedes both `2026-08-03-phase6-deployment-design.md` (all-Hugging-Face
cloud deployment) and `2026-08-05-phase6-hybrid-deployment-design.md` (Render +
HF Spaces hybrid). Both were blocked by real infrastructure limits discovered
during actual deploy attempts: HF now requires a PRO subscription for
Docker-SDK Spaces, and Render's tiers with enough RAM for this backend's
three-model footprint (bge-small embedder, MiniLM cross-encoder reranker,
CLIP) start at $85/month. Given this is a portfolio project, the decision is
to drop public cloud deployment entirely and demonstrate the system as a
**local Docker Compose stack**: Qdrant, the FastAPI backend, and the
Streamlit frontend all running as containers on one machine, brought up with
a single `docker compose up`.

This is a deliberate scope reduction, not an incomplete deployment. The
project still demonstrates containerization, a real vector database
(self-hosted Qdrant, not just local FAISS files), and a working multi-service
architecture — it just doesn't expose a public URL.

Local `venv`-based dev (no Docker at all) is unaffected: `VECTOR_BACKEND`
still defaults to `"faiss"` for running tests and scripts directly, exactly
as before. Docker Compose is an additional way to run the whole stack, not a
replacement for existing local dev.

## What gets removed

Everything tied to the abandoned cloud paths:

- `render.yaml` (Render Blueprint — no longer used).
- `scripts/deploy_frontend_space.py` and `tests/test_deploy_frontend_space.py`
  (HF Space frontend deploy — no longer used).
- `scripts/upload_artifacts_to_hf.py` and `tests/test_upload_artifacts_to_hf.py`
  (HF dataset repo artifact hosting — no longer used; local Docker Compose
  bind-mounts local `data/`/`artifacts/` directly instead).
- `src/ecomsearch/api/app.py`'s `_ensure_artifacts_present()` function, its
  call inside `lifespan()`, the `snapshot_download` import, and the three
  tests in `tests/test_api_app.py` that cover it
  (`test_lifespan_downloads_artifacts_when_missing`,
  `test_lifespan_skips_download_when_artifacts_already_present`,
  `test_ensure_artifacts_present_exits_when_missing_and_no_dataset_repo`) —
  this bootstrap becomes permanently unreachable once local bind-mounts
  guarantee the artifacts are already present.
- `HF_TOKEN`, `HF_DATASET_REPO`, `HF_SPACE_FRONTEND` from
  `src/ecomsearch/config.py` and `.env.example`.

No `.github/workflows` directory exists yet in this repo, so there is no CI
deploy step to strip out — CI (lint/test only, no deploy) is Phase 7 work,
untouched by this spec.

The already-created Qdrant Cloud cluster and HF dataset repo are left alone
(free tier, no ongoing cost, no action needed).

## What's kept, unchanged

- `Dockerfile.api` / `Dockerfile.ui` — both already build and run correctly
  as non-root UID 1000 containers; that constraint came from HF Spaces but
  doesn't hurt local use, so neither Dockerfile changes.
- `.dockerignore` — unchanged.
- `src/ecomsearch/qdrant_index.py` (`QdrantIndex` class) — unchanged; it's
  already agnostic to whether it's talking to a cloud cluster or a local
  container, since it just takes a `QDRANT_URL`.
- `search.py` / `multimodal/search.py`'s `VECTOR_BACKEND`-based factory
  functions — unchanged.
- `scripts/upload_index_to_qdrant.py` / `upload_multimodal_index_to_qdrant.py`
  — unchanged code; only the `QDRANT_URL` they read at runtime changes (from
  a cloud cluster URL to `http://localhost:6333`).
- `tests/test_qdrant_index_integration.py` / `tests/test_search_qdrant_e2e.py`
  — unchanged; both are already gated by
  `pytestmark = pytest.mark.skipif(not QDRANT_URL, ...)`, so they run against
  whatever `QDRANT_URL` points to (local or cloud) and skip cleanly when
  unset, e.g. in CI.

## Architecture & components

### `docker-compose.yml` (new, repo root)

Three services:

- **`qdrant`**: official `qdrant/qdrant` image. Named volume
  (`qdrant_storage:/qdrant/storage`) for persistence across restarts. Port
  `6333` (HTTP/REST) published to the host, so one-time setup scripts
  (`upload_index_to_qdrant.py` etc.) can run against it from a local `venv`
  outside Docker.
- **`backend`**: builds from `Dockerfile.api` (context: repo root).
  Environment: `VECTOR_BACKEND=qdrant`, `QDRANT_URL=http://qdrant:6333`
  (Compose's internal DNS resolves the service name), `QDRANT_COLLECTION_NAME`,
  `QDRANT_IMAGE_COLLECTION_NAME`. No `QDRANT_API_KEY` needed — a local Qdrant
  container has no auth by default, and `QdrantClient(url=..., api_key=None)`
  already handles that. Bind-mounts the repo's local `data/` and `artifacts/`
  directories **read-only** into the container at the same paths
  `REPO_ROOT`-relative config already expects, so the backend finds the
  catalog CSV, BM25 pickle, and CLIP subset files without fetching anything
  remotely. Depends on `qdrant`. Published as `localhost:8000` (matching this
  project's established local-testing port convention from Phase 6's earlier
  Docker verification).
- **`frontend`**: builds from `Dockerfile.ui` (context: repo root).
  Environment: `API_BASE_URL=http://backend:7860` (Compose's internal DNS;
  matches `Dockerfile.api`'s internal port). Published as `localhost:8501`
  (Streamlit's conventional default, per the explicit ask).

### One-time local Qdrant population

A freshly started `qdrant` container has no collections. Setup order
(documented in the README, run by hand once per fresh environment — same
"manual/scripted, not automatic" convention already established for the
cloud version):

1. Local FAISS indexes and other artifacts already exist (built via the
   existing Phase 1-5 pipeline scripts — unaffected by this spec).
2. `docker compose up -d qdrant` — starts just the Qdrant container.
3. Run `upload_index_to_qdrant.py` and `upload_multimodal_index_to_qdrant.py`
   locally via `venv` (with `QDRANT_URL=http://localhost:6333`) — populates
   the two collections from the local FAISS files, unchanged script logic.
4. `docker compose up` — brings up (or leaves running) `qdrant`, and starts
   `backend` + `frontend`, now against a populated local Qdrant.

### Config changes

- `.env.example`: remove `HF_TOKEN`, `HF_SPACE_FRONTEND`, `HF_DATASET_REPO`.
  `QDRANT_URL` default becomes `http://localhost:6333` (for running scripts
  directly via `venv`, outside Docker) with a comment noting Compose
  overrides this internally to `http://qdrant:6333` for the backend
  container. `QDRANT_API_KEY` documented as optional/empty for local use.
  Kaggle section untouched.
- `src/ecomsearch/config.py`: remove `HF_TOKEN`, `HF_DATASET_REPO`,
  `HF_SPACE_FRONTEND`. `VECTOR_BACKEND`, `QDRANT_URL`, `QDRANT_API_KEY`,
  `QDRANT_COLLECTION_NAME` stay.

## Data flow

**One-time setup (per fresh environment), run by hand:**
Build local artifacts (existing pipeline, unaffected) → `docker compose up -d
qdrant` → run the two `upload_*_to_qdrant.py` scripts against
`localhost:6333` → `docker compose up`.

**Container startup (every `docker compose up`):** `backend`'s FastAPI
`lifespan` runs `_warm_up_caches()` (unchanged from Phase 5/6) — calls
`dense_search`, `bm25_search`, `hybrid_search`, `image_search` once each.
Those functions see `VECTOR_BACKEND=qdrant` and connect to
`http://qdrant:6333` instead of loading a local FAISS file. BM25 and the
reranker work exactly as before (local pickle file, local model weights
baked into the image at build time, now read from the bind-mounted
`artifacts/` directory instead of a downloaded copy).

**Request time:** identical to every prior phase — query embedding, fusion,
reranking, response formatting are unchanged code paths. The frontend calls
`API_BASE_URL` (now `http://backend:7860` inside the Compose network, or
`http://localhost:8000` if you're curling the backend directly from the
host).

## Error handling & operational notes

- If the local `qdrant` container isn't up yet or its collections aren't
  populated when `backend` starts, the existing "fail loud" pattern in
  `QdrantIndex.search()` applies — the underlying `qdrant-client` exception
  propagates rather than silently returning empty results. `docker-compose`'s
  `depends_on` ensures `qdrant`'s container process has started before
  `backend` starts, but doesn't guarantee its collections are populated —
  that's the one-time setup step's job, documented clearly in the README so
  it isn't confused with a bug.
- No ephemeral-storage cold-start concern anymore (that was specific to HF
  Spaces) — local bind-mounts and the named Qdrant volume persist across
  restarts on the host machine.

## Testing strategy

- No new automated tests are needed for `docker-compose.yml` itself
  (declarative infra config, not application code) — verified by a real
  `docker compose up` succeeding, consistent with how Docker builds and
  cloud deploys were already verified manually rather than via the pytest
  suite in the earlier specs.
- Removing `_ensure_artifacts_present()` requires deleting its three
  existing tests in `tests/test_api_app.py` (they test code that no longer
  exists) — no replacement tests needed, since there's no new behavior to
  cover; the bind-mount either has the files or the app fails to start via
  the existing `CATALOG_PATH`-driven code paths elsewhere in the app.
- `tests/test_qdrant_index_integration.py` / `tests/test_search_qdrant_e2e.py`
  remain valid as-is and will exercise whatever `QDRANT_URL` is set to
  (local Qdrant, if running, or skip cleanly if not).
- Full existing test suite must still pass after removals (confirms nothing
  else depended on the removed code).

## Out of scope for this revision

- Any public/cloud deployment (explicitly abandoned).
- CI/CD automation — Phase 7, and no existing CI workflow exists yet to
  modify.
- Recording the demo GIF/video — this requires a human to actually run the
  local stack and capture their screen; not something that can be automated
  in this session. Flagged as a manual follow-up step; the README section
  referencing it will be added once the file exists.
- Any change to the already-created Qdrant Cloud cluster or HF dataset repo
  — left alone per explicit instruction, not deleted or modified.
