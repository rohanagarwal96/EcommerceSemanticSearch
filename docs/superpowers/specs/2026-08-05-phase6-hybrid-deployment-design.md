# Phase 6: Hybrid Deployment (Render + HF Spaces) — Design Spec

## Overview & scope

This spec amends the deployment approach from
`2026-08-03-phase6-deployment-design.md`. Everything in that spec still
holds (Qdrant Cloud migration, HF dataset artifact hosting, dual
FAISS/Qdrant backend, both Dockerfiles) *except* where deployed hosts are
mentioned. Hugging Face now requires a PRO subscription to host
Docker-SDK Spaces on the free `cpu-basic` tier — discovered when the
original Task 13 deploy scripts hit `402 Payment Required` on a real
deploy attempt. The fix is a hybrid split:

1. **Backend** (FastAPI, `Dockerfile.api`) deploys to **Render** as a
   Docker web service, free tier, via a committed `render.yaml`
   Blueprint and a one-time GitHub-repo connection in Render's dashboard.
2. **Frontend** (Streamlit) deploys to **HF Spaces** using the native
   `streamlit` SDK (no Docker), which remains free on HF's `cpu-basic`
   tier.

Qdrant Cloud and the HF dataset repo are unaffected by this change. This
supersedes only the deploy-script/hosting portion of Task 13 in the
original plan (`docs/superpowers/plans/2026-08-03-phase6-deployment.md`).

## Architecture & components

### Backend: Render, Docker, Blueprint-driven

- New root-level `render.yaml`:
  - One `web` service, `runtime: docker`, `dockerfilePath: ./Dockerfile.api`,
    `dockerContext: .`, `plan: free`.
  - `envVars` lists the six runtime secrets with `sync: false` (Render
    prompts for values in its dashboard rather than committing them):
    `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`,
    `QDRANT_IMAGE_COLLECTION_NAME`, `HF_DATASET_REPO`, `HF_TOKEN`.
  - `VECTOR_BACKEND=qdrant` and `HF_HUB_DISABLE_XET=1` are **not** listed
    here — they're already baked into `Dockerfile.api` via `ENV` and
    apply regardless of host.
  - Render auto-injects its own `PORT`; `Dockerfile.api`'s
    `CMD ["sh", "-c", "uvicorn ... --port ${PORT:-7860}"]` already reads
    it, so no change needed there.
- `scripts/deploy_backend_space.py` and
  `tests/test_deploy_backend_space.py` are **removed**. There is no
  script-based push for Render's free tier — deploys are triggered by
  `git push origin main` once the GitHub repo is connected.
- `HF_SPACE_BACKEND` is removed from `config.py` and `.env.example` — no
  longer meaningful.

### Frontend: HF Spaces, native Streamlit SDK

- `scripts/deploy_frontend_space.py` is rewritten. Instead of staging
  `Dockerfile.ui`, it stages:
  - `streamlit_app.py` at the Space repo root (verified self-contained:
    only imports `os`, `requests`, `streamlit` — no local package
    dependency, so it can be copied standalone rather than nested under
    `src/ecomsearch/ui/`).
  - `requirements.txt` (copied from `requirements-ui.txt`).
  - `README.md` with HF Spaces YAML front matter using
    `sdk: streamlit`, `sdk_version` (pinned to the `streamlit` version in
    `requirements-ui.txt`), and `app_file: streamlit_app.py` — replacing
    the old `sdk: docker` / `app_port` front matter.
- `tests/test_deploy_frontend_space.py` keeps its existing guard-clause
  test unchanged (`HF_SPACE_FRONTEND` unset → `SystemExit` mentioning the
  var name) — the staging content changes, but the failure-mode contract
  doesn't.
- `Dockerfile.ui` **stays in the repo**, kept for local `docker run`
  parity testing only; it is no longer read by the deploy script.

## Data flow

**One-time setup, run by hand, in order (revised):**
1. Local FAISS indexes already exist (built in earlier phases).
2. `upload_index_to_qdrant.py` + `upload_multimodal_index_to_qdrant.py`
   populate the two Qdrant Cloud collections. *(unchanged, already done)*
3. `upload_artifacts_to_hf.py` pushes the catalog CSV, BM25 pickle, and
   CLIP images/metadata to the HF dataset repo. *(unchanged, already done)*
4. Commit `render.yaml`; connect the GitHub repo to Render in its
   dashboard (one-time, manual); fill in the six secret values there.
   Render builds `Dockerfile.api` and deploys; every future push to
   `main` auto-redeploys.
5. Run `deploy_frontend_space.py` to push the native-SDK Streamlit Space.
6. On the frontend Space's Settings page, set `API_BASE_URL` to the
   Render service's public URL (`https://<service-name>.onrender.com`).

**Container startup / request time:** identical to the original spec —
this change only affects *where* each container runs, not what it does
once running.

## Error handling & operational notes

- Render's free tier also spins down web services after a period of
  inactivity, same category of cold-start cost as Qdrant Cloud's
  free-tier auto-suspend and HF Spaces' ephemeral storage. All three
  compound: a request after a long idle period may wait on Render
  spinning up, Qdrant Cloud resuming, and the backend re-downloading the
  ~160MB artifact bundle from the HF dataset repo, in sequence. This adds
  to (not replaces) the existing "Known limitations" README notes from
  the original spec.
- Render Blueprint deploy failures (bad `render.yaml`, missing secret
  values) surface in Render's own dashboard/build logs — no in-repo
  handling needed, consistent with how HF Space build failures were
  already out of scope for automated handling.

## Testing strategy

- `tests/test_deploy_frontend_space.py`'s existing guard-clause test is
  kept as-is (behavior unchanged).
- `tests/test_deploy_backend_space.py` is deleted along with
  `scripts/deploy_backend_space.py`.
- `render.yaml` is declarative config, not application code — no unit
  test; its correctness is verified by a real Render deploy succeeding,
  the same "verify manually, not automatable" treatment already given to
  Docker builds and HF Space deploys in the original spec.

## Out of scope for this revision

- Everything already out of scope in the original Phase 6 spec (CI/CD,
  rate limiting, structured logging, monitoring — Phase 7).
- Migrating the backend away from Docker entirely — Render's Docker
  support is exactly why it was chosen; no native-runtime rewrite needed.
- Any change to the Qdrant Cloud or HF dataset repo setup — untouched by
  this revision.
