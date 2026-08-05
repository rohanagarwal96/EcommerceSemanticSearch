# Phase 6 Hybrid Deployment (Render + HF Spaces) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocked all-Hugging-Face deployment path (Docker Spaces now require HF PRO) with a hybrid split: FastAPI backend on Render (Docker, free tier, `render.yaml` Blueprint), Streamlit frontend on HF Spaces (native `streamlit` SDK, no Docker, still free).

**Architecture:** Remove the backend's HF-Space deploy script/test and the now-unused `HF_SPACE_BACKEND` config var. Add a `render.yaml` Blueprint describing the backend as a Docker web service. Rewrite `scripts/deploy_frontend_space.py` to stage a native-SDK Streamlit Space instead of a Dockerfile-based one. Some steps (connecting Render's dashboard to GitHub, filling in secret values, setting the frontend's `API_BASE_URL`) are one-time manual actions in a web UI that cannot be scripted or performed by an agent — these are called out explicitly as **HUMAN ACTION REQUIRED** steps.

**Tech Stack:** Render Blueprints (`render.yaml`), Hugging Face Hub native Streamlit Spaces (`huggingface_hub.HfApi`), pytest, PyYAML (already installed transitively, used only for a sanity check).

---

### Task 1: Remove the Hugging Face backend-Space deploy path

**Files:**
- Delete: `scripts/deploy_backend_space.py`
- Delete: `tests/test_deploy_backend_space.py`
- Modify: `src/ecomsearch/config.py:46`
- Modify: `.env.example:4`

- [ ] **Step 1: Delete the backend deploy script and its test**

```bash
git rm --cached scripts/deploy_backend_space.py tests/test_deploy_backend_space.py 2>/dev/null; rm -f scripts/deploy_backend_space.py tests/test_deploy_backend_space.py
```

(These two files are currently untracked leftovers from a deploy attempt that failed with HTTP 402 before ever being committed, so `git rm --cached` will no-op harmlessly if git doesn't know about them — the `rm -f` is what actually removes them.)

- [ ] **Step 2: Remove `HF_SPACE_BACKEND` from config.py**

In `src/ecomsearch/config.py`, delete this line (currently line 46):

```python
HF_SPACE_BACKEND = os.environ.get("HF_SPACE_BACKEND")
```

So the surrounding block reads:

```python
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
HF_SPACE_FRONTEND = os.environ.get("HF_SPACE_FRONTEND")
```

- [ ] **Step 3: Remove `HF_SPACE_BACKEND` from `.env.example`**

In `.env.example`, delete this line (currently line 4):

```
HF_SPACE_BACKEND=your-hf-username/ecommerce-search-api
```

So the HF section reads:

```
# Hugging Face (account + write token, used for model downloads, artifact
# hosting, and Space deployment)
HF_TOKEN=your_huggingface_write_token_here
HF_SPACE_FRONTEND=your-hf-username/ecommerce-search-ui
HF_DATASET_REPO=your-hf-username/ecommerce-search-artifacts
```

- [ ] **Step 4: Confirm nothing else references the removed name**

Run: `grep -rn "HF_SPACE_BACKEND" src/ tests/ scripts/ .env.example`
Expected: no output (empty). If anything besides `docs/` shows up, stop and fix it before continuing.

- [ ] **Step 5: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (no test referenced `HF_SPACE_BACKEND` except the now-deleted `test_deploy_backend_space.py`).

- [ ] **Step 6: Commit**

```bash
git add -A -- scripts/deploy_backend_space.py tests/test_deploy_backend_space.py src/ecomsearch/config.py .env.example
git commit -m "chore: remove HF-Space backend deploy path in favor of Render"
git push origin main
```

---

### Task 2: Add the Render Blueprint for the backend

**Files:**
- Create: `render.yaml`

- [ ] **Step 1: Write `render.yaml`**

```yaml
services:
  - type: web
    name: ecomsearch-search-api
    runtime: docker
    dockerfilePath: ./Dockerfile.api
    dockerContext: .
    plan: free
    envVars:
      - key: QDRANT_URL
        sync: false
      - key: QDRANT_API_KEY
        sync: false
      - key: QDRANT_COLLECTION_NAME
        sync: false
      - key: QDRANT_IMAGE_COLLECTION_NAME
        sync: false
      - key: HF_DATASET_REPO
        sync: false
      - key: HF_TOKEN
        sync: false
```

`VECTOR_BACKEND=qdrant` and `HF_HUB_DISABLE_XET=1` are intentionally not listed — they're already baked into `Dockerfile.api` via `ENV` and apply on any host. `sync: false` means Render will prompt you to fill in each value by hand in its dashboard rather than expecting (or accepting) a value committed to this file.

- [ ] **Step 2: Sanity-check the YAML parses and has the expected shape**

Run:
```bash
venv/Scripts/python.exe -c "
import yaml
with open('render.yaml') as f:
    doc = yaml.safe_load(f)
assert doc['services'][0]['dockerfilePath'] == './Dockerfile.api'
assert doc['services'][0]['plan'] == 'free'
assert {e['key'] for e in doc['services'][0]['envVars']} == {
    'QDRANT_URL', 'QDRANT_API_KEY', 'QDRANT_COLLECTION_NAME',
    'QDRANT_IMAGE_COLLECTION_NAME', 'HF_DATASET_REPO', 'HF_TOKEN',
}
print('render.yaml OK')
"
```
Expected: `render.yaml OK` with no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add render.yaml
git commit -m "feat: add Render Blueprint for backend Docker deployment"
git push origin main
```

---

### Task 3: Rewrite the frontend deploy script for HF Spaces' native Streamlit SDK

**Files:**
- Modify: `scripts/deploy_frontend_space.py` (full rewrite)
- Test: `tests/test_deploy_frontend_space.py` (already exists, unchanged — verifying it still passes)

- [ ] **Step 1: Confirm the existing guard-clause test still describes the desired behavior**

Read `tests/test_deploy_frontend_space.py` — it should already contain exactly this (no edit needed, just confirm):

```python
import pytest

import deploy_frontend_space


def test_main_exits_with_clear_message_when_space_not_set(monkeypatch):
    monkeypatch.setattr(deploy_frontend_space, "HF_SPACE_FRONTEND", None)

    with pytest.raises(SystemExit) as excinfo:
        deploy_frontend_space.main()

    assert "HF_SPACE_FRONTEND" in str(excinfo.value)
```

- [ ] **Step 2: Run the test against the current (Docker-based) script to confirm it currently passes**

Run: `venv/Scripts/python.exe -m pytest tests/test_deploy_frontend_space.py -v`
Expected: PASS (1 test) — this confirms the guard clause behavior we must preserve through the rewrite.

- [ ] **Step 3: Rewrite `scripts/deploy_frontend_space.py`**

```python
"""One-time (or repeat-as-needed) script: push the Streamlit frontend to its
Hugging Face Space using the native Streamlit SDK (no Docker required).

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
sdk: streamlit
sdk_version: "1.60.0"
app_file: streamlit_app.py
---

Streamlit frontend for the E-Commerce Semantic Search project.
"""


def main() -> None:
    if not HF_SPACE_FRONTEND:
        raise SystemExit("HF_SPACE_FRONTEND is not set. Add it to your .env.")

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        shutil.copy2(
            REPO_ROOT / "src" / "ecomsearch" / "ui" / "streamlit_app.py",
            staging_dir / "streamlit_app.py",
        )
        shutil.copy2(REPO_ROOT / "requirements-ui.txt", staging_dir / "requirements.txt")
        (staging_dir / "README.md").write_text(SPACE_README, encoding="utf-8")

        api = HfApi(token=HF_TOKEN)
        print(f"Creating (or reusing) Space '{HF_SPACE_FRONTEND}'...")
        api.create_repo(
            repo_id=HF_SPACE_FRONTEND, repo_type="space", space_sdk="streamlit", exist_ok=True
        )
        print(f"Uploading frontend to '{HF_SPACE_FRONTEND}'...")
        api.upload_folder(
            repo_id=HF_SPACE_FRONTEND,
            folder_path=str(staging_dir),
            repo_type="space",
            commit_message="Deploy frontend (native Streamlit SDK)",
        )

    print(f"Done. https://huggingface.co/spaces/{HF_SPACE_FRONTEND}")


if __name__ == "__main__":
    main()
```

Note `space_sdk="streamlit"` (was `"docker"`), no `Dockerfile.ui` copy, `requirements-ui.txt` is copied and renamed to `requirements.txt` (the name HF's native Streamlit SDK looks for), and the staged README's front matter uses `sdk`/`sdk_version`/`app_file` instead of `sdk`/`app_port` for Docker.

- [ ] **Step 4: Run the guard-clause test again to confirm it still passes against the rewritten script**

Run: `venv/Scripts/python.exe -m pytest tests/test_deploy_frontend_space.py -v`
Expected: PASS (1 test) — same test, now exercising the rewritten `main()`.

- [ ] **Step 5: Run the full test suite**

Run: `venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/deploy_frontend_space.py
git commit -m "feat: deploy frontend to HF Spaces via native Streamlit SDK instead of Docker"
git push origin main
```

---

### Task 4: Deploy the backend to Render — real deploy (HUMAN ACTION REQUIRED for dashboard steps)

**Files:** none (infrastructure setup, no repo changes)

This task cannot be completed by an agent alone: Render's Blueprint flow requires a one-time OAuth connection between your Render account and your GitHub account, done in a browser. An implementing agent should perform Step 1 (confirm the Blueprint file is pushed), then do Step 2 (stop and hand off the numbered manual sub-steps to the human, verbatim), and only attempt Step 3 once told the backend is live. Do not attempt to guess at or fabricate a Render API key/OAuth flow, and do not treat this task as blocked/failed — stopping to hand off is the correct completion state for this task's automatable portion.

- [ ] **Step 1: Confirm `render.yaml` is committed and pushed**

Run: `git log --oneline -1 -- render.yaml` and `git status`
Expected: shows the Task 2 commit; working tree clean; `git log origin/main -1 --oneline` matches local HEAD.

- [ ] **Step 2: Stop and hand off — list the remaining manual steps for the human**

Report exactly these remaining steps back to the user (do not attempt them):

1. Go to <https://dashboard.render.com>, sign in (create a free account if needed).
2. Click **New > Blueprint**, connect your GitHub account if this is the first time, and select the `rohanagarwal96/EcommerceSemanticSearch` repository, branch `main`.
3. Render detects `render.yaml` and shows the `ecomsearch-search-api` service for review. Click **Apply**.
4. During or immediately after creation, Render will prompt for the 6 env vars marked `sync: false` (`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`, `QDRANT_IMAGE_COLLECTION_NAME`, `HF_DATASET_REPO`, `HF_TOKEN`) — fill each in using the real values from your local `.env` (never share these values in chat or commit them).
5. Wait for the first build to finish (Docker build + model download + artifact download on first boot takes several minutes — watch the Logs tab).
6. Once live, copy the service's public URL from the Render dashboard (format: `https://ecomsearch-search-api.onrender.com` or similar, Render may adjust the subdomain if taken) — this is needed for Task 5.

- [ ] **Step 3: Once the human confirms the backend is live, verify it**

Ask the human for the exact Render service URL, then run (substituting the real URL):
```bash
curl -s https://<render-service-url>/health
```
Expected: `{"status":"ok"}`

---

### Task 5: Deploy the frontend to HF Spaces and point it at the backend (HUMAN ACTION REQUIRED for one step)

**Files:** none (deploy execution + one manual settings step)

**Depends on:** Task 3 (script must exist) for Step 1. Depends on Task 4 (need the Render URL) for Step 3.

- [ ] **Step 1: Run the real frontend deploy**

Run: `venv/Scripts/python.exe scripts/deploy_frontend_space.py`
Expected: prints `Creating (or reusing) Space '<HF_SPACE_FRONTEND>'...`, `Uploading frontend to '<HF_SPACE_FRONTEND>'...`, then `Done. https://huggingface.co/spaces/<HF_SPACE_FRONTEND>` with no errors.

- [ ] **Step 2: Confirm the Space is building**

Open (or ask the human to open) `https://huggingface.co/spaces/<HF_SPACE_FRONTEND>` and check its build status shows "Building" or "Running" (not "Build error"). If it errors, check the Space's Logs tab for a missing-dependency or syntax problem in `streamlit_app.py`/`requirements-ui.txt` before proceeding.

- [ ] **Step 3: STOP and hand off — one manual setting is required**

Report back to the human: once the backend's Render URL is known (from Task 4), they must go to `https://huggingface.co/spaces/<HF_SPACE_FRONTEND>/settings` and set a repository secret/variable named `API_BASE_URL` to that Render URL (e.g. `https://ecomsearch-search-api.onrender.com`). This cannot be done by an agent since it's a browser-only settings page action. Setting it triggers the Space to restart and pick up the new value (`streamlit_app.py` reads `API_BASE_URL` from the environment at import time — confirmed at `src/ecomsearch/ui/streamlit_app.py:7`).

---

### Task 6: Verify both live deployments end-to-end

**Files:** none

**Depends on:** Task 4 Step 3 (backend live) and Task 5 (frontend live + `API_BASE_URL` set).

- [ ] **Step 1: Verify the backend directly**

Run (substituting the real Render URL):
```bash
curl -s "https://<render-service-url>/health"
curl -s "https://<render-service-url>/search/text?q=organic+almond+milk&top_k=2"
curl -s "https://<render-service-url>/search/image?q=shoes&top_k=2"
```
Expected: `/health` returns `{"status":"ok"}`; the two search endpoints return real product results (almond milk products for the text query, shoe products for the image query), matching the same results seen in earlier local Docker verification.

- [ ] **Step 2: Verify the frontend end-to-end through the browser**

Ask the human to open `https://huggingface.co/spaces/<HF_SPACE_FRONTEND>` and confirm:
- The Text Search tab returns real results for a query like "organic almond milk".
- The Image Search tab returns real product images for a query like "shoes".

If either tab errors with a connection failure, the most likely cause is `API_BASE_URL` not yet set or the Space not yet finished restarting after Task 5's settings change — wait and retry before treating it as a real bug.

- [ ] **Step 3: Report final status**

Summarize: both live URLs, confirmation both search modes work end-to-end, and that Task 13 (in its hybrid form) is complete. No commit needed for this task (verification only).

---

## Notes for whoever executes Task 14 next

Task 14 in the original `docs/superpowers/plans/2026-08-03-phase6-deployment.md` (README update) was written assuming an all-Hugging-Face deployment. When executing it, use the *hybrid* architecture from this plan instead: one-time setup order becomes `upload_index_to_qdrant.py` → `upload_multimodal_index_to_qdrant.py` → `upload_artifacts_to_hf.py` → connect Render Blueprint (manual) → `deploy_frontend_space.py` → set `API_BASE_URL` (manual); the backend's live URL is the Render URL, not an HF Space URL; and the "Known limitations" addition should mention Render's free-tier spin-down alongside the existing Qdrant and HF Spaces cold-start notes, per `docs/superpowers/specs/2026-08-05-phase6-hybrid-deployment-design.md`.
