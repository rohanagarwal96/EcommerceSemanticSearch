# Phase 7: Production Hygiene — Design Spec

## Overview & scope

Phase 7 adds three things that make this project look and behave like a
production-conscious service, without changing its local-only deployment
model from Phase 6: continuous integration (lint + test on every push),
structured logging in the FastAPI backend, and rate limiting on the two
search endpoints.

This phase does not add a CD/deploy step (consistent with Phase 6's
local-only architecture — there is nothing to deploy to), does not add
external log aggregation, and does not add a distributed rate-limit store.
All three additions are scoped to a single-instance local deployment.

## Architecture & components

### CI — GitHub Actions

- New `.github/workflows/ci.yml`, triggered on `push` to `main` and on
  `pull_request`.
- **Lint job**: installs `ruff`, runs `ruff check .` and
  `ruff format --check .`.
- **Test job**: installs `requirements.txt` and the package
  (`pip install -e .`), then runs the full `pytest` suite — every test file,
  including the real-model integration tests (`test_integration.py`,
  `test_api_integration.py`, `test_multimodal_integration.py`) and the
  Qdrant tests (which skip cleanly since CI has no `QDRANT_URL` set, per
  Phase 6's fix to `.env.example`'s default).
- Both jobs use `actions/cache` for pip's download cache and the
  Hugging Face model cache (`~/.cache/huggingface`) to keep repeat runs
  fast after the first (uncached) run, which will be slow due to
  torch/transformers/faiss-cpu installation and model downloads.
- `pyproject.toml` gains a `[tool.ruff]` block with a standard rule
  selection (pyflakes `F`, pycodestyle `E`/`W`, import-sort `I`) matched to
  this codebase's existing style — not a from-scratch aggressive config.

### Structured logging — structlog

- New dependency: `structlog`.
- Configured once in `src/ecomsearch/api/app.py` at import time: JSON
  renderer, output to stdout (the natural fit for a Docker Compose service,
  already captured via `docker compose logs`).
- A request-logging **middleware** (added to the FastAPI `app`) logs one
  structured event per request: `method`, `path`, `status_code`,
  `duration_ms`.
- Each search route (`search_text` in `routes_text.py`, `search_image` in
  `routes_image.py`) logs one structured event per request with
  route-specific fields: `query`, `mode` (text only), `top_k`,
  `result_count`, `duration_ms`.
- `lifespan()`'s existing `_warm_up_caches()` call is followed by a single
  structured startup-complete log event.
- Uncaught exceptions are logged with a stack trace via a FastAPI
  exception handler before the framework's default 500 response is
  returned.
- CLI scripts (`ecomsearch search`, `build_index.py`, etc.) are **not**
  touched — they keep their existing `rich`-based console output.
  structlog is scoped entirely to the FastAPI backend.

### Rate limiting — slowapi

- New dependency: `slowapi` (built on the `limits` library).
- A shared `Limiter` instance (keyed by client IP, in-memory storage — no
  Redis) is created once and attached to the FastAPI `app`.
- Applied via decorator to `/search/text` and `/search/image`:
  **30 requests/minute** per IP.
- `/health` and `/images/{item_id}` are **not** rate-limited (health
  checks and image thumbnails shouldn't be throttled).
- Exceeding the limit returns HTTP 429 with slowapi's default JSON error
  body (`{"error": "Rate limit exceeded: 30 per 1 minute"}`-style message).

## Data flow

**Request time (search endpoints):** request arrives → rate-limit check
(slowapi middleware) → if within limit, request-logging middleware records
start time → route handler runs (existing search logic, unchanged) →
route handler logs its structured search event → request-logging
middleware logs the completed-request event with status code and duration
→ response returned. If the rate limit is exceeded, slowapi returns 429
before the route handler runs at all.

**CI (on push/PR):** GitHub Actions checks out the repo → lint job and
test job run in parallel → both must pass for the workflow to report
green. No deploy step follows.

## Error handling & operational notes

- Rate-limit state is in-memory and per-process — restarting the backend
  container resets all counters. Acceptable for a local single-instance
  deployment; would need a shared store (Redis) for a multi-instance
  production deployment, which is out of scope.
- Structured logs are not persisted beyond container stdout/whatever
  Docker's own log driver retains — no log rotation or shipping is added.
- CI failures (lint or test) block the workflow from going green but do
  not block local `git push` — this project has no branch protection
  configured (consistent with its established direct-to-main workflow).

## Testing strategy

- Logging: unit tests using `structlog.testing.capture_logs()` to assert
  the expected structured fields (`method`, `path`, `status_code`,
  `duration_ms` for the middleware; `query`, `mode`, `top_k`,
  `result_count`, `duration_ms` for each search route) appear in the
  captured log entries for a given request.
- Rate limiting: `TestClient`-based tests that fire requests at a search
  endpoint past the configured limit and assert the final response is a
  429, and that a request under the limit still succeeds normally.
- CI: verified for real — after the workflow file is committed and pushed,
  confirm the actual GitHub Actions run succeeds (same standard as Phase
  6's real Docker Compose verification, not just reading the YAML).
- Existing test suite must still pass unchanged after middleware/rate
  limiting are added — search routes' response bodies and status codes
  for normal (non-rate-limited) requests are unaffected by this phase.

## Out of scope for this phase

- CD / deploy automation — no deploy target exists post-Phase-6.
- External log aggregation or shipping (Datadog, ELK, etc.).
- Distributed/shared rate-limit storage (Redis-backed slowapi).
- Rate limiting or structured logging for the CLI tools — scoped to the
  FastAPI backend only.
- Changing CI to skip or subset the real-model integration tests — every
  test runs on every CI push, per explicit choice.
