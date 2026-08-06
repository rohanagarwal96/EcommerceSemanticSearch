# Phase 8: Documentation Finalization — Design Spec

## Overview & scope

Phase 8 is documentation-only — no application code, CI, or Dockerfile
changes. Its goal is to make the repository read as a polished, accessible
portfolio project: remove internal AI-development artifacts, and rewrite
the README so a first-time reader with no prior context on this project
(and no assumed ML background) can understand what it does, why it was
built the way it was, and how to run it themselves.

This spec supersedes the Phase 6 README's demo-GIF placeholder (dropped
entirely, not fulfilled) and expands substantially on every prior phase's
incremental README updates.

## What gets removed

- `docs/superpowers/` (20 files: 10 specs, 10 plans) — internal
  AI-assisted-development planning artifacts, full of tool-specific
  language (skill names, subagent references) that conflicts with this
  project's established no-AI-attribution convention. Deleted entirely;
  still recoverable via local git history if ever wanted, but not part of
  the public repo going forward.
- The README's `<!-- Demo: ... -->` HTML comment placeholder (in the
  "Running the App" section) — dropped, no demo media planned.

## Architecture & components

### README restructure

The README grows from its current ~10 flat sections into a more
navigable, narrative document. New table of contents at the top links to
every section below. New/changed sections, in reading order:

1. **Title + pitch** (existing, unchanged).
2. **Table of Contents** (new) — linked list of every `##` section.
3. **Status** (existing checklist — stays as a quick-glance summary,
   unchanged in position/format).
4. **What this project does** (new) — plain-language framing for a reader
   with no ML background: why keyword search alone misses relevant
   results (e.g. "cozy winter coat" vs. "warm jacket"), what semantic
   search and multimodal (text-to-image) search add on top of that, and
   who this project is for.
5. **Architecture** (new) — a Mermaid diagram (renders natively on
   GitHub, no external tooling needed) showing the request flow: Frontend
   → Backend → {dense / BM25 / hybrid retrieval → Qdrant or FAISS} →
   reranker → response, plus the separate multimodal (CLIP) path. A short
   paragraph of prose accompanies the diagram, explaining each box.
6. **How this was built** (new) — phase-by-phase prose narrative (not a
   repeat of the Status checklist): for each of Phases 1-7, what was
   built, why that specific approach was chosen over alternatives, and
   what the outcome was. Written from the real history of this project
   (verified against `docs/eval_results.md`, `docs/latency_results.md`,
   and this repo's actual commit/phase history — not invented).
7. **Data** (existing, unchanged).
8. **Stack** (existing table, unchanged).
9. **Evaluation** (existing, unchanged).
10. **Latency** (existing, unchanged).
11. **Setup** (rewritten) — an explicit, linear, beginner-friendly
    walkthrough assuming zero prior context: clone → create venv →
    install dependencies → build the search indexes (with realistic time
    expectations, as today) → run a first search. More step-by-step than
    the current version, but describing the same underlying commands —
    no new scripts or tooling.
12. **Running the App** (existing — both `venv` and Docker Compose
    options — unchanged in substance, demo-placeholder comment removed).
13. **Production hygiene** (existing, unchanged).
14. **Retrospective / lessons learned** (new) — a short, honest section
    on real engineering-judgment moments from the build, e.g.:
    `rank_bm25`'s brute-force scoring cost discovered during latency
    work (`docs/latency_results.md`), the three-stage cloud-deployment
    path (Qdrant Cloud + HF Spaces → Render/HF hybrid → local-only Docker
    Compose) driven by real infrastructure constraints discovered at each
    step (HF Docker Spaces requiring a paid plan, Render's RAM ceiling for
    a three-model backend), and the CI artifact-caching gap found only
    once a real end-to-end GitHub Actions run was attempted. Written as
    genuine reflection on tradeoffs, not a restated feature list.
15. **Known limitations** (existing, unchanged).
16. **License** (existing, unchanged).

## Data flow

Not applicable — this phase does not change request handling or system
behavior. "Data flow" for this phase is the review flow: read the current
README and every referenced doc (`docs/eval_results.md`,
`docs/latency_results.md`, `data/DATA_DICTIONARY.md`) → verify every claim
being added or kept is still accurate against current code/config →
write the expanded README → verify the new architecture diagram and setup
steps against the actual current code/CI/compose files before treating
the phase as done.

## Error handling & operational notes

Not applicable (no runtime behavior changes). The main risk in this phase
is inaccuracy — an expanded, more narrative README has more surface area
for a claim to drift from what the code actually does. Every new section
gets checked against a concrete source of truth (the diagram against the
actual route/search-function structure, the setup steps against the
actual scripts, the retrospective against the actual eval/latency docs
and this project's real history) rather than being written from memory.

## Testing strategy

- Run the full test suite after `docs/superpowers/` removal to confirm
  nothing in `tests/`, `scripts/`, or `src/` referenced anything under
  that path (expected: nothing does — it was always planning
  documentation, never imported or executed code).
- No new automated tests — this phase produces no new code paths.
- Manual verification: read the finished README top-to-bottom as if
  seeing the project for the first time, confirming the narrative
  actually flows and every command shown is copy-pasteable and correct
  against the current repo state.

## Out of scope for this phase

- Any application code, CI, or Dockerfile change.
- Recording a demo GIF/video (explicitly dropped, not deferred).
- Rewriting `docs/eval_results.md` or `docs/latency_results.md`
  themselves — only the top-level README changes; those two docs are
  linked from it, unchanged.
- Any change to `data/DATA_DICTIONARY.md`.
