# Phase 4a: Evaluation — Design

## Context

This is the first of two sub-phases splitting the original brief's Phase 4
(evaluation and latency engineering). Phases 1-3 (text embedding baseline,
multimodal/CLIP demo, hybrid BM25+dense+RRF+cross-encoder-rerank retrieval
with a `--mode` flag) are complete and merged.

Phase 4a builds a hand-labeled evaluation set and computes Recall@10,
NDCG@10, and MRR for all four retrieval modes (`dense`, `bm25`, `hybrid`,
`hybrid-rerank`), producing a comparison table. Phase 4b (a separate,
later spec) covers latency benchmarking and optimization, including
fixing the known reload-per-call issue in `search.py` (deferred
deliberately — see "Scope boundary" below).

## Scope boundary: no caching/performance work in this phase

`src/ecomsearch/search.py`'s `dense_search`/`bm25_search`/`hybrid_search`
currently reload the FAISS index, BM25 index, and neural models (bge
embedder, cross-encoder reranker) from scratch on every call. Running the
eval harness (30-50 queries × 4 modes = 120-200 calls) against this will
be slow — roughly 10-20 minutes total, run as a background batch job like
Phase 1/2's embedding jobs, not optimized here. This keeps Phase 4a
focused purely on correctness/evaluation; Phase 4b's caching fix will
also speed up future eval re-runs as a side benefit.

## Relevance judgment methodology

- **Binary relevance** (relevant / not relevant), not graded. Simpler to
  label consistently across 30-50 queries, and all three requested
  metrics support binary relevance (NDCG@10 treats "relevant" as a single
  gain value rather than distinguishing degrees of relevance).
- **Pooling**: for each query, the candidate set to judge is the
  deduplicated union of the top-10 results from all 4 modes (typically
  ~15-30 unique candidates after dedup) — not an exhaustive hand-search
  of the 55,516-row catalog. This is the standard IR evaluation technique
  (used by TREC and similar benchmarks) for making hand-labeling
  tractable while avoiding bias toward any single retrieval mode. It
  does mean "relevant items for this query" is scoped to the pooled
  candidates, not a true exhaustive ground truth over the whole catalog
  — documented explicitly as a caveat in the results doc, not hidden.
- **Query domain**: queries must reflect the catalog's actual composition
  (grocery, personal care/drugstore, dietary/attribute-based like "gluten
  free", brand-based, vague/conceptual within-domain) rather than generic
  e-commerce categories the catalog doesn't cover (electronics, apparel,
  etc. — confirmed sparse/absent during Phase 1's manual CLI testing).
  Out-of-domain queries would produce uniformly poor scores for every
  mode, which measures the catalog's coverage, not the retrieval
  systems' relative quality.
- **Review checkpoint**: I (Claude) draft the query set and initial
  relevance judgments, then the user reviews and corrects
  `eval/eval_queries.json` before it's treated as ground truth — a hard
  requirement from the original project brief, not optional.

## Architecture

```
eval/
  eval_queries.json   # committed (curated asset, not a generated artifact):
                       # [{"query": "...", "relevant_item_ids": [...]}, ...]

src/ecomsearch/
  eval.py             # pure metric functions, no I/O:
                       # recall_at_k(retrieved_ids, relevant_ids, k)
                       # ndcg_at_k(retrieved_ids, relevant_ids, k)
                       # mrr(retrieved_ids, relevant_ids)

scripts/
  run_eval.py         # loads eval_queries.json, runs all 4 modes per query,
                       # computes metrics, writes docs/eval_results.md

docs/
  eval_results.md     # full comparison table + methodology notes
```

`eval.py`'s functions are pure (plain lists/sets of `item_id` in, a float
out) — no dependency on `search.py`, FAISS, or any model, matching the
established pattern from Phase 3's `fusion.py`. This keeps them trivially
unit-testable and reusable if a future phase wants different metrics or
cutoffs.

`eval/` is a new top-level directory (sibling to `data/`, `src/`,
`scripts/`, `docs/`) because `eval_queries.json` is a curated, hand-
reviewed asset — it belongs in git like the catalog does, but it isn't
catalog *source data* (which lives in `data/`) and isn't a generated
*artifact* (which lives in the gitignored `artifacts/`).

## Data flow

1. **Drafting** (manual/semi-manual, not runtime code): draft 30-50
   domain-appropriate queries. For each, run all 4 modes' top-10 via the
   existing CLI/`search.py`, take the deduplicated union as the candidate
   pool, judge each candidate relevant/not by reading its name/category/
   description against the query intent. Save to `eval/eval_queries.json`.
   Show to the user for review/correction before treating as ground truth.
2. `scripts/run_eval.py`:
   - Loads `eval/eval_queries.json`.
   - For each query, calls `dense_search`, `bm25_search`,
     `hybrid_search(use_rerank=False)`, and `hybrid_search(use_rerank=True)`
     each at `top_k=10`.
   - Computes `recall_at_k`, `ndcg_at_k`, `mrr` for each (query, mode) pair
     against that query's `relevant_item_ids`.
   - Averages each metric across all queries, per mode — producing a
     4-row (mode) × 3-column (Recall@10 / NDCG@10 / MRR) table.
   - Writes the full table plus a methodology section (binary relevance,
     pooling method, the "relevant = pooled candidates only" caveat) to
     `docs/eval_results.md`, and prints the same table to console.
3. README's Phase 4 section gets a condensed version of the table plus a
   link to `docs/eval_results.md`.

## Testing

- `eval.py`: TDD unit tests with small, hand-computed examples for each
  of the three functions — e.g. a known `retrieved_ids` list and
  `relevant_ids` set where the expected Recall@10/NDCG@10/MRR value is
  verified by hand before being asserted in the test. Pure functions, no
  model, fast — same philosophy as Phase 3's `fusion.py` tests.
- `run_eval.py`: one test for the missing-eval-set-file error path
  (`SystemExit` with the missing path in the message), matching every
  other build script's established pattern in this project. The real,
  full eval run against actual data is a manual verification step (like
  Phase 1/2's build scripts), not something to unit test.

## Error handling

Same established pattern: `run_eval.py` raises a clear `SystemExit` if
`eval/eval_queries.json` doesn't exist yet. Missing search indices are
already handled inside `search.py`'s `load_dense_index`/`load_bm25_index`
(Phase 3) — `run_eval.py` doesn't need to duplicate that.

## Out of scope for Phase 4a (Phase 4b or later)

- Any caching/performance work in `search.py` (explicitly deferred — see
  "Scope boundary" above).
- Latency benchmarking (p50/p95/p99) — Phase 4b.
- Any change to retrieval logic itself (BM25/dense/fusion/reranker code)
  — Phase 4a only measures the existing Phase 3 pipeline, doesn't modify it.
- Graded relevance, additional metrics, or cutoffs beyond @10 — not
  requested by the original brief.
