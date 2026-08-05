# Phase 4b: Latency Engineering — Design

## Context

Phase 4a measured retrieval quality (Recall@10/NDCG@10/MRR across dense/bm25/hybrid/hybrid-rerank) and is complete. Phase 4b is the other half of the original brief's Phase 4 ("Evaluation and latency engineering"): benchmark latency and optimize until it's consistently under 200ms, documenting before/after numbers.

The dominant known cost is that `src/ecomsearch/search.py`'s `dense_search`, `bm25_search`, and `hybrid_search` currently reload the FAISS index, the BM25 pickle, and instantiate fresh neural models (`Embedder()`, `CrossEncoderReranker()`) from scratch on **every single call** — this was deliberately deferred from Phase 3. Reading `search.py` for this design also surfaced a second, previously-undocumented instance of the same problem: `hybrid_search`'s rerank path does `pd.read_csv(CATALOG_PATH)` fresh on every call too. Phase 4a's real eval run empirically confirmed the cost: 140 calls took ~16m43s wall-clock (~7s/call average) on the dev machine's Intel i7-8650U (4-core/8-thread, 15W laptop chip).

Phase 5 (the FastAPI serving layer) does not exist yet, so there is no live server to load-test. This design treats Phase 4b as: fix the caching, then measure what a warm, cached process's latency actually looks like — which is the latency profile Phase 5's server will have once it reuses this same `search.py`.

## Goals

1. Eliminate redundant index/model/catalog reloading in `search.py` via lazy, per-process caching, with no changes to existing function signatures or call sites (CLI, `run_eval.py`).
2. Benchmark p50/p95/p99 latency for all 4 modes in a warm, cached, single-process harness.
3. Hit p95 < 200ms for `dense`, `bm25`, and `hybrid` (RRF, no rerank). `hybrid-rerank` is measured and documented honestly but not forced under this bar — Phase 4a already showed reranking can hurt quality on some queries, and forcing latency down further (e.g. shrinking `RERANK_POOL_SIZE`) is a quality tradeoff decision, not a pure latency-engineering one.
4. Document before/after numbers in `docs/latency_results.md`, then update the README to finally check off the Phase 4 checklist item.

## Non-Goals

- No live FastAPI server or HTTP-level benchmarking (Phase 5's job).
- No concurrent/multi-client load testing — this measures serial per-query latency in a warm process, not throughput under concurrent load.
- No automated CI latency-regression gate (Phase 7: production hygiene).
- No forcing hybrid-rerank under 200ms at the cost of quality.
- No FAISS index type change unless the post-caching benchmark actually shows the index search itself is a meaningful fraction of latency (see below).

## Design

### 1. Caching in `search.py`

Add module-level singleton caches, lazily populated on first use:

```python
_dense_index = None
_bm25_index = None
_embedder = None
_reranker = None
_catalog = None


def _get_dense_index():
    global _dense_index
    if _dense_index is None:
        _dense_index = load_dense_index()
    return _dense_index


def _get_bm25_index():
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = load_bm25_index()
    return _bm25_index


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_catalog():
    global _catalog
    if _catalog is None:
        _catalog = pd.read_csv(CATALOG_PATH, usecols=["item_id", "search_text"]).set_index(
            "item_id"
        )
    return _catalog
```

`dense_search`, `bm25_search`, and `hybrid_search` change internally to call these `_get_*()` helpers instead of `load_dense_index()`/`Embedder()`/`load_bm25_index()`/`CrossEncoderReranker()`/`pd.read_csv(...)` directly. Public signatures and behavior are unchanged — this is a pure internal fix. `load_dense_index()` and `load_bm25_index()` (the existing functions with their `SystemExit`-on-missing-file behavior) stay as-is and are called only from inside the new `_get_*()` wrappers, so the "helpful error if you forgot to build the index" behavior is preserved.

**Testing**: a test that calls `dense_search()` twice with a mocked/spied `Embedder` and `ProductIndex.load`, asserting each was constructed exactly once across both calls — proving the cache actually caches, not just that results are unchanged. Similarly for `bm25_search` (spy on `BM25Index.load`) and `hybrid_search` (spy on `CrossEncoderReranker` and the catalog read).

### 2. Benchmark harness — `scripts/benchmark_latency.py`

- Single warm process. Imports `search.py`, issues one warm-up call per mode first (excluded from timing) so the one-time load cost doesn't pollute the distribution.
- Reuses `eval/eval_queries.json`'s 35 already-approved queries, repeated 10x each and shuffled (`random.shuffle`) → 350 timed calls per mode. No new query set needs drafting.
- Times each call via `time.perf_counter()`.
- Percentile computation lives in a small pure function in `src/ecomsearch/latency.py`:
  ```python
  def percentile(values: list[float], p: float) -> float: ...
  ```
  TDD'd with hand-computed expected values, matching `eval.py`'s test style (pure function, no I/O, `tests/test_latency.py`).
- Writes `docs/latency_results.md` with:
  - Methodology section: cites Phase 4a's real ~7s/call uncached measurement (140 calls, ~16m43s) as the documented "before" baseline — re-running a full formal uncached benchmark (350 calls × 4 modes, hours on this CPU) would just re-prove an already-known number, so it's cited rather than re-measured.
  - Results table: p50/p95/p99 per mode (ms), "after" (cached) numbers.
  - Pass/fail line against the <200ms p95 bar for dense/bm25/hybrid. `hybrid-rerank`'s number is reported without a pass/fail judgment, with a one-line note on why (quality/latency tradeoff, see Phase 4a's `docs/eval_results.md` finding on reranker behavior).

**Testing**: `tests/test_benchmark_latency.py` covers the missing-eval-file `SystemExit` path, mirroring `tests/test_run_eval.py`'s existing pattern. A real run requires live models, so — like `run_eval.py` — there's no fast unit test for the full benchmark; it's validated by actually running it once during implementation.

### 3. FAISS index type — measure, then decide

At ~55,516 products × 384 dimensions (bge-small-en-v1.5), an exact `IndexFlatIP` search is expected to take low single-digit milliseconds — the dominant latency cost has always been model loading and query-embedding inference, not the index search itself. After caching is fixed and the benchmark run, if dense/bm25/hybrid already clear the <200ms p95 bar, the index type stays `IndexFlatIP` and this is documented in `docs/latency_results.md` as a "measured and found unnecessary" decision. If a mode is still over the bar, the plan includes one gated follow-up task (tune `CANDIDATE_POOL_SIZE`/`RERANK_POOL_SIZE` first, since those directly control work-per-query; approximate indexing (IVF/HNSW) only as a last resort) — scoped to whichever mode's measurement actually demands it, not committed upfront.

### 4. Documentation

- `docs/latency_results.md` — new file, same shape/spirit as `docs/eval_results.md`: methodology, before/after table, pass/fail verdict.
- `README.md` — add latency numbers alongside the existing `## Evaluation` section (or a new `## Latency` section), and finally check off `- [ ] Phase 4 — Evaluation and latency engineering` in the Status section, since both halves (4a evaluation, 4b latency) will now be done.

## File Summary

| File | Change |
|---|---|
| `src/ecomsearch/search.py` | Modify — add lazy singleton caches |
| `src/ecomsearch/latency.py` | New — pure `percentile()` helper |
| `tests/test_latency.py` | New — TDD for `percentile()` |
| `tests/test_search_caching.py` | New — proves caching via mock/spy call counts |
| `scripts/benchmark_latency.py` | New — benchmark harness |
| `tests/test_benchmark_latency.py` | New — missing-eval-file `SystemExit` test |
| `docs/latency_results.md` | New — before/after results doc |
| `README.md` | Modify — latency section + Phase 4 checklist checked off |
