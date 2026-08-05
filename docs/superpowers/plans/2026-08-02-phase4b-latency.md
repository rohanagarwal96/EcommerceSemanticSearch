# Phase 4b: Latency Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate `search.py`'s per-call reload of indexes/models via lazy singleton caching, then measure real p50/p95/p99 latency for all 4 retrieval modes in a warm process, documenting before/after numbers and hitting a <200ms p95 target for dense/bm25/hybrid.

**Architecture:** Module-level lazy singleton caches added directly to `src/ecomsearch/search.py` (no signature changes, no new call-site changes). A pure `percentile()` helper in a new `src/ecomsearch/latency.py`. A benchmark harness (`scripts/benchmark_latency.py`) that reuses Phase 4a's approved `eval/eval_queries.json`, repeated and shuffled for statistical stability, timed in a single warm process.

**Tech Stack:** Existing `search.py`/`config.py` from Phases 3-4a, plain Python `time`/`random`/`math` (no new dependencies).

**Spec:** `docs/superpowers/specs/2026-08-02-phase4b-latency-design.md`

**⚠️ Critical process note for whoever executes this plan:** Task 5 is **conditional** — it only runs if Task 4's real benchmark shows `dense`, `bm25`, or `hybrid` missing the <200ms p95 target. Read Task 4's results before deciding whether to run Task 5. If all three pass, skip directly from Task 4 to Task 6.

---

### Task 1: Config additions

**Files:**
- Modify: `src/ecomsearch/config.py`

- [ ] **Step 1: Append these lines to the end of `src/ecomsearch/config.py`**

The file currently ends with the Phase 4a additions (`EVAL_TOP_K = 10`). Append:
```python
LATENCY_RESULTS_PATH = REPO_ROOT / "docs" / "latency_results.md"
LATENCY_TARGET_MS_P95 = 200.0
BENCHMARK_REPEAT_COUNT = 10
BENCHMARK_SEED = 42
```

- [ ] **Step 2: Verify it imports**

Run:
```bash
python -c "from ecomsearch.config import LATENCY_RESULTS_PATH, LATENCY_TARGET_MS_P95, BENCHMARK_REPEAT_COUNT, BENCHMARK_SEED; print(LATENCY_RESULTS_PATH); print(LATENCY_TARGET_MS_P95); print(BENCHMARK_REPEAT_COUNT); print(BENCHMARK_SEED)"
```
Expected: prints the path to `docs/latency_results.md`, then `200.0`, then `10`, then `42`.

- [ ] **Step 3: Commit**

```bash
git add src/ecomsearch/config.py
git commit -m "feat: add Phase 4b latency config constants"
git push origin main
```

---

### Task 2: Percentile helper (TDD)

**Files:**
- Create: `src/ecomsearch/latency.py`
- Test: `tests/test_latency.py`

- [ ] **Step 1: Write the failing tests in `tests/test_latency.py`**

```python
import pytest

from ecomsearch.latency import percentile


def test_percentile_median_of_five_values():
    assert percentile([5, 1, 4, 2, 3], 50) == pytest.approx(3.0)


def test_percentile_ninety_interpolates_between_values():
    assert percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)


def test_percentile_single_value_returns_that_value():
    assert percentile([42.0], 95) == pytest.approx(42.0)


def test_percentile_zero_returns_minimum():
    assert percentile([3, 1, 2], 0) == pytest.approx(1.0)


def test_percentile_hundred_returns_maximum():
    assert percentile([3, 1, 2], 100) == pytest.approx(3.0)


def test_percentile_raises_on_empty_list():
    with pytest.raises(ValueError):
        percentile([], 50)
```

These expected values are hand-computed using linear interpolation between closest ranks (the same method `numpy.percentile` uses by default): for `p`, `rank = (p / 100) * (n - 1)` into the sorted list, interpolating between the values at the floor and ceiling of `rank`. E.g. for `[1,2,3,4,5]` at p=90: sorted is unchanged, `rank = 0.9 * 4 = 3.6`, interpolating 40% of the way from `sorted[3]=4` to `sorted[4]=5` gives `4 + 0.6*(5-4) = 4.6`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_latency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ecomsearch.latency'`

- [ ] **Step 3: Write `src/ecomsearch/latency.py`**

```python
"""Pure latency-measurement helpers (percentile computation, no I/O)."""

import math


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    rank = (p / 100) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]

    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_latency.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ecomsearch/latency.py tests/test_latency.py
git commit -m "feat: add percentile helper for latency benchmarking"
git push origin main
```

---

### Task 3: Cache indexes and models in `search.py` (TDD)

**Files:**
- Modify: `src/ecomsearch/search.py`
- Modify: `tests/test_search.py`

`search.py` currently reloads the FAISS index, BM25 index, embedder, reranker, and catalog CSV from scratch on every call. This task adds per-process lazy caching with no changes to any public function signature.

- [ ] **Step 1: Add the failing tests to `tests/test_search.py`**

First, add these two imports after the existing imports at the top of `tests/test_search.py` (which currently starts with `import numpy as np`, `import pandas as pd`, `import pytest`, then the `ecomsearch` imports):

```python
from ecomsearch.embeddings import Embedder
from ecomsearch.reranker import CrossEncoderReranker
```

Then append this fixture and these 3 tests to the end of `tests/test_search.py`:

```python
@pytest.fixture(autouse=True)
def reset_search_caches(monkeypatch):
    monkeypatch.setattr(search, "_dense_index", None, raising=False)
    monkeypatch.setattr(search, "_bm25_index", None, raising=False)
    monkeypatch.setattr(search, "_embedder", None, raising=False)
    monkeypatch.setattr(search, "_reranker", None, raising=False)
    monkeypatch.setattr(search, "_catalog", None, raising=False)


def test_dense_search_loads_index_and_embedder_only_once_across_calls(
    synthetic_catalog, monkeypatch
):
    load_calls = []
    original_load = ProductIndex.load.__func__

    def counting_load(cls, *args, **kwargs):
        load_calls.append(1)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(ProductIndex, "load", classmethod(counting_load))

    init_calls = []
    original_init = Embedder.__init__

    def counting_init(self, *args, **kwargs):
        init_calls.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Embedder, "__init__", counting_init)

    search.dense_search("almond milk", top_k=1)
    search.dense_search("paper towels", top_k=1)

    assert len(load_calls) == 1
    assert len(init_calls) == 1


def test_bm25_search_loads_index_only_once_across_calls(synthetic_catalog, monkeypatch):
    load_calls = []
    original_load = BM25Index.load.__func__

    def counting_load(cls, *args, **kwargs):
        load_calls.append(1)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(BM25Index, "load", classmethod(counting_load))

    search.bm25_search("almond milk", top_k=1)
    search.bm25_search("paper towels", top_k=1)

    assert len(load_calls) == 1


def test_hybrid_search_with_rerank_loads_reranker_and_catalog_only_once_across_calls(
    synthetic_catalog, monkeypatch
):
    init_calls = []
    original_init = CrossEncoderReranker.__init__

    def counting_init(self, *args, **kwargs):
        init_calls.append(1)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(CrossEncoderReranker, "__init__", counting_init)

    read_csv_calls = []
    original_read_csv = search.pd.read_csv

    def counting_read_csv(*args, **kwargs):
        read_csv_calls.append(1)
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(search.pd, "read_csv", counting_read_csv)

    search.hybrid_search("almond milk", top_k=1, use_rerank=True)
    search.hybrid_search("paper towels", top_k=1, use_rerank=True)

    assert len(init_calls) == 1
    assert len(read_csv_calls) == 1
```

`raising=False` on the reset fixture matters: at this point in the task, `search.py` doesn't have `_dense_index` etc. yet (that's the next step), so a strict `monkeypatch.setattr` would raise `AttributeError` on every test in the file via the autouse fixture. `raising=False` makes it a no-op until Step 3 adds the real attributes.

- [ ] **Step 2: Run tests to verify the 3 new tests fail**

Run: `pytest tests/test_search.py -v`
Expected: the original 6 tests still PASS; the 3 new tests FAIL with `assert 2 == 1` (today's code loads twice per two calls, since nothing is cached yet).

- [ ] **Step 3: Rewrite `src/ecomsearch/search.py`**

Replace the full file with:

```python
"""Retrieval orchestration: dense, keyword (BM25), and hybrid (RRF + rerank) search."""

import pandas as pd

from ecomsearch.bm25 import BM25Index
from ecomsearch.config import (
    BM25_INDEX_PATH,
    CANDIDATE_POOL_SIZE,
    CATALOG_PATH,
    INDEX_PATH,
    ITEM_IDS_PATH,
    RERANK_POOL_SIZE,
)
from ecomsearch.embeddings import Embedder
from ecomsearch.fusion import reciprocal_rank_fusion
from ecomsearch.index import ProductIndex
from ecomsearch.reranker import CrossEncoderReranker

_dense_index = None
_bm25_index = None
_embedder = None
_reranker = None
_catalog = None


def load_dense_index() -> ProductIndex:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No dense index found at {INDEX_PATH}. "
            "Run `python scripts/build_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def load_bm25_index() -> BM25Index:
    if not BM25_INDEX_PATH.exists():
        raise SystemExit(
            f"No BM25 index found at {BM25_INDEX_PATH}. "
            "Run `python scripts/build_bm25_index.py` first to build it."
        )
    return BM25Index.load(BM25_INDEX_PATH)


def _get_dense_index() -> ProductIndex:
    global _dense_index
    if _dense_index is None:
        _dense_index = load_dense_index()
    return _dense_index


def _get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = load_bm25_index()
    return _bm25_index


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        _catalog = pd.read_csv(CATALOG_PATH, usecols=["item_id", "search_text"]).set_index(
            "item_id"
        )
    return _catalog


def dense_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = _get_dense_index()
    embedder = _get_embedder()
    query_vector = embedder.embed_query(query)
    return index.search(query_vector, top_k)


def bm25_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = _get_bm25_index()
    return index.search(query, top_k)


def hybrid_search(query: str, top_k: int, use_rerank: bool = True) -> list[tuple[int, float]]:
    dense_results = dense_search(query, CANDIDATE_POOL_SIZE)
    bm25_results = bm25_search(query, CANDIDATE_POOL_SIZE)

    dense_ids = [item_id for item_id, _ in dense_results]
    bm25_ids = [item_id for item_id, _ in bm25_results]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])

    if not use_rerank:
        return fused[:top_k]

    candidate_ids = [item_id for item_id, _ in fused[:RERANK_POOL_SIZE]]
    catalog = _get_catalog()
    candidates = [(item_id, catalog.loc[item_id, "search_text"]) for item_id in candidate_ids]

    reranker = _get_reranker()
    reranked = reranker.rerank(query, candidates)
    return reranked[:top_k]
```

The only behavioral change from the current file: `load_dense_index()`/`load_bm25_index()` (with their existing helpful `SystemExit` messages) are now called only from inside `_get_dense_index()`/`_get_bm25_index()`, and `Embedder()`/`CrossEncoderReranker()`/the catalog `pd.read_csv()` are wrapped in equivalent lazy-cache getters. `dense_search`/`bm25_search`/`hybrid_search` keep identical signatures and behavior.

- [ ] **Step 4: Run tests to verify they all pass**

Run: `pytest tests/test_search.py -v`
Expected: PASS (9 tests — the original 6 plus the 3 new caching tests)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest -v`
Expected: all tests pass (Phase 1-4a's 50 plus Task 2's 6 plus this task's 3 new = 59 total so far).

- [ ] **Step 6: Commit**

```bash
git add src/ecomsearch/search.py tests/test_search.py
git commit -m "perf: cache indexes and models across search.py calls"
git push origin main
```

---

### Task 4: Benchmark harness (TDD + real run)

**Files:**
- Create: `scripts/benchmark_latency.py`
- Test: `tests/test_benchmark_latency.py`

- [ ] **Step 1: Write the failing test in `tests/test_benchmark_latency.py`**

```python
import pytest

import benchmark_latency


def test_main_exits_with_clear_message_when_eval_queries_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(benchmark_latency, "EVAL_QUERIES_PATH", missing_path)

    with pytest.raises(SystemExit) as excinfo:
        benchmark_latency.main()

    assert "does_not_exist.json" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_latency.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark_latency'`

- [ ] **Step 3: Write `scripts/benchmark_latency.py`**

```python
"""Latency benchmark: measure p50/p95/p99 latency for all 4 retrieval modes
in a warm, cached process, and write results to docs/latency_results.md.

Usage:
    python scripts/benchmark_latency.py
"""

import json
import random
import time

from ecomsearch.config import (
    BENCHMARK_REPEAT_COUNT,
    BENCHMARK_SEED,
    EVAL_QUERIES_PATH,
    EVAL_TOP_K,
    LATENCY_RESULTS_PATH,
    LATENCY_TARGET_MS_P95,
)
from ecomsearch.latency import percentile
from ecomsearch.search import bm25_search, dense_search, hybrid_search

MODES = {
    "dense": lambda query, top_k: dense_search(query, top_k),
    "bm25": lambda query, top_k: bm25_search(query, top_k),
    "hybrid": lambda query, top_k: hybrid_search(query, top_k, use_rerank=False),
    "hybrid-rerank": lambda query, top_k: hybrid_search(query, top_k, use_rerank=True),
}

# Only these modes are held to the <200ms p95 bar. hybrid-rerank's cross-encoder
# pass is inherently the slowest step and is reported without a pass/fail verdict --
# see docs/superpowers/specs/2026-08-02-phase4b-latency-design.md.
LATENCY_GATED_MODES = ("dense", "bm25", "hybrid")


def main() -> None:
    if not EVAL_QUERIES_PATH.exists():
        raise SystemExit(
            f"Eval query set not found at {EVAL_QUERIES_PATH}. "
            "Draft eval/eval_queries.json before running this script."
        )

    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        eval_queries = json.load(f)

    queries = [entry["query"] for entry in eval_queries]
    print(f"Loaded {len(queries)} eval queries.")

    timed_queries = queries * BENCHMARK_REPEAT_COUNT
    random.Random(BENCHMARK_SEED).shuffle(timed_queries)
    print(
        f"Benchmarking {len(timed_queries)} calls per mode "
        f"({BENCHMARK_REPEAT_COUNT}x repeats, shuffled)."
    )

    latencies_ms = {}

    for mode, search_fn in MODES.items():
        print(f"Warming up {mode}...")
        search_fn(queries[0], EVAL_TOP_K)  # untimed warm-up call, loads caches

        print(f"Timing {mode}...")
        samples = []
        for query in timed_queries:
            start = time.perf_counter()
            search_fn(query, EVAL_TOP_K)
            samples.append((time.perf_counter() - start) * 1000)

        latencies_ms[mode] = samples
        print(
            f"{mode}: p50={percentile(samples, 50):.1f}ms "
            f"p95={percentile(samples, 95):.1f}ms "
            f"p99={percentile(samples, 99):.1f}ms"
        )

    lines = [
        "# Latency Results",
        "",
        "## Methodology",
        "",
        f"- {len(timed_queries)} timed calls per mode: the {len(queries)} hand-labeled",
        f"  eval queries ({EVAL_QUERIES_PATH.name}) repeated {BENCHMARK_REPEAT_COUNT}x each,",
        "  shuffled, run serially in a single warm process (one untimed warm-up",
        "  call per mode first, so first-load cost doesn't pollute the distribution).",
        "- 'Before' baseline: Phase 4a's real eval run, 140 calls with zero caching,",
        "  ~16m43s wall-clock (~7s/call average). Not re-measured here -- re-running",
        "  the uncached path for a full percentile breakdown would take hours on this",
        "  CPU and would only re-confirm an already-known number.",
        f"- Target: p95 < {LATENCY_TARGET_MS_P95:.0f}ms for dense/bm25/hybrid.",
        "  hybrid-rerank is measured and reported honestly but not held to this bar --",
        "  see docs/eval_results.md for why reranking is a quality/latency tradeoff,",
        "  not purely a latency question.",
        "",
        "## Results (after caching)",
        "",
        "| Mode | p50 (ms) | p95 (ms) | p99 (ms) | Verdict |",
        "|---|---|---|---|---|",
    ]

    for mode, samples in latencies_ms.items():
        p50 = percentile(samples, 50)
        p95 = percentile(samples, 95)
        p99 = percentile(samples, 99)
        if mode in LATENCY_GATED_MODES:
            verdict = "PASS" if p95 < LATENCY_TARGET_MS_P95 else "FAIL"
        else:
            verdict = "not gated"
        lines.append(f"| {mode} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {verdict} |")

    LATENCY_RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote results to {LATENCY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_latency.py -v`
Expected: PASS

- [ ] **Step 5: Run the real benchmark**

Run: `python scripts/benchmark_latency.py`

Expected timing: `dense`/`bm25`/`hybrid` should each finish their 350 calls quickly now that indexes/models are cached (likely well under a minute each — a cached flat FAISS search and BM25 lookup over ~55k items are each a few ms, with query embedding inference the main remaining cost). `hybrid-rerank` will be the slow one — it still runs the cross-encoder over up to 50 candidates on every one of its 350 calls, so expect several minutes for that mode alone even with caching (caching removes reload cost, not the actual reranker inference cost). Total run time is likely 5-15 minutes, dominated by `hybrid-rerank`.

If you're concerned about a tool-level timeout on a single long-running foreground command, launch it detached, the same technique used in Phase 4a:
```bash
nohup python scripts/benchmark_latency.py > benchmark_latency.log 2>&1 & disown
```
Then track it via `Get-Process python | Select-Object Id,CPU,StartTime` (matching the actual process by `StartTime`, since `$!` after `nohup ... & disown` captures a shell-wrapper PID, not the real Python PID) and check `benchmark_latency.log` for progress.

Confirm afterward: `docs/latency_results.md` exists and contains a 4-row results table. **Read the actual PASS/FAIL verdicts for `dense`/`bm25`/`hybrid`** — this determines whether Task 5 is needed.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_latency.py tests/test_benchmark_latency.py docs/latency_results.md
git commit -m "feat: add latency benchmark script with real results"
git push origin main
```

(`docs/latency_results.md` IS committed, matching Phase 4a's `docs/eval_results.md` precedent — a human-readable results doc, not a regenerable binary artifact.)

---

### Task 5 (CONDITIONAL — only if Task 4 shows a FAIL verdict for `dense`, `bm25`, or `hybrid`)

**Skip this task entirely if Task 4's `docs/latency_results.md` shows PASS for all three gated modes.**

**Files:**
- Modify: `src/ecomsearch/config.py`
- Modify: `docs/latency_results.md`

- [ ] **Step 1: Identify the failing mode(s) and margin**

From `docs/latency_results.md`, note which mode(s) show FAIL and by how much (e.g. "hybrid p95 = 240ms, 40ms over target").

- [ ] **Step 2: Try reducing `CANDIDATE_POOL_SIZE` first**

`hybrid_search` fetches `CANDIDATE_POOL_SIZE` (currently `100`) candidates from *both* `dense_search` and `bm25_search` before fusing — this is the main lever that scales work per query in `hybrid`/`hybrid-rerank`. In `src/ecomsearch/config.py`, change:
```python
CANDIDATE_POOL_SIZE = 50
```

- [ ] **Step 3: Re-run the benchmark and check quality impact**

Run: `python scripts/benchmark_latency.py`
Expected: the previously-failing mode(s) now show PASS.

A smaller candidate pool can reduce Recall@10 by shrinking what's available to fuse — confirm quality hasn't meaningfully regressed from Phase 4a's numbers:
```bash
pytest tests/test_search.py tests/test_fusion.py -v
python scripts/run_eval.py
```
Compare the new `docs/eval_results.md` numbers against Phase 4a's committed ones (`hybrid` was Recall@10=0.4437, NDCG@10=0.9360, MRR=0.9857). A small drop is an acceptable tradeoff; a large one means the pool size cut too deep — try a value between 50 and 100 instead.

- [ ] **Step 4: If dense or bm25 alone still exceed 200ms p95**

This would be unexpected — a single flat FAISS search or BM25 lookup over ~55,516 items should be a few ms. Profile with `py-spy dump --pid <PID> --locals` while a benchmark run is in progress (the same technique used to diagnose the Phase 1 `bge-base` slowdown) to find where time is actually going before considering an index-type change. Only as a last resort, if profiling shows the FAISS search itself (not query embedding) is the bottleneck, investigate switching `src/ecomsearch/index.py`'s `ProductIndex` from `IndexFlatIP` to an approximate index (e.g. `faiss.IndexIVFFlat`) — this needs a training step and changes the save/load format, so treat it as its own follow-up investigation with FAISS's own documentation, not a blind swap here.

- [ ] **Step 5: Document the change and commit**

Add a short note to `docs/latency_results.md`'s methodology section describing what was tuned and why (e.g. "`CANDIDATE_POOL_SIZE` reduced from 100 to 50 to bring `hybrid` p95 under the 200ms target; `docs/eval_results.md` re-run to confirm Recall@10 impact was acceptable — see updated numbers there.").

```bash
git add src/ecomsearch/config.py docs/latency_results.md
git commit -m "perf: tune candidate pool size to hit latency target"
git push origin main
```

If `docs/eval_results.md` also changed from the re-run in Step 3, commit that separately:
```bash
git add docs/eval_results.md
git commit -m "docs: re-run eval after candidate pool size tuning"
git push origin main
```

---

### Task 6: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass — Phase 1-4a's 50 plus Phase 4b's new tests (latency x6, search caching x3, benchmark_latency x1 = 10) = 60 total.

- [ ] **Step 2: Update `README.md`**

Read the current file first, then:
- In the Status section, check off `- [ ] Phase 4 — Evaluation and latency engineering` (now fully done — both evaluation and latency have been measured). Update the status sentence to also mention latency, e.g. append after the existing Phase 4a evaluation sentence: "Latency has been benchmarked in a warmed, cached process and meets a <200ms p95 target for dense/bm25/hybrid modes — see [Latency Results](docs/latency_results.md)."
- Add a new `## Latency` section (after the existing `## Evaluation` section, before `## Setup`), containing the p50/p95/p99 results table copied from `docs/latency_results.md`'s `## Results (after caching)` section — read that file first to get the real numbers — with a link to `docs/latency_results.md` for full methodology.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: add Phase 4b latency results to README"
git push origin main
```
