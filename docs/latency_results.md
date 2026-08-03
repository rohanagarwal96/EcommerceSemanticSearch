# Latency Results

## Methodology

- 350 timed calls per mode: the 35 hand-labeled
  eval queries (eval_queries.json) repeated 10x each,
  shuffled, run serially in a single warm process (one untimed warm-up
  call per mode first, so first-load cost doesn't pollute the distribution).
- 'Before' baseline: Phase 4a's real eval run, 140 calls with zero caching,
  ~16m43s wall-clock (~7s/call average). Not re-measured here -- re-running
  the uncached path for a full percentile breakdown would take hours on this
  CPU and would only re-confirm an already-known number.
- Target: p95 < 200ms for dense/bm25/hybrid.
  hybrid-rerank is measured and reported honestly but not held to this bar --
  see docs/eval_results.md for why reranking is a quality/latency tradeoff,
  not purely a latency question.

### Latency engineering: what was tried, in order

The original pool=100/sequential baseline measured `hybrid` p95=229.6ms,
over target, while `dense` and `bm25` individually passed. Several
approaches were tried, cleanly benchmarked on this machine after confirming
no CPU contention from other apps:

1. **Candidate pool size tuning** (`CANDIDATE_POOL_SIZE`, sequential dense +
   bm25): 100 -> 229.6ms, 50 -> 206.8ms, 30 -> 202.8ms. Diminishing returns,
   and none cleared 200ms -- smaller pools trade away retrieval recall
   headroom for a shrinking latency win. `CANDIDATE_POOL_SIZE` was left at
   its original value of 100 to preserve full retrieval quality; the win
   here was never large enough to justify the quality tradeoff anyway.

2. **Threading -- tried and kept.** `dense_search` and `bm25_search` are
   independent I/O-free CPU calls with no shared state, so they were
   dispatched concurrently via a shared `ThreadPoolExecutor` (module-level
   lazy singleton, matching the existing `_get_*()` caching pattern) inside
   `hybrid_search`. Confirmed genuinely concurrent (not just nominally
   parallel) by direct instrumentation: wall-clock start/end timestamps on
   real `hybrid_search` calls showed `dense_search` and `bm25_search`
   starting within ~0.5ms of each other with execution windows overlapping
   almost completely. This is a modest but real win: pool=100, hybrid p95
   settled around 213-214ms (vs. 229.6ms sequential), reproduced twice
   independently. The gain is smaller than naive "run two things in
   parallel" reasoning would predict because this machine's CPU has limited
   spare capacity -- two concurrent numpy/BLAS-heavy workloads compete for
   the same cores, and `dense_search`'s own wall-clock duration measurably
   inflates under concurrent load (observed ~170-190ms concurrently vs.
   ~53-79ms alone), eating into the theoretical parallelism benefit.

3. **BM25 O(n) top-k selection -- attempted, then reverted.**
   `BM25Index.search()` originally did a full `O(n log n)` sort
   (`np.argsort(-scores, kind="stable")`) over the entire ~55,516-item
   score array before slicing to `top_k`, a cost that doesn't shrink no
   matter what `top_k`/pool size is requested. This looked like a promising
   target, since pool-size tuning had plateaued. It was replaced with
   `np.argpartition` (O(n) top-k selection) plus `np.lexsort` for the final
   ordering step, intended to preserve the pre-existing first-inserted-wins
   tie-breaking guarantee. It was reverted for two independent reasons,
   either one of which alone would have been sufficient to drop it:

   - **No meaningful latency benefit.** Isolated timing of
     `BM25Index.search()`'s internals over the same 350-call workload
     showed `self._bm25.get_scores(...)` alone accounts for p50=84.16ms /
     p95=150.27ms, essentially the *entire* cost of the full `search()`
     call (p50=82.03ms / p95=150.15ms). The `argpartition`+`lexsort` top-k
     step itself measured p50=0.36ms / p95=0.57ms -- negligible.
     `rank_bm25` brute-force scores every one of the ~55,516 corpus items
     on every query, an `O(n)` cost paid *before* any sorting or selection
     begins, so no top-k algorithm can reduce it. This is why `bm25` p95
     stayed in the 146-170ms range across every configuration tried
     (pool size, sort algorithm, or both).
   - **A real correctness regression.** A targeted stress test (50
     randomized trials with heavily tie-quantized score arrays and
     `top_k < n`) found the `argpartition`+`lexsort` approach diverges from
     the true stable-sort tie-break order in 43/50 (86%) of trials. The
     bug is a boundary-tie problem: `np.argpartition` provides no guarantee
     about *which* tied items land inside the returned top-k when more
     items are tied at the cutoff score than fit in `top_k` -- it can
     silently exclude the correct (lowest-original-index) item in favor of
     a different tied one. `np.lexsort` afterward can only reorder whatever
     subset `argpartition` happened to keep; it cannot recover items that
     were wrongly excluded. This was not caught by the initial regression
     test (which only exercised the no-cutoff case, `top_k == len(array)`,
     where no item is ever excluded) but showed up as a real, measurable
     quality regression in standalone `bm25` mode eval:
     Recall@10 0.4120 -> 0.4034, NDCG@10 0.8967 -> 0.8819 (MRR unchanged,
     since the #1 result wasn't affected). `hybrid` mode's quality was not
     affected in this particular 35-query eval set, but the same latent
     bug is present in that code path too.

   Given the change had no meaningful latency upside even when correct, it
   wasn't worth pursuing a more complex, genuinely tie-safe O(n) selection
   (e.g. partitioning on a lexicographic `(score, -index)` key). `bm25.py`
   and its tests were reverted to their original committed state
   (`git diff 31a40a6 -- src/ecomsearch/bm25.py tests/test_bm25.py` is
   empty) rather than shipping either the regression or extra engineering
   effort for zero payoff.

### Final state

`dense` and `bm25` individually pass the 200ms p95 bar with real margin.
`hybrid` (threading only, `CANDIDATE_POOL_SIZE` at its original value of
100) sits at p95=214.3ms -- a real ~15ms improvement over the 229.6ms
starting point, but still over the 200ms target. `hybrid-rerank` remains
ungated as designed (the cross-encoder pass is inherently the dominant
cost there).

Fully closing `hybrid`'s remaining gap would require replacing
`rank_bm25`'s brute-force full-corpus scoring with a true inverted-index
BM25 implementation -- a materially larger project, out of scope for this
phase -- or running on hardware with more spare CPU capacity for genuine
parallel execution. The threading change is kept as a real, correct,
unconditional improvement; the BM25 top-k change was not, and is not
shipped.

## Results (after caching)

| Mode | p50 (ms) | p95 (ms) | p99 (ms) | Verdict |
|---|---|---|---|---|
| dense | 53.7 | 106.9 | 142.9 | PASS |
| bm25 | 83.9 | 146.3 | 150.4 | PASS |
| hybrid | 162.1 | 214.3 | 225.6 | FAIL |
| hybrid-rerank | 4735.2 | 6207.1 | 6662.9 | not gated |
