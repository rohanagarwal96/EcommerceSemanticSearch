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

## Results (after caching)

| Mode | p50 (ms) | p95 (ms) | p99 (ms) | Verdict |
|---|---|---|---|---|
| dense | 52.8 | 73.3 | 127.8 | PASS |
| bm25 | 91.2 | 168.2 | 191.1 | PASS |
| hybrid | 156.6 | 229.6 | 257.5 | FAIL |
| hybrid-rerank | 4623.4 | 8259.8 | 9915.6 | not gated |
