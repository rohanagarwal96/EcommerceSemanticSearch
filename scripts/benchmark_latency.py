"""Latency benchmark: measure p50/p95/p99 latency for all 4 retrieval modes
in a warm, cached process, and write results to docs/latency_results.md.

Usage:
    python scripts/benchmark_latency.py

Warning: this script overwrites docs/latency_results.md entirely, including
the hand-written "what was tried" investigation narrative added after the
Phase 4b tuning work. Re-running it will regenerate the Methodology/Results
sections but silently delete that narrative -- copy it out first if you
want to keep it.
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
