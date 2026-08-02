"""Batch job: run all 4 retrieval modes against the hand-labeled eval set,
compute Recall@10/NDCG@10/MRR per mode, and write a comparison table.

Usage:
    python scripts/run_eval.py
"""
import json

from ecomsearch.config import EVAL_QUERIES_PATH, EVAL_RESULTS_PATH, EVAL_TOP_K
from ecomsearch.eval import mrr, ndcg_at_k, recall_at_k
from ecomsearch.search import bm25_search, dense_search, hybrid_search

MODES = {
    "dense": lambda query, top_k: dense_search(query, top_k),
    "bm25": lambda query, top_k: bm25_search(query, top_k),
    "hybrid": lambda query, top_k: hybrid_search(query, top_k, use_rerank=False),
    "hybrid-rerank": lambda query, top_k: hybrid_search(query, top_k, use_rerank=True),
}


def main() -> None:
    if not EVAL_QUERIES_PATH.exists():
        raise SystemExit(
            f"Eval query set not found at {EVAL_QUERIES_PATH}. "
            "Draft eval/eval_queries.json before running this script."
        )

    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        eval_queries = json.load(f)

    print(f"Loaded {len(eval_queries)} eval queries.")

    scores = {mode: {"recall": [], "ndcg": [], "mrr": []} for mode in MODES}

    for entry in eval_queries:
        query = entry["query"]
        relevant_ids = set(entry["relevant_item_ids"])
        print(f"Evaluating: {query!r}")

        for mode, search_fn in MODES.items():
            results = search_fn(query, EVAL_TOP_K)
            retrieved_ids = [item_id for item_id, _ in results]

            scores[mode]["recall"].append(recall_at_k(retrieved_ids, relevant_ids, EVAL_TOP_K))
            scores[mode]["ndcg"].append(ndcg_at_k(retrieved_ids, relevant_ids, EVAL_TOP_K))
            scores[mode]["mrr"].append(mrr(retrieved_ids, relevant_ids))

    lines = [
        "# Evaluation Results",
        "",
        "## Methodology",
        "",
        f"- {len(eval_queries)} hand-labeled queries, binary relevance.",
        "- Relevant items identified via pooling: the deduplicated union of the",
        "  top-10 results from all 4 modes, judged relevant/not by hand.",
        '  "Relevant" therefore means "relevant among pooled candidates", not',
        "  an exhaustive ground truth over the full 55,516-row catalog.",
        "",
        "## Results",
        "",
        "| Mode | Recall@10 | NDCG@10 | MRR |",
        "|---|---|---|---|",
    ]

    for mode in MODES:
        n = len(eval_queries)
        avg_recall = sum(scores[mode]["recall"]) / n
        avg_ndcg = sum(scores[mode]["ndcg"]) / n
        avg_mrr = sum(scores[mode]["mrr"]) / n
        lines.append(f"| {mode} | {avg_recall:.4f} | {avg_ndcg:.4f} | {avg_mrr:.4f} |")
        print(f"{mode}: Recall@10={avg_recall:.4f} NDCG@10={avg_ndcg:.4f} MRR={avg_mrr:.4f}")

    EVAL_RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote results to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
