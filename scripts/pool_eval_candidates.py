"""Helper: for a list of candidate queries, pool the top-10 results from all
4 retrieval modes and print each unique candidate's details for manual
relevance judging. Not part of the automated eval pipeline -- a one-time
tool used to help draft eval/eval_queries.json.

Usage:
    python scripts/pool_eval_candidates.py "organic almond milk" "gluten free pasta"
"""
import sys

import pandas as pd

from ecomsearch.config import CATALOG_PATH
from ecomsearch.search import bm25_search, dense_search, hybrid_search

MODES = {
    "dense": lambda query: dense_search(query, 10),
    "bm25": lambda query: bm25_search(query, 10),
    "hybrid": lambda query: hybrid_search(query, 10, use_rerank=False),
    "hybrid-rerank": lambda query: hybrid_search(query, 10, use_rerank=True),
}


def main() -> None:
    queries = sys.argv[1:]
    if not queries:
        raise SystemExit(
            'Usage: python scripts/pool_eval_candidates.py "query one" "query two" ...'
        )

    catalog = pd.read_csv(
        CATALOG_PATH, usecols=["item_id", "name", "brand", "category_path", "description"]
    ).set_index("item_id")

    for query in queries:
        pooled_ids = set()
        for search_fn in MODES.values():
            results = search_fn(query)
            pooled_ids.update(item_id for item_id, _ in results)

        print(f"\n=== Query: {query!r} ({len(pooled_ids)} pooled candidates) ===")
        for item_id in sorted(pooled_ids):
            row = catalog.loc[item_id]
            description = str(row["description"])[:120] if pd.notna(row["description"]) else ""
            print(
                f"  {item_id}\t{row['name']}\t{row['brand']}\t"
                f"{row['category_path']}\t{description}"
            )


if __name__ == "__main__":
    main()
