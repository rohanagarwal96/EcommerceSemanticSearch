"""CLI entrypoint for semantic product search."""
import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from ecomsearch.config import CATALOG_PATH, DEFAULT_TOP_K
from ecomsearch.search import bm25_search, dense_search, hybrid_search


def search(query: str, top_k: int, mode: str) -> None:
    if mode == "dense":
        results = dense_search(query, top_k)
    elif mode == "bm25":
        results = bm25_search(query, top_k)
    elif mode == "hybrid":
        results = hybrid_search(query, top_k, use_rerank=False)
    elif mode == "hybrid-rerank":
        results = hybrid_search(query, top_k, use_rerank=True)
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["item_id", "name", "brand", "category_path"],
    ).set_index("item_id")

    table = Table(title=f'Top {len(results)} results for "{query}" (mode={mode})')
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Item ID", justify="right")
    table.add_column("Name")
    table.add_column("Brand")
    table.add_column("Category")

    for rank, (item_id, score) in enumerate(results, start=1):
        row = catalog.loc[item_id]
        table.add_row(
            str(rank),
            f"{score:.4f}",
            str(item_id),
            str(row["name"]),
            str(row["brand"]),
            str(row["category_path"]),
        )

    Console().print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic product search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the catalog")
    search_parser.add_argument("query", help="Free-text search query")
    search_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return"
    )
    search_parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid", "hybrid-rerank"],
        default="hybrid-rerank",
        help="Retrieval mode",
    )

    args = parser.parse_args()

    if args.command == "search":
        search(args.query, args.top_k, args.mode)


if __name__ == "__main__":
    main()
