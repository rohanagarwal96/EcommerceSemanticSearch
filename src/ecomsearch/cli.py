"""CLI entrypoint for semantic product search."""
import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table

from ecomsearch.config import CATALOG_PATH, DEFAULT_TOP_K, INDEX_PATH, ITEM_IDS_PATH
from ecomsearch.embeddings import Embedder
from ecomsearch.index import ProductIndex


def load_index() -> ProductIndex:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No index found at {INDEX_PATH}. "
            "Run `python scripts/build_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def search(query: str, top_k: int) -> None:
    index = load_index()
    embedder = Embedder()
    query_vector = embedder.embed_query(query)
    results = index.search(query_vector, top_k)

    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["item_id", "name", "brand", "category_path"],
    ).set_index("item_id")

    table = Table(title=f'Top {len(results)} results for "{query}"')
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

    args = parser.parse_args()

    if args.command == "search":
        search(args.query, args.top_k)


if __name__ == "__main__":
    main()
