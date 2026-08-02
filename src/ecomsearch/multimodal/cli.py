"""CLI entrypoint for cross-modal (text-to-image) product search."""
import argparse
import re
import shutil

import pandas as pd
from rich.console import Console
from rich.table import Table

from ecomsearch.index import ProductIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import (
    DATASET_IMAGES_DIR,
    DEFAULT_TOP_K,
    DEMO_RESULTS_DIR,
    INDEX_PATH,
    ITEM_IDS_PATH,
    SUBSET_METADATA_PATH,
)


def load_index() -> ProductIndex:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No index found at {INDEX_PATH}. "
            "Run `python scripts/build_multimodal_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def _slugify(query: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug or "query"


def search(query: str, top_k: int) -> None:
    index = load_index()
    embedder = ClipEmbedder()
    query_vector = embedder.embed_text([query])[0]
    results = index.search(query_vector, top_k)

    metadata = pd.read_csv(SUBSET_METADATA_PATH).set_index("item_id")

    table = Table(title=f'Top {len(results)} image results for "{query}"')
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Item ID", justify="right")
    table.add_column("Display Name")
    table.add_column("Category")

    output_dir = DEMO_RESULTS_DIR / _slugify(query)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, (item_id, score) in enumerate(results, start=1):
        row = metadata.loc[item_id]
        table.add_row(
            str(rank),
            f"{score:.4f}",
            str(item_id),
            str(row["display name"]),
            str(row["category"]),
        )
        source_image = DATASET_IMAGES_DIR / row["image"]
        shutil.copy(source_image, output_dir / f"{rank:02d}_{row['image']}")

    Console().print(table)
    print(f"Copied {len(results)} images to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-modal (text-to-image) product search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the image dataset by text query")
    search_parser.add_argument("query", help="Free-text search query")
    search_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return"
    )

    args = parser.parse_args()

    if args.command == "search":
        search(args.query, args.top_k)


if __name__ == "__main__":
    main()
