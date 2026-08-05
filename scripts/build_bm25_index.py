"""Batch job: tokenize the full catalog and build the BM25 index.

Usage:
    python scripts/build_bm25_index.py
"""

import pandas as pd

from ecomsearch.bm25 import BM25Index
from ecomsearch.config import ARTIFACTS_DIR, BM25_INDEX_PATH, CATALOG_PATH


def main() -> None:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            f"Catalog not found at {CATALOG_PATH}. "
            "Make sure data/ecommerce_catalog_enriched.csv is present before building the index."
        )

    print(f"Loading catalog from {CATALOG_PATH}...")
    catalog = pd.read_csv(CATALOG_PATH, usecols=["item_id", "search_text"])

    print(f"Building BM25 index over {len(catalog)} products...")
    index = BM25Index()
    index.build(catalog["search_text"].tolist(), catalog["item_id"].to_numpy())

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(BM25_INDEX_PATH)
    print(f"Saved BM25 index to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
