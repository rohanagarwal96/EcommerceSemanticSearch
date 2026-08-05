"""Batch job: embed the full catalog and build the FAISS index.

Usage:
    python scripts/build_index.py
"""

import pandas as pd

from ecomsearch.config import ARTIFACTS_DIR, CATALOG_PATH, INDEX_PATH, ITEM_IDS_PATH
from ecomsearch.embeddings import Embedder
from ecomsearch.index import ProductIndex


def main() -> None:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            f"Catalog not found at {CATALOG_PATH}. "
            "Make sure data/ecommerce_catalog_enriched.csv is present before building the index."
        )

    print(f"Loading catalog from {CATALOG_PATH}...")
    catalog = pd.read_csv(CATALOG_PATH, usecols=["item_id", "search_text"])

    print(f"Embedding {len(catalog)} products with bge-small-en-v1.5...")
    embedder = Embedder()
    vectors = embedder.embed_documents(catalog["search_text"].tolist())

    print("Building FAISS index...")
    index = ProductIndex(dim=vectors.shape[1])
    index.add(vectors, catalog["item_id"].to_numpy())

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH, ITEM_IDS_PATH)
    print(f"Saved index to {INDEX_PATH} and id mapping to {ITEM_IDS_PATH}")


if __name__ == "__main__":
    main()
