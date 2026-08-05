"""One-time script: upload the local CLIP image FAISS index into Qdrant Cloud.

Usage:
    python scripts/upload_multimodal_index_to_qdrant.py
"""

import faiss
import numpy as np

from ecomsearch.multimodal.config import INDEX_PATH, ITEM_IDS_PATH, QDRANT_IMAGE_COLLECTION_NAME
from ecomsearch.qdrant_index import QdrantIndex

UPSERT_BATCH_SIZE = 256


def main() -> None:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No local multimodal index found at {INDEX_PATH}. "
            "Run `python scripts/build_multimodal_index.py` first to build it."
        )

    print(f"Loading local FAISS index from {INDEX_PATH}...")
    faiss_index = faiss.read_index(str(INDEX_PATH))
    item_ids = np.load(ITEM_IDS_PATH)
    vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    print(f"Loaded {len(item_ids)} vectors of dimension {faiss_index.d}.")

    print(f"Creating Qdrant collection '{QDRANT_IMAGE_COLLECTION_NAME}'...")
    qdrant_index = QdrantIndex(QDRANT_IMAGE_COLLECTION_NAME)
    qdrant_index.create_collection(dim=faiss_index.d)

    print(f"Upserting {len(item_ids)} vectors in batches of {UPSERT_BATCH_SIZE}...")
    for start in range(0, len(item_ids), UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, len(item_ids))
        qdrant_index.upsert(vectors[start:end], item_ids[start:end])
        print(f"  upserted {end}/{len(item_ids)}")

    print(f"Done. Collection '{QDRANT_IMAGE_COLLECTION_NAME}' now holds {len(item_ids)} vectors.")


if __name__ == "__main__":
    main()
