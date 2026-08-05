"""Batch job: stratified-sample the multimodal dataset, embed images with CLIP,
build the FAISS index.

Usage:
    python scripts/build_multimodal_index.py
"""

import numpy as np
import pandas as pd

from ecomsearch.index import ProductIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import (
    ARTIFACTS_DIR,
    DATASET_CSV_PATH,
    DATASET_IMAGES_DIR,
    INDEX_PATH,
    ITEM_IDS_PATH,
    SUBSET_METADATA_PATH,
    SUBSET_SIZE,
)
from ecomsearch.multimodal.sampling import stratified_sample

IMAGE_BATCH_SIZE = 64


def main() -> None:
    if not DATASET_CSV_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATASET_CSV_PATH}. "
            "Run `python scripts/download_multimodal_dataset.py` first."
        )

    print(f"Loading dataset from {DATASET_CSV_PATH}...")
    df = pd.read_csv(DATASET_CSV_PATH)

    print(f"Sampling {SUBSET_SIZE} rows stratified by category...")
    subset = stratified_sample(df, "category", SUBSET_SIZE)
    print(f"Sampled {len(subset)} rows across {subset['category'].nunique()} categories.")

    item_ids = subset["image"].apply(lambda name: int(name.split(".")[0])).to_numpy()
    image_paths = [DATASET_IMAGES_DIR / name for name in subset["image"]]

    print(f"Embedding {len(subset)} images with CLIP in batches of {IMAGE_BATCH_SIZE}...")
    embedder = ClipEmbedder()
    vector_batches = []
    for start in range(0, len(image_paths), IMAGE_BATCH_SIZE):
        batch_paths = image_paths[start : start + IMAGE_BATCH_SIZE]
        vector_batches.append(embedder.embed_images(batch_paths))
        done = min(start + IMAGE_BATCH_SIZE, len(image_paths))
        print(f"  embedded {done}/{len(image_paths)}")
    vectors = np.concatenate(vector_batches, axis=0)

    print("Building FAISS index...")
    index = ProductIndex(dim=vectors.shape[1])
    index.add(vectors, item_ids)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(INDEX_PATH, ITEM_IDS_PATH)
    subset.assign(item_id=item_ids).to_csv(SUBSET_METADATA_PATH, index=False)
    print(
        f"Saved index to {INDEX_PATH}, id mapping to {ITEM_IDS_PATH}, "
        f"metadata to {SUBSET_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()
