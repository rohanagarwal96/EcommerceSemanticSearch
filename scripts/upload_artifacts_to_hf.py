"""One-time script: upload the catalog CSV, BM25 pickle, and CLIP subset
images/metadata to a Hugging Face Hub dataset repo, so production containers can
download them at startup instead of needing gigabytes of raw data baked into the
Docker image.

Usage:
    python scripts/upload_artifacts_to_hf.py
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from ecomsearch.config import BM25_INDEX_PATH, CATALOG_PATH, HF_DATASET_REPO, HF_TOKEN, REPO_ROOT
from ecomsearch.multimodal.config import DATASET_IMAGES_DIR, SUBSET_METADATA_PATH


def _stage_artifacts(staging_dir: Path) -> None:
    catalog_dest = staging_dir / CATALOG_PATH.relative_to(REPO_ROOT)
    catalog_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CATALOG_PATH, catalog_dest)

    bm25_dest = staging_dir / BM25_INDEX_PATH.relative_to(REPO_ROOT)
    bm25_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BM25_INDEX_PATH, bm25_dest)

    metadata_dest = staging_dir / SUBSET_METADATA_PATH.relative_to(REPO_ROOT)
    metadata_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUBSET_METADATA_PATH, metadata_dest)

    subset = pd.read_csv(SUBSET_METADATA_PATH)
    images_dest_dir = staging_dir / DATASET_IMAGES_DIR.relative_to(REPO_ROOT)
    images_dest_dir.mkdir(parents=True, exist_ok=True)
    for image_name in subset["image"]:
        shutil.copy2(DATASET_IMAGES_DIR / image_name, images_dest_dir / image_name)
    print(f"Staged catalog, BM25 index, and {len(subset)} subset images.")


def main() -> None:
    if not HF_DATASET_REPO:
        raise SystemExit(
            "HF_DATASET_REPO is not set. Add it to your .env, e.g. "
            "HF_DATASET_REPO=your-hf-username/ecommerce-search-artifacts"
        )

    for path, build_hint in [
        (CATALOG_PATH, None),
        (BM25_INDEX_PATH, "python scripts/build_bm25_index.py"),
        (SUBSET_METADATA_PATH, "python scripts/build_multimodal_index.py"),
    ]:
        if not path.exists():
            hint = f" Run `{build_hint}` first." if build_hint else ""
            raise SystemExit(f"Required artifact not found at {path}.{hint}")

    api = HfApi(token=HF_TOKEN)
    print(f"Creating (or reusing) dataset repo '{HF_DATASET_REPO}'...")
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        _stage_artifacts(staging_dir)

        print(f"Uploading staged artifacts to '{HF_DATASET_REPO}'...")
        api.upload_folder(
            repo_id=HF_DATASET_REPO,
            folder_path=str(staging_dir),
            repo_type="dataset",
            commit_message="Upload catalog, BM25 index, and CLIP subset images",
        )

    print("Done.")


if __name__ == "__main__":
    main()
