"""One-time job: download the multimodal product image/text dataset from Kaggle.

Usage:
    python scripts/download_multimodal_dataset.py
"""
from ecomsearch.multimodal.config import (
    DATASET_CSV_PATH,
    DATASET_DIR,
    KAGGLE_CREDENTIALS_PATH,
    KAGGLE_DATASET_REF,
)


def main() -> None:
    if DATASET_CSV_PATH.exists():
        print(f"Dataset already present at {DATASET_CSV_PATH}, skipping download.")
        return

    if not KAGGLE_CREDENTIALS_PATH.exists():
        raise SystemExit(
            f"Kaggle credentials not found at {KAGGLE_CREDENTIALS_PATH}. "
            "Set up your Kaggle API token (https://www.kaggle.com/docs/api) "
            "before running this script."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET_REF} to {DATASET_DIR}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET_REF, path=str(DATASET_DIR), unzip=True)
    print(f"Downloaded and extracted to {DATASET_DIR}")


if __name__ == "__main__":
    main()
