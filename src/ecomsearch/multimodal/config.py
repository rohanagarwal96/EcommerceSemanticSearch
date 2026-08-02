"""Shared configuration constants for the multimodal (CLIP) module."""
from pathlib import Path

from ecomsearch.config import REPO_ROOT

DATASET_DIR = REPO_ROOT / "data" / "multimodal"
DATASET_CSV_PATH = DATASET_DIR / "data.csv"
DATASET_IMAGES_DIR = DATASET_DIR / "data"

KAGGLE_DATASET_REF = "nirmalsankalana/mini-product-image-and-text-dataset"
KAGGLE_CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "multimodal"
INDEX_PATH = ARTIFACTS_DIR / "catalog.faiss"
ITEM_IDS_PATH = ARTIFACTS_DIR / "item_ids.npy"
SUBSET_METADATA_PATH = ARTIFACTS_DIR / "subset_metadata.csv"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SUBSET_SIZE = 5000
DEFAULT_TOP_K = 10

DEMO_RESULTS_DIR = REPO_ROOT / "demo_results"
