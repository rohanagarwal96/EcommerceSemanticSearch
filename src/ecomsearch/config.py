"""Shared configuration constants for the ecomsearch package."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = REPO_ROOT / "data" / "ecommerce_catalog_enriched.csv"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "catalog.faiss"
ITEM_IDS_PATH = ARTIFACTS_DIR / "item_ids.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_TOP_K = 10

BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25.pkl"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
CANDIDATE_POOL_SIZE = 100
RERANK_POOL_SIZE = 50

EVAL_QUERIES_PATH = REPO_ROOT / "eval" / "eval_queries.json"
EVAL_RESULTS_PATH = REPO_ROOT / "docs" / "eval_results.md"
EVAL_TOP_K = 10

LATENCY_RESULTS_PATH = REPO_ROOT / "docs" / "latency_results.md"
LATENCY_TARGET_MS_P95 = 200.0
BENCHMARK_REPEAT_COUNT = 10
BENCHMARK_SEED = 42

VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "faiss")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "ecommerce_products")

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
HF_SPACE_BACKEND = os.environ.get("HF_SPACE_BACKEND")
HF_SPACE_FRONTEND = os.environ.get("HF_SPACE_FRONTEND")
