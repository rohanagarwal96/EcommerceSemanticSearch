"""Retrieval orchestration: dense, keyword (BM25), and hybrid (RRF + rerank) search."""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ecomsearch.bm25 import BM25Index
from ecomsearch.config import (
    BM25_INDEX_PATH,
    CANDIDATE_POOL_SIZE,
    CATALOG_PATH,
    INDEX_PATH,
    ITEM_IDS_PATH,
    QDRANT_COLLECTION_NAME,
    RERANK_POOL_SIZE,
    VECTOR_BACKEND,
)
from ecomsearch.embeddings import Embedder
from ecomsearch.fusion import reciprocal_rank_fusion
from ecomsearch.index import ProductIndex
from ecomsearch.qdrant_index import QdrantIndex
from ecomsearch.reranker import CrossEncoderReranker

_dense_index = None
_bm25_index = None
_embedder = None
_reranker = None
_catalog = None
_search_executor = None


def load_dense_index():
    if VECTOR_BACKEND == "qdrant":
        return QdrantIndex(QDRANT_COLLECTION_NAME)
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No dense index found at {INDEX_PATH}. "
            "Run `python scripts/build_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def load_bm25_index() -> BM25Index:
    if not BM25_INDEX_PATH.exists():
        raise SystemExit(
            f"No BM25 index found at {BM25_INDEX_PATH}. "
            "Run `python scripts/build_bm25_index.py` first to build it."
        )
    return BM25Index.load(BM25_INDEX_PATH)


def _get_dense_index() -> ProductIndex:
    global _dense_index
    if _dense_index is None:
        _dense_index = load_dense_index()
    return _dense_index


def _get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = load_bm25_index()
    return _bm25_index


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_search_executor() -> ThreadPoolExecutor:
    global _search_executor
    if _search_executor is None:
        _search_executor = ThreadPoolExecutor(max_workers=2)
    return _search_executor


def _get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        _catalog = pd.read_csv(
            CATALOG_PATH, usecols=["item_id", "search_text"]
        ).set_index("item_id")
    return _catalog


def dense_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = _get_dense_index()
    embedder = _get_embedder()
    query_vector = embedder.embed_query(query)
    return index.search(query_vector, top_k)


def bm25_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = _get_bm25_index()
    return index.search(query, top_k)


def hybrid_search(query: str, top_k: int, use_rerank: bool = True) -> list[tuple[int, float]]:
    executor = _get_search_executor()
    dense_future = executor.submit(dense_search, query, CANDIDATE_POOL_SIZE)
    bm25_future = executor.submit(bm25_search, query, CANDIDATE_POOL_SIZE)
    dense_results = dense_future.result()
    bm25_results = bm25_future.result()

    dense_ids = [item_id for item_id, _ in dense_results]
    bm25_ids = [item_id for item_id, _ in bm25_results]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])

    if not use_rerank:
        return fused[:top_k]

    candidate_ids = [item_id for item_id, _ in fused[:RERANK_POOL_SIZE]]
    catalog = _get_catalog()
    candidates = [(item_id, catalog.loc[item_id, "search_text"]) for item_id in candidate_ids]

    reranker = _get_reranker()
    reranked = reranker.rerank(query, candidates)
    return reranked[:top_k]
