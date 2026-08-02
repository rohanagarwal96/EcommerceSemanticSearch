"""Retrieval orchestration: dense, keyword (BM25), and hybrid (RRF + rerank) search."""
import pandas as pd

from ecomsearch.bm25 import BM25Index
from ecomsearch.config import (
    BM25_INDEX_PATH,
    CANDIDATE_POOL_SIZE,
    CATALOG_PATH,
    INDEX_PATH,
    ITEM_IDS_PATH,
    RERANK_POOL_SIZE,
)
from ecomsearch.embeddings import Embedder
from ecomsearch.fusion import reciprocal_rank_fusion
from ecomsearch.index import ProductIndex
from ecomsearch.reranker import CrossEncoderReranker


def load_dense_index() -> ProductIndex:
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


def dense_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = load_dense_index()
    embedder = Embedder()
    query_vector = embedder.embed_query(query)
    return index.search(query_vector, top_k)


def bm25_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = load_bm25_index()
    return index.search(query, top_k)


def hybrid_search(query: str, top_k: int, use_rerank: bool = True) -> list[tuple[int, float]]:
    dense_results = dense_search(query, CANDIDATE_POOL_SIZE)
    bm25_results = bm25_search(query, CANDIDATE_POOL_SIZE)

    dense_ids = [item_id for item_id, _ in dense_results]
    bm25_ids = [item_id for item_id, _ in bm25_results]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])

    if not use_rerank:
        return fused[:top_k]

    candidate_ids = [item_id for item_id, _ in fused[:RERANK_POOL_SIZE]]
    catalog = pd.read_csv(
        CATALOG_PATH, usecols=["item_id", "search_text"]
    ).set_index("item_id")
    candidates = [(item_id, catalog.loc[item_id, "search_text"]) for item_id in candidate_ids]

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidates)
    return reranked[:top_k]
