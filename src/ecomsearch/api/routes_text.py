"""FastAPI routes for text (catalog) search."""
from typing import Literal

import pandas as pd
from fastapi import APIRouter

from ecomsearch.api.schemas import TextSearchResponse, TextSearchResult
from ecomsearch.config import CATALOG_PATH, DEFAULT_TOP_K
from ecomsearch.search import bm25_search, dense_search, hybrid_search

router = APIRouter()

_catalog = None

MODES = {
    "dense": lambda query, top_k: dense_search(query, top_k),
    "bm25": lambda query, top_k: bm25_search(query, top_k),
    "hybrid": lambda query, top_k: hybrid_search(query, top_k, use_rerank=False),
    "hybrid-rerank": lambda query, top_k: hybrid_search(query, top_k, use_rerank=True),
}


def _get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        _catalog = pd.read_csv(
            CATALOG_PATH, usecols=["item_id", "name", "brand", "category_path"]
        ).set_index("item_id")
    return _catalog


@router.get("/search/text", response_model=TextSearchResponse)
def search_text(
    q: str,
    mode: Literal["dense", "bm25", "hybrid", "hybrid-rerank"] = "hybrid",
    top_k: int = DEFAULT_TOP_K,
) -> TextSearchResponse:
    search_fn = MODES[mode]
    results = search_fn(q, top_k)
    catalog = _get_catalog()

    items = []
    for item_id, score in results:
        row = catalog.loc[item_id]
        items.append(
            TextSearchResult(
                item_id=item_id,
                name=str(row["name"]),
                brand=str(row["brand"]),
                category_path=str(row["category_path"]),
                score=score,
            )
        )

    return TextSearchResponse(query=q, mode=mode, results=items)
