"""FastAPI application: serving layer for text and image product search."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ecomsearch.api.routes_image import router as image_router
from ecomsearch.api.routes_text import router as text_router
from ecomsearch.multimodal.search import image_search
from ecomsearch.search import bm25_search, dense_search, hybrid_search


def _warm_up_caches() -> None:
    dense_search("warm up", top_k=1)
    bm25_search("warm up", top_k=1)
    hybrid_search("warm up", top_k=1, use_rerank=True)
    image_search("warm up", top_k=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_up_caches()
    yield


app = FastAPI(title="E-Commerce Semantic Search API", lifespan=lifespan)
app.include_router(text_router)
app.include_router(image_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
