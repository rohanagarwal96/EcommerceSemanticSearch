"""FastAPI application: serving layer for text and image product search."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from huggingface_hub import snapshot_download

from ecomsearch.api.routes_image import router as image_router
from ecomsearch.api.routes_text import router as text_router
from ecomsearch.config import CATALOG_PATH, HF_DATASET_REPO, HF_TOKEN, REPO_ROOT
from ecomsearch.multimodal.search import image_search
from ecomsearch.search import bm25_search, dense_search, hybrid_search


def _ensure_artifacts_present() -> None:
    if CATALOG_PATH.exists():
        return
    if not HF_DATASET_REPO:
        raise SystemExit(
            "Catalog not found locally and HF_DATASET_REPO is not set -- "
            "cannot bootstrap production artifacts."
        )
    print(f"Downloading artifacts from '{HF_DATASET_REPO}'...")
    snapshot_download(
        repo_id=HF_DATASET_REPO, repo_type="dataset", local_dir=str(REPO_ROOT), token=HF_TOKEN
    )


def _warm_up_caches() -> None:
    dense_search("warm up", top_k=1)
    bm25_search("warm up", top_k=1)
    hybrid_search("warm up", top_k=1, use_rerank=True)
    image_search("warm up", top_k=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_artifacts_present()
    _warm_up_caches()
    yield


app = FastAPI(title="E-Commerce Semantic Search API", lifespan=lifespan)
app.include_router(text_router)
app.include_router(image_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
