"""FastAPI routes for multimodal (image) search."""

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ecomsearch.api.schemas import ImageSearchResponse, ImageSearchResult
from ecomsearch.multimodal.config import DATASET_IMAGES_DIR, DEFAULT_TOP_K, SUBSET_METADATA_PATH
from ecomsearch.multimodal.search import image_search

router = APIRouter()

_metadata = None


def _get_metadata() -> pd.DataFrame:
    global _metadata
    if _metadata is None:
        _metadata = pd.read_csv(SUBSET_METADATA_PATH).set_index("item_id")
    return _metadata


@router.get("/search/image", response_model=ImageSearchResponse)
def search_image(q: str, top_k: int = DEFAULT_TOP_K) -> ImageSearchResponse:
    results = image_search(q, top_k)
    metadata = _get_metadata()

    items = []
    for item_id, score in results:
        row = metadata.loc[item_id]
        items.append(
            ImageSearchResult(
                item_id=item_id,
                display_name=str(row["display name"]),
                category=str(row["category"]),
                score=score,
                image_url=f"/images/{item_id}",
            )
        )

    return ImageSearchResponse(query=q, results=items)


@router.get("/images/{item_id}")
def get_image(item_id: int) -> FileResponse:
    metadata = _get_metadata()
    if item_id not in metadata.index:
        raise HTTPException(status_code=404, detail=f"No image found for item_id {item_id}")

    image_filename = metadata.loc[item_id, "image"]
    image_path = DATASET_IMAGES_DIR / image_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file missing for item_id {item_id}")
    return FileResponse(image_path)
