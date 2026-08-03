"""Pydantic request/response models for the FastAPI app."""
from pydantic import BaseModel


class TextSearchResult(BaseModel):
    item_id: int
    name: str
    brand: str
    category_path: str
    score: float


class TextSearchResponse(BaseModel):
    query: str
    mode: str
    results: list[TextSearchResult]


class ImageSearchResult(BaseModel):
    item_id: int
    display_name: str
    category: str
    score: float
    image_url: str


class ImageSearchResponse(BaseModel):
    query: str
    results: list[ImageSearchResult]
