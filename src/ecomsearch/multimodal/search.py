"""Image search orchestration: cached CLIP-based text-to-image search."""
from ecomsearch.index import ProductIndex
from ecomsearch.multimodal.clip_embedder import ClipEmbedder
from ecomsearch.multimodal.config import INDEX_PATH, ITEM_IDS_PATH

_index = None
_embedder = None


def load_index() -> ProductIndex:
    if not INDEX_PATH.exists() or not ITEM_IDS_PATH.exists():
        raise SystemExit(
            f"No multimodal index found at {INDEX_PATH}. "
            "Run `python scripts/build_multimodal_index.py` first to build it."
        )
    return ProductIndex.load(INDEX_PATH, ITEM_IDS_PATH)


def _get_index() -> ProductIndex:
    global _index
    if _index is None:
        _index = load_index()
    return _index


def _get_embedder() -> ClipEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ClipEmbedder()
    return _embedder


def image_search(query: str, top_k: int) -> list[tuple[int, float]]:
    index = _get_index()
    embedder = _get_embedder()
    query_vector = embedder.embed_text([query])[0]
    return index.search(query_vector, top_k)
