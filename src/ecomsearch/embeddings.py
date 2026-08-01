"""Text embedding utilities wrapping BAAI/bge-base-en-v1.5."""
from sentence_transformers import SentenceTransformer
import numpy as np

from ecomsearch.config import MAX_TOKENS, MODEL_NAME, QUERY_PREFIX


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = MAX_TOKENS

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")
