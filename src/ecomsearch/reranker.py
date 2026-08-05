"""Cross-encoder reranking for search result candidates."""

from sentence_transformers import CrossEncoder

from ecomsearch.config import RERANKER_MODEL_NAME


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        item_ids = [item_id for item_id, _ in candidates]
        ranked = sorted(zip(item_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(int(item_id), float(score)) for item_id, score in ranked]
