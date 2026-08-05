"""Reciprocal Rank Fusion for combining multiple ranked result lists."""

from ecomsearch.config import RRF_K


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
