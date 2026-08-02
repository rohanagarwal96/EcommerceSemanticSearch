"""Evaluation metrics: Recall@k, NDCG@k, and MRR over ranked item_id lists."""
import math


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0
