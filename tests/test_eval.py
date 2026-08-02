import math

import pytest

from ecomsearch.eval import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_full_recall():
    retrieved = [10, 20, 30]
    relevant = {10, 30}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_partial_recall():
    retrieved = [10, 20, 30]
    relevant = {10, 30, 40}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_respects_k_cutoff():
    retrieved = [10, 20, 30, 40]
    relevant = {40}
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)


def test_ndcg_at_k_matches_formula():
    retrieved = [10, 20, 30]
    relevant = {10, 30}
    k = 3

    dcg = 1 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg

    assert ndcg_at_k(retrieved, relevant, k) == pytest.approx(expected)


def test_ndcg_at_k_perfect_ranking_scores_one():
    retrieved = [10, 30, 20]
    relevant = {10, 30}
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_items_scores_zero():
    retrieved = [10, 20, 30]
    relevant = set()
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_mrr_first_relevant_at_rank_two():
    retrieved = [20, 10, 30]
    relevant = {10}
    assert mrr(retrieved, relevant) == pytest.approx(0.5)


def test_mrr_first_relevant_at_rank_one():
    retrieved = [10, 20]
    relevant = {10, 30}
    assert mrr(retrieved, relevant) == pytest.approx(1.0)


def test_mrr_no_relevant_found():
    retrieved = [20, 30]
    relevant = {10}
    assert mrr(retrieved, relevant) == pytest.approx(0.0)
