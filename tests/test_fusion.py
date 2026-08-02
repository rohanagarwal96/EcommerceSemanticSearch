from ecomsearch.fusion import reciprocal_rank_fusion


def test_item_ranked_first_in_both_lists_scores_highest():
    dense = [1, 2, 3]
    bm25 = [1, 3, 2]

    fused = reciprocal_rank_fusion([dense, bm25])

    assert fused[0][0] == 1


def test_item_only_in_one_list_still_included():
    dense = [1, 2]
    bm25 = [3, 4]

    fused = reciprocal_rank_fusion([dense, bm25])

    fused_ids = [item_id for item_id, _ in fused]
    assert set(fused_ids) == {1, 2, 3, 4}


def test_fusion_score_matches_rrf_formula():
    dense = [1, 2]
    bm25 = [2, 1]

    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    scores = dict(fused)

    expected_item_1 = 1 / (60 + 1) + 1 / (60 + 2)
    expected_item_2 = 1 / (60 + 2) + 1 / (60 + 1)

    assert scores[1] == expected_item_1
    assert scores[2] == expected_item_2
