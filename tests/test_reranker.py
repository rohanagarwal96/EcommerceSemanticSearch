def test_reranker_ranks_relevant_text_first(cross_encoder):
    candidates = [
        (1, "Organic almond milk, unsweetened, dairy-free beverage"),
        (2, "Wireless bluetooth headphones with noise cancelling"),
    ]

    results = cross_encoder.rerank("almond milk", candidates)

    assert results[0][0] == 1
