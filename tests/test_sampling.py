import pandas as pd

from ecomsearch.multimodal.sampling import stratified_sample


def test_stratified_sample_preserves_category_proportions():
    df = pd.DataFrame({
        "category": ["a"] * 80 + ["b"] * 20,
        "value": range(100),
    })

    sampled = stratified_sample(df, "category", 10)

    counts = sampled["category"].value_counts()
    assert len(sampled) == 10
    assert counts.get("a", 0) == 8
    assert counts.get("b", 0) == 2


def test_stratified_sample_returns_full_df_when_n_exceeds_length():
    df = pd.DataFrame({"category": ["a", "b", "c"], "value": [1, 2, 3]})

    sampled = stratified_sample(df, "category", 10)

    assert len(sampled) == 3
