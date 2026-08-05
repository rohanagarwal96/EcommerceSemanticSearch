"""Category-stratified sampling utilities for the multimodal dataset."""

import pandas as pd


def stratified_sample(df: pd.DataFrame, category_col: str, n: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)

    fraction = n / len(df)
    # Loop + concat rather than groupby().apply(): in the installed pandas
    # version, .apply() with a same-shape-returning function silently drops
    # the grouping column from the result.
    sampled_groups = []
    for _, group in df.groupby(category_col, sort=False):
        sampled_groups.append(group.sample(frac=fraction, random_state=42))

    sampled = pd.concat(sampled_groups, ignore_index=True)
    return sampled
