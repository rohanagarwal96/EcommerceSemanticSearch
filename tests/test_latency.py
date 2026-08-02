import pytest

from ecomsearch.latency import percentile


def test_percentile_median_of_five_values():
    assert percentile([5, 1, 4, 2, 3], 50) == pytest.approx(3.0)


def test_percentile_ninety_interpolates_between_values():
    assert percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)


def test_percentile_single_value_returns_that_value():
    assert percentile([42.0], 95) == pytest.approx(42.0)


def test_percentile_zero_returns_minimum():
    assert percentile([3, 1, 2], 0) == pytest.approx(1.0)


def test_percentile_hundred_returns_maximum():
    assert percentile([3, 1, 2], 100) == pytest.approx(3.0)


def test_percentile_raises_on_empty_list():
    with pytest.raises(ValueError):
        percentile([], 50)
