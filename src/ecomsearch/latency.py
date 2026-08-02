"""Pure latency-measurement helpers (percentile computation, no I/O)."""
import math


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    rank = (p / 100) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]

    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
