import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import optimizer  # noqa: E402


@pytest.mark.parametrize(
    "trait_counts,total_cost,high_cost,expected",
    [
        ([0] * optimizer.T, 0, 0, 0.0),
        ([2] + [0] * (optimizer.T - 1), 10, 1, 0.15 * 10 + 0.4 * 1),
    ],
)
def test_compute_score(trait_counts, total_cost, high_cost, expected):
    assert optimizer.compute_score(
        trait_counts, total_cost, high_cost
    ) == pytest.approx(expected)


def test_enumerate_half_basic():
    optimizer.TEAM_SIZE = 2
    subset = optimizer.CHAMPIONS[:3]
    sigs = optimizer.enumerate_half(subset, "TEST")
    assert any(s.size <= 2 for s in sigs)


def test_combine_and_search_basic():
    optimizer.TEAM_SIZE = 2
    left = optimizer.CHAMPIONS[:2]
    right = optimizer.CHAMPIONS[2:4]
    left_sigs = optimizer.enumerate_half(left, "L")
    right_sigs = optimizer.enumerate_half(right, "R")
    result = optimizer.combine_and_search(left_sigs, right_sigs, 0.0)
    assert len(result) > 0
