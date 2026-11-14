import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import evaluate, hit_rate_at_k, ndcg_at_k, recall_at_k  # noqa: E402


def test_recall_counts_hits_over_relevant():
    assert recall_at_k([1, 2, 3], {2, 9}, k=3) == 0.5


def test_recall_respects_k():
    assert recall_at_k([1, 2, 3], {3}, k=2) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_hit_rate():
    assert hit_rate_at_k([1, 2], {2}, k=2) == 1.0
    assert hit_rate_at_k([1, 2], {5}, k=2) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([7, 8, 1], {7, 8}, k=3) == pytest.approx(1.0)


def test_ndcg_rewards_higher_position():
    assert ndcg_at_k([9, 7], {7}, k=2) == pytest.approx(1 / math.log2(3))


def test_evaluate_averages_over_users():
    recs = {"a": [1, 2], "b": [3, 4]}
    truth = {"a": {1}, "b": {9}}
    result = evaluate(recs, truth, k=2)
    assert result["recall@2"] == 0.5
    assert result["hit_rate@2"] == 0.5
    assert result["n_users"] == 2


def test_evaluate_missing_user_scores_zero():
    result = evaluate({}, {"a": {1}}, k=10)
    assert result["recall@10"] == 0.0
