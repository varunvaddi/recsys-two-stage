import numpy as np
import pytest

from recsys import metrics


def test_known_values():
    recs, truth = [5, 1, 9, 2], {1, 2}
    assert metrics.recall_at_k(recs, truth, 2) == 0.5
    assert metrics.ndcg_at_k(recs, truth, 2) == pytest.approx(
        (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert metrics.ap_at_k(recs, truth, 4) == pytest.approx(0.5)
    assert metrics.catalog_coverage([[1, 2], [2, 3]], n_items=10, k=2) == 0.3
    assert metrics.recall_at_k([1], set(), 1) == 0.0


def test_perfect_ranking_scores_one():
    assert metrics.ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert metrics.ap_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)