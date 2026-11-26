import numpy as np

from recsys import evalstats


def test_per_user_matches_hand_computation():
    rec = np.array([[1, 2, 3], [4, 5, 6]])
    truth = {0: {2, 9}, 1: {7}}
    recall, ndcg = evalstats.per_user(rec, [0, 1], truth, k=3)
    assert recall.tolist() == [0.5, 0.0]
    assert ndcg[1] == 0.0 and 0 < ndcg[0] < 1


def test_bootstrap_interval_contains_the_mean_and_shrinks_with_more_users():
    rng = np.random.default_rng(0)
    small, large = rng.random(100), rng.random(10_000)
    m, lo, hi = evalstats.bootstrap_ci(small)
    assert lo < m < hi
    _, lo2, hi2 = evalstats.bootstrap_ci(large)
    assert (hi2 - lo2) < (hi - lo)


def test_paired_lift_detects_a_real_improvement_and_a_null_one():
    rng = np.random.default_rng(1)
    base = rng.random(5000) * (rng.random(5000) < 0.2)
    better = base + (rng.random(5000) < 0.1) * 0.5
    lift, lo, _ = evalstats.paired_lift_ci(better, base)
    assert lift > 0 and lo > 0
    lift0, lo0, hi0 = evalstats.paired_lift_ci(base, base)
    assert lift0 == 0 and lo0 == 0 and hi0 == 0