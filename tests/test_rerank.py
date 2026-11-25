import numpy as np
import pandas as pd

from recsys import rerank


def _frame():
    # one user, 6 candidates; items 4 and 5 never appeared in training
    return pd.DataFrame({"user_id": [0] * 6, "item_id": [0, 1, 2, 3, 4, 5],
                         "item_pop": [900, 500, 100, 50, 0, 0]})


def test_lam_zero_keeps_order_and_large_lam_flips_it():
    t = _frame()
    s = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert np.argmax(rerank.adjusted_scores(t, s, 0.0)) == 0
    assert np.argmax(rerank.adjusted_scores(t, s, 10.0)) in (4, 5)


def test_new_slots_reserve_best_new_items():
    t = _frame()
    s = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    kept = rerank.select_with_new_slots(t, s, k=3, n_new=1)
    assert sorted(kept["item_id"]) == [0, 1, 4]  # top-2 overall + best new item (4)


def test_no_new_slots_is_plain_top_k():
    t = _frame()
    s = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    kept = rerank.select_with_new_slots(t, s, k=3, n_new=0)
    assert sorted(kept["item_id"]) == [0, 1, 2]


def test_user_without_new_items_still_gets_k_rows():
    t = pd.DataFrame({"user_id": [0] * 4 + [1] * 4, "item_id": [0, 1, 2, 3] * 2,
                      "item_pop": [9, 8, 7, 6, 9, 8, 0, 6]})
    s = np.arange(8, 0, -1, dtype=float)
    kept = rerank.select_with_new_slots(t, s, k=3, n_new=1)
    assert kept.groupby("user_id").size().tolist() == [3, 3]
    assert 2 in kept[kept["user_id"] == 1]["item_id"].tolist()  # user 1's only new item
    rec = rerank.rerank(t, s, np.array([0, 1]), k=3, lam=0.0, n_new=1)
    assert rec.shape == (2, 3)