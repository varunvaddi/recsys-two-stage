import numpy as np
import pandas as pd
from scipy import sparse

from recsys import cooc


def _x():
    # item 0 = anchor. item 1 is a blockbuster (10 owners), item 2 is niche (2 owners).
    # items 3 and 4 are unused (padding material).
    rows, cols = [], []
    for u in range(5):
        rows.append(u), cols.append(0)
    for u in [0, 1, 2] + list(range(5, 12)):
        rows.append(u), cols.append(1)
    for u in [3, 4]:
        rows.append(u), cols.append(2)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(12, 5))


def test_alpha_decides_blockbuster_vs_niche():
    X = _x()
    raw = cooc.BoughtTogether(alpha=0.0).fit(X)
    lift = cooc.BoughtTogether(alpha=1.0).fit(X)
    assert raw.related(0, 2).tolist() == [1, 2]   # 3 shared owners beats 2
    assert lift.related(0, 2).tolist() == [2, 1]  # 2/2 beats 3/10


def test_min_count_drops_one_off_pairs_and_self():
    X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32))
    m = cooc.BoughtTogether(alpha=0.0, min_count=2).fit(X)
    assert m.related(0, 5).tolist() == []  # every pair co-occurs once
    m1 = cooc.BoughtTogether(alpha=0.0, min_count=1).fit(X)
    assert 0 not in m1.related(0, 5).tolist()


def test_recommend_for_skips_seen_and_pads_with_fallback():
    X = _x()
    m = cooc.BoughtTogether(alpha=0.0).fit(X)
    rec, n_real = m.recommend_for(np.array([0, 0]), 2, seen=X, users=np.array([3, 11]),
                                  fallback=[2, 3, 4])
    # user 3 owns items 0 and 2: one real neighbour left (item 1), then padding with item 3
    assert rec[0].tolist() == [1, 3] and n_real[0] == 1
    # user 11 owns item 1: real neighbour is item 2, padding gives item 3
    assert rec[1].tolist() == [2, 3] and n_real[1] == 1


def test_last_item_is_most_recent_per_user():
    train = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [5, 6, 7], "ts": [1, 2, 3]})
    assert cooc.last_item(train).to_dict() == {0: 6, 1: 7}