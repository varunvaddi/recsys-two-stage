import numpy as np
import pandas as pd
from scipy import sparse

from recsys import baselines, candidates


def test_top_n_table_has_one_row_per_user_rank():
    X = sparse.csr_matrix(np.eye(4, 6, dtype=np.float32))
    model = baselines.Popularity().fit(X)
    t = candidates.top_n_table(model, "pop", np.array([0, 1]), 3, X)
    assert len(t) == 6
    assert t["rank_pop"].tolist() == [1, 2, 3, 1, 2, 3]
    assert t["user_id"].tolist() == [0, 0, 0, 1, 1, 1]


def test_merge_keeps_items_from_either_source():
    a = pd.DataFrame({"user_id": [0, 0], "item_id": [1, 2], "rank_a": [1, 2]})
    b = pd.DataFrame({"user_id": [0, 0], "item_id": [2, 3], "rank_b": [1, 2]})
    m = candidates.merge_candidates([a, b]).sort_values("item_id")
    assert m["item_id"].tolist() == [1, 2, 3]
    assert m["rank_a"].isna().tolist() == [False, False, True]
    assert m["rank_b"].isna().tolist() == [True, False, False]


def test_candidate_recall_union_beats_single_source():
    a = pd.DataFrame({"user_id": [0], "item_id": [1], "rank_a": [1]})
    b = pd.DataFrame({"user_id": [0], "item_id": [3], "rank_b": [1]})
    m = candidates.merge_candidates([a, b])
    truth = {0: {1, 3, 9, 10}}
    assert candidates.candidate_recall(m, truth, [0], "rank_a") == 0.25
    assert candidates.candidate_recall(m, truth, [0], "rank_b") == 0.25
    assert candidates.candidate_recall(m, truth, [0]) == 0.5