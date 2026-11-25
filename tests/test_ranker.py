import numpy as np
import pandas as pd
from scipy import sparse

from recsys import ranker


def _tiny():
    train = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [0, 1, 1],
                          "ts": [86400, 2 * 86400, 3 * 86400]})
    X = sparse.csr_matrix(([1.0, 1.0, 1.0], ([0, 0, 1], [0, 1, 1])), shape=(2, 4))
    emb = np.eye(4, dtype=np.float32)
    cands = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [1, 2, 3],
                          "rank_a": [1.0, np.nan, 2.0], "rank_b": [np.nan, 1.0, 1.0]})
    return cands, X, emb, train


def test_features_are_computed_from_train_only():
    cands, X, emb, train = _tiny()
    f = ranker.build_features(cands, X, X, emb, train)
    assert f["n_sources"].tolist() == [1, 1, 2]
    assert f["best_rank"].tolist() == [1, 1, 1]
    assert f["item_pop"].tolist() == [2, 0, 0]
    assert f["user_n_hist"].tolist() == [2, 2, 1]
    assert f["item_age_days"].iloc[0] == 1  # item 1 first seen day 2, train ends day 3
    assert np.isnan(f["item_age_days"].iloc[1])  # item never in train
    assert f["content_sim"].between(-1, 1).all()


def test_labels_mark_only_true_pairs():
    cands, *_ = _tiny()
    out = ranker.add_labels(cands, {0: {2}, 1: {9}})
    assert out["label"].tolist() == [0, 1, 0]


def test_top_k_orders_by_score_per_user():
    f = pd.DataFrame({"user_id": [0, 0, 0, 1, 1, 1], "item_id": [5, 6, 7, 5, 6, 7]})
    rec = ranker.top_k(f, [0.1, 0.9, 0.5, 0.3, 0.2, 0.8], np.array([0, 1]), k=2)
    assert rec.tolist() == [[6, 7], [7, 5]]


def test_ranker_learns_a_planted_signal():
    rng = np.random.default_rng(0)
    n_u, n_c = 300, 20
    f = pd.DataFrame({"user_id": np.repeat(np.arange(n_u), n_c),
                      "item_id": np.tile(np.arange(n_c), n_u),
                      "signal": rng.random(n_u * n_c),
                      "noise": rng.random(n_u * n_c)})
    f["label"] = (f["signal"] > 0.7).astype(int)
    model = ranker.fit_ranker(f, ["signal", "noise"], n_estimators=40, min_child_samples=5)
    assert model.booster_.feature_importance("gain")[0] > model.booster_.feature_importance("gain")[1]
    top = ranker.top_k(f, model.predict(f[["signal", "noise"]]), np.arange(n_u), k=1)
    hit = np.mean([f[(f.user_id == u) & (f.item_id == top[u, 0])]["label"].iloc[0] for u in range(n_u)])
    assert hit > 0.9