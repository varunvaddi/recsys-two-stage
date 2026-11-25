import numpy as np
import pandas as pd

from recsys import analysis, ranker


def test_drop_source_removes_only_its_exclusive_rows():
    f = pd.DataFrame({"user_id": [0, 0, 0], "item_id": [1, 2, 3],
                      "rank_a": [1.0, np.nan, 2.0], "rank_b": [np.nan, 1.0, 1.0],
                      "n_sources": [1, 1, 2], "best_rank": [1.0, 1.0, 1.0]})
    g = analysis.drop_source(f, "a")
    assert g["item_id"].tolist() == [2, 3]
    assert "rank_a" not in g
    assert g["n_sources"].tolist() == [1, 1]
    assert g["best_rank"].tolist() == [1.0, 1.0]


def test_popularity_shares_sum_to_one_and_detect_head_bias():
    pop = np.arange(100)  # item 99 is the most popular
    out = analysis.popularity_shares({"head": np.array([[95, 96, 97, 98, 99]]),
                                      "tail": np.array([[0, 1, 2, 3, 4]])}, pop)
    assert np.allclose(out.sum(), 1.0)
    assert out.loc["Q5 most popular", "head"] == 1.0
    assert out.loc["Q1 least popular", "tail"] == 1.0
    recs = {"head": np.array([[99, 98]]), "tail": np.array([[0, 1]])}
    hs = analysis.head_share(recs, pop, frac=0.02)
    assert hs["head"] == 1.0 and hs["tail"] == 0.0


def test_shap_contributions_add_up_to_the_model_score():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"user_id": np.repeat(np.arange(200), 10),
                      "item_id": np.tile(np.arange(10), 200),
                      "a": rng.random(2000), "b": rng.random(2000)})
    f["label"] = (f["a"] > 0.7).astype(int)
    model = ranker.fit_ranker(f, ["a", "b"], n_estimators=20, min_child_samples=5)
    X = f[["a", "b"]]
    sv, bias = analysis.shap_values(model.booster_, X)
    raw = model.booster_.predict(X)
    assert np.allclose(sv.sum(axis=1) + bias, raw, atol=1e-5)
    d = analysis.direction(X, sv)
    assert d["a"] > 0.5  # higher a -> higher score