import numpy as np
import pandas as pd

from recsys import baselines, data, ranker, serve

SOURCES = ["pop", "als", "itemcf", "content"]
NAMES = (["rank_" + s for s in SOURCES]
         + ["n_sources", "best_rank", "item_pop", "item_pop_recent", "item_age_days",
            "user_n_hist", "user_n_recent", "user_last_days", "user_avg_log_pop",
            "pop_gap", "content_sim"])


def _world(n_users=60, n_items=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(u, int(i), int(rng.integers(0, 400 * 86400)))
            for u in range(n_users) for i in rng.choice(n_items, 8, replace=False)]
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    emb = rng.normal(size=(n_items, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    als = baselines.ALS(factors=8, iterations=3).fit(X)
    art = type("A", (), {})()
    art.X, art.emb = X, emb
    art.als_user = np.asarray(als.model.user_factors, dtype=np.float32)
    art.als_item = np.asarray(als.model.item_factors, dtype=np.float32)
    art.S = baselines.ItemCF(n_neighbors=10).fit(X).S.tocsr()
    art.cooc_ids = np.tile(np.arange(5, dtype=np.int32), (n_items, 1))
    art.cooc_scores = np.tile(np.array([5, 4, 3, -np.inf, -np.inf], dtype=np.float32), (n_items, 1))
    for k, v in serve.derived_arrays(train, X, X_recent).items():
        setattr(art, k, v)
    return train, X, X_recent, emb, art


def test_time_decay_halves_every_half_life():
    train = pd.DataFrame({"user_id": [0, 0], "item_id": [0, 1], "ts": [0, 30 * 86400]})
    t = serve.time_decayed_popularity(train, 2, half_life_days=30)
    assert np.allclose(t, [0.5, 1.0])


def test_top_unseen_masks_seen_and_sorts():
    out = serve._top_unseen(np.array([5.0, 9.0, 1.0, 7.0]), np.array([1]), 2)
    assert out.tolist() == [3, 0]


def test_serving_features_match_training_features():
    """Training/serving skew guard: live features == the ones the ranker was trained on."""
    train, X, X_recent, emb, art = _world()
    rec = serve.Recommender(art, booster=None, feature_names=NAMES)
    for u in (0, 7, 33):
        items, feats = rec.candidate_features(u)
        cands = pd.DataFrame({"user_id": u, "item_id": items,
                              **{c: feats[c].values for c in NAMES if c.startswith("rank_")}})
        offline = ranker.build_features(cands, X, X_recent, emb, train)
        for c in NAMES:
            np.testing.assert_allclose(feats[c].values.astype(float),
                                       offline[c].values.astype(float), rtol=1e-4, atol=1e-5,
                                       err_msg=f"feature {c} differs for user {u}")


def test_recommend_returns_unseen_unique_items_and_handles_cold_users():
    _, X, _, _, art = _world()
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.random((600, len(NAMES))), columns=NAMES)
    f["user_id"], f["item_id"] = np.repeat(np.arange(30), 20), np.tile(np.arange(20), 30)
    f["label"] = (f["content_sim"] > 0.8).astype(int)
    booster = ranker.fit_ranker(f, NAMES, n_estimators=5, min_child_samples=1).booster_
    rec = serve.Recommender(art, booster=booster)
    out = rec.recommend(3, k=5)
    ids = [i["item_id"] for i in out["items"]]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert not set(ids) & set(X[3].indices.tolist())
    assert out["cold_start"] is False
    assert rec.recommend(10_000, k=5)["cold_start"] is True
    assert len(rec.related(2, k=3)["items"]) == 3
    assert len(rec.up_next(3, k=4)["items"]) == 4
    assert len(rec.trending(5)["items"]) == 5