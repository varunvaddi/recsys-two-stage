import lightgbm as lgb
import numpy as np
import pandas as pd

KEYS = ("user_id", "item_id", "label")


def _days_since(ts_by_id, size, t_end):
    out = np.full(size, np.nan, dtype=np.float32)
    out[ts_by_id.index.values] = (t_end - ts_by_id.values) / 86400
    return out


def _content_sim(u, i, X, emb, chunk=200_000):
    """cosine(user profile, item embedding); profile = sum of the user's train items."""
    uu, inv = np.unique(u, return_inverse=True)
    prof = np.asarray(X[uu] @ emb)
    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    prof = (prof / norm).astype(np.float32)
    out = np.empty(len(u), dtype=np.float32)
    for s in range(0, len(u), chunk):
        sl = slice(s, s + chunk)
        out[sl] = np.einsum("ij,ij->i", prof[inv[sl]], emb[i[sl]])
    return out


def build_features(cands, X, X_recent, emb, train):
    """cands: user_id, item_id, rank_<source>... ; everything below uses TRAIN data only."""
    f = cands.copy()
    rank_cols = [c for c in f.columns if c.startswith("rank_")]
    f["n_sources"] = f[rank_cols].notna().sum(axis=1)
    f["best_rank"] = f[rank_cols].min(axis=1)

    u, i = f["user_id"].values, f["item_id"].values
    t_end = train["ts"].max()
    pop = np.asarray(X.sum(axis=0)).ravel()
    pop_recent = np.asarray(X_recent.sum(axis=0)).ravel()
    n_hist = np.asarray(X.sum(axis=1)).ravel()
    n_recent = np.asarray(X_recent.sum(axis=1)).ravel()
    avg_pop = np.asarray(X @ np.log1p(pop)).ravel() / np.maximum(n_hist, 1)

    f["item_pop"] = pop[i]
    f["item_pop_recent"] = pop_recent[i]
    f["item_age_days"] = _days_since(train.groupby("item_id")["ts"].min(), X.shape[1], t_end)[i]
    f["user_n_hist"] = n_hist[u]
    f["user_n_recent"] = n_recent[u]
    f["user_last_days"] = _days_since(train.groupby("user_id")["ts"].max(), X.shape[0], t_end)[u]
    f["user_avg_log_pop"] = avg_pop[u]
    f["pop_gap"] = np.log1p(f["item_pop"]) - f["user_avg_log_pop"]
    f["content_sim"] = _content_sim(u, i, X, emb)
    return f


def add_labels(f, truth):
    pos = pd.MultiIndex.from_tuples([(u, i) for u, s in truth.items() for i in s])
    idx = pd.MultiIndex.from_frame(f[["user_id", "item_id"]])
    out = f.copy()
    out["label"] = idx.isin(pos).astype(int)
    return out


def feature_cols(f):
    return [c for c in f.columns if c not in KEYS]


def split_users(users, test_frac=0.3, seed=0):
    users = np.random.default_rng(seed).permutation(np.asarray(users))
    n = int(len(users) * test_frac)
    return np.sort(users[n:]), np.sort(users[:n])


def _groups(f):
    f = f.sort_values("user_id", kind="stable")
    f = f[f.groupby("user_id")["label"].transform("max") > 0]  # all-zero users carry no signal
    return f, f.groupby("user_id", sort=False).size().values


def fit_ranker(f, features, es_frac=0.15, seed=0, **params):
    """LambdaRank on candidate rows. A slice of users is held out for early stopping."""
    users = f["user_id"].unique()
    tr_u, es_u = split_users(users, es_frac, seed)
    tr, g_tr = _groups(f[f["user_id"].isin(tr_u)])
    es, g_es = _groups(f[f["user_id"].isin(es_u)])
    kw = {
        "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "random_state": seed, "verbose": -1,
    }
    kw.update(params)
    model = lgb.LGBMRanker(objective="lambdarank", **kw)
    model.fit(tr[features], tr["label"], group=g_tr,
              eval_set=[(es[features], es["label"])], eval_group=[g_es], eval_at=[10],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def top_k(f, scores, users, k=10):
    """Highest-scoring k item ids per user, rows in the order of `users`."""
    g = f[["user_id", "item_id"]].assign(score=np.asarray(scores, dtype=float))
    g = g.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")
    lists = g.groupby("user_id")["item_id"].apply(lambda s: s.values[:k])
    rec = np.vstack([lists[u] for u in users])
    assert rec.shape[1] == k, "some user has fewer than k candidates"
    return rec