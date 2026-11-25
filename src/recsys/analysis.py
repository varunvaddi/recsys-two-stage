import numpy as np
import pandas as pd

from recsys import metrics, ranker

QUINTILES = ["Q1 least popular", "Q2", "Q3", "Q4", "Q5 most popular"]


def drop_source(f, name):
    """Remove one candidate source: its rank column AND the rows only it suggested."""
    g = f.drop(columns=f"rank_{name}")
    ranks = [c for c in g.columns if c.startswith("rank_")]
    g = g[g[ranks].notna().any(axis=1)].copy()
    g["n_sources"] = g[ranks].notna().sum(axis=1)
    g["best_rank"] = g[ranks].min(axis=1)
    return g


def feature_groups(cols):
    return {
        "source ranks": [c for c in cols if c.startswith("rank_")] + ["n_sources", "best_rank"],
        "item popularity & age": ["item_pop", "item_pop_recent", "item_age_days", "pop_gap"],
        "user activity": ["user_n_hist", "user_n_recent", "user_last_days", "user_avg_log_pop"],
        "content similarity": ["content_sim"],
    }


def fit_eval(f, cols, train_users, test_users, truth, n_items, k=10):
    model = ranker.fit_ranker(f[f["user_id"].isin(train_users)], cols)
    test = f[f["user_id"].isin(test_users)].reset_index(drop=True)
    rec = ranker.top_k(test, model.predict(test[cols]), test_users, k)
    r = metrics.evaluate(rec, test_users, truth, n_items, k)
    return {f"recall@{k}": r[f"recall@{k}"], f"ndcg@{k}": r[f"ndcg@{k}"],
            f"coverage@{k}": r[f"coverage@{k}"]}


def shap_values(booster, X):
    """Exact TreeSHAP from LightGBM. Row sums (+ bias) equal the raw model score."""
    c = booster.predict(X, pred_contrib=True)
    return c[:, :-1], c[:, -1]


def direction(X, shap):
    """Spearman(feature value, its SHAP): + means higher values push the score up."""
    return pd.Series({c: X[c].corr(pd.Series(shap[:, j], index=X.index), method="spearman")
                      for j, c in enumerate(X.columns)})


def popularity_shares(recs, pop):
    """Share of each list's items that fall in each popularity quintile (train popularity)."""
    q = pd.qcut(pd.Series(pop).rank(method="first"), 5, labels=False).values
    return pd.DataFrame({name: np.bincount(q[np.asarray(r).ravel()], minlength=5) / np.size(r)
                         for name, r in recs.items()}, index=QUINTILES)


def head_share(recs, pop, frac=0.01):
    head = np.zeros(len(pop), dtype=bool)
    head[np.argsort(-pop)[: max(1, int(len(pop) * frac))]] = True
    return pd.Series({name: float(head[np.asarray(r).ravel()].mean()) for name, r in recs.items()})