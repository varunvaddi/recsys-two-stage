import numpy as np

from recsys import ranker


def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def adjusted_scores(test, scores, lam):
    """Standardised ranker score minus lam * standardised log-popularity.
    lam=0 keeps the ranker's order; bigger lam pushes toward less popular games."""
    return _z(scores) - lam * _z(np.log1p(test["item_pop"].values))


def select_with_new_slots(test, scores, k, n_new):
    """Exactly k rows per user. n_new of them are reserved for the best-scored games
    that never appeared in training (item_pop == 0); the rest are the best overall."""
    g = test[["user_id", "item_id"]].copy()
    g["score"] = np.asarray(scores, dtype=float)
    g["is_new"] = test["item_pop"].values == 0
    g = g.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")
    if n_new == 0:
        return g.groupby("user_id").head(k)
    new_rank = g[g["is_new"]].groupby("user_id").cumcount()
    g["sel_new"] = False
    g.loc[new_rank.index, "sel_new"] = new_rank < n_new
    n_sel = g.groupby("user_id")["sel_new"].transform("sum")
    g["other_rank"] = g[~g["sel_new"]].groupby("user_id").cumcount()
    return g[g["sel_new"] | (g["other_rank"] < k - n_sel)]


def rerank(test, scores, users, k=10, lam=0.0, n_new=0):
    kept = select_with_new_slots(test, adjusted_scores(test, scores, lam), k, n_new)
    return ranker.top_k(kept, kept["score"], users, k)


def new_item_share(rec, pop):
    return float((np.asarray(pop)[np.asarray(rec).ravel()] == 0).mean())