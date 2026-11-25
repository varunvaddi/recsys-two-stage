import numpy as np
import pandas as pd

WARN, ALERT = 0.1, 0.25  # the usual PSI rules of thumb


def psi(ref, cur, bins=10):
    """Population Stability Index of `cur` against `ref`. Bin edges come from the reference
    quantiles, so each reference bin holds ~10% of the data. 0 = same distribution."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # constant feature: nothing to bin
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def status(value):
    return "alert" if value >= ALERT else "warn" if value >= WARN else "ok"


def drift_report(ref, cur, cols):
    """One row per feature: PSI, status, and the means so a human can see which way it moved."""
    rows = []
    for c in cols:
        v = psi(ref[c], cur[c])
        rows.append({"feature": c, "psi": round(v, 4), "status": status(v),
                     "ref_mean": float(np.nanmean(ref[c])), "cur_mean": float(np.nanmean(cur[c]))})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def snapshot(inter, t_end, recent_days=365):
    """The model's input features as they looked at time t_end (history strictly before it).
    Returns (user_features, item_features), one row per user / item."""
    h = inter[inter["ts"] < t_end]
    r = h[h["ts"] >= t_end - recent_days * 86400]
    items = pd.DataFrame({
        "item_pop": h.groupby("item_id").size(),
        "item_age_days": (t_end - h.groupby("item_id")["ts"].min()) / 86400})
    items["item_pop_recent"] = r.groupby("item_id").size().reindex(items.index).fillna(0)
    log_pop = np.log1p(items["item_pop"])
    users = pd.DataFrame({
        "user_n_hist": h.groupby("user_id").size(),
        "user_last_days": (t_end - h.groupby("user_id")["ts"].max()) / 86400})
    users["user_n_recent"] = r.groupby("user_id").size().reindex(users.index).fillna(0)
    users["user_avg_log_pop"] = log_pop.reindex(h["item_id"]).groupby(h["user_id"].values).mean()
    return users, items


def quality_alert(current, baseline, max_drop=0.2):
    """True if a quality metric (e.g. recall@10 on freshly matured labels) fell more than
    `max_drop` (relative) below its baseline."""
    return bool(current < baseline * (1 - max_drop))