import numpy as np
import pandas as pd


def make_synthetic(n_users=2000, n_items=600, n_clusters=8, days=120, seed=0):
    rng = np.random.default_rng(seed)
    user_cluster = rng.integers(0, n_clusters, n_users)
    item_cluster = rng.integers(0, n_clusters, n_items)
    pop = 1.0 / np.arange(1, n_items + 1) ** 0.8
    pop = pop[rng.permutation(n_items)]

    rows = []
    t0 = 1_600_000_000
    for u in range(n_users):
        w = pop * np.where(item_cluster == user_cluster[u], 1.0, 0.05)
        w = w / w.sum()
        n = int(min(n_items // 2, 15 + rng.poisson(25)))
        items = rng.choice(n_items, size=n, replace=False, p=w)
        ts = t0 + rng.integers(0, days * 86400, n)
        rows.extend((u, int(i), int(t)) for i, t in zip(items, ts))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])


def time_split(df, val_frac=0.1, test_frac=0.1):
    t_val = df["ts"].quantile(1 - val_frac - test_frac)
    t_test = df["ts"].quantile(1 - test_frac)
    train = df[df["ts"] < t_val]
    val = df[(df["ts"] >= t_val) & (df["ts"] < t_test)]
    test = df[df["ts"] >= t_test]
    return train, val, test