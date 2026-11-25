import numpy as np
import pandas as pd
from scipy import sparse


def to_csr(df, n_users, n_items):
    m = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df["user_id"], df["item_id"])),
        shape=(n_users, n_items),
    )
    m.data[:] = 1.0
    return m


def truth_dict(eval_df, train_csr):
    out = {}
    for u, i in zip(eval_df["user_id"].values, eval_df["item_id"].values):
        if train_csr[u, i] == 0:
            out.setdefault(int(u), set()).add(int(i))
    return out

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

def last_days(df, days):
    return df[df["ts"] >= df["ts"].max() - days * 86400]

def new_item_truth(truth, train_items):
    out = {u: {i for i in t if i not in train_items} for u, t in truth.items()}
    return {u: t for u, t in out.items() if t}