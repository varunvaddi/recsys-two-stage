import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys import baselines, data, metrics

plt.switch_backend("Agg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    print(f"users={n_users:,} items={n_items:,} "
          f"train/val={len(train):,}/{len(val):,}")

    X = data.to_csr(train, n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    print(f"validation users with train history: {len(users):,}")
    if len(users) > a.sample:
        rng = np.random.default_rng(0)
        users = np.sort(rng.choice(users, a.sample, replace=False))
    print(f"evaluating on {len(users):,} users")

    rows = {}
    for model in [baselines.Popularity(), baselines.ItemCF(), baselines.ALS()]:
        t0 = time.time()
        model.fit(X)
        recs = model.recommend(users, a.k, X)
        res = metrics.evaluate(recs, users, truth, n_items, a.k)
        res["seconds"] = round(time.time() - t0, 1)
        rows[model.name] = res
    table = pd.DataFrame(rows).T
    print(table.round(4))

    os.makedirs(a.out, exist_ok=True)
    table.to_csv(f"{a.out}/real_baselines.csv")
    cols = [c for c in table.columns if c != "seconds"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 3.5))
    for ax, col in zip(axes, cols):
        ax.bar(table.index, table[col], color=["#999", "#4c78a8", "#e45756"])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(f"{a.out}/real_baselines.png", dpi=110)


if __name__ == "__main__":
    main()