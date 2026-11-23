import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys import baselines, data, metrics

plt.switch_backend("Agg")
K = 10
N_USERS, N_ITEMS = 2000, 600


def main():
    df = data.make_synthetic(n_users=N_USERS, n_items=N_ITEMS)
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, N_USERS, N_ITEMS)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    print(f"evaluating on {len(users)} users")

    rows = {}
    for model in [baselines.Popularity(), baselines.ItemCF(), baselines.ALS()]:
        model.fit(X)
        recs = model.recommend(users, K, X)
        rows[model.name] = metrics.evaluate(recs, users, truth, N_ITEMS, K)
    table = pd.DataFrame(rows).T
    print(table.round(4))

    os.makedirs("reports", exist_ok=True)
    table.to_csv("reports/baselines.csv")
    fig, axes = plt.subplots(1, len(table.columns), figsize=(14, 3.5))
    for ax, col in zip(axes, table.columns):
        ax.bar(table.index, table[col], color=["#999", "#4c78a8", "#e45756"])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig("reports/baselines.png", dpi=110)


if __name__ == "__main__":
    main()