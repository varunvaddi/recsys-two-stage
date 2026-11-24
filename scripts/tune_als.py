import itertools

import numpy as np
import pandas as pd

from recsys import baselines, data, metrics

K = 10


def main():
    df = pd.read_parquet("data/processed/core.parquet")
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    rng = np.random.default_rng(0)
    users = np.sort(rng.choice(users, min(20000, len(users)), replace=False))

    pop = baselines.Popularity()
    pop.fit(data.to_csr(data.last_days(train, 365), n_users, n_items))
    ref = metrics.evaluate(pop.recommend(users, K, X), users, truth, n_items, K)
    print(f"reference pop_365d: recall@10={ref['recall@10']:.4f}")

    rows = []
    for f, r, a in itertools.product((16, 64), (1.0, 10.0, 50.0), (1.0, 10.0, 40.0)):
        model = baselines.ALS(factors=f, regularization=r, alpha=a).fit(X)
        res = metrics.evaluate(model.recommend(users, K, X), users, truth, n_items, K)
        rows.append({"factors": f, "reg": r, "alpha": a, **res})
        print(f"done factors={f} reg={r} alpha={a}", flush=True)
    table = pd.DataFrame(rows).sort_values("recall@10", ascending=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()