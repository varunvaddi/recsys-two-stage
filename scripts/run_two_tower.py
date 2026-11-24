import argparse
import time

import numpy as np
import pandas as pd

from recsys import baselines, data, metrics, twotower

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--sample", type=int, default=20000)
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    rng = np.random.default_rng(0)
    users = np.sort(rng.choice(users, min(a.sample, len(users)), replace=False))
    probe = users[:3000]
    print(f"evaluating on {len(users):,} users")

    def show_progress(epoch, loss, model):
        recs = model.recommend(probe, K, X)
        r = metrics.evaluate(recs, probe, truth, n_items, K)["recall@10"]
        print(f"epoch {epoch}: loss {loss:.3f}  val recall@10 {r:.4f}", flush=True)

    runs = [
        ("pop_365d", baselines.Popularity(), X_recent),
        ("als_tuned", baselines.ALS(factors=64, regularization=50.0, alpha=40.0), X),
        ("two_tower", twotower.TwoTowerModel(epochs=a.epochs, callback=show_progress), X),
    ]
    rows = {}
    for name, model, fit_x in runs:
        t0 = time.time()
        model.fit(fit_x)
        recs = model.recommend(users, K, X)
        res = metrics.evaluate(recs, users, truth, n_items, K)
        res["seconds"] = round(time.time() - t0, 1)
        rows[name] = res
    print(pd.DataFrame(rows).T.round(4))


if __name__ == "__main__":
    main()