import argparse

import numpy as np
import pandas as pd

from recsys import baselines, content, data, metrics, twotower

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--emb", default="data/processed/item_emb.npy")
    ap.add_argument("--sample", type=int, default=20000)
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    emb = np.load(a.emb)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    assert emb.shape[0] == n_items, f"{emb.shape[0]} embeddings vs {n_items} items"
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    rng = np.random.default_rng(0)
    users = np.sort(rng.choice(users, min(a.sample, len(users)), replace=False))

    truth_new = data.new_item_truth(truth, set(train["item_id"]))
    mask = np.isin(users, list(truth_new))
    print(f"users: {len(users):,}   users who also bought a brand-new item: {mask.sum():,}")

    runs = [
        ("pop_365d", baselines.Popularity(), X_recent),
        ("als_tuned", baselines.ALS(factors=64, regularization=50.0, alpha=40.0), X),
        ("two_tower", twotower.TwoTowerModel(epochs=2, verbose=False), X),
        ("content", content.ContentBased(emb), X),
    ]
    all_rows, new_rows = {}, {}
    for name, model, fit_x in runs:
        model.fit(fit_x)
        recs = model.recommend(users, K, X)
        all_rows[name] = metrics.evaluate(recs, users, truth, n_items, K)
        new_rows[name] = metrics.evaluate(recs[mask], users[mask], truth_new, n_items, K)
        print(f"done {name}", flush=True)

    print("\nALL next interactions")
    print(pd.DataFrame(all_rows).T.round(4))
    print("\nONLY brand-new items (not seen in training)")
    print(pd.DataFrame(new_rows).T.round(4))


if __name__ == "__main__":
    main()