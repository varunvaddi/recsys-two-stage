import argparse

import numpy as np
import pandas as pd

from recsys import candidates, data, metrics, ranker

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--emb", default="data/processed/item_emb.npy")
    ap.add_argument("--cands", default="data/processed/candidates.parquet")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--out", default="reports/ranker.csv")
    ap.add_argument("--model", default="data/processed/ranker.txt")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    emb = np.load(a.emb)
    cands = pd.read_parquet(a.cands)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    truth = data.truth_dict(val, X)
    truth_new = data.new_item_truth(truth, set(train["item_id"]))

    train_users, test_users = ranker.split_users(cands["user_id"].unique(), a.test_frac)
    f = ranker.add_labels(ranker.build_features(cands, X, X_recent, emb, train), truth)
    cols = ranker.feature_cols(f)
    print(f"{len(f):,} candidate rows, {f['label'].mean():.2%} positive, {len(cols)} features")

    model = ranker.fit_ranker(f[f["user_id"].isin(train_users)], cols)
    print(f"trained {model.best_iteration_} trees (early stopping)")
    test = f[f["user_id"].isin(test_users)].reset_index(drop=True)

    recs = {}
    for c in [c for c in cands.columns if c.startswith("rank_")]:
        recs[c.replace("rank_", "top10 of ")] = ranker.top_k(
            test, -test[c].fillna(1e9), test_users, K)
    recs["RANKER (LightGBM)"] = ranker.top_k(test, model.predict(test[cols]), test_users, K)

    row = {u: j for j, u in enumerate(test_users)}
    users_new = np.array([u for u in test_users if u in truth_new])
    rows = {}
    for name, rec in recs.items():
        r = metrics.evaluate(rec, test_users, truth, n_items, K)
        new = metrics.evaluate(rec[[row[u] for u in users_new]], users_new, truth_new, n_items, K)
        rows[name] = {f"recall@{K}": r[f"recall@{K}"], f"ndcg@{K}": r[f"ndcg@{K}"],
                      f"map@{K}": r[f"map@{K}"], f"coverage@{K}": r[f"coverage@{K}"],
                      f"recall@{K} new items": new[f"recall@{K}"]}
    table = pd.DataFrame(rows).T
    print(f"\n{len(test_users):,} held-out users (never seen by the ranker)")
    print(table.round(4))
    ceiling = candidates.candidate_recall(cands, truth, test_users)
    print(f"\nceiling: the candidates contain {ceiling:.4f} of these users' future games")

    imp = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    print("\nfeature importance (share of total gain):")
    print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())

    table.to_csv(a.out)
    model.booster_.save_model(a.model)


if __name__ == "__main__":
    main()