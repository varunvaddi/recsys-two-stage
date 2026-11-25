import argparse

import numpy as np
import pandas as pd

from recsys import baselines, candidates, content, data, twotower

N = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--emb", default="data/processed/item_emb.npy")
    ap.add_argument("--sample", type=int, default=10000)
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    emb = np.load(a.emb)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    rng = np.random.default_rng(0)
    users = np.sort(rng.choice(users, min(a.sample, len(users)), replace=False))
    truth_new = data.new_item_truth(truth, set(train["item_id"]))
    users_new = np.array([u for u in users if u in truth_new])

    sources = [
        ("pop", baselines.Popularity(), X_recent),
        ("als", baselines.ALS(factors=64, regularization=50.0, alpha=40.0), X),
        ("itemcf", baselines.ItemCF(), X),
        ("twotower", twotower.TwoTowerModel(epochs=2, verbose=False), X),
        ("content", content.ContentBased(emb), X),
    ]
    tables = []
    for name, model, fit_x in sources:
        model.fit(fit_x)
        tables.append(candidates.top_n_table(model, name, users, N, X))
        print(f"done {name}", flush=True)
    cands = candidates.merge_candidates(tables)
    print(f"\ncandidates per user: {len(cands) / len(users):.0f} "
          f"(top {N} from each of {len(sources)} sources, duplicates merged)")

    rows = {}
    for name, _, _ in sources:
        col = f"rank_{name}"
        rows[name] = (candidates.candidate_recall(cands, truth, users, col),
                      candidates.candidate_recall(cands, truth_new, users_new, col))
    rows["ALL SOURCES"] = (candidates.candidate_recall(cands, truth, users),
                           candidates.candidate_recall(cands, truth_new, users_new))
    table = pd.DataFrame(rows, index=[f"recall@{N} (all)", f"recall@{N} (new items)"]).T
    print(table.round(4))
    cands.to_parquet("data/processed/candidates.parquet")


if __name__ == "__main__":
    main()