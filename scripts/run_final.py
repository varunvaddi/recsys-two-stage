"""Final held-out TEST evaluation.

Protocol: the ranker was trained on (train-period features -> validation-period labels) and is
frozen. Here we move one period forward the way production would: refit the five candidate
sources on train+validation history, rebuild every feature from that history, score with the
frozen ranker, and judge against the TEST period, which nothing has ever been tuned on.
"""
import argparse

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from recsys import (
    baselines,
    candidates,
    content,
    data,
    evalstats,
    metrics,
    ranker,
    rerank,
    tracking,
    twotower,
)

K, N = 10, 50
NEW_SLOTS = 2  # the operating point chosen on validation data (phase 6a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--emb", default="data/processed/item_emb.npy")
    ap.add_argument("--model", default="data/processed/ranker.txt")
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--out", default="reports/final_test.csv")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    emb = np.load(a.emb)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, test = data.time_split(df)
    hist = pd.concat([train, val])  # everything known before the test period starts
    day = lambda t: pd.to_datetime(t, unit="s").date()
    print(f"history {day(hist['ts'].min())} .. {day(hist['ts'].max())}   "
          f"test period {day(test['ts'].min())} .. {day(test['ts'].max())}")

    X = data.to_csr(hist, n_users, n_items)
    X_recent = data.to_csr(data.last_days(hist, 365), n_users, n_items)
    truth = data.truth_dict(test, X)
    truth_new = data.new_item_truth(truth, set(hist["item_id"]))
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    users = np.sort(np.random.default_rng(0).choice(users, min(a.sample, len(users)), replace=False))
    users_new = np.array([u for u in users if u in truth_new])
    print(f"{len(users):,} test users ({len(users_new):,} with a never-seen-item purchase)")

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
    ceiling = candidates.candidate_recall(cands, truth, users)
    print(f"candidates per user {len(cands) / len(users):.0f}; ceiling recall {ceiling:.4f}")

    booster = lgb.Booster(model_file=a.model)
    f = ranker.build_features(cands, X, X_recent, emb, hist)
    scores = booster.predict(f[booster.feature_name()], num_threads=1)

    recs = {}
    for c in [c for c in cands.columns if c.startswith("rank_")]:
        label = "popularity (365d)" if c == "rank_pop" else c.replace("rank_", "")
        recs[label] = ranker.top_k(f, -f[c].fillna(1e9), users, K)
    recs["RANKER"] = ranker.top_k(f, scores, users, K)
    recs[f"RANKER + {NEW_SLOTS} new-item slots"] = rerank.rerank(f, scores, users, K, 0.0, NEW_SLOTS)

    row_of = {u: j for j, u in enumerate(users)}
    idx_new = [row_of[u] for u in users_new]
    per, rows = {}, {}
    for name, rec in recs.items():
        r = metrics.evaluate(rec, users, truth, n_items, K)
        new = metrics.evaluate(rec[idx_new], users_new, truth_new, n_items, K)
        recall_u, _ = evalstats.per_user(rec, users, truth, K)
        per[name] = recall_u
        m, lo, hi = evalstats.bootstrap_ci(recall_u)
        rows[name] = {f"recall@{K}": m, "ci95_low": lo, "ci95_high": hi, f"ndcg@{K}": r[f"ndcg@{K}"],
                      f"map@{K}": r[f"map@{K}"], f"coverage@{K}": r[f"coverage@{K}"],
                      f"recall@{K} new items": new[f"recall@{K}"]}
    table = pd.DataFrame(rows).T
    print(f"\nTEST PERIOD, {len(users):,} users, k={K}")
    print(table.round(4).to_string())
    print(f"\ncandidate ceiling: {ceiling:.4f}")

    base = "popularity (365d)"
    lifts = {}
    for name in ["RANKER", f"RANKER + {NEW_SLOTS} new-item slots"]:
        lifts[name] = evalstats.paired_lift_ci(per[name], per[base])
        l, lo, hi = lifts[name]
        print(f"{name} vs {base}: {l:+.1%} recall@{K} (95% CI {lo:+.1%} .. {hi:+.1%})")

    table.to_csv(a.out)
    with tracking.run("final-test", "refit-on-train+val", {"n_users": len(users), "sample": a.sample,
                                                            "new_slots": NEW_SLOTS, "k": K}):
        tracking.log_row(table.loc["RANKER"])
        tracking.log_row({"popularity_recall": table.loc[base, f"recall@{K}"], "ceiling": ceiling,
                          "ranker_lift_vs_pop": lifts["RANKER"][0],
                          "ranker_lift_ci_low": lifts["RANKER"][1], "ranker_lift_ci_high": lifts["RANKER"][2]})
        mlflow.log_artifact(a.out)


if __name__ == "__main__":
    main()