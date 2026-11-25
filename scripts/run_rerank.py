import argparse

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys import analysis, data, metrics, ranker, rerank

plt.switch_backend("Agg")  # draw to files, no window

K = 10
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
NEW_SLOTS = [0, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--emb", default="data/processed/item_emb.npy")
    ap.add_argument("--cands", default="data/processed/candidates.parquet")
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
    pop = np.asarray(X.sum(axis=0)).ravel()

    _, test_users = ranker.split_users(cands["user_id"].unique(), 0.3)
    f = ranker.add_labels(ranker.build_features(cands, X, X_recent, emb, train), truth)
    cols = ranker.feature_cols(f)
    test = f[f["user_id"].isin(test_users)].reset_index(drop=True)
    scores = lgb.Booster(model_file=a.model).predict(test[cols])

    row_of = {u: j for j, u in enumerate(test_users)}
    users_new = np.array([u for u in test_users if u in truth_new])
    idx_new = [row_of[u] for u in users_new]

    rows = []
    for n_new in NEW_SLOTS:
        for lam in LAMBDAS:
            rec = rerank.rerank(test, scores, test_users, K, lam, n_new)
            r = metrics.evaluate(rec, test_users, truth, n_items, K)
            new = metrics.evaluate(rec[idx_new], users_new, truth_new, n_items, K)
            shares = analysis.popularity_shares({"x": rec}, pop)["x"]
            rows.append({
                "new_slots": n_new, "lambda": lam,
                f"recall@{K}": r[f"recall@{K}"], f"ndcg@{K}": r[f"ndcg@{K}"],
                f"coverage@{K}": r[f"coverage@{K}"],
                "top20%_share": shares.iloc[-1], "bottom20%_share": shares.iloc[0],
                "top1%_share": analysis.head_share({"x": rec}, pop)["x"],
                "new_item_recall": new[f"recall@{K}"],
            })
            print(f"new_slots={n_new} lambda={lam} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("reports/rerank.csv", index=False)

    actual = np.concatenate([list(truth[u]) for u in test_users])
    ref = analysis.popularity_shares({"a": actual}, pop)["a"]
    print(f"\nREFERENCE (what users really played): top20% {ref.iloc[-1]:.1%}, "
          f"bottom20% {ref.iloc[0]:.1%}, "
          f"top1% {analysis.head_share({'a': actual}, pop)['a']:.1%}")
    show = t.copy()
    for c in ["top20%_share", "bottom20%_share", "top1%_share"]:
        show[c] = (show[c] * 100).round(1)
    print(show.round(4).to_string(index=False))

    base = t[(t["new_slots"] == 0) & (t["lambda"] == 0)].iloc[0]
    ok = t[t[f"recall@{K}"] >= 0.95 * base[f"recall@{K}"]]
    best = ok.sort_values(["top1%_share", f"recall@{K}"], ascending=[True, False]).iloc[0]
    print("\nOPERATING POINT: least popularity-biased setting that keeps >=95% of baseline recall")
    print(f"  new_slots={int(best['new_slots'])}, lambda={best['lambda']}")
    print(f"  recall@{K}: {base[f'recall@{K}']:.4f} -> {best[f'recall@{K}']:.4f} "
          f"({best[f'recall@{K}'] / base[f'recall@{K}'] - 1:+.1%})")
    print(f"  coverage@{K}: {base[f'coverage@{K}']:.4f} -> {best[f'coverage@{K}']:.4f}")
    print(f"  top-1% share: {base['top1%_share']:.1%} -> {best['top1%_share']:.1%}")
    print(f"  new-item recall@{K}: {base['new_item_recall']:.4f} -> {best['new_item_recall']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for n_new, g in t.groupby("new_slots"):
        label = f"{n_new} new-item slots"
        axes[0].plot(g["top1%_share"] * 100, g[f"recall@{K}"], "o-", label=label)
        axes[1].plot(g[f"coverage@{K}"], g[f"recall@{K}"], "o-", label=label)
    axes[0].set_xlabel("% of recommendations from the top-1% most popular games")
    axes[1].set_xlabel(f"catalog coverage@{K}")
    for ax in axes:
        ax.set_ylabel(f"recall@{K}")
    axes[0].legend()
    fig.suptitle("Accuracy vs popularity bias (each line sweeps lambda 0 to 2)")
    fig.tight_layout()
    fig.savefig("reports/rerank_tradeoff.png", dpi=150)


if __name__ == "__main__":
    main()