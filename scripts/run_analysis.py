import argparse

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys import analysis, data, ranker

plt.switch_backend("Agg")  # draw to files, no window

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--emb", default="data/processed/item_emb.npy")
    ap.add_argument("--cands", default="data/processed/candidates.parquet")
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--model", default="data/processed/ranker.txt")
    ap.add_argument("--abl-users", type=int, default=12000,
                    help="train users per ablation retrain (smaller = faster)")
    ap.add_argument("--skip-ablations", action="store_true")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    emb = np.load(a.emb)
    cands = pd.read_parquet(a.cands)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    truth = data.truth_dict(val, X)
    pop = np.asarray(X.sum(axis=0)).ravel()

    train_users, test_users = ranker.split_users(cands["user_id"].unique(), 0.3)
    f = ranker.add_labels(ranker.build_features(cands, X, X_recent, emb, train), truth)
    cols = ranker.feature_cols(f)
    test = f[f["user_id"].isin(test_users)].reset_index(drop=True)
    booster = lgb.Booster(model_file=a.model)
    test["score"] = booster.predict(test[cols])
    sources = [c.replace("rank_", "") for c in cands.columns if c.startswith("rank_")]

    # ---------------- 1. ablations ----------------
    if not a.skip_ablations:
        rng = np.random.default_rng(1)
        tu = rng.choice(train_users, min(a.abl_users, len(train_users)), replace=False)
        rows = {"full model": analysis.fit_eval(f, cols, tu, test_users, truth, n_items, K)}
        print("full model done", flush=True)
        for s in sources:
            g = analysis.drop_source(f, s)
            c2 = [c for c in cols if c != f"rank_{s}"]
            rows[f"- source: {s}"] = analysis.fit_eval(g, c2, tu, test_users, truth, n_items, K)
            print(f"- source {s} done", flush=True)
        for name, group in analysis.feature_groups(cols).items():
            c2 = [c for c in cols if c not in group]
            rows[f"- features: {name}"] = analysis.fit_eval(f, c2, tu, test_users, truth, n_items, K)
            print(f"- features {name} done", flush=True)
        abl = pd.DataFrame(rows).T
        abl["recall change"] = abl[f"recall@{K}"] / abl.loc["full model", f"recall@{K}"] - 1
        print("\nABLATIONS (retrained without each piece; same test users)")
        print(abl.round(4).to_string())
        abl.to_csv("reports/ablations.csv")

    # ---------------- 2. SHAP on what the ranker actually recommends ----------------
    top = (test.sort_values(["user_id", "score"], ascending=[True, False])
               .groupby("user_id").head(K))
    top = top[top["user_id"].isin(test_users[:3000])]
    sv, _ = analysis.shap_values(booster, top[cols])
    shap_tbl = pd.DataFrame({
        "mean |SHAP|": np.abs(sv).mean(axis=0),
        "direction (+ = higher value raises score)": analysis.direction(top[cols], sv).values,
    }, index=cols).sort_values("mean |SHAP|", ascending=False)
    print(f"\nSHAP over the {len(top):,} games the ranker put in top-{K}")
    print(shap_tbl.round(3).to_string())
    shap_tbl.to_csv("reports/shap_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    shap_tbl["mean |SHAP|"][::-1].plot.barh(ax=ax, color="#3b6fb6")
    ax.set_xlabel("mean |SHAP| (average push on the ranking score)")
    ax.set_title("What drives the ranker's top-10")
    fig.tight_layout()
    fig.savefig("reports/shap_importance.png", dpi=150)

    # ---------------- 3. one user, explained ----------------
    titles = None
    try:
        it = pd.read_parquet(a.items)
        tcol = "title" if "title" in it else ("text" if "text" in it else None)
        if tcol:
            idx = it["item_id"] if "item_id" in it else pd.Series(np.arange(len(it)))
            titles = pd.Series(it[tcol].astype(str).str[:70].values, index=idx.values)
    except (OSError, KeyError, ValueError):
        titles = None
    hits = top[top["label"] == 1]["user_id"]
    u = int(hits.iloc[0]) if len(hits) else int(top["user_id"].iloc[0])
    rows_u = top[top["user_id"] == u].head(5)
    sv_u, _ = analysis.shap_values(booster, rows_u[cols])
    print(f"\nWHY these games for user {u} (top 5; HIT = they really played it)")
    for (_, r), s in zip(rows_u.iterrows(), sv_u):
        order = np.argsort(-np.abs(s))[:3]
        why = ", ".join(f"{cols[j]}={r[cols[j]]:.3g} ({s[j]:+.2f})" for j in order)
        name = titles.get(int(r["item_id"]), "") if titles is not None else ""
        print(f"  {'HIT ' if r['label'] else '    '}item {int(r['item_id'])} {name}\n      because: {why}")

    # ---------------- 4. popularity bias ----------------
    recs = {}
    for s in sources:
        recs[s] = ranker.top_k(test, -test[f"rank_{s}"].fillna(1e9), test_users, K)
    recs["RANKER"] = ranker.top_k(test, test["score"], test_users, K)
    recs["actual (validation)"] = np.concatenate([list(truth[u]) for u in test_users])
    shares = analysis.popularity_shares(recs, pop)
    head = analysis.head_share(recs, pop)
    print("\nPOPULARITY BIAS: share of recommended games in each popularity quintile")
    print((shares * 100).round(1).to_string())
    print("\nshare of recommendations that are the top-1% most popular games")
    print((head * 100).round(1).to_string())
    shares.to_csv("reports/popularity_bias.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    shares.T.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("share of recommendations")
    ax.set_title("Popularity bias by model (Q5 = most popular games)")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig("reports/popularity_bias.png", dpi=150)


if __name__ == "__main__":
    main()