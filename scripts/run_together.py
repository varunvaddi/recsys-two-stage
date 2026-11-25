import argparse

import numpy as np
import pandas as pd

from recsys import baselines, cooc, data, metrics

K = 10
ALPHAS = [0.0, 0.5, 0.75, 1.0]


def titles_for(path):
    try:
        it = pd.read_parquet(path)
        col = "title" if "title" in it else ("text" if "text" in it else None)
        if col is None:
            return None
        idx = it["item_id"] if "item_id" in it else pd.Series(np.arange(len(it)))
        return pd.Series(it[col].astype(str).str[:60].values, index=idx.values)
    except (OSError, KeyError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--out", default="reports/together.csv")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    users = np.sort(np.random.default_rng(0).choice(users, min(a.sample, len(users)), replace=False))
    anchors = cooc.last_item(train).reindex(users).values.astype(int)
    pop = np.asarray(X.sum(axis=0)).ravel()
    fallback = np.argsort(-np.asarray(X_recent.sum(axis=0)).ravel())[:1000]

    terc = pd.qcut(pd.Series(pop[anchors]).rank(method="first"), 3,
                   labels=["niche anchor", "mid anchor", "popular anchor"]).values

    def score(rec):
        r = metrics.evaluate(rec, users, truth, n_items, K)
        out = {f"recall@{K}": r[f"recall@{K}"], f"ndcg@{K}": r[f"ndcg@{K}"],
               f"coverage@{K}": r[f"coverage@{K}"]}
        for t in ["niche anchor", "mid anchor", "popular anchor"]:
            m = terc == t
            out[f"recall ({t})"] = metrics.evaluate(
                rec[m], users[m], truth, n_items, K)[f"recall@{K}"]
        return out

    rows = {"popularity (same list for every game)":
            {**score(baselines.Popularity().fit(X_recent).recommend(users, K, X)),
             "anchors with a full real list": np.nan}}
    models = {}
    for alpha in ALPHAS:
        m = cooc.BoughtTogether(alpha=alpha).fit(X)
        rec, n_real = m.recommend_for(anchors, K, X, users, fallback)
        rows[f"co-occurrence alpha={alpha}"] = {
            **score(rec), "anchors with a full real list": float((n_real >= K).mean())}
        models[alpha] = m
        print(f"alpha={alpha} done", flush=True)
    table = pd.DataFrame(rows).T
    print(f"\n{len(users):,} users; anchor = their most recent train game; "
          f"truth = what they played next (validation)")
    print(table.round(4).to_string())
    table.to_csv(a.out)

    best_name = table.drop(index=table.index[0])[f"recall@{K}"].idxmax()
    best = models[float(best_name.split("=")[1])]
    print(f"\nEXAMPLES with {best_name}")
    titles = titles_for(a.items)
    name = (lambda i: f"{i} {titles.get(i, '')}") if titles is not None else str
    pool = np.where(pop >= 20)[0]
    pool = pool[np.argsort(-pop[pool])]
    for q in (0.01, 0.2, 0.6):
        item = int(pool[int(len(pool) * q)])
        print(f"\n  game page: {name(item)}  (owned by {int(pop[item])} users)")
        for r in best.related(item, 5):
            print(f"    also owned: {name(int(r))}")


if __name__ == "__main__":
    main()