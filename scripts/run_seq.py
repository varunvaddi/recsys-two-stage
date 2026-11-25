import argparse

import numpy as np
import pandas as pd

from recsys import baselines, cooc, data, metrics, seqmodel

K = 10


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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="reports/seq.csv")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    users = np.sort(np.random.default_rng(0).choice(users, min(a.sample, len(users)), replace=False))
    probe = users[:3000]

    def on_epoch(epoch, loss, model):
        rec = model.recommend(probe, K, X)
        r = metrics.evaluate(rec, probe, truth, n_items, K)[f"recall@{K}"]
        print(f"  epoch {epoch}: validation recall@{K} on {len(probe):,} users = {r:.4f}", flush=True)

    model = seqmodel.SeqModel(dim=a.dim, max_len=a.max_len, epochs=a.epochs,
                              callback=on_epoch).fit(train, n_users, n_items)

    n_hist = np.asarray(X[users].sum(axis=1)).ravel()
    terc = pd.qcut(pd.Series(n_hist).rank(method="first"), 3,
                   labels=["short history", "mid history", "long history"]).values

    def score(rec):
        r = metrics.evaluate(rec, users, truth, n_items, K)
        out = {f"recall@{K}": r[f"recall@{K}"], f"ndcg@{K}": r[f"ndcg@{K}"],
               f"coverage@{K}": r[f"coverage@{K}"]}
        for t in ["short history", "mid history", "long history"]:
            m = terc == t
            out[f"recall ({t})"] = metrics.evaluate(
                rec[m], users[m], truth, n_items, K)[f"recall@{K}"]
        return out

    anchors = cooc.last_item(train).reindex(users).values.astype(int)
    fallback = np.argsort(-np.asarray(X_recent.sum(axis=0)).ravel())[:1000]
    together = cooc.BoughtTogether(alpha=0.0).fit(X)
    recs = {
        "popularity (pop_365d)": baselines.Popularity().fit(X_recent).recommend(users, K, X),
        "last game -> co-occurrence (7a)": together.recommend_for(
            anchors, K, X, users, fallback)[0],
        "GRU over last games (7b)": model.recommend(users, K, X),
    }
    table = pd.DataFrame({name: score(r) for name, r in recs.items()}).T
    print(f"\n{len(users):,} users; history = last {a.max_len} games; truth = validation games")
    print(table.round(4).to_string())
    table.to_csv(a.out)

    titles = titles_for(a.items)
    name = (lambda i: f"{i} {titles.get(i, '')}") if titles is not None else str
    gru = recs["GRU over last games (7b)"]
    print("\nEXAMPLES: last 4 games -> predicted next games (HIT = they really played it)")
    for j in np.where(n_hist >= 8)[0][:3]:
        u = users[j]
        print(f"\n  user {u} recently played:")
        for i in model.last[u][-4:]:
            if i < n_items:
                print(f"    {name(int(i))}")
        for i in gru[j][:5]:
            print(f"    -> {'HIT ' if i in truth[u] else '    '}{name(int(i))}")


if __name__ == "__main__":
    main()