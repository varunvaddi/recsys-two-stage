import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from recsys import baselines, cooc, data, serve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--emb", default="data/processed/item_emb.npy")
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--model", default="data/processed/ranker.txt")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--stats-only", action="store_true",
                    help="reuse the saved artifacts, only recompute the re-rank statistics")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def p(name):
        return os.path.join(a.out, name)

    booster = lgb.Booster(model_file=a.model)
    sources = [c[5:] for c in booster.feature_name() if c.startswith("rank_")]
    print("ranker was trained with sources:", sources)

    if a.stats_only:
        X = sparse.load_npz(p("X.npz")).tocsr()
        with open(p("meta.json")) as fh:
            meta = json.load(fh)
    else:
        df = pd.read_parquet(a.src)
        n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
        train, _, _ = data.time_split(df)
        X = data.to_csr(train, n_users, n_items)
        X_recent = data.to_csr(data.last_days(train, 365), n_users, n_items)

        sparse.save_npz(p("X.npz"), X)
        np.savez(p("derived.npz"), **serve.derived_arrays(train, X, X_recent))
        np.save(p("item_emb.npy"), np.load(a.emb))
        booster.save_model(p("ranker.txt"))

        m = cooc.BoughtTogether(alpha=0.0).fit(X)
        np.save(p("cooc_ids.npy"), m.ids)
        np.save(p("cooc_scores.npy"), m.scores)
        print("co-occurrence done", flush=True)
        if "als" in sources:
            als = baselines.ALS(factors=64, regularization=50.0, alpha=40.0).fit(X)
            np.save(p("als_user.npy"), np.asarray(als.model.user_factors, dtype=np.float32))
            np.save(p("als_item.npy"), np.asarray(als.model.item_factors, dtype=np.float32))
            print("ALS done", flush=True)
        if "itemcf" in sources:
            sparse.save_npz(p("itemcf_S.npz"), baselines.ItemCF().fit(X).S.tocsr())
            print("item-CF done", flush=True)
        if "twotower" in sources:
            for name in ("user_vecs.npy", "item_vecs.npy"):
                np.save(p(name), np.load(os.path.join("data/processed", name)))

        meta = {"n_users": n_users, "n_items": n_items, "sources": sources}
        with open(p("meta.json"), "w") as fh:
            json.dump(meta, fh)

    # score / popularity statistics for the re-ranker, from 1,000 sample users
    rec = serve.Recommender.load(a.out, a.items)
    users = np.random.default_rng(0).choice(np.nonzero(np.diff(X.indptr) > 0)[0], 1000, replace=False)
    raws, lps = [], []
    for u in users:
        _, feats = rec.candidate_features(int(u))
        raws.append(booster.predict(feats.values, num_threads=1))
        lps.append(np.log1p(feats["item_pop"].values))
    raws, lps = np.concatenate(raws), np.concatenate(lps)
    meta["rerank_stats"] = {"score_mean": float(raws.mean()), "score_std": float(raws.std()),
                            "logpop_mean": float(lps.mean()), "logpop_std": float(lps.std())}
    with open(p("meta.json"), "w") as fh:
        json.dump(meta, fh)
    print("saved artifacts to", a.out, meta["rerank_stats"])


if __name__ == "__main__":
    main()