import argparse
import time

import numpy as np
import pandas as pd

from recsys import data, metrics, ranker, serve

K = 10


def pct(x):
    return np.percentile(x, [50, 95, 99])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--cands", default="data/processed/candidates.parquet")
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--n", type=int, default=1000, help="requests for the latency benchmark")
    ap.add_argument("--eval-users", type=int, default=3000)
    ap.add_argument("--out", default="reports/serving.csv")
    a = ap.parse_args()

    rec = serve.Recommender.load(a.art, a.items)
    df = pd.read_parquet(a.src)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    truth = data.truth_dict(val, X)
    cands = pd.read_parquet(a.cands, columns=["user_id"])
    _, test_users = ranker.split_users(cands["user_id"].unique(), 0.3)
    print(f"sources in the ranker: {rec.sources}", flush=True)

    # ---- 1. latency of the full two-stage recommend() ----
    rng = np.random.default_rng(0)
    users = rng.choice(test_users, min(a.n, len(test_users)), replace=False)
    for u in users[:30]:
        rec.recommend(int(u), K)  # warm up
    stages = {}
    for u in users:
        t = {}
        rec.recommend(int(u), K, timings=t)
        for k, v in t.items():
            stages.setdefault(k, []).append(v)
    rows = {k: dict(zip(["p50 ms", "p95 ms", "p99 ms"], pct(v), strict=True))
            for k, v in sorted(stages.items())}
    for name, fn in {
        "related()": lambda i: rec.related(int(i), K),
        "up_next()": lambda i: rec.up_next(int(i), K),
        "trending()": lambda i: rec.trending(K),
    }.items():
        ts = []
        for i in users[:500]:
            t0 = time.perf_counter()
            fn(i if name != "related()" else rng.integers(0, n_items))
            ts.append((time.perf_counter() - t0) * 1000)
        rows[name] = dict(zip(["p50 ms", "p95 ms", "p99 ms"], pct(ts), strict=True))
    table = pd.DataFrame(rows).T
    table["max RPS (1 core)"] = np.where(table.index == "total_ms", 1000 / table["p50 ms"], np.nan)
    print(f"\nLATENCY over {len(users):,} requests (one at a time, single process)")
    print(table.round(3).to_string())
    table.to_csv(a.out)

    # ---- 2. does the served pipeline reproduce the offline accuracy? ----
    eval_users = np.array([u for u in test_users if u in truth][: a.eval_users])
    print(f"\nQUALITY of the served pipeline on {len(eval_users):,} held-out users", flush=True)
    for label, kw in {"ranker only": {}, "+ 2 new-item slots": {"n_new": 2},
                      "+ lambda=0.25": {"lam": 0.25}}.items():
        recs = np.array([[i["item_id"] for i in rec.recommend(int(u), K, **kw)["items"]]
                         for u in eval_users])
        r = metrics.evaluate(recs, eval_users, truth, n_items, K)
        print(f"  {label:20s} recall@{K} {r[f'recall@{K}']:.4f}  ndcg@{K} {r[f'ndcg@{K}']:.4f}  "
              f"coverage@{K} {r[f'coverage@{K}']:.4f}")

    # ---- 3. one real example per surface ----
    u = int(eval_users[0])
    print("\nEXAMPLE")
    print(f"  recommend(user {u}):")
    for it in rec.recommend(u, 5)["items"]:
        print(f"    {it['item_id']} {it['title']}  via {it['sources']}")
    nxt = rec.up_next(u, 3)
    print(f"  up_next: after {nxt['anchor']['title'] if nxt['anchor'] else None}")
    for it in nxt["items"]:
        print(f"    {it['item_id']} {it['title']}")
    for it in rec.trending(3)["items"]:
        print(f"  trending: {it['item_id']} {it['title']}")


if __name__ == "__main__":
    main()