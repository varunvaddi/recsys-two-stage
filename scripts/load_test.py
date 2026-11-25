"""Closed-loop load test against a running server.

Start the server first:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE uvicorn recsys.api:app --port 8000
Then:
    python scripts/load_test.py --seconds 10 --levels 1 8 32
"""
import argparse
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp

MIX = [("recommend", 0.70), ("related", 0.15), ("up_next", 0.10), ("trending", 0.05)]


def make_url(kind, rng, n_users, n_items):
    if kind == "recommend":
        return "recommend", f"/recommend/{rng.integers(n_users)}?k=10&n_new=2"
    if kind == "related":
        return "related", f"/related/{rng.integers(n_items)}?k=10"
    if kind == "up_next":
        return "up_next", f"/up-next/{rng.integers(n_users)}?k=10"
    return "trending", "/trending?k=10"


async def worker(client, stop_at, rng, n_users, n_items, out):
    kinds = [m[0] for m in MIX]
    probs = [m[1] for m in MIX]
    while time.perf_counter() < stop_at:
        kind, url = make_url(rng.choice(kinds, p=probs), rng, n_users, n_items)
        t0 = time.perf_counter()
        try:
            r = await client.get(url)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        out.append((kind, (time.perf_counter() - t0) * 1000, ok))


async def run_level(base, conc, seconds, n_users, n_items, warmup=2.0):
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=30) as client:
        junk = []
        stop = time.perf_counter() + warmup
        await asyncio.gather(*[worker(client, stop, np.random.default_rng(i), n_users, n_items, junk)
                               for i in range(conc)])
        out = []
        t0 = time.perf_counter()
        stop = t0 + seconds
        await asyncio.gather(*[worker(client, stop, np.random.default_rng(100 + i), n_users, n_items, out)
                               for i in range(conc)])
        elapsed = time.perf_counter() - t0
    return out, elapsed


def summarize(out, elapsed, conc):
    df = pd.DataFrame(out, columns=["endpoint", "ms", "ok"])
    rows = []
    for name, g in [("ALL", df)] + list(df.groupby("endpoint")):
        p50, p95, p99 = np.percentile(g["ms"], [50, 95, 99])
        rows.append({"concurrency": conc, "endpoint": name, "requests": len(g),
                     "rps": round(len(g) / elapsed, 1), "p50_ms": round(p50, 2),
                     "p95_ms": round(p95, 2), "p99_ms": round(p99, 2),
                     "error_rate": round(1 - g["ok"].mean(), 4)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--out", default="reports/load_test.csv")
    ap.add_argument("--tag", default="1 worker")
    a = ap.parse_args()

    n_users, n_items = sp.load_npz(Path(a.artifacts) / "X.npz").shape
    rows = []
    for conc in a.levels:
        out, elapsed = asyncio.run(run_level(a.base, conc, a.seconds, n_users, n_items))
        level = summarize(out, elapsed, conc)
        rows += level
        r = level[0]
        print(f"concurrency {conc:>3}: {r['rps']:>7} rps  p50 {r['p50_ms']} ms  "
              f"p95 {r['p95_ms']} ms  p99 {r['p99_ms']} ms  errors {r['error_rate']}")
    res = pd.DataFrame(rows)
    res.insert(0, "setup", a.tag)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(res[res.endpoint == "ALL"].to_string(index=False))
    m = httpx.get(a.base + "/metrics").json()
    print("\nserver-side (excludes network + queueing):")
    for name, v in m["endpoints"].items():
        print(f"  {name:<24} p50 {v['p50_ms']} ms  p99 {v['p99_ms']} ms  errors {v['errors']}")
    print("cold-start responses:", m["cold_start_responses"])


if __name__ == "__main__":
    main()