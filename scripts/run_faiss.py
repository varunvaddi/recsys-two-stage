import argparse
import os
import time

import faiss
import numpy as np
import pandas as pd

from recsys import ann, data, metrics

K, K_FETCH = 10, 300
CPUS = os.cpu_count() or 1


def timed_build(fn):
    faiss.omp_set_num_threads(CPUS)
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def real_data_benchmark(a):
    df = pd.read_parquet(a.src)
    uvecs, ivecs = np.load(a.uvecs), np.load(a.ivecs)
    n_users, n_items = uvecs.shape[0], ivecs.shape[0]
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    users = np.sort(np.random.default_rng(0).choice(users, min(a.sample, len(users)), replace=False))
    q = uvecs[users]
    exact = ann.exact_topk(q, ivecs, K_FETCH)
    print(f"{n_items:,} items, dim {ivecs.shape[1]}, {len(users):,} query users", flush=True)

    rows = {}

    def record(name, build_s, batch_fn, single_fn):
        faiss.omp_set_num_threads(CPUS)
        t = time.perf_counter()
        ids = batch_fn(q)
        qps = len(q) / (time.perf_counter() - t)
        rec, _ = ann.filter_seen(ids, users, X, K)
        r = metrics.evaluate(rec, users, truth, n_items, K)[f"recall@{K}"]
        faiss.omp_set_num_threads(1)
        p50, p95, p99 = ann.latency_ms(single_fn, q)
        rows[name] = {"overlap@100": ann.overlap_recall(ids[:, :100], exact[:, :100]),
                      f"recall@{K} vs truth": r, "p50 ms": p50, "p95 ms": p95, "p99 ms": p99,
                      "batch QPS": qps, "build s": build_s}

    record("exact (numpy)", 0.0, lambda x: ann.exact_topk(x, ivecs, K_FETCH),
           ann.numpy_search(ivecs, K_FETCH))
    flat, bs = timed_build(lambda: ann.build_flat(ivecs))
    record("faiss flat (exact)", bs, lambda x: ann.search(flat, x, K_FETCH),
           lambda x: ann.search(flat, x, K_FETCH))
    hnsw, bs = timed_build(lambda: ann.build_hnsw(ivecs))
    for ef in (16, 64, 256):
        hnsw.hnsw.efSearch = ef
        record(f"hnsw efSearch={ef}", bs, lambda x: ann.search(hnsw, x, K_FETCH),
               lambda x: ann.search(hnsw, x, K_FETCH))
    ivf, bs = timed_build(lambda: ann.build_ivf(ivecs))
    for nprobe in (1, 8, 32):
        ivf.nprobe = nprobe
        record(f"ivf nprobe={nprobe}", bs, lambda x: ann.search(ivf, x, K_FETCH),
               lambda x: ann.search(ivf, x, K_FETCH))
    table = pd.DataFrame(rows).T
    print("\nREAL CATALOG: retrieval with the two-tower embeddings")
    print(table.round(4).to_string())
    table.to_csv(a.out)


def scale_benchmark(n_items, dim=64, n_queries=500):
    """Clustered random vectors (like real embeddings, items group into taste clusters)."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(2000, dim)).astype(np.float32)

    def make(n):
        v = centers[rng.integers(0, len(centers), n)] + 0.5 * rng.normal(size=(n, dim)).astype(np.float32)
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    items, queries = make(n_items), make(n_queries)
    print(f"\nSCALE TEST (synthetic clustered vectors): {n_items:,} items, dim {dim}", flush=True)
    flat, _ = timed_build(lambda: ann.build_flat(items))
    exact = ann.search(flat, queries, 10)
    hnsw, bs = timed_build(lambda: ann.build_hnsw(items, m=16, ef_construction=100))
    rows = {}
    faiss.omp_set_num_threads(1)
    p = ann.latency_ms(ann.numpy_search(items, 10), queries, n=200)
    rows["exact (numpy)"] = {"overlap@10": 1.0, "p50 ms": p[0], "p95 ms": p[1], "p99 ms": p[2]}
    p = ann.latency_ms(lambda x: ann.search(flat, x, 10), queries, n=200)
    rows["faiss flat (exact)"] = {"overlap@10": 1.0, "p50 ms": p[0], "p95 ms": p[1], "p99 ms": p[2]}
    for ef in (16, 64, 128, 256):
        hnsw.hnsw.efSearch = ef
        ov = ann.overlap_recall(ann.search(hnsw, queries, 10), exact)
        p = ann.latency_ms(lambda x: ann.search(hnsw, x, 10), queries, n=400)
        rows[f"hnsw efSearch={ef}"] = {"overlap@10": ov, "p50 ms": p[0], "p95 ms": p[1], "p99 ms": p[2]}
    t = pd.DataFrame(rows).T
    print(t.round(4).to_string())
    print(f"(HNSW build took {bs:.0f}s)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--uvecs", default="data/processed/user_vecs.npy")
    ap.add_argument("--ivecs", default="data/processed/item_vecs.npy")
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--scale", type=int, default=0, help="also run a synthetic N-item benchmark")
    ap.add_argument("--out", default="reports/faiss.csv")
    a = ap.parse_args()
    real_data_benchmark(a)
    if a.scale:
        scale_benchmark(a.scale).to_csv("reports/faiss_scale.csv")


if __name__ == "__main__":
    main()