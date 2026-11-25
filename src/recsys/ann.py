import time

import numpy as np


def exact_topk(user_vecs, item_vecs, k, chunk=500):
    """Brute force: score every item for every user, keep the k best. The ground truth."""
    out = np.empty((len(user_vecs), k), dtype=np.int64)
    for s in range(0, len(user_vecs), chunk):
        scores = user_vecs[s:s + chunk] @ item_vecs.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        out[s:s + chunk] = np.take_along_axis(top, order, axis=1)
    return out


def build_flat(item_vecs):
    import faiss
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(np.ascontiguousarray(item_vecs, dtype=np.float32))
    return index


def build_hnsw(item_vecs, m=32, ef_construction=200, ef_search=64):
    import faiss
    index = faiss.IndexHNSWFlat(item_vecs.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(np.ascontiguousarray(item_vecs, dtype=np.float32))
    return index


def build_ivf(item_vecs, nlist=None, nprobe=8):
    import faiss
    n, d = item_vecs.shape
    nlist = nlist or max(16, int(4 * np.sqrt(n)))
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    x = np.ascontiguousarray(item_vecs, dtype=np.float32)
    index.train(x)
    index.add(x)
    index.nprobe = nprobe
    return index


def search(index, queries, k):
    _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return ids


def overlap_recall(approx_ids, exact_ids):
    """Average share of the exact top-k that the approximate search also returned."""
    hits = [len(set(a.tolist()) & set(e.tolist())) / len(e) for a, e in zip(approx_ids, exact_ids)]
    return float(np.mean(hits))


def filter_seen(ids, users, seen, k):
    """First k retrieved items per user that the user hasn't already interacted with.
    Short lists are padded with -1 (never a hit); returns (matrix, n_short)."""
    out = np.full((len(ids), k), -1, dtype=np.int64)
    short = 0
    for r, (row, u) in enumerate(zip(ids, users)):
        owned = set(seen[u].indices)
        picked = [int(i) for i in row if i >= 0 and i not in owned][:k]
        short += len(picked) < k
        out[r, :len(picked)] = picked
    return out, short


def latency_ms(fn, queries, n=500, warmup=30):
    """One query at a time, the way an online request arrives. Returns p50/p95/p99 in ms."""
    qs = queries[: n + warmup]
    for q in qs[:warmup]:
        fn(q[None, :])
    times = []
    for q in qs[warmup:]:
        t = time.perf_counter()
        fn(q[None, :])
        times.append((time.perf_counter() - t) * 1000)
    return tuple(float(x) for x in np.percentile(times, [50, 95, 99]))


def numpy_search(item_vecs, k):
    def fn(q):
        scores = q @ item_vecs.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        return np.take_along_axis(top, order, axis=1)
    return fn