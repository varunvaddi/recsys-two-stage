import numpy as np
import pytest
from scipy import sparse

from recsys import ann


def _vecs(n_items=2000, n_users=50, d=16, seed=0):
    rng = np.random.default_rng(seed)
    items = rng.normal(size=(n_items, d)).astype(np.float32)
    users = rng.normal(size=(n_users, d)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    users /= np.linalg.norm(users, axis=1, keepdims=True)
    return users, items


def test_exact_topk_matches_full_sort():
    users, items = _vecs()
    got = ann.exact_topk(users, items, 10, chunk=16)
    want = np.argsort(-(users @ items.T), axis=1)[:, :10]
    assert (got == want).all()


def test_overlap_recall_known_values():
    exact = np.array([[1, 2, 3, 4]])
    assert ann.overlap_recall(np.array([[4, 3, 9, 8]]), exact) == 0.5
    assert ann.overlap_recall(exact, exact) == 1.0


def test_faiss_flat_is_exact_and_hnsw_ivf_are_close():
    pytest.importorskip("faiss")
    users, items = _vecs()
    exact = ann.exact_topk(users, items, 20)
    assert ann.overlap_recall(ann.search(ann.build_flat(items), users, 20), exact) == 1.0
    hnsw = ann.build_hnsw(items, m=16, ef_construction=100, ef_search=200)
    assert ann.overlap_recall(ann.search(hnsw, users, 20), exact) > 0.95
    ivf = ann.build_ivf(items, nlist=16, nprobe=16)  # probing every list = exact
    assert ann.overlap_recall(ann.search(ivf, users, 20), exact) == 1.0
    ivf1 = ann.build_ivf(items, nlist=16, nprobe=1)
    assert ann.overlap_recall(ann.search(ivf1, users, 20), exact) < 1.0  # fewer probes, less exact


def test_filter_seen_drops_owned_items_and_pads():
    seen = sparse.csr_matrix(np.array([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32))
    ids = np.array([[1, 3, 2, 4], [0, 1, 2, 3]])
    out, short = ann.filter_seen(ids, np.array([0, 1]), seen, 3)
    assert out[0].tolist() == [3, 4, -1]   # items 1 and 2 are owned
    assert out[1].tolist() == [0, 1, 2]
    assert short == 1