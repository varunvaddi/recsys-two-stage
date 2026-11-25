import numpy as np
from scipy import sparse

from recsys import content, data


def _emb():
    emb = np.array([[1.0, 0.0], [0.95, 0.05], [0.9, 0.1],
                    [0.0, 1.0], [0.05, 0.95], [0.1, 0.9]], dtype=np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def test_recommends_similar_items_even_if_never_interacted_with():
    X = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 6))
    model = content.ContentBased(_emb()).fit(X)
    recs = model.recommend(np.array([0]), 2, X)
    assert recs[0].tolist() == [1, 2]


def test_never_recommends_seen_items():
    X = sparse.csr_matrix(([1.0, 1.0], ([0, 0], [0, 3])), shape=(1, 6))
    recs = content.ContentBased(_emb()).fit(X).recommend(np.array([0]), 4, X)
    assert not set(recs[0].tolist()) & {0, 3}


def test_new_item_truth_keeps_only_unseen_items():
    truth = {0: {1, 2}, 1: {2}, 2: {1}}
    out = data.new_item_truth(truth, train_items={1})
    assert out == {0: {2}, 1: {2}}