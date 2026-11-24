import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse


class Popularity:
    name = "popularity"

    def fit(self, train_csr):
        self.pop = np.asarray(train_csr.sum(axis=0)).ravel()
        return self

    def recommend(self, users, k, seen):
        order = np.argsort(-self.pop)
        out = np.empty((len(users), k), dtype=np.int64)
        for row, u in enumerate(users):
            s = set(seen[u].indices)
            picked = [i for i in order[: k + len(s)] if i not in s][:k]
            out[row] = picked
        return out

def topk_unseen(scores, seen_rows, k):
    scores = scores.copy()
    r, c = seen_rows.nonzero()
    scores[r, c] = -np.inf
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


class ItemCF:
    name = "item_cf"

    def __init__(self, n_neighbors=50, chunk=500):
        self.n_neighbors, self.chunk = n_neighbors, chunk

    def fit(self, X):
        Xc = X.tocsc().astype(np.float32)
        norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        Xn = (Xc @ sparse.diags(1.0 / norms)).tocsc()
        n_items = X.shape[1]
        nn = min(self.n_neighbors, n_items - 2)
        rows, cols, vals = [], [], []
        for s in range(0, n_items, self.chunk):
            block = (Xn[:, s:s + self.chunk].T @ Xn).toarray()
            block[np.arange(block.shape[0]), s + np.arange(block.shape[0])] = 0.0
            top = np.argpartition(-block, nn, axis=1)[:, :nn]
            v = np.take_along_axis(block, top, axis=1)
            rows.append(np.repeat(s + np.arange(block.shape[0]), nn))
            cols.append(top.ravel())
            vals.append(v.ravel())
        self.S = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
        )
        self.S.eliminate_zeros()
        return self

    def recommend(self, users, k, seen, user_chunk=512):
        outs = []
        for s in range(0, len(users), user_chunk):
            u = users[s:s + user_chunk]
            scores = (seen[u] @ self.S).toarray()
            outs.append(topk_unseen(scores, seen[u], k))
        return np.vstack(outs)


class ALS:
    name = "als"

    def __init__(self, factors=16, regularization=10.0, iterations=15,
                 alpha=1.0, seed=0):
        self.kw = {
            "factors": factors,
            "regularization": regularization,
            "iterations": iterations,
            "random_state": seed,
        }
        self.alpha = alpha

    def fit(self, X):
        self.model = AlternatingLeastSquares(**self.kw)
        self.model.fit((X * self.alpha).astype(np.float32).tocsr(),
                       show_progress=False)
        return self

    def recommend(self, users, k, seen):
        ids, _ = self.model.recommend(
            np.asarray(users), seen[users], N=k, filter_already_liked_items=True
        )
        return ids