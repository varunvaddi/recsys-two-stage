import numpy as np


class BoughtTogether:
    """Item -> related items from co-ownership counts.
    score(a, b) = (# users who have both a and b) / pop(b) ** alpha
    alpha=0: raw counts (favours blockbusters). alpha=1: lift-like (favours niche matches)."""

    def __init__(self, alpha=0.5, n_neighbors=50, min_count=2, chunk=500):
        self.alpha, self.n_neighbors = alpha, n_neighbors
        self.min_count, self.chunk = min_count, chunk

    def fit(self, X):
        Xc = X.tocsc().astype(np.float32)
        n_items = X.shape[1]
        self.pop = np.asarray(X.sum(axis=0)).ravel()
        denom = np.maximum(self.pop, 1.0) ** self.alpha
        nn = min(self.n_neighbors, n_items - 1)
        self.ids = np.zeros((n_items, nn), dtype=np.int32)
        self.scores = np.full((n_items, nn), -np.inf, dtype=np.float32)
        for s in range(0, n_items, self.chunk):
            e = min(s + self.chunk, n_items)
            counts = (Xc[:, s:e].T @ Xc).toarray()
            score = counts / denom[None, :]
            score[counts < self.min_count] = -np.inf
            score[np.arange(e - s), s + np.arange(e - s)] = -np.inf
            top = np.argpartition(-score, nn - 1, axis=1)[:, :nn]
            vals = np.take_along_axis(score, top, axis=1)
            order = np.argsort(-vals, axis=1)
            self.ids[s:e] = np.take_along_axis(top, order, axis=1)
            self.scores[s:e] = np.take_along_axis(vals, order, axis=1)
        return self

    def related(self, item, k=10):
        ok = np.isfinite(self.scores[item])
        return self.ids[item][ok][:k]

    def recommend_for(self, anchors, k, seen=None, users=None, fallback=()):
        """Related items per anchor, skipping the anchor and (optionally) the user's own
        items. Short lists are padded from `fallback` (e.g. popular items).
        Returns (matrix, n_real) where n_real counts non-padded entries per row."""
        out = np.empty((len(anchors), k), dtype=np.int64)
        n_real = np.zeros(len(anchors), dtype=int)
        for r, a in enumerate(anchors):
            skip = set(seen[users[r]].indices) if seen is not None else set()
            skip.add(int(a))
            picked = [int(i) for i in self.related(a, self.n_neighbors) if i not in skip][:k]
            n_real[r] = len(picked)
            for i in fallback:
                if len(picked) == k:
                    break
                if i not in skip and i not in picked:
                    picked.append(int(i))
            assert len(picked) == k, "fallback list too short"
            out[r] = picked
        return out, n_real


def last_item(train):
    """user_id -> the item of that user's most recent train interaction."""
    idx = train.groupby("user_id")["ts"].idxmax()
    return train.loc[idx].set_index("user_id")["item_id"]