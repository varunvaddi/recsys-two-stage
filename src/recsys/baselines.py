import numpy as np


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