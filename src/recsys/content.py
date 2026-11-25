import numpy as np

from recsys.baselines import topk_unseen


class ContentBased:
    name = "content"

    def __init__(self, emb):
        self.emb = np.asarray(emb, dtype=np.float32)

    def fit(self, X):
        return self

    def recommend(self, users, k, seen, user_chunk=512):
        outs = []
        for s in range(0, len(users), user_chunk):
            u = users[s:s + user_chunk]
            profile = np.asarray(seen[u] @ self.emb)
            norm = np.linalg.norm(profile, axis=1, keepdims=True)
            profile = profile / np.maximum(norm, 1e-9)
            outs.append(topk_unseen(profile @ self.emb.T, seen[u], k))
        return np.vstack(outs)