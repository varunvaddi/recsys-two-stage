import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.baselines import topk_unseen


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def user_vec(self, u):
        return F.normalize(self.user_mlp(self.user_emb(u)), dim=-1)

    def item_vec(self, i):
        return F.normalize(self.item_mlp(self.item_emb(i)), dim=-1)


def in_batch_loss(net, u, i, temperature=0.1, log_q=None):
    logits = net.user_vec(u) @ net.item_vec(i).T / temperature
    if log_q is not None:
        logits = logits - log_q[i][None, :]
    clash = (u[:, None] == u[None, :]) | (i[:, None] == i[None, :])
    clash.fill_diagonal_(False)
    logits = logits.masked_fill(clash, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(i)))


class TwoTowerModel:
    name = "two_tower"

    def __init__(self, dim=64, epochs=5, batch_size=1024, lr=1e-3,
                 temperature=0.1, seed=0, logq=True, verbose=True, callback=None):
        self.dim, self.epochs, self.batch_size = dim, epochs, batch_size
        self.lr, self.temperature, self.seed = lr, temperature, seed
        self.logq, self.verbose, self.callback = logq, verbose, callback

    def _embed(self, n_users, n_items):
        self.net.eval()
        with torch.no_grad():
            self.user_vecs = self.net.user_vec(torch.arange(n_users)).numpy()
            self.item_vecs = self.net.item_vec(torch.arange(n_items)).numpy()
        self.net.train()

    def fit(self, X):
        torch.manual_seed(self.seed)
        n_users, n_items = X.shape
        coo = X.tocoo()
        users = torch.from_numpy(coo.row.astype(np.int64))
        items = torch.from_numpy(coo.col.astype(np.int64))
        self.net = TwoTower(n_users, n_items, self.dim)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n = len(users)
        log_q = None
        if self.logq:
            counts = torch.bincount(items, minlength=n_items).float() + 1.0
            log_q = torch.log(counts / counts.sum())
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                if len(idx) < 2:
                    continue
                loss = in_batch_loss(
                    self.net, users[idx], items[idx], self.temperature, log_q
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if self.callback is not None:
                self._embed(n_users, n_items)
                self.callback(epoch + 1, total / n, self)
            elif self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs}  loss {total / n:.4f}", flush=True)
        self._embed(n_users, n_items)
        self.net.eval()
        return self

    def recommend(self, users, k, seen, user_chunk=512):
        outs = []
        for s in range(0, len(users), user_chunk):
            u = users[s:s + user_chunk]
            scores = self.user_vecs[u] @ self.item_vecs.T
            outs.append(topk_unseen(scores, seen[u], k))
        return np.vstack(outs)