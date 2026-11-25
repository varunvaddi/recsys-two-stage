import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.baselines import topk_unseen


def build_samples(df, max_len, pad):
    """One training sample per interaction (except each user's first):
    input = the previous max_len items in time order (left-padded), target = this item."""
    d = df.sort_values(["user_id", "ts"], kind="stable")
    items = d["item_id"].values
    pos = d.groupby("user_id").cumcount().values
    X = np.full((len(d), max_len), pad, dtype=np.int32)
    for j in range(1, max_len + 1):
        rows = np.nonzero(pos >= j)[0]
        X[rows, max_len - j] = items[rows - j]
    keep = pos >= 1
    return X[keep], items[keep].astype(np.int64)


def last_items(df, n_users, max_len, pad):
    """(n_users, max_len): each user's most recent max_len items, oldest first, left-padded."""
    d = df.sort_values(["user_id", "ts"], kind="stable")
    back = d.groupby("user_id").cumcount(ascending=False).values  # 0 = most recent
    out = np.full((n_users, max_len), pad, dtype=np.int32)
    ok = back < max_len
    out[d["user_id"].values[ok], max_len - 1 - back[ok]] = d["item_id"].values[ok]
    return out


class GRUNet(nn.Module):
    def __init__(self, n_items, dim):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)  # last row = padding
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def encode(self, seq):
        _, h = self.gru(self.emb(seq))
        return F.normalize(h[-1], dim=-1)

    def item_matrix(self):
        return F.normalize(self.emb.weight[:-1], dim=-1)


def seq_loss(h, pos, targets, temperature, log_q):
    """In-batch softmax: each sequence must pick its own next item among the batch's targets.
    logQ correction stops the model from just favouring popular items."""
    logits = h @ pos.T / temperature - log_q[targets][None, :]
    clash = targets[:, None] == targets[None, :]
    clash.fill_diagonal_(False)
    logits = logits.masked_fill(clash, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(h)))


class SeqModel:
    """'Up next': GRU over the user's last few games -> scores for every game."""
    name = "gru_seq"

    def __init__(self, dim=64, max_len=10, epochs=3, batch_size=1024, lr=1e-3,
                 temperature=0.1, seed=0, verbose=True, callback=None):
        self.dim, self.max_len, self.epochs = dim, max_len, epochs
        self.batch_size, self.lr, self.temperature = batch_size, lr, temperature
        self.seed, self.verbose, self.callback = seed, verbose, callback

    def fit(self, df, n_users, n_items):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xs, ys = build_samples(df, self.max_len, n_items)
        self.last = last_items(df, n_users, self.max_len, n_items)
        freq = np.bincount(ys, minlength=n_items) + 1.0
        log_q = torch.log(torch.tensor(freq / freq.sum(), dtype=torch.float32))
        self.net = GRUNet(n_items, self.dim)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = rng.permutation(len(ys))
            total, steps = 0.0, 0
            for b in range(0, len(ys) - self.batch_size + 1, self.batch_size):
                idx = perm[b:b + self.batch_size]
                seq = torch.from_numpy(Xs[idx]).long()
                tgt = torch.from_numpy(ys[idx])
                h = self.net.encode(seq)
                pos = F.normalize(self.net.emb(tgt), dim=-1)
                loss = seq_loss(h, pos, tgt, self.temperature, log_q)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, steps = total + loss.item(), steps + 1
            if self.verbose:
                print(f"epoch {epoch}: loss {total / max(steps, 1):.4f}", flush=True)
            if self.callback:
                self.callback(epoch, total / max(steps, 1), self)
        return self

    def recommend(self, users, k, seen, chunk=512):
        self.net.eval()
        outs = []
        with torch.no_grad():
            items = self.net.item_matrix()
            for s in range(0, len(users), chunk):
                u = users[s:s + chunk]
                h = self.net.encode(torch.from_numpy(self.last[u]).long())
                outs.append(topk_unseen((h @ items.T).numpy(), seen[u], k))
        return np.vstack(outs)