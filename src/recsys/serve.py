import json
import os
import time
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from recsys import ann, cooc, ranker, rerank

N_SRC = 50  # candidates per source, same as offline


def time_decayed_popularity(train, n_items, half_life_days=30.0):
    """'Trending': each interaction counts 1, halved for every half_life_days of age."""
    age_days = (train["ts"].max() - train["ts"].values) / 86400.0
    return np.bincount(train["item_id"].values, weights=0.5 ** (age_days / half_life_days),
                       minlength=n_items)


def derived_arrays(train, X, X_recent):
    """Everything the serving path needs that is computed from train data once."""
    n_users, n_items = X.shape
    t_end = train["ts"].max()
    pop_recent = np.asarray(X_recent.sum(axis=0)).ravel()
    last = cooc.last_item(train).reindex(range(n_users)).fillna(-1).astype(np.int64)
    return {
        "pop": np.asarray(X.sum(axis=0)).ravel(),
        "pop_recent": pop_recent,
        "pop_order": np.argsort(-pop_recent)[:2000],
        "user_n_recent": np.asarray(X_recent.sum(axis=1)).ravel(),
        "item_age_days": ranker._days_since(train.groupby("item_id")["ts"].min(), n_items, t_end),
        "user_last_days": ranker._days_since(train.groupby("user_id")["ts"].max(), n_users, t_end),
        "trending": time_decayed_popularity(train, n_items),
        "last_item": last.values,
    }


def _top_unseen(scores, seen, n):
    s = np.asarray(scores, dtype=np.float32).copy()
    s[seen] = -np.inf
    n = min(n, len(s))
    top = np.argpartition(-s, n - 1)[:n]
    top = top[np.argsort(-s[top])]
    return top[np.isfinite(s[top])]


class Recommender:
    """All four product surfaces behind one object.
    recommend: 5 candidate sources -> LightGBM ranker -> popularity-aware re-rank
    related:   'frequently bought together' (co-occurrence)
    up_next:   co-occurrence on the user's most recent game
    trending:  time-decayed popularity"""

    def __init__(self, art, booster=None, feature_names=None, titles=None, stats=None):
        self.a, self.booster, self.titles, self.stats = art, booster, titles, stats
        self.feature_names = list(feature_names or booster.feature_name())
        self.sources = [c[5:] for c in self.feature_names if c.startswith("rank_")]
        self.a.log1p_pop = np.log1p(art.pop)
        self.index = None
        if "twotower" in self.sources:
            self.index = ann.build_hnsw(art.item_vecs, ef_search=256)

    @classmethod
    def load(cls, art_dir, items_path=None):
        def p(name):
            return os.path.join(art_dir, name)
        booster = lgb.Booster(model_file=p("ranker.txt"))
        with open(p("meta.json")) as fh:
            meta = json.load(fh)
        a = SimpleNamespace(
            X=sparse.load_npz(p("X.npz")).tocsr(), emb=np.load(p("item_emb.npy")),
            cooc_ids=np.load(p("cooc_ids.npy")), cooc_scores=np.load(p("cooc_scores.npy")))
        for k, v in np.load(p("derived.npz")).items():
            setattr(a, k, v)
        sources = [c[5:] for c in booster.feature_name() if c.startswith("rank_")]
        if "als" in sources:
            a.als_user, a.als_item = np.load(p("als_user.npy")), np.load(p("als_item.npy"))
        if "itemcf" in sources:
            a.S = sparse.load_npz(p("itemcf_S.npz")).tocsr()
        if "twotower" in sources:
            a.user_vecs, a.item_vecs = np.load(p("user_vecs.npy")), np.load(p("item_vecs.npy"))
        titles = None
        if items_path and os.path.exists(items_path):
            it = pd.read_parquet(items_path)
            col = "title" if "title" in it else ("text" if "text" in it else None)
            if col:
                idx = it["item_id"] if "item_id" in it else pd.Series(np.arange(len(it)))
                titles = pd.Series(it[col].astype(str).str[:80].values, index=idx.values)
        return cls(a, booster, titles=titles, stats=meta.get("rerank_stats"))

    # ---------- helpers ----------
    def _decorate(self, ids, **extra):
        return [{"item_id": int(i),
                 "title": None if self.titles is None else self.titles.get(int(i)), **extra}
                for i in ids]

    def _fallback(self, k):
        return _top_unseen(self.a.trending, np.array([], dtype=np.int64), k)

    def _known_user(self, user):
        return 0 <= user < self.a.X.shape[0] and self.a.X[user].nnz > 0

    # ---------- stage 1: candidates + features ----------
    def _source_lists(self, u, seen, prof_n):
        a, out, owned = self.a, {}, set(seen.tolist())
        for s in self.sources:
            if s == "pop":
                out[s] = np.array([i for i in a.pop_order if i not in owned][:N_SRC])
            elif s == "als":
                out[s] = _top_unseen(a.als_item @ a.als_user[u], seen, N_SRC)
            elif s == "itemcf":
                out[s] = _top_unseen((a.X[u] @ a.S).toarray().ravel(), seen, N_SRC)
            elif s == "twotower":
                fetch = min(a.item_vecs.shape[0], N_SRC + len(seen) + 5, 256)
                ids = ann.search(self.index, a.user_vecs[u:u + 1], fetch)[0]
                out[s] = np.array([i for i in ids if i >= 0 and i not in owned][:N_SRC])
            elif s == "content":
                out[s] = _top_unseen(a.emb @ prof_n, seen, N_SRC)
        return out

    def candidate_features(self, u, timings=None):
        a, t0 = self.a, time.perf_counter()
        row = a.X[u]
        seen = row.indices
        prof = np.asarray(row @ a.emb).ravel()
        norm = np.linalg.norm(prof)
        prof_n = (prof / norm if norm > 0 else prof).astype(np.float32)
        lists = self._source_lists(u, seen, prof_n)
        items = np.unique(np.concatenate(list(lists.values())))
        ranks = {}
        for s, ids in lists.items():
            r = np.full(len(items), np.nan)
            r[np.searchsorted(items, ids)] = np.arange(1, len(ids) + 1)
            ranks[s] = r
        t1 = time.perf_counter()
        rank_mat = np.column_stack([ranks[s] for s in self.sources])
        n_hist = row.nnz
        avg_pop = float((row @ a.log1p_pop)[0]) / max(n_hist, 1)
        f = {f"rank_{s}": ranks[s] for s in self.sources}
        f["n_sources"] = (~np.isnan(rank_mat)).sum(axis=1)
        f["best_rank"] = np.nanmin(rank_mat, axis=1)
        f["item_pop"] = a.pop[items]
        f["item_pop_recent"] = a.pop_recent[items]
        f["item_age_days"] = a.item_age_days[items]
        f["user_n_hist"] = np.full(len(items), n_hist)
        f["user_n_recent"] = np.full(len(items), a.user_n_recent[u])
        f["user_last_days"] = np.full(len(items), a.user_last_days[u])
        f["user_avg_log_pop"] = np.full(len(items), avg_pop)
        f["pop_gap"] = np.log1p(f["item_pop"]) - avg_pop
        f["content_sim"] = a.emb[items] @ prof_n
        feats = pd.DataFrame(f)[self.feature_names]
        if timings is not None:
            timings["1_candidate_sources_ms"] = (t1 - t0) * 1000
            timings["2_features_ms"] = (time.perf_counter() - t1) * 1000
        return items, feats

    # ---------- stage 2+3: ranker, then popularity-aware re-rank ----------
    def _adjusted(self, raw, item_pop, lam):
        lp = np.log1p(item_pop)
        st = self.stats
        if st is None:  # fall back to this request's own statistics
            st = {"score_mean": raw.mean(), "score_std": raw.std() or 1.0,
                  "logpop_mean": lp.mean(), "logpop_std": lp.std() or 1.0}
        return ((raw - st["score_mean"]) / st["score_std"]
                - lam * (lp - st["logpop_mean"]) / st["logpop_std"])

    def recommend(self, user, k=10, lam=0.0, n_new=0, timings=None):
        t_all = time.perf_counter()
        if not self._known_user(user):
            return {"user_id": user, "cold_start": True,
                    "items": self._decorate(self._fallback(k), sources=["trending"])}
        items, feats = self.candidate_features(user, timings)
        t = time.perf_counter()
        raw = self.booster.predict(feats.values, num_threads=1)
        if timings is not None:
            timings["3_ranker_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        adj = self._adjusted(raw, feats["item_pop"].values, lam)
        if n_new == 0:  # fast path: plain top-k by adjusted score
            pos = np.argsort(-adj)[:k]
        else:
            frame = pd.DataFrame({"user_id": user, "item_id": items,
                                  "item_pop": feats["item_pop"].values})
            kept = rerank.select_with_new_slots(frame, adj, k, n_new)
            pos = np.searchsorted(items, kept["item_id"].values)
        rank_cols = feats[[f"rank_{s}" for s in self.sources]].values
        out = []
        for j in pos:
            src = [s for s, r in zip(self.sources, rank_cols[j], strict=True) if not np.isnan(r)]
            out.append(self._decorate([items[j]], score=float(raw[j]), sources=src)[0])
        if timings is not None:
            timings["4_rerank_ms"] = (time.perf_counter() - t) * 1000
            timings["total_ms"] = (time.perf_counter() - t_all) * 1000
        return {"user_id": user, "cold_start": False, "items": out}

    # ---------- the other three surfaces ----------
    def related(self, item, k=10):
        a = self.a
        if not 0 <= item < len(a.pop):
            return {"item_id": item, "fallback": True,
                    "items": self._decorate(self._fallback(k), sources=["trending"])}
        ids = a.cooc_ids[item][np.isfinite(a.cooc_scores[item])][:k]
        if len(ids) == 0:
            return {"item_id": item, "fallback": True,
                    "items": self._decorate(self._fallback(k), sources=["trending"])}
        return {"item_id": item, "fallback": False, "items": self._decorate(ids)}

    def up_next(self, user, k=10):
        a = self.a
        last = int(a.last_item[user]) if 0 <= user < len(a.last_item) else -1
        if last < 0:
            return {"user_id": user, "anchor": None,
                    "items": self._decorate(self._fallback(k), sources=["trending"])}
        owned = set(a.X[user].indices.tolist()) | {last}
        ids = [int(i) for i in a.cooc_ids[last][np.isfinite(a.cooc_scores[last])]
               if int(i) not in owned][:k]
        for i in self._fallback(k + len(owned)):
            if len(ids) == k:
                break
            if int(i) not in owned and int(i) not in ids:
                ids.append(int(i))
        return {"user_id": user, "anchor": self._decorate([last])[0],
                "items": self._decorate(ids)}

    def trending(self, k=10):
        return {"items": self._decorate(self._fallback(k))}