"""Small helpers for the Streamlit app, kept out of the app file so they can be unit-tested."""
import numpy as np
import pandas as pd


def search_titles(titles, query, limit=25):
    """Items whose title contains `query` (case-insensitive), as a Series of title indexed by item id."""
    query = (query or "").strip()
    if not query or titles is None:
        return titles.iloc[0:0] if titles is not None else pd.Series(dtype=str)
    return titles[titles.str.contains(query, case=False, regex=False, na=False)].head(limit)


def example_users(X, n=20, min_hist=5, seed=0):
    """A reproducible handful of users that have enough history to make a demo interesting."""
    sizes = np.diff(X.indptr)
    eligible = np.nonzero(sizes >= min_hist)[0]
    return np.sort(np.random.default_rng(seed).choice(eligible, min(n, len(eligible)), replace=False))


def items_frame(items, pop):
    """API-style item dicts -> a display table with the item's play count and a 'new' flag
    (new = nobody bought it in the training period)."""
    rows = []
    for rank, it in enumerate(items, 1):
        i = it["item_id"]
        rows.append({"rank": rank, "title": it.get("title") or f"item {i}", "item_id": i,
                     "plays": int(pop[i]) if 0 <= i < len(pop) else 0,
                     "new": bool(0 <= i < len(pop) and pop[i] == 0),
                     "sources": ", ".join(it.get("sources") or [])})
    return pd.DataFrame(rows)