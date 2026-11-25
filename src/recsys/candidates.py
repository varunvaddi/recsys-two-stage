import numpy as np
import pandas as pd


def top_n_table(model, name, users, n, seen):
    recs = model.recommend(users, n, seen)
    return pd.DataFrame({
        "user_id": np.repeat(users, n),
        "item_id": recs.ravel(),
        f"rank_{name}": np.tile(np.arange(1, n + 1), len(users)),
    })


def merge_candidates(tables):
    out = tables[0]
    for t in tables[1:]:
        out = out.merge(t, on=["user_id", "item_id"], how="outer")
    return out


def candidate_recall(cands, truth, users, col=None):
    pairs = pd.DataFrame(
        [(u, i) for u in users for i in truth.get(u, ())],
        columns=["user_id", "item_id"],
    )
    if col is not None:
        cands = cands[cands[col].notna()]
    found = cands[["user_id", "item_id"]].drop_duplicates()
    hit = pairs.merge(found, on=["user_id", "item_id"], how="left", indicator=True)
    hit["hit"] = hit["_merge"] == "both"
    return float(hit.groupby("user_id")["hit"].mean().mean())