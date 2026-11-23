import argparse

import pandas as pd

from recsys import prepare


def describe(df):
    n_u, n_i = df["user_id"].nunique(), df["item_id"].nunique()
    return f"rows={len(df):,} users={n_u:,} items={n_i:,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/interactions.parquet")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--min-rating", type=float, default=4.0)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    raw = pd.read_parquet(a.src)
    pos = raw[raw["rating"] >= a.min_rating]
    print(f"positives (rating >= {a.min_rating}): {len(pos):,} of {len(raw):,}")

    for k in (3, 5, 10):
        print(f"k={k}: {describe(prepare.k_core(pos, k, k))}")

    core = prepare.k_core(pos, a.k, a.k)
    core, _user_ids, item_ids = prepare.encode_ids(core)
    core[["user_id", "item_id", "ts"]].to_parquet(f"{a.out}/core.parquet")
    pd.DataFrame({"raw_item_id": item_ids}).to_parquet(f"{a.out}/item_ids.parquet")
    lo = pd.to_datetime(core["ts"].min(), unit="s").date()
    hi = pd.to_datetime(core["ts"].max(), unit="s").date()
    print(f"saved k={a.k} core: {describe(core)}, dates {lo} to {hi}")


if __name__ == "__main__":
    main()