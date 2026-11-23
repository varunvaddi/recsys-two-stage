import argparse

import pandas as pd

from recsys import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/review_categories/Video_Games.jsonl")
    ap.add_argument("--dst", default="data/processed/interactions.parquet")
    a = ap.parse_args()

    prepare.reviews_to_parquet(a.src, a.dst)
    df = pd.read_parquet(a.dst)
    print(f"rows: {len(df):,}")
    print(f"users: {df['user_id'].nunique():,}  items: {df['item_id'].nunique():,}")
    lo = pd.to_datetime(df["ts"].min(), unit="s").date()
    hi = pd.to_datetime(df["ts"].max(), unit="s").date()
    print(f"dates: {lo} to {hi}")
    print(df["rating"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    main()