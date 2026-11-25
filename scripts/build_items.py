import argparse

import pandas as pd

from recsys import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/raw/meta_categories/meta_Video_Games.jsonl")
    ap.add_argument("--item-ids", default="data/processed/item_ids.parquet")
    ap.add_argument("--out", default="data/processed/items.parquet")
    a = ap.parse_args()

    prepare.meta_to_parquet(a.meta, "data/processed/meta.parquet")
    meta = pd.read_parquet("data/processed/meta.parquet")
    items = prepare.build_items(meta, pd.read_parquet(a.item_ids))
    items.to_parquet(a.out)

    print(f"items: {len(items):,}  with metadata: {items['has_meta'].mean():.1%}")
    print(f"average text length: {items['text'].str.len().mean():.0f} characters")
    sample = items[items["has_meta"]].sample(3, random_state=0)
    for _, r in sample.iterrows():
        print(f"- {r['text'][:150]}")


if __name__ == "__main__":
    main()