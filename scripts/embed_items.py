import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from recsys import embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/processed/items.parquet")
    ap.add_argument("--out", default="data/processed/item_emb.npy")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()

    items = pd.read_parquet(a.items)
    encoder = SentenceTransformer(a.model)
    emb = embed.embed_texts(items["text"].tolist(), encoder)
    np.save(a.out, emb)
    print(f"saved {emb.shape} to {a.out}")

    rng = np.random.default_rng(1)
    for i in rng.choice(len(items), 3, replace=False):
        print(f"\n{items['title'][i][:90]}")
        top, sims = embed.nearest_items(emb, int(i), 5)
        for j, s in zip(top, sims):
            print(f"   {s:.2f}  {items['title'][j][:90]}")


if __name__ == "__main__":
    main()