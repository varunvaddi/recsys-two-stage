import argparse

import numpy as np
import pandas as pd

from recsys import data, twotower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/core.parquet")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, _, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    model = twotower.TwoTowerModel(dim=a.dim, epochs=a.epochs).fit(X)
    np.save("data/processed/user_vecs.npy", model.user_vecs.astype(np.float32))
    np.save("data/processed/item_vecs.npy", model.item_vecs.astype(np.float32))
    print(f"saved {model.user_vecs.shape} user vectors and {model.item_vecs.shape} item vectors")


if __name__ == "__main__":
    main()