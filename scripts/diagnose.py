import numpy as np
import pandas as pd

from recsys import baselines, data, metrics

K = 10
DAY = 86400


def main():
    df = pd.read_parquet("data/processed/core.parquet")
    n_users, n_items = int(df["user_id"].max()) + 1, int(df["item_id"].max()) + 1
    train, val, _ = data.time_split(df)
    t_end = train["ts"].max()

    known_item = val["item_id"].isin(set(train["item_id"]))
    known_user = val["user_id"].isin(set(train["user_id"]))
    print(f"val events with a known item:  {known_item.mean():.1%}")
    print(f"val events with a known user:  {known_user.mean():.1%}")
    print(f"val events with both known:    {(known_item & known_user).mean():.1%}")

    pop = train["item_id"].value_counts()
    for n in (100, 1000):
        share = val["item_id"].isin(pop.index[:n]).mean()
        print(f"top-{n} train items cover {share:.1%} of val events")
    recent_items = set(train.loc[train["ts"] >= t_end - 90 * DAY, "item_id"])
    share = val["item_id"].isin(recent_items).mean()
    print(f"items active in the last 90 train days cover {share:.1%} of val events")

    X = data.to_csr(train, n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    rng = np.random.default_rng(0)
    users = np.sort(rng.choice(users, min(20000, len(users)), replace=False))

    rows = {}
    windows = {"pop_30d": 30, "pop_90d": 90, "pop_365d": 365, "pop_all": None}
    for name, days in windows.items():
        recent = train if days is None else train[train["ts"] >= t_end - days * DAY]
        model = baselines.Popularity().fit(data.to_csr(recent, n_users, n_items))
        recs = model.recommend(users, K, X)
        rows[name] = metrics.evaluate(recs, users, truth, n_items, K)
    print(pd.DataFrame(rows).T.round(4))


if __name__ == "__main__":
    main()