import numpy as np

from recsys import baselines, data, metrics

K = 10
N_USERS, N_ITEMS = 2000, 600


def main():
    df = data.make_synthetic(n_users=N_USERS, n_items=N_ITEMS)
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, N_USERS, N_ITEMS)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    print(f"evaluating on {len(users)} users")

    for model in [baselines.Popularity(), baselines.ItemCF()]:
        model.fit(X)
        recs = model.recommend(users, K, X)
        result = metrics.evaluate(recs, users, truth, N_ITEMS, K)
        print(model.name, {k: round(v, 4) for k, v in result.items()})


if __name__ == "__main__":
    main()