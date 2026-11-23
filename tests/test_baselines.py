import numpy as np

from recsys import baselines, data


def test_popularity_recommends_unseen_items():
    df = data.make_synthetic(n_users=100, n_items=50, seed=1)
    X = data.to_csr(df, 100, 50)
    model = baselines.Popularity().fit(X)
    users = np.arange(10)
    recs = model.recommend(users, 5, X)
    assert recs.shape == (10, 5)
    for row, u in zip(recs, users):
        assert len(set(row)) == 5
        assert not set(row) & set(X[u].indices)