import numpy as np

from recsys import baselines, data, metrics


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

def test_item_cf_recommends_unseen_items():
    df = data.make_synthetic(n_users=100, n_items=50, seed=1)
    X = data.to_csr(df, 100, 50)
    model = baselines.ItemCF(n_neighbors=10).fit(X)
    users = np.arange(10)
    recs = model.recommend(users, 5, X)
    assert recs.shape == (10, 5)
    for row, u in zip(recs, users):
        assert len(set(row)) == 5
        assert not set(row) & set(X[u].indices)


def test_item_cf_beats_popularity():
    n_users, n_items = 800, 300
    df = data.make_synthetic(n_users=n_users, n_items=n_items, seed=1)
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    scores = {}
    for m in [baselines.Popularity(), baselines.ItemCF(n_neighbors=30)]:
        m.fit(X)
        recs = m.recommend(users, 10, X)
        scores[m.name] = metrics.evaluate(recs, users, truth, n_items, 10)["recall@10"]
    assert scores["item_cf"] > scores["popularity"]