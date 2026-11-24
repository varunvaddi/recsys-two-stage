import numpy as np
import torch

from recsys import baselines, data, metrics, twotower


def test_loss_is_finite_with_duplicate_items_and_users():
    net = twotower.TwoTower(n_users=10, n_items=10, dim=8)
    u = torch.tensor([0, 1, 2, 2])
    i = torch.tensor([5, 5, 6, 7])
    assert torch.isfinite(twotower.in_batch_loss(net, u, i))
    assert torch.isfinite(twotower.in_batch_loss(net, u, i, log_q=torch.zeros(10)))


def test_two_tower_recommends_unseen_items():
    df = data.make_synthetic(n_users=100, n_items=50, seed=1)
    X = data.to_csr(df, 100, 50)
    model = twotower.TwoTowerModel(dim=16, epochs=2, verbose=False).fit(X)
    users = np.arange(10)
    recs = model.recommend(users, 5, X)
    assert recs.shape == (10, 5)
    for row, u in zip(recs, users):
        assert len(set(row)) == 5
        assert not set(row) & set(X[u].indices)


def test_callback_runs_once_per_epoch():
    df = data.make_synthetic(n_users=100, n_items=50, seed=1)
    X = data.to_csr(df, 100, 50)
    seen_epochs = []
    twotower.TwoTowerModel(
        dim=8, epochs=3, callback=lambda e, loss, m: seen_epochs.append(e)
    ).fit(X)
    assert seen_epochs == [1, 2, 3]


def test_two_tower_beats_popularity_on_clustered_data():
    n_users, n_items = 800, 300
    df = data.make_synthetic(n_users=n_users, n_items=n_items, seed=1)
    train, val, _ = data.time_split(df)
    X = data.to_csr(train, n_users, n_items)
    truth = data.truth_dict(val, X)
    users = np.array(sorted(u for u in truth if X[u].nnz > 0))
    scores = {}
    models = [baselines.Popularity(),
              twotower.TwoTowerModel(dim=32, epochs=15, batch_size=256, verbose=False)]
    for m in models:
        m.fit(X)
        recs = m.recommend(users, 10, X)
        scores[m.name] = metrics.evaluate(recs, users, truth, n_items, 10)["recall@10"]
    assert scores["two_tower"] > scores["popularity"]