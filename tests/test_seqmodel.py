import numpy as np
import pandas as pd
from scipy import sparse

from recsys import seqmodel


def _df():
    return pd.DataFrame({"user_id": [0, 0, 0, 1], "item_id": [5, 6, 7, 9],
                         "ts": [1, 2, 3, 1]})


def test_build_samples_uses_previous_items_in_time_order():
    X, y = seqmodel.build_samples(_df(), max_len=2, pad=99)
    assert X.tolist() == [[99, 5], [5, 6]]  # user 1 has one item -> no sample
    assert y.tolist() == [6, 7]


def test_last_items_are_left_padded_and_most_recent():
    out = seqmodel.last_items(_df(), n_users=3, max_len=2, pad=99)
    assert out.tolist() == [[6, 7], [99, 9], [99, 99]]


def test_gru_learns_a_cycle():
    """Everyone walks the cycle 0 -> 1 -> ... -> 19 -> 0. The next item is predictable."""
    rng = np.random.default_rng(0)
    n_items, n_users, length = 20, 200, 10
    rows = []
    for u in range(n_users):
        start = int(rng.integers(0, n_items))
        for t in range(length):
            rows.append((u, (start + t) % n_items, t))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])
    model = seqmodel.SeqModel(dim=16, max_len=3, epochs=8, batch_size=128, lr=1e-2,
                              verbose=False).fit(df, n_users, n_items)
    users = np.arange(100)
    empty = sparse.csr_matrix((n_users, n_items), dtype=np.float32)
    rec = model.recommend(users, 1, empty)[:, 0]
    last = df[df["ts"] == length - 1].set_index("user_id")["item_id"].reindex(users).values
    assert (rec == (last + 1) % n_items).mean() > 0.8