import pandas as pd

from recsys import prepare


def _toy():
    rows = [("U1", "A"), ("U1", "B"), ("U2", "A"), ("U2", "B"),
            ("U3", "A"), ("U3", "C")]
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def test_k_core_removes_in_cascade():
    core = prepare.k_core(_toy(), k_user=2, k_item=2)
    assert len(core) == 4
    assert set(core["user_id"]) == {"U1", "U2"}
    assert set(core["item_id"]) == {"A", "B"}


def test_encode_ids_dense_and_reversible():
    core = prepare.k_core(_toy(), k_user=2, k_item=2)
    enc, _user_ids, item_ids = prepare.encode_ids(core)
    assert sorted(enc["user_id"].unique()) == [0, 1]
    assert sorted(enc["item_id"].unique()) == [0, 1]
    back = [item_ids[i] for i in enc["item_id"]]
    assert back == core["item_id"].tolist()