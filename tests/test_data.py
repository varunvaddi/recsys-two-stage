import pandas as pd

from recsys import data


def test_synthetic_shape():
    df = data.make_synthetic(n_users=100, n_items=50, seed=1)
    assert set(df.columns) == {"user_id", "item_id", "ts"}
    assert df["user_id"].nunique() == 100
    assert not df.duplicated(["user_id", "item_id"]).any()


def test_time_split_has_no_leakage():
    df = data.make_synthetic(n_users=200, n_items=100, seed=1)
    train, val, test = data.time_split(df)
    assert train["ts"].max() < val["ts"].min()
    assert val["ts"].max() < test["ts"].min()
    assert len(train) + len(val) + len(test) == len(df)

def test_last_days_keeps_only_recent_events():
    df = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [0, 1, 2],
                       "ts": [0, 86400 * 10, 86400 * 11]})
    out = data.last_days(df, 2)
    assert out["item_id"].tolist() == [1, 2]