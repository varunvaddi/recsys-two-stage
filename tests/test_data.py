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