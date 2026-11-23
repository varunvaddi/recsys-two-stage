import json

import pandas as pd

from recsys import prepare


def test_reviews_to_parquet(tmp_path):
    src = tmp_path / "reviews.jsonl"
    rows = [
        {"rating": 4.0, "text": "a", "images": [], "parent_asin": "I1",
         "user_id": "U1", "timestamp": 1608186804795},
        {"rating": 1.0, "text": "b", "images": [{"x": "y"}], "parent_asin": "I2",
         "user_id": "U2", "timestamp": 1608186900000},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows))
    dst = tmp_path / "out" / "interactions.parquet"
    prepare.reviews_to_parquet(str(src), str(dst))
    df = pd.read_parquet(dst).sort_values("item_id")
    assert list(df.columns) == ["user_id", "item_id", "rating", "ts"]
    assert df["item_id"].tolist() == ["I1", "I2"]
    assert df["ts"].tolist() == [1608186804, 1608186900]