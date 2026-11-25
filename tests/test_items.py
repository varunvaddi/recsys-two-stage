import json

import pandas as pd

from recsys import prepare


def _write_meta(path):
    rows = [
        {"parent_asin": "A1", "title": "Space Quest", "store": "Acme",
         "main_category": "Video Games", "categories": ["Video Games", "PC"],
         "description": ["A long space adventure."], "features": ["Fast", "Fun", "x"],
         "price": None, "images": []},
        {"parent_asin": "A2", "title": "Farm Days", "store": None,
         "main_category": None, "categories": [], "description": [], "features": []},
        {"parent_asin": "A4", "title": "T" * 500, "store": "S",
         "categories": ["C"], "description": ["D"], "features": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_build_items_keeps_order_and_flags_missing(tmp_path):
    src = tmp_path / "meta.jsonl"
    _write_meta(src)
    dst = tmp_path / "meta.parquet"
    prepare.meta_to_parquet(str(src), str(dst))
    raw_ids = pd.DataFrame({"raw_item_id": ["A2", "A1", "A3", "A4"]})
    items = prepare.build_items(pd.read_parquet(dst), raw_ids)

    assert items["item_id"].tolist() == [0, 1, 2, 3]
    assert items["raw_item_id"].tolist() == ["A2", "A1", "A3", "A4"]
    assert items["has_meta"].tolist() == [True, True, False, True]
    assert items.loc[1, "text"] == (
        "Space Quest | Acme | Video Games | PC | Fast | Fun | A long space adventure."
    )
    assert items.loc[0, "text"] == "Farm Days"
    assert items.loc[2, "text"] == ""
    assert len(items.loc[3, "text"]) == 400