import json
import os

import duckdb
import pandas as pd


def reviews_to_parquet(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT user_id, parent_asin AS item_id, rating,
                   CAST(timestamp // 1000 AS BIGINT) AS ts
            FROM read_json('{src}', format='newline_delimited',
                columns={{user_id: 'VARCHAR', parent_asin: 'VARCHAR',
                          rating: 'DOUBLE', timestamp: 'BIGINT'}})
        ) TO '{dst}' (FORMAT PARQUET)
    """)
    con.close()

def k_core(df, k_user=5, k_item=5):
    while True:
        n = len(df)
        df = df[df.groupby("user_id")["item_id"].transform("size") >= k_user]
        df = df[df.groupby("item_id")["user_id"].transform("size") >= k_item]
        if len(df) == n:
            return df


def encode_ids(df):
    df = df.copy()
    df["user_id"], user_ids = pd.factorize(df["user_id"])
    df["item_id"], item_ids = pd.factorize(df["item_id"])
    return df, user_ids, item_ids

def meta_to_parquet(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT parent_asin, title, store, main_category,
                   categories, description, features
            FROM read_json('{src}', format='newline_delimited',
                columns={{parent_asin: 'VARCHAR', title: 'VARCHAR',
                          store: 'VARCHAR', main_category: 'VARCHAR',
                          categories: 'JSON', description: 'JSON',
                          features: 'JSON'}})
        ) TO '{dst}' (FORMAT PARQUET)
    """)
    con.close()


def _text(x):
    return x if isinstance(x, str) else ""


def _as_list(x):
    if not isinstance(x, str):
        return []
    value = json.loads(x)
    if isinstance(value, list):
        return [str(s) for s in value]
    return [str(value)] if value else []


def build_item_text(row, max_chars=400):
    parts = [_text(row["title"]), _text(row["store"])]
    parts += _as_list(row["categories"])
    parts += _as_list(row["features"])[:2]
    parts += _as_list(row["description"])[:1]
    return " | ".join(p.strip() for p in parts if p.strip())[:max_chars]


def build_items(meta, raw_ids):
    out = raw_ids.reset_index().rename(columns={"index": "item_id"})
    meta = meta.drop_duplicates("parent_asin")
    out = out.merge(meta, left_on="raw_item_id", right_on="parent_asin", how="left")
    out["text"] = [build_item_text(r) for r in out.to_dict("records")]
    out["has_meta"] = out["title"].notna()
    return out[["item_id", "raw_item_id", "title", "store", "text", "has_meta"]]