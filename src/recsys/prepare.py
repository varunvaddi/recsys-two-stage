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