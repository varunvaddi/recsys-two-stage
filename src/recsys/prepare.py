import os

import duckdb


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