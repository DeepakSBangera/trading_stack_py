import argparse

import duckdb

p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--n", type=int, default=50)
p.add_argument("--cols", default="", help="comma-separated columns, blank = all")
a = p.parse_args()

cols = [c.strip() for c in a.cols.split(",") if c.strip()]
select = ", ".join(cols) if cols else "*"

con = duckdb.connect()
df = con.sql(f"SELECT {select} FROM read_parquet('{a.file}') LIMIT {a.n}").df()
print(df.to_string(index=False))
