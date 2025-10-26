import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def repartition_to_year_symbol(
    src_dir: str, dst_dir: str, pattern=r".+\.parquet$", date_col="date"
):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in Path(src_dir).glob("*.parquet") if re.match(pattern, p.name)]
    for fp in files:
        tbl = pq.read_table(fp)
        # ensure date is date32 or timestamp[ms, UTC]
        if tbl.schema.field(date_col).type not in (pa.timestamp("ms", "UTC"), pa.date32()):
            df = tbl.to_pandas()
            df[date_col] = pd.to_datetime(df[date_col], utc=True)
            tbl = pa.Table.from_pandas(df, preserve_index=False)
        years = pc.year(tbl[date_col])
        tbl = tbl.append_column("_year", years)
        for y in set(tbl.column("_year").to_pylist()):
            mask = pc.equal(tbl["_year"], pa.scalar(y, pa.int32()))
            part = tbl.filter(mask).drop_columns(tbl.schema.get_all_field_indices("_year"))
            sym = df["symbol"].iloc[0] if "symbol" in df.columns else Path(fp).stem.split("_")[0]
            out = Path(dst_dir) / f"symbol={sym}" / f"year={y}"
            out.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                part,
                out / f"part-{Path(fp).stem}.parquet",
                compression="zstd",
                use_dictionary=True,
                data_page_size=1 << 20,
            )  # 1MB pages
