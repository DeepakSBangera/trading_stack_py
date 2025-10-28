import pathlib as P
import sys

import pandas as pd


def find_datetime_col(df):
    # common candidates
    for c in ["__index_level_0__", "dt", "timestamp", "time"]:
        if c in df.columns:
            dtype = df[c].dtype
            if str(dtype).startswith("datetime64"):
                return c
            # try coercion
            try:
                pd.to_datetime(df[c], errors="raise", utc=True)  # validate only
                return c
            except Exception:
                pass
    # no candidate
    return None


def main():
    if len(sys.argv) < 3:
        print("usage: python add_date_if_missing.py <in.parquet> <out.parquet>")
        sys.exit(2)
    src = P.Path(sys.argv[1])
    dst = P.Path(sys.argv[2])

    df = pd.read_parquet(src)
    cols_lower = [c.lower() for c in df.columns]
    if "date" in cols_lower:
        # already fine; standardize tz to UTC if datetime
        real_date = df.columns[cols_lower.index("date")]
        try:
            if not str(df[real_date].dtype).startswith("datetime64"):
                df[real_date] = pd.to_datetime(df[real_date], utc=True, errors="coerce")
            else:
                # localize to UTC if tz-naive
                if getattr(df[real_date].dt, "tz", None) is None:
                    df[real_date] = df[real_date].dt.tz_localize("UTC")
        except Exception:
            pass
        df.to_parquet(dst, index=False)
        print("OK: date already present → wrote", dst)
        return

    # no date column — try to construct it
    cand = find_datetime_col(df)
    if cand is not None:
        s = pd.to_datetime(df[cand], utc=True, errors="coerce")
        if s.isna().all():
            print(
                "FAIL: candidate column existed but could not be parsed to datetime:",
                cand,
            )
            sys.exit(3)
        df.insert(0, "date", s)
        # drop special index column if that’s what we used
        if cand == "__index_level_0__":
            df = df.drop(columns=[cand])
        df.to_parquet(dst, index=False)
        print(f"OK: created 'date' from {cand} → wrote {dst}")
        return

    # last resort: if the *pandas index* carries timestamps (rare in saved parquet)
    if isinstance(df.index, pd.DatetimeIndex):
        s = (
            pd.DatetimeIndex(df.index).tz_convert("UTC")
            if df.index.tz is not None
            else df.index.tz_localize("UTC")
        )
        df = df.reset_index(drop=False).rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df.to_parquet(dst, index=False)
        print("OK: created 'date' from pandas index → wrote", dst)
        return

    print("FAIL: could not find/construct a datetime 'date' column")
    sys.exit(4)


if __name__ == "__main__":
    main()
