import pathlib

import pandas as pd

root = pathlib.Path(r"F:\Projects\trading_stack_py")
candidates = [
    root / "reports/wk6_weights_capped.csv",
    root / "reports/wk11_blend_targets.csv",
] + sorted(root.glob("reports/weights_*.csv"))
src = next((p for p in candidates if p.exists()), None)
if not src:
    print("No weights source found; leaving stub.")
else:
    df = pd.read_csv(src)
    # Expect columns: date, symbol, weight (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(
        columns={
            cols.get("date", "date"): "date",
            cols.get("symbol", "symbol"): "symbol",
            cols.get("weight", "weight"): "weight",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"])
    df["weight_prev"] = df.groupby("symbol")["weight"].shift(1)
    df["d_weight"] = (df["weight"] - df["weight_prev"]).fillna(0.0)
    df["turnover_abs"] = df["d_weight"].abs()
    out = df[
        ["date", "symbol", "weight_prev", "weight", "d_weight", "turnover_abs"]
    ].rename(columns={"weight": "weight_curr"})
    out.to_csv(root / "reports/wk3_turnover_profile.csv", index=False)
    print("Wrote reports/wk3_turnover_profile.csv from", src.name)
