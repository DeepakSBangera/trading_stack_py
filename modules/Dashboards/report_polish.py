import glob
import os

import pandas as pd


def main():
    os.makedirs("reports", exist_ok=True)
    rows = []

    # 1) Latest CLI runs
    for f in glob.glob("reports/run_*.csv"):
        ticker = os.path.basename(f)[4:-4]  # run_<ticker>.csv
        try:
            df = pd.read_csv(f)
            rows.append({"type": "run", "ticker": ticker, "path": f, "rows": len(df)})
        except Exception:
            pass

    # 2) Grid searches
    for f in glob.glob("reports/grid_*.csv"):
        try:
            df = pd.read_csv(f)
            top = df.sort_values(["Sharpe", "CAGR"], ascending=[False, False]).head(1).to_dict("records")[0]
            rows.append(
                {
                    "type": "grid",
                    "ticker": os.path.basename(f)[5:-4],
                    "path": f,
                    "best_fast": int(top["fast"]),
                    "best_slow": int(top["slow"]),
                    "best_Sharpe": float(top["Sharpe"]),
                    "best_CAGR": float(top["CAGR"]),
                }
            )
        except Exception:
            pass

    # 3) Walk-forward
    for f in glob.glob("reports/walkforward_*.csv"):
        try:
            df = pd.read_csv(f)
            psr_med = float(df["PSR"].median()) if "PSR" in df.columns and not df.empty else None
            rows.append(
                {
                    "type": "walkforward",
                    "ticker": os.path.basename(f)[12:-4],
                    "path": f,
                    "windows": len(df),
                    "PSR_median": psr_med,
                }
            )
        except Exception:
            pass

    out = pd.DataFrame(rows)
    outp = os.path.join("reports", "build_summary.csv")
    out.to_csv(outp, index=False)
    print(f"Saved: {outp}")
    if not out.empty:
        print(out.to_string(index=False))
    else:
        print("No artifacts found to summarize.")


if __name__ == "__main__":
    main()
