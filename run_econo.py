# Run diagnostics + ARIMA forecasts for all CSVs in econo/timeseries
import os, glob
import pandas as pd
from datetime import date
from econo.diagnostics import adf_test, kpss_test
from econo.forecast import arima_forecast

def main():
    files = glob.glob("econo/timeseries/*.csv")
    if not files:
        print("[INFO] No time series under econo/timeseries. Add CSVs with columns: date,value")
        return
    diag_rows, fc_frames = [], []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
        s = df.iloc[:, 0] if df.shape[1] == 1 else df["value"]
        try:
            adf = adf_test(s)
            kps = kpss_test(s)
            diag_rows.append({"series": name, **adf, **{f"kpss_{k}": v for k, v in kps.items()}})
        except Exception as e:
            diag_rows.append({"series": name, "error": str(e)})
        try:
            fit, sf = arima_forecast(s)
            sf["series"] = name
            sf["step"] = range(1, len(sf) + 1)
            fc_frames.append(sf.reset_index(drop=True))
        except Exception:
            pass
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv("econo/diagnostics.csv", index=False)
        print("Wrote econo/diagnostics.csv")
    if fc_frames:
        fc = pd.concat(fc_frames, ignore_index=True)
        fc.to_csv(f"econo/forecasts_{date.today().isoformat()}.csv", index=False)
        print(f"Wrote econo/forecasts_{date.today().isoformat()}.csv")

if __name__ == "__main__":
    main()
