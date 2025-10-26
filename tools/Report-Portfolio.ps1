function Report-Portfolio {
  [CmdletBinding()]
  param(
    [string]$Tickers   = "RELIANCE.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,TCS.NS",
    [string]$Start     = "2018-01-01",
    [int]   $Lookback  = 126,
    [int]   $TopN      = 3,
    [int]   $CostBps   = 10,
    [string]$Freq      = "ME",         # month-end
    [string]$Benchmark = "",           # e.g. "^NSEI"
    [switch]$Open
  )

  # Build Python list literal
  $parts = ($Tickers -split ',' | ForEach-Object { '"{0}"' -f $_.Trim() })
  $tickersPy = '[' + ($parts -join ', ') + ']'
  $benchLit  = if ($Benchmark) { '"{0}"' -f $Benchmark.Trim() } else { 'None' }

  @"
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers = $tickersPy
start   = "$Start"
lookback = $Lookback
top_n    = $TopN
cost_bps = $CostBps
rebalance_freq = "$Freq"
bench = $benchLit
outdir = Path("reports"); outdir.mkdir(parents=True, exist_ok=True)

def safe_pct(x):
    return x.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

# --- prices (force Adj Close availability) ---
raw = yf.download(tickers, start=start, progress=False, auto_adjust=False)
if isinstance(raw.columns, pd.MultiIndex):
    if "Adj Close" not in raw.columns.levels[0]:
        raise KeyError("Expected 'Adj Close' in yfinance output (multi-index).")
    data = raw["Adj Close"].copy()
else:
    if "Adj Close" not in raw.columns:
        raise KeyError("Expected 'Adj Close' in yfinance output (single-index).")
    one = raw["Adj Close"].to_frame()
    one.columns = [tickers[0] if len(tickers)==1 else one.columns[0]]
    data = one

data = data.dropna(how="all")
if data.empty:
    raise SystemExit("No adjusted close data returned — check tickers or start date.")

rets = safe_pct(data)

# --- rebalance (month-end that exist in index) ---
month_ends = rets.resample("ME").last().index
rebalance_dates = month_ends.intersection(rets.index)
if len(rebalance_dates)==0:
    raise SystemExit("No month-end rebalance dates found in returns index.")

weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
prev_hold = pd.Series(0.0, index=rets.columns)

for i, d in enumerate(rebalance_dates):
    end_idx = rets.index.get_loc(d)
    start_idx = max(0, end_idx - lookback)
    win = (1.0 + rets.iloc[start_idx:end_idx]).prod() - 1.0
    win = win.replace([np.inf, -np.inf], np.nan).fillna(-1e9)
    picks = win.nlargest(top_n).index
    w = pd.Series(0.0, index=rets.columns)
    if len(picks)>0:
        w.loc[picks] = 1.0/len(picks)

    start_loc = end_idx
    end_loc = len(rets)
    if i+1 < len(rebalance_dates):
        end_loc = rets.index.get_loc(rebalance_dates[i+1])
    weights.iloc[start_loc:end_loc] = w.values

    turnover = (w - prev_hold).abs().sum()
    if start_loc < len(rets):
        rets.iloc[start_loc] = rets.iloc[start_loc] - (cost_bps/1e4)*turnover
    prev_hold = w

port_ret = (weights * rets).sum(axis=1).fillna(0.0)
equity = (1.0 + port_ret).cumprod()
dd = equity/equity.cummax() - 1.0

years = max((equity.index[-1] - equity.index[0]).days/365.25, 1e-9)
total_ret = float(equity.iloc[-1] - 1.0)
cagr = float((equity.iloc[-1])**(1.0/years) - 1.0) if equity.iloc[-1] > 0 else 0.0
maxdd = float(dd.min())

# --- optional benchmark ---
bench_cagr = None
bench_label = None
if bench is not None:
    try:
        b = yf.download([bench], start=str(equity.index[0].date()), progress=False, auto_adjust=False)
        if isinstance(b.columns, pd.MultiIndex):
            bpx = b["Adj Close"].iloc[:,0].dropna()
        else:
            bpx = b["Adj Close"].dropna()
        bpx = bpx.reindex(equity.index).ffill().dropna()
        if len(bpx) > 1:
            beq = (bpx / bpx.iloc[0])
            byears = max((beq.index[-1] - beq.index[0]).days/365.25, 1e-9)
            bench_cagr = float(beq.iloc[-1]**(1.0/byears) - 1.0)
            bench_label = bench
    except Exception as e:
        bench_cagr = None
        bench_label = None

# --- save ---
stem = f"portfolio_{'-'.join([t.replace('.','_') for t in tickers])}"
csv_path = outdir / f"{stem}_equity.csv"
png_path = outdir / f"{stem}_equity.png"

df_out = pd.DataFrame({"equity": equity, "drawdown": dd, "port_ret": port_ret})
df_out.index.name = "date"
df_out.to_csv(csv_path, float_format="%.10f")

plt.figure(figsize=(10,5))
equity.plot(label="Portfolio")
if bench_label is not None:
    # normalize benchmark to 1 on first overlapping point
    # plot on same axis for visual comparison
    bnorm = (bpx / bpx.iloc[0]).reindex_like(equity, method="ffill")
    bnorm.plot(ax=plt.gca(), style="--", label=bench_label)
plt.title(f"Equity Curve ({top_n} of {len(tickers)}; lookback={lookback}d; cost={cost_bps}bps)")
plt.xlabel("Date"); plt.ylabel("Equity (base=1.0)")
plt.legend()
plt.tight_layout(); plt.savefig(png_path, dpi=140); plt.close()

print(json.dumps({
    "csv": str(csv_path),
    "png": str(png_path),
    "total_return": total_ret,
    "CAGR": cagr,
    "MaxDD": maxdd,
    "Benchmark": bench_label,
    "Benchmark_CAGR": bench_cagr,
    "first_date": str(equity.index[0].date()),
    "last_date": str(equity.index[-1].date()),
    "points": int(len(equity))
}, indent=2))
"@ | python -

  if ($Open) {
    # open PNG; then CSV
    if (Test-Path reports) {
      $png = Get-ChildItem reports -Filter *equity.png | Sort-Object LastWriteTime -Descending | Select-Object -First 1
      $csv = Get-ChildItem reports -Filter *equity.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
      if ($png) { Start-Process $png.FullName }
      if ($csv) { Start-Process $csv.FullName }
    }
  }
}