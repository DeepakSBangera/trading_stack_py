# bootstrap.py — generates Trading + Pricing + Econometrics project on F:\Projects\trading_stack_py
import os
from textwrap import dedent

# Repo root = parent of this file's folder (…/trading_stack_py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def write(path: str, content: str) -> None:
    """Create parent folders and write UTF-8 text."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dedent(content).strip() + "\n")


# ---------------- Files to create (kept quote-safe) ----------------

requirements = """
pandas==2.2.2
numpy==1.26.4
yfinance==0.2.52
pyyaml==6.0.2
plotly==5.22.0
streamlit==1.38.0
pyarrow
statsmodels==0.14.2
scipy==1.13.1
"""

readme = f"""\
# Trading · Pricing · Econometrics — Python-First (Low-Cost)

## What this repo gives you
- Trading: fetch data, compute indicators/signals, write a buy list, view in Streamlit.
- Pricing: estimate log–log price elasticity; write price recommendations.
- Econometrics: run ADF/KPSS diagnostics + ARIMA baseline forecasts.

## Quickstart (Windows, F:)
```bat
cd /d {BASE_DIR}
python -m venv .venv
.\\.venv\\Scripts\activate
pip install -r requirements.txt

python run_daily.py
streamlit run app\\dashboard.py
```

### Pricing (later)
Place `pricing\\data\\transactions.csv` with columns:
`date,product_id,price,qty,promo_flag,cost`
Then run:
```bat
python pricing\\run_pricing.py
```

### Econometrics (later)
Put `econo\\timeseries\\*.csv` with columns: `date,value`
Then run:
```bat
python run_econo.py
```
"""

config_yaml = """
data:
  source: yfinance
  start: 2018-01-01
  universe_csv: data/universe/watchlist.csv
  csv_dir: data/csv

signals:
  rule: R1_trend_breakout_obv
  params:
    sma_fast: 20
    sma_slow: 50
    lookback_high: 252
    obv_window: 10
    atr_window: 14
    atr_min_pct: 0.01
    atr_max_pct: 0.06
    avoid_spike_ret: 0.12
    rsi_window: 14
    rsi_buy: 55

risk:
  risk_per_trade_pct: 0.01
  atr_mult_stop: 2.5
  max_pos_value_pct: 0.08

scheduler:
  daily_run_hour_ist: 19
"""

watchlist_csv = """\
symbol
RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
NIFTYBEES.NS
"""

indicators_py = """\
# Minimal indicators used by the signal rule
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    \"\"\"Simple moving average.\"\"\"
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    \"\"\"Average True Range (volatility proxy).\"\"\"
    high, low_, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low_), (high - prev_close).abs(), (low_ - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    \"\"\"On-Balance Volume: cumulative volume with sign of price change.\"\"\"
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).fillna(0).cumsum()


def slope(s: pd.Series, n: int) -> pd.Series:
    \"\"\"Rolling linear slope over window n.\"\"\"
    x = np.arange(n)

    def _fit(y: np.ndarray) -> float:
        if len(y) < n or np.isnan(y).any():
            return np.nan
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]
        return float(m)

    return s.rolling(n).apply(lambda w: _fit(w.values), raw=False)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    \"\"\"Relative Strength Index.\"\"\"
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
"""

signals_py = """\
# Feature engineering + pluggable rules (select by name)
from __future__ import annotations

import pandas as pd

from .indicators import atr, obv, rsi, slope, sma


def compute_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    p = params or {}
    out = df.copy()
    sma_fast = int(p.get("sma_fast", 20))
    sma_slow = int(p.get("sma_slow", 50))
    obv_win = int(p.get("obv_window", 10))
    atr_win = int(p.get("atr_window", 14))
    look_hi = int(p.get("lookback_high", 252))
    rsi_win = int(p.get("rsi_window", 14))

    out["sma_f"] = sma(out["close"], sma_fast)
    out["sma_s"] = sma(out["close"], sma_slow)
    out["atr"] = atr(out, atr_win)
    out["obv"] = obv(out)
    out["obv_slope"] = slope(out["obv"], obv_win)
    out["hi52"] = out["close"].rolling(look_hi).max()
    out["atr_pct"] = out["atr"] / out["close"]
    out["ret_5d"] = out["close"].pct_change(5)
    out["rsi"] = rsi(out["close"], rsi_win)
    return out


def rule_R1_trend_breakout_obv(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    \"\"\"Trend + near 52W high + OBV slope + sane vol + no big recent spike.\"\"\"
    p = params or {}
    atr_min = float(p.get("atr_min_pct", 0.01))
    atr_max = float(p.get("atr_max_pct", 0.06))
    avoid_spike = float(p.get("avoid_spike_ret", 0.12))

    cross_up = (
        (features["sma_f"] > features["sma_s"])
        & (features["sma_f"].shift(1) <= features["sma_s"].shift(1))
    )
    breakout = features["close"] >= 0.995 * features["hi52"]
    obv_ok = features["obv_slope"] > 0
    vol_ok = features["atr_pct"].between(atr_min, atr_max)
    anti_spk = features["ret_5d"].fillna(0) < avoid_spike
    buy = cross_up & breakout & obv_ok & vol_ok & anti_spk
    return pd.DataFrame(
        {
            "buy": buy.astype(int),
            "score": (
                cross_up.astype(int)
                + breakout.astype(int)
                + obv_ok.astype(int)
                + vol_ok.astype(int)
                + anti_spk.astype(int)
            ),
        },
        index=features.index,
    )


def rule_R2_momo_rsi(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    \"\"\"Momentum + RSI filter + ATR band + anti-spike.\"\"\"
    p = params or {}
    atr_min = float(p.get("atr_min_pct", 0.01))
    atr_max = float(p.get("atr_max_pct", 0.06))
    avoid_spike = float(p.get("avoid_spike_ret", 0.12))
    rsi_buy = float(p.get("rsi_buy", 55))

    trend_up = features["sma_f"] > features["sma_s"]
    rsi_ok = features["rsi"] > rsi_buy
    vol_ok = features["atr_pct"].between(atr_min, atr_max)
    anti_spk = features["ret_5d"].fillna(0) < avoid_spike
    buy = trend_up & rsi_ok & vol_ok & anti_spk
    return pd.DataFrame(
        {
            "buy": buy.astype(int),
            "score": (
                trend_up.astype(int)
                + rsi_ok.astype(int)
                + vol_ok.astype(int)
                + anti_spk.astype(int)
            ),
        },
        index=features.index,
    )


RULES = {
    "R1_trend_breakout_obv": rule_R1_trend_breakout_obv,
    "R2_momo_rsi": rule_R2_momo_rsi,
}


def make_signals(
    df: pd.DataFrame,
    params: dict,
    rule_name: str = "R1_trend_breakout_obv",
) -> pd.DataFrame:
    feats = compute_features(df, params)
    rule_fn = RULES.get(rule_name, rule_R1_trend_breakout_obv)
    sigs = rule_fn(feats, params)
    return feats.join(sigs)
"""

data_io_py = """\
# Data adapters (yfinance or CSV) + lightweight I/O helpers
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_ohlcv(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False, threads=False)
    if df.empty:
        return (
            pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
            .set_index(pd.DatetimeIndex([]))
        )
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df


def fetch_ohlcv_from_csv(symbol: str, csv_dir: str = "data/csv") -> pd.DataFrame:
    fp = Path(csv_dir) / f"{symbol}.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.DataFrame()
    return df[cols]


def save_parquet(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_watchlist(csv_path: str) -> list[str]:
    return pd.read_csv(csv_path)["symbol"].dropna().astype(str).tolist()
"""

portfolio_py = """\
# Toy position sizing + paper portfolio snapshotting (hook up later if needed)
from __future__ import annotations

import pandas as pd


def size_position(equity: float, price: float, atr: float, cfg: dict) -> int:
    risk_amt = equity * cfg["risk_per_trade_pct"]
    stop = price - cfg["atr_mult_stop"] * atr
    per_share_risk = max(price - stop, 0.01)
    qty_risk = int(risk_amt // per_share_risk)
    max_val = equity * cfg["max_pos_value_pct"]
    qty_val = int(max_val // max(price, 0.01))
    return max(0, min(qty_risk, qty_val))


def update_positions(
    buylist: pd.DataFrame,
    prices_next_open: pd.Series,
    positions_csv: str,
    portfolio_csv: str,
    cfg: dict,
    init_equity: float = 1_000_000.0,
):
    try:
        positions = pd.read_csv(positions_csv)
    except FileNotFoundError:
        positions = pd.DataFrame(columns=["symbol", "qty", "avg_price"])
    try:
        portfolio = pd.read_csv(portfolio_csv)
    except FileNotFoundError:
        portfolio = pd.DataFrame(columns=["date", "equity"])

    equity = portfolio["equity"].iloc[-1] if not portfolio.empty else init_equity
    held = set(positions["symbol"])
    entries = []
    for sym in buylist["symbol"]:
        if sym in held:
            continue
        entry_price = float(buylist.set_index("symbol").loc[sym, "close"])
        atr = float(buylist.set_index("symbol").loc[sym, "atr"])
        if entry_price > 0:
            qty = size_position(equity, entry_price, atr, cfg)
            if qty > 0:
                entries.append({"symbol": sym, "qty": qty, "avg_price": entry_price})

    if entries:
        positions = pd.concat([positions, pd.DataFrame(entries)], ignore_index=True)

    snap = {"date": pd.Timestamp.today().date().isoformat(), "equity": equity}
    portfolio = pd.concat([portfolio, pd.DataFrame([snap])], ignore_index=True)

    positions.to_csv(positions_csv, index=False)
    portfolio.to_csv(portfolio_csv, index=False)
    return positions, portfolio
"""

run_daily_py = """\
# Daily trading driver: fetch -> features -> signals -> today's BUY list CSV
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import yaml

from src import data_io
from src.signals import make_signals

with open("config/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

os.makedirs("reports", exist_ok=True)
os.makedirs("data/prices", exist_ok=True)
os.makedirs("data/features", exist_ok=True)

symbols = data_io.load_watchlist(CFG["data"]["universe_csv"])
start = CFG["data"].get("start", "2018-01-01")
source = CFG["data"].get("source", "yfinance")  # "yfinance" or "csv"
csv_dir = CFG["data"].get("csv_dir", "data/csv")  # used only if source == "csv"

rule_name = CFG["signals"].get("rule", "R1_trend_breakout_obv")
params = CFG["signals"].get("params", {})

all_buys = []
for sym in symbols:
    # Choose data source
    if source == "yfinance":
        df = data_io.fetch_ohlcv(sym, start=start)
    elif source == "csv":
        df = data_io.fetch_ohlcv_from_csv(sym, csv_dir=csv_dir)
    else:
        raise ValueError(f"Unknown data source: {source}")

    if df.empty:
        print(f"[WARN] No data for {sym}")
        continue

    # Cache raw & features (parquet)
    df.to_parquet(f"data/prices/{sym}.parquet")
    sigdf = make_signals(df, params, rule_name)
    sigdf.to_parquet(f"data/features/{sym}.parquet")

    last = sigdf.iloc[-1].copy()
    if int(last.get("buy", 0)) == 1:
        all_buys.append(
            {
                "symbol": sym,
                "close": float(last["close"]),
                "atr": float(last["atr"]),
                "score": int(last["score"]),
            }
        )

today = datetime.today().date().isoformat()
buylist_path = f"reports/buylist_{today}.csv"
(
    pd.DataFrame(all_buys)
    .sort_values(by=["score", "symbol"], ascending=[False, True])
    .to_csv(buylist_path, index=False)
)
print(f"Wrote {buylist_path}")
"""

dashboard_py = """\
# Streamlit dashboard: shows latest BUY list and (later) portfolio
from __future__ import annotations

import glob
import os

import pandas as pd
import streamlit as st
import yaml

st.title("Trading Dashboard — Signals & Portfolio (Milestone-1)")

# Show config summary
try:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    rule = CFG["signals"].get("rule", "?")
    source = CFG["data"].get("source", "?")
    st.caption(f"Using rule: {rule}, data source: {source}")
except Exception:
    st.caption("Could not read config/config.yaml")

files = sorted(glob.glob("reports/buylist_*.csv"))
if files:
    latest = files[-1]
    st.subheader(f"Today's BUY list: {os.path.basename(latest)}")
    buylist = pd.read_csv(latest)
    st.dataframe(buylist)
else:
    st.info("Run python run_daily.py to generate today's BUY list.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Positions (paper)")
    try:
        pos = pd.read_csv("db/positions.csv")
        st.dataframe(pos)
    except FileNotFoundError:
        st.write("No positions yet.")

with col2:
    st.subheader("Portfolio snapshots")
    try:
        pf = pd.read_csv("db/portfolio.csv")
        st.line_chart(pf.set_index("date")["equity"])
    except Exception:
        st.write("No portfolio snapshots yet.")
"""

pricing_models = """\
# Pricing model: log-log elasticity per product using OLS
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_loglog_elasticity(
    df: pd.DataFrame,
    price_col: str,
    qty_col: str,
    extra_cols: list[str] | None = None,
):
    extra_cols = extra_cols or []
    d = df[[price_col, qty_col] + extra_cols].dropna()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)].copy()
    d["ln_qty"] = np.log(d[qty_col])
    d["ln_price"] = np.log(d[price_col])
    X = d[["ln_price"] + extra_cols]
    X = sm.add_constant(X)
    model = sm.OLS(d["ln_qty"], X, hasconst=True).fit()
    beta = model.params.get("ln_price", np.nan)  # elasticity (often negative)
    return beta, model
"""

pricing_reco = """\
# Convert elasticity to recommended price within +/-band
from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_price_from_elasticity(
    cost: float,
    elasticity_abs: float | None,
    p_min: float | None = None,
    p_max: float | None = None,
) -> float:
    if elasticity_abs is None or np.isnan(elasticity_abs):
        return np.nan
    if elasticity_abs <= 1:
        p_star = p_max if p_max is not None else cost * 1.05
    else:
        p_star = cost * elasticity_abs / (elasticity_abs - 1.0)
    if p_min is not None:
        p_star = max(p_star, p_min)
    if p_max is not None:
        p_star = min(p_star, p_max)
    return float(p_star)


def apply_recommendations(
    df: pd.DataFrame,
    price_col: str,
    qty_col: str,
    cost_col: str,
    elasticity_abs: float,
    pct_band: float = 0.1,
) -> dict:
    current_price = df[price_col].median()
    c = df[cost_col].median() if cost_col in df else current_price * 0.6
    p_min = current_price * (1 - pct_band)
    p_max = current_price * (1 + pct_band)
    p_star = optimal_price_from_elasticity(c, elasticity_abs, p_min, p_max)
    return {
        "current": current_price,
        "cost": c,
        "recommended": p_star,
        "band": (p_min, p_max),
    }
"""

run_pricing = """\
# Batch pricing: estimate elasticity and write recommendations CSV
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from pricing.models import fit_loglog_elasticity
from pricing.recommend import apply_recommendations

INFILE = "pricing/data/transactions.csv"


def main() -> None:
    if not os.path.exists(INFILE):
        print(f"[INFO] No pricing data at {INFILE}. Add a CSV to get recommendations.")
        return
    df = pd.read_csv(INFILE)
    out_rows = []
    for pid, d in df.groupby("product_id"):
        try:
            beta, _ = fit_loglog_elasticity(
                d,
                price_col="price",
                qty_col="qty",
                extra_cols=["promo_flag"],
            )
            eps = abs(beta)
            rec = apply_recommendations(
                d,
                price_col="price",
                qty_col="qty",
                cost_col="cost",
                elasticity_abs=eps,
                pct_band=0.1,
            )
            out_rows.append(
                {
                    "product_id": pid,
                    "elasticity_abs": eps,
                    "current_price": rec["current"],
                    "cost_proxy": rec["cost"],
                    "rec_price": rec["recommended"],
                    "band_lo": rec["band"][0],
                    "band_hi": rec["band"][1],
                }
            )
        except Exception as e:  # noqa: BLE001
            out_rows.append({"product_id": pid, "error": str(e)})
    out = pd.DataFrame(out_rows)
    os.makedirs("pricing", exist_ok=True)
    outfile = f"pricing/recommendations_{date.today().isoformat()}.csv"
    out.to_csv(outfile, index=False)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
"""

econo_diag = """\
# ADF and KPSS stationarity tests
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(s: pd.Series) -> dict:
    s = s.dropna()
    res = adfuller(s, autolag="AIC")
    return {"stat": res[0], "pvalue": res[1], "lags": res[2], "nobs": res[3]}


def kpss_test(s: pd.Series) -> dict:
    s = s.dropna()
    stat, pval, lags, _ = kpss(s, nlags="auto")
    return {"stat": stat, "pvalue": pval, "lags": lags}
"""

econo_forecast = """\
# Baseline ARIMA forecast helper
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def arima_forecast(s: pd.Series, order=(1, 1, 1), steps: int = 12):
    s = s.dropna()
    model = sm.tsa.ARIMA(s, order=order)
    fit = model.fit()
    f = fit.get_forecast(steps=steps)
    return fit, f.summary_frame()
"""

run_econo = """\
# Run diagnostics + ARIMA forecasts for all CSVs in econo/timeseries
from __future__ import annotations

import glob
import os
from datetime import date

import pandas as pd

from econo.diagnostics import adf_test, kpss_test
from econo.forecast import arima_forecast


def main() -> None:
    files = glob.glob("econo/timeseries/*.csv")
    if not files:
        print(
            "[INFO] No time series under econo/timeseries. "
            "Add CSVs with columns: date,value"
        )
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
            diag_rows.append(
                {"series": name, **adf, **{f"kpss_{k}": v for k, v in kps.items()}}
            )
        except Exception as e:  # noqa: BLE001
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
        outfile = f"econo/forecasts_{date.today().isoformat()}.csv"
        fc.to_csv(outfile, index=False)
        print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
"""

# ---------------- Write all files ----------------

paths = {
    "requirements.txt": requirements,
    "README.md": readme,
    "config/config.yaml": config_yaml,
    "data/universe/watchlist.csv": watchlist_csv,
    "src/indicators.py": indicators_py,
    "src/signals.py": signals_py,
    "src/data_io.py": data_io_py,
    "src/portfolio.py": portfolio_py,
    "run_daily.py": run_daily_py,
    "app/dashboard.py": dashboard_py,
    "pricing/models.py": pricing_models,
    "pricing/recommend.py": pricing_reco,
    "pricing/run_pricing.py": run_pricing,
    "econo/diagnostics.py": econo_diag,
    "econo/forecast.py": econo_forecast,
    "run_econo.py": run_econo,
}

for rel, content in paths.items():
    write(os.path.join(BASE_DIR, rel), content)

for d in [
    "data/prices",
    "data/features",
    "db",
    "reports",
    "pricing/data",
    "econo/timeseries",
    "data/csv",
]:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

print(f"Project created at: {BASE_DIR}")
print("Next steps:")
print(f"  1) cd /d {BASE_DIR}")
print("  2) python -m venv .venv && .\\.venv\\Scripts\\activate")
print("  3) pip install -r requirements.txt")
print("  4) python run_daily.py  (then)  streamlit run app\\dashboard.py")
