import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"F:\Projects\trading_stack_py")
DATA = ROOT / "data" / "csv"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "policy_w1.yaml"
REPORTS.mkdir(parents=True, exist_ok=True)

OUT = REPORTS / "wk1_entry_exit_baseline.csv"


def write_empty_scaffold(reason: str) -> None:
    cols = [
        "date",
        "ticker",
        "score",
        "signal",
        "action",
        "list",
        "stop_atr_x",
        "position_w",
        "notes",
    ]
    pd.DataFrame(columns=cols).to_csv(OUT, index=False)
    print(f"[W1] Wrote empty scaffold to {OUT} ({reason}).")


def read_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


# policy defaults if file missing
cfg = read_yaml(CONFIG) if CONFIG.exists() else {}
cfg.setdefault("fallback_windows", {"m12": 252, "m6": 126, "m1": 21})
cfg.setdefault("vol_lookback_days", 20)
cfg.setdefault("atr_lookback_days", 14)
cfg.setdefault("weights", {"z12_1": 0.5, "z6m": 0.3, "z1m": 0.2, "zvol": -0.3})
cfg.setdefault("top_n", 30)
cfg.setdefault("per_name_cap", 0.06)
cfg.setdefault("list3_file", "config/lists/list3_quality.csv")
cfg.setdefault("list2_file", "config/lists/list2_conviction.csv")

LB12 = int(cfg["fallback_windows"]["m12"])
LB6 = int(cfg["fallback_windows"]["m6"])
LB1 = int(cfg["fallback_windows"]["m1"])
VOL_LB = int(cfg["vol_lookback_days"])
ATR_LB = int(cfg["atr_lookback_days"])


def load_optional_list(rel_path: str) -> set[str]:
    try:
        pth = ROOT / rel_path
        if rel_path and pth.exists():
            df = pd.read_csv(pth, dtype=str)
            df.columns = [c.strip().lower() for c in df.columns]
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            return set(df[col].astype(str).str.upper())
    except Exception:
        return set()
    return set()


list3 = load_optional_list(cfg.get("list3_file"))
list2 = load_optional_list(cfg.get("list2_file"))


def compute_metrics_from_df(df: pd.DataFrame) -> dict | None:
    """Needs date + (adj close or close); high/low/close for ATR if available."""
    if "date" not in df.columns:
        return None

    if "adj close" in df.columns:
        px_col = "adj close"
    elif "close" in df.columns:
        px_col = "close"
    else:
        return None

    tmp = df.dropna(subset=["date"]).copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"]).sort_values("date")

    px = pd.to_numeric(tmp[px_col], errors="coerce").dropna()
    if px.empty:
        return None

    def ret_n(n: int) -> float:
        if len(px) <= n or px.iloc[-n] == 0:
            return float("nan")
        return px.iloc[-1] / px.iloc[-n] - 1.0

    r_252 = ret_n(LB12)
    r_21 = ret_n(LB1)

    if pd.isna(r_252) or pd.isna(r_21) or (1 + r_21) == 0:
        r_12_1 = float("nan")
    else:
        r_12_1 = (1 + r_252) / (1 + r_21) - 1.0

    r_6m = ret_n(LB6)
    r_1m = r_21

    rets = px.pct_change().dropna()
    vol = float("nan")
    if len(rets) >= max(VOL_LB, 5):
        vol = rets.tail(VOL_LB).std(ddof=0) * math.sqrt(252)

    atr = float("nan")
    if {"high", "low", "close"}.issubset(tmp.columns):
        high = pd.to_numeric(tmp["high"], errors="coerce")
        low = pd.to_numeric(tmp["low"], errors="coerce")
        close = pd.to_numeric(tmp["close"], errors="coerce")
        a = (high - low).abs()
        b = (high - close.shift()).abs()
        d = (low - close.shift()).abs()
        tr = pd.concat([a, b, d], axis=1).max(axis=1)
        tr = tr.dropna()
        if len(tr) >= ATR_LB:
            atr = tr.rolling(ATR_LB).mean().iloc[-1]

    return {"r12_1": r_12_1, "r6m": r_6m, "r1m": r_1m, "vol": vol, "atr": atr}


rows: list[dict] = []
files = glob.glob(str(DATA / "*.csv"))
print(f"[W1] Found {len(files)} CSV file(s) under {DATA}")

for path in files:
    tkr = Path(path).stem.upper()
    try:
        df_file = pd.read_csv(path)
        df_file.columns = [c.strip().lower() for c in df_file.columns]
        met = compute_metrics_from_df(df_file)
        if met is None:
            continue
        met["ticker"] = tkr
        rows.append(met)
    except Exception:
        continue

if not rows:
    write_empty_scaffold("no usable rows (check csv presence / headers)")
    raise SystemExit(0)

raw = pd.DataFrame(rows)

needed = {"r12_1", "r6m", "r1m", "vol"}
missing = needed - set(raw.columns)
if missing:
    write_empty_scaffold(f"missing required cols after load: {missing}")
    raise SystemExit(0)

raw = raw.dropna(subset=["r12_1", "r6m", "r1m", "vol"])
if raw.empty:
    write_empty_scaffold("all tickers had insufficient history for momentum/vol")
    raise SystemExit(0)


def zscore(x: pd.Series) -> pd.Series:
    s = x.std(ddof=0)
    if s in (0, float("nan")) or pd.isna(s):
        return x * 0
    return (x - x.mean()) / s


raw["z12_1"] = zscore(raw["r12_1"])
raw["z6m"] = zscore(raw["r6m"])
raw["z1m"] = zscore(raw["r1m"])
raw["zvol"] = zscore(raw["vol"])

w = cfg["weights"]
raw["score"] = (
    w.get("z12_1", 0.5) * raw["z12_1"]
    + w.get("z6m", 0.3) * raw["z6m"]
    + w.get("z1m", 0.2) * raw["z1m"]
    + w.get("zvol", -0.3) * raw["zvol"]
)
raw = raw.sort_values("score", ascending=False)

top_n = int(cfg["top_n"])
sel: list[str] = []
added: set[str] = set()


def which_list(t: str) -> str:
    if t in list3:
        return "List-3"
    if t in list2:
        return "List-2"
    return "List-1"


# List-3 first
for t in [t for t in raw["ticker"] if t in list3]:
    sel.append(t)
    added.add(t)
# then List-2
for t in [t for t in raw["ticker"] if t in list2 and t not in added]:
    sel.append(t)
    added.add(t)
# fill with List-1 by score
for t in raw["ticker"]:
    if len(sel) >= top_n:
        break
    if t not in added:
        sel.append(t)
        added.add(t)

chosen = raw.set_index("ticker").loc[sel].reset_index()

inv_vol = 1.0 / chosen["vol"].replace(0, np.nan)
wts = inv_vol / inv_vol.sum()
wts = wts.clip(upper=float(cfg["per_name_cap"]))
wts = wts / wts.sum()

today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
out_df = pd.DataFrame(
    {
        "date": today,
        "ticker": chosen["ticker"],
        "score": chosen["score"].round(4),
        "signal": np.where(chosen["score"] > 0, "UP", "DOWN"),
        "action": np.where(chosen["score"] > 0, "BUY", "AVOID"),
        "list": [which_list(t) for t in chosen["ticker"]],
        "stop_atr_x": np.round(chosen["atr"].fillna(0.0) * 3.0, 2),
        "position_w": wts.round(4),
        "notes": "",
    }
)

out_df.to_csv(OUT, index=False)
print(f"[W1] Wrote {OUT} with {len(out_df)} row(s).")
print(f"[W1] Weight sum ≈ {float(out_df['position_w'].sum()):.6f}")
