from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BT_DIR = ROOT / "reports" / "backtests"
OUT = ROOT / "reports" / "wk4_voltarget_stops.csv"
CFG = ROOT / "config" / "w4_risk.yaml"


@dataclass
class RiskCfg:
    target_vol_ann: float = 0.20
    lookback_days: int = 20
    min_mult: float = 0.50
    max_mult: float = 1.50
    dd_warn: float = 0.06
    dd_halt: float = 0.12
    throttle_warn: float = 0.75
    throttle_halt: float = 0.50


def load_cfg(p: Path) -> RiskCfg:
    with open(p, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    tl = y.get("throttle_levels") or {}
    return RiskCfg(
        target_vol_ann=float(y.get("target_vol_ann", 0.20)),
        lookback_days=int(y.get("lookback_days", 20)),
        min_mult=float(y.get("min_mult", 0.50)),
        max_mult=float(y.get("max_mult", 1.50)),
        dd_warn=float(y.get("dd_warn", 0.06)),
        dd_halt=float(y.get("dd_halt", 0.12)),
        throttle_warn=float(tl.get("warn", 0.75)),
        throttle_halt=float(tl.get("halt", 0.50)),
    )


def latest_run_dir(base: Path) -> Path:
    runs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No backtest runs under reports/backtests. Run: python -m scripts.w2_backtest")
    return runs[-1]


def load_inputs(run: Path) -> tuple[pd.Series, pd.DataFrame]:
    # equity curve is required; prices optional (for dates)
    eq = pd.read_csv(run / "equity_curve.csv", index_col=0)
    eq.index = pd.to_datetime(eq.index)
    if eq.shape[1] == 1:
        eq = eq.iloc[:, 0].rename("equity")
    else:
        eq = eq["equity"]
    px_path = run / "px.csv"
    px = pd.read_csv(px_path, index_col=0) if px_path.exists() else pd.DataFrame(index=eq.index)
    px.index = pd.to_datetime(px.index)
    return eq, px


def realized_vol_daily(ret: pd.Series, lb: int) -> pd.Series:
    # daily realized vol (rolling) using population std
    return ret.rolling(lb, min_periods=max(5, lb // 2)).std(ddof=0)


def main() -> None:
    cfg = load_cfg(CFG)
    run = latest_run_dir(BT_DIR)
    eq, _px = load_inputs(run)

    # Recreate daily portfolio returns from equity
    port_ret = eq.pct_change().fillna(0.0)

    # Rolling daily vol -> annualized target multiplier
    rv_d = realized_vol_daily(port_ret, cfg.lookback_days)
    rv_ann = rv_d * (252.0**0.5)

    # desired multiplier so ann vol -> target
    mult = (cfg.target_vol_ann / rv_ann).clip(lower=cfg.min_mult, upper=cfg.max_mult)
    mult = mult.fillna(1.0)

    # drawdown throttle on equity
    roll_peak = eq.cummax()
    dd = 1.0 - (eq / roll_peak).clip(upper=1.0)  # e.g., 0.07 means -7% from peak

    throttle = pd.Series(1.0, index=eq.index, name="throttle")
    throttle.loc[dd >= cfg.dd_warn] = cfg.throttle_warn
    throttle.loc[dd >= cfg.dd_halt] = cfg.throttle_halt

    # combined risk multiplier
    risk_mult = (mult * throttle).clip(lower=cfg.min_mult / 2, upper=cfg.max_mult)

    # produce a risk-scaled equity (informational)
    scaled_ret = port_ret * risk_mult.shift(1).fillna(1.0)
    scaled_eq = (1.0 + scaled_ret).cumprod()

    out = pd.DataFrame(
        {
            "equity": eq,
            "ret": port_ret,
            "roll_vol_daily": rv_d,
            "roll_vol_ann": rv_ann,
            "vt_multiplier": mult,
            "drawdown": dd,
            "throttle": throttle,
            "risk_multiplier": risk_mult,
            "equity_scaled": scaled_eq,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=True)
    print(f"Wrote {OUT}")
    print("Preview:")
    print(out.tail(5).to_string(index=True))


if __name__ == "__main__":
    main()
