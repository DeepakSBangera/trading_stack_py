from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config" / "capacity_policy.yaml"

sys.path.append(str(ROOT))
from modules.RiskSizing.turnover import (
    annualized_churn,
    compute_turnover,
    liquidity_screens,
    pretrade_violations,
)


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def open_win(p: Path):
    if sys.platform.startswith("win") and p.exists():
        os.startfile(p)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def safe_write_csv(df: pd.DataFrame, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(base_path, index=False)
        return base_path
    except PermissionError:
        stamped = base_path.with_name(f"{base_path.stem}_{_timestamp()}{base_path.suffix}")
        df.to_csv(stamped, index=False)
        return stamped


def ensure_prereqs():
    REPORTS.mkdir(parents=True, exist_ok=True)
    need = [REPORTS / "positions_daily.parquet", REPORTS / "adv_value.parquet", CONFIG]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(miss))


def main():
    ensure_prereqs()
    policy = load_yaml(CONFIG)
    positions = pd.read_parquet(REPORTS / "positions_daily.parquet")
    adv = pd.read_parquet(REPORTS / "adv_value.parquet")

    # Prefer throttled orders if available
    orders_th = REPORTS / "orders_daily_throttled.parquet"
    orders_p = orders_th if orders_th.exists() else (REPORTS / "orders_daily.parquet")
    orders = pd.read_parquet(orders_p) if orders_p.exists() else None

    # Turnover
    turnover_daily = compute_turnover(positions[["date", "ticker", "weight"]])
    path_turn = safe_write_csv(turnover_daily, REPORTS / "turnover_profile.csv")
    ann = round(float(annualized_churn(turnover_daily)), 2)

    # Liquidity
    liq = liquidity_screens(
        holdings_value=positions[["date", "ticker", "position_value", "list_tier"]],
        adv_value=adv[["date", "ticker", "adv_value"]],
        policy=policy,
    )
    path_liq = safe_write_csv(liq, REPORTS / "liquidity_screens.csv")

    # Pre-trade
    if orders is None:
        pos = positions.sort_values(["ticker", "date"]).copy()
        pos["prev_val"] = pos.groupby("ticker")["position_value"].shift(1).fillna(0.0)
        pos["order_value"] = (pos["position_value"] - pos["prev_val"]).abs()
        orders = pos[["date", "ticker", "order_value", "list_tier", "port_value"]]
    pv = pretrade_violations(orders_value=orders, policy=policy)
    path_pv = safe_write_csv(pv, REPORTS / "pretrade_violations.csv")

    print(
        json.dumps(
            {
                "artifacts": [str(path_turn), str(path_liq), str(path_pv)],
                "orders_source": str(orders_p),
                "annualized_churn_pct": ann,
                "liquidity_violations": int(liq["violation"].sum()),
                "pretrade_violations": int(pv["violation"].sum()),
            },
            indent=2,
        )
    )

    open_win(path_turn)
    open_win(path_liq)
    open_win(path_pv)
    open_win(CONFIG)


if __name__ == "__main__":
    main()
