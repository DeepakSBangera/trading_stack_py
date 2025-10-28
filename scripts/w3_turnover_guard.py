from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BT_BASE = ROOT / "reports" / "backtests"
POLICY = ROOT / "reports" / "wk3_turnover_policy.json"
OUT = ROOT / "reports" / "wk3_turnover_violations.csv"


def _as_bps(x: float | int | None) -> float | None:
    """Normalize a threshold to basis points (≤1.5 treated as fraction, else already bps)."""
    if x is None:
        return None
    x = float(x)
    return x * 10_000.0 if x <= 1.5 else x


def load_policy(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    caps = raw.get("policy_caps", raw)
    return {
        "avg_turnover_bps": _as_bps(caps.get("target_avg_turnover_per_rebalance")),
        "p95_turnover_bps": _as_bps(caps.get("target_p95_turnover")),
        "max_turnover_bps": _as_bps(caps.get("max_turnover_bps")),  # optional
        "advisory_turnover_bps": _as_bps(caps.get("advisory_turnover_bps")),  # optional
        "max_weekly_adds": caps.get("max_weekly_adds"),  # optional
        "max_weekly_drops": caps.get("max_weekly_drops"),  # optional
    }


def latest_backtest_dir(base: Path) -> Path:
    runs = sorted(
        [d for d in base.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime
    )
    if not runs:
        raise SystemExit("No backtest runs found. Run: python -m scripts.w2_backtest")
    return runs[-1]


def _load_turnover_csv(tfile: Path) -> pd.DataFrame:
    """
    Load turnover CSV with flexible column names; return columns:
    ['rebalance_date','turnover_bps','adds','drops'].
    """
    df = pd.read_csv(tfile)
    orig_cols = list(df.columns)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # --- locate date column ---
    date_candidates = ["rebalance_date", "date", "dt", "rebal_date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        if df.index.name and str(df.index.name).lower() in date_candidates:
            df = df.reset_index()
            date_col = str(df.columns[0]).lower()
        else:
            for c in df.columns:
                try:
                    parsed = pd.to_datetime(df[c], errors="raise")
                    if parsed.notna().mean() > 0.8:
                        date_col = c
                        break
                except Exception:
                    pass
    if date_col is None:
        raise SystemExit(
            f"Could not find a date column in {tfile}. Headers were: {orig_cols}"
        )
    df["rebalance_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # --- locate/normalize turnover ---
    if "turnover_bps" in df.columns:
        df["turnover_bps"] = pd.to_numeric(df["turnover_bps"], errors="coerce")
    elif "turnover" in df.columns:
        # assume fraction (e.g., 0.23) and convert to bps
        df["turnover_bps"] = pd.to_numeric(df["turnover"], errors="coerce") * 10_000.0
    else:
        guess = next((c for c in df.columns if "turnover" in c), None)
        if not guess:
            raise SystemExit(
                "Could not find a turnover column (looked for 'turnover_bps' or 'turnover'). "
                f"Headers: {orig_cols}"
            )
        df["turnover_bps"] = pd.to_numeric(df[guess], errors="coerce")
        if df["turnover_bps"].dropna().between(0, 1.5).mean() > 0.8:
            df["turnover_bps"] *= 10_000.0

    # --- adds/drops (optional) ---
    for c in ("adds", "drops"):
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    out = df[["rebalance_date", "turnover_bps", "adds", "drops"]].copy()
    out = out.dropna(subset=["rebalance_date", "turnover_bps"]).sort_values(
        "rebalance_date"
    )
    return out


def main() -> None:
    if not POLICY.exists():
        raise SystemExit(f"Policy file not found: {POLICY}")

    pol = load_policy(POLICY)
    latest = latest_backtest_dir(BT_BASE)
    tfile = latest / "turnover_by_rebalance.csv"
    if not tfile.exists():
        raise SystemExit(
            f"{tfile} not found. Run: python -m scripts.w3_turnover first."
        )

    df = _load_turnover_csv(tfile)

    # Basic metrics
    avg_turnover = df["turnover_bps"].mean()
    p95_turnover = df["turnover_bps"].quantile(0.95)

    # Evaluate breaches
    conds: list[dict[str, Any]] = []

    if pol["avg_turnover_bps"] is not None and avg_turnover > pol["avg_turnover_bps"]:
        conds.append(
            {
                "scope": "GLOBAL",
                "rule": "AVG",
                "value_bps": round(avg_turnover, 1),
                "limit_bps": pol["avg_turnover_bps"],
            }
        )

    if pol["p95_turnover_bps"] is not None and p95_turnover > pol["p95_turnover_bps"]:
        conds.append(
            {
                "scope": "GLOBAL",
                "rule": "P95",
                "value_bps": round(p95_turnover, 1),
                "limit_bps": pol["p95_turnover_bps"],
            }
        )

    # Per-rebalance caps (optional)
    hard_cap = pol["max_turnover_bps"]
    warn_cap = pol["advisory_turnover_bps"]
    if hard_cap is not None:
        m = df["turnover_bps"] > hard_cap
        for _, row in df.loc[m].iterrows():
            conds.append(
                {
                    "scope": "REB",
                    "rule": "HARD",
                    "date": row["rebalance_date"].date().isoformat(),
                    "value_bps": round(float(row["turnover_bps"]), 1),
                    "adds": int(row["adds"]),
                    "drops": int(row["drops"]),
                    "limit_bps": hard_cap,
                }
            )
    if warn_cap is not None:
        m = (df["turnover_bps"] > warn_cap) & (
            ~(df["turnover_bps"] > (hard_cap or 9e9))
        )
        for _, row in df.loc[m].iterrows():
            conds.append(
                {
                    "scope": "REB",
                    "rule": "WARN",
                    "date": row["rebalance_date"].date().isoformat(),
                    "value_bps": round(float(row["turnover_bps"]), 1),
                    "adds": int(row["adds"]),
                    "drops": int(row["drops"]),
                    "limit_bps": warn_cap,
                }
            )

    # Adds/drops caps (optional)
    if pol["max_weekly_adds"] is not None:
        m = df["adds"] > int(pol["max_weekly_adds"])
        for _, row in df.loc[m].iterrows():
            conds.append(
                {
                    "scope": "REB",
                    "rule": "ADDS",
                    "date": row["rebalance_date"].date().isoformat(),
                    "value_bps": round(float(row["turnover_bps"]), 1),
                    "adds": int(row["adds"]),
                    "drops": int(row["drops"]),
                    "limit": int(pol["max_weekly_adds"]),
                }
            )
    if pol["max_weekly_drops"] is not None:
        m = df["drops"] > int(pol["max_weekly_drops"])
        for _, row in df.loc[m].iterrows():
            conds.append(
                {
                    "scope": "REB",
                    "rule": "DROPS",
                    "date": row["rebalance_date"].date().isoformat(),
                    "value_bps": round(float(row["turnover_bps"]), 1),
                    "adds": int(row["adds"]),
                    "drops": int(row["drops"]),
                    "limit": int(pol["max_weekly_drops"]),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conds).to_csv(OUT, index=False)

    print("Turnover guard summary")
    print("----------------------")
    print(f"Latest run: {latest.name}")
    print(
        f"Avg turnover (bps): {avg_turnover:.1f}  | target ≤ {pol['avg_turnover_bps'] or 'N/A'}"
    )
    print(
        f"P95 turnover (bps): {p95_turnover:.1f} | target ≤ {pol['p95_turnover_bps'] or 'N/A'}"
    )
    print(f"Violations: {len(conds)}  → {OUT}")


if __name__ == "__main__":
    main()
