from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

"""
W31 — Execution Bandit
Goal: Choose an execution style per order (VWAP/TWAP/POV-10/POV-20) to minimize realized slippage.
Arms: ["VWAP","TWAP","POV10","POV20"]
Reward (higher is better):  - (slippage_bps + commission_bps + tax_bps)
  If no realized fills yet, use prior belief (Bayesian-ish) or global median.

Inputs (auto-detected, optional fallbacks):
- reports/wk13_dryrun_fills.csv      # realized slippage_bps, commission_bps, tax_bps (historical sim)
- reports/wires/w19_orders_wire.csv  # today's orders to route (ticker, side, qty, px_ref/limit etc.)
- reports/exec_bandit_log.csv        # cumulative arm stats per (symbol bucket, TOD bucket), optional

Outputs:
- reports/w31_bandit_assignments.csv   # order_id,ticker,chosen_arm,explore/exploit flag, est_reward
- reports/exec_bandit_log.csv          # updated bandit stats (counts, mean reward, UCB bonus)
- reports/w31_exec_bandit_diag.json    # summary, knobs, coverage
"""

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"
WIRES = REPORTS / "wires"

FILLS_CSV = REPORTS / "wk13_dryrun_fills.csv"
ORDERS_CSV = WIRES / "w19_orders_wire.csv"
BANDIT_LOG_CSV = REPORTS / "exec_bandit_log.csv"

OUT_ASSIGN = REPORTS / "w31_bandit_assignments.csv"
OUT_DIAG = REPORTS / "w31_exec_bandit_diag.json"

# ---- knobs ----
ARMS = ["VWAP", "TWAP", "POV10", "POV20"]
EPSILON = 0.10  # ε-greedy exploration prob
UCB_C = 1.5  # UCB bonus scale (if using UCB fallback)
USE_EPSILON_GREEDY = True
RANDOM_SEED = 31

# For per-bucket bandits
SYMBOL_BUCKETS = [
    "BIG",
    "MID",
    "SMALL",
]  # by notional vs ADV if available; we’ll infer by notional quantiles
TOD_BUCKETS = [
    "OPEN",
    "MID",
    "CLOSE",
]  # time-of-day; for EOD workflow we default to MID


def _safe_read_csv(p: Path) -> pd.DataFrame | None:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


def _reward_from_fills(df: pd.DataFrame) -> pd.Series:
    # reward = -(slippage + commission + tax) in bps
    cols = {c.lower(): c for c in df.columns}
    s = pd.to_numeric(df[cols.get("slippage_bps", "slippage_bps")], errors="coerce")
    com = pd.to_numeric(
        df.get(cols.get("commission_bps", "commission_bps"), 0.0), errors="coerce"
    ).fillna(0.0)
    tax = pd.to_numeric(
        df.get(cols.get("tax_bps", "tax_bps"), 0.0), errors="coerce"
    ).fillna(0.0)
    return -(s.fillna(s.median())) - com - tax


def _prep_historical_rewards() -> pd.DataFrame:
    """
    Build a table: [ticker, arm, reward_bps, notional_ref, ts, tod_bucket]
    We infer arm via 'exec_arm' if present; otherwise map heuristically (unknown->VWAP).
    """
    df = _safe_read_csv(FILLS_CSV)
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ticker", "arm", "reward_bps", "notional", "tod_bucket", "ts"]
        )

    cols = {c.lower(): c for c in df.columns}
    tic = cols.get("ticker", "ticker")
    notional_col = cols.get("notional_ref") or cols.get("notional") or None
    ts_col = cols.get("date") or cols.get("ts") or None
    arm_col = None
    # Try detect arm column if previously logged
    for k in ["exec_arm", "arm", "schedule", "style"]:
        if k in cols:
            arm_col = cols[k]
            break

    rewards = _reward_from_fills(df)
    out = pd.DataFrame(
        {
            "ticker": df[tic].astype(str),
            "arm": df[arm_col].astype(str) if arm_col else "VWAP",
            "reward_bps": rewards,
            "notional": (
                pd.to_numeric(df[notional_col], errors="coerce")
                if notional_col
                else np.nan
            ),
            "ts": pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.NaT,
        }
    )
    # TOD bucket (coarse)
    tod = []
    for t in out["ts"]:
        if pd.isna(t):
            tod.append("MID")
        else:
            h = t.hour
            if h <= 10:
                tod.append("OPEN")
            elif h >= 14:
                tod.append("CLOSE")
            else:
                tod.append("MID")
    out["tod_bucket"] = tod
    return out


def _bucket_symbol(notional_series: pd.Series) -> pd.Series:
    # If notional missing, return MID bucket
    x = pd.to_numeric(notional_series, errors="coerce").fillna(
        notional_series.median() if notional_series.size else 0.0
    )
    if x.empty:
        return pd.Series(["MID"] * 0)
    # Quantile cut
    q1, q2 = x.quantile(0.33), x.quantile(0.66)
    bucket = []
    for v in x:
        if v <= q1:
            bucket.append("SMALL")
        elif v <= q2:
            bucket.append("MID")
        else:
            bucket.append("BIG")
    return pd.Series(bucket)


def _load_orders() -> pd.DataFrame:
    df = _safe_read_csv(ORDERS_CSV)
    if df is None or df.empty:
        raise FileNotFoundError(f"Missing or empty {ORDERS_CSV}. Please run W19 first.")
    cols = {c.lower(): c for c in df.columns}
    need = [
        cols.get("ticker"),
        cols.get("qty") or cols.get("quantity"),
        cols.get("limit_price") or cols.get("px_ref") or cols.get("price"),
    ]
    if any(c is None for c in need):
        # Permit minimal: ticker, qty present; price optional
        pass
    # add notional proxy if price present
    if (cols.get("limit_price") or cols.get("px_ref") or cols.get("price")) and (
        cols.get("qty") or cols.get("quantity")
    ):
        pcol = cols.get("limit_price") or cols.get("px_ref") or cols.get("price")
        qcol = cols.get("qty") or cols.get("quantity")
        df["notional"] = pd.to_numeric(df[pcol], errors="coerce").fillna(
            0.0
        ) * pd.to_numeric(df[qcol], errors="coerce").fillna(0.0)
    else:
        df["notional"] = np.nan
    # default TOD to MID (end-of-day workflow)
    df["tod_bucket"] = "MID"
    # make an order_id
    if "order_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "order_id"})
        df["order_id"] = df["order_id"].astype(int) + 1
    return df


def _init_bandit_table(arms: list[str]) -> pd.DataFrame:
    # Multi-index by (symbol_bucket, tod_bucket, arm)
    rows = []
    for sb in SYMBOL_BUCKETS:
        for tb in TOD_BUCKETS:
            for a in arms:
                rows.append(
                    {
                        "symbol_bucket": sb,
                        "tod_bucket": tb,
                        "arm": a,
                        "n": 0,
                        "mean_reward": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _update_bandit_from_history(btab: pd.DataFrame, hist: pd.DataFrame):
    if hist is None or hist.empty:
        return btab
    # symbol_bucket from notional
    sb = _bucket_symbol(hist["notional"])
    hist = hist.copy()
    hist["symbol_bucket"] = sb
    g = (
        hist.groupby(["symbol_bucket", "tod_bucket", "arm"], as_index=False)[
            "reward_bps"
        ]
        .agg(["count", "mean"])
        .reset_index()
    )
    g = g.rename(columns={"count": "n", "mean": "mean_reward"})
    for _, r in g.iterrows():
        mask = (
            (btab["symbol_bucket"] == r["symbol_bucket"])
            & (btab["tod_bucket"] == r["tod_bucket"])
            & (btab["arm"] == r["arm"])
        )
        btab.loc[mask, "n"] = btab.loc[mask, "n"].astype(int) + int(r["n"])
        # running mean (simple overwrite fine for aggregated history)
        btab.loc[mask, "mean_reward"] = float(r["mean_reward"])
    return btab


def _merge_existing_log(
    btab: pd.DataFrame, log_df: pd.DataFrame | None
) -> pd.DataFrame:
    if log_df is None or log_df.empty:
        return btab
    # bring forward prior n/mean
    key = ["symbol_bucket", "tod_bucket", "arm"]
    merged = pd.merge(
        btab,
        log_df[key + ["n", "mean_reward"]],
        on=key,
        how="left",
        suffixes=("", "_old"),
    )
    merged["n"] = merged["n_old"].fillna(merged["n"]).astype(int)
    merged["mean_reward"] = merged["mean_reward_old"].fillna(merged["mean_reward"])
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_old")])
    return merged


def _choose_arm(row, btab: pd.DataFrame, explore_prob: float) -> tuple[str, str, float]:
    sb, tb = row["symbol_bucket"], row["tod_bucket"]
    # Slice bandit rows
    candidates = btab[(btab["symbol_bucket"] == sb) & (btab["tod_bucket"] == tb)].copy()
    if candidates.empty:
        # fallback: global means
        candidates = btab.groupby("arm", as_index=False)["mean_reward"].mean()
        # Choose max
        arm = candidates.sort_values("mean_reward", ascending=False).iloc[0]["arm"]
        return arm, "fallback", float(candidates["mean_reward"].max())

    # ε-greedy
    if USE_EPSILON_GREEDY and random.random() < explore_prob:
        # explore: pick a minimally tried arm (or random among ties)
        min_n = candidates["n"].min()
        pool = candidates[candidates["n"] == min_n]
        pick = pool.sample(1, random_state=random.randint(1, 10**6)).iloc[0]
        return str(pick["arm"]), "explore", float(pick["mean_reward"])

    # exploit: pick best estimated mean; slight UCB bonus for low n
    T = max(int(btab["n"].sum()), 1)
    bonus = UCB_C * np.sqrt(np.log(T) / np.maximum(1, candidates["n"].values))
    score = candidates["mean_reward"].values + bonus
    i = int(np.argmax(score))
    return str(candidates.iloc[i]["arm"]), "exploit", float(score[i])


def _append_log(log_df: pd.DataFrame | None, updates: list[dict]) -> pd.DataFrame:
    add = pd.DataFrame(updates)
    if log_df is None or log_df.empty:
        out = add
    else:
        out = pd.concat([log_df, add], ignore_index=True)
    # roll-up to maintain summary table per (sb,tb,arm): n & mean
    roll = (
        out.groupby(["symbol_bucket", "tod_bucket", "arm"], as_index=False)[
            "reward_bps"
        ]
        .agg(["count", "mean"])
        .reset_index()
    )
    roll = roll.rename(columns={"count": "n", "mean": "mean_reward"})
    return roll


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    REPORTS.mkdir(parents=True, exist_ok=True)
    WIRES.mkdir(parents=True, exist_ok=True)

    orders = _load_orders()
    hist = _prep_historical_rewards()
    prior_log = _safe_read_csv(BANDIT_LOG_CSV)

    # Build base bandit table and update from history+log
    btab = _init_bandit_table(ARMS)
    btab = _update_bandit_from_history(btab, hist)
    btab = _merge_existing_log(btab, prior_log)

    # Bucket symbols for incoming orders
    orders = orders.copy()
    orders["symbol_bucket"] = _bucket_symbol(orders["notional"]).fillna("MID")
    # Choose per order
    assigns = []
    updates_for_log = []
    for _, r in orders.iterrows():
        arm, mode, est = _choose_arm(r, btab, EPSILON)
        assigns.append(
            {
                "order_id": r["order_id"],
                "ticker": str(r["ticker"]),
                "qty": int(
                    pd.to_numeric(
                        r.get("qty", r.get("quantity", 0)), errors="coerce"
                    ).fillna(0)
                ),
                "tod_bucket": r["tod_bucket"],
                "symbol_bucket": r["symbol_bucket"],
                "chosen_arm": arm,
                "mode": mode,
                "est_reward_bps": round(float(est), 4),
            }
        )
        # We only update mean rewards once realized fills arrive (next day).
        # For now we log a pseudo-observation using current bandit mean as a weak prior (weight n=0).
        # No immediate update here to avoid bias.

    # Save assignments
    pd.DataFrame(assigns).to_csv(OUT_ASSIGN, index=False)

    # Keep bandit log as a *summary* table (n,mean_reward). We don't change it until fills arrive.
    btab.to_csv(BANDIT_LOG_CSV, index=False)

    diag = {
        "arms": ARMS,
        "epsilon": EPSILON,
        "ucb_c": UCB_C,
        "use_epsilon_greedy": USE_EPSILON_GREEDY,
        "orders_in": int(orders.shape[0]),
        "assignments_out": int(len(assigns)),
        "history_rows": 0 if hist is None else int(hist.shape[0]),
        "bandit_rows": int(btab.shape[0]),
        "files": {
            "orders_csv": str(ORDERS_CSV),
            "fills_csv": str(FILLS_CSV),
            "bandit_log_csv": str(BANDIT_LOG_CSV),
            "assignments_csv": str(OUT_ASSIGN),
        },
        "notes": "Assigns an execution style per order using ε-greedy + UCB tie-break. Rewards are negative total bps drag. Update bandit_log after fills.",
    }
    OUT_DIAG.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"assignments_csv": str(OUT_ASSIGN), "diag_json": str(OUT_DIAG)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
