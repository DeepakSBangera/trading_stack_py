from __future__ import annotations

import pandas as pd


def _ensure_dt(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out.sort_values(col)


def compute_turnover(positions: pd.DataFrame, date_col="date", id_col="ticker", w_col="weight") -> pd.DataFrame:
    """
    positions: long-form with [date, ticker, weight] where weight in [-1,1]
    Returns daily turnover % = 0.5 * sum(|w_t - w_{t-1}|) * 100.
    """
    df = _ensure_dt(positions, date_col)[[date_col, id_col, w_col]].copy()
    df["prev_w"] = df.groupby(id_col)[w_col].shift(1).fillna(0.0)
    df["abs_chg"] = (df[w_col] - df["prev_w"]).abs()
    daily = df.groupby(date_col, as_index=False)["abs_chg"].sum()
    daily["turnover_pct"] = 0.5 * daily["abs_chg"] * 100.0
    return daily[[date_col, "turnover_pct"]]


def annualized_churn(daily_turnover: pd.DataFrame, date_col="date") -> float:
    """
    Rough annualization: sum daily turnover over observed days * (252 / n).
    Cap to 10000 to avoid pathologies.
    """
    n = max(1, daily_turnover.shape[0])
    factor = 252 / n
    ann = float(daily_turnover["turnover_pct"].sum() * factor)
    return min(ann, 10000.0)


def join_with_adv(positions: pd.DataFrame, adv: pd.DataFrame, date_col="date", id_col="ticker") -> pd.DataFrame:
    a = _ensure_dt(positions, date_col)
    b = _ensure_dt(adv, date_col)
    return a.merge(b[[date_col, id_col, "adv_value"]], on=[date_col, id_col], how="left")


def liquidity_screens(
    holdings_value: pd.DataFrame,
    adv_value: pd.DataFrame,
    policy: dict,
    date_col="date",
    id_col="ticker",
    val_col="position_value",
) -> pd.DataFrame:
    """
    Holdings vs ADV: flag where position_value > cap * ADV.
    Expected columns:
      holdings_value: [date, ticker, position_value, list_tier]
      adv_value:      [date, ticker, adv_value]
    """
    cap_default = float(policy.get("adv_cap_pct", 10)) / 100.0
    caps = {
        "L1": float(policy.get("adv_cap_pct_L1", cap_default * 100)) / 100.0,
        "L2": float(policy.get("adv_cap_pct_L2", cap_default * 100)) / 100.0,
        "L3": float(policy.get("adv_cap_pct_L3", cap_default * 100)) / 100.0,
    }
    df = join_with_adv(holdings_value, adv_value, date_col, id_col)
    df["cap"] = df["list_tier"].map(caps).fillna(cap_default)
    df["limit_value"] = df["cap"] * df["adv_value"]
    df["violation"] = df[val_col] > df["limit_value"]
    cols = [
        date_col,
        id_col,
        "list_tier",
        val_col,
        "adv_value",
        "limit_value",
        "violation",
    ]
    return df[cols].sort_values([date_col, id_col])


def pretrade_violations(
    orders_value: pd.DataFrame,
    policy: dict,
    date_col="date",
    id_col="ticker",
    order_val_col="order_value",
) -> pd.DataFrame:
    """
    Pre-trade check: daily order value must be within turnover band for the tier.
    Expected columns:
      orders_value: [date, ticker, order_value, list_tier, port_value]
    """
    bands = policy.get("turnover_bands_pct_per_day", {"L1": 0.8, "L2": 1.2, "L3": 1.8})
    df = orders_value.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["band_pct"] = df["list_tier"].map(bands).fillna(bands.get("L3", 1.8))
    df["limit_value"] = (df["band_pct"] / 100.0) * df["port_value"]
    df["violation"] = df[order_val_col] > df["limit_value"]
    cols = [
        date_col,
        id_col,
        "list_tier",
        order_val_col,
        "port_value",
        "band_pct",
        "limit_value",
        "violation",
    ]
    return df[cols].sort_values([date_col, id_col])
