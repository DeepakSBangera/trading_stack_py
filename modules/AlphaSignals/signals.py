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
    """Trend + near 52W high + OBV slope + sane vol + no big recent spike."""
    p = params or {}
    atr_min = float(p.get("atr_min_pct", 0.01))
    atr_max = float(p.get("atr_max_pct", 0.06))
    avoid_spike = float(p.get("avoid_spike_ret", 0.12))

    cross_up = (features["sma_f"] > features["sma_s"]) & (
        features["sma_f"].shift(1) <= features["sma_s"].shift(1)
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
    """Momentum + RSI filter + ATR band + anti-spike."""
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
