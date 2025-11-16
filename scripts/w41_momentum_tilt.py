from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- paths ----------
ROOT = Path(r"F:\Projects\trading_stack_py")
DATA_DIRS = [ROOT / "data" / "prices", ROOT / "data" / "csv"]
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "wk41_momentum_tilt.csv"
OUT_SUMMARY = REPORTS / "wk41_momentum_tilt_summary.json"


# ---------- helpers ----------
def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _load_panel(max_tickers: int = 50) -> pd.DataFrame:
    """Return wide price panel with columns per ticker and DateTimeIndex."""
    files = []
    for d in DATA_DIRS:
        if d.exists():
            files += sorted(list(d.glob("*.parquet"))) + sorted(list(d.glob("*.csv")))
    if not files:
        raise SystemExit("No price files found under data/prices or data/csv.")

    frames = []
    for p in files[:max_tickers]:
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            date_col = _pick_col(df.columns.tolist(), ["date", "dt", "timestamp"])
            px_col = _pick_col(df.columns.tolist(), ["adj_close", "adjusted_close", "close", "price"])
            if date_col is None or px_col is None:
                continue
            df = df[[date_col, px_col]].copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.dropna().sort_values(date_col)
            df = df.set_index(date_col).rename(columns={px_col: p.stem})
            frames.append(df)
        except Exception:
            # keep going; we only need the ones that load cleanly
            pass

    if not frames:
        raise SystemExit("Could not build any price series (no usable columns).")
    panel = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    return panel


def _rank_normalize(s: pd.Series) -> pd.Series:
    """Convert scores to 0..1 by rank (ties average)."""
    r = s.rank(method="average", na_option="keep")
    return (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else s * 0 + 0.5


def _softmax(s: pd.Series, temp: float = 1.0) -> pd.Series:
    v = s.fillna(s.min() - 1.0)
    v = v / max(1e-9, temp)
    v = v - v.max()
    e = np.exp(v.clip(-50, 50))  # numeric safety
    out = e / e.sum() if e.sum() > 0 else pd.Series(np.full(len(s), 1.0 / len(s)), index=s.index)
    return pd.Series(out, index=s.index)


# ---------- core ----------
def compute_momentum_tilt(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Score = 0.5 * 12m + 0.3 * 6m + 0.2 * 1m total return;
    then apply inverse-vol (63d) scaling; final weights via softmax.
    """
    if panel.shape[0] < 260:
        # still works—use what we have
        pass

    # trailing returns
    r12 = (
        panel.iloc[-1] / panel.iloc[-252:].iloc[0] - 1
        if panel.shape[0] >= 252
        else (panel.pct_change(21).add(1).rolling(12).apply(np.prod) - 1).iloc[-1]
    )
    r06 = (
        panel.iloc[-1] / panel.iloc[-126:].iloc[0] - 1
        if panel.shape[0] >= 126
        else (panel.pct_change(21).add(1).rolling(6).apply(np.prod) - 1).iloc[-1]
    )
    r01 = (
        panel.iloc[-1] / panel.iloc[-21:].iloc[0] - 1
        if panel.shape[0] >= 21
        else panel.pct_change().rolling(21).sum().iloc[-1]
    )

    # combine with weights
    raw = 0.5 * r12 + 0.3 * r06 + 0.2 * r01
    score = raw.replace([np.inf, -np.inf], np.nan)

    # inverse vol (lower vol -> higher weight), 63d
    vol = panel.pct_change().rolling(63).std().iloc[-1].replace(0, np.nan)
    inv_vol = 1.0 / vol
    inv_vol = inv_vol / inv_vol.max()

    # rank-normalize momentum & inv-vol, then multiply (geometric blend)
    m_norm = _rank_normalize(score)
    v_norm = _rank_normalize(inv_vol)
    blend = (m_norm.clip(0, 1) * v_norm.clip(0, 1)).fillna(0)

    # final weights via softmax to avoid zeroing
    w = _softmax(blend, temp=0.15)

    out = (
        pd.DataFrame(
            {
                "ticker": w.index.astype(str),
                "mom_12m": r12.values,
                "mom_6m": r06.values,
                "mom_1m": r01.values,
                "score_raw": score.values,
                "vol_63d": vol.reindex(w.index).values,
                "rank_mom": m_norm.values,
                "rank_invvol": v_norm.values,
                "w_tilt": w.values,
            }
        )
        .sort_values("w_tilt", ascending=False)
        .reset_index(drop=True)
    )

    # select top 30 names (hard cap), renormalize
    out["sel"] = False
    cap = min(30, out.shape[0])
    out.loc[: cap - 1, "sel"] = True
    sel_sum = out.loc[out["sel"], "w_tilt"].sum()
    if sel_sum > 0:
        out.loc[out["sel"], "w_tilt"] = out.loc[out["sel"], "w_tilt"] / sel_sum
        out.loc[~out["sel"], "w_tilt"] = 0.0

    return out


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    panel = _load_panel(max_tickers=200)
    res = compute_momentum_tilt(panel)

    # write csv
    res.to_csv(OUT_CSV, index=False)

    # summary
    as_of = panel.index.max()
    info = {
        "as_of_ist": pd.Timestamp(as_of, tz="Asia/Kolkata").isoformat(),
        "tickers": int(res.shape[0]),
        "selected": int(res["sel"].sum()),
        "files": {"detail_csv": str(OUT_CSV)},
        "notes": "Momentum (12/6/1m) × inverse-vol blend; top-30 normalized.",
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
