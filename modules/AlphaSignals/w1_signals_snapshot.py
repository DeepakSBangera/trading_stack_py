# scripts/w1_signals_snapshot.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FEATURES = ROOT / "data" / "features"
SNAPSHOT = REPORTS / "signals_snapshot.csv"


def _load_from_week1_report() -> pd.DataFrame | None:
    """Try the canonical Week-1 output first."""
    # pick the latest baseline CSV if there are many
    cands = sorted(REPORTS.glob("wk1_entry_exit_baseline*.csv"))
    if not cands:
        return None
    latest = max(cands, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest)
    # Expect typical columns from Week 1; keep whatever exists
    keep_cols = [
        c
        for c in ["ticker", "score", "atr", "atr_pct", "rank", "list_priority"]
        if c in df.columns
    ]
    if "ticker" not in df.columns:
        # try common alternate name
        for alt in ("symbol", "asset"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                keep_cols = ["ticker"] + [c for c in keep_cols if c != "ticker"]
                break
    if "ticker" not in df.columns:
        # give up—structure not recognized
        return None
    out = df[keep_cols].copy()
    # sort by score/rank if present
    if "score" in out.columns:
        out = out.sort_values("score", ascending=False)
    elif "rank" in out.columns:
        out = out.sort_values("rank", ascending=True)
    return out


def _load_from_features_fallback() -> pd.DataFrame:
    """If Week-1 baseline isn’t present, synthesize a simple snapshot from features."""
    rows = []
    for p in sorted(FEATURES.glob("*.parquet")):
        try:
            f = pd.read_parquet(p)
            # try to find common momentum/atr columns; otherwise approximate
            score = None
            for c in ("mom_12_1", "r12_1", "momentum", "score"):
                if c in f.columns:
                    score = pd.to_numeric(f[c], errors="coerce").iloc[-1]
                    break
            if score is None:
                # rough proxy: 252d return minus 21d return
                # (works if we have 'close' or 'adj close' present)
                price_col = None
                for c in ("adj close", "Adj Close", "close", "Close"):
                    if c in f.columns:
                        price_col = c
                        break
                if price_col is not None:
                    s = pd.to_numeric(f[price_col], errors="coerce")
                    r252 = (s / s.shift(252) - 1.0).iloc[-1]
                    r21 = (s / s.shift(21) - 1.0).iloc[-1]
                    score = r252 - r21
            atr = None
            for c in ("atr", "ATR", "atr_14"):
                if c in f.columns:
                    atr = pd.to_numeric(f[c], errors="coerce").iloc[-1]
                    break
            rows.append({"ticker": p.stem, "score": score, "atr": atr})
        except Exception:
            # ignore unreadable files
            pass
    df = pd.DataFrame(rows)
    if not df.empty and "score" in df.columns:
        df = df.sort_values("score", ascending=False)
        df["rank"] = range(1, len(df) + 1)
    return df


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = _load_from_week1_report()
    if df is None or df.empty:
        print("[info] Week-1 baseline not found or empty; using features fallback …")
        df = _load_from_features_fallback()
    if df is None or df.empty:
        raise SystemExit(
            "No signals could be built: make sure Week-1 scripts ran or features exist."
        )
    df.to_csv(SNAPSHOT, index=False)
    # Show a tiny preview
    print(f"Wrote: {SNAPSHOT}")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(df.head(10))


if __name__ == "__main__":
    main()
