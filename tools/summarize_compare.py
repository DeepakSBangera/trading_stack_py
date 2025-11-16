import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

WANT = ["CAGR", "Sharpe", "MaxDD", "VolAnn"]  # extend if you add more


def fmt_pct(x):
    try:
        return f"{100 * x:.2f}%"
    except Exception:
        return str(x)


def fmt_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default=r"reports\compare_runs.parquet")
    ap.add_argument("--out-csv", default=r"reports\compare_runs.csv")
    ap.add_argument("--decision-out", default=r"reports\promotion_decision.txt")
    # simple gates
    ap.add_argument("--min-calmar", type=float, default=0.60)
    ap.add_argument(
        "--max-dd", type=float, default=-0.40
    )  # e.g. -40% floor (higher is better)
    ap.add_argument(
        "--prefer", default="CAGR,Sharpe"
    )  # metrics to require improvement on
    args = ap.parse_args()

    p = pathlib.Path(args.compare)
    if not p.exists():
        sys.exit(f"compare parquet not found: {p}")

    df = pd.read_parquet(p)
    # Expect columns: metric, A_num, A_str, B_num, B_str, delta_num
    # Some tearsheet versions may lack strings; guard lightly
    for col in ["A_num", "B_num", "delta_num"]:
        if col not in df.columns:
            sys.exit(f"missing column '{col}' in {p}")

    # show useful subset
    sdf = df[df["metric"].isin(WANT)].copy()
    if sdf.empty:
        # fallback: show everything
        sdf = df.copy()

    # CSV export
    sdf.to_csv(args.out_csv, index=False)

    # Pretty print to console
    print("\n=== Run Comparison (A → B) ===")
    for _, r in sdf.iterrows():
        m = r["metric"]
        A = r.get("A_num", np.nan)
        B = r.get("B_num", np.nan)
        d = r.get("delta_num", np.nan)
        if m in ("CAGR", "VolAnn"):
            A_s, B_s, d_s = fmt_pct(A), fmt_pct(B), fmt_pct(d)
        elif m == "MaxDD":
            # MaxDD is negative; higher (less negative) is better
            A_s, B_s, d_s = fmt_pct(A), fmt_pct(B), fmt_pct(d)
        else:
            A_s, B_s, d_s = fmt_num(A), fmt_num(B), fmt_num(d)
        print(f"{m:>10}:  A={A_s:>10}   B={B_s:>10}   Δ={d_s:>10}")

    # Simple promotion decision:
    # - Prefer list (default CAGR, Sharpe) must improve (delta > 0)
    # - MaxDD must not worsen beyond floor (>-40%) and ideally improve (delta > 0)
    # - Optional Calmar gate if present in compare
    prefer = [m.strip() for m in args.prefer.split(",") if m.strip()]
    dd_ok, calmar_ok, prefer_ok = True, True, True

    def value(metric, col):
        row = df[df["metric"] == metric]
        return None if row.empty else float(row.iloc[0][col])

    # Prefer metrics
    for m in prefer:
        d = value(m, "delta_num")
        if d is None or d <= 0:
            prefer_ok = False

    # MaxDD (less negative is better → delta > 0)
    mdd_B = value("MaxDD", "B_num")
    mdd_d = value("MaxDD", "delta_num")
    if mdd_B is not None:
        # must be above hard floor (e.g. -40%)
        if mdd_B < args.max_dd:  # e.g., -0.45 < -0.40 → fail
            dd_ok = False
        # optional: require improvement if delta exists
        if mdd_d is not None and mdd_d <= 0:
            # not a hard fail; mark soft warning
            pass

    # Calmar (if present)
    calmar_B = value("Calmar", "B_num")
    if calmar_B is not None:
        calmar_ok = calmar_B >= args.min_calmar

    decision = "PROMOTE" if (prefer_ok and dd_ok and calmar_ok) else "HOLD"
    reasons = []
    if not prefer_ok:
        reasons.append(f"prefer metrics {prefer} not strictly improved")
    if not dd_ok:
        reasons.append(
            f"MaxDD {fmt_pct(mdd_B) if mdd_B is not None else 'NA'} below floor {fmt_pct(args.max_dd)}"
        )
    if not calmar_ok and calmar_B is not None:
        reasons.append(f"Calmar {calmar_B:.2f} < {args.min_calmar:.2f}")

    print("\nDecision:", decision)
    if reasons:
        print("Reasons:", "; ".join(reasons))

    # Write decision file
    with open(args.decision_out, "w", encoding="utf-8") as f:
        f.write("Run comparison: A vs B\n")
        f.write(f"Source: {p}\n\n")
        f.write(sdf.to_string(index=False))
        f.write("\n\n")
        f.write(f"Decision: {decision}\n")
        if reasons:
            f.write("Reasons: " + "; ".join(reasons) + "\n")
    print(f"\n✓ Wrote {args.out_csv}")
    print(f"✓ Wrote {args.decision_out}")


if __name__ == "__main__":
    main()
