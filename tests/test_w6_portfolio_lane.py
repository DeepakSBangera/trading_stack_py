import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RPT = ROOT / "reports"


def test_w6_portfolio_compare_exists_and_sane():
    path = RPT / "wk6_portfolio_compare.csv"
    assert path.exists(), "wk6_portfolio_compare.csv missing – run W6 lane"

    df = pd.read_csv(path)
    assert not df.empty, "wk6_portfolio_compare.csv is empty"

    expected_cols = {
        "ticker",
        "sector",
        "w_current",
        "w_capped",
        "w_final",
        "delta_vs_current",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"wk6_portfolio_compare.csv missing columns: {missing}"

    # Sanity on final weights: sum close to 1, no insane exposures
    w_final = df["w_final"].dropna()

    # Sum of weights should be ~1
    assert abs(w_final.sum() - 1.0) < 1e-6, f"w_final sum != 1 (got {w_final.sum():.6f})"

    # No single name should have crazy weight (0.2 is very generous vs your 0.12 cap)
    assert (w_final.abs() <= 0.2).all(), "w_final has weights > 0.2; check caps logic"


def test_w6_capacity_curve_exists_and_sane():
    path = RPT / "capacity_curve.csv"
    assert path.exists(), "capacity_curve.csv missing – run w6_capacity_curve.py"

    df = pd.read_csv(path)
    assert not df.empty, "capacity_curve.csv is empty"

    expected_cols = {
        "date",
        "total_capacity_value",
        "p50_per_name_cap",
        "p75_per_name_cap",
        "p90_per_name_cap",
        "p95_per_name_cap",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"capacity_curve.csv missing columns: {missing}"

    # Capacity numbers should be positive and within a sane band
    cap = df["total_capacity_value"].dropna()
    assert (cap > 0).all(), "total_capacity_value has non-positive entries"
    # loose sanity band: between 1e6 and 1e9 (you are ~3e7)
    assert cap.max() < 1e9, f"total_capacity_value too large: {cap.max():.2f}"
    assert cap.min() > 1e5, f"total_capacity_value suspiciously tiny: {cap.min():.2f}"
