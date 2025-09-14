# Batch pricing: estimate elasticity and write recommendations CSV
import os
from datetime import date

import pandas as pd
from pricing.models import fit_loglog_elasticity
from pricing.recommend import apply_recommendations

INFILE = "pricing/data/transactions.csv"


def main():
    if not os.path.exists(INFILE):
        print(f"[INFO] No pricing data at {INFILE}. Add a CSV to get recommendations.")
        return
    df = pd.read_csv(INFILE)
    out_rows = []
    for pid, d in df.groupby("product_id"):
        try:
            beta, model = fit_loglog_elasticity(
                d, price_col="price", qty_col="qty", extra_cols=["promo_flag"]
            )
            eps = abs(beta)
            rec = apply_recommendations(
                d,
                price_col="price",
                qty_col="qty",
                cost_col="cost",
                elasticity_abs=eps,
                pct_band=0.1,
            )
            out_rows.append(
                {
                    "product_id": pid,
                    "elasticity_abs": eps,
                    "current_price": rec["current"],
                    "cost_proxy": rec["cost"],
                    "rec_price": rec["recommended"],
                    "band_lo": rec["band"][0],
                    "band_hi": rec["band"][1],
                }
            )
        except Exception as e:
            out_rows.append({"product_id": pid, "error": str(e)})
    out = pd.DataFrame(out_rows)
    os.makedirs("pricing", exist_ok=True)
    outfile = f"pricing/recommendations_{date.today().isoformat()}.csv"
    out.to_csv(outfile, index=False)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
