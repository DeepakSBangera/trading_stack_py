import os
import time

import numpy as np
import pandas as pd
from trading_stack_py.cv.walkforward import WalkForwardCV
from trading_stack_py.metrics.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

from trading_stack_py.metrics.mtl import minimum_track_record_length


def segment_sr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return 0.0
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 else float(x.mean() / sd)


def main():
    np.random.seed(7)
    n = 756
    true_edge = 0.00025
    rets = true_edge + 0.01 * np.random.standard_t(df=6, size=n)

    cv = WalkForwardCV(train_size=378, test_size=63, step_size=21, expanding=True, embargo=5)
    rows = []
    seg = 0
    sr_list = []
    for tr, te in cv.split(n):
        seg += 1
        r_te = rets[te]
        sr = segment_sr(r_te)
        psr = probabilistic_sharpe_ratio(sr_hat=sr, sr_threshold=0.0, n=len(r_te), skewness=0.0, kurt=3.0)
        rows.append(
            {
                "segment": seg,
                "train_start": int(tr[0]),
                "train_end": int(tr[-1]),
                "test_start": int(te[0]),
                "test_end": int(te[-1]),
                "test_len": int(len(r_te)),
                "test_mean": float(np.mean(r_te)),
                "test_std": float(np.std(r_te, ddof=1)),
                "test_sr": float(sr),
                "test_psr_vs_0": float(psr),
            }
        )
        sr_list.append(sr)

    sr_arr = np.array(sr_list, dtype=float)
    dsr_val = deflated_sharpe_ratio(sr_arr, num_trials=len(sr_arr))
    mtl_90 = minimum_track_record_length(sr_hat=float(np.nanmean(sr_arr)), p_star=0.90)
    mtl_95 = minimum_track_record_length(sr_hat=float(np.nanmean(sr_arr)), p_star=0.95)

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("reports", "W5", ts)
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "segment_metrics.csv")
    md_path = os.path.join(outdir, "README.md")
    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# W5 Walk-Forward Report\n\n")
        f.write(f"- Segments: **{len(sr_arr)}**\n")
        f.write(f"- Mean segment SR: **{float(np.nanmean(sr_arr)):.3f}**\n")
        f.write(f"- DSR over segment SRs (N={len(sr_arr)}): **{dsr_val:.3f}**\n")
        f.write(f"- MTL (PSR≥90%): **{mtl_90:.1f} periods**\n")
        f.write(f"- MTL (PSR≥95%): **{mtl_95:.1f} periods**\n\n")
        f.write("## Segments (first 5)\n\n")
        f.write(df.head().to_markdown(index=False))
        f.write("\n")

    print(f"Report written:\n - {csv_path}\n - {md_path}")


if __name__ == "__main__":
    main()
