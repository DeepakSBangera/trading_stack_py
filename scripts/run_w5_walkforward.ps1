# Runs a simple walk-forward demo and computes SR + DSR on residual PnL
# Usage: pwsh -File scripts/run_w5_walkforward.ps1
\Stop = 'Stop'

python - <<'PY'
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from trading_stack_py.cv.walkforward import WalkForwardCV, WalkForwardConfig
from trading_stack_py.metrics.dsr import sharpe_ratio, deflated_sharpe_ratio
import os

np.random.seed(42)
n = 1500
# Toy features: lagged returns + noise; target is next-period return
r = 0.001 * np.random.randn(n) + 0.0002  # mildly positive drift
df = pd.DataFrame({
    "ret": r,
})
df["lag1"] = df["ret"].shift(1)
df["lag2"] = df["ret"].shift(2)
df["lag3"] = df["ret"].shift(3)
df = df.dropna().reset_index(drop=True)

X = df[["lag1","lag2","lag3"]]
y = df["ret"].shift(-1).dropna()
X = X.iloc[:len(y)]
df = df.iloc[:len(y)]

cfg = WalkForwardConfig(train_size=250, test_size=50, expanding=False)
wfcv = WalkForwardCV(cfg)

def model_factory():
    return Ridge(alpha=1.0)

res = wfcv.run(X, y, model_factory)

# Simple PnL proxy: sign(pred) * actual
res["pnl"] = np.sign(res["pred"]).astype(float) * res["y_true"]
sr = sharpe_ratio(res["pnl"].values)                # per-period SR
dsr = deflated_sharpe_ratio(sr, n=len(res), skew=float(pd.Series(res["pnl"]).skew()), kurt=float(pd.Series(res["pnl"]).kurt()+3), trials=20)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = "reports"
os.makedirs(outdir, exist_ok=True)
outcsv = os.path.join(outdir, f"w5_walkforward_demo_{stamp}.csv")
res.to_csv(outcsv, index=True)

summary = pd.DataFrame({
    "metric": ["obs","SR","DSR","pnl_mean","pnl_std"],
    "value": [len(res), sr, dsr, float(res["pnl"].mean()), float(res["pnl"].std(ddof=1))]
})
sumcsv = os.path.join(outdir, f"w5_summary_{stamp}.csv")
summary.to_csv(sumcsv, index=False)

print("SUMMARY")
print(summary.to_string(index=False))
print(f"results_csv={outcsv}")
print(f"summary_csv={sumcsv}")
PY
