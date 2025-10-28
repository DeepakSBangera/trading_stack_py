import numpy as np

from trading_stack_py.cv.walkforward import WalkForwardCV
from trading_stack_py.metrics.dsr import deflated_sharpe_ratio

np.random.seed(42)

# Synthetic daily returns for 3 years (~756 obs), slight edge + noise
n = 756
true_edge = 0.0003  # ~7.5% annualized mean if 252 days
rets = true_edge + 0.01 * np.random.standard_t(df=6, size=n)  # fat tails

# Walk-forward splits: 1.5y train, 3m test, step = 1m
cv = WalkForwardCV(train_size=378, test_size=63, step_size=21, expanding=True)
segments = 0
sr_list = []
for _tr, te in cv.split(n):
    segments += 1
    r_te = rets[te]
    # simple SR on test segment
    sr = r_te.mean() / r_te.std(ddof=1)
    sr_list.append(sr)

# Treat each segment as a "trial" from the same strategy family
num_trials = max(1, len(sr_list))
dsr_val = deflated_sharpe_ratio(np.array(sr_list), num_trials=num_trials)

print(f"Walk-forward segments: {segments}")
print(f"Segment SRs (first 5): {np.array(sr_list)[:5].round(3)}")
print(f"DSR over segment SR distribution (N={num_trials}): {dsr_val:.3f}")
