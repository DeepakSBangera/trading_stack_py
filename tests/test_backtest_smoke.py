import numpy as np
import pandas as pd

from scripts.w2_backtest import run_backtest


def test_smoke():
    dates = pd.bdate_range("2021-01-01", periods=260)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, (len(dates), 4)), index=dates, columns=list("ABCD")
    )
    px = (1 + rets).cumprod() * 100
    res = run_backtest(px, top_n=2)
    assert res.equity.iloc[-1] > 0  # basic sanity
