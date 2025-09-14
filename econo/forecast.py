# Baseline ARIMA forecast helper
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def arima_forecast(s: pd.Series, order=(1, 1, 1), steps: int = 12):
    s = s.dropna()
    model = sm.tsa.ARIMA(s, order=order)
    fit = model.fit()
    f = fit.get_forecast(steps=steps)
    return fit, f.summary_frame()
