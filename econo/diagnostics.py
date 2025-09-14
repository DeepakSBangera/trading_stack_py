# ADF and KPSS stationarity tests
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(s: pd.Series) -> dict:
    s = s.dropna()
    res = adfuller(s, autolag="AIC")
    return {"stat": res[0], "pvalue": res[1], "lags": res[2], "nobs": res[3]}


def kpss_test(s: pd.Series) -> dict:
    s = s.dropna()
    stat, pval, lags, _ = kpss(s, nlags="auto")
    return {"stat": stat, "pvalue": pval, "lags": lags}
