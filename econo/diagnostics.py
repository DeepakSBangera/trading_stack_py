# ADF and KPSS stationarity tests
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(s):
    s = s.dropna()
    res = adfuller(s, autolag="AIC")
    return {"stat": res[0], "pvalue": res[1], "lags": res[2], "nobs": res[3]}

def kpss_test(s):
    s = s.dropna()
    stat, pval, lags, crit = kpss(s, nlags="auto")
    return {"stat": stat, "pvalue": pval, "lags": lags}
