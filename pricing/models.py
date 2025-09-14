# Pricing model: log-log elasticity per product using OLS
import numpy as np
import pandas as pd
import statsmodels.api as sm

def fit_loglog_elasticity(df, price_col, qty_col, extra_cols=None):
    extra_cols = extra_cols or []
    d = df[[price_col, qty_col] + extra_cols].dropna()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)].copy()
    d["ln_qty"] = np.log(d[qty_col])
    d["ln_price"] = np.log(d[price_col])
    X = d[["ln_price"] + extra_cols]
    X = sm.add_constant(X)
    model = sm.OLS(d["ln_qty"], X, hasconst=True).fit()
    beta = model.params.get("ln_price", np.nan)  # elasticity (often negative)
    return beta, model

