# Baseline ARIMA forecast helper
import pandas as pd
import statsmodels.api as sm

def arima_forecast(s, order=(1,1,1), steps=12):
    s = s.dropna()
    model = sm.tsa.ARIMA(s, order=order)
    fit = model.fit()
    f = fit.get_forecast(steps=steps)
    return fit, f.summary_frame()
