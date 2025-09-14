# Trading · Pricing · Econometrics — Python-First (Low-Cost)

## What this repo gives you
- Trading: fetch data, compute indicators/signals, write a buy list, view in Streamlit.
- Pricing: estimate log–log price elasticity; write price recommendations.
- Econometrics: run ADF/KPSS diagnostics + ARIMA baseline forecasts.

## Quickstart (Windows, F:)
```bat
cd /d F:\Projects\trading_stack_py
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt

python run_daily.py
streamlit run app\dashboard.py
```

### Pricing (later)
Place `pricing\data\transactions.csv` with columns:
`date,product_id,price,qty,promo_flag,cost`
Then run:
```bat
python pricing\run_pricing.py
```

### Econometrics (later)
Put `econo\timeseries\*.csv` with columns: `date,value`
Then run:
```bat
python run_econo.py
```
