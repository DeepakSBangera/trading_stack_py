\# Execution Profiles — W25



We compare \*\*TWAP\*\*, \*\*VWAP\*\*, and \*\*POV\*\* (participation-of-volume) on the W12 last-day order set.



\- \*\*TWAP\*\*: Equal time slices across the session. Slight drift penalty if market volume is back-loaded.

\- \*\*VWAP\*\*: Volume-weighted schedule across the session. Assumed neutral drift.

\- \*\*POV\*\*: Join the tape at a fixed participation %, pays up modestly to maintain pace.



\## Model Notes (simulated)

\- Base slippage floor = 2 bps; impact = 8 bps × (participation^0.65).

\- Participation proxy = min(100 × notional / ADV, 12.5% cap). ADV from `reports/adv\_value.parquet` if present, else fallback.

\- Strategy drift: VWAP 0 bps, TWAP +2 bps, POV +4 bps (buy+ / sell−).

\- Commission 1.5 bps; taxes 0 bps (placeholder).



Outputs:

\- `reports/wk25\_exec\_engineering.csv` — per order \& strategy fill estimate and TCA components.

\- `reports/wk25\_exec\_summary.csv` — per-strategy totals, med/p90 slippage.



