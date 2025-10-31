# Modular Architecture (Plug & Play)

Each module has **code**, **scripts**, **tests**, **reports**, and a **README**:
- DataPipe        — raw→clean, schema/QC, PIT
- AlphaFactors    — style factors (momentum/quality/vol)
- AlphaSignals    — rankings, entry/exit policies
- Portfolio       — optimizers (EW, shrinkage MV), constraints, exposures
- RiskSizing      — vol targeting, ATR stops, Kelly, DD throttle
- Testing         — walk-forward, DSR, PBO/CPCV
- Regime          — regime labeling and policy switches
- Execution       — TCA, scheduling (VWAP/TWAP/POV), capacity hooks
- Governance      — change control, kill-switch, lineage, pre-trade
- Monitoring      — IC half-life, attribution, tiles
- Dashboards      — Streamlit/HTML reports & tearsheets
- Adapters        — broker/data connectors (e.g., Kite)

Back-compat wrappers remain in **scripts/** so legacy commands continue to work.
