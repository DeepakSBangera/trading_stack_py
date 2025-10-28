"""
Streamlit dashboard: shows latest BUY list and (later) portfolio)
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ---------- Setup ----------
st.set_page_config(page_title="Trading Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parents[1]  # repo root (…/trading_stack_py)
REPORTS = ROOT / "reports"
DB = ROOT / "db"
CFG_FILES = [ROOT / "config" / "config.yaml", ROOT / "config" / "policy_w1.yaml"]

# ---------- Header ----------
st.title("Trading Dashboard — Signals & Portfolio (Milestone 1)")

# ---------- Config summary ----------
cfg_loaded = None
for p in CFG_FILES:
    try:
        with open(p, encoding="utf-8") as f:
            cfg_loaded = yaml.safe_load(f)
        cfg_name = p.relative_to(ROOT)
        break
    except FileNotFoundError:
        continue

if cfg_loaded:
    rule = (cfg_loaded.get("signals", {}) or {}).get("rule") or "?"
    source = (cfg_loaded.get("data", {}) or {}).get("source") or "?"
    st.caption(f"Config: `{cfg_name}` · rule: **{rule}** · data source: **{source}**")
else:
    st.caption(
        "No config found (looked for `config/config.yaml` and `config/policy_w1.yaml`)."
    )

# ---------- Latest BUY list ----------
buylist_files = sorted(glob.glob(str(REPORTS / "buylist_*.csv")))
if buylist_files:
    latest_path = Path(buylist_files[-1])
    st.subheader(f"Today's BUY list — {latest_path.name}")
    try:
        buylist = pd.read_csv(latest_path)
        st.dataframe(buylist, use_container_width=True)
    except Exception as e:
        st.error(f"Could not read `{latest_path}`: {type(e).__name__}: {e}")
else:
    st.info(
        "No `buylist_*.csv` found under `reports/`.\n\n"
        "Tip: run `python scripts/w1_build_entry_exit.py` or your daily pipeline to generate it."
    )

# ---------- Two-column layout ----------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Positions (paper)")
    pos_path = DB / "positions.csv"
    if pos_path.exists():
        try:
            pos = pd.read_csv(pos_path)
            st.dataframe(pos, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read `{pos_path}`: {type(e).__name__}: {e}")
    else:
        st.write("No positions yet. Create `db/positions.csv` to see them here.")

with col2:
    st.subheader("Portfolio snapshots")
    pf_path = DB / "portfolio.csv"
    if pf_path.exists():
        try:
            pf = pd.read_csv(pf_path)
            if "date" in pf.columns and "equity" in pf.columns:
                pf = pf.copy()
                pf["date"] = pd.to_datetime(pf["date"], errors="coerce")
                pf = pf.dropna(subset=["date"]).set_index("date").sort_index()
                st.line_chart(pf["equity"])
            else:
                st.warning("`db/portfolio.csv` should have columns: `date,equity`.")
        except Exception as e:
            st.error(f"Could not read `{pf_path}`: {type(e).__name__}: {e}")
    else:
        st.write(
            "No portfolio snapshots yet. Create `db/portfolio.csv` to chart equity."
        )

# ---------- Sidebar (quick diagnostics) ----------
with st.sidebar:
    st.header("Quick checks")
    st.write("**Reports dir**", f"`{REPORTS}`")
    st.write("**DB dir**", f"`{DB}`")
    st.write("**Latest buylist found?**", "✅" if buylist_files else "❌")
    st.write(
        "**Positions.csv present?**", "✅" if (DB / "positions.csv").exists() else "❌"
    )
    st.write(
        "**Portfolio.csv present?**", "✅" if (DB / "portfolio.csv").exists() else "❌"
    )
