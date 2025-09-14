# Streamlit dashboard: shows latest BUY list + shows current rule & source from config
import streamlit as st
import pandas as pd
import glob, os, yaml

st.title("Trading Dashboard — Signals & Portfolio (Milestone-1)")

# Show config summary
try:
    with open("config/config.yaml","r",encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    rule = CFG["signals"].get("rule","?")
    source = CFG["data"].get("source","?")
    st.caption(f"Using rule: **{rule}**, data source: **{source}**")
except Exception:
    st.caption("Could not read config/config.yaml")

files = sorted(glob.glob("reports/buylist_*.csv"))
if files:
    latest = files[-1]
    st.subheader(f"Today's BUY list: {os.path.basename(latest)}")
    buylist = pd.read_csv(latest)
    st.dataframe(buylist)
else:
    st.info("Run `python run_daily.py` to generate today's BUY list.")

col1,col2 = st.columns(2)
with col1:
    st.subheader("Positions (paper)")
    try:
        pos = pd.read_csv("db/positions.csv")
        st.dataframe(pos)
    except FileNotFoundError:
        st.write("No positions yet.")

with col2:
    st.subheader("Portfolio snapshots")
    try:
        pf = pd.read_csv("db/portfolio.csv")
        st.line_chart(pf.set_index("date")["equity"])
    except Exception:
        st.write("No portfolio snapshots yet.")
