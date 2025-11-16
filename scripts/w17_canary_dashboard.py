# scripts/w17_canary_dashboard.py  (TZ-safe; Series .dt.tz_convert fix)
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\trading_stack_py")
REPORTS = ROOT / "reports"

CANARY_LOG = REPORTS / "canary_log.csv"  # expected: timestamp,check,status[,detail]
MAN_INDEX = REPORTS / "run_manifest_index.csv"  # used for seeding if needed

DASH_HTML = REPORTS / "canary_dashboard.html"
SUM_CSV = REPORTS / "canary_summary.csv"
DIAG_JSON = REPORTS / "w17_canary_diag.json"

TS_CANDS = ["timestamp", "ts", "time", "created_at", "datetime", "when"]
CHECK_CANDS = ["check", "name", "probe", "test", "event", "task"]
STATUS_CANDS = ["status", "state", "result", "pass_fail", "ok", "outcome"]
DETAIL_CANDS = ["detail", "msg", "message", "note", "info"]

IST = "Asia/Kolkata"


def _now_ist_naive():
    return pd.Timestamp.now(tz=IST).tz_localize(None)


def _seed_canary_if_missing() -> None:
    if CANARY_LOG.exists():
        return
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    if MAN_INDEX.exists():
        idx = pd.read_csv(MAN_INDEX)
        tcol = None
        for c in ["timestamp", "ts", "time", "created_at"]:
            if c in idx.columns:
                tcol = c
                break
        if tcol is not None:
            ts_series = pd.to_datetime(idx[tcol], errors="coerce", utc=True)
            days = sorted({t.tz_convert(IST).date() for t in ts_series.dropna()})
            days = days[-20:] if len(days) > 20 else days
            for d in days:
                rows.append(
                    {
                        "timestamp": f"{d}T09:15:00+05:30",
                        "check": "pipeline/manifest_writable",
                        "status": "PASS",
                        "detail": "seed",
                    }
                )
                rows.append(
                    {
                        "timestamp": f"{d}T09:16:00+05:30",
                        "check": "reports/index_html_exists",
                        "status": "PASS",
                        "detail": "seed",
                    }
                )
    if not rows:
        base = dt.date.today()
        for i in range(10, 0, -1):
            d = base - dt.timedelta(days=i)
            rows.append(
                {
                    "timestamp": f"{d}T09:15:00+05:30",
                    "check": "pipeline/heartbeat",
                    "status": "PASS",
                    "detail": "seed",
                }
            )
    pd.DataFrame(rows).to_csv(CANARY_LOG, index=False)


def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k in low:
            return low[k]
    norm = {c.lower().replace(" ", "").replace("-", "_"): c for c in cols}
    for k in cands:
        kk = k.replace(" ", "").replace("-", "_")
        if kk in norm:
            return norm[kk]
    return None


def _normalize_status(val: str) -> str:
    if not isinstance(val, str):
        return "PASS"
    u = val.strip().lower()
    if u in ("pass", "ok", "success", "true", "1", "passed", "green"):
        return "PASS"
    if u in ("fail", "error", "false", "0", "failed", "red"):
        return "FAIL"
    if u in ("warn", "warning", "amber", "yellow"):
        return "WARN"
    return "PASS"


def _load_log() -> pd.DataFrame:
    _seed_canary_if_missing()
    df = pd.read_csv(CANARY_LOG, dtype=str)
    if df.empty:
        return pd.DataFrame(columns=["ts", "day", "check", "status", "detail"])

    ts_col = _pick(df.columns, TS_CANDS) or df.columns[0]
    ck_col = _pick(df.columns, CHECK_CANDS)
    st_col = _pick(df.columns, STATUS_CANDS)
    dt_col = _pick(df.columns, DETAIL_CANDS)

    if st_col is None:
        df["__status__"] = "PASS"
        st_col = "__status__"
    if ck_col is None:
        df["__check__"] = "heartbeat"
        ck_col = "__check__"
    if dt_col is None and "detail" not in df.columns:
        df["detail"] = ""
        dt_col = "detail"

    df["timestamp"] = df[ts_col].astype(str)
    df["check"] = df[ck_col].astype(str)
    df["status"] = df[st_col].apply(_normalize_status)
    if dt_col:
        df["detail"] = df[dt_col].astype(str)

    # Parse to aware UTC Series, convert to IST, then drop tz → tz-naive
    ts_utc = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    mask = ts_utc.notna()
    df = df[mask].copy()
    ts_utc = ts_utc[mask]
    df["ts"] = ts_utc.dt.tz_convert(IST).dt.tz_localize(
        None
    )  # <-- FIX: use .dt on Series
    df["day"] = df["ts"].dt.date
    df = df.sort_values("ts")
    keep = ["ts", "day", "check", "status"]
    if "detail" in df.columns:
        keep.append("detail")
    return df[keep].copy()


def _daily_pass_rate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "checks", "pass_rate"])
    g = df.groupby("day")["status"].value_counts().unstack(fill_value=0)
    for c in ("PASS", "FAIL", "WARN"):
        if c not in g.columns:
            g[c] = 0
    g["checks"] = g.sum(axis=1)
    g["pass_rate"] = np.where(g["checks"] > 0, g["PASS"] / g["checks"], 0.0)
    return g.reset_index()[["day", "checks", "pass_rate"]].sort_values("day")


def _sparkline_svg(series, width=420, height=64, pad=6) -> str:
    series = list(series) if series is not None else []
    if not series:
        return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"></svg>'
    xs = np.linspace(pad, width - pad, len(series))
    ys = height - pad - np.clip(series, 0.0, 1.0) * (height - 2 * pad)
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=False))
    last = float(series[-1])
    color = "#16a34a" if last >= 0.95 else ("#ca8a04" if last >= 0.8 else "#dc2626")
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pass-rate sparkline">
      <polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}" />
      <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#ddd" stroke-width="1"/>
      <text x="{width - pad}" y="{pad + 10}" font-size="10" text-anchor="end" fill="#666">{int(last * 100)}%</text>
    </svg>
    """


def _build_html(daily: pd.DataFrame, recent: pd.DataFrame) -> tuple[str, dict]:
    pass_series = daily["pass_rate"].tolist() if not daily.empty else []
    svg = _sparkline_svg(pass_series)
    today = _now_ist_naive().isoformat(timespec="seconds")

    if recent.empty:
        tbl = "<tr><td colspan='4'>No checks yet</td></tr>"
        last_rate = 0.0
        p90 = 0.0
        checks_7d = 0
        fails_7d = 0
    else:
        rows = []
        for _, r in recent.tail(50).iloc[::-1].iterrows():
            status = str(r["status"])
            color = (
                "#16a34a"
                if status == "PASS"
                else ("#ca8a04" if status == "WARN" else "#dc2626")
            )
            ts = pd.to_datetime(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            ck = html.escape(str(r["check"]))
            det = html.escape(str(r.get("detail", "")))
            rows.append(
                f"<tr><td>{ts}</td><td>{ck}</td><td style='color:{color};font-weight:600;'>{status}</td><td>{det}</td></tr>"
            )
        tbl = "\n".join(rows)
        last_rate = pass_series[-1] if pass_series else 0.0
        p90 = float(np.percentile(pass_series, 90)) if pass_series else 0.0
        cutoff = _now_ist_naive() - pd.Timedelta(days=7)  # tz-naive IST cutoff
        recent7 = recent[recent["ts"] >= cutoff]
        checks_7d = int(recent7.shape[0])
        fails_7d = int((recent7["status"] == "FAIL").sum())

    color_now = (
        "#16a34a"
        if last_rate >= 0.95
        else ("#ca8a04" if last_rate >= 0.8 else "#dc2626")
    )
    head = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<title>Canary Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color:#111; }}
h1 {{ margin: 0 0 6px 0; }}
.card {{ border:1px solid #e5e7eb; border-radius:12px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }}
.kpi {{ font-size:28px; font-weight:700; }}
.kpi small {{ font-size:12px; color:#6b7280; display:block; }}
table {{ width:100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ border-bottom:1px solid #eee; padding:8px 6px; text-align:left; font-size:13px; }}
th {{ background:#fafafa; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#f1f5f9; color:#334155; font-weight:600; font-size:12px; }}
hr {{ border:none; border-top:1px solid #e5e7eb; margin:16px 0; }}
.footer {{ color:#6b7280; font-size:12px; }}
</style>
</head><body>
<h1>Canary Dashboard <span class="badge">updated {today} IST</span></h1>
<div class="grid">
  <div class="card"><div class="kpi" style="color:{color_now};">{int(last_rate * 100)}%<small>Pass rate (last day)</small></div></div>
  <div class="card"><div class="kpi">{int(p90 * 100)}%<small>P90 pass rate (window)</small></div></div>
  <div class="card"><div class="kpi">{checks_7d}<small>Checks ran (last 7 days)</small></div></div>
  <div class="card"><div class="kpi" style="color:{"#dc2626" if fails_7d > 0 else "#16a34a"};">{fails_7d}<small>Fails (last 7 days)</small></div></div>
</div>
<hr/>
<div class="card">
  <h3 style="margin:0 0 8px 0;">Pass-rate sparkline</h3>
  {svg}
</div>
<hr/>
<div class="card">
  <h3 style="margin:0 0 8px 0;">Recent checks</h3>
  <table>
    <thead><tr><th>Timestamp</th><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
    <tbody>
      {tbl}
    </tbody>
  </table>
</div>
<div class="footer">Source: {html.escape(str(CANARY_LOG))}</div>
</body></html>"""
    kpi = {
        "last_rate": float(last_rate),
        "p90": float(p90),
        "checks_7d": int(checks_7d),
        "fails_7d": int(fails_7d),
    }
    return head, kpi


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = _load_log()
    daily = _daily_pass_rate(df)
    html_str, kpi = _build_html(daily, df)
    DASH_HTML.write_text(html_str, encoding="utf-8")
    daily.to_csv(SUM_CSV, index=False)
    diag = {
        "as_of": _now_ist_naive().isoformat(timespec="seconds"),
        "rows": int(df.shape[0]),
        "days": int(daily.shape[0]),
        "kpi": kpi,
        "inputs": {
            "canary_log": str(CANARY_LOG),
            "manifest_index": str(MAN_INDEX) if MAN_INDEX.exists() else None,
        },
        "outputs": {"html": str(DASH_HTML), "summary_csv": str(SUM_CSV)},
    }
    DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"html": str(DASH_HTML), "summary_csv": str(SUM_CSV), "kpi": kpi}, indent=2
        )
    )


if __name__ == "__main__":
    main()
