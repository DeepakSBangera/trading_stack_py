"""
Simple return attribution utilities.

Assumptions:
- weights_df: wide DataFrame indexed by date, columns = symbols, values = weight (0..1).
- rets_df:    wide DataFrame indexed by date, columns = symbols, values = simple return per period.
- Both aligned to same frequency; weights are applied with a 1-period lag (prev close).

APIs:
- contribution_by_symbol(weights_df, rets_df) -> (contrib_df, summary_df)
- contribution_by_group(weights_df, rets_df, group_map: dict) -> (group_contrib_df, group_summary_df)

If rets_df is missing or empty, you can approximate contributions by
proportional allocation of the *portfolio return* (crude); not provided here
to avoid confusion—feed real asset returns if you can.
"""

import numpy as np
import pandas as pd


def _align(
    weights_df: pd.DataFrame, rets_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = weights_df.sort_index()
    rets = rets_df.sort_index()
    common_index = weights.index.intersection(rets.index)
    common_cols = weights.columns.intersection(rets.columns)
    if len(common_index) == 0 or len(common_cols) == 0:
        raise ValueError(
            "No alignment between weights and returns (index/columns mismatch)."
        )
    w = weights.loc[common_index, common_cols].copy()
    r = rets.loc[common_index, common_cols].copy()
    # lag weights by 1 period (prev holdings drive today's P&L)
    w = w.shift(1).fillna(0.0)
    # sanitize
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return w, r


def contribution_by_symbol(
    weights_df: pd.DataFrame, rets_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      contrib_df: same shape as inputs; per-symbol daily contribution (weights_lag * returns)
      summary_df: rows = symbols, cols = ['total_contrib', 'avg_contrib', 'hit_rate']
    """
    w, r = _align(weights_df, rets_df)
    contrib = w * r
    # summary per symbol
    total = contrib.sum(axis=0)
    avg = contrib.mean(axis=0)
    hit = (contrib > 0).sum(axis=0) / contrib.shape[0] if contrib.shape[0] else 0.0
    summary = pd.DataFrame(
        {"total_contrib": total, "avg_contrib": avg, "hit_rate": hit}
    )
    summary.index.name = "symbol"
    return contrib, summary.sort_values("total_contrib", ascending=False)


def contribution_by_group(
    weights_df: pd.DataFrame, rets_df: pd.DataFrame, group_map: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grouped attribution using a symbol->group map.
    Unknown symbols go to group '<OTHER>'.
    """
    contrib, _ = contribution_by_symbol(weights_df, rets_df)
    # build a column MultiIndex with group labels
    groups = [group_map.get(sym, "<OTHER>") for sym in contrib.columns]
    grouped = {}
    for g in set(groups):
        cols = [c for c, gg in zip(contrib.columns, groups) if gg == g]
        grouped[g] = (
            contrib[cols].sum(axis=1) if cols else pd.Series(0.0, index=contrib.index)
        )
    group_df = pd.DataFrame(grouped).reindex(index=contrib.index).fillna(0.0)
    # summary
    total = group_df.sum(axis=0)
    avg = group_df.mean(axis=0)
    hit = (group_df > 0).sum(axis=0) / group_df.shape[0] if group_df.shape[0] else 0.0
    summary = pd.DataFrame(
        {"total_contrib": total, "avg_contrib": avg, "hit_rate": hit}
    )
    summary.index.name = "group"
    return group_df, summary.sort_values("total_contrib", ascending=False)
