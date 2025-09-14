# Toy position sizing + paper portfolio snapshotting (hook up later if needed)
import pandas as pd


def size_position(equity, price, atr, cfg):
    risk_amt = equity * cfg["risk_per_trade_pct"]
    stop = price - cfg["atr_mult_stop"] * atr
    per_share_risk = max(price - stop, 0.01)
    qty_risk = int(risk_amt // per_share_risk)
    max_val = equity * cfg["max_pos_value_pct"]
    qty_val = int(max_val // max(price, 0.01))
    return max(0, min(qty_risk, qty_val))


def update_positions(
    buylist, prices_next_open, positions_csv, portfolio_csv, cfg, init_equity=1_000_000.0
):
    try:
        positions = pd.read_csv(positions_csv)
    except FileNotFoundError:
        positions = pd.DataFrame(columns=["symbol", "qty", "avg_price"])
    try:
        portfolio = pd.read_csv(portfolio_csv)
    except FileNotFoundError:
        portfolio = pd.DataFrame(columns=["date", "equity"])

    equity = portfolio["equity"].iloc[-1] if not portfolio.empty else init_equity
    held = set(positions["symbol"])
    entries = []
    for sym in buylist["symbol"]:
        if sym in held:
            continue
        entry_price = float(buylist.set_index("symbol").loc[sym, "close"])
        atr = float(buylist.set_index("symbol").loc[sym, "atr"])
        if entry_price > 0:
            qty = size_position(equity, entry_price, atr, cfg)
            if qty > 0:
                entries.append({"symbol": sym, "qty": qty, "avg_price": entry_price})

    if entries:
        positions = pd.concat([positions, pd.DataFrame(entries)], ignore_index=True)

    snap = {"date": pd.Timestamp.today().date().isoformat(), "equity": equity}
    portfolio = pd.concat([portfolio, pd.DataFrame([snap])], ignore_index=True)

    positions.to_csv(positions_csv, index=False)
    portfolio.to_csv(portfolio_csv, index=False)
    return positions, portfolio
