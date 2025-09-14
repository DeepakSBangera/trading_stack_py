# Convert elasticity to recommended price within +/-band
import numpy as np

def optimal_price_from_elasticity(cost, elasticity_abs, p_min=None, p_max=None):
    if elasticity_abs is None or np.isnan(elasticity_abs):
        return np.nan
    if elasticity_abs <= 1:
        p_star = p_max if p_max is not None else cost * 1.05
    else:
        p_star = cost * elasticity_abs / (elasticity_abs - 1.0)
    if p_min is not None:
        p_star = max(p_star, p_min)
    if p_max is not None:
        p_star = min(p_star, p_max)
    return float(p_star)

def apply_recommendations(df, price_col, qty_col, cost_col, elasticity_abs, pct_band=0.1):
    current_price = df[price_col].median()
    c = df[cost_col].median() if cost_col in df else current_price * 0.6
    p_min = current_price * (1 - pct_band)
    p_max = current_price * (1 + pct_band)
    p_star = optimal_price_from_elasticity(c, elasticity_abs, p_min, p_max)
    return dict(current=current_price, cost=c, recommended=p_star, band=(p_min, p_max))
