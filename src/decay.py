"""
Stage 7a — Influence decay via local projections (Jordà 2005).
Regress cumulative curve change at each horizon h on the auction shock.
Coefficient path {β_h} = impulse response function (IRF).
Fit β_h ≈ β₀·exp(-h/τ)  →  influence half-life = τ·ln(2) business days.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit


def local_projection_decay(decay_panel, shock_col='auction_shock',
                            controls=None, maxlags_hac=5):
    """Jordà (2005) local projections at each horizon h.

    OLS: cum_change ~ const + shock [+ controls], HAC standard errors.
    Returns DataFrame indexed by h with columns: beta, se, ci_lo, ci_hi, t_stat.

    Parameters
    ----------
    shock_col   : auction shock column (default 'auction_shock' = tail_bps)
    controls    : extra RHS columns (e.g. ['regime_eod', 'vix_eod'])
    maxlags_hac : Newey-West bandwidth
    """
    controls = controls or []
    rows     = []

    for h, g in decay_panel.groupby('h'):
        g = g.dropna(subset=['cum_change', shock_col])
        if len(g) < 15:
            continue
        rhs  = [shock_col] + [c for c in controls if c in g.columns]
        X    = sm.add_constant(g[rhs].fillna(0))
        try:
            res  = sm.OLS(g['cum_change'], X).fit(
                cov_type='HAC', cov_kwds={'maxlags': maxlags_hac})
            beta = res.params[shock_col]
            se   = res.bse[shock_col]
            rows.append({'h': h, 'beta': beta, 'se': se,
                         'ci_lo': beta - 1.96 * se,
                         'ci_hi': beta + 1.96 * se,
                         't_stat': res.tvalues[shock_col]})
        except Exception as exc:
            print(f'  LP h={h} failed: {exc}')

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index('h')


def fit_half_life(irf_df):
    """Fit β_h ≈ β₀·exp(-h/τ) to the IRF.
    Returns half-life = τ·ln(2) business days.

    Uses only the monotone-decay region (same sign as β₀, > 10% of peak)
    to avoid fitting the noisy tail.
    """
    df = irf_df.reset_index()
    if df.empty:
        return np.nan

    sign = np.sign(df['beta'].iloc[0])
    mask = ((df['beta'] * sign > 0) &
            (df['beta'].abs() > 0.1 * df['beta'].abs().max()))
    df   = df[mask]

    if len(df) < 3:
        print('  Too few valid IRF points for half-life fit.')
        return np.nan

    def _decay(h, b0, tau):
        return b0 * np.exp(-h / np.maximum(tau, 0.1))

    try:
        popt, _ = curve_fit(_decay, df['h'].values, df['beta'].values,
                            p0=[df['beta'].iloc[0], 3.0], maxfev=10_000)
        return float(popt[1] * np.log(2))
    except Exception:
        return np.nan
