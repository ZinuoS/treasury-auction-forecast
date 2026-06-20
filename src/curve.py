"""
Stage 1 — Per-minute Svensson curve fit + micro factor + PCA infrastructure.
Refs: Gürkaynak-Sack-Wright (2007) for parametrisation;
      Hu-Pan-Wang (2013) for OTR-residual-as-illiquidity-factor.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from config import MATURITIES, N_PCA, CACHE_DIR, RANDOM_SEED


# ── Svensson parametrisation ──────────────────────────────────────────────────

def svensson(tau, b0, b1, b2, b3, l1, l2):
    """Svensson (1994) extended Nelson-Siegel.
    τ: maturity in years (scalar or array).
    Decay params l1, l2 must be > 0 (enforced by bounds in caller).
    """
    eps     = 1e-8
    l1, l2  = max(l1, eps), max(l2, eps)
    tau     = np.asarray(tau, dtype=float)
    t1      = (1 - np.exp(-tau / l1)) / (tau / l1)
    t2      = t1 - np.exp(-tau / l1)
    t3      = (1 - np.exp(-tau / l2)) / (tau / l2) - np.exp(-tau / l2)
    return b0 + b1 * t1 + b2 * t2 + b3 * t3


def fit_curve_one_minute(taus, yields, p0=None):
    """Fit Svensson to one minute's yield cross-section.

    Parameters
    ----------
    taus   : array of maturities in years (e.g. [2,3,5,7,10,20,30])
    yields : array of mid-yields in % (NaN allowed — at least 4 finite needed)
    p0     : warm-start parameters from previous minute (speeds convergence)

    Returns
    -------
    (params_list, fitted_30y, noise_xsec)  or  (None, nan, nan) on failure.
    """
    taus   = np.asarray(taus,   dtype=float)
    yields = np.asarray(yields, dtype=float)
    mask   = np.isfinite(yields)
    if mask.sum() < 4:
        return None, np.nan, np.nan

    taus_m, yields_m = taus[mask], yields[mask]

    if p0 is None:
        b0_0 = float(yields_m[-1])                   # level ≈ long-end yield
        b1_0 = float(yields_m[0] - yields_m[-1])     # slope ≈ short – long
        p0   = [b0_0, b1_0, 0.0, 0.0, 2.0, 5.0]

    lo = [-np.inf, -np.inf, -10., -10., 0.1, 0.1]
    hi = [ np.inf,  np.inf,  10.,  10., 30., 30.]

    try:
        popt, _ = curve_fit(svensson, taus_m, yields_m,
                            p0=p0, bounds=(lo, hi),
                            maxfev=10_000, method='trf')
    except Exception:
        return None, np.nan, np.nan

    fitted_all = svensson(taus, *popt)
    noise_xsec = float(np.sqrt(np.mean((yields[mask] - fitted_all[mask]) ** 2)))
    fitted_30y = float(svensson(30.0, *popt))
    return list(popt), fitted_30y, noise_xsec


def add_micro_factor(df, use_cache=True, progress=True):
    """Fit Svensson at every (auction_id × timestamp_et); attach:
       fitted_30y, noise_xsec, micro_factor, wi_otr_spread, bid_ask_spread.

    Warm-starts across minutes within each auction for speed + continuity.
    Heavy (~105K fits for 220 auctions) — cached to cache/intraday_curved.parquet.
    """
    cache = CACHE_DIR / 'intraday_curved.parquet'
    if use_cache and cache.exists():
        print(f'Loading Svensson fits from cache ({cache.name})…')
        cached = pd.read_parquet(cache)
        return df.merge(cached, on=['auction_id', 'timestamp_et'], how='left')

    ycols   = [f'y_{m}y' for m in MATURITIES]
    taus    = np.array(MATURITIES, dtype=float)
    records = []
    n_auc   = df['auction_id'].nunique()

    for i, (aid, grp) in enumerate(df.groupby('auction_id', sort=False)):
        prev = None
        for ts, g in grp.sort_values('timestamp_et').groupby('timestamp_et', sort=True):
            yv            = g[ycols].iloc[0].values.astype(float)
            params, f30y, noise = fit_curve_one_minute(taus, yv, p0=prev)
            if params is not None:
                prev = params
            records.append({'auction_id': aid, 'timestamp_et': ts,
                            'fitted_30y': f30y, 'noise_xsec': noise})
        if progress:
            print(f'  Svensson fit: {i+1}/{n_auc} auctions', end='\r')
    print()

    fit_df = pd.DataFrame(records)
    df     = df.merge(fit_df, on=['auction_id', 'timestamp_et'], how='left')

    df['micro_factor']   = df['otr_30y_yield'] - df['fitted_30y']
    df['wi_otr_spread']  = (df.get('wi_30y_yield',  pd.Series(dtype=float))
                             - df['otr_30y_yield'])
    df['bid_ask_spread'] = (df.get('ask', pd.Series(dtype=float))
                             - df.get('bid', pd.Series(dtype=float)))

    # Cache only the derived columns
    cache_cols = ['auction_id', 'timestamp_et', 'fitted_30y', 'noise_xsec',
                  'micro_factor', 'wi_otr_spread', 'bid_ask_spread']
    df[cache_cols].to_parquet(cache, index=False)
    print(f'  Cached → {cache}')
    return df


# ── PCA infrastructure ────────────────────────────────────────────────────────

def fit_pca_basis(intraday_train, n_components=N_PCA):
    """Fit PCA on training-set yield cross-sections.

    ⚠ Always call with ONLY training-set rows inside every CV fold.
    Returns fitted sklearn.decomposition.PCA object.
    """
    ycols = [f'y_{m}y' for m in MATURITIES]
    Y     = intraday_train[ycols].dropna(how='any').values
    pca   = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(Y)
    ev = pca.explained_variance_ratio_.cumsum()
    print(f'  PCA var explained: '
          f'PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  PC3={ev[2]:.1%}')
    return pca


def add_pca_projections(df, pca_model):
    """Project per-minute yields onto a pre-fit PCA basis.
    Adds pc1_level, pc2_slope, pc3_curv.  NaN rows imputed with column mean.
    """
    ycols     = [f'y_{m}y' for m in MATURITIES]
    Y         = df[ycols].values.astype(float)
    col_means = np.nanmean(Y, axis=0)
    Y_filled  = np.where(np.isfinite(Y), Y, col_means)
    scores    = pca_model.transform(Y_filled)
    df        = df.copy()
    df['pc1_level'] = scores[:, 0]
    df['pc2_slope'] = scores[:, 1]
    df['pc3_curv']  = scores[:, 2]
    return df
