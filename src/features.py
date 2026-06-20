"""
Stages 4–5 — Collapse to auction level + targets + decay panel.
Stage 4: aggregate intraday substrate → one row per auction (df_forecast).
Stage 5: forward curve changes as targets; long decay panel for local projections.
Ref: Litterman-Scheinkman (1991) for level/slope/curvature decomposition.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import (MATURITIES, N_PCA, H_FORECAST, H_DECAY, RANDOM_SEED)


# ── Stage 4 ───────────────────────────────────────────────────────────────────

def collapse_to_auction(intraday, auction_results, regime_df):
    """Aggregate intraday substrate → one feature row per auction.

    All output columns must be known by the auction-day close (no leakage).

    Parameters
    ----------
    intraday        : micro_panel with phase, micro_factor, pc1_level, pc2_slope, pc3_curv
    auction_results : Bloomberg table (one row/auction): tail_bps, bid_to_cover, …
    regime_df       : output of regime.regime_features_per_auction()
    """
    # ── post-auction micro aggregates ─────────────────────────────────────────
    post     = intraday[intraday['phase'] == 'post']
    agg_post = (post.groupby('auction_id')
                    .agg(
                        micro_eod           = ('micro_factor', 'last'),
                        micro_max_dev       = ('micro_factor', lambda s: s.abs().max()),
                        micro_vol_post      = ('micro_factor', 'std'),
                        pc1_level_eod_intra = ('pc1_level',   'last'),
                        pc2_slope_eod_intra = ('pc2_slope',   'last'),
                        pc3_curv_eod_intra  = ('pc3_curv',    'last'),
                    )
                    .reset_index())

    # ── at-auction (13:00) snapshot ───────────────────────────────────────────
    at13 = (intraday[intraday['phase'] == 'at']
            .groupby('auction_id')
            .agg(micro_at13=('micro_factor', 'last'))
            .reset_index())

    # ── pre-auction last observation ──────────────────────────────────────────
    pre     = intraday[intraday['phase'] == 'pre']
    agg_pre = (pre.groupby('auction_id')
                  .agg(micro_pre_last=('micro_factor', 'last'))
                  .reset_index())

    agg = (agg_post
           .merge(at13,    on='auction_id', how='left')
           .merge(agg_pre, on='auction_id', how='left'))
    agg['micro_postauction_drift'] = agg['micro_eod'] - agg['micro_at13']

    df = (agg
          .merge(auction_results, on='auction_id', how='left')
          .merge(regime_df,       on='auction_id', how='left'))

    # ── meta columns for PurgedKFold ─────────────────────────────────────────
    df['auction_date']      = pd.to_datetime(df['auction_date'])
    df['target_start_time'] = df['auction_date'] + pd.Timedelta('16:00:00')
    df['target_end_time']   = (df['auction_date']
                               + pd.tseries.offsets.BDay(H_FORECAST)
                               + pd.Timedelta('16:00:00'))
    return df.sort_values('auction_date').reset_index(drop=True)


# ── Stage 5 ───────────────────────────────────────────────────────────────────

def fit_daily_pca(daily_df, train_mask, n_components=N_PCA):
    """Fit PCA on training-set daily yield cross-sections.

    Returns (pca_model, score_df) where score_df columns PC1/PC2/PC3
    are projected using the train-only basis (applied to all rows).
    ⚠ Call inside every CV fold — never on the full dataset.
    """
    ycols = [f'y_{m}y' for m in MATURITIES]
    avail = [c for c in ycols if c in daily_df.columns]
    Y     = (daily_df[avail]
             .ffill()
             .fillna(daily_df[avail].median())
             .values.astype(float))

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(Y[train_mask])                  # ← train-only fit

    scores   = pca.transform(Y)
    score_df = pd.DataFrame(scores,
                            index=daily_df.index,
                            columns=['PC1', 'PC2', 'PC3'])
    return pca, score_df


def build_targets(df_forecast, daily_curve_with_pcs, h=H_FORECAST):
    """Add h-day-forward change targets to df_forecast.

    daily_curve_with_pcs must have PC1, PC2, PC3, y30_close indexed by date.
    Adds d_level_h{h}, d_slope_h{h}, d_curv_h{h}, d_30y_h{h}.
    """
    df   = df_forecast.copy()
    cmap = {'PC1': 'level', 'PC2': 'slope', 'PC3': 'curv', 'y30_close': '30y'}

    for col, name in cmap.items():
        if col not in daily_curve_with_pcs.columns:
            continue
        s = daily_curve_with_pcs[col]
        changes = []
        for d in df['auction_date']:
            d   = pd.Timestamp(d)
            d_h = d + pd.tseries.offsets.BDay(h)
            changes.append(s.get(d_h, np.nan) - s.get(d, np.nan))
        df[f'd_{name}_h{h}'] = changes
    return df


def build_decay_panel(df_forecast, daily_curve_with_pcs,
                      target_col='PC1', H=H_DECAY):
    """Long panel: (auction × horizon h) → cumulative curve change t → t+h.
    Feeds Jordà (2005) local projections in src/decay.py.
    target_start/end_time columns are needed by PurgedKFold.
    """
    c = (daily_curve_with_pcs[target_col]
         if target_col in daily_curve_with_pcs.columns
         else daily_curve_with_pcs.get('y30_close', pd.Series(dtype=float)))

    recs = []
    for _, r in df_forecast.iterrows():
        d0   = pd.Timestamp(r['auction_date'])
        base = c.get(d0, np.nan)
        for h in range(H + 1):
            dh = d0 + pd.tseries.offsets.BDay(h)
            recs.append({
                'auction_id':        r['auction_id'],
                'h':                 h,
                'cum_change':        c.get(dh, np.nan) - base,
                'auction_shock':     r.get('tail_bps',   np.nan),
                'regime_eod':        r.get('regime_eod', np.nan),
                'target_start_time': r.get('target_start_time'),
                'target_end_time':   dh + pd.Timedelta('16:00:00'),
            })
    return pd.DataFrame(recs)
