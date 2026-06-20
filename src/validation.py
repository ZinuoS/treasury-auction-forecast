"""
Stage 7b — Leakage-safe validation.
• h=1 forecast: expanding walk-forward (non-overlapping targets → in src/model.py).
• Decay layer: PurgedKFold — purged + embargoed CV for overlapping-label panels.
• Event study: model-independent sanity check on micro_factor around t=0.
Ref: López de Prado (2018) Advances in Financial ML, ch. 7.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import EMBARGO_TD
from src.decay import local_projection_decay


# ── Inline PurgedKFold ────────────────────────────────────────────────────────

class PurgedKFold:
    """Expanding-window purged + embargoed CV for overlapping-label panels.

    Purge  : removes training obs whose label window overlaps the test
             observation's feature window (avoids label look-ahead).
    Embargo: time buffer after the test window before training resumes
             (avoids leakage via autocorrelated slow-moving features).

    Inline from López de Prado (2018) ch. 7.
    Use when mlfinlab is blocked by the package mirror.

    X must have columns 'target_start_time' and 'target_end_time'
    (populated by features.build_decay_panel).
    """

    def __init__(self, n_splits=5, embargo_td=None):
        self.n_splits   = n_splits
        self.embargo_td = embargo_td if embargo_td is not None else EMBARGO_TD

    def split(self, X, y=None, groups=None):
        n    = len(X)
        idx  = np.arange(n)
        t_st = pd.to_datetime(X['target_start_time'].values)
        t_en = pd.to_datetime(X['target_end_time'].values)

        fold_size = n // (self.n_splits + 1)
        min_train = fold_size

        for fold in range(self.n_splits):
            lo = min_train + fold * fold_size
            hi = (min_train + (fold + 1) * fold_size
                  if fold < self.n_splits - 1 else n)
            if lo >= n:
                break

            test_idx    = idx[lo:hi]
            test_start  = t_st[lo]
            test_end    = t_en[hi - 1]
            embargo_end = test_end + self.embargo_td

            # Purge: label overlaps test window
            purge   = (t_en[:lo] > test_start) & (t_st[:lo] < test_end)
            # Embargo: label ends within embargo buffer after test window
            embargo = (t_en[:lo] >= test_start) & (t_en[:lo] <= embargo_end)

            train_idx = idx[:lo][~(purge | embargo)[:lo]]
            if len(train_idx) < 10:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ── Purged CV for decay layer ─────────────────────────────────────────────────

def decay_cv(decay_panel, shock_col='auction_shock',
             n_splits=5, embargo_td=None):
    """OOS evaluation of local-projection IRF using PurgedKFold.

    For each fold: fit LP on train → predict cum_change on test.
    Returns (oos_df, r2_by_horizon).
    """
    embargo_td = embargo_td or EMBARGO_TD
    panel      = decay_panel.sort_values('target_start_time').reset_index(drop=True)
    pkf        = PurgedKFold(n_splits=n_splits, embargo_td=embargo_td)

    oos = []
    for tr_idx, te_idx in pkf.split(panel):
        train = panel.iloc[tr_idx]
        test  = panel.iloc[te_idx]
        irf   = local_projection_decay(train, shock_col=shock_col)

        for _, row in test.iterrows():
            h = int(row['h'])
            if h not in irf.index:
                continue
            beta = irf.loc[h, 'beta']
            pred = beta * row[shock_col] if np.isfinite(row[shock_col]) else np.nan
            oos.append({'h': h, 'actual': row['cum_change'], 'pred': pred})

    oos_df = pd.DataFrame(oos)
    if oos_df.empty:
        print('No OOS records — check panel size and CV settings.')
        return oos_df, pd.Series(dtype=float)

    r2_by_h = (oos_df.dropna()
                     .groupby('h')
                     .apply(lambda g: r2_score(g['actual'], g['pred'])
                            if len(g) > 2 else np.nan))
    print('OOS R² by horizon:')
    print(r2_by_h.to_string())
    return oos_df, r2_by_h


# ── Event study ───────────────────────────────────────────────────────────────

def event_study(intraday):
    """Mean micro_factor by event_minute across all auctions.
    Model-independent sanity check: concession build-up and post-auction reversal.
    """
    return (intraday
            .groupby('event_minute')['micro_factor']
            .agg(['mean', 'std', 'count'])
            .rename(columns={'mean': 'micro_mean',
                             'std':  'micro_std',
                             'count': 'n'}))
