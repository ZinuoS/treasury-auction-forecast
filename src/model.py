"""
Stage 6 — h=1 forecast model + leakage-safe CV pipeline.
RandomForestRegressor (shallow, small N≈220).
ALL preprocessing models (PCA, macro Ridge, HMM) are refit inside each fold.
Ref: Breiman (2001); CLAUDE.md §"Non-negotiable rules".
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from config import (
    FEATURES, TARGET, MACRO_COLS,
    RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS, RANDOM_SEED,
    CV_N_SPLITS, CV_MIN_TRAIN, N_PCA,
    MATURITIES,
)
from src.curve    import fit_pca_basis, add_pca_projections
from src.macro    import macro_baseline
from src.regime   import fit_pooled_hmm, regime_features_per_auction, HAVE_HMM
from src.features import collapse_to_auction, fit_daily_pca, build_targets


# ── Point forecast helpers ────────────────────────────────────────────────────

def regime_baseline(train, test, target=TARGET):
    """Regime-conditional mean — the benchmark any model must beat.
    Falls back to unconditional mean if regime_eod column is absent.
    """
    if 'regime_eod' not in train.columns or 'regime_eod' not in test.columns:
        fallback = train[target].mean()
        return np.full(len(test), fallback)
    means    = train.groupby('regime_eod')[target].mean()
    fallback = train[target].mean()
    return test['regime_eod'].map(means).fillna(fallback).values


def fit_rf(train, test, features=None, target=TARGET):
    """Fit shallow RF; return (predictions, feature_importance Series).
    Shallow (max_depth=4, min_samples_leaf=5) appropriate for N≈220.
    """
    features = features or FEATURES
    avail    = [f for f in features if f in train.columns and f in test.columns]
    rf       = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED)
    rf.fit(train[avail].fillna(0), train[target])
    pred = rf.predict(test[avail].fillna(0))
    imp  = pd.Series(rf.feature_importances_, index=avail).sort_values(ascending=False)
    return pred, imp


# ── Leakage-safe CV fold ──────────────────────────────────────────────────────

def run_cv_fold(intraday_all, auction_df, daily_df, macro_cols,
                train_ids, test_ids, target=TARGET):
    """Execute one fold of the leakage-safe expanding walk-forward.

    Inside this fold (train_ids only):
      1. Fit PCA on intraday yields          → project all (train+test)
      2. Fit Ridge macro baseline            → macro_fair_30y + macro_resid_daily
      3. Fit HMM on micro_factor sequences   → regime features
      4. Collapse to auction level           → df_forecast rows
      5. Fit daily PCA for PC targets        → d_level_h1, d_slope_h1, …
      6. Fit RF on train → predict test

    Returns dict with predictions, feature importance, fitted models.
    """
    train_ids, test_ids = set(train_ids), set(test_ids)
    all_ids             = train_ids | test_ids

    intra_tr  = intraday_all[intraday_all['auction_id'].isin(train_ids)]
    intra_all = intraday_all[intraday_all['auction_id'].isin(all_ids)]
    auc_tr    = auction_df[auction_df['auction_id'].isin(train_ids)]
    auc_all   = auction_df[auction_df['auction_id'].isin(all_ids)]

    # 1. Intraday PCA (train-only basis)
    pca_intra = fit_pca_basis(intra_tr)
    intra_all = add_pca_projections(intra_all, pca_intra)

    # 2. Macro baseline (train-only)
    train_dates = set(pd.to_datetime(auc_tr['auction_date']).dt.date)
    train_mask  = np.array([d.date() in train_dates
                             for d in pd.to_datetime(daily_df.index)])
    daily_sub   = macro_baseline(daily_df, macro_cols, train_mask)

    # 3. HMM (train-only)
    hmm_model = None
    regime_df = pd.DataFrame({'auction_id': list(all_ids)})
    if HAVE_HMM:
        try:
            hmm_model = fit_pooled_hmm(intra_tr)
            regime_df = regime_features_per_auction(intra_all, hmm_model)
        except Exception as exc:
            print(f'  HMM skipped: {exc}')

    # 4. Collapse to auction level
    df_fc = collapse_to_auction(intra_all, auc_all, regime_df)
    df_fc = df_fc.merge(
        daily_sub[['macro_fair_30y', 'macro_resid_daily']],
        left_on='auction_date', right_index=True, how='left')

    # 5. Daily PCA for targets (train-only)
    ycols_d = [c for c in [f'y_{m}y' for m in MATURITIES] if c in daily_df.columns]
    if ycols_d:
        _, pc_scores   = fit_daily_pca(daily_df, train_mask)
        daily_with_pcs = daily_sub.join(pc_scores)
    else:
        daily_with_pcs = daily_sub

    df_fc = build_targets(df_fc, daily_with_pcs)

    # 6. RF
    train_fc = df_fc[df_fc['auction_id'].isin(train_ids)]
    test_fc  = df_fc[df_fc['auction_id'].isin(test_ids)]

    if target not in train_fc.columns or train_fc[target].isna().all():
        print('  Target unavailable — skipping fold.')
        return None

    pred_rf,  imp  = fit_rf(train_fc, test_fc, target=target)
    pred_base      = regime_baseline(train_fc, test_fc, target=target)

    return {
        'fold_n_train':       len(train_ids),
        'fold_n_test':        len(test_ids),
        'actual':             test_fc[target].values,
        'pred_rf':            pred_rf,
        'pred_base':          pred_base,
        'feature_importance': imp,
        'pca_model':          pca_intra,
        'hmm_model':          hmm_model,
        'df_test':            test_fc,
    }


def walk_forward_leakage_safe(intraday_all, auction_df, daily_df,
                               macro_cols=None, n_splits=None, target=TARGET):
    """Expanding walk-forward CV with all preprocessing models refit per fold.

    First fold trains on the earliest ≥CV_MIN_TRAIN auctions, then expands.
    Returns list of fold result dicts.
    """
    macro_cols = macro_cols or MACRO_COLS
    n_splits   = n_splits   or CV_N_SPLITS

    auctions  = auction_df.sort_values('auction_date').reset_index(drop=True)
    aids      = auctions['auction_id'].values
    n         = len(aids)
    min_train = max(CV_MIN_TRAIN, n // (n_splits + 1))
    fold_size = max(1, (n - min_train) // n_splits)

    results = []
    for i in range(n_splits):
        lo = min_train + i * fold_size
        hi = min_train + (i + 1) * fold_size if i < n_splits - 1 else n
        if lo >= n:
            break
        train_ids = set(aids[:lo])
        test_ids  = set(aids[lo:hi])
        print(f'\n── Fold {i+1}/{n_splits}  train={len(train_ids)}  test={len(test_ids)} ──')
        result = run_cv_fold(intraday_all, auction_df, daily_df,
                             macro_cols, train_ids, test_ids, target=target)
        if result is not None:
            results.append(result)
    return results


def summarise_cv(results, target=TARGET):
    """Aggregate CV results: print R²/RMSE, return mean feature importance."""
    actual  = np.concatenate([r['actual']    for r in results])
    pred_rf = np.concatenate([r['pred_rf']   for r in results])
    pred_bl = np.concatenate([r['pred_base'] for r in results])
    mask    = np.isfinite(actual)

    r2_rf  = r2_score(actual[mask], pred_rf[mask])
    r2_bl  = r2_score(actual[mask], pred_bl[mask])
    rmse_rf = np.sqrt(mean_squared_error(actual[mask], pred_rf[mask]))
    rmse_bl = np.sqrt(mean_squared_error(actual[mask], pred_bl[mask]))

    print(f'h=1 RF     R²={r2_rf:.4f}  RMSE={rmse_rf:.4f}')
    print(f'h=1 Regime R²={r2_bl:.4f}  RMSE={rmse_bl:.4f}')

    mean_imp = (pd.concat([r['feature_importance'] for r in results], axis=1)
                  .mean(axis=1)
                  .sort_values(ascending=False))
    print(f'\nTop-10 features (mean importance across {len(results)} folds):')
    print(mean_imp.head(10).to_string())
    return mean_imp, {'r2_rf': r2_rf, 'r2_base': r2_bl,
                      'rmse_rf': rmse_rf, 'rmse_base': rmse_bl}
