#!/usr/bin/env python3
"""
Visualisation test — runs all six desk exhibits on synthetic test data.

Run from the project root (after generate_test_data.py + run_pipeline_test.py):
    python tests/test_visualisation.py

Saves PNGs to: tests/outputs/exhibits/
"""
import sys, os, time
sys.path.insert(0, os.path.abspath('.'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path('tests/outputs/exhibits')
OUT_DIR.mkdir(parents=True, exist_ok=True)

from config import CACHE_DIR, TARGET
from src.decay import local_projection_decay
from src.visualisation import (
    exhibit_event_study,
    exhibit_regime_overlay,
    exhibit_transition_matrix,
    exhibit_decay_curve,
    exhibit_feature_importance,
    exhibit_forecast_scorecard,
)

results = {}

def _save(name, fig, t_start):
    path = OUT_DIR / f'{name}.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    elapsed = time.time() - t_start
    print(f'  ✓  {name}  ({elapsed:.1f}s)  → {path.name}')
    results[name] = True

def _fail(name, exc):
    results[name] = False
    print(f'  ✗  {name}  — {type(exc).__name__}: {exc}')

t0 = time.time()

# ── Load all caches ───────────────────────────────────────────────────────────
print('\nLoading caches…')
intraday_df  = pd.read_parquet(CACHE_DIR / 'intraday_stage1.parquet')
bloomberg_df = pd.read_parquet(CACHE_DIR / 'bloomberg.parquet')
df_forecast  = pd.read_parquet(CACHE_DIR / 'df_forecast.parquet')
decay_panel  = pd.read_parquet(CACHE_DIR / 'decay_panel.parquet')
oos_preds    = pd.read_parquet(CACHE_DIR / 'oos_predictions.parquet')

print(f'  intraday    : {len(intraday_df):,} rows')
print(f'  df_forecast : {len(df_forecast)} auctions')
print(f'  decay_panel : {len(decay_panel):,} rows')
print(f'  oos_preds   : {len(oos_preds)} rows')

# Pre-compute IRF (shared by Ex4 and displayed in summary)
print('\nPre-computing IRF…')
irf_full = local_projection_decay(decay_panel, shock_col='auction_shock')
print(f'  {len(irf_full)} horizons  |  '
      f'β_h range [{irf_full["beta"].min():.4f}, {irf_full["beta"].max():.4f}]')

# Try loading HMM
hmm_model = None
try:
    from src.regime import HAVE_HMM, fit_pooled_hmm
    if HAVE_HMM:
        print('\nFitting HMM for Exhibits 2 & 3…')
        hmm_model = fit_pooled_hmm(intraday_df)
        print(f'  HMM fitted  K={hmm_model.n_components}')
    else:
        print('\n  hmmlearn not installed — Exhibits 2 & 3 will be skipped.')
except Exception as e:
    print(f'\n  HMM skipped: {e}')


# ── Exhibit 1 — Event Study ───────────────────────────────────────────────────
print('\n─── Exhibit 1 — Event Study ────────────────────────────────')
t = time.time()
try:
    fig = exhibit_event_study(intraday_df, df_forecast)
    _save('ex1_event_study', fig, t)
except Exception as e:
    _fail('ex1_event_study', e)


# ── Exhibit 2 — Regime Overlay ────────────────────────────────────────────────
print('\n─── Exhibit 2 — Regime Overlay ─────────────────────────────')
t = time.time()
if hmm_model is None:
    print('  Skipped — no HMM model')
    results['ex2_regime_overlay'] = 'skipped'
else:
    try:
        fig = exhibit_regime_overlay(intraday_df, hmm_model)
        _save('ex2_regime_overlay', fig, t)
    except Exception as e:
        _fail('ex2_regime_overlay', e)


# ── Exhibit 3 — Transition Matrix ────────────────────────────────────────────
print('\n─── Exhibit 3 — Transition Matrix ──────────────────────────')
t = time.time()
if hmm_model is None:
    print('  Skipped — no HMM model')
    results['ex3_transition_matrix'] = 'skipped'
else:
    try:
        fig = exhibit_transition_matrix(hmm_model)
        _save('ex3_transition_matrix', fig, t)
    except Exception as e:
        _fail('ex3_transition_matrix', e)


# ── Exhibit 4 — Decay Curve ───────────────────────────────────────────────────
print('\n─── Exhibit 4 — Decay Curve ────────────────────────────────')
t = time.time()
try:
    fig = exhibit_decay_curve(decay_panel, df_forecast, irf_full=irf_full)
    _save('ex4_decay_curve', fig, t)
except Exception as e:
    _fail('ex4_decay_curve', e)


# ── Exhibit 5 — Feature Importance ───────────────────────────────────────────
print('\n─── Exhibit 5 — Feature Importance ─────────────────────────')
t = time.time()
try:
    fig = exhibit_feature_importance(df_forecast, oos_preds)
    _save('ex5_feature_importance', fig, t)
except Exception as e:
    _fail('ex5_feature_importance', e)


# ── Exhibit 6 — Forecast Scorecard ───────────────────────────────────────────
print('\n─── Exhibit 6 — Forecast Scorecard ─────────────────────────')
t = time.time()
try:
    fig = exhibit_forecast_scorecard(oos_preds)
    _save('ex6_forecast_scorecard', fig, t)
except Exception as e:
    _fail('ex6_forecast_scorecard', e)


# ── Summary ───────────────────────────────────────────────────────────────────
total = time.time() - t0
print(f'\n{"═"*56}')
print(f'  VISUALISATION TEST  ({total:.1f}s)')
print(f'{"═"*56}')
for name, status in results.items():
    icon = '✓' if status is True else ('—' if status == 'skipped' else '✗')
    print(f'  {icon}  {name}')

passed  = sum(1 for v in results.values() if v is True)
skipped = sum(1 for v in results.values() if v == 'skipped')
failed  = sum(1 for v in results.values() if v is False)
print(f'\n  {passed} passed  |  {skipped} skipped  |  {failed} failed')
print(f'  Figures saved → {OUT_DIR}')
print(f'{"═"*56}')
