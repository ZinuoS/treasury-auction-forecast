#!/usr/bin/env python3
"""
Generate synthetic test data for the treasury-auction-forecast pipeline.

Run from the project root:
    python tests/generate_test_data.py

Writes:
    data/intraday.csv                       — minute-bar intraday tape (60 auctions)
    data/bloomberg_results.csv              — one row per auction
    data/cache/intraday_curved.parquet      — pre-computed Svensson outputs (skip slow fitting)

Design choices:
    • 60 auctions on consecutive business days — ensures d+BDay(1) lands on another
      auction date, so build_targets() returns non-NaN targets.
    • 3 regimes (calm / normal / stressed) of 20 auctions each, shuffled chronologically.
    • micro_factor has a realistic pre-auction build-up and post-auction exponential decay.
    • tail_bps is correlated with micro_factor peak so the LP IRF has signal.
    • Svensson-derived columns are pre-computed analytically and saved as the cache
      parquet so add_micro_factor() skips the expensive per-minute fitting step.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

from config import DATA_DIR, CACHE_DIR, MATURITIES

SEED = 42
rng  = np.random.default_rng(SEED)


# ── Svensson helper (local copy — no import dependency at data-gen stage) ─────
def _sv(tau, b0, b1, b2, b3, l1=2.0, l2=8.0):
    eps = 1e-8
    l1  = max(l1, eps)
    l2  = max(l2, eps)
    tau = np.asarray(tau, float)
    t1  = (1 - np.exp(-tau / l1)) / (tau / l1)
    t2  = t1 - np.exp(-tau / l1)
    t3  = (1 - np.exp(-tau / l2)) / (tau / l2) - np.exp(-tau / l2)
    return b0 + b1 * t1 + b2 * t2 + b3 * t3


# ── Generation parameters ─────────────────────────────────────────────────────
N_AUCTIONS   = 60
EVENT_WINDOW = 120   # ± minutes generated; config.EVENT_WINDOW_MIN=240 so all pass filter
TAUS         = np.array(MATURITIES, dtype=float)  # [2,3,5,7,10,20,30]

# Consecutive business days — critical for build_targets() to find d+BDay(1) in index
auction_dates = pd.bdate_range('2020-01-02', periods=N_AUCTIONS)

# 20 calm / 20 normal / 20 stressed, shuffled
regime_ids = np.repeat([0, 1, 2], [N_AUCTIONS // 3] * 3)
rng.shuffle(regime_ids)

REGIME = {
    0: dict(mf_peak_bps=3.0,  mf_vol_bps=0.6,  decay_half_min=20,  tail_base=0.3,  label='Calm'),
    1: dict(mf_peak_bps=6.0,  mf_vol_bps=1.5,  decay_half_min=40,  tail_base=0.8,  label='Normal'),
    2: dict(mf_peak_bps=12.0, mf_vol_bps=3.5,  decay_half_min=90,  tail_base=2.0,  label='Stressed'),
}

# Yield-level random walk (b0 parameter)
b0_path = 4.0 + np.cumsum(rng.normal(0, 0.04, N_AUCTIONS))
b0_path = np.clip(b0_path, 2.5, 6.5)

MINUTES = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)  # 241 minutes per auction


# ── Generate rows ─────────────────────────────────────────────────────────────
intra_rows = []
bb_rows    = []
cache_rows = []

for i, (adate, reg) in enumerate(zip(auction_dates, regime_ids)):
    aid = adate.strftime('%Y%m%d')
    p   = REGIME[reg]
    b0  = b0_path[i]
    b1  = rng.normal(-0.4, 0.08)    # slope factor
    b2  = rng.normal(0.8,  0.15)    # curvature factor
    b3  = rng.normal(0.0,  0.05)

    base_y = _sv(TAUS, b0, b1, b2, b3)          # base yield curve (%)
    fit30  = float(_sv(30.0, b0, b1, b2, b3))   # 30Y fair value (%)

    # tail_bps: correlated with micro_factor peak (this creates the IRF signal)
    mf_peak_bps = p['mf_peak_bps'] + rng.normal(0, p['mf_vol_bps'] * 0.5)
    tail_bps    = mf_peak_bps * rng.uniform(0.4, 0.9) + rng.normal(0, p['tail_base'])

    for m in MINUTES:
        ts = adate + pd.Timedelta(hours=13) + pd.Timedelta(minutes=int(m))

        # micro_factor path (bps, then convert to % for yield arithmetic)
        if m < 0:
            # linear pre-auction build-up from 0 to mf_peak_bps
            mf_bps = mf_peak_bps * (-m / EVENT_WINDOW)
        else:
            # exponential decay post-auction
            mf_bps = mf_peak_bps * np.exp(-m * np.log(2) / p['decay_half_min'])

        mf_bps += rng.normal(0, p['mf_vol_bps'] * 0.12)   # intraday noise
        mf_pct  = mf_bps / 100                              # convert to % yield

        # Cross-section yields (base curve + small random noise)
        yields_t = base_y + rng.normal(0, 0.025, len(TAUS))

        # OTR 30Y = fair value + micro_factor
        otr30  = fit30 + mf_pct
        wi30   = otr30 - rng.uniform(0.003, 0.012)  # WI slightly cheaper than OTR
        half_spread = rng.uniform(0.001, 0.004)

        intra_row = {
            'auction_id':    aid,
            'timestamp_et':  ts.strftime('%Y-%m-%d %H:%M:%S'),
            'otr_30y_yield': round(otr30, 6),
            'wi_30y_yield':  round(wi30, 6),
            'bid':           round(otr30 - half_spread, 6),
            'ask':           round(otr30 + half_spread, 6),
            'volume':        int(rng.integers(500, 20_000)),
        }
        for mat, y in zip(MATURITIES, yields_t):
            intra_row[f'y_{mat}y'] = round(float(y), 6)
        intra_rows.append(intra_row)

        # Cache row: pre-computed Svensson outputs so add_micro_factor() hits cache
        cache_rows.append({
            'auction_id':    aid,
            'timestamp_et':  ts,
            'fitted_30y':    round(fit30, 6),
            'noise_xsec':    round(float(rng.uniform(0.01, 0.05)), 6),
            'micro_factor':  round(mf_pct, 6),      # in % (otr - fitted)
            'wi_otr_spread': round(wi30 - otr30, 6),
            'bid_ask_spread':round(2 * half_spread, 6),
        })

    # Bloomberg row
    y30_close = fit30 + rng.normal(0, 0.008)
    bb_rows.append({
        'auction_id':        aid,
        'auction_date':      adate.strftime('%Y-%m-%d'),
        'cusip':             f'912810{i:03d}',
        'tail_bps':          round(float(tail_bps), 3),
        'bid_to_cover':      round(float(rng.uniform(2.1, 3.5)), 3),
        'indirect_pct':      round(float(rng.uniform(55, 75)), 1),
        'direct_pct':        round(float(rng.uniform(10, 20)), 1),
        'dealer_pct':        round(float(rng.uniform(10, 25)), 1),
        'issue_size':        round(float(rng.uniform(15, 25)), 1),
        'wi_otr_concession': round(float(rng.uniform(0.1, 2.0)), 3),
        'move_eod':          round(float(rng.uniform(80, 140)), 1),
        'vix_eod':           round(float(rng.uniform(12, 35)), 1),
        'csi_surprise':      round(float(rng.normal(0, 1.0)), 3),
        'ois_3m':            round(float(b0 - rng.uniform(0.5, 2.5)), 4),
        'policy_exp':        round(float(b0 - rng.uniform(0.3, 2.0)), 4),
        'y30_close':         round(float(y30_close), 4),
        'level_eod':         round(float(b0), 4),
        'slope_2s10s_eod':   round(float(_sv(10, b0, b1, b2, b3) - _sv(2, b0, b1, b2, b3)), 4),
        'slope_5s30s_eod':   round(float(_sv(30, b0, b1, b2, b3) - _sv(5, b0, b1, b2, b3)), 4),
        'curv_eod':          round(float(2*_sv(10, b0, b1, b2, b3)
                                         - _sv(2, b0, b1, b2, b3)
                                         - _sv(30, b0, b1, b2, b3)), 4),
    })


# ── Save ──────────────────────────────────────────────────────────────────────
intra_df  = pd.DataFrame(intra_rows)
bb_df     = pd.DataFrame(bb_rows)
cache_df  = pd.DataFrame(cache_rows)
cache_df['timestamp_et'] = pd.to_datetime(cache_df['timestamp_et'])
# Ensure auction_id is string in all outputs so merges are type-consistent
cache_df['auction_id'] = cache_df['auction_id'].astype(str)

intra_df.to_csv(DATA_DIR / 'intraday.csv', index=False)
bb_df.to_csv(DATA_DIR / 'bloomberg_results.csv', index=False)
cache_df.to_parquet(CACHE_DIR / 'intraday_curved.parquet', index=False)

print(f'✓  Intraday CSV  : {len(intra_df):>7,} rows  → data/intraday.csv')
print(f'✓  Bloomberg CSV : {len(bb_df):>7,} rows  → data/bloomberg_results.csv')
print(f'✓  Svensson cache: {len(cache_df):>7,} rows  → data/cache/intraday_curved.parquet')
print()
print(f'   Auctions  : {N_AUCTIONS}  '
      f'(calm={sum(regime_ids==0)}  normal={sum(regime_ids==1)}  stressed={sum(regime_ids==2)})')
print(f'   Date range: {auction_dates[0].date()}  →  {auction_dates[-1].date()}')
print(f'   Minutes/auction: {len(MINUTES)}  (±{EVENT_WINDOW} min around 13:00 ET)')
