"""
Stage 0 — Assembly & event clock.
Ingest desk intraday tape + Bloomberg CSV exports, clean, validate, build event clock.
Ref: Fleming, Hrung, Mizrach & Romero (intraday auction effects).
"""
import glob
import os

import numpy as np
import pandas as pd

from config import (
    AUCTION_TIME_ET, EVENT_WINDOW_MIN, MATURITIES, MACRO_COLS,
    BLOOMBERG_NA, CACHE_DIR, DATA_DIR,
    INTRADAY_SKIP_ROWS, BLOOMBERG_SKIP_ROWS,
    INTRADAY_COL_MAP, BLOOMBERG_COL_MAP,
    INTRADAY_CSV_PATH, BLOOMBERG_CSV_PATH,
)

try:
    import polars as pl
    HAVE_POLARS = True
except ImportError:
    HAVE_POLARS = False


# ── Inspection ────────────────────────────────────────────────────────────────

def upload_and_inspect(path, skip_rows=0, n_rows=5):
    """Print column names, dtypes, NA%, and sample values for a CSV.
    Increase skip_rows if Bloomberg metadata rows appear above the header.
    """
    path = str(path)
    if not os.path.exists(path):
        print(f'File not found: {path}')
        print(f'Drop your CSV in: {os.path.abspath(os.path.dirname(path))}')
        return None

    df = pd.read_csv(path, skiprows=skip_rows, na_values=BLOOMBERG_NA,
                     keep_default_na=True, nrows=200)

    print(f'\n{"─"*62}')
    print(f'File  : {os.path.basename(path)}')
    print(f'Shape : {len(df)} rows × {len(df.columns)} cols  (first 200 rows)')
    print(f'{"─"*62}')
    for i, col in enumerate(df.columns):
        sample  = repr(df[col].dropna().iloc[0]) if df[col].notna().any() else 'ALL NaN'
        pct_na  = 100 * df[col].isna().mean()
        print(f'  [{i:2d}] {col!r:45s}  {str(df[col].dtype):10s}  '
              f'na={pct_na:4.0f}%  e.g. {sample}')
    print(f'\nFirst {n_rows} rows:')
    print(df.head(n_rows).to_string(max_cols=10))
    return df


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _coerce_numerics(df, exclude=('timestamp_et', 'auction_date', 'auction_id',
                                   'cusip', 'issue_type', 'regime_transition')):
    """Coerce object columns to numeric where ≥50% of values convert cleanly."""
    for col in df.columns:
        if col in exclude or df[col].dtype != object:
            continue
        conv = pd.to_numeric(df[col], errors='coerce')
        if conv.notna().sum() >= 0.5 * df[col].notna().sum():
            df[col] = conv
    return df


def clean_intraday_csv(path=None, col_map=None, skip_rows=None):
    """Load + clean the desk intraday tape from CSV.

    Handles Bloomberg NA tokens, column renaming, timestamp parsing,
    yield column coercion, and auction_id derivation.
    """
    path      = str(path or INTRADAY_CSV_PATH)
    col_map   = col_map   if col_map   is not None else INTRADAY_COL_MAP
    skip_rows = skip_rows if skip_rows is not None else INTRADAY_SKIP_ROWS

    df = pd.read_csv(path, skiprows=skip_rows,
                     na_values=BLOOMBERG_NA, keep_default_na=True)
    if col_map:
        df = df.rename(columns=col_map)

    # Timestamp
    if 'timestamp_et' in df.columns:
        df['timestamp_et'] = pd.to_datetime(df['timestamp_et'],
                                             infer_datetime_format=True, errors='coerce')
    else:
        cands = [c for c in df.columns
                 if any(t in c.lower() for t in ('date', 'time', 'timestamp'))]
        print(f'  ⚠ No timestamp_et — candidates: {cands}. Add to INTRADAY_COL_MAP.')

    # Yield + numeric coercion
    ycols = [f'y_{m}y' for m in MATURITIES]
    for c in ycols + ['otr_30y_yield', 'wi_30y_yield', 'bid', 'ask', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = _coerce_numerics(df)

    # Drop unusable rows
    before = len(df)
    for drop_col in ['timestamp_et', 'otr_30y_yield']:
        if drop_col in df.columns:
            df = df.dropna(subset=[drop_col])
    if (dropped := before - len(df)):
        print(f'  Dropped {dropped:,} rows (null timestamp or OTR yield)')

    # auction_id fallback
    if 'auction_id' not in df.columns and 'timestamp_et' in df.columns:
        df['auction_id'] = df['timestamp_et'].dt.strftime('%Y%m%d')
        print('  ⚠ auction_id derived from date. Add to INTRADAY_COL_MAP if available.')

    # Ensure auction_id is always string so merges with parquet caches are type-consistent
    if 'auction_id' in df.columns:
        df['auction_id'] = df['auction_id'].astype(str)

    sort_cols = [c for c in ['auction_id', 'timestamp_et'] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def clean_bloomberg_csv(path=None, col_map=None, skip_rows=None):
    """Load + clean the Bloomberg auction results CSV. One row per auction."""
    path      = str(path or BLOOMBERG_CSV_PATH)
    col_map   = col_map   if col_map   is not None else BLOOMBERG_COL_MAP
    skip_rows = skip_rows if skip_rows is not None else BLOOMBERG_SKIP_ROWS

    df = pd.read_csv(path, skiprows=skip_rows,
                     na_values=BLOOMBERG_NA, keep_default_na=True)
    if col_map:
        df = df.rename(columns=col_map)

    # Date
    if 'auction_date' in df.columns:
        df['auction_date'] = pd.to_datetime(df['auction_date'],
                                             infer_datetime_format=True, errors='coerce')
        df = df.dropna(subset=['auction_date'])
    else:
        cands = [c for c in df.columns if 'date' in c.lower()]
        print(f'  ⚠ No auction_date — candidates: {cands}. Add to BLOOMBERG_COL_MAP.')

    df = _coerce_numerics(df)

    # auction_id fallback
    if 'auction_id' not in df.columns:
        if 'auction_date' in df.columns and 'cusip' in df.columns:
            df['auction_id'] = (df['auction_date'].dt.strftime('%Y%m%d')
                                + '_' + df['cusip'].astype(str))
        elif 'auction_date' in df.columns:
            df['auction_id'] = df['auction_date'].dt.strftime('%Y%m%d')
        print('  ⚠ auction_id derived from date+cusip. Add to BLOOMBERG_COL_MAP if available.')

    # Percentage scale guard (0–1 → 0–100)
    for c in ['indirect_pct', 'direct_pct', 'dealer_pct']:
        if c in df.columns and df[c].dropna().max() <= 1.5:
            df[c] = df[c] * 100
            print(f'  {c}: converted 0–1 → 0–100 scale')

    # Ensure auction_id is always string for type-consistent merges
    if 'auction_id' in df.columns:
        df['auction_id'] = df['auction_id'].astype(str)

    return (df.sort_values('auction_date').reset_index(drop=True)
            if 'auction_date' in df.columns else df)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_uploads(intraday_df, bloomberg_df):
    """Schema + sanity check before running the expensive Stage 1.
    Returns True if both tables are ready to proceed.
    """
    ok = True

    required_intra = (['auction_id', 'timestamp_et', 'otr_30y_yield']
                      + [f'y_{m}y' for m in MATURITIES])
    missing_intra  = [c for c in required_intra if c not in intraday_df.columns]

    print('\n── Intraday tape ──')
    n_auctions = intraday_df['auction_id'].nunique() if 'auction_id' in intraday_df.columns else '?'
    print(f'  Rows: {len(intraday_df):,}  |  Auctions: {n_auctions}')
    if missing_intra:
        print(f'  ✗ MISSING: {missing_intra}  → add to INTRADAY_COL_MAP')
        ok = False
    else:
        print(f'  ✓ Schema OK ({len(required_intra)} required cols present)')

    if 'otr_30y_yield' in intraday_df.columns:
        lo, hi = intraday_df['otr_30y_yield'].min(), intraday_df['otr_30y_yield'].max()
        sane   = 0.0 < lo and hi < 25.0
        print(f'  otr_30y_yield ∈ [{lo:.3f}, {hi:.3f}]  —  {"OK" if sane else "⚠ SUSPICIOUS"}')
        if not sane:
            ok = False

    required_bb = ['auction_id', 'auction_date', 'tail_bps', 'bid_to_cover',
                   'indirect_pct', 'y30_close']
    missing_bb  = [c for c in required_bb if c not in bloomberg_df.columns]

    print('\n── Bloomberg table ──')
    print(f'  Rows (auctions): {len(bloomberg_df):,}')
    if missing_bb:
        print(f'  ✗ MISSING: {missing_bb}  → add to BLOOMBERG_COL_MAP')
        ok = False
    else:
        print(f'  ✓ Schema OK ({len(required_bb)} required cols present)')

    if 'auction_id' in intraday_df.columns and 'auction_id' in bloomberg_df.columns:
        i_ids = set(intraday_df['auction_id'].unique())
        b_ids = set(bloomberg_df['auction_id'].unique())
        print(f'\n── Join check ──')
        print(f'  Matched: {len(i_ids & b_ids):,}  |  '
              f'only-intraday: {len(i_ids - b_ids)}  |  '
              f'only-bloomberg: {len(b_ids - i_ids)}')
        if not (i_ids & b_ids):
            print('  ⚠ Zero matches — check auction_id encoding in both files')
            ok = False

    print(f'\n{"─"*40}')
    print(f'Status: {"READY ✓" if ok else "NOT READY ✗ — fix issues above"}')
    print(f'{"─"*40}')
    return ok


# ── Load ─────────────────────────────────────────────────────────────────────

def load_intraday(path=None, use_cache=True):
    """Load + clean intraday CSV; cache to parquet after first run."""
    from config import CACHE_DIR
    cache = CACHE_DIR / 'intraday_raw.parquet'

    if use_cache and cache.exists():
        print(f'Loading intraday from cache ({cache.name})…')
        return pd.read_parquet(cache)

    path = path or INTRADAY_CSV_PATH
    # Auto-detect if path not found
    if not os.path.exists(str(path)):
        csvs = sorted(glob.glob(str(DATA_DIR / '*.csv')))
        hits = [f for f in csvs if any(k in os.path.basename(f).lower()
                                        for k in ('intraday', 'tape', 'tick', 'minute'))]
        if hits:
            path = hits[0]
            print(f'Auto-detected: {os.path.basename(path)}')
        elif csvs:
            raise FileNotFoundError(
                f'Could not auto-detect intraday file.\n'
                f'CSVs in data/: {[os.path.basename(f) for f in csvs]}\n'
                f'Pass path= explicitly or rename your file to include "intraday".')
        else:
            raise FileNotFoundError(f'No CSV files in {DATA_DIR}/')

    print(f'Loading intraday: {os.path.basename(str(path))}…')
    df = clean_intraday_csv(path)
    print(f'  {len(df):,} rows  |  {df["auction_id"].nunique()} auctions')

    df.to_parquet(cache, index=False)
    print(f'  Cached → {cache}')
    return df


def load_bloomberg(path=None):
    """Load + clean Bloomberg auction results CSV."""
    path = path or BLOOMBERG_CSV_PATH
    if not os.path.exists(str(path)):
        csvs = sorted(glob.glob(str(DATA_DIR / '*.csv')))
        hits = [f for f in csvs if any(k in os.path.basename(f).lower()
                                        for k in ('bloomberg', 'auction', 'result', 'bbg'))]
        if hits:
            path = hits[0]
            print(f'Auto-detected: {os.path.basename(path)}')
        elif csvs:
            raise FileNotFoundError(
                f'Could not auto-detect Bloomberg file.\n'
                f'CSVs in data/: {[os.path.basename(f) for f in csvs]}\n'
                f'Pass path= explicitly or rename to include "bloomberg" or "auction".')
        else:
            raise FileNotFoundError(f'No CSV files in {DATA_DIR}/')

    print(f'Loading Bloomberg: {os.path.basename(str(path))}…')
    df = clean_bloomberg_csv(path)
    print(f'  {len(df):,} auctions')
    return df


# ── Event clock ───────────────────────────────────────────────────────────────

def add_event_clock(df):
    """Attach event_minute (minutes since 13:00 ET) and phase (pre/at/post).
    Filters to ±EVENT_WINDOW_MIN around the auction close.
    """
    df     = df.copy()
    ts     = pd.to_datetime(df['timestamp_et'])
    anchor = ts.dt.normalize() + pd.Timedelta('13:00:00')
    df['event_minute'] = (ts - anchor).dt.total_seconds() / 60.0
    df['phase'] = np.select(
        [df['event_minute'] < 0, df['event_minute'] == 0],
        ['pre', 'at'], default='post')
    before = len(df)
    df = df[df['event_minute'].abs() <= EVENT_WINDOW_MIN].copy()
    print(f'  Event clock: {len(df):,}/{before:,} rows in ±{EVENT_WINDOW_MIN} min window')
    return df


def build_daily_curve(bloomberg_df, intraday_df, macro_cols=None):
    """Assemble daily yield curve table indexed by auction_date.
    Forward-fills weekly/monthly macro releases to auction frequency.
    """
    macro_cols = macro_cols or MACRO_COLS
    ycols      = [f'y_{m}y' for m in MATURITIES]
    avail_y    = [c for c in ycols if c in intraday_df.columns]

    eod_yields = (intraday_df
                  .sort_values('timestamp_et')
                  .groupby('auction_id')[avail_y]
                  .last()
                  .reset_index())
    daily = bloomberg_df.merge(eod_yields, on='auction_id', how='left')
    daily = daily.set_index('auction_date').sort_index()

    avail_macro = [c for c in macro_cols if c in daily.columns]
    if avail_macro:
        daily[avail_macro] = daily[avail_macro].ffill()
    return daily
