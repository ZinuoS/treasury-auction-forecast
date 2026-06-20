"""
Stage 2 — Daily macro baseline.
Ridge regression: daily 30y level ~ Bloomberg macro factors.
Used as a context feature (macro_resid_daily), NOT as a differencing target.
Refs: Ang-Piazzesi (2003); Diebold-Rudebusch-Aruoba (2006); Adrian-Crump-Moench (2013).

⚠ fit train-only inside every CV fold — never on the full dataset.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import MACRO_COLS


def macro_baseline(daily_df, macro_cols, train_mask,
                   target='y30_close', alpha=1.0):
    """Fit Ridge on training rows; apply to all rows.

    Parameters
    ----------
    daily_df   : DataFrame indexed by auction_date
    macro_cols : list of Bloomberg column names (from config.MACRO_COLS)
    train_mask : boolean array aligned to daily_df index (True = train)
    target     : column to regress on (default 'y30_close')
    alpha      : Ridge regularisation (increase if macro_cols are collinear)

    Returns daily_df copy with 'macro_fair_30y' and 'macro_resid_daily' added.
    ⚠ train_mask must come ONLY from the current CV fold's training set.
    """
    df    = daily_df.copy()
    avail = [c for c in macro_cols
             if c in df.columns and df[c].notna().sum() > 5]

    if not avail:
        print(f'  ⚠ No macro cols found ({macro_cols}). '
              'Falling back to unconditional train mean.')
        df['macro_fair_30y']    = df.loc[train_mask, target].mean()
        df['macro_resid_daily'] = df[target] - df['macro_fair_30y']
        return df

    X = (df[avail]
         .ffill()
         .fillna(df[avail].median())
         .values.astype(float))
    y = df[target].ffill().values.astype(float)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X[train_mask])   # fit on train only
    X_all  = scaler.transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_tr, y[train_mask])

    df['macro_fair_30y']    = model.predict(X_all)
    df['macro_resid_daily'] = df[target] - df['macro_fair_30y']
    return df
