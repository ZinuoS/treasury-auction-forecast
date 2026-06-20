"""
Stage 3 — Pooled HMM regime features.
One global GaussianHMM fit on all training auction micro_factor sequences
(per-day lengths passed so chain resets at each auction boundary).
Ref: Hamilton (1989); alamgirhossain-tech/Markov-Switch-Regime-Analysis.

⚠ Fit train-only inside every CV fold.
⚠ Use FILTERED (forward-algorithm) probabilities for feature columns;
   model.predict_proba() returns SMOOTHED posteriors (uses future data) — do not use
   it for anything predictive.
"""
import numpy as np
import pandas as pd

from config import N_REGIMES, RANDOM_SEED

try:
    from hmmlearn.hmm import GaussianHMM
    HAVE_HMM = True
except ImportError:
    HAVE_HMM = False
    print('hmmlearn not available — regime stage skipped (request via Artifactory)')


def fit_pooled_hmm(intraday_train, n_regimes=N_REGIMES, seed=None):
    """Fit one global HMM on all training-auction micro_factor sequences.

    Per-day sequence lengths are passed so the HMM treats each auction day
    as an independent chain (correct pooling — no bleed between auctions).

    ⚠ Call only on training-set rows.
    """
    if not HAVE_HMM:
        raise RuntimeError('hmmlearn required — install via Artifactory.')

    seed  = seed or RANDOM_SEED
    seqs, lengths = [], []
    for _, g in (intraday_train
                 .sort_values(['auction_id', 'event_minute'])
                 .groupby('auction_id', sort=False)):
        x = g['micro_factor'].fillna(0.0).values
        if len(x) >= 10:
            seqs.append(x)
            lengths.append(len(x))

    if not seqs:
        raise ValueError('No valid auction sequences found for HMM fitting.')

    X     = np.concatenate(seqs).reshape(-1, 1)
    model = GaussianHMM(n_components=n_regimes, covariance_type='diag',
                        n_iter=300, tol=1e-5, random_state=seed)
    model.fit(X, lengths)
    return model


def _forward_filtered_probs(model, X):
    """Forward algorithm: P(s_t | y_{1:t}) — FILTERED, not smoothed.

    Returns array shape (T, K).
    model.predict_proba() returns SMOOTHED P(s_t | y_{1:T}) which conditions
    on future observations — never use it for predictive features.
    """
    T, K  = len(X), model.n_components
    log_e = model._compute_log_likelihood(X)   # (T, K)

    log_a       = np.full((T, K), -np.inf)
    log_a[0]    = np.log(model.startprob_ + 1e-300) + log_e[0]
    log_trans   = np.log(model.transmat_    + 1e-300)   # (K, K)

    for t in range(1, T):
        for k in range(K):
            log_a[t, k] = (np.logaddexp.reduce(log_a[t - 1] + log_trans[:, k])
                           + log_e[t, k])

    log_norms = np.logaddexp.reduce(log_a, axis=1, keepdims=True)
    return np.exp(log_a - log_norms)   # (T, K)


def regime_features_per_auction(intraday, model, n_regimes=N_REGIMES):
    """Apply pre-fit HMM to derive per-auction regime feature row.

    Discrete states: Viterbi (most likely path).
    Probability features: filtered posteriors from forward algorithm.
    Returns one-row-per-auction DataFrame.
    """
    rows = []
    for aid, g in (intraday
                   .sort_values(['auction_id', 'event_minute'])
                   .groupby('auction_id', sort=False)):
        mf = g['micro_factor'].fillna(0.0).values
        X  = mf.reshape(-1, 1)

        states     = model.predict(X)             # Viterbi
        filt_probs = _forward_filtered_probs(model, X)   # (T, K) filtered

        phase   = g['phase'].values
        is_pre  = phase == 'pre'
        is_post = phase == 'post'

        pre_state   = int(states[is_pre][-1]) if is_pre.any()  else np.nan
        eod_state   = int(states[-1])
        post_states = states[is_post]
        eod_filt    = filt_probs[-1]             # (K,)

        row = {
            'auction_id':             aid,
            'regime_eod':             eod_state,
            'regime_transition':      (f'{int(pre_state)}->{eod_state}'
                                       if not np.isnan(pre_state)
                                       else f'?->{eod_state}'),
            'minutes_in_post_regime': int((post_states == eod_state).sum())
                                      if len(post_states) > 0 else 0,
        }
        for k in range(n_regimes):
            row[f'prob_regime_{k}_eod'] = float(eod_filt[k])

        rows.append(row)
    return pd.DataFrame(rows)
