# Treasury Auction Curve Forecast

A leakage-safe machine-learning pipeline that quantifies the intraday dislocation at US Treasury auctions and forecasts its influence on the yield curve over the following 1–10 business days.
---

## Pipeline stages

| Stage | Notebook | Module | What it does |
|-------|----------|--------|--------------|
| 0 | `00_assembly` | `src/assembly.py` | Ingest desk intraday tape + Bloomberg CSVs; validate schemas; build event clock (±4h around 13:00 ET) |
| 1 | `01_curve_micro` | `src/curve.py` | Svensson (1994) curve fit at every minute × auction; compute `micro_factor` = OTR yield − model fair value |
| 2 | `02_macro_baseline` | `src/macro.py` | Ridge regression of daily 30Y on macro factors; residual used as a supply/term-premium feature |
| 3 | `03_regime_hmm` | `src/regime.py` | Pooled GaussianHMM on `micro_factor` sequences; filtered (causal) posteriors as regime features |
| 4–5 | `04_05_features_targets` | `src/features.py` | Collapse intraday → one row/auction; add h-day-forward curve-change targets; build long decay panel |
| 6 | `06_model_cv` | `src/model.py` | Shallow RandomForest with expanding walk-forward CV; **all preprocessing models refit per fold** |
| 7a–7b | `07_decay_validation` | `src/decay.py`, `src/validation.py` | Jordà local projections → IRF → half-life; PurgedKFold OOS; event study; six desk exhibits |

---

## Key design choices

### Leakage prevention (non-negotiable)
Every preprocessing model — intraday PCA, macro Ridge, HMM — is refit **inside each CV fold on training rows only**. No global fits leak future information into test features.

### Causal regime features only
`regime.py` uses the **forward-filtered** (forward-algorithm) posteriors `P(s_t | y_{1:t})`.  
`model.predict_proba()` returns **smoothed** posteriors that condition on future observations — it is never used for predictive features.

### Purged + embargoed CV for the decay layer
Because local-projection targets overlap across horizons, standard k-fold leaks. `validation.PurgedKFold` removes training observations whose label window overlaps the test window (purge) and adds a 2-business-day embargo after each test window.

---

## Repository structure

```
treasury-auction-forecast/
├── config.py                  # single source of truth: paths, seeds, hyperparameters
├── data/
│   ├── intraday.csv           # desk intraday tape (not committed)
│   ├── bloomberg_results.csv  # Bloomberg auction results (not committed)
│   └── cache/                 # parquet caches written by each stage
├── src/
│   ├── assembly.py            # Stage 0
│   ├── curve.py               # Stage 1
│   ├── macro.py               # Stage 2
│   ├── regime.py              # Stage 3
│   ├── features.py            # Stages 4–5
│   ├── model.py               # Stage 6
│   ├── decay.py               # Stage 7a
│   └── validation.py          # Stage 7b
└── notebooks/
    ├── 00_assembly.ipynb
    ├── 01_curve_micro.ipynb
    ├── 02_macro_baseline.ipynb
    ├── 03_regime_hmm.ipynb
    ├── 04_05_features_targets.ipynb
    ├── 06_model_cv.ipynb
    └── 07_decay_validation.ipynb   ← desk exhibits (six charts)
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/ZinuoS/treasury-auction-forecast.git
cd treasury-auction-forecast

# 2. Install dependencies
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
pip install hmmlearn          # optional — regime stage skipped if unavailable
pip install polars            # optional — faster CSV loading in Stage 0

# 3. Drop your data files into data/
#    data/intraday.csv          — minute-bar intraday tape
#    data/bloomberg_results.csv — Bloomberg auction results

# 4. Fill in column mappings in config.py
#    INTRADAY_COL_MAP and BLOOMBERG_COL_MAP

# 5. Run notebooks in order: 00 → 01 → … → 07
```

---

## Configuration

All constants live in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVENT_WINDOW_MIN` | 240 | ±minutes around 13:00 ET |
| `MATURITIES` | [2,3,5,7,10,20,30] | Yield curve tenors |
| `N_REGIMES` | 3 | HMM states (calm / normal / stressed) |
| `H_FORECAST` | 1 | Forecast horizon (business days) |
| `H_DECAY` | 10 | Max horizon for local projections |
| `CV_N_SPLITS` | 5 | Walk-forward folds |
| `CV_MIN_TRAIN` | 20 | Minimum auctions in first training fold |
| `EMBARGO_TD` | 2BD | PurgedKFold embargo after test window |

---

## Deck exhibits (notebook 07)

Six self-contained charts ordered so the thesis builds without a formula in the body:

1. **Event study** — average `micro_factor` by minute, split by tail-size tertile. Model-independent.
2. **Regime overlay** — one auction day's micro-factor path with HMM regime shading.
3. **Transition heatmap** — 3×3 Markov stickiness table.
4. **Decay curve** — IRF β_h with HAC bands, fitted exponential, half-life annotation, calm vs. stressed.
5. **Feature importance** — permutation importance (not impurity); `tail_bps` annotated.
6. **Forecast scorecard** — directional accuracy and OOS R² per fold vs. regime-conditional baseline.

---

## References

- Gürkaynak, Sack & Wright (2007) — Svensson curve parametrisation
- Hu, Pan & Wang (2013) — OTR residual as illiquidity proxy
- Hamilton (1989) — Hidden Markov regime switching
- Jordà (2005) — Local projection impulse response functions
- Ang & Piazzesi (2003) — Macro factors in yield curve models
- López de Prado (2018) — *Advances in Financial ML*, ch. 7 (PurgedKFold)
- Litterman & Scheinkman (1991) — Level/slope/curvature decomposition

---

## Data privacy

Raw data files (`data/*.csv`) and parquet caches (`data/cache/`) are excluded from version control via `.gitignore`. Only source code and notebooks are committed.
