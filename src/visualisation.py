"""
Visualisation layer — six desk exhibits.

Each function is self-contained: pass in the DataFrames it needs,
get back a matplotlib Figure. No plt.show() calls inside — the caller
decides whether to display interactively or save to disk.

Usage (notebook):
    from src.visualisation import (
        exhibit_event_study,
        exhibit_regime_overlay,
        exhibit_transition_matrix,
        exhibit_decay_curve,
        exhibit_feature_importance,
        exhibit_forecast_scorecard,
    )
    fig = exhibit_event_study(intraday_df, df_forecast)
    fig.savefig('ex1_event_study.png', bbox_inches='tight', dpi=150)

Usage (script):
    from src.visualisation import render_all_exhibits
    figs = render_all_exhibits(...)
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from config import N_REGIMES, H_DECAY, FEATURES, TARGET

# ── Palette & style ───────────────────────────────────────────────────────────
CALM_C    = '#2166AC'
NORMAL_C  = '#4DAC26'
STRESS_C  = '#D01C8B'
REGIME_COLORS  = [CALM_C, NORMAL_C, STRESS_C]
REGIME_LABELS  = ['Calm', 'Normal', 'Stressed']

_STYLE = {
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linewidth':    0.6,
    'font.family':       'sans-serif',
    'font.size':         11,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'legend.fontsize':   10,
    'legend.frameon':    False,
    'figure.dpi':        120,
}


def _apply_style():
    plt.rcParams.update(_STYLE)


# ══════════════════════════════════════════════════════════════════════════════
# Exhibit 1 — Event Study
# ══════════════════════════════════════════════════════════════════════════════

def exhibit_event_study(intraday_df: pd.DataFrame,
                        df_forecast: pd.DataFrame,
                        micro_col: str = 'micro_factor') -> plt.Figure:
    """
    Exhibit 1 — Event Study: The Dislocation is Real.

    Mean micro_factor by event-minute, split by tail-size tertile.

    Parameters
    ----------
    intraday_df : minute-bar data with columns [auction_id, event_minute, micro_factor]
    df_forecast : auction-level data with column [tail_bps]
    micro_col   : column to aggregate (default 'micro_factor')
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 5))

    # ── Pooled SE band ────────────────────────────────────────────────────────
    grp = intraday_df.groupby('event_minute')[micro_col]
    pooled_mean = grp.mean()
    pooled_se   = grp.std() / np.sqrt(grp.count())
    ax.fill_between(pooled_mean.index,
                    pooled_mean - pooled_se,
                    pooled_mean + pooled_se,
                    color='#BBBBBB', alpha=0.35, label='Pooled ±1 SE', zorder=1)
    ax.plot(pooled_mean.index, pooled_mean,
            color='#555555', linewidth=1.4, alpha=0.6, zorder=2)

    # ── Tail-size split ───────────────────────────────────────────────────────
    if ('tail_bps' in df_forecast.columns and
            df_forecast['tail_bps'].notna().sum() >= 9):
        tq = df_forecast['tail_bps'].quantile([1/3, 2/3]).values
        def _tag(x):
            if pd.isna(x): return 'Mid tail'
            return 'Small tail' if x <= tq[0] else ('Large tail' if x > tq[1] else 'Mid tail')
        tag_map = df_forecast.set_index('auction_id')['tail_bps'].map(_tag).to_dict()
        df_w = intraday_df.copy()
        df_w['_tag'] = df_w['auction_id'].map(tag_map).fillna('Mid tail')

        split_cfg = [
            ('Small tail', '#4393C3', 1.8, '-'),
            ('Mid tail',   '#878787', 1.2, '--'),
            ('Large tail', '#D01C8B', 2.2, '-'),
        ]
        for tag, col, lw, ls in split_cfg:
            sub = df_w[df_w['_tag'] == tag]
            if len(sub) < 10:
                continue
            m = sub.groupby('event_minute')[micro_col].mean()
            ax.plot(m.index, m.values, color=col, linewidth=lw,
                    linestyle=ls, label=tag, zorder=3)

    # ── Annotations ───────────────────────────────────────────────────────────
    ax.axvline(0, color='#CC0000', linestyle='--', linewidth=1.6,
               label='Auction close (13:00 ET)', zorder=4)
    ax.axhline(0, color='black', linewidth=0.6)

    ax.set_xlabel('Minutes relative to auction close (13:00 ET)')
    ax.set_ylabel(f'{micro_col} (%)')
    ax.set_title(
        'Exhibit 1 — Event Study: OTR Dislocation Around Auction Close',
        fontweight='bold', pad=10)
    ax.legend(loc='upper right')
    fig.tight_layout()

    # Phase labels — placed after tight_layout() so y-limits are finalised
    y_lo, y_hi = ax.get_ylim()
    label_y = y_lo + (y_hi - y_lo) * 0.06   # 6% up from bottom
    x_range  = ax.get_xlim()
    pre_x  = (x_range[0] + 0) / 2           # midpoint of pre-auction range
    post_x = (0 + x_range[1]) / 2           # midpoint of post-auction range
    for x_pos, lbl in [(pre_x, 'Pre-auction\nbuild-up'),
                        (post_x, 'Post-auction\nreversal')]:
        ax.text(x_pos, label_y, lbl, ha='center', fontsize=9,
                color='#888888', style='italic', zorder=5)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Exhibit 2 — Regime Overlay
# ══════════════════════════════════════════════════════════════════════════════

def exhibit_regime_overlay(intraday_df: pd.DataFrame,
                           hmm_model,
                           auction_id=None,
                           micro_col: str = 'micro_factor') -> plt.Figure:
    """
    Exhibit 2 — Regime Overlay: What 'Stressed' Looks Like on a Real Day.

    Top panel : micro_factor path with Viterbi-decoded regime shading.
    Bottom panel: forward-filtered P(regime | data_1:t) — causal posteriors.

    Parameters
    ----------
    intraday_df : minute-bar data
    hmm_model   : fitted GaussianHMM (from src.regime.fit_pooled_hmm)
    auction_id  : which auction to show (default: highest micro_factor range)
    """
    from src.regime import _forward_filtered_probs
    _apply_style()

    # Pick most dramatic auction if not specified
    if auction_id is None:
        ranges = (intraday_df.groupby('auction_id')[micro_col]
                             .agg(lambda s: s.max() - s.min()))
        auction_id = ranges.idxmax()

    day = (intraday_df[intraday_df['auction_id'] == auction_id]
           .sort_values('event_minute').reset_index(drop=True))

    mf     = day[micro_col].fillna(0).values.reshape(-1, 1)
    states = hmm_model.predict(mf)
    filt   = _forward_filtered_probs(hmm_model, mf)
    mins   = day['event_minute'].values
    K      = hmm_model.n_components

    fig, (ax_mf, ax_pr) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
        sharex=True)

    # ── Regime shading ────────────────────────────────────────────────────────
    for t in range(len(mins) - 1):
        ax_mf.axvspan(mins[t], mins[t + 1],
                      color=REGIME_COLORS[states[t] % K],
                      alpha=0.15, linewidth=0)

    ax_mf.plot(mins, day[micro_col], color='#111111', linewidth=1.6, zorder=3)
    ax_mf.axvline(0, color='#CC0000', linestyle='--', linewidth=1.5,
                  label='13:00 auction close', zorder=4)
    ax_mf.axhline(0, color='black', linewidth=0.5)
    ax_mf.set_ylabel(f'{micro_col} (%)')
    ax_mf.set_title(
        f'Exhibit 2 — Regime Overlay: Auction {auction_id}',
        fontweight='bold', pad=10)

    # Legend with regime colour patches
    patches = [mpatches.Patch(color=REGIME_COLORS[k], alpha=0.5,
                               label=REGIME_LABELS[k] if k < len(REGIME_LABELS) else f'R{k}')
               for k in range(K)]
    patches.append(plt.Line2D([0], [0], color='#CC0000', linestyle='--',
                               label='13:00 close'))
    ax_mf.legend(handles=patches, loc='upper left')

    # ── Filtered posteriors ───────────────────────────────────────────────────
    for k in range(K):
        lbl = REGIME_LABELS[k] if k < len(REGIME_LABELS) else f'Regime {k}'
        ax_pr.plot(mins, filt[:, k], color=REGIME_COLORS[k],
                   linewidth=1.4, label=lbl)
    ax_pr.set_ylim(-0.02, 1.05)
    ax_pr.set_ylabel('P(regime | data)', fontsize=10)
    ax_pr.set_xlabel('Event minute')
    ax_pr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_pr.legend(ncol=K, loc='lower right')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Exhibit 3 — Transition Matrix Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def exhibit_transition_matrix(hmm_model) -> plt.Figure:
    """
    Exhibit 3 — Regime Transition Matrix.

    3×3 heatmap with probability annotations.
    Diagonal dominance = regime persistence.
    """
    _apply_style()
    trans  = hmm_model.transmat_
    K      = trans.shape[0]
    labels = (REGIME_LABELS[:K] if K <= len(REGIME_LABELS)
              else [f'Regime {k}' for k in range(K)])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(trans, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    for i in range(K):
        for j in range(K):
            v = trans[i, j]
            ax.text(j, i, f'{v:.2f}',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if v > 0.55 else '#222222')

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f'→ {l}' for l in labels])
    ax.set_yticklabels(labels)
    ax.set_xlabel('To regime')
    ax.set_ylabel('From regime')
    ax.set_title('Exhibit 3 — Regime Transition Matrix\n(one-step probabilities)',
                 fontweight='bold', pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability')

    # Stickiness footnote
    stick_str = '   '.join(
        f'{labels[k]}: stays {trans[k, k]:.0%}' for k in range(K))
    fig.text(0.5, -0.02, stick_str, ha='center', fontsize=9, color='#555555')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Exhibit 4 — Decay Curve
# ══════════════════════════════════════════════════════════════════════════════

def exhibit_decay_curve(decay_panel: pd.DataFrame,
                        df_forecast: pd.DataFrame,
                        irf_full: pd.DataFrame = None) -> plt.Figure:
    """
    Exhibit 4 — Decay Curve: The Auction's Footprint.

    β_h from Jordà local projections with HAC CI bands.
    Fitted exponential + annotated half-life.
    Regime-conditional lines where data sufficient.

    Parameters
    ----------
    decay_panel : (auction × horizon) long panel from build_decay_panel()
    df_forecast : auction-level frame (needs regime_eod if available)
    irf_full    : pre-computed IRF DataFrame (optional; recomputed if None)
    """
    from src.decay import local_projection_decay, fit_half_life
    _apply_style()

    if irf_full is None:
        irf_full = local_projection_decay(decay_panel, shock_col='auction_shock')

    half_life = fit_half_life(irf_full)

    # Regime-conditional IRFs
    regime_irfs = {}
    if 'regime_eod' in df_forecast.columns:
        reg_map = (df_forecast.set_index('auction_id')['regime_eod']
                              .dropna().astype(int).to_dict())
        dp = decay_panel.copy()
        dp['_reg'] = dp['auction_id'].map(reg_map)
        for k in range(N_REGIMES):
            sub = dp[dp['_reg'] == k]
            if sub['auction_id'].nunique() >= 8:
                irf_k = local_projection_decay(sub, shock_col='auction_shock')
                if not irf_k.empty:
                    regime_irfs[k] = (irf_k, fit_half_life(irf_k))

    fig, ax = plt.subplots(figsize=(11, 5))

    # Full-sample CI band
    ax.fill_between(irf_full.index, irf_full['ci_lo'], irf_full['ci_hi'],
                    color='#888888', alpha=0.18, label='Full-sample 95% HAC CI')
    ax.plot(irf_full.index, irf_full['beta'], 'o-', color='#333333',
            linewidth=2.2, markersize=5.5, zorder=3, label='β_h (all regimes)')

    # Fitted exponential
    h_grid = np.linspace(0, max(irf_full.index), 200)
    if not np.isnan(half_life) and not irf_full.empty:
        tau = half_life / np.log(2)
        b0  = irf_full['beta'].iloc[0]
        ax.plot(h_grid, b0 * np.exp(-h_grid / tau),
                '--', color='#333333', linewidth=1.6, alpha=0.6)

        # half-life arrow annotation
        hl_y = b0 * 0.5
        ax.annotate(
            f' t½ = {half_life:.1f} bd',
            xy=(half_life, hl_y),
            xytext=(half_life + max(irf_full.index) * 0.08, hl_y * 1.25),
            fontsize=10, color='#333333', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.3),
            zorder=5)

    # Regime-conditional lines
    for k, (irf_k, hl_k) in regime_irfs.items():
        col = REGIME_COLORS[k % N_REGIMES]
        lbl = REGIME_LABELS[k % N_REGIMES] if k < len(REGIME_LABELS) else f'R{k}'
        ax.plot(irf_k.index, irf_k['beta'], 's-', color=col,
                linewidth=1.8, markersize=4, alpha=0.85, label=f'β_h — {lbl}')
        if not np.isnan(hl_k) and not irf_k.empty:
            tau_k = hl_k / np.log(2)
            b0_k  = irf_k['beta'].iloc[0]
            ax.plot(h_grid, b0_k * np.exp(-h_grid / tau_k),
                    '--', color=col, linewidth=1.2, alpha=0.55)
            ax.annotate(
                f' t½={hl_k:.1f} ({lbl.lower()})',
                xy=(hl_k, b0_k * 0.5),
                xytext=(hl_k + max(irf_k.index) * 0.1, b0_k * 0.6),
                fontsize=9, color=col,
                arrowprops=dict(arrowstyle='->', color=col, lw=1.0))

    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xlabel('Horizon h (business days after auction)')
    ax.set_ylabel('β_h  —  curve level shift per 1 bps tail')
    ax.set_title('Exhibit 4 — Decay Curve: How Long Does the Auction Footprint Last?',
                 fontweight='bold', pad=10)
    ax.legend(loc='upper right')

    # Half-life summary in corner
    hl_lines = [f'Full-sample half-life: {half_life:.1f} bd' if not np.isnan(half_life)
                else 'Half-life: insufficient data']
    for k, (_, hl_k) in regime_irfs.items():
        lbl = REGIME_LABELS[k] if k < len(REGIME_LABELS) else f'R{k}'
        hl_lines.append(f'{lbl}: {hl_k:.1f} bd' if not np.isnan(hl_k) else f'{lbl}: n/a')
    fig.text(0.015, 0.015, '\n'.join(hl_lines), fontsize=9, color='#555555',
             va='bottom', family='monospace')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Exhibit 5 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def exhibit_feature_importance(df_forecast: pd.DataFrame,
                               oos_preds: pd.DataFrame,
                               top_n: int = 15) -> plt.Figure:
    """
    Exhibit 5 — Feature Importance (Permutation, not impurity).

    Uses a walk-forward split mirroring CV: trains on first 75%, permutes on last 25%.
    Permutation importance avoids inflating correlated rate features.

    Parameters
    ----------
    df_forecast : auction-level frame with all features and TARGET column
    oos_preds   : OOS prediction frame (used for fold count annotation)
    top_n       : number of features to display
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance as sk_perm_imp
    from config import (RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
                        RF_N_ESTIMATORS, RANDOM_SEED)
    _apply_style()

    avail = [f for f in FEATURES if f in df_forecast.columns]
    mask  = df_forecast[TARGET].notna()
    X_all = df_forecast.loc[mask, avail].fillna(0)
    y_all = df_forecast.loc[mask, TARGET]
    n     = len(X_all)
    split = max(int(n * 0.75), n - 15)  # at least 15 in validation

    if split >= n - 3:
        # Fallback: use impurity importance on all data
        rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,
                                   max_depth=RF_MAX_DEPTH,
                                   min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                   random_state=RANDOM_SEED)
        rf.fit(X_all, y_all)
        imp  = pd.Series(rf.feature_importances_, index=avail).sort_values()
        mode = 'Impurity (fallback — small N)'
    else:
        rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,
                                   max_depth=RF_MAX_DEPTH,
                                   min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                   random_state=RANDOM_SEED)
        rf.fit(X_all.iloc[:split], y_all.iloc[:split])
        perm = sk_perm_imp(rf, X_all.iloc[split:], y_all.iloc[split:],
                           n_repeats=25, random_state=RANDOM_SEED, scoring='r2')
        imp  = pd.Series(perm.importances_mean, index=avail).sort_values()
        mode = 'Permutation'

    top = imp.tail(top_n)
    colors = ['#D01C8B' if v > 0 else '#AAAAAA' for v in top.values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color=colors, edgecolor='none', height=0.65)
    ax.axvline(0, color='black', linewidth=0.8)

    # Annotate tail_bps if in view
    if 'tail_bps' in top.index:
        v = top.loc['tail_bps']
        offset = max(top.abs().max() * 0.02, abs(v) * 0.05)
        ax.annotate('← auction result', xy=(v, list(top.index).index('tail_bps')),
                    xytext=(v + offset, list(top.index).index('tail_bps') + 0.35),
                    fontsize=9, color='#D01C8B')

    ax.set_xlabel(('Mean decrease in R² when feature permuted'
                   if mode == 'Permutation'
                   else 'Mean decrease in impurity'))
    ax.set_title(f'Exhibit 5 — Feature Importance ({mode}, top {len(top)})',
                 fontweight='bold', pad=10)
    n_folds = oos_preds['fold'].nunique() if 'fold' in oos_preds.columns else '?'
    fig.text(0.99, 0.01, f'Based on {n} auctions  |  {n_folds} CV folds',
             ha='right', fontsize=9, color='#888888')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Exhibit 6 — Forecast Scorecard
# ══════════════════════════════════════════════════════════════════════════════

def exhibit_forecast_scorecard(oos_preds: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 6 — Forecast Scorecard: Honest Edge vs. Naïve Benchmark.

    Left  : directional accuracy (hit rate) per fold — with 50% coin-flip line.
    Right : OOS R² per fold — with 0 line.
    Bottom: pooled summary table.

    Parameters
    ----------
    oos_preds : DataFrame with columns [fold, TARGET, pred_rf, pred_base]
    """
    from sklearn.metrics import r2_score as _r2
    _apply_style()

    rows = []
    for fold_n, g in oos_preds.groupby('fold'):
        act  = g[TARGET].values
        p_rf = g['pred_rf'].values
        p_bl = g['pred_base'].values
        m    = np.isfinite(act) & np.isfinite(p_rf) & np.isfinite(p_bl)
        if m.sum() < 3:
            continue
        rows.append({
            'fold':    int(fold_n),
            'n':       int(m.sum()),
            'dir_rf':  float(np.mean(np.sign(act[m]) == np.sign(p_rf[m]))),
            'dir_bl':  float(np.mean(np.sign(act[m]) == np.sign(p_bl[m]))),
            'r2_rf':   float(_r2(act[m], p_rf[m])),
            'r2_bl':   float(_r2(act[m], p_bl[m])),
        })

    fm = pd.DataFrame(rows)
    if fm.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No OOS data', ha='center', va='center')
        return fig

    x = np.arange(len(fm))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Directional accuracy ────────────────────────────────────────────
    b1 = axes[0].bar(x - w / 2, fm['dir_rf'], w, label='Random Forest',
                     color='#2166AC', alpha=0.85)
    b2 = axes[0].bar(x + w / 2, fm['dir_bl'], w, label='Regime baseline',
                     color='#878787', alpha=0.70)
    axes[0].axhline(0.5, color='#CC0000', linestyle='--', linewidth=1.4,
                    label='50% (coin flip)', zorder=5)

    # Value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f'{h:.0%}', ha='center', va='bottom', fontsize=8.5,
                     color='#333333')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f'Fold {int(r.fold)}\n(n={r.n})' for _, r in fm.iterrows()], fontsize=9)
    axes[0].set_ylim(0, 1.12)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[0].set_ylabel('Directional accuracy')
    axes[0].set_title('Directional Accuracy per Fold', fontweight='bold')
    axes[0].legend(loc='lower right')

    # ── Right: OOS R² ────────────────────────────────────────────────────────
    b3 = axes[1].bar(x - w / 2, fm['r2_rf'], w, label='Random Forest',
                     color='#2166AC', alpha=0.85)
    b4 = axes[1].bar(x + w / 2, fm['r2_bl'], w, label='Regime baseline',
                     color='#878787', alpha=0.70)
    axes[1].axhline(0, color='black', linewidth=0.9, zorder=5)

    for bar in list(b3) + list(b4):
        h = bar.get_height()
        offset = 0.005 if h >= 0 else -0.015
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + offset,
                     f'{h:.3f}', ha='center',
                     va='bottom' if h >= 0 else 'top',
                     fontsize=8, color='#333333')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f'Fold {int(r.fold)}\n(n={r.n})' for _, r in fm.iterrows()], fontsize=9)
    axes[1].set_ylabel('OOS R²')
    axes[1].set_title('OOS R² per Fold', fontweight='bold')
    axes[1].legend(loc='lower right')

    # ── Pooled summary row ────────────────────────────────────────────────────
    act_all = oos_preds[TARGET].values
    rf_all  = oos_preds['pred_rf'].values
    bl_all  = oos_preds['pred_base'].values
    m_all   = np.isfinite(act_all) & np.isfinite(rf_all) & np.isfinite(bl_all)

    dir_rf_p = np.mean(np.sign(act_all[m_all]) == np.sign(rf_all[m_all]))
    dir_bl_p = np.mean(np.sign(act_all[m_all]) == np.sign(bl_all[m_all]))
    r2_rf_p  = _r2(act_all[m_all], rf_all[m_all])
    r2_bl_p  = _r2(act_all[m_all], bl_all[m_all])

    summary = (f'Pooled ({m_all.sum()} obs) — '
               f'RF: dir={dir_rf_p:.1%}  R²={r2_rf_p:.4f}   '
               f'Baseline: dir={dir_bl_p:.1%}  R²={r2_bl_p:.4f}   '
               f'Edge: Δdir={dir_rf_p-dir_bl_p:+.1%}  ΔR²={r2_rf_p-r2_bl_p:+.4f}')
    fig.text(0.5, -0.02, summary, ha='center', fontsize=9.5,
             color='#333333', family='monospace')

    fig.suptitle('Exhibit 6 — Forecast Scorecard: RF vs. Regime-Conditional Baseline',
                 fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: render all exhibits at once
# ══════════════════════════════════════════════════════════════════════════════

def render_all_exhibits(intraday_df: pd.DataFrame,
                        df_forecast: pd.DataFrame,
                        decay_panel: pd.DataFrame,
                        oos_preds: pd.DataFrame,
                        hmm_model=None,
                        irf_full: pd.DataFrame = None,
                        save_dir: str = None) -> dict:
    """
    Render all six exhibits and optionally save to disk.

    Parameters
    ----------
    intraday_df  : minute-bar intraday data (with micro_factor, event_minute)
    df_forecast  : auction-level features + targets
    decay_panel  : (auction × horizon) panel for local projections
    oos_preds    : OOS predictions from walk_forward_leakage_safe
    hmm_model    : fitted GaussianHMM (or None to skip Ex 2 & 3)
    irf_full     : pre-computed IRF DataFrame (or None to compute inside Ex 4)
    save_dir     : path to save PNGs (or None to skip saving)

    Returns
    -------
    dict mapping exhibit name → Figure
    """
    figs = {}

    print('Rendering Exhibit 1 — Event Study…')
    figs['ex1_event_study'] = exhibit_event_study(intraday_df, df_forecast)

    if hmm_model is not None:
        print('Rendering Exhibit 2 — Regime Overlay…')
        figs['ex2_regime_overlay'] = exhibit_regime_overlay(intraday_df, hmm_model)

        print('Rendering Exhibit 3 — Transition Matrix…')
        figs['ex3_transition_matrix'] = exhibit_transition_matrix(hmm_model)
    else:
        print('  Exhibits 2 & 3 skipped — no HMM model provided.')

    print('Rendering Exhibit 4 — Decay Curve…')
    figs['ex4_decay_curve'] = exhibit_decay_curve(decay_panel, df_forecast, irf_full)

    print('Rendering Exhibit 5 — Feature Importance…')
    figs['ex5_feature_importance'] = exhibit_feature_importance(df_forecast, oos_preds)

    print('Rendering Exhibit 6 — Forecast Scorecard…')
    figs['ex6_forecast_scorecard'] = exhibit_forecast_scorecard(oos_preds)

    if save_dir is not None:
        from pathlib import Path
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            path = out / f'{name}.png'
            fig.savefig(path, bbox_inches='tight', dpi=150)
            print(f'  Saved → {path}')

    return figs
