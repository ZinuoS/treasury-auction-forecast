"""
Central configuration — single source of truth for paths, constants, and seeds.
Import from every src/ module and notebook:
    from config import DATA_DIR, MATURITIES, TARGET, ...
"""
from pathlib import Path
import pandas as pd

# ── project layout ────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent
DATA_DIR  = ROOT_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── intraday / event window ───────────────────────────────────────────────────
AUCTION_TIME_ET  = '13:00'
EVENT_WINDOW_MIN = 240          # ± minutes around 13:00 ET
MATURITIES       = [2, 3, 5, 7, 10, 20, 30]

# ── model ─────────────────────────────────────────────────────────────────────
N_REGIMES           = 3
N_PCA               = 3
H_FORECAST          = 1
H_DECAY             = 10
RF_MAX_DEPTH        = 4
RF_MIN_SAMPLES_LEAF = 5
RF_N_ESTIMATORS     = 300
RANDOM_SEED         = 42

# ── CV ────────────────────────────────────────────────────────────────────────
CV_N_SPLITS  = 5
CV_MIN_TRAIN = 20               # minimum auctions in first training fold
EMBARGO_TD   = pd.Timedelta('2BD')

# ── feature / target names ────────────────────────────────────────────────────
TARGET = f'd_level_h{H_FORECAST}'

FEATURES = [
    # auction quality
    'tail_bps', 'bid_to_cover', 'indirect_pct', 'direct_pct', 'dealer_pct',
    'issue_size', 'wi_otr_concession',
    # intraday micro factor
    'micro_eod', 'micro_postauction_drift', 'micro_max_dev', 'micro_vol_post',
    # regime (filtered probs — causal)
    'minutes_in_post_regime',
    'prob_regime_0_eod', 'prob_regime_1_eod', 'prob_regime_2_eod',
    # macro context
    'move_eod', 'vix_eod', 'csi_surprise', 'ois_3m', 'policy_exp',
    'macro_resid_daily',
    # curve shape at EOD
    'level_eod', 'slope_2s10s_eod', 'slope_5s30s_eod', 'curv_eod',
]

MACRO_COLS = ['ois_3m', 'policy_exp', 'vix_eod', 'csi_surprise']

# ── Bloomberg CSV import settings ─────────────────────────────────────────────
BLOOMBERG_NA = [
    '#N/A N/A', '#N/A', 'N/A', 'N.A.', 'N.M.', '#VALUE!',
    '#REF!', '--', '', ' ', 'nan', 'NaN',
]

# ── ↓ Update these to match your actual exported file names ──────────────────
INTRADAY_CSV_PATH  = DATA_DIR / 'intraday.csv'
BLOOMBERG_CSV_PATH = DATA_DIR / 'bloomberg_results.csv'

# ── ↓ Fill in after running upload_and_inspect() in notebook 1 ───────────────
INTRADAY_SKIP_ROWS  = 0     # metadata rows to skip at top of Bloomberg export
BLOOMBERG_SKIP_ROWS = 0

INTRADAY_COL_MAP = {
    # 'Your CSV column' : 'schema column'
    # 'Date'             : 'timestamp_et',
    # 'Auction_ID'       : 'auction_id',
    # 'OTR_Yield_30Y'    : 'otr_30y_yield',
    # 'WI_Yield_30Y'     : 'wi_30y_yield',
    # 'Yield_2Y'         : 'y_2y',
    # 'Yield_3Y'         : 'y_3y',
    # 'Yield_5Y'         : 'y_5y',
    # 'Yield_7Y'         : 'y_7y',
    # 'Yield_10Y'        : 'y_10y',
    # 'Yield_20Y'        : 'y_20y',
    # 'Yield_30Y'        : 'y_30y',
    # 'Bid'              : 'bid',
    # 'Ask'              : 'ask',
    # 'Volume'           : 'volume',
}

BLOOMBERG_COL_MAP = {
    # 'Auction Date'     : 'auction_date',
    # 'CUSIP'            : 'cusip',
    # 'Issue Type'       : 'issue_type',
    # 'Tail (bps)'       : 'tail_bps',
    # 'Bid-to-Cover'     : 'bid_to_cover',
    # 'Indirect (%)'     : 'indirect_pct',
    # 'Direct (%)'       : 'direct_pct',
    # 'Dealer (%)'       : 'dealer_pct',
    # 'Issue Size ($B)'  : 'issue_size',
    # 'WI-OTR Concession': 'wi_otr_concession',
    # 'MOVE Index'       : 'move_eod',
    # 'VIX'              : 'vix_eod',
    # 'CSI Surprise'     : 'csi_surprise',
    # 'OIS 3M'           : 'ois_3m',
    # 'Policy Exp'       : 'policy_exp',
    # 'Yield 30Y Close'  : 'y30_close',
    # 'Level EOD'        : 'level_eod',
    # 'Slope 2s10s'      : 'slope_2s10s_eod',
    # 'Slope 5s30s'      : 'slope_5s30s_eod',
    # 'Curvature'        : 'curv_eod',
}
