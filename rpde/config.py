# =============================================================
# RPDE Configuration
# Single source of truth for all RPDE tunable parameters.
# =============================================================

# ── PATTERN DISCOVERY — Big Move Scanner ──────────────────

# Minimum pip move to consider a "big move" (per pair overrides below)
DEFAULT_MIN_MOVE_PIPS = 20.0

# Maximum bars to look forward for a big move to materialize
# At M5 timeframe: 24 bars = 2 hours, 48 bars = 4 hours
FORWARD_LOOK_BARS = 24

# Timeframe used for scanning historical bars
SCAN_TIMEFRAME = "M5"

# Minimum bars between golden moments (avoid clustering at same candle)
MIN_BAR_SEPARATION = 5

# Per-pair pip thresholds (pairs have different volatility personalities)
PAIR_MOVE_THRESHOLDS = {
    # JPY Crosses (usually 20-30 pip moves are significant)
    "EURJPY": 25.0,
    "GBPJPY": 30.0,
    "CHFJPY": 20.0,
    "CADJPY": 20.0,
    "AUDJPY": 25.0,
    # Commodities (bigger moves needed)
    "XAGUSD": 80.0,
    # USD Pairs
    "AUDUSD": 20.0,
    "EURUSD": 20.0,
    "GBPUSD": 25.0,
    # Commodity Cross
    "AUDCAD": 20.0,
}

# ── PATTERN MINING — Clustering ───────────────────────────

# Clustering algorithm: "dbscan" or "kmeans"
CLUSTER_ALGORITHM = "dbscan"

# DBSCAN: Maximum distance between two samples to be considered neighbors
# Lower = more patterns (stricter), Higher = fewer patterns (broader)
DBSCAN_EPS = 0.6

# DBSCAN: Minimum samples in a neighborhood to form a cluster
DBSCAN_MIN_SAMPLES = 8

# Feature groups used for clustering (subset of 93 features)
CLUSTER_FEATURES = [
    # Market quality
    "fq_final_score", "fq_market_score", "fq_smc_score",
    # Order flow
    "of_delta", "of_imbalance",
    # VWAP
    "vw_pip_from_vwap", "vw_pip_to_poc",
    # SMC structure
    "smc_pd_zone", "smc_pips_to_eq", "smc_smc_bias",
    # Momentum
    "vs_atr", "vs_momentum_velocity", "vs_choppy",
    # Currency strength
    "cs_base_strength", "cs_quote_strength", "cs_strength_delta",
    # ATR percentile
    "ap_atr_percentile", "ap_atr_ratio",
    # MTF RSI
    "mr_m5_rsi", "mr_m15_rsi", "mr_h1_rsi", "mr_h4_rsi",
    # MTF Score
    "mt_mtf_score", "mt_trend_agreement",
    # Intermarket
    "im_vix", "im_risk_env",
]

# ── PATTERN VALIDATION — Statistical Requirements ─────────

# Minimum occurrences to consider a pattern valid
MIN_PATTERN_OCCURRENCES = 30

# Minimum win rate for a valid pattern
MIN_PATTERN_WIN_RATE = 0.55

# Minimum profit factor (total wins / total losses in R)
MIN_PATTERN_PROFIT_FACTOR = 1.3

# Minimum backtest period in days to consider pattern validated
MIN_BACKTEST_DAYS = 60

# Pattern confidence tiers
PATTERN_TIERS = {
    "GOD_TIER": {
        "min_occurrences": 200,
        "min_win_rate": 0.70,
        "min_profit_factor": 2.0,
        "min_backtest_days": 180,
        "description": "Maximum confidence — deploy with full size",
    },
    "STRONG": {
        "min_occurrences": 80,
        "min_win_rate": 0.65,
        "min_profit_factor": 1.8,
        "min_backtest_days": 120,
        "description": "High confidence — normal position sizing",
    },
    "VALID": {
        "min_occurrences": 50,
        "min_win_rate": 0.60,
        "min_profit_factor": 1.5,
        "min_backtest_days": 90,
        "description": "Confirmed — standard position sizing",
    },
    "PROBATIONARY": {
        "min_occurrences": 30,
        "min_win_rate": 0.55,
        "min_profit_factor": 1.3,
        "min_backtest_days": 60,
        "description": "Needs more data — reduced position sizing (0.5x)",
    },
}

# ── PATTERN MODEL — XGBoost Classifier ────────────────────

# Target variable for pattern model:
# "binary" — predicts win/loss (1/0)
# "regression" — predicts expected R-multiple (continuous)
PATTERN_MODEL_TARGET = "regression"

# Minimum golden moments per pair to train a pattern model
MIN_TRAINING_SAMPLES = 80

# XGBoost parameters
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "mae",
    "early_stopping_rounds": 50,
}

# ── PATTERN GATE — L2 Confidence Scorer ───────────────────

# Minimum confidence to take a pattern-based trade
GATE_MIN_CONFIDENCE = 0.65

# Minimum predicted R to take a pattern-based trade
GATE_MIN_PREDICTED_R = 0.3

# Maximum patterns per pair (keep the best N)
MAX_PATTERNS_PER_PAIR = 50

# ── REPLAY BUFFER (for RPDE incremental training) ────────

REPLAY_BUFFER_SIZE = 5000
REPLAY_BUFFER_SAMPLE_SIZE = 3000
REPLAY_BUFFER_DECAY_HOURS = 720  # 30 days half-life for priority weighting

# ── CURRENCY CORRELATION BOOST ────────────────────────────

# Minimum correlation threshold to consider patterns currency-specific
CURRENCY_CORRELATION_THRESHOLD = 0.65

# Confidence boost when pattern confirmed across same-base-currency pairs
CURRENCY_CONFIRM_BOOST = 0.08

# ── LIVE TRADING ─────────────────────────────────────────

# Minimum time between pattern-based trades on same pair (minutes)
PATTERN_COOLDOWN_MINUTES = 30

# Maximum daily pattern-based trades
MAX_DAILY_PATTERN_TRADES = 8

# Maximum concurrent pattern-based positions
MAX_PATTERN_POSITIONS = 5
