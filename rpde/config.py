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
# Higher value needed because we cluster in 26-dimensional feature space.
# In high dimensions, inter-point distances grow naturally.
# Too low = everything is noise (0 clusters). Too high = 1 giant cluster.
DBSCAN_EPS = 2.5

# DBSCAN: Minimum samples in a neighborhood to form a cluster
# Reduced from 8 to 5 — with strict validation gates (PF≥3, WR≥55%),
# smaller seed clusters are fine since they'll be filtered later.
DBSCAN_MIN_SAMPLES = 5

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
# Higher threshold = only keep patterns with strong edge
MIN_PATTERN_PROFIT_FACTOR = 3.0

# Minimum backtest period in days to consider pattern validated
MIN_BACKTEST_DAYS = 60

# Maximum Average MAE as fraction of move (stop loss quality)
# e.g., 0.6 means: avg adverse excursion must be < 60% of the average move
# Patterns that require deep pullbacks before succeeding are less reliable
MAX_AVG_MAE_MOVE_RATIO = 0.6

# Pattern confidence tiers
PATTERN_TIERS = {
    "GOD_TIER": {
        "min_occurrences": 200,
        "min_win_rate": 0.70,
        "min_profit_factor": 5.0,
        "min_backtest_days": 180,
        "description": "Maximum confidence — deploy with full size",
    },
    "STRONG": {
        "min_occurrences": 80,
        "min_win_rate": 0.65,
        "min_profit_factor": 4.0,
        "min_backtest_days": 120,
        "description": "High confidence — normal position sizing",
    },
    "VALID": {
        "min_occurrences": 50,
        "min_win_rate": 0.60,
        "min_profit_factor": 3.5,
        "min_backtest_days": 90,
        "description": "Confirmed — standard position sizing",
    },
    "PROBATIONARY": {
        "min_occurrences": 30,
        "min_win_rate": 0.55,
        "min_profit_factor": 3.0,
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

# ── PHASE 2: TEMPORAL FUSION TRANSFORMER (TFT) ────────────

# Multi-timeframe sequence lengths (number of candles per timeframe)
# Chosen to capture meaningful price action at each granularity:
#   H4: 30 candles = 5 days of 4H bars
#   H1: 48 candles = 2 days of 1H bars
#   M15: 96 candles = 24 hours of 15min bars
#   M5: 60 candles = 5 hours of 5min bars (recent microstructure)
TFT_TIMEFRAMES = {
    "H4":  30,
    "H1":  48,
    "M15": 96,
    "M5":  60,
}

# Raw OHLCV features per candle fed to TFT (no engineered features)
# The TFT learns its own representations from raw price data
TFT_RAW_FEATURES = [
    "open", "high", "low", "close", "tick_volume",
    # Derived per-candle features (computed in dataset builder)
    "body_ratio",      # |close - open| / (high - low)
    "upper_wick",      # (high - max(open, close)) / (high - low)
    "lower_wick",      # (min(open, close) - low) / (high - low)
    "range_pct",       # (high - low) / close (normalized range)
    "return_1",        # close / prev_close - 1
    "return_3",        # close / close_3bars_ago - 1
]  # 11 features per candle per timeframe

# TFT model hyperparameters
TFT_HIDDEN_SIZE = 64           # Hidden dimension for all encoders/decoders
TFT_ATTENTION_HEADS = 4        # Multi-head attention heads
TFT_NUM_ENCODER_LAYERS = 2     # Transformer encoder layers per timeframe
TFT_DROPOUT = 0.15             # Dropout rate (regularization)
TFT_LEARNING_RATE = 1e-4       # AdamW learning rate
TFT_WEIGHT_DECAY = 1e-5        # L2 regularization
TFT_BATCH_SIZE = 32            # Training batch size
TFT_EPOCHS = 50                # Maximum training epochs
TFT_PATIENCE = 8               # Early stopping patience (epochs)
TFT_GRADIENT_CLIP = 1.0        # Max gradient norm for clipping

# Variable Selection Network (VSN) — per-timeframe importance weights
VSN_ENABLED = True             # Enable variable selection per timeframe
VSN_HIDDEN_SIZE = 16           # VSN MLP hidden size

# Cross-timeframe attention
CROSS_TF_ATTENTION_ENABLED = True   # Enable cross-TF attention mechanism
CROSS_TF_ATTENTION_HEADS = 4        # Heads for cross-TF attention
CROSS_TF_ATTENTION_LAYERS = 2       # Layers of cross-TF attention

# TFT output heads (3 prediction targets)
TFT_OUTPUT_HEADS = {
    "candle_pattern_match": {
        "type": "classification",    # Binary: does current state match known pattern?
        "hidden_dim": 32,
    },
    "momentum_score": {
        "type": "regression",        # Continuous: how strong is the momentum? [-1, 1]
        "hidden_dim": 32,
    },
    "reversal_probability": {
        "type": "regression",        # Continuous: probability of imminent reversal [0, 1]
        "hidden_dim": 32,
    },
}

# TFT minimum training data
TFT_MIN_TRAINING_SAMPLES = 500    # Minimum sequences to start training
TFT_TRAIN_VAL_SPLIT = 0.8         # Time-based split (80% train, 20% val)
TFT_SEQUENCE_STRIDE = 5           # Steps between consecutive training sequences
TFT_RETRAIN_DAYS = 14             # Bi-weekly retraining cadence (days)
TFT_MIN_PAIR_TRADES = 30          # Min trades on a pair before including in TFT training

# TFT device configuration
TFT_DEVICE = "auto"               # "auto" (GPU if available, else CPU), "cuda", "cpu"
TFT_MIXED_PRECISION = True        # Use fp16 for faster GPU training
TFT_NUM_WORKERS = 2               # DataLoader workers

# ── PHASE 2: FUSION LAYER ──────────────────────────────────

# How to combine XGBoost and TFT predictions
# Weights are LEARNED per pair via a small meta-learner
FUSION_DEFAULT_XGB_WEIGHT = 0.55  # Default: XGBoost slightly favored (more mature)
FUSION_DEFAULT_TFT_WEIGHT = 0.45  # Default: TFT weight (adjusts with training)

# Fusion layer meta-learner
FUSION_META_LR = 0.01            # Learning rate for fusion weight optimizer
FUSION_WEIGHT_SMOOTHING = 0.9    # EMA smoothing for weight updates (prevents jitter)
FUSION_MIN_WEIGHT = 0.1          # Minimum weight for either signal (never fully ignore)
FUSION_MAX_WEIGHT = 0.9          # Maximum weight for either signal

# Direction agreement boost — when XGB and TFT agree on direction
FUSION_DIRECTION_AGREE_BOOST = 0.10   # +10% confidence when both agree
FUSION_DIRECTION_DISAGREE_PENALTY = 0.20  # -20% confidence when they disagree

# TFT contribution thresholds for gate decisions
TFT_MIN_PATTERN_MATCH = 0.5      # Minimum pattern match score from TFT
TFT_MIN_MOMENTUM_SCORE = 0.3     # Minimum momentum to consider TFT signal valid
TFT_REVERSAL_THRESHOLD = 0.7     # Above this, consider reversal warning

# ── LIVE TRADING ─────────────────────────────────────────

# Minimum time between pattern-based trades on same pair (minutes)
PATTERN_COOLDOWN_MINUTES = 30

# Maximum daily pattern-based trades
MAX_DAILY_PATTERN_TRADES = 8

# Maximum concurrent pattern-based positions
MAX_PATTERN_POSITIONS = 5

# ── PHASE 3: RL DECISION ENGINE ─────────────────────────────

# Observation / action space dimensions
RL_OBS_DIM = 28               # 8 fusion + 12 market + 8 portfolio
RL_ACTION_DIM = 7             # SKIP + 3 BUY sizes + 3 SELL sizes

# Episode management
RL_MAX_STEPS_PER_EPISODE = 50  # Max signals per trading day

# Reward function
RL_REWARD_SCALING = 1.0        # Multiplier on R-multiple rewards
RL_BAD_ENTRY_PENALTY = -0.1    # Additional penalty when trade loses > 1R
RL_GOOD_ENTRY_BONUS = 0.05     # Additional bonus when trade profits > 1R
RL_DRAWDOWN_SHAPING = -0.001   # Per 1% drawdown increase per step

# Risk limits enforced inside the environment
RL_MAX_DAILY_LOSS_PCT = 3.0    # Hard daily loss limit (pct of equity)
RL_MAX_POSITIONS = 5           # Max concurrent open positions
RL_MAX_CONSECUTIVE_LOSSES = 3  # Pause after this many consecutive losses

# Size → SL/TP mapping (in R-multiples)
# Maps each size tier to its stop-loss width and take-profit target
RL_SIZE_MAP = {
    0.5: {"sl_r": 0.5, "tp_r": 1.0},   # Conservative
    1.0: {"sl_r": 1.0, "tp_r": 2.0},   # Standard
    1.5: {"sl_r": 1.5, "tp_r": 3.0},   # Aggressive
}

# PPO network architecture
RL_ACTOR_HIDDEN = 128           # Actor (policy) hidden dimension
RL_CRITIC_HIDDEN = 128          # Critic (value) hidden dimension

# PPO algorithm hyperparameters
RL_LEARNING_RATE = 3e-4         # Adam optimizer learning rate
RL_GAMMA = 0.99                 # Discount factor for future rewards
RL_GAE_LAMBDA = 0.95            # GAE lambda for advantage estimation
RL_CLIP_RATIO = 0.2             # PPO clipping ratio (epsilon)
RL_VALUE_COEFF = 0.5            # Value loss coefficient
RL_ENTROPY_COEFF = 0.01         # Entropy bonus coefficient (encourages exploration)
RL_MAX_GRAD_NORM = 0.5          # Max gradient norm for clipping

# PPO training loop
RL_PPO_EPOCHS = 4               # PPO update epochs per rollout
RL_MINI_BATCH_SIZE = 64         # Mini-batch size for PPO updates
RL_ROLLOUT_STEPS = 2048         # Steps per rollout before PPO update

# RL training cadence and data requirements
RL_RETRAIN_DAYS = 7             # Retrain RL agent weekly
RL_MIN_TRAINING_EPISODES = 50   # Minimum episodes before RL agent is considered trained

# RL device configuration (shared with TFT)
RL_DEVICE = "auto"              # "auto" (GPU if available), "cuda", "cpu"
RL_MIXED_PRECISION = True       # Use fp16 for faster GPU training

# ── PHASE 3: SAFETY GUARDS (non-overridable) ──────────────────

# Drawdown limits (HARD — trigger system shutdown)
SAFETY_MAX_DAILY_LOSS_PCT = 3.0     # Daily loss > 3% → SHUT DOWN
SAFETY_MAX_WEEKLY_LOSS_PCT = 5.0    # Weekly loss > 5% → SHUT DOWN

# Position limits (SOFT — skip trade)
SAFETY_MAX_POSITIONS = 5            # Max concurrent positions across all pairs
SAFETY_MAX_PER_PAIR = 2             # Max concurrent positions per pair

# Consecutive loss protection
SAFETY_MAX_CONSECUTIVE_LOSSES = 5    # Pause trading after N consecutive losses

# Margin protection (HARD — trigger system shutdown)
SAFETY_MARGIN_LEVEL_MIN = 150.0     # margin_level < 150% → SHUT DOWN
SAFETY_EQUITY_MIN_PCT = 50.0        # equity < 50% of balance → SHUT DOWN
SAFETY_FREE_MARGIN_BUFFER = 2.0     # free_margin must be > required_margin * this

# News filter (SOFT — skip trade near high-impact events)
SAFETY_NEWS_BUFFER_MINUTES = 15     # Skip if high-impact news within N minutes
SAFETY_MEDIUM_NEWS_BUFFER_MINUTES = 5  # Skip if medium-impact news within N minutes

# Weekend filter (SOFT — no trading around weekends)
SAFETY_FRIDAY_CLOSE_HOUR = 20       # UTC hour: no new trades after Friday 20:00
SAFETY_MONDAY_OPEN_MINUTE = 5       # UTC minute: wait until Monday 00:05 (spread wide)

# Dynamic filters
SAFETY_SPREAD_MULTIPLIER = 3.0      # Skip if spread > N × average spread
SAFETY_ATR_EXTREME_MULTIPLIER = 3.0 # Skip if ATR > N × average ATR (too volatile)

# Shutdown cooldown
SAFETY_COOLDOWN_AFTER_SHUTDOWN_HOURS = 2  # Min hours after manual reset before trading

# Per-pair maximum spread limits (pips) — static floor
SPREAD_LIMITS = {
    "EURUSD": 2.0, "GBPUSD": 3.0, "EURJPY": 3.0,
    "GBPJPY": 4.0, "AUDUSD": 2.5, "AUDJPY": 3.5,
    "CHFJPY": 3.0, "CADJPY": 3.0, "XAGUSD": 5.0,
    "AUDCAD": 3.5,
}

# ── LIVE ENGINE (Phase 3: Live Trading Orchestrator) ──────────

# Default risk percent per trade (fraction of equity)
LIVE_RISK_PERCENT_PER_TRADE = 0.01    # 1% of equity per trade

# Minimum SL in pips (broker floor)
LIVE_MIN_SL_PIPS = 3.0

# RL mid-trade check interval (seconds)
LIVE_MID_TRADE_INTERVAL_SEC = 30

# Trailing stop activation threshold (R-multiples in profit)
LIVE_TRAILING_ACTIVATE_R = 1.0

# Trailing stop distance (fraction of ATR once activated)
LIVE_TRAILING_ATR_FRACTION = 0.5

# Breakeven move threshold (R-multiples in profit to move SL to entry)
LIVE_BREAKEVEN_ACTIVATE_R = 0.5

# Partial exit thresholds — close 50% at 1R, let rest run to TP
LIVE_PARTIAL_EXIT_R = 1.0
LIVE_PARTIAL_EXIT_PCT = 0.5          # Close 50% of position

# Early exit — RL can suggest early close if confidence drops
LIVE_EARLY_EXIT_ENABLED = True
LIVE_EARLY_EXIT_MIN_CONFIDENCE = 0.3  # Below this, RL suggests exit

# Real-time P&L feedback → RL env
LIVE_PNL_FEEDBACK_ENABLED = True
LIVE_PNL_UPDATE_INTERVAL_SEC = 10

# Max execution delay (ms) before abandoning a signal
LIVE_MAX_EXECUTION_DELAY_MS = 5000

# Position management loop interval (seconds)
LIVE_MANAGE_INTERVAL_SEC = 1.0

# Paper mode default (always True for safety)
LIVE_PAPER_MODE_DEFAULT = True
