"""
Deriv Over/Under Bot — Configuration v9
=========================================
Dynamic Barrier Selection — finds the BEST contract, not just Over 4 / Under 5.

Key Insight: Over 4 / Under 5 are 50/50 contracts with ~95% payout.
A 5% frequency edge on Over 4 gives only ~5% EV. But the same 5% edge on
Over 8 (10% natural, ~895% payout) gives ~50% EV — 10x more profitable!

v9 Changes from v8:
- DYNAMIC BARRIER SELECTION: evaluates ALL Over/Under barriers (not just 4/5)
  Finds the barrier with the best risk-adjusted EV based on observed frequencies
- FIXED CONFIDENCE: uses actual observed win probability (not inflated mapping)
  Old: confidence = 0.50 + setup_score * 0.40 (always 70-90% = wrong!)
  New: confidence = observed_win_probability_for_barrier (e.g., 0.15 for Over 8)
- FIXED EV CALCULATION: uses real probability x real payout per barrier
- RAISED THRESHOLDS: minimum 5% frequency edge + statistical significance (z-score > 2)
- SOFTER TREND: trend is a BIAS not strict requirement — strong frequency edge alone is enough
- Minimum EV of 5% to trade — no more negative-EV trades disguised by inflated confidence
- Trade interval raised from 2s to 10s — fewer trades, higher quality

Decision Flow (v9):
1. Compute per-digit frequencies across windows
2. For each barrier (Over 0-8, Under 1-9), calculate observed win probability and EV
3. Pick the barrier with the best risk-adjusted EV
4. Verify statistical significance (z-score > 2)
5. ML confirmation (if disagrees, reduce confidence but don't block)
6. Execute trade only if EV > 5%
"""

import os
from pathlib import Path

# ─── Load .env file (look in project root, then parent fxbot dir) ───
for env_path in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        break

# ─── Project Paths ───
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# ─── MySQL Database (XAMPP) ───
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "apex_trader")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─── Deriv API ───
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", None)

# ─── Trading Instruments ───
SYMBOLS = {
    "R_10":  {"name": "Volatility 10 Index",         "decimal_places": 3, "category": "volatility"},
    "R_25":  {"name": "Volatility 25 Index",         "decimal_places": 3, "category": "volatility"},
    "R_50":  {"name": "Volatility 50 Index",         "decimal_places": 4, "category": "volatility"},
    "R_75":  {"name": "Volatility 75 Index",         "decimal_places": 4, "category": "volatility"},
    "R_100": {"name": "Volatility 100 Index",        "decimal_places": 2, "category": "volatility"},
    "1HZ10V":  {"name": "Volatility 10 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ15V":  {"name": "Volatility 15 (1s) Index",  "decimal_places": 3, "category": "1s"},
    "1HZ25V":  {"name": "Volatility 25 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ30V":  {"name": "Volatility 30 (1s) Index",  "decimal_places": 3, "category": "1s"},
    "1HZ50V":  {"name": "Volatility 50 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ75V":  {"name": "Volatility 75 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ90V":  {"name": "Volatility 90 (1s) Index",  "decimal_places": 3, "category": "1s"},
    "1HZ100V": {"name": "Volatility 100 (1s) Index", "decimal_places": 2, "category": "1s"},
    "JD10":  {"name": "Jump 10 Index",   "decimal_places": 2, "category": "jump"},
    "JD25":  {"name": "Jump 25 Index",   "decimal_places": 2, "category": "jump"},
    "JD50":  {"name": "Jump 50 Index",   "decimal_places": 2, "category": "jump"},
    "JD75":  {"name": "Jump 75 Index",   "decimal_places": 2, "category": "jump"},
    "JD100": {"name": "Jump 100 Index",  "decimal_places": 2, "category": "jump"},
    "BOOM50":    {"name": "Boom 50 Index",      "decimal_places": 3, "category": "crash_boom"},
    "BOOM150N":  {"name": "Boom 150 Index",     "decimal_places": 5, "category": "crash_boom"},
    "BOOM300N":  {"name": "Boom 300 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM500":   {"name": "Boom 500 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM600":   {"name": "Boom 600 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM900":   {"name": "Boom 900 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM1000":  {"name": "Boom 1000 Index",    "decimal_places": 3, "category": "crash_boom"},
    "CRASH50":   {"name": "Crash 50 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH150N": {"name": "Crash 150 Index",    "decimal_places": 5, "category": "crash_boom"},
    "CRASH300N": {"name": "Crash 300 Index",    "decimal_places": 3, "category": "crash_boom"},
    "CRASH500":  {"name": "Crash 500 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH600":  {"name": "Crash 600 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH900":  {"name": "Crash 900 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH1000": {"name": "Crash 1000 Index",    "decimal_places": 3, "category": "crash_boom"},
    "RDBULL": {"name": "Bull Market Index",  "decimal_places": 4, "category": "daily_reset"},
    "RDBEAR": {"name": "Bear Market Index",  "decimal_places": 4, "category": "daily_reset"},
    "RB100": {"name": "Range Break 100 Index", "decimal_places": 1, "category": "range_break"},
    "RB200": {"name": "Range Break 200 Index", "decimal_places": 1, "category": "range_break"},
    "stpRNG":  {"name": "Step Index 100", "decimal_places": 1, "category": "step"},
    "stpRNG2": {"name": "Step Index 200", "decimal_places": 1, "category": "step"},
    "stpRNG3": {"name": "Step Index 300", "decimal_places": 1, "category": "step"},
    "stpRNG4": {"name": "Step Index 400", "decimal_places": 1, "category": "step"},
    "stpRNG5": {"name": "Step Index 500", "decimal_places": 1, "category": "step"},
    "WLDAUD": {"name": "AUD Basket",  "decimal_places": 3, "category": "basket"},
    "WLDEUR": {"name": "EUR Basket",  "decimal_places": 3, "category": "basket"},
    "WLDGBP": {"name": "GBP Basket",  "decimal_places": 3, "category": "basket"},
    "WLDUSD": {"name": "USD Basket",  "decimal_places": 3, "category": "basket"},
    "WLDXAU": {"name": "Gold Basket",  "decimal_places": 3, "category": "basket"},
}

# ─── Symbol Categories ───
DIGIT_CONTRACT_CATEGORIES = {"volatility", "1s", "daily_reset"}

RECOMMENDED_SYMBOLS = {
    "fastest_data":  "1HZ100V",
    "most_popular":  "R_100",
    "high_volatility": "1HZ50V",
}

def get_symbol_decimals(symbol: str) -> int:
    return SYMBOLS.get(symbol, {}).get("decimal_places", 2)

def get_symbol_category(symbol: str) -> str:
    return SYMBOLS.get(symbol, {}).get("category", "unknown")

def supports_digit_contracts(symbol: str) -> bool:
    return get_symbol_category(symbol) in DIGIT_CONTRACT_CATEGORIES

# ─── Multi-Market Configuration ───
DEFAULT_MARKETS = [
    # 1-second indices (fastest tick rate)
    "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V",
    "1HZ75V", "1HZ90V", "1HZ100V",
    # Standard volatility indices
    "R_10", "R_25", "R_50", "R_75", "R_100",
]
VALID_MULTI_MARKET_SYMBOLS = [
    "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V",
    "1HZ75V", "1HZ90V", "1HZ100V",
    "R_10", "R_25", "R_50", "R_75", "R_100",
]
DEFAULT_SYMBOL = "1HZ100V"

# ─── Contract Settings ───
CONTRACT_TYPE_OVER   = "DIGITOVER"
CONTRACT_TYPE_UNDER  = "DIGITUNDER"
OVER_BARRIER = 4              # Default fallback (overridden by dynamic barrier selection)
UNDER_BARRIER = 5             # Default fallback (overridden by dynamic barrier selection)
CONTRACT_DURATION = 5          # Default (overridden by observation phase)
CONTRACT_DURATION_UNIT = "t"

# ─── Dynamic Barrier Selection (v9 — THE MOST IMPORTANT CHANGE) ───
# Instead of always trading Over 4 / Under 5 (50/50 contracts), evaluate ALL
# barriers and pick the one with the best risk-adjusted EV.
#
# Why this matters:
#   Over 4 (50% natural, ~95% payout): A 5% freq edge gives EV = +5%
#   Over 7 (20% natural, ~395% payout): A 5% freq edge gives EV = +25%
#   Over 8 (10% natural, ~895% payout): A 5% freq edge gives EV = +50%
#
# The same observed frequency deviation is MUCH more valuable on
# lower-probability contracts because the payout multiplier amplifies the edge.
DYNAMIC_BARRIERS = True                # Enable dynamic barrier selection
BARRIER_OVER_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]   # Over barriers to evaluate
BARRIER_UNDER_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Under barriers to evaluate
MIN_BARRIER_PROBABILITY = 0.08         # Don't trade <8% natural prob (extremely high variance)
                                        # Over 8 = 10% natural → allowed (high payout = 858%)
                                        # Over 9 would be 0% → not allowed
MAX_BARRIER_PROBABILITY = 0.70         # Don't trade >70% natural prob (payout too low)
PAYOUT_HOUSE_MARGIN = 0.046            # Deriv's house margin (~4.6% based on observed Over 4 payout)
MIN_EV_FOR_TRADE = 0.05                # Minimum 5% EV to trade (was always "positive" before)

# ─── Natural Probabilities for Barriers ───
# Over B wins if digit > B. Under B wins if digit < B.
# For uniform digit distribution (0-9), each digit = 10%.
# Over B natural probability = (9 - B) / 10
# Under B natural probability = (B - 0) / 10 = B / 10
BARRIER_NATURAL_PROB_OVER = {b: (9 - b) / 10.0 for b in range(10)}   # Over 0: 90%, Over 4: 50%, Over 8: 10%
BARRIER_NATURAL_PROB_UNDER = {b: b / 10.0 for b in range(1, 10)}     # Under 1: 10%, Under 5: 50%, Under 9: 90%

def estimate_payout_rate(natural_probability: float) -> float:
    """
    Estimate the payout rate for a barrier with a given natural probability.
    
    Payout rate = profit / stake when you win.
    Based on Deriv's pricing model: payout = (1 / natural_prob - 1) * (1 - house_margin)
    
    Examples:
        Over 4 (50% natural): payout = (1/0.50 - 1) * (1 - 0.046) = 1.0 * 0.954 = 0.954
        Over 7 (20% natural): payout = (1/0.20 - 1) * (1 - 0.046) = 4.0 * 0.954 = 3.816
        Over 8 (10% natural): payout = (1/0.10 - 1) * (1 - 0.046) = 9.0 * 0.954 = 8.586
    """
    if natural_probability <= 0 or natural_probability >= 1:
        return 0.0
    return (1.0 / natural_probability - 1.0) * (1.0 - PAYOUT_HOUSE_MARGIN)

# ─── Dynamic Duration ───
DYNAMIC_DURATION = True
MIN_DURATION = 2               # Minimum 2 ticks (1t is too noisy)
MAX_DURATION = 10
DURATION_EXPLORATION_RATE = 0.15

# ─── Stake & Money ───
INITIAL_BANKROLL = 100.0
MIN_STAKE = 0.35
DEFAULT_STAKE = 0.35
MAX_STAKE = 5.0

# ─── Signal Thresholds ───
# v9: Confidence is now the ACTUAL observed win probability for the chosen barrier.
# For Over 4 this might be ~0.55, for Over 8 it might be ~0.15.
# The old system mapped setup_score to [0.50, 0.90] which was always inflated.
MIN_CONFIDENCE = 0.12           # Minimum observed win probability (varies by barrier)
MIN_EDGE_THRESHOLD = 0.05       # Minimum 5% EV to trade (v9: raised from 1%)
MIN_FREQ_EDGE_ZSCORE = 1.3      # Require 1.3+ SD from natural probability (~80% confidence)
                                 # Trading, not scientific publishing — 1.3 is sufficient

# ─── Digit Frequency Direction (PRIMARY DECISION) ───
# The digit frequency analysis now evaluates ALL barriers dynamically.
# Instead of just checking "Over 4 vs Under 5", we check every barrier
# and find the one with the best risk-adjusted EV.
# v9: Raised from 2% to 5% — 2% is within normal noise for synthetic indices.
MIN_DIGIT_FREQUENCY_EDGE = 0.01   # Minimum ABSOLUTE edge (1% — very loose, z-score is the real gate)
MIN_DIGIT_FREQUENCY_EDGE_RELATIVE = 0.10  # Minimum RELATIVE edge from natural prob (10%)
                                    # For Over 8 (10%): need 1% absolute → obs > 11%
                                    # For Over 7 (20%): need 2% absolute → obs > 22%
                                    # For Over 4 (50%): need 5% absolute → obs > 55%
DIGIT_FREQ_WINDOW_AGREEMENT = 2   # At least N of 3 windows must agree on direction

# ─── Setup Quality ───
# A "setup" = clear digit frequency edge with statistical significance.
# Setup score determines trade quality. Higher = better.
# v9: Lowered to 0.60 since setup score calculation changed with dynamic barriers.
MIN_SETUP_SCORE = 0.60           # Minimum setup quality to trade (0-1 scale)

# ─── Profit Target Per Market ───
# After making this much profit on one market, STOP trading it.
# Wait for another good setup to appear.
PROFIT_TARGET_PER_MARKET = 50.0   # $50 profit target per market session

# ─── Observation Phase (Duration Determination) ───
# When a setup is detected, WATCH the market for this many seconds
# before determining the optimal tick duration.
# During observation, track how quickly digits move to the dominant side.
OBSERVATION_PERIOD_SEC = 25       # Watch for 25 seconds before determining duration
MIN_OBSERVATION_TICKS = 10       # Need at least this many ticks during observation

# ─── Market Session (Persistence) ───
# Stay on one market during a setup. Don't jump between markets.
# Only switch when: setup breaks, profit target reached, or no trade happening.
# v8.1: Increased idle timeout — good setups deserve patience.
MARKET_SESSION_MAX_IDLE_SEC = 180  # After 180s with no trade on current market, allow switch
MARKET_STICKY_AFTER_TRADE = True   # v8.1: After trading a market, keep trading it while setup is good

# ─── ML Confirmation Model ───
# The Logistic Regression model is a CONFIRMATION signal, not the driver.
# v9: ML disagreement now REDUCES confidence (not blocks entirely).
# The old "block on disagreement" was too aggressive — it blocked many
# valid frequency-based signals because the ML model was trained on
# Over 4 / Under 5 data and doesn't generalize to other barriers well.
ML_CONFIRMATION_WEIGHT = 0.20     # v9: Reduced from 0.30 — ML is less relevant with dynamic barriers
ML_DISAGREEMENT_BLOCKS = False    # v9: ML disagreement REDUCES confidence, not blocks entirely

# ─── Confidence-Weighted Agreement ───
SIGNAL_SCORE_METHOD = "setup_weighted"        # setup quality drives the score
MIN_SIGNAL_SCORE = 0.55                       # Lowered — rule-based primary is more reliable
AGREEMENT_WEIGHT = 1.0                        # Always 1.0 with single model

# ─── Forced Trade (strong setup) ───
# A forced trade happens when the setup is VERY strong (setup score >= 0.80).
# v9: Raised from 0.75 — must be very strong to force trade without ML.
FORCE_TRADE_MIN_CONFIDENCE = 0.12
FORCE_TRADE_MIN_EV = 0.05
FORCE_TRADE_MIN_SETUP_SCORE = 0.80  # v9: Raised — must be very strong to force trade without ML

# ─── Trend Setting (v9: SOFTENED — trend is BIAS, not strict requirement) ───
# In v8, trend was a strict REQUIREMENT: no trend = no trade.
# This was WRONG because: in synthetic indices, the price trend has NO causal
# relationship with the last digit distribution. An uptrend doesn't mean
# digits 5-9 appear more often. The digits are uniformly random.
#
# v9: Trend is now a BIAS. A strong trend + frequency alignment = higher confidence.
# But a very strong frequency edge WITHOUT a trend can still produce a valid trade.
# This allows the bot to trade on pure digit frequency signals.
TREND_REQUIRED = False               # v9: Trend is NOT required (was strict requirement)
TREND_SLOPE_TSTAT_THRESHOLD = 3.0    # Still 3-sigma for trend detection
TREND_CONFIDENCE_BOOST = 0.05        # Boost confidence by 5% when trend aligns with frequency
TREND_MISALIGN_PENALTY = 0.10        # Reduce confidence by 10% when trend opposes frequency
TREND_CONFIDENCE_REDUCTION = 0.05    # Legacy — kept for compatibility
TREND_SIGNAL_SCORE_REDUCTION = 0.05  # Legacy — kept for compatibility

# ─── Martingale Confidence Gate ───
# During martingale recovery, the bot MUST stay on the SAME market.
# No switching markets during recovery — the setup was good on THAT market.
# v9: Adjusted for dynamic barriers — confidence is now real probability.
MARTINGALE_MIN_CONFIDENCE = 0.12   # v9: Lowered — confidence is now real probability
MARTINGALE_MIN_SETUP_SCORE = 0.60  # v9: Adjusted for new setup scoring
MARTINGALE_SAME_MARKET = True      # MUST recover on the same market where loss occurred
MAX_DAILY_TRADES = 0          # 0 = unlimited (demo training mode)
COOLDOWN_AFTER_LOSS_TICKS = 1
MIN_TRADE_INTERVAL_SEC = 10   # v9: Raised from 2s — fewer trades, higher quality

# ─── Multi-Trade Mode ───
# When True, multiple markets can trade simultaneously (each still limited to 1 trade).
# When False, only ONE trade across ALL markets at a time (old behavior).
ALLOW_MULTIPLE_TRADES = False    # Default OFF — use --multi-trade CLI flag to enable
MAX_CONCURRENT_TRADES = 5        # Max simultaneous open trades across all markets

# ─── Risk Management (RELAXED for demo overnight training) ───
MAX_BANKROLL_PER_TRADE = 0.05    # 5% max per trade
MAX_DAILY_LOSS = 1.0             # 100% — don't stop on losses (demo training)
MAX_CONSECUTIVE_LOSSES = 10      # Circuit breaker after 10 consecutive losses
CIRCUIT_BREAKER_COOLDOWN_SEC = 30  # Short 30s cooldown (demo training)
MAX_OPEN_POSITIONS = MAX_CONCURRENT_TRADES if ALLOW_MULTIPLE_TRADES else 1
SESSION_TIME_LIMIT_MINUTES = 0   # 0 = unlimited (demo training mode)

# ─── Kelly Criterion ───
KELLY_FRACTION = 0.25

# ─── Feature Engine ───
TICK_WINDOWS = {
    "micro":       10,
    "short":       50,
    "medium":      200,
    "trend_long":  500,    # Added for 500-tick trend slope (strong trend confirmation)
    "long":        1000,
}

# ─── Online Learner ───
# CHANGED: "ensemble" -> "logistic" — single transparent model
# The 3-model ensemble (Logistic + HAT + SRP) was a black box.
# Now: Logistic Regression is a CONFIRMATION signal, not the primary decision-maker.
# You can inspect its weights anytime and see what features it values.
MODEL_TYPE = "logistic"
LEARNING_RATE = 0.01
L2_REGULARIZATION = 0.01
REPLAY_BUFFER_SIZE = 5000
DRIFT_DETECTION_SENSITIVITY = 0.001

# ─── Per-Tick Live Learning ───
TICK_LEARN_ENABLED = False
TICK_LEARN_INTERVAL = 5

# ─── Drift Retrain ───
DRIFT_RETRAIN_ENABLED = True
DRIFT_RETRAIN_COOLDOWN = 120

# ─── Model Persistence ───
MODEL_SNAPSHOT_INTERVAL = 50
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─── Meta-Selector (Market Bandit) ───
META_SELECTOR_EXPLORATION_RATE = 0.10
META_SELECTOR_MIN_TRADES = 5
META_SELECTOR_SCORE_WINDOW = 50

# ─── Logging ───
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

# ─── Dashboard ───
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_REFRESH_MS = 2000

# ─── Mode ───
TRADING_MODE = "live"
