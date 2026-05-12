"""
Deriv Over/Under Bot — Configuration v8
=========================================
Structured Quality Trading — mirrors manual trading philosophy:
1. Trend check (3-window agreement, 3-sigma threshold)
2. Digit frequency Over/Under analysis (the PRIMARY direction signal)
3. Observation phase (20-30s watching digit movement -> determines duration)
4. Execute few high-quality trades, then stop that market
5. Profit target per market session ($50)
6. Single Logistic Regression as ML CONFIRMATION (not primary decision-maker)

v8 Changes from v7:
- Digit frequency Over/Under split as PRIMARY direction decision
- Single Logistic Regression model replaces 3-model ensemble (transparent, verifiable)
- Observation phase for duration (watch digits for 20-30s before trading)
- Profit target per market ($50) -> stop trading that market until new setup
- Market persistence: stay on one market during a setup, don't jump around
- Setup quality scoring: trend + digit frequency edge must be strong
- Increased martingale steps to 3 (trusted setups, 85% payout adjusted)
- Setup detector: encapsulates trend + digit frequency analysis
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
OVER_BARRIER = 4
UNDER_BARRIER = 5
CONTRACT_DURATION = 5          # Default (overridden by observation phase)
CONTRACT_DURATION_UNIT = "t"

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
MIN_CONFIDENCE = 0.55           # Lowered — rule-based primary + ML confirmation
MIN_EDGE_THRESHOLD = 0.01

# ─── Digit Frequency Direction (PRIMARY DECISION) ───
# The digit frequency Over/Under split is the PRIMARY direction signal.
# If Over-frequency > 50% + MIN_DIGIT_FREQUENCY_EDGE across multiple windows,
# the direction is Over. Vice versa for Under.
# This replaces the ML model as the primary decision-maker.
MIN_DIGIT_FREQUENCY_EDGE = 0.02   # Minimum 2% edge from 50/50 (e.g., 52% Over)
DIGIT_FREQ_WINDOW_AGREEMENT = 2   # At least N of 3 windows must agree on direction

# ─── Setup Quality ───
# A "setup" = strong trend + clear digit frequency edge.
# Setup score determines trade quality. Higher = better.
# v8.1: Raised from 0.60 to 0.70 — only trade STRONG setups.
# Weak setups (0.60-0.70) lead to losses that martingale can't recover from.
MIN_SETUP_SCORE = 0.70           # Minimum setup quality to trade (0-1 scale)

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
# When rules say Over and ML also says Over -> HIGH confidence -> trade
# When rules say Over but ML says Under -> NO TRADE — the setup isn't clear enough
# v8.1: ML disagreement now BLOCKS trades entirely. If the setup is truly
# strong, the ML model should agree. Disagreement = setup is questionable.
ML_CONFIRMATION_WEIGHT = 0.30     # How much ML confirmation affects confidence (0-1)
ML_DISAGREEMENT_BLOCKS = True     # v8.1: ML disagreement now BLOCKS trades (not just penalty)

# ─── Confidence-Weighted Agreement ───
SIGNAL_SCORE_METHOD = "setup_weighted"        # NEW: setup quality drives the score
MIN_SIGNAL_SCORE = 0.55                       # Lowered — rule-based primary is more reliable
AGREEMENT_WEIGHT = 1.0                        # Always 1.0 with single model

# ─── Forced Trade (strong setup) ───
# A forced trade happens when the setup is VERY strong (setup score >= 0.75).
# This replaces the old "100% model agreement" forced trade.
FORCE_TRADE_MIN_CONFIDENCE = 0.55
FORCE_TRADE_MIN_EV = 0.0
FORCE_TRADE_MIN_SETUP_SCORE = 0.75  # Setup must be VERY strong to force trade

# ─── Trend Requirement (NOT bias — TRADES ONLY IN STRONG TRENDS) ───
# Linear regression slope on price detects market trend direction.
# Trades ONLY happen when a VERY STRONG trend is confirmed across all windows.
# Uptrend -> ONLY Over trades allowed
# Downtrend -> ONLY Under trades allowed
# Ranging (no trend) -> NO TRADES AT ALL — wait for a strong trend
#
# This is a REQUIREMENT, not a bias — no trend = no trade.
# Uses 50, 200, and 500 tick windows. All must agree on direction.
# 200-tick and 500-tick must BOTH have t-stat > threshold (very significant).
# 50-tick must at least agree in direction (catches recent turns).
TREND_SLOPE_TSTAT_THRESHOLD = 3.0    # 3-sigma = 99.7% confidence
TREND_CONFIDENCE_REDUCTION = 0.05    # Lower confidence by 5% for trend-aligned trades
TREND_SIGNAL_SCORE_REDUCTION = 0.05  # Lower signal_score threshold for trend-aligned trades

# ─── Martingale Confidence Gate ───
# During martingale recovery, the bot MUST stay on the SAME market.
# No switching markets during recovery — the setup was good on THAT market.
# v8.1: Removed ML confidence gate during martingale — it was blocking recovery
# on good setups because the ML model's confidence was low (~55%).
# Instead, we trust the SETUP quality (trend + frequency edge) for recovery.
MARTINGALE_MIN_CONFIDENCE = 0.55   # v8.1: Lowered — trust setup quality, not ML confidence
MARTINGALE_MIN_SETUP_SCORE = 0.65  # v8.1: Setup must still be decent during recovery
MARTINGALE_SAME_MARKET = True      # v8.1: MUST recover on the same market where loss occurred
MAX_DAILY_TRADES = 0          # 0 = unlimited (demo training mode)
COOLDOWN_AFTER_LOSS_TICKS = 1
MIN_TRADE_INTERVAL_SEC = 2    # Minimum seconds between trades

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
