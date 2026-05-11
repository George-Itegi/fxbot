"""
Deriv Over/Under Bot — Configuration v7
=========================================
Multi-market architecture — ALL volatility markets with independent models.
Each market gets its OWN ensemble model (logistic + HAT + SRP).
LIVE trading on demo account — limits removed for overnight training.

v7 Changes:
- Added ALLOW_MULTIPLE_TRADES flag for concurrent trades across markets
- MAX_CONCURRENT_TRADES caps simultaneous open positions
- Per-symbol loss cooldown when multi-trade is on (only losing market waits)
- Set ALLOW_MULTIPLE_TRADES=False for old one-at-a-time behavior
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
    "WLDXAU": {"name": "Gold Basket", "decimal_places": 3, "category": "basket"},
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
# ALL volatility + 1s markets — each gets its own independent ensemble model
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
CONTRACT_DURATION = 5
CONTRACT_DURATION_UNIT = "t"

# ─── Dynamic Duration ───
DYNAMIC_DURATION = True
MIN_DURATION = 1
MAX_DURATION = 10
DURATION_EXPLORATION_RATE = 0.15

# ─── Stake & Money ───
INITIAL_BANKROLL = 100.0
MIN_STAKE = 0.35
DEFAULT_STAKE = 0.35
MAX_STAKE = 5.0

# ─── Signal Thresholds ───
MIN_CONFIDENCE = 0.56
MIN_EDGE_THRESHOLD = 0.01

# ─── Forced Trade (100% agreement) ───
# All 3 models must not only VOTE the same way but also have REAL confidence.
# A 54% probability with 3 binary votes = fake agreement.
# All models must have confidence >= this threshold to force a trade.
FORCE_TRADE_MIN_CONFIDENCE = 0.60   # 60% — models must be meaningfully confident
FORCE_TRADE_MIN_EV = 0.0           # EV must be positive to force a trade
MAX_DAILY_TRADES = 0          # 0 = unlimited (demo training mode)
COOLDOWN_AFTER_LOSS_TICKS = 1
MIN_TRADE_INTERVAL_SEC = 2    # Minimum seconds between trades

# ─── Multi-Trade Mode ───
# When True, multiple markets can trade simultaneously (each still limited to 1 trade).
# When False, only ONE trade across ALL markets at a time (old behavior).
ALLOW_MULTIPLE_TRADES = False    # Default OFF — use --multi-trade CLI flag to enable
MAX_CONCURRENT_TRADES = 5        # Max simultaneous open trades across all markets (when ALLOW_MULTIPLE_TRADES=True)
                                   # Each market still limited to 1 trade at a time regardless

# ─── Risk Management (RELAXED for demo overnight training) ───
MAX_BANKROLL_PER_TRADE = 0.05    # 5% max per trade (dynamic: min stake at low conf, up to 5% at high conf)
MAX_DAILY_LOSS = 1.0             # 100% — don't stop on losses (demo training)
MAX_CONSECUTIVE_LOSSES = 10      # Circuit breaker after 10 consecutive losses
CIRCUIT_BREAKER_COOLDOWN_SEC = 30  # Short 30s cooldown (demo training)
MAX_OPEN_POSITIONS = MAX_CONCURRENT_TRADES if ALLOW_MULTIPLE_TRADES else 1  # Dynamic based on multi-trade mode
SESSION_TIME_LIMIT_MINUTES = 0   # 0 = unlimited (demo training mode)

# ─── Kelly Criterion ───
KELLY_FRACTION = 0.25

# ─── Feature Engine ───
TICK_WINDOWS = {
    "micro":  10,
    "short":  50,
    "medium": 200,
    "long":   1000,
}

# ─── Online Learner ───
MODEL_TYPE = "ensemble"
LEARNING_RATE = 0.01
L2_REGULARIZATION = 0.01
REPLAY_BUFFER_SIZE = 5000
DRIFT_DETECTION_SENSITIVITY = 0.001

# ─── Per-Tick Live Learning ───
# When enabled, models learn from EVERY tick during live trading (not just trade outcomes).
# This makes models adapt MUCH faster to changing patterns.
# The interval controls how often: 1 = every tick, 5 = every 5th tick, etc.
# Default OFF — use --tick-learn CLI flag to enable.
TICK_LEARN_ENABLED = False
TICK_LEARN_INTERVAL = 5    # Learn every Nth tick (1=all, 5=every 5th, 10=every 10th)

# ─── Drift Retrain ───
# When drift is detected, automatically retrain the model from the replay buffer.
# This wipes the current model and rebuilds it from recent data.
DRIFT_RETRAIN_ENABLED = True   # Retrain from buffer on critical drift
DRIFT_RETRAIN_COOLDOWN = 120   # Minimum seconds between drift retrains (avoid thrashing)

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
