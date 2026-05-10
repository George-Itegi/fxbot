"""
Deriv Over/Under Bot — Configuration
=====================================
All settings, thresholds, and parameters in one place.
Demo account is the default — switch to real ONLY after full paper trading.
"""

import os
from pathlib import Path

# ─── Project Paths ───
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "data" / "deriv_bot.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─── Deriv API ───
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")  # Default demo app_id
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Use demo token from env, or None for anonymous demo
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", None)

# ─── Trading Instruments ───
SYMBOLS = {
    "R_10":  {"name": "Volatility 10 Index",  "digits": 2},
    "R_25":  {"name": "Volatility 25 Index",  "digits": 2},
    "R_50":  {"name": "Volatility 50 Index",  "digits": 2},
    "R_75":  {"name": "Volatility 75 Index",  "digits": 2},
    "R_100": {"name": "Volatility 100 Index", "digits": 2},
}

DEFAULT_SYMBOL = "R_100"

# ─── Contract Settings ───
CONTRACT_TYPE_OVER   = "DIGITOVER"
CONTRACT_TYPE_UNDER  = "DIGITUNDER"

# Over/Under barriers — configurable
OVER_BARRIER = 4    # Last digit > 4 (i.e., digits 5,6,7,8,9)
UNDER_BARRIER = 5   # Last digit < 5 (i.e., digits 0,1,2,3,4)

CONTRACT_DURATION = 5       # Number of ticks
CONTRACT_DURATION_UNIT = "t"  # 't' = ticks

# ─── Stake & Money ───
INITIAL_BANKROLL = 100.0      # Starting balance (USD)
MIN_STAKE = 0.35              # Deriv minimum
DEFAULT_STAKE = 0.35          # Start with minimum (demo)
MAX_STAKE = 5.0               # Cap for scaling

# ─── Signal Thresholds ───
MIN_CONFIDENCE = 0.56         # Model must be >56% confident
MIN_EDGE_THRESHOLD = 0.05     # Need 5%+ expected edge to trade
MAX_DAILY_TRADES = 50         # Don't overtrade
COOLDOWN_AFTER_LOSS_TICKS = 3 # Wait N ticks after a loss
MIN_TRADE_INTERVAL_SEC = 5    # Minimum seconds between trades

# ─── Risk Management (HARD LIMITS) ───
MAX_BANKROLL_PER_TRADE = 0.02    # 2% max per trade
MAX_DAILY_LOSS = 0.10            # Stop after 10% daily loss
MAX_CONSECUTIVE_LOSSES = 5       # Circuit breaker
MAX_OPEN_POSITIONS = 1           # One position at a time
SESSION_TIME_LIMIT_MINUTES = 480 # 8-hour session cap

# ─── Kelly Criterion ───
KELLY_FRACTION = 0.25  # Quarter-Kelly (conservative)
# Full Kelly = optimal but brutal drawdowns
# Quarter-Kelly = ~75% of full Kelly growth with ~25% of the variance

# ─── Feature Engine ───
TICK_WINDOWS = {
    "micro":  10,    # Immediate microstructure
    "short":  50,    # Recent momentum
    "medium": 200,   # Regime context
    "long":   1000,  # Session baseline
}

# ─── Online Learner ───
LEARNING_RATE = 0.01
L2_REGULARIZATION = 0.01
REPLAY_BUFFER_SIZE = 5000
DRIFT_DETECTION_SENSITIVITY = 0.001  # ADWIN delta (lower = more sensitive)

# ─── Model Persistence ───
MODEL_SNAPSHOT_INTERVAL = 100  # Save model every N trades
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─── Logging ───
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

# ─── Dashboard ───
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_REFRESH_MS = 2000

# ─── Mode ───
# "paper" = log signals but don't place real orders
# "live"  = place real orders (ONLY after full validation)
TRADING_MODE = os.getenv("TRADING_MODE", "paper")
