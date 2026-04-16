# =============================================================
# APEX TRADER — MASTER CONFIGURATION
# All system-wide settings live here. Never hardcode values.
# =============================================================

# --- WATCHLIST ---
WATCHLIST = [
    # Major Forex pairs
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD",
    "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD",
    "CADCHF", "CADJPY",
    "CHFJPY",
    "NZDCAD", "NZDCHF", "NZDJPY",
    # Cross pairs
    "GBPJPY", "EURJPY", "AUDJPY",
    # Commodities
    "XAUUSD",  # Gold
    "XAGUSD",  # Silver
    "WTIUSD",  # Crude Oil
    "BRNUSD",  # Brent Oil
    # Indices
    "US30",    # Dow Jones
    "US500",   # S&P 500
    "USTEC",   # Nasdaq
    "DE30",    # DAX
    "UK100",   # FTSE 100
    "JP225",   # Nikkei 225
]

# --- TIMEFRAMES USED BY THE SYSTEM ---
# H4=trend, H1=structure, M30=context, M15=entry, M5=timing
TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]

# --- RISK MANAGEMENT ---
RISK_PERCENT_PER_TRADE = 1.0    # % of balance risked per trade
MAX_OPEN_TRADES        = 5      # Max simultaneous positions
MAX_DAILY_LOSS_PERCENT = 3.0    # Bot shuts down if this is hit
MAX_WEEKLY_LOSS_PERCENT= 8.0    # Weekly circuit breaker
MAGIC_NUMBER           = 200001 # Unique ID for this bot's trades

# --- SIGNAL QUALITY ---
MIN_AI_SCORE           = 72     # Minimum score (0-100) to place a trade
MIN_CONFLUENCE_COUNT   = 3      # Minimum factors that must agree

# --- SPREAD LIMITS (in pips) ---
MAX_SPREAD = {
    # Majors
    "EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 2.0,
    "AUDUSD": 2.0, "USDCAD": 2.5, "USDCHF": 2.5, "NZDUSD": 2.5,
    # JPY crosses — wider spreads normal
    "GBPJPY": 4.0, "EURJPY": 3.5, "AUDJPY": 3.5,
    # Commodities
    "XAUUSD": 30.0, "XAGUSD": 50.0,
    # Indices
    "US30": 50.0, "US500": 10.0, "USTEC": 15.0,
    # Default fallback
    "DEFAULT": 3.0,
}

# --- TRADE COOLDOWN (minutes per symbol) ---
# Prevents the same symbol from being traded repeatedly
SYMBOL_COOLDOWN_MINUTES = 60   # Min 60 mins between trades on same symbol

# --- MINIMUM RISK/REWARD ---
MIN_RISK_REWARD_RATIO = 1.5    # TP must be at least 1.5x the SL distance

# --- SESSION WINDOWS (UTC hours) ---
SESSIONS = {
    "ASIAN":          {"start": 0,  "end": 7},
    "LONDON_OPEN":    {"start": 7,  "end": 10},
    "LONDON_KILLZONE":{"start": 8,  "end": 11},
    "NY_KILLZONE":    {"start": 13, "end": 16},
    "NY_LONDON_OVERLAP": {"start": 12, "end": 16},
    "DEAD_ZONE":      {"start": 20, "end": 23},
}
PREFERRED_SESSIONS = ["LONDON_KILLZONE", "NY_KILLZONE", "NY_LONDON_OVERLAP"]
