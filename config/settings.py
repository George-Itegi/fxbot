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
    "EURUSD": 2.0, "GBPUSD": 2.5, "USDJPY": 2.5,
    "AUDUSD": 2.5, "USDCAD": 3.0, "USDCHF": 3.0, "NZDUSD": 3.0,
    # Minor Forex pairs (increased tolerance)
    "EURGBP": 3.0, "EURAUD": 4.0, "EURCAD": 4.0, "EURCHF": 4.0, "EURNZD": 5.0,
    "GBPAUD": 5.0, "GBPCAD": 5.0, "GBPCHF": 5.0, "GBPNZD": 6.0,
    "AUDCAD": 3.5, "AUDCHF": 3.5, "AUDNZD": 3.5,
    "CADCHF": 4.0, "CADJPY": 4.0,
    "CHFJPY": 4.5,
    "NZDCAD": 3.5, "NZDCHF": 3.5, "NZDJPY": 4.0,
    # JPY crosses — wider spreads normal
    "GBPJPY": 5.0, "EURJPY": 4.5, "AUDJPY": 4.5,
    # Commodities
    "XAUUSD": 40.0, "XAGUSD": 60.0,
    "WTIUSD": 5.0,  # Crude Oil
    "BRNUSD": 5.0,  # Brent Oil
    # Indices
    "US30": 60.0, "US500": 15.0, "USTEC": 20.0,
    "DE30": 30.0, "UK100": 20.0, "JP225": 40.0,
    # Default fallback
    "DEFAULT": 4.0,
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
