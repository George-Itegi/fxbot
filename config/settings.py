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
MAX_DAILY_LOSS_PERCENT = 6.0    # Bot shuts down if this is hit
MAX_WEEKLY_LOSS_PERCENT= 8.0    # Weekly circuit breaker
MAGIC_NUMBER           = 200001 # Unique ID for this bot's trades

# --- SIGNAL QUALITY ---
MIN_AI_SCORE           = 80     # Minimum score (0-100) to place a trade (Increased for higher quality)
MIN_CONFLUENCE_COUNT   = 4      # Minimum factors that must agree (Increased for stricter confluence)

<<<<<<< HEAD
# --- SPREAD LIMITS (very loose testing mode) ---
=======
# --- SPREAD LIMITS (in pips) ---
PROFIT_GUARD_TRIGGER_PIPS = 5.0  # Pips profit to activate profit protection
TRAILING_STOP_PIPS        = 10.0 # Pips to trail SL behind current price
DYNAMIC_TP_MULTIPLIER     = 2.0  # Multiplier for initial TP when trailing

>>>>>>> 0d62e3f3f66c6487e7062bd8a3b3394013b385f5
MAX_SPREAD = {
    # Majors
    "EURUSD": 20.0, "GBPUSD": 25.0, "USDJPY": 25.0,
    "AUDUSD": 25.0, "USDCAD": 30.0, "USDCHF": 30.0, "NZDUSD": 30.0,

    # Minor Forex pairs
    "EURGBP": 30.0, "EURAUD": 40.0, "EURCAD": 40.0, "EURCHF": 40.0, "EURNZD": 50.0,
    "GBPAUD": 50.0, "GBPCAD": 50.0, "GBPCHF": 50.0, "GBPNZD": 60.0,
    "AUDCAD": 35.0, "AUDCHF": 35.0, "AUDNZD": 35.0,
    "CADCHF": 40.0, "CADJPY": 40.0,
    "CHFJPY": 45.0,
    "NZDCAD": 35.0, "NZDCHF": 35.0, "NZDJPY": 40.0,

    # JPY crosses
    "GBPJPY": 50.0, "EURJPY": 45.0, "AUDJPY": 45.0,

    # Commodities
    "XAUUSD": 400.0, "XAGUSD": 600.0,
    "WTIUSD": 50.0,
    "BRNUSD": 50.0,

    # Indices
    "US30": 600.0, "US500": 150.0, "USTEC": 200.0,
    "DE30": 300.0, "UK100": 200.0, "JP225": 400.0,

    # Default fallback
    "DEFAULT": 40.0,
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
    "NY_LONDON_OVERLAP": {"start": 12, "end": 17},
    "NY_CLOSE":       {"start": 17, "end": 20},
    "DEAD_ZONE":      {"start": 20, "end": 23},
}
<<<<<<< HEAD
PREFERRED_SESSIONS = ["LONDON_KILLZONE", "NY_KILLZONE", "NY_LONDON_OVERLAP, ASIAN"]
=======
PREFERRED_SESSIONS = ["NY_KILLZONE", "NY_LONDON_OVERLAP", "LONDON_KILLZONE"] # Prioritize NY session
>>>>>>> 0d62e3f3f66c6487e7062bd8a3b3394013b385f5
