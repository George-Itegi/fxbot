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
# H4=trend, H1=structure, M30=context, M15=bias, M5=structure confirm, M1=entry
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4"]

# --- SCALPING PARAMETERS (NEW) ---
# These control the hybrid intraday + scalping behavior
SCALPING = {
    # Volume Surge Detection
    "VOLUME_SURGE_MULTIPLIER": 2.0,   # 2x average = institutional entry
    "VOLUME_SURGE_MIN_TICKS": 20,     # Minimum ticks for analysis

    # Order Flow Imbalance
    "OF_IMBALANCE_WINDOW": 50,         # Last N ticks for imbalance calc
    "OF_IMBALANCE_BUY_THRESHOLD": 0.3,  # Only BUY when imbalance > +0.3
    "OF_IMBALANCE_SELL_THRESHOLD": -0.3, # Only SELL when imbalance < -0.3

    # Momentum Velocity
    "VELOCITY_WINDOW_SECONDS": 60,     # Measurement window (seconds)
    "MIN_SCALP_VELOCITY": 1.0,         # Pips/min — minimum for scalping
    "CHOPPY_VELOCITY": 0.5,            # Pips/min — below this = skip

    # M1 Entry Confirmation
    "M1_VOLUME_MULTIPLIER": 1.5,       # M1 candle volume > 1.5x avg = confirmed
    "M1_STOCHRSI_OVERSOLD": 40,        # M1 StochRSI below this = oversold
    "M1_STOCHRSI_OVERBOUGHT": 60,      # M1 StochRSI above this = overbought
}

# --- RISK MANAGEMENT ---
RISK_PERCENT_PER_TRADE = 1.0    # % of balance risked per trade
MAX_OPEN_TRADES        = 5      # Max simultaneous positions
MAX_DAILY_LOSS_PERCENT = 3.0    # Bot shuts down if this is hit
MAX_WEEKLY_LOSS_PERCENT= 8.0    # Weekly circuit breaker
MAGIC_NUMBER           = 200001 # Unique ID for this bot's trades

# --- SIGNAL QUALITY ---
MIN_AI_SCORE           = 80     # Minimum score (0-100) to place a trade (Increased for higher quality)
MIN_CONFLUENCE_COUNT   = 4      # Minimum factors that must agree (Increased for stricter confluence)

# --- PROFIT PROTECTION ---
PROFIT_GUARD_TRIGGER_PIPS = 5.0  # Pips profit to activate profit protection
TRAILING_STOP_PIPS        = 10.0 # Pips to trail SL behind current price
DYNAMIC_TP_MULTIPLIER     = 2.0  # Multiplier for initial TP when trailing

# --- SPREAD LIMITS (in pips) ---
# Temporarily setting all MAX_SPREAD values to 999.0 to bypass the check for testing.
MAX_SPREAD = {
    "EURUSD": 999.0, "GBPUSD": 999.0, "USDJPY": 999.0,
    "AUDUSD": 999.0, "USDCAD": 999.0, "USDCHF": 999.0, "NZDUSD": 999.0,
    "EURGBP": 999.0, "EURAUD": 999.0, "EURCAD": 999.0, "EURCHF": 999.0, "EURNZD": 999.0,
    "GBPAUD": 999.0, "GBPCAD": 999.0, "GBPCHF": 999.0, "GBPNZD": 999.0,
    "AUDCAD": 999.0, "AUDCHF": 999.0, "AUDNZD": 999.0,
    "CADCHF": 999.0, "CADJPY": 999.0,
    "CHFJPY": 999.0,
    "NZDCAD": 999.0, "NZDCHF": 999.0, "NZDJPY": 999.0,
    "GBPJPY": 999.0, "EURJPY": 999.0, "AUDJPY": 999.0,
    "XAUUSD": 999.0, "XAGUSD": 999.0,
    "WTIUSD": 999.0, "BRNUSD": 999.0,
    "US30": 999.0, "US500": 999.0, "USTEC": 999.0,
    "DE30": 999.0, "UK100": 999.0, "JP225": 999.0,
    "DEFAULT": 999.0,
}

# --- TRADE COOLDOWN (minutes per symbol) ---
SYMBOL_COOLDOWN_MINUTES = 60

# --- MINIMUM RISK/REWARD ---
MIN_RISK_REWARD_RATIO = 1.5

# --- SESSION WINDOWS (UTC hours) ---
SESSIONS = {
    "ASIAN":          {"start": 0,  "end": 7},
    "LONDON_OPEN":    {"start": 7,  "end": 10},
    "LONDON_KILLZONE":{"start": 8,  "end": 11},
    "NY_KILLZONE":    {"start": 13, "end": 17},
    "NY_LONDON_OVERLAP": {"start": 12, "end": 17},
    "NY_CLOSE":       {"start": 17, "end": 20},
    "DEAD_ZONE":      {"start": 20, "end": 23},
}
PREFERRED_SESSIONS = ["NY_KILLZONE", "NY_LONDON_OVERLAP", "LONDON_KILLZONE"]
