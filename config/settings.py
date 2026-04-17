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
MAX_OPEN_TRADES        = 999    # Max simultaneous positions (999 = effectively unlimited for testing)
MAX_DAILY_LOSS_PERCENT = 3.0    # Bot shuts down if this is hit
MAX_WEEKLY_LOSS_PERCENT= 8.0    # Weekly circuit breaker
MAGIC_NUMBER           = 200001 # Unique ID for this bot's trades

# --- SIGNAL QUALITY ---
MIN_AI_SCORE           = 80     # Minimum score (0-100) to place a trade (Increased for higher quality)
MIN_CONFLUENCE_COUNT   = 4      # Minimum factors that must agree (Increased for stricter confluence)

# --- PROFIT PROTECTION ---
PROFIT_GUARD_TRIGGER_PIPS = 5.0  # Pips profit to activate break-even move
TRAILING_STOP_PIPS        = 8.0  # Pips to trail SL behind current price (fallback if no ATR)
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
SYMBOL_COOLDOWN_MINUTES = 10     # Reduced for testing — was 60

# --- MINIMUM RISK/REWARD ---
MIN_RISK_REWARD_RATIO = 1.5

# --- CORRELATION RISK MANAGEMENT ---
MAX_CORRELATED_EXPOSURE = 2      # Max correlated pairs in same direction
MAX_SAME_CURRENCY_EXPOSURE = 3  # Max net exposure per single currency
# Example: If already long EURUSD + EURGBP (EUR exposure = +2),
# and you try to BUY EURJPY (EUR = +3), it would be blocked if
# MAX_SAME_CURRENCY_EXPOSURE = 2.

# --- RE-ENTRY LOGIC ---
ALLOW_REENTRY = True                  # Allow re-entering after TP
REENTRY_COOLDOWN_MINUTES = 15        # Min wait between exits and re-entry
REENTRY_MIN_SCORE_INCREASE = 5       # Score must be this much higher on re-entry

# --- LIMIT ORDER ENTRY ---
LIMIT_ORDER_ENABLED = True           # Use limit orders for pullback entries
LIMIT_ORDER_PRICE_OFFSET_PIPS = 3.0  # Offset from ideal price (pips)
LIMIT_ORDER_EXPIRE_MINUTES = 30      # Cancel if not filled within this time

# --- CONSECUTIVE LOSS PROTECTION ---
MAX_CONSECUTIVE_LOSSES = 8           # Raised for testing — was 4
CONSECUTIVE_LOSS_PAUSE_MINUTES = 15 # Reduced for testing — was 30

# --- SESSION WINDOWS (UTC hours) ---
# Aligned with real institutional forex session behaviors (EAT/UTC+3 reference)
# Covers all 24 hours with no gaps.
SESSIONS = {
    "SYDNEY":             {"start": 21, "end": 24},  # Price Discovery
    "TOKYO":              {"start": 0,  "end": 7},   # Accumulation
    "LONDON_OPEN":        {"start": 7,  "end": 8},   # Manipulation
    "LONDON_SESSION":     {"start": 8,  "end": 12},  # Expansion
    "NY_LONDON_OVERLAP":  {"start": 12, "end": 16},  # Distribution (peak)
    "NY_AFTERNOON":       {"start": 16, "end": 21},  # Late Distribution
}

# Session behavior descriptions (for logging/reference)
SESSION_BEHAVIORS = {
    "SYDNEY":            "Price Discovery — reacts to weekend news, thin liquidity",
    "TOKYO":             "Accumulation — tight ranges, smart money builds positions",
    "LONDON_OPEN":       "Manipulation — Judas Swing, false breakouts, stop hunts",
    "LONDON_SESSION":    "Expansion — sets daily trend, strong directional moves",
    "NY_LONDON_OVERLAP": "Distribution — highest liquidity, institutional exit",
    "NY_AFTERNOON":      "Late Distribution — liquidation, reversals or continuation",
}

# Preferred sessions — London + NY (manipulation, expansion, distribution)
PREFERRED_SESSIONS = [
    "LONDON_OPEN",
    "LONDON_SESSION",
    "NY_LONDON_OVERLAP",
    "NY_AFTERNOON",
    "TOKYO",
]
