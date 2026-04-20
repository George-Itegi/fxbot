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
MAX_OPEN_TRADES        = 5      # Max simultaneous positions (was 999 — 14 trades killed your account)
MAX_DAILY_LOSS_PERCENT = 3.0    # Bot shuts down if this is hit
MAX_WEEKLY_LOSS_PERCENT= 8.0    # Weekly circuit breaker
MAGIC_NUMBER           = 200001 # Unique ID for this bot's trades

# --- SIGNAL QUALITY (v4.3 STRICT MODE) ---
MIN_AI_SCORE           = 85     # Only trade on very high quality signals
MIN_CONFLUENCE_COUNT   = 6      # At least 6 factors must agree (was 4)
MIN_STRATEGY_SCORE     = 70     # Minimum strategy internal score to trade (was 55-65)

# --- PROFIT PROTECTION (v4.3) ---
PROFIT_GUARD_TRIGGER_PIPS = 8.0  # More profit before BE — only lock in real winners
TRAILING_STOP_PIPS        = 10.0 # Wider trail distance (was 8.0)
DYNAMIC_TP_MULTIPLIER     = 2.5  # Let winners run further

# --- SPREAD LIMITS (in pips) ---
# Tight spreads = only enter when liquidity is good.
# Major pairs: very tight (these are the most liquid).
# Cross pairs: wider (naturally less liquid, but still must be reasonable).
# Commodities/Indices: wider still but still capped.
MAX_SPREAD = {
    # Major USD pairs — tightest spreads, highest liquidity
    "EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5,
    "AUDUSD": 2.0, "USDCAD": 2.5, "USDCHF": 2.0, "NZDUSD": 2.5,
    # EUR crosses — moderate spreads
    "EURGBP": 3.0, "EURAUD": 3.5, "EURCAD": 4.0, "EURCHF": 3.5, "EURNZD": 5.0,
    # GBP crosses
    "GBPAUD": 4.0, "GBPCAD": 4.5, "GBPCHF": 4.0, "GBPNZD": 6.0,
    # AUD crosses
    "AUDCAD": 4.0, "AUDCHF": 4.0, "AUDNZD": 5.0,
    # Minor crosses
    "CADCHF": 4.0, "CADJPY": 3.5,
    "CHFJPY": 4.0,
    "NZDCAD": 5.0, "NZDCHF": 5.0, "NZDJPY": 4.0,
    # JPY crosses — moderate
    "GBPJPY": 4.0, "EURJPY": 3.0, "AUDJPY": 3.5,
    # Commodities — wider spreads
    "XAUUSD": 4.0, "XAGUSD": 5.0,
    "WTIUSD": 5.0, "BRNUSD": 5.0,
    # Indices — point-based spreads
    "US30": 5.0, "US500": 3.0, "USTEC": 4.0,
    "DE30": 5.0, "UK100": 5.0, "JP225": 5.0,
    "DEFAULT": 4.0,
}

# --- TRADE COOLDOWN (minutes per symbol) ---
SYMBOL_COOLDOWN_MINUTES = 30     # Don't spam the same symbol (was 10)

# --- MINIMUM RISK/REWARD (v4.3 STRICT) ---
MIN_RISK_REWARD_RATIO = 2.0  # Must have at least 2:1 reward (was 1.5)

# --- CORRELATION RISK MANAGEMENT ---
MAX_CORRELATED_EXPOSURE = 2      # Max correlated pairs in same direction (was 2, now enforced)
MAX_SAME_CURRENCY_EXPOSURE = 2  # Max net exposure per single currency (was 3 — reduced for safety)
# Example: If already long EURUSD + EURGBP (EUR exposure = +2),
# and you try to BUY EURJPY (EUR = +3), it would be blocked.
# This prevents the 6-EUR-pair disaster that lost -$600+ in one go.

# --- RE-ENTRY LOGIC (v4.3 STRICT) ---
ALLOW_REENTRY = True                  # Allow re-entering after TP
REENTRY_COOLDOWN_MINUTES = 30        # Longer cooldown between re-entries (was 15)
REENTRY_MIN_SCORE_INCREASE = 10      # Must be much stronger signal (was 5)

# --- LIMIT ORDER ENTRY ---
LIMIT_ORDER_ENABLED = True           # Use limit orders for pullback entries
LIMIT_ORDER_PRICE_OFFSET_PIPS = 3.0  # Offset from ideal price (pips)
LIMIT_ORDER_EXPIRE_MINUTES = 30      # Cancel if not filled within this time

# --- CONSECUTIVE LOSS PROTECTION ---
MAX_CONSECUTIVE_LOSSES = 5           # Pause after 5 consecutive losses (was 8)
CONSECUTIVE_LOSS_PAUSE_MINUTES = 30 # Full 30-min cooldown after losing streak (was 15)

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
]
