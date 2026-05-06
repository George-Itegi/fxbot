# =============================================================
# backtest/config.py  v2.0
# Backtest settings — symbols, date ranges, spreads, parameters
# Upgraded to match live system (partial TP, dynamic sizing, etc.)
# =============================================================

import datetime

# --- Symbols to backtest ---
# Matches the full WATCHLIST in config/settings.py
# SYMBOLS = [
#     # Major Forex pairs (7)
#     "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
#     # JPY crosses (5)
#     "GBPJPY", "EURJPY", "AUDJPY", "CADJPY", "NZDJPY",
#     # Popular crosses (3)
#     "EURGBP", "GBPAUD", "GBPNZD",
#     # Additional crosses (5) — added for model diversity
#     "EURCAD", "GBPCHF", "AUDNZD", "AUDCAD", "NZDCAD",
#     # CHF crosses (2)
#     "CHFJPY", "EURCHF",
#     # Commodities (2)
#     "XAUUSD",  # Gold
#     "XAGUSD",  # Silver
# ]
PAIR_WHITELIST = [
    # JPY Crosses (Core Edge)
    "EURJPY",  # +$11,506 | JPY Cross
    "GBPJPY",  # +$10,346 | JPY Cross
    "CHFJPY",  # +$6,083  | JPY Cross
    "CADJPY",  # +$3,413  | JPY Cross
    "AUDJPY",  # +$2,739  | JPY Cross

    # Commodities
    "XAGUSD",  # +$9,290  | Metal

    # USD Pairs
    "AUDUSD",  # +$2,253  | USD
    "EURUSD",  # +$1,592  | USD
    "GBPUSD",  # +$1,419  | USD

    # Commodity Cross
    "AUDCAD",  # +$2,277  | Commodity
]

# SYMBOLS = [
#     # JPY Crosses (Core Edge)
#     "CHFJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",

#     # GBP Base (Secondary Edge)
#     "GBPUSD", "GBPNZD",

#     # Commodities
#     "XAGUSD",  # Silver
# ]
# v2.0: 8-pair optimized portfolio (removed GBPAUD/GBPCAD/XAUUSD/EURGBP/AUDCAD)
SYMBOLS = PAIR_WHITELIST

# Cut pairs: negative P&L, low Sharpe, or structural dependency (see config/settings.py)
# Cut sessions: NY_AFTERNOON and SYDNEY — enforced in data_loader.py _tag_session
# --- Timeframes to download ---
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4"]

# --- Date range (UTC) ---
END_DATE   = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
START_DATE = END_DATE - datetime.timedelta(days=180)  # 6 months

# --- Realistic average spreads (pips) per symbol ---
# Use typical ICMarkets raw spreads during London/NY session
# AVG_SPREAD_PIPS = {
#     # Major pairs

#     "EURUSD": 0.3,  "GBPUSD": 0.5,  "USDJPY": 0.3,
#     "AUDUSD": 0.4,  "USDCAD": 0.6,  "NZDUSD": 0.4,  "USDCHF": 0.5,
#     # JPY crosses
#     "GBPJPY": 1.0,  "EURJPY": 0.5,  "AUDJPY": 0.7,
#     "CADJPY": 0.7,  "NZDJPY": 1.0,  "CHFJPY": 0.8,
#     # Popular crosses
#     "EURGBP": 0.6,  "GBPAUD": 1.2,  "GBPNZD": 2.5,
#     # Other crosses
#     "EURCAD": 0.8,  "GBPCHF": 0.9,  "AUDNZD": 1.2,
#     "AUDCAD": 0.8,  "NZDCAD": 0.8,  "EURCHF": 0.7,
#     # Commodities
#     "XAUUSD": 2.0,  "XAGUSD": 3.0,
#     # Default
#     "DEFAULT": 0.5,
# }

# AVG_SPREAD_PIPS = {
#     # JPY Crosses (Core Edge)
#     "CHFJPY": 0.6,  "EURJPY": 0.3, "GBPJPY": 0.7,
#     "AUDJPY": 0.4, "CADJPY": 0.5,

#     # GBP Base (Secondary Edge)
#     "GBPUSD": 0.2, "GBPNZD": 2.2,

#     # Commodities
#     "XAGUSD": 2.0,   # Silver: ~0.020 USD

#     # Fallback
#     "DEFAULT": 0.5,
# }
MAX_SPREAD = {
    # JPY Crosses (Core Edge)
    "EURJPY": 3.0, "GBPJPY": 4.0, "CHFJPY": 3.5,
    "AUDJPY": 3.5, "CADJPY": 4.0,
    # Commodities
    "XAGUSD": 5.0,
    # USD Pairs
    "AUDUSD": 2.0, "EURUSD": 2.0, "GBPUSD": 2.0,
    # Commodity Cross
    "AUDCAD": 4.0,
    # Default fallback
    "DEFAULT": 4.0,
}

# Alias: AVG_SPREAD_PIPS maps to MAX_SPREAD for backward compat with engine.py
AVG_SPREAD_PIPS = MAX_SPREAD

# --- Slippage (pips) ---
SLIPPAGE_PIPS = 0.3

# --- Scan frequency ---
# How often to run the strategy engine (in M1 bars)
SCAN_EVERY_N_BARS = 15  # Every 15 M1 bars = every M15 candle close

# --- Starting balance for P&L calculations ---
STARTING_BALANCE = 20000.0

# --- Risk settings (match live) ---
BASE_RISK_PERCENT = 1.0   # Default 1% risk per trade

# --- Partial TP (matches live v4.4) ---
PARTIAL_TP_ENABLED = True
PARTIAL_TP_RATIO = 0.50       # Close 50% at 1R (default)
PARTIAL_TP_AT_R_MULTIPLE = 1.0  # Trigger at 1R profit

# --- Per-Strategy TP2 Primary Target ---
# When True, engine uses tp2_pips as the main TP target instead of tp1_pips.
# The existing partial TP mechanism triggers at 1R (=TP1 distance for
# strategies with SL == TP1 distance like LIQUIDITY_SWEEP v2.1).
STRATEGY_TP2_PRIMARY = {
    'LIQUIDITY_SWEEP_ENTRY': True,  # TP2=3.0xATR as main target
}

# --- Per-Strategy Partial TP Ratio Override ---
# Allows strategies to use different partial close sizes.
# LIQUIDITY_SWEEP: 33% at TP1, trail remaining 67% to TP2.
STRATEGY_PARTIAL_TP_RATIO = {
    'LIQUIDITY_SWEEP_ENTRY': 0.33,  # Close 33% at TP1, trail rest
}

# --- ATR Trailing Stop (matches live) ---
ATR_TRAIL_ENABLED = True
ATR_TRAIL_MULTIPLIER = 1.0   # Trail distance = ATR × multiplier

# --- Dynamic TP Extension (matches live) ---
DYNAMIC_TP_EXTENSION_ENABLED = True
DYNAMIC_TP_TRIGGER_PCT = 0.60   # Trigger when 60% to target
DYNAMIC_TP_MULTIPLIER_ATR = 1.5  # Extend by max(2×ATR, 1.5×trail)

# --- Dynamic Position Sizing (matches live) ---
DYNAMIC_SIZING_ENABLED = True
# Conviction tiers
CONVICTION_LOW_SCORE_MAX = 75      # 0.5% risk
CONVICTION_MED_SCORE_MAX = 85      # 1.0% risk (default)
CONVICTION_HIGH_MIN_GROUPS = 3     # 1.5% risk (high score + 3+ groups)
# Consecutive loss halving
CONSECUTIVE_LOSS_HALVE_THRESHOLD = 3  # Halve after 3 consecutive losses

# --- Max open positions (matches live) ---
MAX_OPEN_TRADES = 5
MAX_PER_SYMBOL = 1

# --- Confluence minimum (matches live) ---
MIN_CONFLUENCE = 6

# --- Strategy min R:R (matches live) ---
MIN_RR_RATIO = 2.0

# --- Master score gate (matches live) ---
MASTER_MIN_SCORE = 45

# --- Data cache directory ---
CACHE_DIR = "backtest/.cache"

# --- Strategies to test ---
# Empty = test all active strategies from registry
STRATEGIES_FILTER = []

# --- Pip values per lot (USD) ---
# Standard lot pip value for USD-account
# For pairs ending in USD (EURUSD, GBPUSD, etc.) = $10/pip/lot
# For JPY pairs = varies (~$6.50-$9.50 depending on rate)
# Simplified — we compute dynamically based on current rate
PIP_VALUE_PER_LOT = {
    # JPY Crosses
    "EURJPY": 6.50, "GBPJPY": 6.50, "CHFJPY": 6.50,
    "AUDJPY": 6.50, "CADJPY": 6.50,
    # Commodities
    "XAGUSD": 50.0,
    # USD Pairs
    "AUDUSD": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0,
    # Commodity Cross
    "AUDCAD": 6.50,
    # Fallback
    "DEFAULT": 10.0,
}

# --- Lot size reference ---
# We use 0.01 lot (micro lot) as base for all calculations
# This matches a $20k account with 0.5-1.5% risk
BASE_LOT_SIZE = 0.01
