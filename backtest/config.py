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

SYMBOLS = [
    # JPY Crosses (Priority - High Volatility Trenders)
    "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
    
    # USD Majors (Liquid, Reliable Trends)
    "EURUSD", "GBPUSD", "AUDUSD",
    
    # Other Crosses (Added for Trend Diversity)
    "EURGBP", "GBPAUD", "AUDCAD", "GBPNZD", "GBPCAD", "EURNZD",
    
    # Commodities (High Volatility)
    "XAGUSD",  # Silver
    "XAUUSD",  # Gold ⭐ NEW
]
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

AVG_SPREAD_PIPS = {
    # JPY Crosses
    "EURJPY": 0.3,  "GBPJPY": 0.7,  "AUDJPY": 0.4,
    "CADJPY": 0.5,  "CHFJPY": 0.6,  "NZDJPY": 0.8,
    
    # USD Majors
    "EURUSD": 0.1,  "GBPUSD": 0.2,  "AUDUSD": 0.2,
    
    # Other Crosses
    "EURGBP": 0.3,  "GBPAUD": 0.7,  "AUDCAD": 0.4,
    "GBPNZD": 2.2,  "GBPCAD": 1.0,  "EURNZD": 1.2,
    
    # Commodities
    "XAGUSD": 2.0,   # Silver: ~0.020 USD
    "XAUUSD": 0.4,   # Gold: ~0.40 USD (IC Markets quotes to 2 decimals; 1 pip = 0.01)
    
    # Fallback
    "DEFAULT": 0.5,
}

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
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0,
    "NZDUSD": 10.0, "USDCAD": 7.50, "USDCHF": 11.00,
    "USDJPY": 6.50,  "EURJPY": 6.50,  "GBPJPY": 6.50,
    "AUDJPY": 6.50, "CADJPY": 6.50, "NZDJPY": 6.50,
    "CHFJPY": 6.50,
    "EURGBP": 13.00, "GBPAUD": 15.00, "GBPNZD": 6.00,
    "EURCAD": 7.50,  "GBPCHF": 11.00, "AUDNZD": 6.00,
    "AUDCAD": 7.50,  "NZDCAD": 7.50,  "EURCHF": 11.00,
    "XAUUSD": 1.0,   "XAGUSD": 50.0,
    "DEFAULT": 10.0,
}

# --- Lot size reference ---
# We use 0.01 lot (micro lot) as base for all calculations
# This matches a $20k account with 0.5-1.5% risk
BASE_LOT_SIZE = 0.01
