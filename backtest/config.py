# =============================================================
# backtest/config.py
# Backtest settings — symbols, date ranges, spreads, parameters
# =============================================================

# --- Symbols to backtest ---
# Start with the core 9 (reduce later based on results)
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "GBPJPY", "EURJPY", "XAUUSD",
]

# --- Timeframes to download ---
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4"]

# --- Date range (UTC) ---
# Start: 6 months back from now
# End: yesterday (don't include today — incomplete data)
import datetime
END_DATE   = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
START_DATE = END_DATE - datetime.timedelta(days=180)  # 6 months

# --- Realistic average spreads (pips) per symbol ---
# Use typical ICMarkets raw spreads during London/NY session
AVG_SPREAD_PIPS = {
    "EURUSD": 0.3,  "GBPUSD": 0.5,  "USDJPY": 0.3,
    "AUDUSD": 0.4,  "USDCAD": 0.6,  "GBPJPY": 1.0,
    "EURJPY": 0.5,  "XAUUSD": 2.0,
    # Defaults for any symbol not listed
    "DEFAULT": 0.5,
}

# --- Slippage (pips) ---
# Add this on top of spread for realistic execution
SLIPPAGE_PIPS = 0.3

# --- Scan frequency ---
# How often to run the strategy engine (in M1 bars)
SCAN_EVERY_N_BARS = 15  # Every 15 M1 bars = every M15 candle close

# --- Starting balance for P&L calculations ---
STARTING_BALANCE = 20000.0

# --- Risk per trade (matches live settings) ---
RISK_PERCENT_PER_TRADE = 1.0

# --- Max open positions (matches live settings) ---
MAX_OPEN_TRADES = 5

# --- Data cache directory ---
CACHE_DIR = "backtest/.cache"

# --- Strategies to test ---
# Empty = test all active strategies from registry
# Specific = test only these — keeping the 5 active strategies
STRATEGIES_FILTER = []  # [] = use registry's ACTIVE status automatically
