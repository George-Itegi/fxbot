# =============================================================
# APEX TRADER — MASTER CONFIGURATION
# All system-wide settings live here. Never hardcode values.
# =============================================================

# --- PAIR WHITELIST / BLACKLIST (v2.0 — data-driven) ---
# Single source of truth for which pairs and sessions are allowed.
# Backtested 180 days on 13 pairs. Cuts based on negative P&L, low Sharpe,
# insufficient sample size, or structural dependency issues.
#
# CUT PAIRS (do NOT trade — backtest proven negative):
#   XAUUSD  -$246  Sharpe -0.92  21.4% WR  (gold microstructure mismatch)
#   AUDCAD +$1886 Sharpe  0.20  30.0% WR  (one-strategy dependency, thin edge)
#   EURUSD  -$4538 Sharpe -3.82  20.5% WR  (EUR base toxic for momentum)
#   EURAUD  -$3304 Sharpe -2.89  22.0% WR  (EUR base toxic for momentum)
#   EURNZD  -$4162 Sharpe -2.89  23.7% WR  (EUR base toxic for momentum)
#   EURCAD  -$3041 Sharpe -2.11  23.0% WR  (EUR base toxic for momentum)
#
# To ADD a new pair: backtest 180 days, must beat GBPUSD minimum bar
#   (Sharpe > 1.79, PF > 1.55, positive P&L, 30+ trades).
# To CUT an existing pair: add to PAIR_BLACKLIST with reason.

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

PAIR_BLACKLIST = {
    # Pair: (date_cut, reason, backtest_pnl, backtest_sharpe)
    "XAUUSD": ("2025-05", "Gold microstructure mismatch", -246, -0.92),
    "GBPAUD": ("2025-05", "AUD quote toxic for momentum", -524, -0.50),
    "GBPCAD": ("2025-05", "Insufficient edge in 120d", -3026, -2.08),
    "GBPNZD": ("2025-05", "NZD quote toxic in 120d", -3026, -2.08),
    "EURGBP": ("2025-05", "Model rejects all signals, 0 trades", -691, -42.6),
}

# Alias for backward compat — WATCHLIST = WHITELIST
WATCHLIST = PAIR_WHITELIST

# --- SESSION WHITELIST / BLACKLIST (v2.0 — data-driven) ---
# Backtested across all 13 pairs (736 trades). Sessions below lose money.
#
# CUT SESSIONS (do NOT trade — backtest proven negative):
#   NY_AFTERNOON:  36 trades, 11.1% WR, -$2,607 (fading volume, erratic)
#   SYDNEY:       ~67 trades, ~20% WR, -$3,361 (thin liquidity, false signals)
#
# ALLOWED SESSIONS (3 of 6):
#   LONDON_OPEN:       29 trades, 37.9% WR, +$2,806
#   LONDON_SESSION:   275 trades, 40.0% WR, +$17,995
#   NY_LONDON_OVERLAP: 306 trades, 43.1% WR, +$27,632

SESSION_WHITELIST = [
    "LONDON_OPEN",
    "LONDON_SESSION",
    "NY_LONDON_OVERLAP",
]

SESSION_BLACKLIST = {
    # Session: (date_cut, reason, trades, win_rate, pnl)
    "NY_AFTERNOON": ("2025-05", "36 trades, 11.1% WR, -$2,607", 36, 0.111, -2607),
    "SYDNEY":      ("2025-05", "~67 trades, ~20% WR, -$3,361", 67, 0.20, -3361),
}

# Tokyo is marginal (+$1,417 on 23 trades, 34.8% WR) —
# kept for monitoring but NOT in active whitelist yet.
# Add to SESSION_WHITELIST if live data confirms edge.

# --- TIMEFRAMES USED BY THE SYSTEM ---
# H4=trend, H1=structure, M30=context, M15=bias, M5=structure confirm, M1=entry
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4"]

# --- ORDER FLOW & VOLUME PARAMETERS ---
# Used by institutional confirmation gate (NOT scalping-specific).
# These determine what counts as "institutions are active".
ORDER_FLOW = {
    # Volume Surge Detection
    "VOLUME_SURGE_MULTIPLIER": 2.0,   # 2x average = institutional entry
    "VOLUME_SURGE_MIN_TICKS": 20,     # Minimum ticks for analysis

    # Order Flow Imbalance
    "OF_IMBALANCE_WINDOW": 50,         # Last N ticks for imbalance calc
    "OF_IMBALANCE_BUY_THRESHOLD": 0.3,  # Only BUY when imbalance > +0.3
    "OF_IMBALANCE_SELL_THRESHOLD": -0.3, # Only SELL when imbalance < -0.3
}

# --- RISK MANAGEMENT ---
RISK_PERCENT_PER_TRADE = 1.0    # % of balance risked per trade
MAX_OPEN_TRADES        = 5      # Max simultaneous positions (was 999 — 14 trades killed your account)
MAX_DAILY_LOSS_PERCENT = 8.0    # Bot shuts down if this is hit
MAX_WEEKLY_LOSS_PERCENT= 8.0    # Weekly circuit breaker
MAGIC_NUMBER           = 200001 # Unique ID for this bot's trades

# --- SCANNING MODE (v4.3) ---
SCAN_MODE         = "sequential"   # "sequential" (one pair at a time) or "parallel"
SCAN_PAIR_INTERVAL= 3.0            # Seconds between sequential pair scans (parallel ignores this)
SCAN_VERBOSE      = True           # Detailed per-strategy logs on console (like backtesting)

# --- SIGNAL QUALITY (v4.3 STRICT MODE) ---
MIN_AI_SCORE           = 85     # Only trade on very high quality signals
MIN_CONFLUENCE_COUNT   = 6      # At least 6 factors must agree (was 4)
MIN_STRATEGY_SCORE     = 70     # Minimum strategy internal score to trade (was 55-65)

# --- PROFIT PROTECTION (v4.3) ---
PROFIT_GUARD_TRIGGER_PIPS = 8.0  # More profit before BE — only lock in real winners
TRAILING_STOP_PIPS        = 10.0 # Wider trail distance (was 8.0)
DYNAMIC_TP_MULTIPLIER     = 2.5  # Let winners run further

# --- PARTIAL TP (v4.4) ---
# Close 50% at 1R profit, move SL to breakeven, trail remainder with ATR.
# This transforms a 50% win rate with 2:1 R:R into a much more consistent
# equity curve. Even losing trades that hit partial TP before reversing
# cost you nothing (the 50% closed at profit covers the remaining SL).
PARTIAL_TP_ENABLED          = True
PARTIAL_TP_RATIO            = 0.50   # Close 50% of position at 1R
PARTIAL_TP_AT_R_MULTIPLE    = 1.0    # Trigger at 1x SL distance (1R profit)

# --- DYNAMIC POSITION SIZING (v4.4) ---
# Scale risk by conviction: AI score + number of agreeing strategy groups.
# Low conviction signals risk less, high conviction signals risk more.
# After consecutive losses, all sizes are halved until a win resets.
DYNAMIC_SIZING_ENABLED      = True
SIZING_LOW_RISK_PCT         = 0.50   # Low conviction: 0.5% risk
SIZING_MED_RISK_PCT         = 1.00   # Medium conviction: 1.0% risk (default)
SIZING_HIGH_RISK_PCT        = 1.50   # High conviction: 1.5% risk
SIZING_LOW_SCORE_MIN        = 65     # Low conviction score range start
SIZING_LOW_SCORE_MAX        = 75     # Low conviction score range end
SIZING_HIGH_SCORE_MIN       = 85     # High conviction: score >= this
SIZING_HIGH_MIN_GROUPS      = 3      # High conviction: 3+ groups must agree
SIZING_CONSEC_LOSS_HALVE    = 3      # Halve all sizes after N consecutive losses

# --- SPREAD LIMITS (in pips) ---
# Only whitelisted pairs. Cut pairs removed.
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

# Preferred sessions — ONLY whitelisted sessions (v2.0)
PREFERRED_SESSIONS = SESSION_WHITELIST
