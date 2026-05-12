"""
Deriv Over/Under Bot — Configuration v11
==========================================
Over 4 / Under 5 ONLY — Simple, Stable, Profitable.

v10 FAILED because moderate barriers (Over 5, Over 6, Under 4, Under 6)
still have too much variance. Over 5 has 40% natural prob, Over 6 has 30%.
These are still lottery-ish — you need long winning streaks to profit.

v11 Strategy: "Over 4 / Under 5 Only"
1. ONLY trade Over 4 (digit > 4) and Under 5 (digit < 5)
   - Both have 50% natural probability — FAIR coin flip contracts
   - ~95% payout rate — breakeven at 52.6% win rate
   - Lowest variance of all digit contracts
   - Even small frequency edges are profitable
2. Adjusted thresholds for 50/50 contracts:
   - z-score 2.0 minimum (we only test 2 barriers, not 17 — less multiple testing)
   - 3% minimum EV (thin but positive — 95% payout means small edges work)
   - 52% minimum confidence (must be above the 50% natural probability)
   - 3-window agreement still required
3. LESS Bayesian shrinkage (prior=50 instead of 100)
   - For 50/50 contracts, the observed frequency is more reliable
   - Shrinkage toward 50% is less needed since 50% IS the natural probability
4. FIXED 5-tick duration
5. ML disagreement BLOCKS trades
6. Market persistence — stay on one market while setup is good
"""

import os
from pathlib import Path

# ─── Load .env file (look in project root, then parent fxbot dir) ───
for env_path in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        break

# ─── Project Paths ───
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# ─── MySQL Database (XAMPP) ───
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "apex_trader")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─── Deriv API ───
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", None)

# ─── Trading Instruments ───
SYMBOLS = {
    "R_10":  {"name": "Volatility 10 Index",         "decimal_places": 3, "category": "volatility"},
    "R_25":  {"name": "Volatility 25 Index",         "decimal_places": 3, "category": "volatility"},
    "R_50":  {"name": "Volatility 50 Index",         "decimal_places": 4, "category": "volatility"},
    "R_75":  {"name": "Volatility 75 Index",         "decimal_places": 4, "category": "volatility"},
    "R_100": {"name": "Volatility 100 Index",        "decimal_places": 2, "category": "volatility"},
    "1HZ10V":  {"name": "Volatility 10 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ15V":  {"name": "Volatility 15 (1s) Index",  "decimal_places": 3, "category": "1s"},
    "1HZ25V":  {"name": "Volatility 25 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ30V":  {"name": "Volatility 30 (1s) Index",  "decimal_places": 3, "category": "1s"},
    "1HZ50V":  {"name": "Volatility 50 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ75V":  {"name": "Volatility 75 (1s) Index",  "decimal_places": 2, "category": "1s"},
    "1HZ90V":  {"name": "Volatility 90 (1s) Index",  "decimal_places": 3, "category": "1s"},
    "1HZ100V": {"name": "Volatility 100 (1s) Index", "decimal_places": 2, "category": "1s"},
    "JD10":  {"name": "Jump 10 Index",   "decimal_places": 2, "category": "jump"},
    "JD25":  {"name": "Jump 25 Index",   "decimal_places": 2, "category": "jump"},
    "JD50":  {"name": "Jump 50 Index",   "decimal_places": 2, "category": "jump"},
    "JD75":  {"name": "Jump 75 Index",   "decimal_places": 2, "category": "jump"},
    "JD100": {"name": "Jump 100 Index",  "decimal_places": 2, "category": "jump"},
    "BOOM50":    {"name": "Boom 50 Index",      "decimal_places": 3, "category": "crash_boom"},
    "BOOM150N":  {"name": "Boom 150 Index",     "decimal_places": 5, "category": "crash_boom"},
    "BOOM300N":  {"name": "Boom 300 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM500":   {"name": "Boom 500 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM600":   {"name": "Boom 600 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM900":   {"name": "Boom 900 Index",     "decimal_places": 3, "category": "crash_boom"},
    "BOOM1000":  {"name": "Boom 1000 Index",    "decimal_places": 3, "category": "crash_boom"},
    "CRASH50":   {"name": "Crash 50 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH150N": {"name": "Crash 150 Index",    "decimal_places": 5, "category": "crash_boom"},
    "CRASH300N": {"name": "Crash 300 Index",    "decimal_places": 3, "category": "crash_boom"},
    "CRASH500":  {"name": "Crash 500 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH600":  {"name": "Crash 600 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH900":  {"name": "Crash 900 Index",     "decimal_places": 3, "category": "crash_boom"},
    "CRASH1000": {"name": "Crash 1000 Index",    "decimal_places": 3, "category": "crash_boom"},
    "RDBULL": {"name": "Bull Market Index",  "decimal_places": 4, "category": "daily_reset"},
    "RDBEAR": {"name": "Bear Market Index",  "decimal_places": 4, "category": "daily_reset"},
    "RB100": {"name": "Range Break 100 Index", "decimal_places": 1, "category": "range_break"},
    "RB200": {"name": "Range Break 200 Index", "decimal_places": 1, "category": "range_break"},
    "stpRNG":  {"name": "Step Index 100", "decimal_places": 1, "category": "step"},
    "stpRNG2": {"name": "Step Index 200", "decimal_places": 1, "category": "step"},
    "stpRNG3": {"name": "Step Index 300", "decimal_places": 1, "category": "step"},
    "stpRNG4": {"name": "Step Index 400", "decimal_places": 1, "category": "step"},
    "stpRNG5": {"name": "Step Index 500", "decimal_places": 1, "category": "step"},
    "WLDAUD": {"name": "AUD Basket",  "decimal_places": 3, "category": "basket"},
    "WLDEUR": {"name": "EUR Basket",  "decimal_places": 3, "category": "basket"},
    "WLDGBP": {"name": "GBP Basket",  "decimal_places": 3, "category": "basket"},
    "WLDUSD": {"name": "USD Basket",  "decimal_places": 3, "category": "basket"},
    "WLDXAU": {"name": "Gold Basket",  "decimal_places": 3, "category": "basket"},
}

# ─── Symbol Categories ───
DIGIT_CONTRACT_CATEGORIES = {"volatility", "1s", "daily_reset"}

RECOMMENDED_SYMBOLS = {
    "fastest_data":  "1HZ100V",
    "most_popular":  "R_100",
    "high_volatility": "1HZ50V",
}

def get_symbol_decimals(symbol: str) -> int:
    return SYMBOLS.get(symbol, {}).get("decimal_places", 2)

def get_symbol_category(symbol: str) -> str:
    return SYMBOLS.get(symbol, {}).get("category", "unknown")

def supports_digit_contracts(symbol: str) -> bool:
    return get_symbol_category(symbol) in DIGIT_CONTRACT_CATEGORIES

# ─── Multi-Market Configuration ───
DEFAULT_MARKETS = [
    # 1-second indices (fastest tick rate)
    "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V",
    "1HZ75V", "1HZ90V", "1HZ100V",
    # Standard volatility indices
    "R_10", "R_25", "R_50", "R_75", "R_100",
]
VALID_MULTI_MARKET_SYMBOLS = [
    "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V",
    "1HZ75V", "1HZ90V", "1HZ100V",
    "R_10", "R_25", "R_50", "R_75", "R_100",
]
DEFAULT_SYMBOL = "1HZ100V"

# ─── Contract Settings ───
CONTRACT_TYPE_OVER   = "DIGITOVER"
CONTRACT_TYPE_UNDER  = "DIGITUNDER"
OVER_BARRIER = 4              # Default fallback (overridden by dynamic barrier selection)
UNDER_BARRIER = 5             # Default fallback (overridden by dynamic barrier selection)
CONTRACT_DURATION = 5          # v10: FIXED at 5 ticks
CONTRACT_DURATION_UNIT = "t"

# ─── Barrier Selection (v11 — OVER 4 / UNDER 5 ONLY) ───
# v10 tested moderate barriers (Over 3-6, Under 4-7) but even Over 5 (40% natural)
# and Over 6 (30% natural) have too much variance for consistent profits.
#
# v11: ONLY Over 4 and Under 5. These are the ONLY contracts where:
#   - Natural probability is 50% (fair coin flip)
#   - Payout is ~95% (breakeven at 52.6%)
#   - Variance is LOW — you win roughly half your trades
#   - Even a small frequency edge (3-5%) is profitable
#
# Over 4: wins if digit > 4 (digits 5,6,7,8,9) = 50% natural, ~95% payout
# Under 5: wins if digit < 5 (digits 0,1,2,3,4) = 50% natural, ~95% payout
DYNAMIC_BARRIERS = False               # v11: DISABLED — only Over 4 / Under 5
BARRIER_OVER_OPTIONS = [4]             # v11: ONLY Over 4
BARRIER_UNDER_OPTIONS = [5]            # v11: ONLY Under 5
MIN_BARRIER_PROBABILITY = 0.45         # v11: Only ~50% natural prob barriers
MAX_BARRIER_PROBABILITY = 0.55         # v11: Only ~50% natural prob barriers
PAYOUT_HOUSE_MARGIN = 0.046            # Deriv's house margin (~4.6% based on observed Over 4 payout)
MIN_EV_FOR_TRADE = 0.03                # v11: 3% minimum EV (was 8% — too strict for 95% payout contracts)

# ─── Bayesian Shrinkage (v11 — REDUCED for 50/50 contracts) ───
# For Over 4/Under 5 (50% natural prob), the observed frequency is more
# reliable because 50% is the CENTER of the distribution — less sampling noise.
# v11: Reduced prior from 100 to 50 — less aggressive shrinkage.
# With n=200 and k=50: 80% observed, 20% natural
# Example: Over 4 with 55% observed -> adjusted = (200*0.55 + 50*0.50)/250 = 0.54
#   EV = 0.54 * 1.95 - 1 = 0.053 = 5.3% edge
BAYESIAN_SHRINKAGE_PRIOR = 50           # v11: Less shrinkage for 50/50 contracts
                                        # With n=200: 80% observed, 20% natural
                                        # With n=500: 91% observed, 9% natural

# ─── Natural Probabilities for Barriers ───
# Over B wins if digit > B. Under B wins if digit < B.
# For uniform digit distribution (0-9), each digit = 10%.
# Over B natural probability = (9 - B) / 10
# Under B natural probability = (B - 0) / 10 = B / 10
BARRIER_NATURAL_PROB_OVER = {b: (9 - b) / 10.0 for b in range(10)}   # Over 0: 90%, Over 4: 50%, Over 8: 10%
BARRIER_NATURAL_PROB_UNDER = {b: b / 10.0 for b in range(1, 10)}     # Under 1: 10%, Under 5: 50%, Under 9: 90%

def estimate_payout_rate(natural_probability: float) -> float:
    """
    Estimate the payout rate for a barrier with a given natural probability.
    
    Payout rate = profit / stake when you win.
    Based on Deriv's pricing model: payout = (1 / natural_prob - 1) * (1 - house_margin)
    
    Examples:
        Over 4 (50% natural): payout = (1/0.50 - 1) * (1 - 0.046) = 1.0 * 0.954 = 0.954
        Over 5 (40% natural): payout = (1/0.40 - 1) * (1 - 0.046) = 1.5 * 0.954 = 1.431
        Over 6 (30% natural): payout = (1/0.30 - 1) * (1 - 0.046) = 2.33 * 0.954 = 2.223
    """
    if natural_probability <= 0 or natural_probability >= 1:
        return 0.0
    return (1.0 / natural_probability - 1.0) * (1.0 - PAYOUT_HOUSE_MARGIN)

# ─── Duration (v10: FIXED at 5 ticks) ───
# v9 had dynamic duration (2t-10t) which caused chaos — trades bounced
# between 2t, 3t, 5t with no clear logic. The observation phase didn't help
# because digit patterns don't have predictable "flip durations."
# v10: Fixed at 5 ticks — simple, consistent, no observation phase needed.
DYNAMIC_DURATION = False              # v10: DISABLED — fixed 5t duration
CONTRACT_DURATION = 5                 # Fixed 5-tick duration
MIN_DURATION = 5                      # v10: Always 5
MAX_DURATION = 5                      # v10: Always 5
DURATION_EXPLORATION_RATE = 0.0       # v10: No exploration

# ─── Stake & Money ───
INITIAL_BANKROLL = 100.0
MIN_STAKE = 0.35
DEFAULT_STAKE = 0.35
MAX_STAKE = 5.0

# ─── Signal Thresholds ───
# v11: Calibrated for Over 4 / Under 5 ONLY (50% natural, ~95% payout).
# Since we only test 2 barriers, the multiple testing problem is minimal.
# z=2.0 (95% confidence) is sufficient — we're not data-mining across 17 options.
MIN_CONFIDENCE = 0.52           # v11: Must be above 50% natural — need a real edge
MIN_EDGE_THRESHOLD = 0.03       # v11: Minimum 3% EV
MIN_FREQ_EDGE_ZSCORE = 2.0      # v11: 2-sigma (95% confidence) — was 3.0

# ─── Digit Frequency Direction (PRIMARY DECISION) ───
# v11: Only Over 4 / Under 5 + ALL 3 windows must agree.
MIN_DIGIT_FREQUENCY_EDGE = 0.03   # v11: Minimum 3% ABSOLUTE edge
MIN_DIGIT_FREQUENCY_EDGE_RELATIVE = 0.06  # v11: Minimum 6% RELATIVE edge (3%/50% = 6%)
DIGIT_FREQ_WINDOW_AGREEMENT = 3   # v11: ALL 3 windows must agree

# ─── Setup Quality ───
# v11: Lowered to 0.55 since Over 4/Under 5 have manageable variance.
MIN_SETUP_SCORE = 0.55           # v11: Minimum setup quality (was 0.70)

# ─── Profit Target Per Market ───
# After making this much profit on one market, STOP trading it.
# Wait for another good setup to appear.
PROFIT_TARGET_PER_MARKET = 50.0   # $50 profit target per market session

# ─── Observation Phase (v10: DISABLED — using fixed 5t duration) ───
# v9's observation phase determined duration dynamically, but this caused
# chaos (2t/3t/5t switching) and the flip-duration analysis was unreliable.
# v10: Fixed 5-tick duration. Observation phase is skipped entirely.
OBSERVATION_PERIOD_SEC = 0        # v10: No observation phase (was 25s)
MIN_OBSERVATION_TICKS = 0         # v10: Not needed

# ─── Market Session (Persistence) ───
# Stay on one market during a setup. Don't jump between markets.
# Only switch when: setup breaks, profit target reached, or no trade happening.
# v8.1: Increased idle timeout — good setups deserve patience.
MARKET_SESSION_MAX_IDLE_SEC = 300  # v11: 5 min idle before switching (was 3 min)
MARKET_STICKY_AFTER_TRADE = True   # Stay on market after trading it

# ─── ML Confirmation Model ───
# v10: ML disagreement now BLOCKS trades again.
# v9 reduced ML disagreement to just 20% confidence reduction — this was wrong.
# If the ML model disagrees with the signal, the signal is probably noise.
# On synthetic indices, we need EVERY confirmation we can get.
ML_CONFIRMATION_WEIGHT = 0.20     # Weight of ML opinion when it agrees
ML_DISAGREEMENT_BLOCKS = True     # v10: BLOCK on ML disagreement (was False — too permissive)

# ─── Confidence-Weighted Agreement ───
SIGNAL_SCORE_METHOD = "setup_weighted"        # setup quality drives the score
MIN_SIGNAL_SCORE = 0.55                       # Lowered — rule-based primary is more reliable
AGREEMENT_WEIGHT = 1.0                        # Always 1.0 with single model

# ─── Forced Trade (strong setup) ───
# A forced trade happens when the setup is VERY strong (setup score >= 0.85).
# v10: Much harder to force without ML — need 85% setup + 35% confidence + 8% EV.
FORCE_TRADE_MIN_CONFIDENCE = 0.52   # v11: Must be above 50% natural
FORCE_TRADE_MIN_EV = 0.03           # v11: Aligned with MIN_EV_FOR_TRADE
FORCE_TRADE_MIN_SETUP_SCORE = 0.75  # v11: Strong setup to force without ML

# ─── Trend Setting (v10: SOFT BIAS — not required, but helpful) ───
# On synthetic indices, price trend has NO causal relationship with digit distribution.
# However, trend alignment with digit frequency is a nice-to-have confirmation.
# v10: Trend is a SOFT BIAS — it boosts/penalizes but never blocks a strong freq signal.
TREND_REQUIRED = False               # Trend is NOT required (digit freq is primary)
TREND_SLOPE_TSTAT_THRESHOLD = 2.0    # v11: 2-sigma for trend
TREND_CONFIDENCE_BOOST = 0.02        # v11: Small boost when trend aligns
TREND_MISALIGN_PENALTY = 0.10        # v11: Moderate penalty when trend opposes
TREND_CONFIDENCE_REDUCTION = 0.05    # Legacy
TREND_SIGNAL_SCORE_REDUCTION = 0.05  # Legacy

# ─── Martingale Confidence Gate ───
# During martingale recovery, the bot MUST stay on the SAME market.
# No switching markets during recovery — the setup was good on THAT market.
# v10: Much stricter — same as regular thresholds.
MARTINGALE_MIN_CONFIDENCE = 0.50   # v11: Slightly relaxed for recovery (was 35%)
MARTINGALE_MIN_SETUP_SCORE = 0.50  # v11: Relaxed for martingale recovery (was 0.70)
MARTINGALE_SAME_MARKET = True      # MUST recover on the same market where loss occurred
MARTINGALE_MAX_STEPS = 2           # Max 2 recovery steps
MAX_DAILY_TRADES = 0          # 0 = unlimited (demo training mode)
COOLDOWN_AFTER_LOSS_TICKS = 1
MIN_TRADE_INTERVAL_SEC = 10   # v11: 10s between trades (was 15s)

# ─── Multi-Trade Mode ───
# When True, multiple markets can trade simultaneously (each still limited to 1 trade).
# When False, only ONE trade across ALL markets at a time (old behavior).
ALLOW_MULTIPLE_TRADES = False    # Default OFF — use --multi-trade CLI flag to enable
MAX_CONCURRENT_TRADES = 5        # Max simultaneous open trades across all markets

# ─── Risk Management (RELAXED for demo overnight training) ───
MAX_BANKROLL_PER_TRADE = 0.05    # 5% max per trade
MAX_DAILY_LOSS = 1.0             # 100% — don't stop on losses (demo training)
MAX_CONSECUTIVE_LOSSES = 10      # Circuit breaker after 10 consecutive losses
CIRCUIT_BREAKER_COOLDOWN_SEC = 30  # Short 30s cooldown (demo training)
MAX_OPEN_POSITIONS = MAX_CONCURRENT_TRADES if ALLOW_MULTIPLE_TRADES else 1
SESSION_TIME_LIMIT_MINUTES = 0   # 0 = unlimited (demo training mode)

# ─── Kelly Criterion ───
KELLY_FRACTION = 0.25

# ─── Feature Engine ───
TICK_WINDOWS = {
    "micro":       10,
    "short":       50,
    "medium":      200,
    "trend_long":  500,    # Added for 500-tick trend slope (strong trend confirmation)
    "long":        1000,
}

# ─── Online Learner ───
# CHANGED: "ensemble" -> "logistic" — single transparent model
# The 3-model ensemble (Logistic + HAT + SRP) was a black box.
# Now: Logistic Regression is a CONFIRMATION signal, not the primary decision-maker.
# You can inspect its weights anytime and see what features it values.
MODEL_TYPE = "logistic"
LEARNING_RATE = 0.01
L2_REGULARIZATION = 0.01
REPLAY_BUFFER_SIZE = 5000
DRIFT_DETECTION_SENSITIVITY = 0.001

# ─── Per-Tick Live Learning ───
TICK_LEARN_ENABLED = False
TICK_LEARN_INTERVAL = 5

# ─── Drift Retrain ───
DRIFT_RETRAIN_ENABLED = True
DRIFT_RETRAIN_COOLDOWN = 120

# ─── Model Persistence ───
MODEL_SNAPSHOT_INTERVAL = 50
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─── Meta-Selector (Market Bandit) ───
META_SELECTOR_EXPLORATION_RATE = 0.10
META_SELECTOR_MIN_TRADES = 5
META_SELECTOR_SCORE_WINDOW = 50

# ─── Logging ───
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

# ─── Dashboard ───
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
DASHBOARD_REFRESH_MS = 2000

# ─── Mode ───
TRADING_MODE = "live"
