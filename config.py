"""
Deriv Over/Under Bot — Configuration v10
==========================================
Conservative Edge Detection — trade RARELY but RELIABLY.

v9 FAILED because it chased extreme barriers (Over 7/8). These are lottery tickets:
  - Over 8 has 20% natural prob — you LOSE 80% of trades
  - The "observed probability" (30-40%) was just noise, not a real edge
  - On 200-tick samples, z=1.3 catches ~10% random fluctuations as "significant"
  - With 17 barriers tested, you ALWAYS find one that looks good by pure chance

v10 ROOT CAUSE FIX: Multiple testing problem + noise chasing.
  - Testing 17 barriers at z>1.3 means ~170% chance of finding a false positive
  - Extreme barriers amplify noise because the EV formula multiplies by the payout
  - The observed probability is unreliable — it reverts to natural probability immediately

v10 Strategy: "Conservative Edge Detection"
1. RESTRICT barriers to moderate range (Over 3-6, Under 4-7)
   These have 30-60% natural prob — even without an edge, you win 30-60% of trades
2. BAYESIAN SHRINKAGE: blend observed prob with natural prob based on sample size
   This prevents the bot from chasing noise — if n=200, shrink 50% toward natural
3. MUCH stricter z-score: 3.0 minimum (was 1.3 — caught 10% of random fluctuations!)
4. ALL 3 windows must agree (was 2/3)
5. FIXED 5-tick duration — no more chaos between 2t/3t/5t
6. ML disagreement BLOCKS trades (was 20% reduction — not strong enough)
7. Higher minimums: 35% confidence, 8% EV, 70% setup score
8. Fewer trades, higher quality — aim for 60%+ win rate instead of 13%
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

# ─── Dynamic Barrier Selection (v10 — CONSERVATIVE RANGE ONLY) ───
# v9 tested ALL barriers (Over 0-8, Under 1-9) and chased extreme ones.
# This was WRONG because:
#   - Testing 17 barriers creates a massive multiple testing problem
#   - Extreme barriers (Over 7/8) amplify noise via the payout multiplier
#   - The observed probability on extreme barriers is unreliable
#
# v10: Only test MODERATE barriers where:
#   - Natural probability is 30-60% (you win 30-60% even without an edge)
#   - Payout is 65-233% (enough to profit, not so high it's a lottery)
#   - Variance is manageable (you survive long enough to profit from real edges)
#
# Over 3 (60% natural, ~59% payout): Moderate prob, low payout
# Over 4 (50% natural, ~95% payout): Balanced
# Over 5 (40% natural, ~143% payout): Good balance of prob/payout
# Over 6 (30% natural, ~233% payout): Lower prob, higher payout
# Under 4 (40% natural, ~143% payout): Mirror of Over 5
# Under 5 (50% natural, ~95% payout): Mirror of Over 4
# Under 6 (60% natural, ~59% payout): Mirror of Over 3
# Under 7 (70% natural, ~37% payout): High prob, very low payout
DYNAMIC_BARRIERS = True                # Enable dynamic barrier selection
BARRIER_OVER_OPTIONS = [3, 4, 5, 6]    # v10: MODERATE only (was 0-8)
BARRIER_UNDER_OPTIONS = [4, 5, 6, 7]   # v10: MODERATE only (was 1-9)
MIN_BARRIER_PROBABILITY = 0.25         # v10: Minimum 25% natural prob (was 8%)
                                        # Over 7 (20%) = excluded, Over 6 (30%) = allowed
MAX_BARRIER_PROBABILITY = 0.75         # v10: Maximum 75% natural prob (was 70%)
PAYOUT_HOUSE_MARGIN = 0.046            # Deriv's house margin (~4.6% based on observed Over 4 payout)
MIN_EV_FOR_TRADE = 0.08                # v10: Minimum 8% EV (was 5% — too low, caught noise trades)

# ─── Bayesian Shrinkage (v10 — THE KEY FIX) ───
# Instead of using the raw observed probability (which is noisy), blend it
# with the natural probability using Bayesian shrinkage:
#   adjusted_prob = (n * observed + k * natural) / (n + k)
#
# Where n = sample size and k = prior strength.
# With n=200 (medium window) and k=100:
#   adjusted_prob = (200 * observed + 100 * natural) / 300
#   = 67% weight on observed, 33% weight on natural
#
# This means: even if the observed probability says 40% on Over 8 (10% natural),
# the adjusted probability would be (200*0.40 + 100*0.10) / 300 = 0.30
# And the EV would be 0.30 * 9.59 - 1 = +1.88 — wait, that's still positive!
# BUT: the z-score threshold of 3.0 would prevent this from being selected
# because a jump from 10% to 40% on 200 ticks has z = (0.40-0.10)/sqrt(0.10*0.90/200) = 14.1
# which IS very significant... but with ONLY 8 moderate barriers, the
# multiple testing problem is much reduced, so z=3.0 is more reliable.
#
# More importantly: for moderate barriers like Over 5 (40% natural),
# if we observe 50%, adjusted = (200*0.50 + 100*0.40)/300 = 0.467
# EV = 0.467 * 2.43 - 1 = +0.135 = 13.5% — reasonable.
# And Over 5 actually wins 40% of the time, so even if the edge is noise,
# you still win 40% (not 10% like Over 8).
BAYESIAN_SHRINKAGE_PRIOR = 100          # v10: Shrinkage prior strength
                                        # With n=200: 67% observed, 33% natural
                                        # With n=500: 83% observed, 17% natural

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
# v10: MUCH stricter than v9. v9 traded at 13% confidence with z=1.3 — pure gambling.
# z=1.3 catches ~10% of random fluctuations as "significant."
# z=3.0 catches only 0.3% — much more likely to be a real edge.
# On synthetic indices, genuine edges are RARE. Trade less, win more.
MIN_CONFIDENCE = 0.35           # v10: Minimum 35% observed win probability (was 12%!)
MIN_EDGE_THRESHOLD = 0.08       # v10: Minimum 8% EV (was 5%)
MIN_FREQ_EDGE_ZSCORE = 3.0      # v10: 3-sigma (99.7% confidence) — was 1.3 (80% = noise!)
                                 # z=3.0 means the observed edge is VERY unlikely to be random

# ─── Digit Frequency Direction (PRIMARY DECISION) ───
# v10: Only moderate barriers + ALL 3 windows must agree.
# v9 tested 17 barriers with 2/3 window agreement — massive false positive rate.
# With 8 moderate barriers and z=3.0, we still need all windows to confirm.
MIN_DIGIT_FREQUENCY_EDGE = 0.03   # v10: Minimum 3% ABSOLUTE edge (was 1% — too loose)
MIN_DIGIT_FREQUENCY_EDGE_RELATIVE = 0.15  # v10: Minimum 15% RELATIVE edge (was 10%)
                                    # For Over 6 (30%): need 4.5% absolute -> obs > 34.5%
                                    # For Over 4 (50%): need 7.5% absolute -> obs > 57.5%
                                    # For Over 5 (40%): need 6% absolute -> obs > 46%
DIGIT_FREQ_WINDOW_AGREEMENT = 3   # v10: ALL 3 windows must agree (was 2 — too many false signals)

# ─── Setup Quality ───
# v10: Raised to 0.70 — only trade when we have strong evidence.
# v9's 0.60 threshold allowed too many weak setups through.
MIN_SETUP_SCORE = 0.70           # v10: Minimum setup quality (was 0.60)

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
MARKET_SESSION_MAX_IDLE_SEC = 180  # After 180s with no trade on current market, allow switch
MARKET_STICKY_AFTER_TRADE = True   # v8.1: After trading a market, keep trading it while setup is good

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
FORCE_TRADE_MIN_CONFIDENCE = 0.35   # v10: Aligned with MIN_CONFIDENCE
FORCE_TRADE_MIN_EV = 0.08           # v10: Aligned with MIN_EV_FOR_TRADE
FORCE_TRADE_MIN_SETUP_SCORE = 0.85  # v10: Must be VERY strong to force without ML

# ─── Trend Setting (v10: SOFT BIAS — not required, but helpful) ───
# On synthetic indices, price trend has NO causal relationship with digit distribution.
# However, trend alignment with digit frequency is a nice-to-have confirmation.
# v10: Trend is a SOFT BIAS — it boosts/penalizes but never blocks a strong freq signal.
TREND_REQUIRED = False               # Trend is NOT required (digit freq is primary)
TREND_SLOPE_TSTAT_THRESHOLD = 3.0    # 3-sigma for trend detection
TREND_CONFIDENCE_BOOST = 0.03        # v10: Small boost when trend aligns (was 5%)
TREND_MISALIGN_PENALTY = 0.15        # v10: Larger penalty when trend opposes (was 10%)
TREND_CONFIDENCE_REDUCTION = 0.05    # Legacy
TREND_SIGNAL_SCORE_REDUCTION = 0.05  # Legacy

# ─── Martingale Confidence Gate ───
# During martingale recovery, the bot MUST stay on the SAME market.
# No switching markets during recovery — the setup was good on THAT market.
# v10: Much stricter — same as regular thresholds.
MARTINGALE_MIN_CONFIDENCE = 0.35   # v10: Same as MIN_CONFIDENCE (was 12%!)
MARTINGALE_MIN_SETUP_SCORE = 0.70  # v10: Same as MIN_SETUP_SCORE (was 0.60)
MARTINGALE_SAME_MARKET = True      # MUST recover on the same market where loss occurred
MARTINGALE_MAX_STEPS = 2           # v10: Only 2 recovery steps (3 was too risky)
MAX_DAILY_TRADES = 0          # 0 = unlimited (demo training mode)
COOLDOWN_AFTER_LOSS_TICKS = 1
MIN_TRADE_INTERVAL_SEC = 15   # v10: Raised from 10s — even fewer trades, higher quality

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
