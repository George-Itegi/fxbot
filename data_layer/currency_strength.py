# =============================================================
# data_layer/currency_strength.py
# Currency Strength Meter — measures relative strength of 10
# major currencies/commodities from forex pair price changes.
#
# Algorithm:
#   1. Fetch recent candles for all known pairs
#   2. Compute % change over multiple lookback windows
#   3. For each pair, add % change to BASE, subtract from QUOTE
#   4. Average contributions per currency
#   5. Normalize to -100 to +100 via percentile ranking
# =============================================================

import numpy as np
from collections import defaultdict
from data_layer.price_feed import get_candles
from core.logger import get_logger

log = get_logger(__name__)

# All currencies/commodities we track
TRACKED_CURRENCIES = [
    'USD', 'EUR', 'GBP', 'JPY', 'AUD',
    'CAD', 'CHF', 'NZD', 'XAU', 'XAG',
]

# Standard pair-to-currency mapping (BASE, QUOTE)
PAIR_CURRENCY_MAP = {
    'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'), 'AUDUSD': ('AUD', 'USD'),
    'NZDUSD': ('NZD', 'USD'), 'USDCAD': ('USD', 'CAD'), 'USDCHF': ('USD', 'CHF'),
    'USDJPY': ('USD', 'JPY'), 'EURJPY': ('EUR', 'JPY'), 'GBPJPY': ('GBP', 'JPY'),
    'AUDJPY': ('AUD', 'JPY'), 'CADJPY': ('CAD', 'JPY'), 'CHFJPY': ('CHF', 'JPY'),
    'NZDJPY': ('NZD', 'JPY'), 'EURGBP': ('EUR', 'GBP'), 'EURAUD': ('EUR', 'AUD'),
    'GBPAUD': ('GBP', 'AUD'), 'AUDCAD': ('AUD', 'CAD'), 'EURNZD': ('EUR', 'NZD'),
    'GBPNZD': ('GBP', 'NZD'), 'AUDNZD': ('AUD', 'NZD'), 'EURCAD': ('EUR', 'CAD'),
    'EURCHF': ('EUR', 'CHF'), 'GBPCHF': ('GBP', 'CHF'), 'NZDCAD': ('NZD', 'CAD'),
    'XAUUSD': ('XAU', 'USD'), 'XAGUSD': ('XAG', 'USD'),
}

# Lookback periods for multi-window averaging (in bars)
LOOKBACK_WINDOWS = [20, 50, 100]

# Safe default return: all currencies at 0
_SAFE_DEFAULT = {c: 0.0 for c in TRACKED_CURRENCIES}


def calculate_currency_strength(pairs: list = None, timeframe: str = 'M15',
                                lookback: int = 200) -> dict:
    """
    Calculate relative currency strength from forex pair price changes.
    Returns dict with strength values for each currency (-100 to +100).

    Args:
        pairs:     List of forex pair symbols. Defaults to PAIR_WHITELIST.
        timeframe: MT5 timeframe string (e.g. 'M15', 'H1').
        lookback:  Number of bars to fetch per pair.

    Returns:
        Dict mapping currency code -> float (-100 to +100).
        Positive = strong, Negative = weak.
        Returns safe defaults on MT5 failure.
    """
    if pairs is None:
        try:
            from config.settings import PAIR_WHITELIST
            pairs = PAIR_WHITELIST
        except Exception:
            pairs = list(PAIR_CURRENCY_MAP.keys())

    # Accumulators: currency -> list of % change contributions
    contributions = defaultdict(list)

    for pair in pairs:
        pair_upper = pair.upper()
        if pair_upper not in PAIR_CURRENCY_MAP:
            continue

        base_ccy, quote_ccy = PAIR_CURRENCY_MAP[pair_upper]

        # Fetch candles — need the longest lookback window + some buffer
        required_bars = max(LOOKBACK_WINDOWS) + 10
        df = get_candles(pair_upper, timeframe, min(lookback, required_bars))
        if df is None or len(df) < required_bars:
            continue

        try:
            closes = df['close'].values.astype(float)
            # Compute % change for each lookback window
            window_changes = []
            for n in LOOKBACK_WINDOWS:
                if len(closes) > n:
                    pct_change = (closes[-1] - closes[-(n + 1)]) / closes[-(n + 1)] * 100
                    window_changes.append(pct_change)

            if not window_changes:
                continue

            # Average % change across all lookback windows
            avg_change = np.mean(window_changes)

            # BASE strengthens when pair goes up, QUOTE weakens
            contributions[base_ccy].append(avg_change)
            contributions[quote_ccy].append(-avg_change)

        except Exception as e:
            log.warning(f"[CURRENCY_STRENGTH] Failed to compute for {pair}: {e}")
            continue

    if not contributions:
        log.warning("[CURRENCY_STRENGTH] No pair data available — returning defaults")
        return dict(_SAFE_DEFAULT)

    # Average all contributions per currency
    avg_strength = {}
    for ccy in TRACKED_CURRENCIES:
        if ccy in contributions and contributions[ccy]:
            avg_strength[ccy] = float(np.mean(contributions[ccy]))
        else:
            avg_strength[ccy] = 0.0

    # Normalize to -100 to +100 using percentile ranking
    values = np.array([avg_strength[ccy] for ccy in TRACKED_CURRENCIES])
    normalized = _normalize_to_range(values)

    result = {}
    for i, ccy in enumerate(TRACKED_CURRENCIES):
        result[ccy] = round(float(normalized[i]), 1)

    log.debug(f"[CURRENCY_STRENGTH] "
              f"Strongest: {max(result, key=result.get)} ({max(result.values()):.1f}) | "
              f"Weakest: {min(result, key=result.get)} ({min(result.values()):.1f})")

    return result


def _normalize_to_range(values: np.ndarray) -> np.ndarray:
    """
    Normalize an array of values to -100..+100 range using
    percentile-like scaling based on the data distribution.

    Uses median centering + MAD (Median Absolute Deviation) scaling
    for robustness against outlier currencies.
    """
    if len(values) == 0:
        return np.array([])

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad < 1e-10:
        # All values are the same — return all zeros
        return np.zeros_like(values)

    # Scale by MAD (more robust than std dev)
    scaled = (values - median) / (mad * 1.4826)  # 1.4826 = MAD→σ conversion

    # Clip to ±3 sigma then scale to ±100
    scaled = np.clip(scaled, -3.0, 3.0)
    normalized = scaled * (100.0 / 3.0)

    return normalized


def get_currency_ranking(strength: dict, direction: str = 'strong',
                         top_n: int = 3) -> list:
    """
    Get top N strongest or weakest currencies from a strength dict.

    Args:
        strength:  Dict from calculate_currency_strength()
        direction: 'strong' (highest values) or 'weak' (lowest values)
        top_n:     Number of currencies to return

    Returns:
        List of (currency, strength_value) tuples, sorted.
    """
    if not strength:
        return []

    sorted_ccys = sorted(strength.items(), key=lambda x: x[1], reverse=True)

    if direction == 'weak':
        sorted_ccys.reverse()

    return sorted_ccys[:top_n]


def get_pair_strength_bias(strength: dict, pair: str) -> dict:
    """
    Determine directional bias for a specific pair from currency strength.

    Args:
        strength: Dict from calculate_currency_strength()
        pair:     Forex pair string (e.g. 'EURJPY')

    Returns:
        Dict with:
            - bias: 'BUY' | 'SELL' | 'NEUTRAL'
            - base_strength: float
            - quote_strength: float
            - strength_diff: float (base - quote)
    """
    pair_upper = pair.upper()
    if pair_upper not in PAIR_CURRENCY_MAP:
        return {'bias': 'NEUTRAL', 'base_strength': 0.0,
                'quote_strength': 0.0, 'strength_diff': 0.0}

    base_ccy, quote_ccy = PAIR_CURRENCY_MAP[pair_upper]
    base_val = strength.get(base_ccy, 0.0)
    quote_val = strength.get(quote_ccy, 0.0)
    diff = base_val - quote_val

    # Threshold for directional conviction
    threshold = 15.0  # Only recommend direction if diff > 15 points

    if diff > threshold:
        bias = 'BUY'   # Base is stronger than quote → pair should go up
    elif diff < -threshold:
        bias = 'SELL'  # Quote is stronger → pair should go down
    else:
        bias = 'NEUTRAL'

    return {
        'bias': bias,
        'base_currency': base_ccy,
        'quote_currency': quote_ccy,
        'base_strength': base_val,
        'quote_strength': quote_val,
        'strength_diff': round(diff, 1),
    }
