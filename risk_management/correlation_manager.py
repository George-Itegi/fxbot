# =============================================================
# risk_management/correlation_manager.py
# PURPOSE: Manage currency exposure across correlated pairs.
# Prevents over-concentration in a single currency.
# Example: If already long EURUSD and long EURGBP, block new EUR
# longs to prevent double EUR exposure.
# =============================================================

from core.logger import get_logger
from config.settings import MAX_CORRELATED_EXPOSURE, MAX_SAME_CURRENCY_EXPOSURE

log = get_logger(__name__)

# ── Currency Exposure Groups ────────────────────────────────
# Each pair maps to (base_currency, quote_currency)
# Trading BUY on a pair = long base, short quote
# Trading SELL on a pair = short base, long quote

CURRENCY_MAP = {
    # Major USD pairs
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"),
    # JPY crosses
    "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"),
    # Commodities
    "XAUUSD": ("XAU", "USD"),
}

# ── Highly correlated pairs (trade same direction ~80%+) ────
# If one is in a position, treat the other as correlated exposure
CORRELATION_GROUPS = {
    "EUR_MAJORS": ["EURUSD", "EURJPY"],
    "GBP_MAJORS": ["GBPUSD", "GBPJPY"],
    "JPY_MAJORS": ["USDJPY", "EURJPY", "GBPJPY"],
    "USD_MAJORS": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
    "GOLD_USD":  ["XAUUSD"],
}

# Currencies that share strong positive correlation
ALIGNED_CURRENCIES = {
    "EUR": ["EUR"],
    "GBP": ["GBP"],
    "AUD": ["AUD"],
    "NZD": ["NZD"],
    "CAD": ["CAD"],
    "CHF": ["CHF"],
    "JPY": ["JPY"],
    "USD": ["USD"],
}


def get_currency_exposure(symbol: str, direction: str) -> dict:
    """
    Calculate the net currency exposure for a proposed trade.
    
    Returns:
        dict with currency -> net exposure (positive = long, negative = short)
        e.g. {"EUR": +1, "USD": -1} for BUY EURUSD
    """
    pair = CURRENCY_MAP.get(symbol, (None, None))
    if pair[0] is None:
        return {}

    base, quote = pair

    if direction == "BUY":
        return {base: +1, quote: -1}
    else:  # SELL
        return {base: -1, quote: +1}


def check_correlation_risk(symbol: str, direction: str,
                           open_positions: list) -> tuple:
    """
    Check if a new trade would create excessive currency correlation.
    
    Args:
        symbol: Proposed symbol to trade
        direction: "BUY" or "SELL"
        open_positions: List of MT5 position objects
    
    Returns:
        (allowed: bool, reason: str)
    """
    # Calculate current total exposure across all open positions
    current_exposure = {}
    for pos in open_positions:
        pair = CURRENCY_MAP.get(pos.symbol, (None, None))
        if pair[0] is None:
            continue
        base, quote = pair
        
        # Position type: 0 = BUY (long base), 1 = SELL (short base)
        multiplier = +1 if pos.type == 0 else -1
        
        current_exposure[base] = current_exposure.get(base, 0) + multiplier
        current_exposure[quote] = current_exposure.get(quote, 0) - multiplier

    # Calculate proposed exposure
    proposed = get_currency_exposure(symbol, direction)
    
    # Calculate combined exposure
    combined = dict(current_exposure)
    for currency, exposure in proposed.items():
        combined[currency] = combined.get(currency, 0) + exposure

    # ── Check 1: Max same-direction exposure per currency ──
    for currency, net in combined.items():
        if currency in ("IDX", "XAU", "XAG", "OIL"):
            continue  # Skip commodities and indices for currency check
        if abs(net) > MAX_SAME_CURRENCY_EXPOSURE:
            direction_str = "LONG" if net > 0 else "SHORT"
            log.info(f"[CORR] ❌ {symbol} {direction} blocked — "
                     f"{currency} exposure would be {net}x {direction_str} "
                     f"(max: {MAX_SAME_CURRENCY_EXPOSURE})")
            return False, f"excessive_{currency}_exposure"

    # ── Check 2: Correlated pair overlap ──
    pair_info = CURRENCY_MAP.get(symbol, (None, None))
    if pair_info[0] is None:
        return True, "ok"

    base, quote = pair_info

    # Check if any open position is in the same correlation group
    open_symbols = [p.symbol for p in open_positions]
    for group_name, group_pairs in CORRELATION_GROUPS.items():
        if symbol not in group_pairs:
            continue
        
        # Count how many correlated pairs are already open in same direction
        correlated_count = 0
        for open_sym in open_symbols:
            if open_sym in group_pairs and open_sym != symbol:
                # Check if direction aligns
                for pos in open_positions:
                    if pos.symbol == open_sym:
                        open_dir = "BUY" if pos.type == 0 else "SELL"
                        if open_dir == direction:
                            correlated_count += 1
                        break
        
        if correlated_count >= MAX_CORRELATED_EXPOSURE:
            log.info(f"[CORR] ❌ {symbol} {direction} blocked — "
                     f"{correlated_count} correlated {group_name} positions "
                     f"already open (max: {MAX_CORRELATED_EXPOSURE})")
            return False, f"correlated_{group_name}"

    # ── Check 3: Opposing direction warning (not a block) ──
    for currency, exposure in proposed.items():
        current = current_exposure.get(currency, 0)
        if current != 0 and ((current > 0 and exposure > 0) or 
                             (current < 0 and exposure < 0)):
            net = current + exposure
            log.info(f"[CORR] ⚠️ {symbol} {direction} — {currency} "
                     f"exposure: {current} → {net} (aligned)")
        elif current != 0 and ((current > 0 and exposure < 0) or 
                               (current < 0 and exposure > 0)):
            net = current + exposure
            if abs(net) > 0:
                log.info(f"[CORR] ⚠️ {symbol} {direction} — {currency} "
                         f"partial hedge: {current} → {net}")

    return True, "ok"


def get_portfolio_exposure_summary(open_positions: list) -> dict:
    """
    Get a summary of current currency exposure across all positions.
    Used for dashboard display.
    """
    exposure = {}
    for pos in open_positions:
        pair = CURRENCY_MAP.get(pos.symbol, (None, None))
        if pair[0] is None:
            continue
        base, quote = pair
        multiplier = +1 if pos.type == 0 else -1
        
        exposure[base] = exposure.get(base, 0) + multiplier
        exposure[quote] = exposure.get(quote, 0) - multiplier
    
    return exposure
