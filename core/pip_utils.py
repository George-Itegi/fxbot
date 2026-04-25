# =============================================================
# core/pip_utils.py
# SINGLE SOURCE OF TRUTH for pip size calculations.
# All strategies AND the engine MUST use this.
#
# Why: Strategies used price-based detection (price > 500 → 1.0)
# while the engine used symbol-name detection ("XAU" → 0.1).
# This caused:
#   - XAGUSD: 100x mismatch (strategy 0.0001 vs engine 0.01)
#   - XAUUSD: 10x mismatch (strategy 1.0 vs engine 0.1)
# Resulting in impossible SL/TP levels and random trade outcomes.
# =============================================================


def get_pip_size(symbol: str, price: float = None) -> float:
    """
    Return pip size for a symbol.

    Priority: symbol name > price heuristic.
    Symbol-name is authoritative (matches broker conventions).
    Price heuristic is a fallback only.

    Args:
        symbol: Symbol string like "EURUSD", "XAUUSD", "GBPJPY"
        price: Optional current price (used as fallback)

    Returns:
        float: pip size (e.g. 0.0001 for EURUSD, 0.01 for USDJPY, 0.1 for XAUUSD)
    """
    sym = symbol.upper()

    # Index CFDs
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        return 1.0

    # Gold: 1 pip = 0.1 (moves in $0.10 increments)
    if "XAU" in sym:
        return 0.1

    # Silver: 1 pip = 0.01 (moves in $0.01 increments)
    if "XAG" in sym:
        return 0.01

    # Oil
    if any(x in sym for x in ["WTI", "BRN"]):
        return 0.01

    # JPY pairs: 2 decimal places
    if "JPY" in sym:
        return 0.01

    # Standard forex: 4 decimal places
    if price is not None and price < 10:
        return 0.0001

    # Fallback: use price heuristic
    if price is not None:
        if price > 500:
            return 1.0
        elif price > 50:
            return 0.01
        else:
            return 0.0001

    return 0.0001
