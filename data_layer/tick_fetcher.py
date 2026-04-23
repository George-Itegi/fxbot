# =============================================================
# data_layer/tick_fetcher.py
# PURPOSE: Fetch raw tick data from MT5 for a single symbol.
# This is the foundation of order flow analysis.
# v2.0: FIXED tick classification for ALL symbol types.
#       Uses spread-proportional thresholds + consecutive tick
#       comparison instead of hardcoded 0.000015.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()

# Symbol categories for pip/point handling
JPY_PAIRS = {'USDJPY', 'EURJPY', 'GBPJPY'}
COMMODITIES = {'XAUUSD'}


def get_symbol_category(symbol: str) -> str:
    """Classify symbol into category for proper tick handling."""
    sym = str(symbol).upper()
    if sym in COMMODITIES:
        return 'COMMODITY'
    if sym in JPY_PAIRS:
        return 'JPY'
    return 'FOREX'


def get_tick_threshold(symbol: str, ask: float, bid: float) -> float:
    """
    Get an appropriate threshold for comparing last price to bid/ask.
    Uses spread-proportional threshold so it works for ALL symbol types.
    """
    spread = ask - bid
    if spread <= 0:
        spread = ask * 0.00001  # Fallback: ~0.001%
    # Use 30% of spread as threshold — tight enough to detect aggressive fills
    return spread * 0.3


def classify_tick(flags: int, bid: float, ask: float, last: float,
                  prev_bid: float = 0, prev_ask: float = 0,
                  symbol: str = '') -> str:
    """
    Classify a tick as BUY, SELL, or NEUTRAL.

    Priority:
    1. If 'last' price exists (actual trade on exchanges/indices),
       compare to bid/ask using spread-proportional threshold.
    2. If no last price (pure forex/OTC), use consecutive bid/ask
       movement to determine direction.
    3. Flag bits as fallback.

    FIXED: Now works correctly for JPY pairs, Gold, Oil, and Indices.
    """
    # ── Method 1: Last price comparison (indices, futures, CFDs) ──
    if last > 0:
        threshold = get_tick_threshold(symbol, ask, bid)

        # Last price near ask = aggressive buyer (lifting the offer)
        if abs(last - ask) <= threshold:
            return 'BUY'
        # Last price near bid = aggressive seller (hitting the bid)
        if abs(last - bid) <= threshold:
            return 'SELL'

        # For high-priced instruments, also check if last is above mid
        # and closer to ask (buying) or below mid and closer to bid (selling)
        mid = (ask + bid) / 2
        half_spread = (ask - bid) / 2
        if half_spread > 0:
            if last > mid:
                return 'BUY'
            elif last < mid:
                return 'SELL'

    # ── Method 2: Consecutive tick comparison (forex OTC) ──
    # Compare current bid/ask to previous tick to detect direction
    if prev_bid > 0 and prev_ask > 0:
        bid_diff = bid - prev_bid
        ask_diff = ask - prev_ask

        # Both bid and ask moved up = buying pressure
        if bid_diff > 0 and ask_diff > 0:
            return 'BUY'
        # Both moved down = selling pressure
        if bid_diff < 0 and ask_diff < 0:
            return 'SELL'
        # Only ask moved up (offer lifted) = buying
        if ask_diff > abs(bid_diff) and ask_diff > 0:
            return 'BUY'
        # Only bid moved down (bid hit) = selling
        if abs(bid_diff) > ask_diff and bid_diff < 0:
            return 'SELL'

    # ── Method 3: Flag bits as last resort ──
    # Bit 1 (value 2) = bid changed, Bit 2 (value 4) = ask changed
    if flags & 4 and not flags & 2:   # ask rose, bid didn't
        return 'BUY'
    if flags & 2 and not flags & 4:   # bid dropped, ask didn't
        return 'SELL'

    return 'NEUTRAL'


def get_ticks(symbol: str, num_ticks: int = 500) -> pd.DataFrame | None:
    """
    Fetch the last N ticks for a symbol.
    Returns a clean DataFrame with direction classified per tick.
    v2.0: Uses improved classification with spread-proportional thresholds
          and consecutive tick comparison.
    """
    ticks = mt5.copy_ticks_from(
        symbol,
        datetime.now(timezone.utc),
        num_ticks,
        mt5.COPY_TICKS_ALL
    )

    if ticks is None or len(ticks) == 0:
        return None

    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Sort by time to ensure chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # Pre-compute previous bid/ask for consecutive comparison
    df['prev_bid'] = df['bid'].shift(1)
    df['prev_ask'] = df['ask'].shift(1)

    # Classify each tick using improved multi-method logic
    symbol_cat = get_symbol_category(symbol)
    df['side'] = df.apply(
        lambda row: classify_tick(
            row['flags'], row['bid'], row['ask'], row['last'],
            row['prev_bid'], row['prev_ask'], symbol
        ),
        axis=1
    )

    df = df[['time', 'bid', 'ask', 'last', 'volume', 'flags', 'side']]

    return df


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        exit()
    if not mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER")
    ):
        print(f"Login failed: {mt5.last_error()}")
        exit()
    print("Connected to MT5\n")

    SYMBOLS = ["EURUSD", "USDJPY", "XAUUSD"]
    NUM_TICKS = 500

    for SYMBOL in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"  {SYMBOL} ({get_symbol_category(SYMBOL)}) — last {NUM_TICKS} ticks")
        print(f"{'='*50}")

        df = get_ticks(SYMBOL, NUM_TICKS)
        if df is not None:
            buys     = len(df[df['side'] == 'BUY'])
            sells    = len(df[df['side'] == 'SELL'])
            neutrals = len(df[df['side'] == 'NEUTRAL'])
            total    = buys + sells + neutrals
            delta    = buys - sells

            print(f"  Total ticks  : {total}")
            print(f"  BUY  ticks   : {buys} ({buys/total*100:.1f}%)")
            print(f"  SELL ticks   : {sells} ({sells/total*100:.1f}%)")
            print(f"  NEUTRAL ticks: {neutrals} ({neutrals/total*100:.1f}%)")
            print(f"  Delta        : {delta:+d}")
            print(f"  Bias         : {'BUYERS' if delta > 0 else 'SELLERS' if delta < 0 else 'NEUTRAL'}")
        else:
            print(f"  No tick data available")

    mt5.shutdown()
