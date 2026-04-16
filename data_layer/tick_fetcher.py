# =============================================================
# data_layer/tick_fetcher.py
# PURPOSE: Fetch raw tick data from MT5 for a single symbol.
# This is the foundation of order flow analysis.
# Run this file standalone to test and see real tick data.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()


def connect():
    """Simple connection for standalone testing."""
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    if not mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER")
    ):
        print(f"Login failed: {mt5.last_error()}")
        return False
    print("Connected to MT5\n")
    return True


def classify_tick(flags: int, bid: float, ask: float, last: float) -> str:
    """
    Classify a tick as BUY, SELL, or NEUTRAL.
    MT5 uses bitmask flags — we check the bits, not exact values.
    Bit 1 (value 2)  = bid changed
    Bit 2 (value 4)  = ask changed
    Bit 4 (value 16) = last price changed (actual trade)
    We approximate direction from bid/ask movement.
    """
    # If last price exists and matches ask → aggressive buyer
    if last > 0 and abs(last - ask) < 0.000015:
        return 'BUY'
    # If last price exists and matches bid → aggressive seller
    if last > 0 and abs(last - bid) < 0.000015:
        return 'SELL'
    # No last price (forex) — use flag bits to approximate
    if flags & 4 and not flags & 2:   # ask changed, bid did not → buying pressure
        return 'BUY'
    if flags & 2 and not flags & 4:   # bid changed, ask did not → selling pressure
        return 'SELL'
    return 'NEUTRAL'


def get_ticks(symbol: str, num_ticks: int = 500) -> pd.DataFrame | None:
    """
    Fetch the last N ticks for a symbol.
    Returns a clean DataFrame with direction classified per tick.
    """
    ticks = mt5.copy_ticks_from(
        symbol,
        datetime.now(timezone.utc),
        num_ticks,
        mt5.COPY_TICKS_ALL
    )

    if ticks is None or len(ticks) == 0:
        print(f"No tick data for {symbol}")
        return None

    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Classify each tick using improved logic
    df['side'] = df.apply(
        lambda row: classify_tick(row['flags'], row['bid'], row['ask'], row['last']),
        axis=1
    )

    df = df[['time', 'bid', 'ask', 'last', 'volume', 'flags', 'side']]
    return df


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL   = "EURUSD"
    NUM_TICKS = 500

    print(f"Fetching last {NUM_TICKS} ticks for {SYMBOL}...\n")
    df = get_ticks(SYMBOL, NUM_TICKS)

    if df is not None:
        print(f"Total ticks fetched : {len(df)}")
        print(f"Time range          : {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

        print(f"\n--- Last 10 ticks ---")
        print(df.tail(10).to_string(index=False))

        # Breakdown of all flag values seen
        print(f"\n--- Flag Values Found ---")
        print(df['flags'].value_counts().to_string())

        # Summary
        buys     = len(df[df['side'] == 'BUY'])
        sells    = len(df[df['side'] == 'SELL'])
        neutrals = len(df[df['side'] == 'NEUTRAL'])
        delta    = buys - sells

        print(f"\n--- Tick Summary ---")
        print(f"BUY    ticks : {buys}")
        print(f"SELL   ticks : {sells}")
        print(f"NEUTRAL ticks: {neutrals}")
        print(f"Delta        : {delta}")
        print(f"Bias         : {'BUYERS in control 📈' if delta > 0 else 'SELLERS in control 📉' if delta < 0 else 'NEUTRAL ↔️'}")

    mt5.shutdown()
