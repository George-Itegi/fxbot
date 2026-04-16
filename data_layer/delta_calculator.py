# =============================================================
# data_layer/delta_calculator.py
# PURPOSE: Calculate order flow delta from tick data.
# Delta = Aggressive Buyers - Aggressive Sellers
# This tells us WHO is in control of the market right now.
# Run this file standalone to test.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

# Import our tick fetcher we already built and tested
from data_layer.tick_fetcher import get_ticks

load_dotenv()


def connect():
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


def calculate_delta(df_ticks: pd.DataFrame) -> dict:
    """
    Calculate delta metrics from a tick DataFrame.

    Returns a dictionary with:
    - delta        : total buyers minus sellers
    - buy_ticks    : number of aggressive buy ticks
    - sell_ticks   : number of aggressive sell ticks
    - cumulative   : running delta over time (list)
    - bias         : 'BULLISH', 'BEARISH', or 'NEUTRAL'
    - strength     : 'STRONG', 'MODERATE', or 'WEAK'
    """
    if df_ticks is None or df_ticks.empty:
        return {}

    buy_ticks  = len(df_ticks[df_ticks['side'] == 'BUY'])
    sell_ticks = len(df_ticks[df_ticks['side'] == 'SELL'])
    total      = buy_ticks + sell_ticks
    delta      = buy_ticks - sell_ticks

    # Cumulative delta — shows how delta built up over time
    df_ticks = df_ticks.copy()
    df_ticks['delta_tick'] = df_ticks['side'].map(
        {'BUY': 1, 'SELL': -1, 'NEUTRAL': 0}
    )
    cumulative = df_ticks['delta_tick'].cumsum().tolist()

    # Bias — which side is winning
    if delta > 0:
        bias = 'BULLISH'
    elif delta < 0:
        bias = 'BEARISH'
    else:
        bias = 'NEUTRAL'

    # Strength — how dominant is one side
    if total == 0:
        strength = 'WEAK'
    else:
        dominance = abs(delta) / total  # 0.0 to 1.0
        if dominance >= 0.6:
            strength = 'STRONG'
        elif dominance >= 0.35:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

    return {
        'delta':       delta,
        'buy_ticks':   buy_ticks,
        'sell_ticks':  sell_ticks,
        'total_ticks': total,
        'cumulative':  cumulative,
        'bias':        bias,
        'strength':    strength,
    }


def get_rolling_delta(df_ticks: pd.DataFrame, window: int = 100) -> dict:
    """
    Calculate delta over the last N ticks only (rolling window).
    More responsive to recent market pressure than full delta.
    """
    if df_ticks is None or len(df_ticks) < window:
        recent = df_ticks
    else:
        recent = df_ticks.tail(window)

    result = calculate_delta(recent)
    result['window'] = len(recent)
    return result


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL    = "EURUSD"
    NUM_TICKS = 500

    print(f"Fetching {NUM_TICKS} ticks for {SYMBOL}...\n")
    df = get_ticks(SYMBOL, NUM_TICKS)

    if df is not None:
        # --- Full delta (all ticks) ---
        full = calculate_delta(df)
        print("=" * 45)
        print("  FULL DELTA REPORT")
        print("=" * 45)
        print(f"  Total ticks analysed : {full['total_ticks']}")
        print(f"  BUY  ticks           : {full['buy_ticks']}")
        print(f"  SELL ticks           : {full['sell_ticks']}")
        print(f"  Delta                : {full['delta']:+d}")
        print(f"  Bias                 : {full['bias']}")
        print(f"  Strength             : {full['strength']}")

        # --- Rolling delta (last 100 ticks only) ---
        rolling = get_rolling_delta(df, window=100)
        print()
        print("=" * 45)
        print("  ROLLING DELTA (last 100 ticks)")
        print("=" * 45)
        print(f"  BUY  ticks           : {rolling['buy_ticks']}")
        print(f"  SELL ticks           : {rolling['sell_ticks']}")
        print(f"  Delta                : {rolling['delta']:+d}")
        print(f"  Bias                 : {rolling['bias']}")
        print(f"  Strength             : {rolling['strength']}")

        # --- Cumulative delta (first 10 and last 10 values) ---
        cum = full['cumulative']
        print()
        print("=" * 45)
        print("  CUMULATIVE DELTA (trend over time)")
        print("=" * 45)
        print(f"  Started at : {cum[0]:+d}")
        print(f"  Ended at   : {cum[-1]:+d}")
        trend = cum[-1] - cum[0]
        print(f"  Net change : {trend:+d}  "
              f"({'Building UP 📈' if trend > 0 else 'Building DOWN 📉' if trend < 0 else 'Flat ↔️'})")

    mt5.shutdown()
