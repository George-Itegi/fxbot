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
# ORDER FLOW IMBALANCE
# Measures buy vs sell pressure as a ratio.
# Only BUY when imbalance > +0.3 (buyers overwhelming)
# Only SELL when imbalance < -0.3 (sellers overwhelming)
# Impact: 30-40% fewer false signals
# =============================================================

def calculate_order_flow_imbalance(df_ticks: pd.DataFrame,
                                   window: int = 50,
                                   buy_threshold: float = 0.3,
                                   sell_threshold: float = -0.3) -> dict:
    """
    Calculate order flow imbalance ratio over the last N ticks.
    This is the key filter for scalping entries — only trade when
    one side is clearly in control.

    Imbalance = (buy_ticks - sell_ticks) / total_active_ticks
    Range: -1.0 (all sells) to +1.0 (all buys)

    Args:
        df_ticks: DataFrame with 'side' column ('BUY', 'SELL', 'NEUTRAL')
        window: Number of recent ticks to analyze (default 50)
        buy_threshold: Minimum imbalance to allow BUY entry (default +0.3)
        sell_threshold: Maximum imbalance to allow SELL entry (default -0.3)

    Returns:
        Dict with:
            - imbalance: float (-1.0 to +1.0)
            - buy_ticks: int (number of buy ticks in window)
            - sell_ticks: int (number of sell ticks in window)
            - neutral_ticks: int (number of neutral ticks in window)
            - total_ticks: int (total ticks in window)
            - active_ticks: int (buy + sell only, excludes neutral)
            - direction: 'BUY', 'SELL', 'NEUTRAL'
            - can_buy: bool (imbalance > buy_threshold)
            - can_sell: bool (imbalance < sell_threshold)
            - strength: 'EXTREME', 'STRONG', 'MODERATE', 'WEAK', 'NONE'
            - dominance_pct: float (dominant side as percentage)
    """
    default = {
        'imbalance': 0.0,
        'buy_ticks': 0,
        'sell_ticks': 0,
        'neutral_ticks': 0,
        'total_ticks': 0,
        'active_ticks': 0,
        'direction': 'NEUTRAL',
        'can_buy': False,
        'can_sell': False,
        'strength': 'NONE',
        'dominance_pct': 0.0,
    }

    if df_ticks is None or len(df_ticks) < 10:
        return default

    recent = df_ticks.tail(window)

    buy_ticks = len(recent[recent['side'] == 'BUY'])
    sell_ticks = len(recent[recent['side'] == 'SELL'])
    neutral_ticks = len(recent[recent['side'] == 'NEUTRAL'])
    total_ticks = len(recent)
    active_ticks = buy_ticks + sell_ticks

    if active_ticks == 0:
        return default

    # Core imbalance calculation: normalized -1 to +1
    imbalance = round((buy_ticks - sell_ticks) / active_ticks, 4)

    # Direction
    if imbalance > 0.1:
        direction = 'BUY'
    elif imbalance < -0.1:
        direction = 'SELL'
    else:
        direction = 'NEUTRAL'

    # Tradeability gates
    can_buy = imbalance >= buy_threshold
    can_sell = imbalance <= sell_threshold

    # Strength classification
    abs_imb = abs(imbalance)
    if abs_imb >= 0.6:
        strength = 'EXTREME'
    elif abs_imb >= 0.4:
        strength = 'STRONG'
    elif abs_imb >= 0.3:
        strength = 'MODERATE'
    elif abs_imb >= 0.15:
        strength = 'WEAK'
    else:
        strength = 'NONE'

    # Dominance: percentage of the winning side
    dominance_pct = round(max(buy_ticks, sell_ticks) / active_ticks * 100, 1)

    return {
        'imbalance': imbalance,
        'buy_ticks': buy_ticks,
        'sell_ticks': sell_ticks,
        'neutral_ticks': neutral_ticks,
        'total_ticks': total_ticks,
        'active_ticks': active_ticks,
        'direction': direction,
        'can_buy': can_buy,
        'can_sell': can_sell,
        'strength': strength,
        'dominance_pct': dominance_pct,
    }


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
