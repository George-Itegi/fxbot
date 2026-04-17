# =============================================================
# data_layer/smc/liquidity_pools.py
# PURPOSE: Detect Liquidity Pools — areas where retail stop
# losses cluster. Institutions sweep these before reversing.
# Buy-side liquidity = above swing highs (retail longs stopped)
# Sell-side liquidity = below swing lows (retail shorts stopped)
# Run standalone to test.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv
import os

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


def get_candles(symbol: str, timeframe, count: int) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def detect_liquidity_pools(df: pd.DataFrame,
                           swing_length: int = 5,
                           max_pools: int = 5) -> dict:
    """
    Detect buy-side and sell-side liquidity pools.

    Buy-side liquidity  = clusters above swing highs
                          (retail stop losses from short sellers)
    Sell-side liquidity = clusters below swing lows
                          (retail stop losses from long buyers)

    When institutions need to fill large orders they sweep these
    levels first to trigger stops and grab liquidity.
    """
    df   = df.copy().reset_index(drop=True)
    buyside   = []  # above swing highs
    sellside  = []  # below swing lows

    for i in range(swing_length, len(df) - swing_length):
        window = df.iloc[i - swing_length: i + swing_length + 1]

        # Swing High = local peak (buy-side liquidity sits above it)
        if df['high'].iloc[i] == window['high'].max():
            # Count how many times price touched this high (equal highs)
            level     = df['high'].iloc[i]
            touches   = ((df['high'] >= level * 0.9999) &
                         (df['high'] <= level * 1.0001)).sum()
            buyside.append({
                'type':    'BUY_SIDE',
                'level':   round(float(level), 5),
                'time':    str(df['time'].iloc[i]),
                'touches': int(touches),
                'swept':   False,
                'note':    'Retail stop losses above — institutions may sweep here',
            })

        # Swing Low = local trough (sell-side liquidity sits below it)
        if df['low'].iloc[i] == window['low'].min():
            level   = df['low'].iloc[i]
            touches = ((df['low'] >= level * 0.9999) &
                       (df['low'] <= level * 1.0001)).sum()
            sellside.append({
                'type':    'SELL_SIDE',
                'level':   round(float(level), 5),
                'time':    str(df['time'].iloc[i]),
                'touches': int(touches),
                'swept':   False,
                'note':    'Retail stop losses below — institutions may sweep here',
            })

    # Sort by touches descending (most touched = most liquidity)
    buyside  = sorted(buyside,  key=lambda x: x['touches'], reverse=True)
    sellside = sorted(sellside, key=lambda x: x['touches'], reverse=True)

    return {
        'buyside_pools':  buyside[:max_pools],
        'sellside_pools': sellside[:max_pools],
    }


def check_sweeps(pools: list, df: pd.DataFrame) -> list:
    """
    Mark pools as swept if price has already traded through them.
    A swept pool = liquidity already taken. Less useful now.
    An unswept pool = liquidity still sitting there. High value target.
    """
    result       = []
    latest_high  = df['high'].iloc[-1]
    latest_low   = df['low'].iloc[-1]

    for pool in pools:
        pool = pool.copy()
        if pool['type'] == 'BUY_SIDE':
            pool['swept'] = latest_high > pool['level']
        elif pool['type'] == 'SELL_SIDE':
            pool['swept'] = latest_low < pool['level']
        result.append(pool)
    return result


def get_nearest_pool(pools: list, current_price: float) -> dict | None:
    """Return the closest unswept liquidity pool to current price."""
    active = [p for p in pools if not p['swept']]
    if not active:
        return None
    return min(active, key=lambda p: abs(current_price - p['level']))


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"
    print(f"Detecting Liquidity Pools for {SYMBOL} H1...\n")

    df = get_candles(SYMBOL, mt5.TIMEFRAME_H1, 200)
    if df is None:
        print("No data.")
        mt5.shutdown()
        exit()

    tick          = mt5.symbol_info_tick(SYMBOL)
    current_price = tick.bid
    pools         = detect_liquidity_pools(df)

    # Check sweeps
    pools['buyside_pools']  = check_sweeps(pools['buyside_pools'],  df)
    pools['sellside_pools'] = check_sweeps(pools['sellside_pools'], df)

    nearest = get_nearest_pool(
        pools['buyside_pools'] + pools['sellside_pools'],
        current_price)

    print("=" * 58)
    print(f"  LIQUIDITY POOLS — {SYMBOL} H1")
    print(f"  Current Price : {current_price:.5f}")
    print("=" * 58)

    print(f"\n  BUY-SIDE Liquidity (above swing highs — stop hunts):")
    for p in pools['buyside_pools']:
        status = "✅ UNSWEPT" if not p['swept'] else "❌ SWEPT"
        print(f"  {status} | Level: {p['level']}"
              f" | Touches: {p['touches']}"
              f" | {p['time'][:16]}")

    print(f"\n  SELL-SIDE Liquidity (below swing lows — stop hunts):")
    for p in pools['sellside_pools']:
        status = "✅ UNSWEPT" if not p['swept'] else "❌ SWEPT"
        print(f"  {status} | Level: {p['level']}"
              f" | Touches: {p['touches']}"
              f" | {p['time'][:16]}")

    if nearest:
        print(f"\n  NEAREST UNSWEPT POOL:")
        print(f"  Type   : {nearest['type']}")
        print(f"  Level  : {nearest['level']}")
        print(f"  Touches: {nearest['touches']}")
        print(f"  Note   : {nearest['note']}")
    print("=" * 58)

    mt5.shutdown()
