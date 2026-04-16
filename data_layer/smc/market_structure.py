# =============================================================
# data_layer/smc/market_structure.py
# PURPOSE: Detect market structure — swing highs/lows,
# Higher Highs/Lows, Break of Structure (BOS),
# Change of Character (CHOCH).
# This is the FOUNDATION of all SMC analysis.
# Every other SMC module depends on this.
# Run standalone to test.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv
import os

from data_layer.momentum_velocity import get_pip_size

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


def find_swing_points(df: pd.DataFrame,
                      swing_length: int = 5) -> pd.DataFrame:
    """
    Find swing highs and swing lows.
    A swing high = candle whose high is highest in swing_length
                   candles on both left and right.
    A swing low  = candle whose low is lowest in swing_length
                   candles on both left and right.
    These are the turning points institutions react to.
    """
    df = df.copy()
    df['swing_high'] = False
    df['swing_low']  = False

    for i in range(swing_length, len(df) - swing_length):
        window_highs = df['high'].iloc[i - swing_length: i + swing_length + 1]
        window_lows  = df['low'].iloc[i  - swing_length: i + swing_length + 1]

        if df['high'].iloc[i] == window_highs.max():
            df.at[df.index[i], 'swing_high'] = True

        if df['low'].iloc[i] == window_lows.min():
            df.at[df.index[i], 'swing_low'] = True

    return df


def get_last_swings(df: pd.DataFrame, n: int = 5) -> dict:
    """
    Return the last N confirmed swing highs and lows.
    Used by other SMC modules to find key levels.
    """
    highs = df[df['swing_high'] == True].tail(n)
    lows  = df[df['swing_low']  == True].tail(n)

    return {
        'swing_highs': highs[['time', 'high']].values.tolist(),
        'swing_lows':  lows[['time', 'low']].values.tolist(),
    }


def detect_structure(df: pd.DataFrame) -> dict:
    """
    Detect market structure from swing points.
    Returns:
    - trend       : BULLISH | BEARISH | RANGING
    - last_bos    : last Break of Structure (price, direction, time)
    - last_choch  : last Change of Character (reversal signal)
    - hh_hl       : Higher High + Higher Low count (bull confirmation)
    - lh_ll       : Lower High + Lower Low count (bear confirmation)
    """
    df = find_swing_points(df)

    swing_highs = df[df['swing_high']]['high'].values
    swing_lows  = df[df['swing_low']]['low'].values
    high_times  = df[df['swing_high']]['time'].values
    low_times   = df[df['swing_low']]['time'].values

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {'trend': 'RANGING', 'bos': None, 'choch': None,
                'hh_count': 0, 'hl_count': 0,
                'lh_count': 0, 'll_count': 0}

    # Count HH/HL (bullish structure)
    hh_count = sum(1 for i in range(1, len(swing_highs))
                   if swing_highs[i] > swing_highs[i-1])
    hl_count = sum(1 for i in range(1, len(swing_lows))
                   if swing_lows[i] > swing_lows[i-1])

    # Count LH/LL (bearish structure)
    lh_count = sum(1 for i in range(1, len(swing_highs))
                   if swing_highs[i] < swing_highs[i-1])
    ll_count = sum(1 for i in range(1, len(swing_lows))
                   if swing_lows[i] < swing_lows[i-1])

    # Determine trend
    if hh_count >= 2 and hl_count >= 2:
        trend = 'BULLISH'
    elif lh_count >= 2 and ll_count >= 2:
        trend = 'BEARISH'
    else:
        trend = 'RANGING'

    # Break of Structure (BOS) — with minimum size validation
    # Only count BOS if the break is at least 5 pips significant
    current_price = df['close'].iloc[-1]
    last_sh = float(swing_highs[-1])
    last_sl = float(swing_lows[-1])
    last_sh_time = str(high_times[-1])
    last_sl_time = str(low_times[-1])
    min_break_pips = 5.0
    # v4.1 FIX: Use centralized get_pip_size() instead of hardcoded 0.0001
    pip_size = get_pip_size(df['close'].iloc[-1])

    bos = None
    bull_break = (current_price - last_sh) / pip_size
    bear_break = (last_sl - current_price) / pip_size

    if bull_break >= min_break_pips:
        bos = {'type': 'BULLISH_BOS', 'level': last_sh,
               'break_pips': round(bull_break, 1),
               'time': last_sh_time,
               'note': f'Price broke last swing high by {bull_break:.1f} pips — bull continuation'}
    elif bear_break >= min_break_pips:
        bos = {'type': 'BEARISH_BOS', 'level': last_sl,
               'break_pips': round(bear_break, 1),
               'time': last_sl_time,
               'note': f'Price broke last swing low by {bear_break:.1f} pips — bear continuation'}

    # CHOCH — only count if prior structure really shifts
    # Must have at least 2 HH before a CHOCH down, or 2 LL before CHOCH up
    choch = None
    if trend == 'BULLISH' and hh_count >= 2 and bear_break >= min_break_pips:
        choch = {'type': 'BEARISH_CHOCH', 'level': last_sl,
                 'note': 'Confirmed bull structure broken — reversal to bear likely'}
    elif trend == 'BEARISH' and ll_count >= 2 and bull_break >= min_break_pips:
        choch = {'type': 'BULLISH_CHOCH', 'level': last_sh,
                 'note': 'Confirmed bear structure broken — reversal to bull likely'}

    return {
        'trend':     trend,
        'bos':       bos,
        'choch':     choch,
        'hh_count':  hh_count,
        'hl_count':  hl_count,
        'lh_count':  lh_count,
        'll_count':  ll_count,
        'last_swing_high': round(last_sh, 5),
        'last_swing_low':  round(last_sl, 5),
    }


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"

    print(f"Detecting Market Structure for {SYMBOL}...\n")
    df = get_candles(SYMBOL, mt5.TIMEFRAME_H1, 200)

    if df is not None:
        df     = find_swing_points(df, swing_length=5)
        swings = get_last_swings(df, n=5)
        struct = detect_structure(df)

        print("=" * 52)
        print(f"  MARKET STRUCTURE — {SYMBOL} H1")
        print("=" * 52)
        print(f"  Trend         : {struct['trend']}")
        print(f"  HH count      : {struct['hh_count']}  "
              f"| HL count : {struct['hl_count']}")
        print(f"  LH count      : {struct['lh_count']}  "
              f"| LL count : {struct['ll_count']}")
        print(f"  Last Swing High: {struct['last_swing_high']}")
        print(f"  Last Swing Low : {struct['last_swing_low']}")
        print("-" * 52)

        if struct['bos']:
            b = struct['bos']
            print(f"  BOS  : {b['type']}")
            print(f"         Level : {b['level']}")
            print(f"         Note  : {b['note']}")
        else:
            print("  BOS  : None detected")

        if struct['choch']:
            c = struct['choch']
            print(f"  CHOCH: {c['type']}")
            print(f"         Level : {c['level']}")
            print(f"         Note  : {c['note']}")
        else:
            print("  CHOCH: None detected")

        print("-" * 52)
        print(f"\n  Last 5 Swing Highs:")
        for sh in swings['swing_highs']:
            print(f"    {sh[0]}  →  {sh[1]:.5f}")
        print(f"\n  Last 5 Swing Lows:")
        for sl in swings['swing_lows']:
            print(f"    {sl[0]}  →  {sl[1]:.5f}")
        print("=" * 52)

    mt5.shutdown()
