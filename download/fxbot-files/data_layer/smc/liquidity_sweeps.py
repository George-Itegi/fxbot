# =============================================================
# data_layer/smc/liquidity_sweeps.py
# PURPOSE: Detect liquidity sweeps — when price pierces a key
# level (stop hunt) then REVERSES. This is the actual SMC
# trade trigger. Not just pools being nearby — pools being TAKEN.
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


def detect_sweeps(df: pd.DataFrame,
                  swing_length: int = 5,
                  reversal_pips: float = 3.0,
                  pip_size: float = 0.0001) -> list:
    """
    Detect liquidity sweeps — price took stops then reversed.

    A sweep happens when:
    1. Price pierces above a swing high (triggers buy stops)
       then closes BACK BELOW it in the same or next candle
    2. Price pierces below a swing low (triggers sell stops)
       then closes BACK ABOVE it in the same or next candle

    The reversal candle after the sweep = actual entry signal.
    """
    df     = df.copy().reset_index(drop=True)
    sweeps = []

    # First find all swing points
    swing_highs = []
    swing_lows  = []

    for i in range(swing_length, len(df) - swing_length):
        window = df.iloc[i - swing_length: i + swing_length + 1]
        if df['high'].iloc[i] == window['high'].max():
            swing_highs.append((i, float(df['high'].iloc[i])))
        if df['low'].iloc[i] == window['low'].min():
            swing_lows.append((i, float(df['low'].iloc[i])))

    # Check each subsequent candle for a sweep of those levels
    for i in range(swing_length + 1, len(df)):
        candle = df.iloc[i]

        # --- Bearish Sweep (stop hunt above swing high then reverse down) ---
        for sh_idx, sh_level in swing_highs:
            if sh_idx >= i:
                continue
            # Price wick above the swing high
            pierced_above = candle['high'] > sh_level
            # But closed back below it (rejection)
            closed_below  = candle['close'] < sh_level
            reversal_size = (sh_level - candle['close']) / pip_size

            if pierced_above and closed_below and reversal_size >= reversal_pips:
                sweeps.append({
                    'type':          'BEARISH_SWEEP',
                    'time':          str(candle['time']),
                    'swept_level':   round(sh_level, 5),
                    'sweep_high':    round(float(candle['high']), 5),
                    'close':         round(float(candle['close']), 5),
                    'reversal_pips': round(reversal_size, 1),
                    'candle_idx':    i,
                    'bias':          'BEARISH',
                    'note':          'Buy stops taken above swing high — expect move DOWN',
                })
                break  # one sweep per candle max

        # --- Bullish Sweep (stop hunt below swing low then reverse up) ---
        for sl_idx, sl_level in swing_lows:
            if sl_idx >= i:
                continue
            pierced_below = candle['low'] < sl_level
            closed_above  = candle['close'] > sl_level
            reversal_size = (candle['close'] - sl_level) / pip_size

            if pierced_below and closed_above and reversal_size >= reversal_pips:
                sweeps.append({
                    'type':          'BULLISH_SWEEP',
                    'time':          str(candle['time']),
                    'swept_level':   round(sl_level, 5),
                    'sweep_low':     round(float(candle['low']), 5),
                    'close':         round(float(candle['close']), 5),
                    'reversal_pips': round(reversal_size, 1),
                    'candle_idx':    i,
                    'bias':          'BULLISH',
                    'note':          'Sell stops taken below swing low — expect move UP',
                })
                break

    return sweeps


def get_recent_sweeps(sweeps: list, n: int = 3) -> list:
    """Return the N most recent sweeps."""
    return sweeps[-n:] if sweeps else []


def get_last_sweep(sweeps: list) -> dict | None:
    """Return the single most recent sweep — the most actionable."""
    return sweeps[-1] if sweeps else None


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"
    print(f"Detecting Liquidity Sweeps for {SYMBOL} H1...\n")

    df = get_candles(SYMBOL, mt5.TIMEFRAME_H1, 200)
    if df is None:
        print("No data.")
        mt5.shutdown()
        exit()

    sym_info = mt5.symbol_info(SYMBOL)
    pip_size = sym_info.point * 10 if sym_info else 0.0001

    sweeps = detect_sweeps(df, swing_length=5,
                           reversal_pips=3.0, pip_size=pip_size)
    recent = get_recent_sweeps(sweeps, n=5)

    print("=" * 58)
    print(f"  LIQUIDITY SWEEPS — {SYMBOL} H1")
    print(f"  Total sweeps detected : {len(sweeps)}")
    print("=" * 58)

    if recent:
        print(f"\n  Last {len(recent)} Sweeps:")
        for sw in recent:
            icon = "📈" if sw['bias'] == 'BULLISH' else "📉"
            print(f"\n  {icon} {sw['type']}")
            print(f"     Time          : {sw['time'][:16]}")
            print(f"     Swept Level   : {sw['swept_level']}")
            print(f"     Reversal      : {sw['reversal_pips']} pips")
            print(f"     Note          : {sw['note']}")
    else:
        print("  No sweeps detected in last 200 candles.")

    last = get_last_sweep(sweeps)
    if last:
        print(f"\n  MOST RECENT SWEEP (trade signal):")
        print(f"  Type    : {last['type']}")
        print(f"  Time    : {last['time'][:16]}")
        print(f"  Level   : {last['swept_level']}")
        print(f"  Bias    : {last['bias']}")
        print(f"  Note    : {last['note']}")
    print("=" * 58)

    mt5.shutdown()
