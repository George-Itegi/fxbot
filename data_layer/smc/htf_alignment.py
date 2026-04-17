# =============================================================
# data_layer/smc/htf_alignment.py
# PURPOSE: Check Higher Timeframe alignment before any trade.
# H1 signal must agree with H4 and D1 bias.
# Prevents trading against the big picture.
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


def get_tf_bias(df: pd.DataFrame) -> str:
    """
    Get directional bias for a single timeframe using EMA + structure.
    Returns BULLISH, BEARISH, or NEUTRAL.
    """
    if df is None or len(df) < 50:
        return 'NEUTRAL'

    close  = df['close']
    ema21  = close.ewm(span=21, adjust=False).mean().iloc[-1]
    ema50  = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    price  = float(close.iloc[-1])

    bull_points = sum([price > ema21, price > ema50,
                       price > ema200, ema21 > ema50])
    bear_points = sum([price < ema21, price < ema50,
                       price < ema200, ema21 < ema50])

    if bull_points >= 3:
        return 'BULLISH'
    elif bear_points >= 3:
        return 'BEARISH'
    return 'NEUTRAL'


def check_htf_alignment(symbol: str, entry_bias: str) -> dict:
    """
    Check if H4 agrees with the entry bias. (D1 removed)
    Returns alignment score and recommendation.

    alignment: STRONG / WEAK / AGAINST
    """
    df_h4 = get_candles(symbol, mt5.TIMEFRAME_H4, 100)
    h4_bias = get_tf_bias(df_h4)

    if h4_bias == entry_bias:
        alignment = 'STRONG'
        score     = 100
        note      = f'H4 {entry_bias} — HTF alignment confirmed'
        approved  = True
    elif h4_bias == 'NEUTRAL':
        alignment = 'WEAK'
        score     = 50
        note      = f'H4 neutral — proceed with caution'
        approved  = True
    else:
        alignment = 'AGAINST'
        score     = 0
        note      = f'H4={h4_bias} — trading against HTF, SKIP'
        approved  = False

    return {
        'h4_bias':    h4_bias,
        'd1_bias':    'REMOVED',
        'entry_bias': entry_bias,
        'alignment':  alignment,
        'score':      score,
        'approved':   approved,
        'note':       note,
    }


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"
    print(f"Checking HTF Alignment for {SYMBOL}...\n")

    for bias in ['BULLISH', 'BEARISH']:
        result = check_htf_alignment(SYMBOL, bias)
        print("=" * 55)
        print(f"  HTF ALIGNMENT — {SYMBOL} | Entry: {bias}")
        print("=" * 55)
        print(f"  H4 Bias    : {result['h4_bias']}")
        print(f"  D1 Bias    : {result['d1_bias']}")
        print(f"  Alignment  : {result['alignment']}")
        print(f"  Score      : {result['score']}/100")
        print(f"  Approved   : {'✅ YES' if result['approved'] else '❌ NO'}")
        print(f"  Note       : {result['note']}")
        print()

    mt5.shutdown()
