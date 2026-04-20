# =============================================================
# data_layer/vwap_calculator.py
# PURPOSE: Calculate VWAP (Volume Weighted Average Price).
# VWAP = where institutions benchmark fair value TODAY.
# Price above VWAP = bullish. Below VWAP = bearish.
# Run this file standalone to test.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
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
    """Fetch OHLCV candles from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VWAP for each candle in the DataFrame.
    VWAP = Sum(Typical Price x Volume) / Sum(Volume)
    Typical Price = (High + Low + Close) / 3
    Resets at the start of each new day automatically.
    """
    df = df.copy()

    # Typical price for each candle
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # Add a date column so VWAP resets daily
    df['date'] = df['time'].dt.date

    # Calculate cumulative VWAP per day
    df['tp_x_vol']   = df['typical_price'] * df['tick_volume']
    df['cum_tp_vol'] = df.groupby('date')['tp_x_vol'].cumsum()
    df['cum_vol']    = df.groupby('date')['tick_volume'].cumsum()

    # Avoid division by zero
    df['vwap'] = np.where(
        df['cum_vol'] > 0,
        df['cum_tp_vol'] / df['cum_vol'],
        df['typical_price']
    )
    df['vwap'] = df['vwap'].round(5)
    return df


def calculate_vwap_bands(df: pd.DataFrame,
                         multipliers: list = [1.0, 2.0]) -> pd.DataFrame:
    """
    Add VWAP standard deviation bands.
    These act like Bollinger Bands but anchored to VWAP.
    Price touching upper band = overbought relative to VWAP.
    Price touching lower band = oversold relative to VWAP.
    """
    df = df.copy()
    df['date'] = df['time'].dt.date

    # Rolling std of typical price per day
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3

    # Calculate std dev grouped by day
    df['vwap_std'] = df.groupby('date')['tp'].transform('std').fillna(0)

    for m in multipliers:
        df[f'vwap_upper_{m}'] = (df['vwap'] + m * df['vwap_std']).round(5)
        df[f'vwap_lower_{m}'] = (df['vwap'] - m * df['vwap_std']).round(5)

    return df


def get_vwap_context(symbol: str, timeframe=None,
                     candle_count: int = 200) -> dict:
    """
    Master function — returns full VWAP report for a symbol.
    This is what the bot calls every scan cycle.
    """
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M15

    df = get_candles(symbol, timeframe, candle_count)
    if df is None:
        return {}

    # Current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {}
    current_price = tick.bid

    # Pip size — use centralized get_pip_size() for correct calculation
    from data_layer.momentum_velocity import get_pip_size
    pip_size = get_pip_size(symbol)

    # Calculate VWAP and bands
    df = calculate_vwap(df)
    df = calculate_vwap_bands(df, multipliers=[1.0, 2.0])

    # Get latest values
    latest       = df.iloc[-1]
    vwap         = float(latest['vwap'])
    upper_1      = float(latest['vwap_upper_1.0'])
    lower_1      = float(latest['vwap_lower_1.0'])
    upper_2      = float(latest['vwap_upper_2.0'])
    lower_2      = float(latest['vwap_lower_2.0'])

    # Price position relative to VWAP
    pip_from_vwap = round((current_price - vwap) / pip_size, 1)

    if current_price > upper_2:
        position = "FAR_ABOVE_VWAP"
        bias     = "STRONG_BULL"
        note     = "Price far above VWAP — overbought, possible mean reversion"
    elif current_price > upper_1:
        position = "ABOVE_VWAP_BAND1"
        bias     = "BULLISH"
        note     = "Price above VWAP band 1 — bullish but extended"
    elif current_price > vwap:
        position = "ABOVE_VWAP"
        bias     = "BULLISH"
        note     = "Price above VWAP — buyers in control today"
    elif current_price > lower_1:
        position = "BELOW_VWAP"
        bias     = "BEARISH"
        note     = "Price below VWAP — sellers in control today"
    elif current_price > lower_2:
        position = "BELOW_VWAP_BAND1"
        bias     = "BEARISH"
        note     = "Price below VWAP band 1 — bearish but extended"
    else:
        position = "FAR_BELOW_VWAP"
        bias     = "STRONG_BEAR"
        note     = "Price far below VWAP — oversold, possible mean reversion"

    return {
        'symbol':        symbol,
        'current_price': round(current_price, 5),
        'vwap':          round(vwap, 5),
        'upper_band_1':  round(upper_1, 5),
        'lower_band_1':  round(lower_1, 5),
        'upper_band_2':  round(upper_2, 5),
        'lower_band_2':  round(lower_2, 5),
        'pip_from_vwap': pip_from_vwap,
        'position':      position,
        'bias':          bias,
        'note':          note,
    }


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"

    print(f"Calculating VWAP for {SYMBOL}...\n")
    result = get_vwap_context(
        symbol       = SYMBOL,
        timeframe    = mt5.TIMEFRAME_M15,
        candle_count = 200
    )

    if result:
        print("=" * 50)
        print(f"  VWAP REPORT — {result['symbol']}")
        print("=" * 50)
        print(f"  Current Price  : {result['current_price']}")
        print(f"  VWAP           : {result['vwap']}  ← Today's fair value")
        print(f"  Upper Band 1   : {result['upper_band_1']}")
        print(f"  Upper Band 2   : {result['upper_band_2']}")
        print(f"  Lower Band 1   : {result['lower_band_1']}")
        print(f"  Lower Band 2   : {result['lower_band_2']}")
        print("-" * 50)
        print(f"  Pips from VWAP : {result['pip_from_vwap']:+.1f} pips")
        print(f"  Position       : {result['position']}")
        print(f"  Bias           : {result['bias']}")
        print(f"  Note           : {result['note']}")
        print("=" * 50)
    else:
        print("Failed to calculate VWAP.")

    mt5.shutdown()
