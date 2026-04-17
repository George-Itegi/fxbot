# =============================================================
# data_layer/momentum_velocity.py
# PURPOSE: Measure price momentum velocity in pips/minute.
# Only scalp when velocity > 1.0 pips/min (strong directional move).
# Skip choppy/ranging markets when velocity < 0.5 pips/min.
# This avoids 50%+ of noise trades and catches moves at origin.
# =============================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from core.logger import get_logger

log = get_logger("MOMENTUM_VELOCITY")

# --- Default thresholds ---
DEFAULT_MIN_SCALP_VELOCITY = 1.0    # pips/min — minimum for scalping entry
DEFAULT_CHOPPY_VELOCITY    = 0.5    # pips/min — below this = choppy/ranging, skip
DEFAULT_VELOCITY_WINDOW    = 60     # seconds — measurement window for velocity


def calculate_momentum_velocity(tick_data: pd.DataFrame,
                                pip_size: float,
                                window_seconds: int = DEFAULT_VELOCITY_WINDOW,
                                min_ticks: int = 10) -> Dict:
    """
    Calculate real-time momentum velocity from tick data.
    Measures price movement rate in pips per minute.

    Args:
        tick_data: DataFrame with 'time' and 'bid' columns
        pip_size: Pip size for the symbol (0.0001 or 0.01)
        window_seconds: How many seconds of recent data to analyze
        min_ticks: Minimum ticks required for analysis

    Returns:
        Dict with:
            - velocity_pips_min: float (pips moved per minute)
            - velocity_direction: 'UP', 'DOWN', 'FLAT'
            - is_scalpable: bool (velocity > 1.0 pips/min)
            - is_choppy: bool (velocity < 0.5 pips/min)
            - max_price: float (highest price in window)
            - min_price: float (lowest price in window)
            - price_range_pips: float (total range in window)
            - net_movement_pips: float (net directional movement)
            - window_ticks: int (number of ticks in window)
            - window_duration_sec: float (actual duration of window)
    """
    default = {
        'velocity_pips_min': 0.0,
        'velocity_direction': 'FLAT',
        'is_scalpable': False,
        'is_choppy': True,
        'max_price': 0.0,
        'min_price': 0.0,
        'price_range_pips': 0.0,
        'net_movement_pips': 0.0,
        'window_ticks': 0,
        'window_duration_sec': 0.0,
    }

    if tick_data is None or len(tick_data) < min_ticks:
        return default

    df = tick_data.copy()
    if 'time' not in df.columns or 'bid' not in df.columns:
        return default

    df['time'] = pd.to_datetime(df['time'])
    latest_time = df['time'].iloc[-1]
    cutoff = latest_time - pd.Timedelta(seconds=window_seconds)

    # Filter to recent window
    window_df = df[df['time'] >= cutoff]

    if len(window_df) < min_ticks:
        return default

    # Calculate price statistics
    max_price = window_df['bid'].max()
    min_price = window_df['bid'].min()
    first_price = window_df['bid'].iloc[0]
    last_price = window_df['bid'].iloc[-1]
    price_range_pips = (max_price - min_price) / pip_size
    net_movement_pips = (last_price - first_price) / pip_size

    # Actual window duration
    window_duration = (window_df['time'].iloc[-1] - window_df['time'].iloc[0]).total_seconds()
    if window_duration <= 0:
        window_duration = 1.0  # Avoid division by zero

    # Velocity: pips per minute
    # Use the absolute range (not net) to measure true momentum energy
    velocity_pips_min = round((price_range_pips / window_duration) * 60, 2)

    # Determine direction from net movement
    if net_movement_pips > pip_size:  # > 1 pip net move
        velocity_direction = 'UP'
    elif net_movement_pips < -pip_size:
        velocity_direction = 'DOWN'
    else:
        velocity_direction = 'FLAT'

    # Classification
    is_scalpable = velocity_pips_min >= DEFAULT_MIN_SCALP_VELOCITY
    is_choppy = velocity_pips_min <= DEFAULT_CHOPPY_VELOCITY

    return {
        'velocity_pips_min': velocity_pips_min,
        'velocity_direction': velocity_direction,
        'is_scalpable': is_scalpable,
        'is_choppy': is_choppy,
        'max_price': max_price,
        'min_price': min_price,
        'price_range_pips': round(price_range_pips, 1),
        'net_movement_pips': round(net_movement_pips, 1),
        'window_ticks': len(window_df),
        'window_duration_sec': round(window_duration, 1),
    }


def calculate_candle_velocity(df_candles: pd.DataFrame,
                              pip_size: float,
                              candle_minutes: int = 1,
                              lookback: int = 10) -> Dict:
    """
    Alternative: calculate velocity from candle data when tick data
    is unavailable. Uses candle ranges over recent lookback.

    Args:
        df_candles: DataFrame with 'high', 'low', 'close', 'open' columns
        pip_size: Pip size for the symbol
        candle_minutes: Minutes per candle (for velocity calculation)
        lookback: Number of recent candles to analyze

    Returns:
        Dict with velocity info (same schema as tick-based function)
    """
    default = {
        'velocity_pips_min': 0.0,
        'velocity_direction': 'FLAT',
        'is_scalpable': False,
        'is_choppy': True,
        'max_price': 0.0,
        'min_price': 0.0,
        'price_range_pips': 0.0,
        'net_movement_pips': 0.0,
        'window_ticks': 0,
        'window_duration_sec': 0.0,
    }

    if df_candles is None or len(df_candles) < lookback:
        return default

    recent = df_candles.tail(lookback)
    max_price = recent['high'].max()
    min_price = recent['low'].min()
    first_close = recent['close'].iloc[0]
    last_close = recent['close'].iloc[-1]

    price_range_pips = (max_price - min_price) / pip_size
    net_movement_pips = (last_close - first_close) / pip_size

    # Total time in minutes
    total_minutes = lookback * candle_minutes
    if total_minutes <= 0:
        total_minutes = 1.0

    # Average velocity: total range / total time in minutes
    velocity_pips_min = round(price_range_pips / total_minutes, 2)

    if net_movement_pips > pip_size:
        velocity_direction = 'UP'
    elif net_movement_pips < -pip_size:
        velocity_direction = 'DOWN'
    else:
        velocity_direction = 'FLAT'

    is_scalpable = velocity_pips_min >= DEFAULT_MIN_SCALP_VELOCITY
    is_choppy = velocity_pips_min <= DEFAULT_CHOPPY_VELOCITY

    return {
        'velocity_pips_min': velocity_pips_min,
        'velocity_direction': velocity_direction,
        'is_scalpable': is_scalpable,
        'is_choppy': is_choppy,
        'max_price': max_price,
        'min_price': min_price,
        'price_range_pips': round(price_range_pips, 1),
        'net_movement_pips': round(net_movement_pips, 1),
        'window_ticks': lookback,
        'window_duration_sec': total_minutes * 60,
    }


def get_pip_size(symbol: str) -> float:
    """Get pip size for a symbol. Matches order_manager._get_pip_point exactly."""
    sym = str(symbol).upper()
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        return 1.0  # Index points
    if "XAU" in sym:
        return 0.1   # Gold
    if "XAG" in sym:
        return 0.01  # Silver
    if any(x in sym for x in ["WTI", "BRN"]):
        return 0.01  # Oil
    if any(x in sym for x in ["JPY", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]):
        return 0.01  # JPY pairs
    return 0.0001  # Standard forex
