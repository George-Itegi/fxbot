# =============================================================
# data_layer/tick_volume_surge.py
# PURPOSE: Detect institutional volume surges in tick data.
# When volume spikes to 2x+ the rolling average, it signals
# smart money (institutions) entering the market.
# Only trade when a surge is detected + direction confirmed.
# =============================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional
from core.logger import get_logger

log = get_logger("TICK_VOLUME_SURGE")


def detect_tick_volume_surge(tick_data: pd.DataFrame,
                             surge_multiplier: float = 2.0,
                             rolling_window: int = 50,
                             min_ticks: int = 20) -> Dict:
    """
    Detect tick volume surges by comparing recent volume against
    the rolling average. A surge indicates institutional participation.

    Args:
        tick_data: DataFrame with 'time', 'bid', 'ask', 'volume', 'side' columns
        surge_multiplier: How many times the average volume constitutes a surge (default 2.0)
        rolling_window: Number of ticks to calculate the rolling average volume (default 50)
        min_ticks: Minimum ticks required for analysis (default 20)

    Returns:
        Dict with:
            - surge_detected: bool
            - surge_ratio: float (current volume / average volume)
            - current_volume: int (tick count in latest window)
            - avg_volume: float (rolling average tick count)
            - surge_direction: 'BUY', 'SELL', 'NEUTRAL' (aggressive side during surge)
            - surge_strength: 'EXTREME', 'STRONG', 'MODERATE', 'NONE'
            - buy_pct: float (% of ticks that were buys in surge window)
            - sell_pct: float (% of ticks that were sells in surge window)
    """
    default = {
        'surge_detected': False,
        'surge_ratio': 0.0,
        'current_volume': 0,
        'avg_volume': 0.0,
        'surge_direction': 'NEUTRAL',
        'surge_strength': 'NONE',
        'buy_pct': 0.0,
        'sell_pct': 0.0,
    }

    if tick_data is None or len(tick_data) < min_ticks:
        return default

    # If 'side' column doesn't exist, try to add it
    df = tick_data.copy()
    if 'side' not in df.columns:
        # Simple bid/ask comparison fallback
        if 'bid' in df.columns and 'ask' in df.columns:
            df['side'] = df.apply(
                lambda row: 'BUY' if row['ask'] <= row['bid'] else (
                    'SELL' if row['bid'] >= row['ask'] else 'NEUTRAL'
                ), axis=1
            )
        else:
            return default

    # Use tick count as volume proxy (tick_volume from MT5)
    # Group ticks into small time buckets for smoother analysis
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        # Create 1-second buckets for surge analysis
        df['bucket'] = df['time'].dt.floor('1s')
        buckets = df.groupby('bucket').agg(
            tick_count=('side', 'count'),
            buy_count=('side', lambda x: (x == 'BUY').sum()),
            sell_count=('side', lambda x: (x == 'SELL').sum()),
            neutral_count=('side', lambda x: (x == 'NEUTRAL').sum()),
        ).reset_index()
    else:
        # No timestamps — use raw tick index windows
        buckets = pd.DataFrame({
            'tick_count': [len(df)],
            'buy_count': [(df['side'] == 'BUY').sum()],
            'sell_count': [(df['side'] == 'SELL').sum()],
            'neutral_count': [(df['side'] == 'NEUTRAL').sum()],
        })

    if len(buckets) < 2:
        # Not enough data points for meaningful rolling average
        # Return current data as-is
        current = buckets.iloc[-1]
        return {
            'surge_detected': False,
            'surge_ratio': 1.0,
            'current_volume': int(current['tick_count']),
            'avg_volume': float(current['tick_count']),
            'surge_direction': 'NEUTRAL',
            'surge_strength': 'NONE',
            'buy_pct': 0.0,
            'sell_pct': 0.0,
        }

    # Rolling average of tick counts per bucket
    buckets['rolling_avg'] = buckets['tick_count'].rolling(
        window=min(rolling_window, len(buckets)), min_periods=2
    ).mean()

    # Current (latest) bucket volume
    current = buckets.iloc[-1]
    avg = current['rolling_avg']
    current_vol = current['tick_count']

    if pd.isna(avg) or avg == 0:
        return default

    surge_ratio = round(current_vol / avg, 2)

    # Classify strength
    if surge_ratio >= 3.0:
        surge_strength = 'EXTREME'
    elif surge_ratio >= 2.0:
        surge_strength = 'STRONG'
    elif surge_ratio >= 1.5:
        surge_strength = 'MODERATE'
    else:
        surge_strength = 'NONE'

    # Determine dominant direction during the latest window
    total_active = current['buy_count'] + current['sell_count']
    if total_active > 0:
        buy_pct = round(current['buy_count'] / total_active * 100, 1)
        sell_pct = round(current['sell_count'] / total_active * 100, 1)
    else:
        buy_pct = 0.0
        sell_pct = 0.0

    # Direction: which side is more aggressive during the surge
    if buy_pct > 60:
        surge_direction = 'BUY'
    elif sell_pct > 60:
        surge_direction = 'SELL'
    else:
        surge_direction = 'NEUTRAL'

    surge_detected = surge_ratio >= surge_multiplier

    return {
        'surge_detected': surge_detected,
        'surge_ratio': surge_ratio,
        'current_volume': int(current_vol),
        'avg_volume': round(float(avg), 1),
        'surge_direction': surge_direction,
        'surge_strength': surge_strength,
        'buy_pct': buy_pct,
        'sell_pct': sell_pct,
    }


def get_candle_volume_surge(df_candles: pd.DataFrame,
                            lookback: int = 20,
                            surge_multiplier: float = 2.0) -> Dict:
    """
    Alternative: detect volume surge from candle data (M1/M5).
    Useful when tick data is unavailable.

    Args:
        df_candles: DataFrame with 'tick_volume' column from price_feed
        lookback: Number of candles for rolling average
        surge_multiplier: Minimum ratio for surge

    Returns:
        Dict with surge info (same schema as tick-based function)
    """
    default = {
        'surge_detected': False,
        'surge_ratio': 0.0,
        'current_volume': 0,
        'avg_volume': 0.0,
        'surge_direction': 'NEUTRAL',
        'surge_strength': 'NONE',
        'buy_pct': 0.0,
        'sell_pct': 0.0,
    }

    if df_candles is None or len(df_candles) < lookback + 1:
        return default

    volumes = df_candles['tick_volume'].tail(lookback + 1)
    current_vol = volumes.iloc[-1]
    avg_vol = volumes.iloc[:-1].mean()

    if avg_vol == 0:
        return default

    surge_ratio = round(current_vol / avg_vol, 2)

    if surge_ratio >= 3.0:
        surge_strength = 'EXTREME'
    elif surge_ratio >= 2.0:
        surge_strength = 'STRONG'
    elif surge_ratio >= 1.5:
        surge_strength = 'MODERATE'
    else:
        surge_strength = 'NONE'

    # Determine direction from candle body (close vs open)
    last_candle = df_candles.iloc[-1]
    if last_candle['close'] > last_candle['open']:
        surge_direction = 'BUY'
    elif last_candle['close'] < last_candle['open']:
        surge_direction = 'SELL'
    else:
        surge_direction = 'NEUTRAL'

    return {
        'surge_detected': surge_ratio >= surge_multiplier,
        'surge_ratio': surge_ratio,
        'current_volume': int(current_vol),
        'avg_volume': round(float(avg_vol), 1),
        'surge_direction': surge_direction,
        'surge_strength': surge_strength,
        'buy_pct': 0.0,
        'sell_pct': 0.0,
    }
