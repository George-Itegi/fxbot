# =============================================================
# backtest/tick_simulator.py
# Simulates order flow metrics from M1 candle data.
# In the live bot, these come from real tick data.
# In backtest, we derive them from candle characteristics.
# =============================================================

import numpy as np
import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)


def simulate_order_flow(df_m1_recent: pd.DataFrame,
                        pip_size: float,
                        window_bars: int = 50) -> dict:
    """
    Simulate order flow metrics from the last N M1 bars.
    This replaces real tick data for backtesting.

    Logic:
      - Bullish candle (close > open) → net buy pressure
      - Bearish candle (close < open) → net sell pressure
      - Volume above average → surge
      - Body ratio (body/range) → how decisive the move was

    Returns dict with same schema as delta_calculator + market_scanner
    """
    if df_m1_recent is None or len(df_m1_recent) < 10:
        return _empty_flow()

    recent = df_m1_recent.tail(window_bars).copy()
    n = len(recent)

    # --- Simulate delta from candle bodies ---
    # Each M1 bar: if close > open → buy_ticks proportional to body
    #              if close < open → sell_ticks proportional to body
    # Volume acts as a multiplier for each bar's tick count
    avg_vol = recent['tick_volume'].mean()
    if avg_vol <= 0:
        return _empty_flow()

    buy_ticks = 0
    sell_ticks = 0
    neutral_ticks = 0

    for _, bar in recent.iterrows():
        body = bar['close'] - bar['open']
        bar_range = bar['high'] - bar['low']
        vol_ratio = bar['tick_volume'] / avg_vol if avg_vol > 0 else 1.0

        # Base tick count proportional to volume
        base_ticks = max(5, int(20 * vol_ratio))

        if bar_range > 0:
            body_ratio = abs(body) / bar_range
        else:
            body_ratio = 0.5

        if abs(body) < bar_range * 0.1:
            # Doji — mostly neutral
            neutral_ticks += base_ticks
            buy_ticks += int(base_ticks * 0.1)
            sell_ticks += int(base_ticks * 0.1)
        elif body > 0:
            # Bullish bar
            buy_ticks += int(base_ticks * (0.5 + 0.5 * body_ratio))
            sell_ticks += int(base_ticks * 0.5 * (1 - body_ratio))
            neutral_ticks += int(base_ticks * 0.1)
        else:
            # Bearish bar
            sell_ticks += int(base_ticks * (0.5 + 0.5 * body_ratio))
            buy_ticks += int(base_ticks * 0.5 * (1 - body_ratio))
            neutral_ticks += int(base_ticks * 0.1)

    total_ticks = buy_ticks + sell_ticks + neutral_ticks
    active_ticks = buy_ticks + sell_ticks
    delta = buy_ticks - sell_ticks

    # Bias
    if delta > 0:
        bias = 'BULLISH'
    elif delta < 0:
        bias = 'BEARISH'
    else:
        bias = 'NEUTRAL'

    # Strength
    if active_ticks == 0:
        strength = 'WEAK'
    else:
        dominance = abs(delta) / active_ticks
        if dominance >= 0.6:
            strength = 'STRONG'
        elif dominance >= 0.35:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

    # --- Full delta ---
    full_delta = {
        'delta': delta,
        'buy_ticks': buy_ticks,
        'sell_ticks': sell_ticks,
        'total_ticks': total_ticks,
        'bias': bias,
        'strength': strength,
    }

    # --- Rolling delta (last 100 equivalent → last ~5 bars worth of ticks) ---
    # Use last 5 M1 bars for rolling
    recent_5 = df_m1_recent.tail(5) if len(df_m1_recent) >= 5 else recent
    buy5 = sell5 = 0
    for _, bar in recent_5.iterrows():
        body = bar['close'] - bar['open']
        bar_range = bar['high'] - bar['low']
        vol_ratio = bar['tick_volume'] / avg_vol if avg_vol > 0 else 1.0
        base = max(5, int(20 * vol_ratio))
        body_ratio = abs(body) / bar_range if bar_range > 0 else 0.5
        if body > bar_range * 0.1:
            buy5 += int(base * (0.5 + 0.5 * body_ratio))
            sell5 += int(base * 0.5 * (1 - body_ratio))
        elif body < -bar_range * 0.1:
            sell5 += int(base * (0.5 + 0.5 * body_ratio))
            buy5 += int(base * 0.5 * (1 - body_ratio))
        else:
            buy5 += int(base * 0.3)
            sell5 += int(base * 0.3)

    roll_delta = buy5 - sell5
    if roll_delta > 0:
        roll_bias = 'BULLISH'
    elif roll_delta < 0:
        roll_bias = 'BEARISH'
    else:
        roll_bias = 'NEUTRAL'

    if (buy5 + sell5) > 0:
        roll_dom = abs(roll_delta) / (buy5 + sell5)
    else:
        roll_dom = 0
    if roll_dom >= 0.6:
        roll_strength = 'STRONG'
    elif roll_dom >= 0.35:
        roll_strength = 'MODERATE'
    else:
        roll_strength = 'WEAK'

    rolling_delta = {
        'delta': roll_delta,
        'buy_ticks': buy5,
        'sell_ticks': sell5,
        'total_ticks': buy5 + sell5,
        'bias': roll_bias,
        'strength': roll_strength,
        'window': buy5 + sell5,
    }

    # --- Order flow imbalance ---
    if active_ticks == 0:
        imbalance = 0.0
    else:
        imbalance = round((buy_ticks - sell_ticks) / active_ticks, 4)

    if imbalance > 0.1:
        of_direction = 'BUY'
    elif imbalance < -0.1:
        of_direction = 'SELL'
    else:
        of_direction = 'NEUTRAL'

    abs_imb = abs(imbalance)
    if abs_imb >= 0.6:
        imb_strength = 'EXTREME'
    elif abs_imb >= 0.4:
        imb_strength = 'STRONG'
    elif abs_imb >= 0.3:
        imb_strength = 'MODERATE'
    elif abs_imb >= 0.15:
        imb_strength = 'WEAK'
    else:
        imb_strength = 'NONE'

    order_flow_imbalance = {
        'imbalance': imbalance,
        'buy_ticks': buy_ticks,
        'sell_ticks': sell_ticks,
        'neutral_ticks': neutral_ticks,
        'total_ticks': total_ticks,
        'active_ticks': active_ticks,
        'direction': of_direction,
        'can_buy': imbalance >= 0.3,
        'can_sell': imbalance <= -0.3,
        'strength': imb_strength,
        'dominance_pct': round(
            max(buy_ticks, sell_ticks) / active_ticks * 100, 1
        ) if active_ticks > 0 else 0.0,
    }

    # --- Volume surge detection ---
    current_vol = recent.iloc[-1]['tick_volume']
    avg_vol = recent['tick_volume'].mean()
    surge_ratio = round(current_vol / avg_vol, 2) if avg_vol > 0 else 0.0
    surge_detected = surge_ratio >= 2.0

    if surge_ratio >= 3.0:
        surge_strength = 'EXTREME'
    elif surge_ratio >= 2.5:
        surge_strength = 'STRONG'
    elif surge_ratio >= 2.0:
        surge_strength = 'MODERATE'
    else:
        surge_strength = 'NONE'

    # Determine surge direction from last 3 bars
    last_3 = df_m1_recent.tail(3) if len(df_m1_recent) >= 3 else recent.tail(3)
    if last_3.iloc[-1]['close'] > last_3.iloc[0]['open']:
        surge_dir = 'BUY'
    elif last_3.iloc[-1]['close'] < last_3.iloc[0]['open']:
        surge_dir = 'SELL'
    else:
        surge_dir = 'NEUTRAL'

    volume_surge = {
        'surge_detected': surge_detected,
        'surge_ratio': surge_ratio,
        'surge_strength': surge_strength,
        'surge_direction': surge_dir,
    }

    # --- Momentum velocity (from candle data) ---
    from data_layer.momentum_velocity import calculate_candle_velocity
    momentum = calculate_candle_velocity(
        df_m1_recent, pip_size, candle_minutes=1, lookback=10
    )

    return {
        'delta': full_delta,
        'rolling_delta': rolling_delta,
        'order_flow_imbalance': order_flow_imbalance,
        'volume_surge': volume_surge,
        'momentum': momentum,
    }


def simulate_order_flow_from_full(df_m1: pd.DataFrame,
                                   pip_size: float) -> dict:
    """
    Wrapper: takes the full M1 DataFrame and simulates all
    order flow metrics for the LAST bar's moment.
    """
    return simulate_order_flow(df_m1, pip_size, window_bars=50)


def _empty_flow() -> dict:
    """Return empty/default flow metrics."""
    return {
        'delta': {'delta': 0, 'buy_ticks': 0, 'sell_ticks': 0,
                  'total_ticks': 0, 'bias': 'NEUTRAL', 'strength': 'WEAK'},
        'rolling_delta': {'delta': 0, 'buy_ticks': 0, 'sell_ticks': 0,
                          'total_ticks': 0, 'bias': 'NEUTRAL',
                          'strength': 'WEAK', 'window': 0},
        'order_flow_imbalance': {
            'imbalance': 0.0, 'buy_ticks': 0, 'sell_ticks': 0,
            'neutral_ticks': 0, 'total_ticks': 0, 'active_ticks': 0,
            'direction': 'NEUTRAL', 'can_buy': False, 'can_sell': False,
            'strength': 'NONE', 'dominance_pct': 0.0,
        },
        'volume_surge': {
            'surge_detected': False, 'surge_ratio': 0.0,
            'surge_strength': 'NONE', 'surge_direction': 'NEUTRAL',
        },
        'momentum': {
            'velocity_pips_min': 0.0, 'velocity_direction': 'FLAT',
            'is_scalpable': False, 'is_choppy': True,
            'max_price': 0.0, 'min_price': 0.0,
            'price_range_pips': 0.0, 'net_movement_pips': 0.0,
            'window_ticks': 0, 'window_duration_sec': 0.0,
        },
    }
