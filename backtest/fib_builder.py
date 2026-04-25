# =============================================================
# backtest/fib_builder.py
# Builds Fibonacci level report from historical candles.
# Detects: swing highs/lows, retracement levels, extension levels,
#          golden zone (0.618-0.786), confluence with price.
# =============================================================

import numpy as np
import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

# Standard Fibonacci retracement levels
FIB_RETRACEMENTS = {
    0.0: '0.0',
    0.236: '0.236',
    0.382: '0.382',
    0.5: '0.5',
    0.618: '0.618',
    0.786: '0.786',
    1.0: '1.0',
}

# Fibonacci extension levels (for take-profit targets)
FIB_EXTENSIONS = {
    1.0: '1.0',
    1.272: '1.272',
    1.414: '1.414',
    1.618: '1.618',
    2.0: '2.0',
    2.618: '2.618',
}

# Golden zone (most important retracement area)
GOLDEN_ZONE_LOW = 0.618
GOLDEN_ZONE_HIGH = 0.786

# Confluence zone tolerance (in pips) — how close price needs to be to a Fib level
DEFAULT_ZONE_TOLERANCE_PIPS = 3.0


def _guess_pip_size(price: float) -> float:
    """Guess pip size based on price level."""
    if price > 1000:
        return 0.1    # Gold / Silver
    elif price > 500:
        return 1.0    # Indices
    elif price > 50:
        return 0.01   # JPY pairs
    else:
        return 0.0001 # Standard forex


def _find_swing_highs(df: pd.DataFrame, lookback: int = 5,
                      max_swings: int = 5) -> list:
    """
    Find recent swing highs in the dataframe.
    A swing high is a bar where high > high of [lookback] bars on each side.
    Returns list of dicts: [{'price': float, 'index': int, 'time': datetime}, ...]
    Sorted by most recent first.
    """
    if df is None or len(df) < lookback * 2 + 1:
        return []

    swings = []
    for i in range(lookback, len(df) - lookback):
        current_high = df.iloc[i]['high']
        is_swing = True
        for j in range(1, lookback + 1):
            if df.iloc[i - j]['high'] >= current_high:
                is_swing = False
                break
            if df.iloc[i + j]['high'] >= current_high:
                is_swing = False
                break
        if is_swing:
            swings.append({
                'price': float(current_high),
                'index': i,
                'time': df.iloc[i].get('time'),
            })

    # Return most recent, max N
    swings.sort(key=lambda x: x['index'], reverse=True)
    return swings[:max_swings]


def _find_swing_lows(df: pd.DataFrame, lookback: int = 5,
                     max_swings: int = 5) -> list:
    """
    Find recent swing lows in the dataframe.
    A swing low is a bar where low < low of [lookback] bars on each side.
    Returns list of dicts: [{'price': float, 'index': int, 'time': datetime}, ...]
    Sorted by most recent first.
    """
    if df is None or len(df) < lookback * 2 + 1:
        return []

    swings = []
    for i in range(lookback, len(df) - lookback):
        current_low = df.iloc[i]['low']
        is_swing = True
        for j in range(1, lookback + 1):
            if df.iloc[i - j]['low'] <= current_low:
                is_swing = False
                break
            if df.iloc[i + j]['low'] <= current_low:
                is_swing = False
                break
        if is_swing:
            swings.append({
                'price': float(current_low),
                'index': i,
                'time': df.iloc[i].get('time'),
            })

    swings.sort(key=lambda x: x['index'], reverse=True)
    return swings[:max_swings]


def _calc_retracement_levels(swing_high: float, swing_low: float,
                             direction: str = 'UP') -> dict:
    """
    Calculate Fibonacci retracement levels between swing_high and swing_low.

    For UP trend (price going up): swing_low is the start, swing_high is the peak.
        Retracement measures how much price pulls BACK from the peak.
        0.618 level = swing_high - 0.618 * (swing_high - swing_low)

    For DOWN trend (price going down): swing_high is the start, swing_low is the trough.
        Retracement measures how much price pulls BACK from the trough.
        0.618 level = swing_low + 0.618 * (swing_high - swing_low)

    Returns dict of {level_name: price_value}
    """
    levels = {}
    diff = swing_high - swing_low

    if direction == 'UP':
        # Price was going up. Retracement levels are below the swing high.
        for ratio, name in FIB_RETRACEMENTS.items():
            levels[name] = swing_high - ratio * diff
    else:
        # Price was going down. Retracement levels are above the swing low.
        for ratio, name in FIB_RETRACEMENTS.items():
            levels[name] = swing_low + ratio * diff

    return levels


def _calc_extension_levels(swing_high: float, swing_low: float,
                           direction: str = 'UP') -> dict:
    """
    Calculate Fibonacci extension levels.
    Extensions project BEYOND the swing for take-profit targets.

    For UP trend: extensions are above the swing_high.
    For DOWN trend: extensions are below the swing_low.

    Returns dict of {level_name: price_value}
    """
    levels = {}
    diff = swing_high - swing_low

    if direction == 'UP':
        for ratio, name in FIB_EXTENSIONS.items():
            levels[name] = swing_high + (ratio - 1.0) * diff
    else:
        for ratio, name in FIB_EXTENSIONS.items():
            levels[name] = swing_low - (ratio - 1.0) * diff

    return levels


def _find_nearest_fib_level(current_price: float, fib_levels: dict,
                            pip_size: float,
                            tolerance_pips: float = DEFAULT_ZONE_TOLERANCE_PIPS) -> dict:
    """
    Check if current_price is near any Fibonacci level.
    Returns dict with nearest level info or empty dict if none within tolerance.
    """
    tolerance = tolerance_pips * pip_size
    best = None
    best_distance = float('inf')

    for name, level_price in fib_levels.items():
        distance = abs(current_price - level_price)
        if distance < tolerance and distance < best_distance:
            best = {
                'level_name': name,
                'level_price': level_price,
                'distance_pips': round(distance / pip_size, 1),
                'above': current_price >= level_price,
            }
            best_distance = distance

    return best or {}


def build_fib_report(df_h1: pd.DataFrame = None,
                     df_h4: pd.DataFrame = None,
                     df_m15: pd.DataFrame = None,
                     current_price: float = None) -> dict:
    """
    Build a Fibonacci level report from historical candles.

    Uses the most recent significant swing high/low on H4 (major) and H1 (minor)
    to draw retracement and extension levels.

    Returns dict:
        swing_high: {'price': float, 'time': ...}
        swing_low:  {'price': float, 'time': ...}
        direction:  'UP' or 'DOWN'
        retracement_levels: {name: price, ...}
        extension_levels: {name: price, ...}
        golden_zone: {'low': float, 'high': float}
        nearest_level: {level_name, level_price, distance_pips} or {}
        levels_in_zone: list of level names within tolerance of current_price
        confluence_score: int (0-20, higher = more fib confluence)
        fib_bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    """
    if df_h1 is None or len(df_h1) < 50:
        return _empty_fib()

    if current_price is None:
        current_price = float(df_h1.iloc[-1]['close'])

    pip_size = _guess_pip_size(current_price)

    # Use H4 for major swings if available, else H1
    major_df = df_h4 if df_h4 is not None and len(df_h4) >= 30 else df_h1
    lookback = 5 if major_df is df_h4 else 5

    # Find swing highs and lows
    swing_highs = _find_swing_highs(major_df, lookback=lookback, max_swings=5)
    swing_lows = _find_swing_lows(major_df, lookback=lookback, max_swings=5)

    if not swing_highs or not swing_lows:
        return _empty_fib()

    # Use the most recent completed swing (most recent swing that's NOT the current bar)
    # Find the most recent swing high and low that form a valid structure
    swing_high = swing_highs[0]
    swing_low = swing_lows[0]

    # Determine trend direction based on which swing is more recent
    if swing_high['index'] > swing_low['index']:
        # High is more recent — price was going UP, now pulling back
        direction = 'UP'
        # Swing low should be BEFORE the swing high for a valid structure
        if swing_low['index'] >= swing_high['index']:
            # Try to find an earlier swing low
            for sl in swing_lows:
                if sl['index'] < swing_high['index']:
                    swing_low = sl
                    break
            else:
                return _empty_fib()
    else:
        # Low is more recent — price was going DOWN, now pulling back
        direction = 'DOWN'
        if swing_high['index'] >= swing_low['index']:
            for sh in swing_highs:
                if sh['index'] < swing_low['index']:
                    swing_high = sh
                    break
            else:
                return _empty_fib()

    # Calculate Fibonacci levels
    retracement_levels = _calc_retracement_levels(
        swing_high['price'], swing_low['price'], direction)
    extension_levels = _calc_extension_levels(
        swing_high['price'], swing_low['price'], direction)

    # Golden zone
    if direction == 'UP':
        gz_low = retracement_levels['0.786']
        gz_high = retracement_levels['0.618']
    else:
        gz_low = retracement_levels['0.618']
        gz_high = retracement_levels['0.786']

    # Check if current price is in golden zone
    in_golden_zone = gz_low <= current_price <= gz_high

    # Find nearest Fib level to current price
    all_levels = {**retracement_levels, **extension_levels}
    nearest = _find_nearest_fib_level(current_price, all_levels, pip_size)

    # Count levels within tolerance
    tolerance = DEFAULT_ZONE_TOLERANCE_PIPS * pip_size
    levels_in_zone = []
    for name, level_price in all_levels.items():
        if abs(current_price - level_price) <= tolerance:
            levels_in_zone.append(name)

    # Confluence score (0-20)
    confluence_score = 0
    if in_golden_zone:
        confluence_score += 10
    if nearest:
        confluence_score += 5
    if len(levels_in_zone) >= 2:
        confluence_score += 5

    # Fib bias — does the Fib structure support the current direction?
    if direction == 'UP':
        if current_price <= retracement_levels['0.618']:
            fib_bias = 'BULLISH'    # Price pulled back deep — good BUY zone
        elif current_price <= retracement_levels['0.382']:
            fib_bias = 'BULLISH'    # Shallow pullback — still bullish
        else:
            fib_bias = 'NEUTRAL'
    else:
        if current_price >= retracement_levels['0.618']:
            fib_bias = 'BEARISH'    # Price retraced deep — good SELL zone
        elif current_price >= retracement_levels['0.382']:
            fib_bias = 'BEARISH'
        else:
            fib_bias = 'NEUTRAL'

    return {
        'swing_high': {'price': swing_high['price'], 'time': swing_high['time']},
        'swing_low': {'price': swing_low['price'], 'time': swing_low['time']},
        'direction': direction,
        'retracement_levels': retracement_levels,
        'extension_levels': extension_levels,
        'golden_zone': {'low': gz_low, 'high': gz_high},
        'in_golden_zone': in_golden_zone,
        'nearest_level': nearest,
        'levels_in_zone': levels_in_zone,
        'confluence_score': confluence_score,
        'fib_bias': fib_bias,
    }


def check_fib_confluence(current_price: float, direction: str,
                         fib_report: dict, pip_size: float = None) -> dict:
    """
    Check Fibonacci confluence for a specific trade direction.
    Called by strategies to see if their entry aligns with Fib levels.

    Returns dict:
        fib_bonus: int (score bonus for the strategy, 0-15)
        confluence: list of strings (to append to strategy confluence)
        fib_level_name: str or None (nearest Fib level name)
        fib_distance_pips: float (distance to nearest level)
        in_golden_zone: bool
        fib_bias_aligned: bool (Fib bias matches trade direction)
    """
    if fib_report is None or fib_report.get('confluence_score', 0) == 0:
        return {'fib_bonus': 0, 'confluence': [], 'fib_level_name': None,
                'fib_distance_pips': 0, 'in_golden_zone': False,
                'fib_bias_aligned': False}

    if pip_size is None:
        pip_size = _guess_pip_size(current_price)

    bonus = 0
    conf = []
    nearest = fib_report.get('nearest_level', {})
    in_gz = fib_report.get('in_golden_zone', False)
    fib_bias = fib_report.get('fib_bias', 'NEUTRAL')

    # Check if Fib bias aligns with trade direction
    bias_aligned = (
        (direction == 'BUY' and fib_bias == 'BULLISH') or
        (direction == 'SELL' and fib_bias == 'BEARISH')
    )

    # Golden zone confluence (strongest signal)
    if in_gz:
        bonus += 8
        conf.append(f'FIB_GOLDEN_ZONE')

    # Near a specific Fib level
    if nearest:
        level_name = nearest.get('level_name', '')
        dist = nearest.get('distance_pips', 0)

        # 0.618 and 0.786 are the most significant
        if level_name in ('0.618', '0.786'):
            bonus += 7
            conf.append(f'FIB_{level_name}_RETRACEMENT({dist}p)')
        elif level_name == '0.5':
            bonus += 5
            conf.append(f'FIB_50_RETRACEMENT({dist}p)')
        elif level_name in ('0.382', '0.236'):
            bonus += 4
            conf.append(f'FIB_{level_name}_RETRACEMENT({dist}p)')
        elif level_name == '0.0':
            bonus += 3
            conf.append(f'FIB_0.0_BASE({dist}p)')

    # Fib bias alignment
    if bias_aligned:
        bonus += 3
        conf.append(f'FIB_BIAS_ALIGNED_{fib_bias}')

    return {
        'fib_bonus': min(bonus, 15),  # Cap at 15
        'confluence': conf,
        'fib_level_name': nearest.get('level_name'),
        'fib_distance_pips': nearest.get('distance_pips', 0),
        'in_golden_zone': in_gz,
        'fib_bias_aligned': bias_aligned,
    }


def _empty_fib() -> dict:
    """Return empty Fibonacci report when not enough data."""
    return {
        'swing_high': {'price': 0, 'time': None},
        'swing_low': {'price': 0, 'time': None},
        'direction': 'NEUTRAL',
        'retracement_levels': {},
        'extension_levels': {},
        'golden_zone': {'low': 0, 'high': 0},
        'in_golden_zone': False,
        'nearest_level': {},
        'levels_in_zone': [],
        'confluence_score': 0,
        'fib_bias': 'NEUTRAL',
    }
