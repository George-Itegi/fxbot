# =============================================================
# data_layer/smart_money_score.py
# Simplified Smart Money Footprint Score (candle-only, no tick data)
#
# Combines multiple candle-based signals to estimate whether
# smart money is accumulating (BUY) or distributing (SELL).
#
# Components:
#   1. Volume surge ratio (institutional activity proxy)
#   2. Momentum velocity (rate of price change)
#   3. RSI divergence detection (price vs oscillator)
#   4. Bollinger Band rejection (price rejection at bands)
#   5. Candle body ratio (conviction vs indecision)
#
# Returns: {'score': float(-100 to +100), 'bias': str, 'confidence': int}
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

_SM_DEFAULT = {
    'score': 0.0,
    'bias': 'NEUTRAL',
    'confidence': 0,
    'components': {},
}


def calculate_smart_money_score_simple(symbol: str) -> dict:
    """
    Simplified smart money score using only candle data (no tick data required).
    Combines: volume surge + momentum velocity + RSI divergence +
    Bollinger Band rejection + candle body ratio.

    Returns:
        dict: {
            'score': float(-100 to +100),
            'bias': str (STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL),
            'confidence': int (0-100),
            'components': dict,
        }
    """
    result = dict(_SM_DEFAULT)
    components = {}
    total_score = 0.0

    try:
        from data_layer.price_feed import get_candles

        # Fetch M15 candles with all indicators pre-calculated
        df = get_candles(symbol, 'M15', 60)
        if df is None or len(df) < 30:
            log.debug(f"[SM_SCORE] Insufficient candle data for {symbol}")
            return result

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last['close'])

        # ── Component 1: Volume Surge (±20 points) ──
        vol_score, vol_details = _score_volume_surge(df, last)
        total_score += vol_score
        components['volume_surge'] = vol_details

        # ── Component 2: Momentum Velocity (±20 points) ──
        mom_score, mom_details = _score_momentum_velocity(df)
        total_score += mom_score
        components['momentum'] = mom_details

        # ── Component 3: RSI Divergence (±25 points) ──
        rsi_score, rsi_details = _score_rsi_divergence(df)
        total_score += rsi_score
        components['rsi_divergence'] = rsi_details

        # ── Component 4: Bollinger Band Rejection (±20 points) ──
        bb_score, bb_details = _score_bollinger_rejection(df, last)
        total_score += bb_score
        components['bollinger_rejection'] = bb_details

        # ── Component 5: Candle Body Ratio (±15 points) ──
        body_score, body_details = _score_candle_body(df, last)
        total_score += body_score
        components['candle_body'] = body_details

        # Clamp to [-100, +100]
        total_score = max(-100.0, min(100.0, total_score))

        # Determine bias
        if total_score >= 50:
            bias = 'STRONG_BUY'
        elif total_score >= 20:
            bias = 'BUY'
        elif total_score <= -50:
            bias = 'STRONG_SELL'
        elif total_score <= -20:
            bias = 'SELL'
        else:
            bias = 'NEUTRAL'

        # Confidence: number of components with |score| > 5
        confirming = sum(1 for k in ['volume_surge', 'rsi_divergence',
                                     'bollinger_rejection', 'candle_body']
                        if abs(components.get(k, {}).get('contribution', 0)) > 5)
        confidence = min(100, confirming * 20 + int(abs(total_score) / 5))

        result['score'] = round(total_score, 1)
        result['bias'] = bias
        result['confidence'] = confidence
        result['components'] = components

        if abs(total_score) > 40:
            log.debug(f"[SM_SCORE] {symbol}: {total_score:+.1f} ({bias}) "
                      f"conf={confidence}")

        return result

    except Exception as e:
        log.debug(f"[SM_SCORE] Error for {symbol}: {e}")
        return result


def _score_volume_surge(df, last) -> tuple:
    """
    Score volume surge: high volume on the last candle relative to average.
    +20 for bullish surge, -20 for bearish surge.
    """
    try:
        if 'vol_ma20' not in df.columns or 'tick_volume' not in df.columns:
            return 0.0, {'contribution': 0.0, 'surge_ratio': 1.0}

        vol_ma = float(last.get('vol_ma20', 0))
        current_vol = float(last.get('tick_volume', 0))

        if vol_ma <= 0:
            return 0.0, {'contribution': 0.0, 'surge_ratio': 1.0}

        surge_ratio = current_vol / vol_ma

        # Check candle direction
        candle_body = float(last['close']) - float(last['open'])
        direction = 1.0 if candle_body > 0 else -1.0 if candle_body < 0 else 0.0

        # Surge strength: sigmoid-like mapping
        if surge_ratio >= 2.5:
            magnitude = 1.0
        elif surge_ratio >= 1.8:
            magnitude = 0.7
        elif surge_ratio >= 1.3:
            magnitude = 0.3
        else:
            magnitude = 0.0

        contribution = 20.0 * magnitude * direction
        return contribution, {
            'contribution': round(contribution, 1),
            'surge_ratio': round(surge_ratio, 2),
            'direction': 'BULL' if direction > 0 else 'BEAR' if direction < 0 else 'FLAT',
        }

    except Exception:
        return 0.0, {'contribution': 0.0, 'surge_ratio': 1.0}


def _score_momentum_velocity(df) -> tuple:
    """
    Score momentum velocity: rate of price change over last N candles.
    +20 for strong upward velocity, -20 for strong downward.
    """
    try:
        recent = df.tail(10)
        if len(recent) < 5:
            return 0.0, {'contribution': 0.0, 'velocity': 0.0}

        # Average change per candle over last 5
        closes = recent['close'].astype(float).values
        changes = np.diff(closes)
        avg_change = float(np.mean(changes[-5:]))

        # Normalize by ATR
        atr = float(recent.iloc[-1].get('atr', 0))
        if atr <= 0:
            return 0.0, {'contribution': 0.0, 'velocity': 0.0}

        # Velocity as fraction of ATR
        velocity = avg_change / atr

        # Clamp and map to ±20
        clamped = max(-1.0, min(1.0, velocity * 5))
        contribution = 20.0 * clamped

        return contribution, {
            'contribution': round(contribution, 1),
            'velocity': round(velocity, 4),
            'avg_change': round(avg_change, 6),
        }

    except Exception:
        return 0.0, {'contribution': 0.0, 'velocity': 0.0}


def _score_rsi_divergence(df) -> tuple:
    """
    Score RSI divergence: price makes new high/low but RSI doesn't confirm.
    Bullish divergence = price lower low, RSI higher low → +25
    Bearish divergence = price higher high, RSI lower high → -25
    """
    try:
        if 'rsi' not in df.columns:
            return 0.0, {'contribution': 0.0, 'divergence': 'NONE'}

        recent = df.tail(30).copy()
        if len(recent) < 20:
            return 0.0, {'contribution': 0.0, 'divergence': 'NONE'}

        closes = recent['close'].astype(float).values
        rsi_vals = recent['rsi'].astype(float).values

        # Split into two halves
        mid = len(closes) // 2
        first_half_close = closes[:mid]
        second_half_close = closes[mid:]
        first_half_rsi = rsi_vals[:mid]
        second_half_rsi = rsi_vals[mid:]

        # Find extremes
        first_price_high = float(np.max(first_half_close))
        first_price_low = float(np.min(first_half_close))
        second_price_high = float(np.max(second_half_close))
        second_price_low = float(np.min(second_half_close))

        # RSI extremes (in the vicinity of price extremes)
        first_rsi_at_high = float(np.max(first_half_rsi))
        first_rsi_at_low = float(np.min(first_half_rsi))
        second_rsi_at_high = float(np.max(second_half_rsi))
        second_rsi_at_low = float(np.min(second_half_rsi))

        contribution = 0.0
        divergence = 'NONE'

        # Bearish divergence: price higher high, RSI lower high
        if second_price_high > first_price_high and second_rsi_at_high < first_rsi_at_high:
            price_diff_pct = (second_price_high - first_price_high) / first_price_high * 100
            rsi_drop = first_rsi_at_high - second_rsi_at_high
            strength = min(1.0, (price_diff_pct * 5 + rsi_drop) / 20)
            contribution = -25.0 * strength
            divergence = 'BEARISH'

        # Bullish divergence: price lower low, RSI higher low
        elif second_price_low < first_price_low and second_rsi_at_low > first_rsi_at_low:
            price_diff_pct = (first_price_low - second_price_low) / first_price_low * 100
            rsi_gain = second_rsi_at_low - first_rsi_at_low
            strength = min(1.0, (price_diff_pct * 5 + rsi_gain) / 20)
            contribution = 25.0 * strength
            divergence = 'BULLISH'

        return contribution, {
            'contribution': round(contribution, 1),
            'divergence': divergence,
            'current_rsi': round(float(rsi_vals[-1]), 1),
        }

    except Exception:
        return 0.0, {'contribution': 0.0, 'divergence': 'NONE'}


def _score_bollinger_rejection(df, last) -> tuple:
    """
    Score Bollinger Band rejection: price touches band and reverses.
    +20 for rejection at lower band (bullish),
    -20 for rejection at upper band (bearish).
    """
    try:
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            return 0.0, {'contribution': 0.0, 'rejection': 'NONE'}

        bb_upper = float(last.get('bb_upper', 0))
        bb_lower = float(last.get('bb_lower', 0))
        bb_mid = float(last.get('bb_mid', 0))
        close = float(last['close'])
        high = float(last['high'])
        low = float(last['low'])
        open_price = float(last['open'])

        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return 0.0, {'contribution': 0.0, 'rejection': 'NONE'}

        contribution = 0.0
        rejection = 'NONE'

        # Bullish rejection: price touches or breaches lower band, closes above it
        proximity_lower = (low - bb_lower) / bb_width
        if proximity_lower <= 0.05 and close > bb_lower:
            # Wick touched lower band but closed above = bullish rejection
            wick_ratio = (bb_lower - low) / max(high - low, 0.0001)
            strength = min(1.0, wick_ratio * 2 + (close - open_price) / bb_width * 5)
            strength = max(0.0, strength)
            contribution = 20.0 * strength
            rejection = 'BULLISH'

        # Bearish rejection: price touches or breaches upper band, closes below it
        proximity_upper = (bb_upper - high) / bb_width
        if proximity_upper <= 0.05 and close < bb_upper:
            wick_ratio = (high - bb_upper) / max(high - low, 0.0001)
            strength = min(1.0, wick_ratio * 2 + (open_price - close) / bb_width * 5)
            strength = max(0.0, strength)
            contribution = -20.0 * strength
            rejection = 'BEARISH'

        return contribution, {
            'contribution': round(contribution, 1),
            'rejection': rejection,
            'bb_width_pct': round(bb_width / bb_mid * 100, 2) if bb_mid > 0 else 0,
        }

    except Exception:
        return 0.0, {'contribution': 0.0, 'rejection': 'NONE'}


def _score_candle_body(df, last) -> tuple:
    """
    Score candle body ratio: large body = conviction, small body = indecision.
    Direction of conviction matters: bullish body → +15, bearish → -15.
    """
    try:
        close = float(last['close'])
        open_price = float(last['open'])
        high = float(last['high'])
        low = float(last['low'])

        candle_range = high - low
        if candle_range <= 0:
            return 0.0, {'contribution': 0.0, 'body_ratio': 0.0}

        body = abs(close - open_price)
        body_ratio = body / candle_range  # 0.0 = doji, 1.0 = full body

        # Direction
        direction = 1.0 if close > open_price else -1.0 if close < open_price else 0.0

        # Map body_ratio to conviction
        # Doji (0-0.1) = indecision, no contribution
        # Medium (0.1-0.5) = moderate
        # Strong (0.5-1.0) = high conviction
        if body_ratio <= 0.1:
            magnitude = 0.0  # Indecision — no conviction signal
        elif body_ratio <= 0.3:
            magnitude = 0.3
        elif body_ratio <= 0.6:
            magnitude = 0.6
        else:
            magnitude = 1.0

        # Also consider volume confirmation if available
        if 'vol_ma20' in df.columns and 'tick_volume' in df.columns:
            vol_ma = float(last.get('vol_ma20', 0))
            current_vol = float(last.get('tick_volume', 0))
            if vol_ma > 0 and current_vol < vol_ma * 0.7:
                # Low volume on a body candle = weak conviction
                magnitude *= 0.5

        contribution = 15.0 * magnitude * direction

        return contribution, {
            'contribution': round(contribution, 1),
            'body_ratio': round(body_ratio, 3),
            'direction': 'BULL' if direction > 0 else 'BEAR' if direction < 0 else 'DOJI',
        }

    except Exception:
        return 0.0, {'contribution': 0.0, 'body_ratio': 0.0}
