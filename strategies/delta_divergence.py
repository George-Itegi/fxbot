# =============================================================
# strategies/delta_divergence.py
# Strategy 8: Delta Divergence
# Detects when price makes a new high/low but order flow delta
# weakens, indicating a fake breakout / trapped retail positions.
# This catches institutional reversal setups at their origin.
#
# BUY  when: price makes new low BUT delta strengthens (bullish)
# SELL when: price makes new high BUT delta weakens (bearish)
#
# Win rate target: 60-65%
# Best session: LONDON_SESSION, NY_LONDON_OVERLAP, NY_SESSION
# Best state:  BREAKOUT_REJECTED, REVERSAL_RISK, BALANCED
# =============================================================

import pandas as pd
import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "DELTA_DIVERGENCE"
MIN_SCORE     = 70
VERSION       = "1.0"

# --- Delta Divergence Parameters ---
LOOKBACK_SWINGS = 20       # Candles to look back for swing highs/lows (lowered from 50)
DELTA_LOOKBACK  = 50       # Ticks for rolling delta comparison
MIN_SWING_PIPS  = 2.0      # Minimum swing size to consider (lowered from 3.0)
DELTA_WEAKEN_THRESHOLD = 0.3  # Delta must weaken by this ratio (lowered from 0.4)


def _get_pip_size(price: float) -> float:
    """Return pip size for a symbol based on its price."""
    if price > 500:
        return 1.0
    elif price > 50:
        return 0.01
    else:
        return 0.0001


def _find_swing_highs_lows(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Find recent swing highs and swing lows in candle data.
    A swing high: candle whose high is higher than N candles on each side.
    A swing low: candle whose low is lower than N candles on each side.
    
    Returns:
        dict with 'swing_highs' and 'swing_lows' lists
        Each entry: {'price': float, 'index': int, 'pips_from_current': float}
    """
    if df is None or len(df) < lookback * 2 + 1:
        return {"swing_highs": [], "swing_lows": []}
    
    recent = df.tail(lookback * 2 + 1).copy()
    current_close = float(recent.iloc[-1]['close'])
    pip_size = _get_pip_size(current_close)
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(recent) - lookback):
        candle = recent.iloc[i]
        window_left = recent.iloc[i - lookback:i]
        window_right = recent.iloc[i + 1:i + lookback + 1]
        
        # Check swing high
        if candle['high'] >= window_left['high'].max() and \
           candle['high'] >= window_right['high'].max():
            swing_highs.append({
                'price': float(candle['high']),
                'index': int(i),
                'time': str(candle.get('time', '')),
                'pips_from_current': round((float(candle['high']) - current_close) / pip_size, 1),
            })
        
        # Check swing low
        if candle['low'] <= window_left['low'].min() and \
           candle['low'] <= window_right['low'].min():
            swing_lows.append({
                'price': float(candle['low']),
                'index': int(i),
                'time': str(candle.get('time', '')),
                'pips_from_current': round((current_close - float(candle['low'])) / pip_size, 1),
            })
    
    # Keep most recent (closest to current price)
    swing_highs = sorted(swing_highs, key=lambda x: x['pips_from_current'])[:3]
    swing_lows = sorted(swing_lows, key=lambda x: x['pips_from_current'])[:3]
    
    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def _detect_bearish_divergence(df: pd.DataFrame, swings: dict,
                                delta_at_swings: list,
                                current_delta: float) -> dict:
    """
    Bearish divergence: price makes higher high BUT delta weakens.
    Indicates sellers absorbing buying pressure — fake breakout.
    Returns divergence info or None.
    """
    if len(swings['swing_highs']) < 2:
        return None
    
    highs = swings['swing_highs']
    pip_size = _get_pip_size(float(df.iloc[-1]['close']))
    
    # Need at least 2 swing highs: one previous, one recent
    prev_high = highs[-2] if len(highs) >= 2 else None
    curr_high = highs[-1]
    
    if prev_high is None:
        return None
    
    # Price must be making a HIGHER high (bearish divergence condition)
    if curr_high['price'] <= prev_high['price']:
        return None
    
    swing_range = (curr_high['price'] - prev_high['price']) / pip_size
    if swing_range < MIN_SWING_PIPS:
        return None
    
    # Delta must be WEAKENING despite higher price
    # If delta was +100 at previous high but only +30 at current high = divergence
    # We use the delta_at_swings list (delta values at the time of each swing)
    if len(delta_at_swings) < 2:
        # Fallback: use current rolling delta vs a simple heuristic
        if current_delta <= 0:
            # Current delta is negative while price makes higher high = STRONG divergence
            return {
                'type': 'BEARISH',
                'prev_high': prev_high['price'],
                'curr_high': curr_high['price'],
                'swing_range_pips': swing_range,
                'delta_strength': 'EXTREME',
                'description': f"Higher high +{swing_range:.1f}p but delta negative"
            }
        return None
    
    prev_delta = delta_at_swings[-2]
    curr_delta = delta_at_swings[-1]
    
    # Bearish: price up, delta down
    if curr_delta < prev_delta:
        if prev_delta > 0:
            weaken_ratio = 1.0 - (curr_delta / prev_delta) if prev_delta != 0 else 1.0
        else:
            weaken_ratio = abs(curr_delta - prev_delta) / max(abs(prev_delta), 1)
        
        strength = 'MODERATE'
        if curr_delta < 0:
            strength = 'EXTREME'
        elif weaken_ratio > 0.6:
            strength = 'STRONG'
        
        return {
            'type': 'BEARISH',
            'prev_high': prev_high['price'],
            'curr_high': curr_high['price'],
            'swing_range_pips': swing_range,
            'delta_strength': strength,
            'weaken_ratio': round(weaken_ratio, 2),
            'description': f"Higher high +{swing_range:.1f}p, delta {prev_delta} -> {curr_delta}"
        }
    
    return None


def _detect_bullish_divergence(df: pd.DataFrame, swings: dict,
                                delta_at_swings: list,
                                current_delta: float) -> dict:
    """
    Bullish divergence: price makes lower low BUT delta strengthens.
    Indicates buyers absorbing selling pressure — potential reversal.
    Returns divergence info or None.
    """
    if len(swings['swing_lows']) < 2:
        return None
    
    lows = swings['swing_lows']
    pip_size = _get_pip_size(float(df.iloc[-1]['close']))
    
    prev_low = lows[-2] if len(lows) >= 2 else None
    curr_low = lows[-1]
    
    if prev_low is None:
        return None
    
    # Price must be making a LOWER low (bullish divergence condition)
    if curr_low['price'] >= prev_low['price']:
        return None
    
    swing_range = (prev_low['price'] - curr_low['price']) / pip_size
    if swing_range < MIN_SWING_PIPS:
        return None
    
    # Delta must be STRENGTHENING despite lower price
    if len(delta_at_swings) < 2:
        if current_delta >= 0:
            return {
                'type': 'BULLISH',
                'prev_low': prev_low['price'],
                'curr_low': curr_low['price'],
                'swing_range_pips': swing_range,
                'delta_strength': 'EXTREME',
                'description': f"Lower low -{swing_range:.1f}p but delta positive"
            }
        return None
    
    prev_delta = delta_at_swings[-2]
    curr_delta = delta_at_swings[-1]
    
    # Bullish: price down, delta up
    if curr_delta > prev_delta:
        if prev_delta < 0:
            strengthen_ratio = 1.0 - (curr_delta / prev_delta) if prev_delta != 0 else 1.0
        else:
            strengthen_ratio = abs(curr_delta - prev_delta) / max(abs(prev_delta), 1)
        
        strength = 'MODERATE'
        if curr_delta > 0:
            strength = 'EXTREME'
        elif strengthen_ratio > 0.6:
            strength = 'STRONG'
        
        return {
            'type': 'BULLISH',
            'prev_low': prev_low['price'],
            'curr_low': curr_low['price'],
            'swing_range_pips': swing_range,
            'delta_strength': strength,
            'strengthen_ratio': round(strengthen_ratio, 2),
            'description': f"Lower low -{swing_range:.1f}p, delta {prev_delta} -> {curr_delta}"
        }
    
    return None


def evaluate(symbol: str,
             df_m1: pd.DataFrame = None,
             df_m5: pd.DataFrame = None,
             df_m15: pd.DataFrame = None,
             df_h1: pd.DataFrame = None,
             smc_report: dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None,
             relaxed: bool = False) -> dict | None:
    """
    Fires when a delta divergence is detected:
    - BEARISH: price higher high + delta weakening = SELL
    - BULLISH: price lower low + delta strengthening = BUY
    
    This catches fake breakouts where retail is trapped and
    institutions are positioning for reversal.
    
    Entry criteria:
      1. Price makes a new swing high/low (breakout attempt)
      2. Order flow delta opposes the price move (divergence)
      3. Volume surge confirms institutional activity
      4. M15 StochRSI supports reversal (overbought/oversold)
      5. Premium/Discount zone confirms (premium = sell, discount = buy)
    """
    # ── Data validation ─────────────────────────────────────
    if df_m5 is None or df_m15 is None:
        return None
    if len(df_m5) < 60 or len(df_m15) < 50:
        return None
    
    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size
    
    if atr_pips < 2.0:
        return None
    
    # ── Get Delta Data ──────────────────────────────────────
    if not market_report:
        return None
    
    rolling_delta = market_report.get('rolling_delta', {})
    full_delta    = market_report.get('delta', {})
    current_delta = rolling_delta.get('delta', 0)
    delta_bias    = rolling_delta.get('bias', 'NEUTRAL')
    delta_strength = rolling_delta.get('strength', 'WEAK')
    of_imbalance  = market_report.get('order_flow_imbalance', {})
    volume_surge  = market_report.get('volume_surge', {})
    
    # ── Find Swing Highs/Lows on M15 ────────────────────────
    swing_lb = 3 if relaxed else 5  # Relaxed: tighter window, more swings
    swings = _find_swing_highs_lows(df_m15, lookback=swing_lb)
    
    # Simplified delta-at-swings: use current delta and delta_strength
    # In a full implementation, we'd store delta values at each swing point
    # For now, we detect divergence using current delta state + swing patterns
    delta_values = [
        current_delta,
        full_delta.get('delta', 0),
    ]
    
    score = 0
    confluence = []
    
    # Relaxed: wider proximity for swing+delta alternative
    swing_proximity = 10 if relaxed else 5
    
    # ── Detect Bearish Divergence (SELL) ────────────────────
    bear_div = _detect_bearish_divergence(df_m15, swings, delta_values, current_delta)
    
    # ── Detect Bullish Divergence (BUY) ─────────────────────
    bull_div = _detect_bullish_divergence(df_m15, swings, delta_values, current_delta)
    
    # Pick the strongest divergence
    divergence = None
    direction = None
    
    if bear_div and (bull_div is None or bear_div.get('delta_strength') == 'EXTREME'):
        divergence = bear_div
        direction = "SELL"
    elif bull_div:
        divergence = bull_div
        direction = "BUY"
    elif bear_div:
        divergence = bear_div
        direction = "SELL"
    else:
        # No divergence detected — try an alternative approach
        # Check: price at/near recent swing extreme + delta opposes
        if swings['swing_highs'] and delta_bias == "BEARISH":
            nearest_high = swings['swing_highs'][0]
            dist = nearest_high['pips_from_current']
            if dist < swing_proximity:  # Price close to or above swing high
                direction = "SELL"
                divergence = {
                    'type': 'BEARISH',
                    'delta_strength': 'MODERATE',
                    'description': f"Near swing high ({dist:.1f}p) + bearish delta",
                    'swing_range_pips': dist,
                }
        elif swings['swing_lows'] and delta_bias == "BULLISH":
            nearest_low = swings['swing_lows'][0]
            dist = nearest_low['pips_from_current']
            if dist < swing_proximity:
                direction = "BUY"
                divergence = {
                    'type': 'BULLISH',
                    'delta_strength': 'MODERATE',
                    'description': f"Near swing low ({dist:.1f}p) + bullish delta",
                    'swing_range_pips': dist,
                }
    
    if divergence is None or direction is None:
        return None
    
    div_strength = divergence.get('delta_strength', 'MODERATE')
    swing_pips = divergence.get('swing_range_pips', 0)
    
    # ── Score: Divergence Strength ──────────────────────────
    if div_strength == 'EXTREME':
        score += 35
        confluence.append("DIVERGENCE_EXTREME")
    elif div_strength == 'STRONG':
        score += 25
        confluence.append("DIVERGENCE_STRONG")
    else:
        score += 15
        confluence.append("DIVERGENCE_MODERATE")
    
    confluence.append(divergence['description'])
    
    # ── Volume Surge Confirmation ───────────────────────────
    if volume_surge.get('surge_detected', False):
        score += 15
        confluence.append(f"VOL_SURGE_{volume_surge.get('surge_ratio', 0)}x")
        # Surge direction should oppose the price move (confirms institutions fading it)
        surge_dir = volume_surge.get('surge_direction', '')
        if direction == "SELL" and surge_dir == "BUY":
            # Surge is buying but we want to sell = institutional absorption
            score += 5
            confluence.append("SURGE_ABSORPTION")
        elif direction == "BUY" and surge_dir == "SELL":
            score += 5
            confluence.append("SURGE_ABSORPTION")
    
    # ── Order Flow Imbalance ───────────────────────────────
    imb = of_imbalance.get('imbalance', 0)
    imb_strength = of_imbalance.get('strength', 'NONE')
    
    if direction == "BUY":
        if imb > 0.15:
            score += 10
            confluence.append(f"OF_BULL_{imb:+.2f}")
        if of_imbalance.get('can_buy', False):
            score += 5
    elif direction == "SELL":
        if imb < -0.15:
            score += 10
            confluence.append(f"OF_BEAR_{imb:+.2f}")
        if of_imbalance.get('can_sell', False):
            score += 5
    
    # ── StochRSI Reversal Confirmation (M15) ────────────────
    if df_m15 is not None and len(df_m15) >= 3:
        stoch_k = float(df_m15.iloc[-1].get('stoch_rsi_k', 50))
        prev_k  = float(df_m15.iloc[-2].get('stoch_rsi_k', 50))
        
        if direction == "SELL" and stoch_k > 70:
            score += 10
            confluence.append("STOCHRSI_OVERBOUGHT")
            # StochRSI turning down from overbought = extra confirmation
            if prev_k > stoch_k:
                score += 5
                confluence.append("STOCHRSI_TURNING_DOWN")
        elif direction == "BUY" and stoch_k < 30:
            score += 10
            confluence.append("STOCHRSI_OVERSOLD")
            if prev_k < stoch_k:
                score += 5
                confluence.append("STOCHRSI_TURNING_UP")
    
    # ── Premium/Discount Confluence ────────────────────────
    if smc_report:
        pd_info = smc_report.get('premium_discount', {})
        pd_zone = pd_info.get('zone', '')
        
        if direction == "SELL" and "PREMIUM" in pd_zone:
            score += 10
            confluence.append("PREMIUM_ZONE_SELL")
        elif direction == "BUY" and "DISCOUNT" in pd_zone:
            score += 10
            confluence.append("DISCOUNT_ZONE_BUY")
    
    # ── M5 Candle Confirmation ──────────────────────────────
    if df_m5 is not None and len(df_m5) >= 3:
        m5_last = df_m5.iloc[-1]
        m5_prev = df_m5.iloc[-2]
        m5_body = m5_last['close'] - m5_last['open']
        m5_range = m5_last['high'] - m5_last['low']
        
        if m5_range > 0:
            body_ratio = abs(m5_body) / m5_range
            if body_ratio > 0.6:
                aligned = (direction == "SELL" and m5_body < 0) or \
                          (direction == "BUY" and m5_body > 0)
                if aligned:
                    score += 5
                    confluence.append("M5_REVERSAL_CANDLE")
    
    # ── Momentum Check ──────────────────────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        vel_dir = momentum.get('velocity_direction', 'FLAT')
        velocity = momentum.get('velocity_pips_min', 0)
        
        if direction == "SELL" and vel_dir == "UP":
            # Price still moving up but we expect reversal
            # Higher velocity = more momentum to reverse into = bigger potential
            if velocity > 1.0:
                score += 5
                confluence.append("HIGH_MOMENTUM_REVERSAL")
        elif direction == "BUY" and vel_dir == "DOWN":
            if velocity > 1.0:
                score += 5
                confluence.append("HIGH_MOMENTUM_REVERSAL")
    
    # ── Score threshold ─────────────────────────────────────
    min_confluence = 3 if relaxed else 5
    min_score = (MIN_SCORE - 20) if relaxed else MIN_SCORE  # Relaxed: 50 vs 70

    if len(confluence) < min_confluence:
        return None
    if score < min_score:
        return None
    
    # ── Calculate SL/TP ─────────────────────────────────────
    # SL: below the swing extreme (recent swing high for sell, swing low for buy)
    # TP: based on ATR multiple — target the opposite extreme
    
    sl_atr_mult = 1.0
    tp_atr_mult = 1.5
    
    if direction == "SELL":
        # SL above the recent swing high
        if swings['swing_highs']:
            ref_price = swings['swing_highs'][0]['price']
            sl_pips = max(5.0, round((ref_price - close_price) / pip_size + atr_pips * 0.3, 1))
        else:
            sl_pips = round(atr_pips * sl_atr_mult, 1)
        
        sl_price  = round(close_price + sl_pips * pip_size, 5)
        tp1_price = round(close_price - atr_pips * tp_atr_mult * pip_size, 5)
        tp2_price = round(close_price - atr_pips * 2.5 * pip_size, 5)
        tp1_pips  = round(atr_pips * tp_atr_mult, 1)
        tp2_pips  = round(atr_pips * 2.5, 1)
    else:  # BUY
        if swings['swing_lows']:
            ref_price = swings['swing_lows'][0]['price']
            sl_pips = max(5.0, round((close_price - ref_price) / pip_size + atr_pips * 0.3, 1))
        else:
            sl_pips = round(atr_pips * sl_atr_mult, 1)
        
        sl_price  = round(close_price - sl_pips * pip_size, 5)
        tp1_price = round(close_price + atr_pips * tp_atr_mult * pip_size, 5)
        tp2_price = round(close_price + atr_pips * 2.5 * pip_size, 5)
        tp1_pips  = round(atr_pips * tp_atr_mult, 1)
        tp2_pips  = round(atr_pips * 2.5, 1)
    
    log.info(f"[{STRATEGY_NAME}] {direction} {symbol}"
             f" Score:{score} | {', '.join(confluence)}")
    
    return {
        "direction":   direction,
        "entry_price": close_price,
        "sl_price":    sl_price,
        "tp1_price":   tp1_price,
        "tp2_price":   tp2_price,
        "sl_pips":     sl_pips,
        "tp1_pips":    tp1_pips,
        "tp2_pips":    tp2_pips,
        "strategy":    STRATEGY_NAME,
        "version":     VERSION,
        "score":       score,
        "confluence":  confluence,
        "divergence":  divergence,
        "spread":      0,
    }
