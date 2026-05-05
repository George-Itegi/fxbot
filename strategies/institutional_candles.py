# =============================================================
# strategies/institutional_candles.py  v1.0
# Strategy: Institutional Candlestick Patterns
#
# Professional price action trading — NOT retail "I see a hammer, buy."
# This strategy fires ONLY when a high-quality candlestick pattern
# (engulfing, pin bar, marubozu) forms AT a confirmed institutional
# level (order blocks, S&D zones, VWAP, PD zones, BOS levels).
#
# The edge is the CONFLUENCE of candlestick pattern + institutional
# level + delta confirmation + order flow. Any single element alone
# is noise; together they identify where banks are defending levels.
#
# Group: PRICE_ACTION (5th group — critical for consensus diversity)
# Best sessions: London Open, London Session, NY-London Overlap
# Best states: TRENDING_STRONG, BREAKOUT_ACCEPTED, REVERSAL_RISK
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "INSTITUTIONAL_CANDLES"
MIN_SCORE     = 70
VERSION       = "1.0"


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


# ================================================================
# Candle Pattern Detection Helpers
# ================================================================

def _detect_m15_candle_patterns(df_m15: pd.DataFrame,
                                 pip_size: float) -> dict | None:
    """
    Analyze the last 2-3 M15 candles and detect institutional-quality
    candlestick patterns.

    Detects:
      1. BULLISH_ENGULFING / BEARISH_ENGULFING
      2. PIN_BAR_BULL / PIN_BAR_BEAR (hammer / shooting star)
      3. MARUBOZU_BULL / MARUBOZU_BEAR (full body, no wicks)

    Returns dict with pattern details, or None if no valid pattern.
    """
    if df_m15 is None or len(df_m15) < 3:
        return None

    # Latest candle and the one before it
    curr = df_m15.iloc[-1]
    prev = df_m15.iloc[-2]

    curr_open  = float(curr['open'])
    curr_close = float(curr['close'])
    curr_high  = float(curr['high'])
    curr_low   = float(curr['low'])

    prev_open  = float(prev['open'])
    prev_close = float(prev['close'])

    curr_body = abs(curr_close - curr_open)
    curr_body_pips = curr_body / pip_size
    curr_upper_wick = curr_high - max(curr_open, curr_close)
    curr_lower_wick = min(curr_open, curr_close) - curr_low
    curr_range = curr_high - curr_low
    curr_upper_wick_pips = curr_upper_wick / pip_size
    curr_lower_wick_pips = curr_lower_wick / pip_size
    curr_range_pips = curr_range / pip_size

    # ── 1. BULLISH ENGULFING ──
    # Current bullish body completely engulfs previous bearish body
    if (curr_close > prev_open and curr_open <= prev_close
            and curr_close > curr_open
            and prev_close < prev_open
            and curr_body_pips > 3.0):
        # Quality: ratio of engulfing body to total range
        wick_total = curr_upper_wick_pips + curr_lower_wick_pips
        wick_ratio = wick_total / curr_body_pips if curr_body_pips > 0 else 99
        quality = "STRONG" if wick_ratio < 0.3 else ("MODERATE" if wick_ratio < 0.6 else "WEAK")
        return {
            'pattern_type': 'BULLISH_ENGULFING',
            'direction': 'BUY',
            'body_pips': curr_body_pips,
            'wick_ratio': round(wick_ratio, 2),
            'quality': quality,
            'candle_high': curr_high,
            'candle_low': curr_low,
            'prev_high': float(prev['high']),
            'prev_low': float(prev['low']),
        }

    # ── 2. BEARISH ENGULFING ──
    # Current bearish body completely engulfs previous bullish body
    if (curr_close < prev_open and curr_open >= prev_close
            and curr_close < curr_open
            and prev_close > prev_open
            and curr_body_pips > 3.0):
        wick_total = curr_upper_wick_pips + curr_lower_wick_pips
        wick_ratio = wick_total / curr_body_pips if curr_body_pips > 0 else 99
        quality = "STRONG" if wick_ratio < 0.3 else ("MODERATE" if wick_ratio < 0.6 else "WEAK")
        return {
            'pattern_type': 'BEARISH_ENGULFING',
            'direction': 'SELL',
            'body_pips': curr_body_pips,
            'wick_ratio': round(wick_ratio, 2),
            'quality': quality,
            'candle_high': curr_high,
            'candle_low': curr_low,
            'prev_high': float(prev['high']),
            'prev_low': float(prev['low']),
        }

    # ── 3. PIN BAR BULL (Hammer) ──
    # Long lower wick >= 2x body, small upper wick, body in upper third
    if (curr_lower_wick_pips >= 5.0
            and curr_body_pips > 0
            and curr_lower_wick >= 2.0 * curr_body
            and curr_upper_wick < curr_body * 0.5
            and curr_range_pips > 0):
        # Body should be in upper third of the range
        body_mid = (curr_open + curr_close) / 2
        range_pos = (body_mid - curr_low) / curr_range  # 0=bottom, 1=top
        if range_pos >= 0.6:  # Body in upper 40% of range
            wick_ratio = curr_lower_wick / curr_body if curr_body > 0 else 0
            quality = "STRONG" if wick_ratio >= 3.0 else ("MODERATE" if wick_ratio >= 2.0 else "WEAK")
            return {
                'pattern_type': 'PIN_BAR_BULL',
                'direction': 'BUY',
                'body_pips': curr_body_pips,
                'wick_ratio': round(wick_ratio, 2),
                'quality': quality,
                'candle_high': curr_high,
                'candle_low': curr_low,
                'prev_high': float(prev['high']),
                'prev_low': float(prev['low']),
            }

    # ── 4. PIN BAR BEAR (Shooting Star) ──
    # Long upper wick >= 2x body, small lower wick, body in lower third
    if (curr_upper_wick_pips >= 5.0
            and curr_body_pips > 0
            and curr_upper_wick >= 2.0 * curr_body
            and curr_lower_wick < curr_body * 0.5
            and curr_range_pips > 0):
        body_mid = (curr_open + curr_close) / 2
        range_pos = (body_mid - curr_low) / curr_range
        if range_pos <= 0.4:  # Body in lower 40% of range
            wick_ratio = curr_upper_wick / curr_body if curr_body > 0 else 0
            quality = "STRONG" if wick_ratio >= 3.0 else ("MODERATE" if wick_ratio >= 2.0 else "WEAK")
            return {
                'pattern_type': 'PIN_BAR_BEAR',
                'direction': 'SELL',
                'body_pips': curr_body_pips,
                'wick_ratio': round(wick_ratio, 2),
                'quality': quality,
                'candle_high': curr_high,
                'candle_low': curr_low,
                'prev_high': float(prev['high']),
                'prev_low': float(prev['low']),
            }

    # ── 5. MARUBOZU BULL ──
    # Full body candle — tiny wicks, strong bullish body
    if (curr_close > curr_open
            and curr_upper_wick_pips < 1.0
            and curr_lower_wick_pips < 1.0
            and curr_body_pips > 5.0):
        return {
            'pattern_type': 'MARUBOZU_BULL',
            'direction': 'BUY',
            'body_pips': curr_body_pips,
            'wick_ratio': round((curr_upper_wick_pips + curr_lower_wick_pips) / curr_body_pips, 2),
            'quality': "STRONG" if curr_body_pips > 10 else "MODERATE",
            'candle_high': curr_high,
            'candle_low': curr_low,
            'prev_high': float(prev['high']),
            'prev_low': float(prev['low']),
        }

    # ── 6. MARUBOZU BEAR ──
    # Full body candle — tiny wicks, strong bearish body
    if (curr_close < curr_open
            and curr_upper_wick_pips < 1.0
            and curr_lower_wick_pips < 1.0
            and curr_body_pips > 5.0):
        return {
            'pattern_type': 'MARUBOZU_BEAR',
            'direction': 'SELL',
            'body_pips': curr_body_pips,
            'wick_ratio': round((curr_upper_wick_pips + curr_lower_wick_pips) / curr_body_pips, 2),
            'quality': "STRONG" if curr_body_pips > 10 else "MODERATE",
            'candle_high': curr_high,
            'candle_low': curr_low,
            'prev_high': float(prev['high']),
            'prev_low': float(prev['low']),
        }

    return None


# ================================================================
# Institutional Context Checker
# ================================================================

def _check_institutional_context(smc_report: dict,
                                  market_report: dict,
                                  close_price: float,
                                  pip_size: float,
                                  direction: str) -> dict:
    """
    Check if the candle pattern occurs at an institutional level.

    Checks 5 types of institutional confluence:
      1. PD Zone (Premium/Discount)
      2. VWAP proximity
      3. BOS Level proximity
      4. Order Block proximity
      5. Supply/Demand Zone proximity

    Returns dict with context details and cumulative score.
    """
    context_types = []
    context_score = 0
    details = []

    tolerance_pips = 5.0
    tolerance = tolerance_pips * pip_size

    # ── 1. PD Zone (Premium/Discount) ──
    pd_zone = (smc_report or {}).get('premium_discount', {})
    zone_name = pd_zone.get('zone', '')

    if direction == 'BUY':
        if 'EXTREME_DISCOUNT' in zone_name:
            context_types.append("AT_EXTREME_DISCOUNT")
            context_score += 12
            details.append("BUY at extreme discount zone — deep value")
        elif 'DISCOUNT' in zone_name:
            context_types.append("AT_DISCOUNT_ZONE")
            context_score += 12
            details.append("BUY at discount zone — institutional buy area")
    elif direction == 'SELL':
        if 'EXTREME_PREMIUM' in zone_name:
            context_types.append("AT_EXTREME_PREMIUM")
            context_score += 12
            details.append("SELL at extreme premium zone — deep overbought")
        elif 'PREMIUM' in zone_name:
            context_types.append("AT_PREMIUM_ZONE")
            context_score += 12
            details.append("SELL at premium zone — institutional sell area")

    # ── 2. VWAP proximity ──
    vwap_data = market_report.get('vwap', {})
    vwap_price = vwap_data.get('vwap', 0)
    if vwap_price > 0:
        pip_from_vwap = abs(close_price - vwap_price) / pip_size
        if pip_from_vwap < tolerance_pips:
            # Direction-aware: BUY below VWAP, SELL above VWAP
            if direction == 'BUY' and close_price <= vwap_price:
                context_types.append("NEAR_VWAP_BELOW")
                context_score += 8
                details.append(f"BUY below VWAP by {pip_from_vwap:.1f}p — mean reversion up")
            elif direction == 'SELL' and close_price >= vwap_price:
                context_types.append("NEAR_VWAP_ABOVE")
                context_score += 8
                details.append(f"SELL above VWAP by {pip_from_vwap:.1f}p — mean reversion down")

    # ── 3. BOS Level proximity ──
    structure = (smc_report or {}).get('structure', {})
    bos = structure.get('bos', {})
    if bos:
        bos_level = float(bos.get('level', 0))
        bos_type = str(bos.get('type', ''))
        if bos_level > 0:
            pip_from_bos = abs(close_price - bos_level) / pip_size
            if pip_from_bos < tolerance_pips:
                bos_aligned = (
                    (direction == 'BUY' and 'BULLISH' in bos_type)
                    or (direction == 'SELL' and 'BEARISH' in bos_type)
                )
                if bos_aligned:
                    context_types.append("NEAR_BOS_LEVEL")
                    context_score += 10
                    details.append(f"Near {'bullish' if 'BULLISH' in bos_type else 'bearish'} BOS level ({pip_from_bos:.1f}p)")

    # ── 4. Order Block proximity ──
    order_blocks = (smc_report or {}).get('order_blocks', [])
    if order_blocks and isinstance(order_blocks, list):
        for ob in order_blocks:
            if not isinstance(ob, dict):
                continue
            ob_top = float(ob.get('top', 0))
            ob_bottom = float(ob.get('bottom', 0))
            ob_type = str(ob.get('type', ''))
            if ob_top <= 0 or ob_bottom <= 0:
                continue

            # Check if price is near this order block
            near_top = abs(close_price - ob_top) / pip_size < tolerance_pips
            near_bottom = abs(close_price - ob_bottom) / pip_size < tolerance_pips
            inside = ob_bottom <= close_price <= ob_top

            if near_top or near_bottom or inside:
                ob_aligned = (
                    (direction == 'BUY' and 'BULLISH' in ob_type.upper())
                    or (direction == 'SELL' and 'BEARISH' in ob_type.upper())
                )
                if ob_aligned:
                    context_types.append("NEAR_OB_LEVEL")
                    context_score += 8
                    details.append(f"Near aligned {ob_type} OB ({ob_bottom:.5f}–{ob_top:.5f})")
                    break  # Only count one OB

    # ── 5. Supply/Demand Zone proximity ──
    # Check from the SMC report zone data
    sd_zones = (smc_report or {}).get('sd_zones', [])
    if sd_zones and isinstance(sd_zones, list):
        for zone in sd_zones:
            if not isinstance(zone, dict):
                continue
            zone_top = float(zone.get('top', zone.get('zone_top', 0)))
            zone_bottom = float(zone.get('bottom', zone.get('zone_bottom', 0)))
            zone_type = str(zone.get('type', ''))

            if zone_top <= 0 or zone_bottom <= 0:
                continue

            near_top = abs(close_price - zone_top) / pip_size < tolerance_pips
            near_bottom = abs(close_price - zone_bottom) / pip_size < tolerance_pips
            inside = zone_bottom <= close_price <= zone_top

            if near_top or near_bottom or inside:
                zone_aligned = (
                    (direction == 'BUY' and 'DEMAND' in zone_type.upper())
                    or (direction == 'SELL' and 'SUPPLY' in zone_type.upper())
                )
                if zone_aligned:
                    context_types.append("NEAR_SD_ZONE")
                    context_score += 8
                    details.append(f"Near aligned {zone_type} zone ({zone_bottom:.5f}–{zone_top:.5f})")
                    break  # Only count one SD zone

    has_context = len(context_types) >= 1

    return {
        'has_context': has_context,
        'context_types': context_types,
        'context_score': context_score,
        'context_count': len(context_types),
        'details': details,
    }


# ================================================================
# Main Strategy Evaluate
# ================================================================

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
    Fires when a high-quality M15 candlestick pattern forms AT a
    confirmed institutional level WITH delta and order flow confirmation.

    MANDATORY gates:
      - Recognized candle pattern (engulfing, pin bar, or marubozu)
      - Pattern occurs at at least one institutional level
      - Delta bias matches pattern direction
    """
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_m15) < 3 or len(df_h1) < 20 or len(df_h4) < 20:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 3.0:
        return None

    # ── STEP 1: Detect M15 Candle Pattern ──
    pattern = _detect_m15_candle_patterns(df_m15, pip_size)
    if pattern is None:
        return None

    pattern_type = pattern['pattern_type']
    direction    = pattern['direction']
    quality      = pattern['quality']
    body_pips    = pattern['body_pips']
    wick_ratio   = pattern['wick_ratio']

    # Only use STRONG or MODERATE quality patterns
    if quality == "WEAK":
        return None

    # ── STEP 2: MANDATORY — Institutional Context ──
    ctx = _check_institutional_context(
        smc_report, market_report, close_price, pip_size, direction)

    if not ctx['has_context']:
        return None

    # ── Get SMC data ──
    structure = (smc_report or {}).get('structure', {})
    trend     = structure.get('trend', 'RANGING')
    smc_bias  = (smc_report or {}).get('smc_bias', 'NEUTRAL')
    htf_ok    = (smc_report or {}).get('htf_alignment', {}).get('approved', True)
    pd_zone   = (smc_report or {}).get('premium_discount', {}).get('zone', '')

    # Delta data (MANDATORY)
    rolling_delta  = market_report.get('rolling_delta', {})
    delta_bias     = rolling_delta.get('bias', 'NEUTRAL')
    delta_strength = rolling_delta.get('strength', 'WEAK')

    # Order flow data
    of_imb      = market_report.get('order_flow_imbalance', {})
    of_dir      = of_imb.get('direction', 'NEUTRAL')
    of_strength = of_imb.get('strength', 'NONE')
    of_imb_value = of_imb.get('imbalance', 0)

    # Volume data
    volume_surge     = market_report.get('volume_surge', {})
    vol_surge_active = volume_surge.get('surge_detected', False)
    surge_ratio      = volume_surge.get('surge_ratio', 1.0)

    # Spread data
    spread_data = market_report.get('spread', {})
    spread_pips = float(spread_data.get('spread_pips', 99))

    # Candle data for indicators
    h1   = df_h1.iloc[-1]
    m15  = df_m15.iloc[-1]
    h4   = df_h4.iloc[-1]
    supertrend_dir_h1 = int(h1.get('supertrend_dir', 0))
    supertrend_dir_h4 = int(h4.get('supertrend_dir', 0))
    stoch_k = float(m15.get('stoch_rsi_k', 50))

    # ── STEP 3: MANDATORY — Delta must match direction ──
    if direction == 'BUY' and delta_bias != 'BULLISH':
        return None
    if direction == 'SELL' and delta_bias != 'BEARISH':
        return None

    # ── Build Score ──
    score      = 0
    confluence = []

    # Base score: pattern detected
    score += 20
    confluence.append(f"PATTERN_{pattern_type}")

    # Institutional context score (8-12 per level found)
    score += ctx['context_score']
    confluence.extend(ctx['context_types'])

    # Delta mandatory
    score += 15
    confluence.append(f"DELTA_{delta_bias}_MANDATORY")

    # Pattern quality bonus
    if quality == "STRONG":
        score += 10
        confluence.append("PATTERN_STRONG_QUALITY")
    elif quality == "MODERATE":
        score += 5
        confluence.append("PATTERN_MODERATE_QUALITY")

    # Spread filter
    is_jpy = 'JPY' in symbol.upper()
    max_spread = 0.5 if is_jpy else 2.0
    if spread_pips < max_spread:
        score += 8
        confluence.append("LOW_SPREAD")
    elif spread_pips > max_spread * 1.5:
        # High spread penalty — reduce score
        score -= 5
        confluence.append("HIGH_SPREAD_PENALTY")

    # Volume surge
    if vol_surge_active:
        score += 8
        confluence.append("VOLUME_SURGE")

    # H4 supertrend alignment
    if direction == 'BUY' and supertrend_dir_h4 == 1:
        score += 10
        confluence.append("H4_SUPERTREND_BULL")
    elif direction == 'SELL' and supertrend_dir_h4 == -1:
        score += 10
        confluence.append("H4_SUPERTREND_BEAR")

    # H1 supertrend alignment
    if direction == 'BUY' and supertrend_dir_h1 == 1:
        score += 5
        confluence.append("H1_SUPERTREND_BULL")
    elif direction == 'SELL' and supertrend_dir_h1 == -1:
        score += 5
        confluence.append("H1_SUPERTREND_BEAR")

    # StochRSI: oversold for BUY, overbought for SELL
    if direction == 'BUY' and stoch_k < 30:
        score += 8
        confluence.append("STOCHRSI_OVERSOLD")
    elif direction == 'SELL' and stoch_k > 70:
        score += 8
        confluence.append("STOCHRSI_OVERBOUGHT")

    # Order flow imbalance confirming direction
    if direction == 'BUY':
        if of_dir in ('BUY', 'BULLISH') or of_imb_value > 0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 12
                confluence.append("OF_BULL_STRONG")
            else:
                score += 6
                confluence.append("OF_BULL_CONFIRMS")
    elif direction == 'SELL':
        if of_dir in ('SELL', 'BEARISH') or of_imb_value < -0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 12
                confluence.append("OF_BEAR_STRONG")
            else:
                score += 6
                confluence.append("OF_BEAR_CONFIRMS")

    # HTF alignment
    if htf_ok and smc_bias == direction.replace('BUY', 'BULLISH').replace('SELL', 'BEARISH'):
        score += 8
        confluence.append("HTF_SMC_ALIGNED")
    elif htf_ok:
        score += 3
        confluence.append("HTF_APPROVED")

    # BOS from SMC confirming direction
    bos = structure.get('bos', {})
    has_bos = False
    if bos:
        bos_type = str(bos.get('type', ''))
        if direction == 'BUY' and 'BULLISH' in bos_type:
            score += 8
            confluence.append("BOS_BULL_CONFIRM")
            has_bos = True
        elif direction == 'SELL' and 'BEARISH' in bos_type:
            score += 8
            confluence.append("BOS_BEAR_CONFIRM")
            has_bos = True

    # Multi-level confluence bonus (2+ institutional context types)
    if ctx['context_count'] >= 2:
        score += 10
        confluence.append("MULTI_LEVEL_CONFLUENCE")

    # London/NY session bonus
    session = (master_report or {}).get('session', '')
    if session in ('LONDON_OPEN', 'LONDON_SESSION', 'NY_LONDON_OVERLAP'):
        score += 5
        confluence.append("SESSION_BONUS")

    # Delta strength bonus
    if delta_strength in ('STRONG', 'MODERATE'):
        score += 5
        confluence.append("DELTA_STRONG")

    # Fibonacci confluence bonus
    fib_bonus = 0
    try:
        from backtest.fib_builder import build_fib_report, check_fib_confluence
        fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4,
                                       current_price=close_price)
        fib_check = check_fib_confluence(close_price, direction,
                                          fib_report, pip_size)
        if fib_check['fib_bonus'] > 0:
            score += fib_check['fib_bonus']
            confluence.extend(fib_check['confluence'])
            fib_bonus = fib_check['fib_bonus']
    except Exception:
        pass

    # PD zone penalty for wrong direction
    if direction == 'BUY' and 'EXTREME_PREMIUM' in pd_zone:
        score -= 10
        confluence.append("PD_PREMIUM_PENALTY")
    elif direction == 'SELL' and 'EXTREME_DISCOUNT' in pd_zone:
        score -= 10
        confluence.append("PD_DISCOUNT_PENALTY")

    # ── Minimum confluence check ──
    if len(confluence) < 5:
        return None

    if score < MIN_SCORE:
        return None

    # ═══════════════════════════════════════════════════════════
    # SL/TP Calculation
    # ═══════════════════════════════════════════════════════════
    entry_price = close_price

    # SL: beyond the wick or institutional level (whichever is wider)
    # For engulfing patterns: beyond the engulfed candle
    # For pin bar patterns: beyond the wick tip
    # For marubozu: beyond the candle body
    min_sl_pips = atr_pips * 0.5  # ATR * 0.5 minimum

    if 'ENGULFING' in pattern_type:
        # SL beyond the engulfed candle
        if direction == 'BUY':
            sl_reference = pattern['prev_low']  # Below the previous (bearish) candle
            raw_sl_pips = (entry_price - sl_reference) / pip_size
            sl_pips = max(raw_sl_pips, min_sl_pips)
            sl_price = round(entry_price - sl_pips * pip_size, 5)
        else:
            sl_reference = pattern['prev_high']  # Above the previous (bullish) candle
            raw_sl_pips = (sl_reference - entry_price) / pip_size
            sl_pips = max(raw_sl_pips, min_sl_pips)
            sl_price = round(entry_price + sl_pips * pip_size, 5)
    elif 'PIN_BAR' in pattern_type:
        # SL beyond the wick tip
        if direction == 'BUY':
            raw_sl_pips = (entry_price - pattern['candle_low']) / pip_size
            sl_pips = max(raw_sl_pips, min_sl_pips)
            sl_price = round(entry_price - sl_pips * pip_size, 5)
        else:
            raw_sl_pips = (pattern['candle_high'] - entry_price) / pip_size
            sl_pips = max(raw_sl_pips, min_sl_pips)
            sl_price = round(entry_price + sl_pips * pip_size, 5)
    else:
        # Marubozu — SL beyond the candle
        if direction == 'BUY':
            raw_sl_pips = (entry_price - pattern['candle_low']) / pip_size
            sl_pips = max(raw_sl_pips, min_sl_pips)
            sl_price = round(entry_price - sl_pips * pip_size, 5)
        else:
            raw_sl_pips = (pattern['candle_high'] - entry_price) / pip_size
            sl_pips = max(raw_sl_pips, min_sl_pips)
            sl_price = round(entry_price + sl_pips * pip_size, 5)

    sl_pips = round(sl_pips, 1)

    # TP1 at 1.5R, TP2 at 2.5R
    if direction == 'BUY':
        tp1_price = round(entry_price + sl_pips * 1.5 * pip_size, 5)
        tp2_price = round(entry_price + sl_pips * 2.5 * pip_size, 5)
        tp1_pips  = round(sl_pips * 1.5, 1)
        tp2_pips  = round(sl_pips * 2.5, 1)
    else:
        tp1_price = round(entry_price - sl_pips * 1.5 * pip_size, 5)
        tp2_price = round(entry_price - sl_pips * 2.5 * pip_size, 5)
        tp1_pips  = round(sl_pips * 1.5, 1)
        tp2_pips  = round(sl_pips * 2.5, 1)

    log.info(f"[{STRATEGY_NAME} v{VERSION}] {direction} {symbol}"
             f" entry={entry_price:.5f} SL={sl_price:.5f} "
             f"Score:{score} Pattern:{pattern_type} | "
             f"{', '.join(confluence)}")

    return {
        "direction":   direction,
        "entry_price": entry_price,
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
        "pattern":     pattern_type,
        "_inst_candles_features": {
            'pattern_type':    pattern_type,
            'direction':       direction,
            'body_pips':       round(body_pips, 1),
            'wick_ratio':      wick_ratio,
            'quality':         quality,
            'context_types':   ctx['context_types'],
            'context_count':   ctx['context_count'],
            'context_score':   ctx['context_score'],
            'delta_bias':      delta_bias,
            'delta_strength':  delta_strength,
            'spread_pips':     round(spread_pips, 1),
            'of_imbalance':    of_imb_value,
            'of_strength':     of_strength,
            'stoch_rsi_k':     stoch_k,
            'supertrend_dir_h1': supertrend_dir_h1,
            'supertrend_dir_h4': supertrend_dir_h4,
            'htf_ok':          1 if htf_ok else 0,
            'smc_bias':        smc_bias,
            'pd_zone':         pd_zone,
            'vol_surge':       1 if vol_surge_active else 0,
            'has_bos':         1 if has_bos else 0,
            'fib_bonus':       fib_bonus,
            'atr_pips':        round(atr_pips, 1),
            'sl_pips':         sl_pips,
            'trend':           trend,
        },
    }
