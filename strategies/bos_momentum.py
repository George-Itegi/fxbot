# =============================================================
# strategies/bos_momentum.py  v1.0
# Strategy: Break of Structure Momentum
#
# Catches the explosive move right after a Break of Structure
# or Change of Character. When H1 prints a fresh BOS, that's
# often the start of a 50-100+ pip move. Entering on the first
# pullback to the broken level is one of the highest-R:R setups.
#
# Key differentiator vs existing strategies:
#   TREND_CONTINUATION = catches mid-trend pullbacks to EMA
#   EMA_CROSS_MOMENTUM = catches crossover entries
#   SMC_OB_REVERSAL    = catches reversals at order blocks
#   BOS_MOMENTUM       = catches the INITIAL breakout move
#
# Entry: Fresh BOS/CHoCH on H1 → first pullback to broken level
# → displacement candle on M15 confirms continuation.
#
# Group: SMC_STRUCTURE
# Best sessions: London Open, London Session, NY-London Overlap
# Best states: TRENDING_STRONG, BREAKOUT_ACCEPTED, TRENDING_EXTENDED
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "BREAK_OF_STRUCTURE_MOMENTUM"
MIN_SCORE     = 70
VERSION       = "1.0"


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _detect_fresh_bos(df_h1: pd.DataFrame, pip_size: float) -> dict | None:
    """
    Detect a fresh Break of Structure or Change of Character on H1.
    
    BOS (Bullish): Current bar's high > previous swing high, with structure shift
    CHoCH (Bearish): Current bar's low < previous swing low, with structure shift
    
    Returns dict with:
      type: 'BULLISH_BOS', 'BEARISH_BOS', 'BULLISH_CHOCH', 'BEARISH_CHOCH'
      broken_level: The price level that was broken
      displacement_pips: How strong the breakout candle was
      bars_since_bos: How many bars ago the BOS occurred
    or None if no fresh BOS found.
    """
    if df_h1 is None or len(df_h1) < 20:
        return None

    # Look at the last 10 bars for a fresh BOS
    lookback = min(10, len(df_h1) - 5)

    for i in range(len(df_h1) - 1, max(len(df_h1) - lookback - 1, 4), -1):
        current = df_h1.iloc[i]
        prev    = df_h1.iloc[i - 1]

        # Find swing high/low from the preceding bars
        start = max(0, i - 10)
        recent_highs = [float(df_h1.iloc[j]['high']) for j in range(start, i)]
        recent_lows  = [float(df_h1.iloc[j]['low'])  for j in range(start, i)]

        if not recent_highs or not recent_lows:
            continue

        swing_high = max(recent_highs)
        swing_low  = min(recent_lows)

        curr_high = float(current['high'])
        curr_low  = float(current['low'])
        curr_close = float(current['close'])
        curr_open  = float(current['open'])

        # ── Bullish BOS: High breaks above swing high ──
        if curr_high > swing_high:
            # Displacement: strong bullish candle that broke the level
            body = curr_close - curr_open
            disp_pips = body / pip_size

            # Must be a meaningful displacement (not a tiny wick break)
            if disp_pips < 3:
                continue

            # Confirm structure: close should be above the broken level
            if curr_close <= swing_high:
                continue

            bars_since = len(df_h1) - 1 - i

            # Determine if BOS or CHoCH based on prior trend
            # CHoCH = trend was bearish, now breaks bull (trend change)
            # BOS = trend was already bullish, continues higher
            # Use simple EMA check
            ema21 = float(current.get('ema_21', 0))
            bos_type = 'BULLISH_CHOCH' if curr_close < ema21 else 'BULLISH_BOS'

            return {
                'type': bos_type,
                'broken_level': swing_high,
                'displacement_pips': disp_pips,
                'bars_since_bos': bars_since,
                'breakout_candle_close': curr_close,
            }

        # ── Bearish BOS: Low breaks below swing low ──
        if curr_low < swing_low:
            body = curr_open - curr_close
            disp_pips = body / pip_size

            if disp_pips < 3:
                continue

            if curr_close >= swing_low:
                continue

            bars_since = len(df_h1) - 1 - i

            ema21 = float(current.get('ema_21', 0))
            bos_type = 'BEARISH_CHOCH' if curr_close > ema21 else 'BEARISH_BOS'

            return {
                'type': bos_type,
                'broken_level': swing_low,
                'displacement_pips': disp_pips,
                'bars_since_bos': bars_since,
                'breakout_candle_close': curr_close,
            }

    return None


def _check_pullback_to_broken_level(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                                     broken_level: float, direction: str,
                                     pip_size: float) -> dict:
    """
    Check if price has pulled back to the broken level (support/resistance flip).
    
    For bullish BOS: old resistance becomes support — price should pull back to
    or slightly below the broken level, then show rejection.
    
    For bearish BOS: old support becomes resistance — price should pull back to
    or slightly above the broken level, then show rejection.
    
    Returns dict with:
      is_pullback: bool
      pullback_depth_pips: how far price pulled back past the level
      rejection_strength: weak/strong based on M15 candle
    """
    if df_h1 is None or df_m15 is None or len(df_m15) < 3:
        return {"is_pullback": False, "pullback_depth_pips": 0, "rejection_strength": "NONE"}

    h1_last = df_h1.iloc[-1]
    h1_close = float(h1_last['close'])
    m15_last = df_m15.iloc[-1]
    m15_close = float(m15_last['close'])
    m15_low   = float(m15_last['low'])
    m15_high  = float(m15_last['high'])
    m15_open  = float(m15_last['open'])

    max_pullback_pips = 10.0  # Don't allow deep pullbacks (invalidates BOS)

    if direction == "BUY":
        # Price should have pulled back to or slightly below the broken level
        if h1_close > broken_level:
            # Price is above — check if recent low touched the level
            recent_lows = []
            for j in range(max(0, len(df_h1) - 5), len(df_h1)):
                recent_lows.append(float(df_h1.iloc[j]['low']))
            min_low = min(recent_lows)
            pullback_depth = (broken_level - min_low) / pip_size

            if 0 <= pullback_depth <= max_pullback_pips:
                # M15 rejection check
                body = m15_close - m15_open
                lower_wick = m15_open - m15_low if m15_open < m15_close else m15_close - m15_low

                rejection = "NONE"
                if body > 0 and lower_wick > 0:
                    # Bullish candle with lower wick near the level = good rejection
                    if lower_wick / pip_size > 2:
                        rejection = "STRONG"
                    else:
                        rejection = "WEAK"

                return {
                    "is_pullback": True,
                    "pullback_depth_pips": pullback_depth,
                    "rejection_strength": rejection,
                }
    else:  # SELL
        if h1_close < broken_level:
            recent_highs = []
            for j in range(max(0, len(df_h1) - 5), len(df_h1)):
                recent_highs.append(float(df_h1.iloc[j]['high']))
            max_high = max(recent_highs)
            pullback_depth = (max_high - broken_level) / pip_size

            if 0 <= pullback_depth <= max_pullback_pips:
                body = m15_open - m15_close
                upper_wick = m15_high - m15_open if m15_open > m15_close else m15_high - m15_close

                rejection = "NONE"
                if body > 0 and upper_wick > 0:
                    if upper_wick / pip_size > 2:
                        rejection = "STRONG"
                    else:
                        rejection = "WEAK"

                return {
                    "is_pullback": True,
                    "pullback_depth_pips": pullback_depth,
                    "rejection_strength": rejection,
                }

    return {"is_pullback": False, "pullback_depth_pips": 0, "rejection_strength": "NONE"}


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
    Fires when a fresh BOS/CHoCH is detected on H1 and price
    pulls back to the broken level with continuation confirmation.
    """
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_h1) < 20 or len(df_h4) < 20:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 3.0:
        return None

    # Detect fresh BOS
    bos_info = _detect_fresh_bos(df_h1, pip_size)
    if bos_info is None:
        return None

    # Gate: BOS must be fresh (within last 10 bars)
    if bos_info['bars_since_bos'] > 8:
        return None

    bos_type     = bos_info['type']
    broken_level = bos_info['broken_level']
    disp_pips    = bos_info['displacement_pips']
    bars_since   = bos_info['bars_since_bos']

    # Determine direction from BOS type
    if 'BULLISH' in bos_type:
        direction = "BUY"
    elif 'BEARISH' in bos_type:
        direction = "SELL"
    else:
        return None

    # Check for pullback to broken level
    pullback = _check_pullback_to_broken_level(
        df_h1, df_m15, broken_level, direction, pip_size)

    # MANDATORY: Must have a pullback (don't chase breakouts)
    if not pullback['is_pullback']:
        return None

    # Get confirmation data
    structure = (smc_report or {}).get('structure', {})
    trend     = structure.get('trend', 'RANGING')
    smc_bias  = (smc_report or {}).get('smc_bias', 'NEUTRAL')
    htf_ok    = (smc_report or {}).get('htf_alignment', {}).get('approved', True)
    pd_zone   = (smc_report or {}).get('premium_discount', {}).get('zone', '')

    rolling_delta   = market_report.get('rolling_delta', {})
    delta_bias      = rolling_delta.get('bias', 'NEUTRAL')
    delta_strength  = rolling_delta.get('strength', 'WEAK')

    of_imb      = market_report.get('order_flow_imbalance', {})
    of_dir      = of_imb.get('direction', 'NEUTRAL')
    of_strength = of_imb.get('strength', 'NONE')
    of_imb_value = of_imb.get('imbalance', 0)

    volume_surge     = market_report.get('volume_surge', {})
    vol_surge_active = volume_surge.get('surge_detected', False)

    h1  = df_h1.iloc[-1]
    m15 = df_m15.iloc[-1]
    h4  = df_h4.iloc[-1]
    supertrend_dir_h1 = int(h1.get('supertrend_dir', 0))
    supertrend_dir_h4 = int(h4.get('supertrend_dir', 0))
    stoch_k = float(m15.get('stoch_rsi_k', 50))

    score      = 0
    confluence = []

    # ═══════════════════════════════════════════════════════════
    # SCORING (works for both BUY and SELL)
    # ═══════════════════════════════════════════════════════════

    # Core signal: Fresh BOS detected
    score += 20
    confluence.append(f"FRESH_{bos_type}")

    # Pullback to broken level (MANDATORY, already verified)
    score += 15
    confluence.append(f"PULLBACK_TO_LEVEL_{pullback['pullback_depth_pips']:.1f}p")

    # MANDATORY: Delta must confirm direction
    if direction == "BUY" and delta_bias == "BULLISH":
        score += 15
        confluence.append("DELTA_BULL_MANDATORY")
    elif direction == "SELL" and delta_bias == "BEARISH":
        score += 15
        confluence.append("DELTA_BEAR_MANDATORY")
    else:
        return None  # Delta doesn't confirm — skip

    # Displacement quality — stronger break = better setup
    if disp_pips > 15:
        score += 12
        confluence.append(f"STRONG_DISPLACEMENT_{disp_pips:.0f}p")
    elif disp_pips > 8:
        score += 6
        confluence.append(f"DISPLACEMENT_{disp_pips:.0f}p")

    # CHoCH bonus — trend changes tend to have stronger follow-through
    if 'CHOCH' in bos_type:
        score += 8
        confluence.append("CHOCH_BONUS")

    # H4 structure alignment
    if direction == "BUY" and supertrend_dir_h4 == 1:
        score += 12
        confluence.append("H4_SUPERTREND_BULL")
    elif direction == "SELL" and supertrend_dir_h4 == -1:
        score += 12
        confluence.append("H4_SUPERTREND_BEAR")

    # H1 supertrend alignment
    if direction == "BUY" and supertrend_dir_h1 == 1:
        score += 8
        confluence.append("H1_SUPERTREND_BULL")
    elif direction == "SELL" and supertrend_dir_h1 == -1:
        score += 8
        confluence.append("H1_SUPERTREND_BEAR")

    # M15 rejection candle at the broken level
    rejection = pullback.get('rejection_strength', 'NONE')
    if rejection == "STRONG":
        score += 12
        confluence.append("M15_STRONG_REJECTION")
    elif rejection == "WEAK":
        score += 5
        confluence.append("M15_WEAK_REJECTION")

    # Order flow confirmation
    if direction == "BUY" and (of_dir in ('BUY', 'BULLISH') or of_imb_value > 0.2):
        if of_strength in ('STRONG', 'EXTREME'):
            score += 12
            confluence.append("OF_BULL_STRONG")
        else:
            score += 6
            confluence.append("OF_BULL_CONFIRMS")
    elif direction == "SELL" and (of_dir in ('SELL', 'BEARISH') or of_imb_value < -0.2):
        if of_strength in ('STRONG', 'EXTREME'):
            score += 12
            confluence.append("OF_BEAR_STRONG")
        else:
            score += 6
            confluence.append("OF_BEAR_CONFIRMS")

    # Volume surge — institutional participation (strong bonus for breakouts)
    if vol_surge_active:
        score += 10
        confluence.append("VOLUME_SURGE")

    # StochRSI
    if direction == "BUY" and stoch_k < 35:
        score += 5
        confluence.append("STOCHRSI_ROOM_UP")
    elif direction == "SELL" and stoch_k > 65:
        score += 5
        confluence.append("STOCHRSI_ROOM_DOWN")

    # Delta strength
    if delta_strength in ('STRONG', 'MODERATE'):
        score += 5
        confluence.append("DELTA_STRONG")

    # HTF alignment
    if htf_ok and ((direction == "BUY" and smc_bias == "BULLISH") or
                   (direction == "SELL" and smc_bias == "BEARISH")):
        score += 5
        confluence.append("HTF_SMC_ALIGNED")

    # BOS from SMC structure (bonus)
    smc_bos = structure.get('bos')
    has_smc_bos = False
    if smc_bos and direction.upper() in smc_bos.get('type', '').upper():
        score += 5
        confluence.append("SMC_BOS_CONFIRMS")
        has_smc_bos = True

    # PD zone penalty
    if direction == "BUY" and 'EXTREME_PREMIUM' in pd_zone:
        score -= 10
        confluence.append("PD_PREMIUM_PENALTY")
    elif direction == "SELL" and 'EXTREME_DISCOUNT' in pd_zone:
        score -= 10
        confluence.append("PD_DISCOUNT_PENALTY")

    # Pullback depth penalty — too deep = weak breakout
    if pullback['pullback_depth_pips'] > 7:
        score -= 10
        confluence.append("DEEP_PULLBACK_PENALTY")

    # Fibonacci confluence bonus
    try:
        from backtest.fib_builder import build_fib_report, check_fib_confluence
        fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4, current_price=close_price)
        fib_check = check_fib_confluence(close_price, direction, fib_report, pip_size)
        if fib_check['fib_bonus'] > 0:
            score += fib_check['fib_bonus']
            confluence.extend(fib_check['confluence'])
    except Exception:
        pass

    if len(confluence) < 5:
        return None

    if score >= MIN_SCORE:
        # Entry near the broken level
        if direction == "BUY":
            entry_price = max(close_price, broken_level)
            sl_price  = round(broken_level - atr_pips * 0.5 * pip_size, 5)
            tp1_price = round(entry_price + atr_pips * 2.0 * pip_size, 5)
            tp2_price = round(entry_price + atr_pips * 4.0 * pip_size, 5)
            sl_pips   = round((entry_price - sl_price) / pip_size, 1)
            tp1_pips  = round((tp1_price - entry_price) / pip_size, 1)
            tp2_pips  = round((tp2_price - entry_price) / pip_size, 1)
        else:
            entry_price = min(close_price, broken_level)
            sl_price  = round(broken_level + atr_pips * 0.5 * pip_size, 5)
            tp1_price = round(entry_price - atr_pips * 2.0 * pip_size, 5)
            tp2_price = round(entry_price - atr_pips * 4.0 * pip_size, 5)
            sl_pips   = round((sl_price - entry_price) / pip_size, 1)
            tp1_pips  = round((entry_price - tp1_price) / pip_size, 1)
            tp2_pips  = round((entry_price - tp2_price) / pip_size, 1)

        log.info(f"[{STRATEGY_NAME} v{VERSION}] {direction} {symbol}"
                 f" entry={entry_price:.5f} Score:{score} | "
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
            "broken_level": broken_level,
            "_bos_momentum_features": {
                'bos_type': bos_type,
                'broken_level': broken_level,
                'displacement_pips': disp_pips,
                'bars_since_bos': bars_since,
                'pullback_depth_pips': pullback['pullback_depth_pips'],
                'rejection_strength': rejection,
                'trend': trend,
                'delta_bias': delta_bias,
                'delta_strength': delta_strength,
                'of_imbalance': of_imb_value,
                'of_strength': of_strength,
                'stoch_rsi_k': stoch_k,
                'supertrend_dir_h1': supertrend_dir_h1,
                'supertrend_dir_h4': supertrend_dir_h4,
                'htf_ok': 1 if htf_ok else 0,
                'smc_bias': smc_bias,
                'pd_zone': pd_zone,
                'vol_surge': 1 if vol_surge_active else 0,
                'has_smc_bos': 1 if has_smc_bos else 0,
                'atr_pips': atr_pips,
            },
        }

    return None
