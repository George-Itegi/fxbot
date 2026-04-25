# =============================================================
# strategies/rsi_divergence_smc.py  v1.0
# Strategy 13: RSI Divergence + SMC Confirmation (OSCILLATOR group)
#
# Purpose: 5th strategy group — oscillator-based approach that is
# fundamentally different from trend, structure, order flow, and
# mean reversion. Uses RSI divergence on M15 as the PRIMARY trigger
# with BOS/CHoCH from SMC as confirmation.
#
# Entry logic:
#   1. M15 RSI divergence (price new high/low but RSI diverges)
#   2. BOS or CHoCH from SMC report in divergence direction
#   3. FVG or OB zone near the divergence point (bonus)
#   4. Delta divergence alignment (bonus)
#   5. Volume surge at the reversal candle
#
# Win rate target: 55-65%
# Best session: LONDON_OPEN, LONDON_SESSION, NY_LONDON_OVERLAP
# Best state:  REVERSAL_RISK, BREAKOUT_REJECTED, BALANCED
# =============================================================

import pandas as pd
import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "RSI_DIVERGENCE_SMC"
MIN_SCORE     = 68
VERSION       = "1.0"

# --- Parameters ---
RSI_OVERBOUGHT   = 70    # M15 RSI overbought threshold
RSI_OVERSOLD     = 30    # M15 RSI oversold threshold
RSI_LOOKBACK     = 20    # Bars to look back for swing points in RSI
SWING_LOOKBACK   = 10    # Bars either side for swing detection
MIN_RSI_DIVERGENCE = 5.0 # Minimum RSI difference for divergence
OB_FVG_PROXIMITY = 25.0  # Pips — how close OB/FVG must be


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _find_swing_points(df: pd.DataFrame, lookback: int = 10, symbol: str = '') -> dict:
    """
    Find swing highs and lows using candle data.
    Returns dict with 'swing_highs' and 'swing_lows' lists.
    """
    if df is None or len(df) < lookback * 2 + 1:
        return {"swing_highs": [], "swing_lows": []}

    recent = df.tail(lookback * 2 + 1).copy()
    current_close = float(recent.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, current_close)

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(recent) - lookback):
        candle = recent.iloc[i]
        window_left = recent.iloc[i - lookback:i]
        window_right = recent.iloc[i + 1:i + lookback + 1]

        if candle['high'] >= window_left['high'].max() and \
           candle['high'] >= window_right['high'].max():
            swing_highs.append({
                'price': float(candle['high']),
                'index': int(i),
                'pips_from_current': round(
                    (float(candle['high']) - current_close) / pip_size, 1),
            })

        if candle['low'] <= window_left['low'].min() and \
           candle['low'] <= window_right['low'].min():
            swing_lows.append({
                'price': float(candle['low']),
                'index': int(i),
                'pips_from_current': round(
                    (current_close - float(candle['low'])) / pip_size, 1),
            })

    swing_highs = sorted(swing_highs, key=lambda x: x['pips_from_current'])[:3]
    swing_lows = sorted(swing_lows, key=lambda x: x['pips_from_current'])[:3]

    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def _detect_rsi_divergence(df: pd.DataFrame, symbol: str = '',
                            swing_lookback: int = SWING_LOOKBACK,
                            min_rsi_diff: float = MIN_RSI_DIVERGENCE) -> dict | None:
    """
    Detect RSI divergence on M15 timeframe.
    Returns divergence info dict or None.
    """
    if df is None or len(df) < RSI_LOOKBACK + 5:
        return None

    close_price = float(df.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)

    # Get RSI values
    rsi_col = 'rsi'
    if rsi_col not in df.columns:
        return None

    rsi_values = df[rsi_col].values[-RSI_LOOKBACK:]
    if len(rsi_values) < RSI_LOOKBACK or np.any(np.isnan(rsi_values)):
        return None

    # Find swing points in both price and RSI
    swings = _find_swing_points(df.tail(RSI_LOOKBACK), lookback=swing_lookback, symbol=symbol)
    swing_highs = swings['swing_highs']
    swing_lows = swings['swing_lows']

    result = None

    # Bearish divergence: price makes higher high, RSI makes lower high
    if len(swing_highs) >= 2:
        prev = swing_highs[-2]
        curr = swing_highs[-1]

        if curr['price'] > prev['price']:
            prev_idx = len(df) - RSI_LOOKBACK + prev['index']
            curr_idx = len(df) - RSI_LOOKBACK + curr['index']

            if 0 <= prev_idx < len(df) and 0 <= curr_idx < len(df):
                prev_rsi = float(df.iloc[prev_idx][rsi_col])
                curr_rsi = float(df.iloc[curr_idx][rsi_col])

                rsi_diff = prev_rsi - curr_rsi
                if rsi_diff > min_rsi_diff:
                    price_range = (curr['price'] - prev['price']) / pip_size
                    strength = 'EXTREME' if curr_rsi > RSI_OVERBOUGHT else \
                               'STRONG' if rsi_diff > 10 else 'MODERATE'
                    result = {
                        'type': 'BEARISH',
                        'direction': 'SELL',
                        'price_range_pips': round(price_range, 1),
                        'rsi_diff': round(rsi_diff, 1),
                        'prev_rsi': round(prev_rsi, 1),
                        'curr_rsi': round(curr_rsi, 1),
                        'strength': strength,
                        'description': (f"Price HH +{price_range:.1f}p "
                                        f"but RSI {prev_rsi:.0f} -> {curr_rsi:.0f}"),
                    }

    # Bullish divergence: price makes lower low, RSI makes higher low
    if result is None and len(swing_lows) >= 2:
        prev = swing_lows[-2]
        curr = swing_lows[-1]

        if curr['price'] < prev['price']:
            prev_idx = len(df) - RSI_LOOKBACK + prev['index']
            curr_idx = len(df) - RSI_LOOKBACK + curr['index']

            if 0 <= prev_idx < len(df) and 0 <= curr_idx < len(df):
                prev_rsi = float(df.iloc[prev_idx][rsi_col])
                curr_rsi = float(df.iloc[curr_idx][rsi_col])

                rsi_diff = curr_rsi - prev_rsi
                if rsi_diff > min_rsi_diff:
                    price_range = (prev['price'] - curr['price']) / pip_size
                    strength = 'EXTREME' if curr_rsi < RSI_OVERSOLD else \
                               'STRONG' if rsi_diff > 10 else 'MODERATE'
                    result = {
                        'type': 'BULLISH',
                        'direction': 'BUY',
                        'price_range_pips': round(price_range, 1),
                        'rsi_diff': round(rsi_diff, 1),
                        'prev_rsi': round(prev_rsi, 1),
                        'curr_rsi': round(curr_rsi, 1),
                        'strength': strength,
                        'description': (f"Price LL -{price_range:.1f}p "
                                        f"but RSI {prev_rsi:.0f} -> {curr_rsi:.0f}"),
                    }

    return result


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
    RSI Divergence + SMC Confirmation Strategy:
    Fires when M15 RSI diverges from price AND SMC structure confirms.
    """
    if df_m15 is None or len(df_m15) < RSI_LOOKBACK + 10:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 2.0:
        return None

    # ── Detect RSI Divergence (mandatory) ───────────────
    # In relaxed mode: use wider swing lookback and lower RSI diff
    if relaxed:
        orig_swing_lookback = SWING_LOOKBACK
        orig_min_rsi_div = MIN_RSI_DIVERGENCE
        # Temporarily widen parameters for detection
        swing_lb = 5   # wider window catches more swings
        min_rsi_div = 2.0  # smaller divergences are valid
    else:
        swing_lb = SWING_LOOKBACK
        min_rsi_div = MIN_RSI_DIVERGENCE

    divergence = _detect_rsi_divergence(df_m15, symbol, swing_lookback=swing_lb,
                                          min_rsi_diff=min_rsi_div)

    if divergence is None:
        return None

    direction = divergence['direction']
    div_strength = divergence['strength']
    rsi_diff = divergence['rsi_diff']

    score = 0
    confluence = []

    # ── Score: Divergence strength ──────────────────────
    if div_strength == 'EXTREME':
        score += 30
        confluence.append("RSI_DIV_EXTREME")
    elif div_strength == 'STRONG':
        score += 22
        confluence.append("RSI_DIV_STRONG")
    else:
        score += 22 if relaxed else 14  # Relaxed: MODERATE gets STRONG points
        confluence.append("RSI_DIV_MODERATE")

    confluence.append(divergence['description'])

    # ── SMC Confirmation: BOS/CHoCH (mandatory) ─────────
    smc_confirmed = False
    if smc_report:
        _bos = smc_report.get('structure', {}).get('bos')
        bos_list = [_bos] if _bos and isinstance(_bos, dict) else []
        _choch = smc_report.get('structure', {}).get('choch')
        choch_list = [_choch] if _choch and isinstance(_choch, dict) else []

        for bos in bos_list:
            bos_type = bos.get('type', '')
            bos_dir = bos.get('direction', '')
            if direction == "BUY" and 'BULL' in bos_type.upper():
                score += 15
                confluence.append("BOS_BULL_CONFIRM")
                smc_confirmed = True
                break
            elif direction == "SELL" and 'BEAR' in bos_type.upper():
                score += 15
                confluence.append("BOS_BEAR_CONFIRM")
                smc_confirmed = True
                break

        if not smc_confirmed:
            for choch in choch_list:
                choch_type = choch.get('type', '')
                choch_dir = choch.get('direction', '')
                if direction == "BUY" and 'BULL' in choch_type.upper():
                    score += 12
                    confluence.append("CHoCH_BULL_CONFIRM")
                    smc_confirmed = True
                    break
                elif direction == "SELL" and 'BEAR' in choch_type.upper():
                    score += 12
                    confluence.append("CHoCH_BEAR_CONFIRM")
                    smc_confirmed = True
                    break

    if not smc_confirmed:
        # Soft fallback: check SMC bias direction
        if smc_report:
            smc_bias = smc_report.get('smc_bias', 'NEUTRAL')
            if (direction == "BUY" and smc_bias == "BULLISH") or \
               (direction == "SELL" and smc_bias == "BEARISH"):
                score += 5
                confluence.append("SMC_BIAS_ALIGNED")
                # Still proceed — not strictly mandatory
            else:
                if relaxed:
                    # Relaxed: no SMC alignment = penalty, not a kill
                    score -= 10
                    confluence.append("NO_SMC_CONFIRM_PENALTY")
                else:
                    return None  # Strict: no SMC confirmation at all
        else:
            if relaxed:
                score -= 10
                confluence.append("NO_SMC_REPORT_PENALTY")
            else:
                return None

    # ── OB/FVG Zone proximity (bonus) ───────────────────
    if smc_report:
        ob_list = [smc_report.get('nearest_ob')] if smc_report.get('nearest_ob') else []
        fvg_list = smc_report.get('quality_fvgs', [])

        # Check order blocks
        for ob in ob_list:
            ob_type = ob.get('type', '')
            ob_top = float(ob.get('top', 0))
            ob_bottom = float(ob.get('bottom', 0))
            ob_mid = (ob_top + ob_bottom) / 2

            dist = abs(close_price - ob_mid) / pip_size
            if dist > OB_FVG_PROXIMITY:
                continue

            if direction == "BUY" and 'BULL' in ob_type.upper():
                score += 8
                confluence.append(f"OB_BULL_NEAR({dist:.1f}p)")
            elif direction == "SELL" and 'BEAR' in ob_type.upper():
                score += 8
                confluence.append(f"OB_BEAR_NEAR({dist:.1f}p)")

        # Check FVGs
        for fvg in fvg_list:
            fvg_type = fvg.get('type', '')
            fvg_top = float(fvg.get('top', 0))
            fvg_bottom = float(fvg.get('bottom', 0))
            fvg_mid = (fvg_top + fvg_bottom) / 2

            dist = abs(close_price - fvg_mid) / pip_size
            if dist > OB_FVG_PROXIMITY:
                continue

            if direction == "BUY" and 'BULL' in fvg_type.upper():
                score += 6
                confluence.append(f"FVG_BULL_NEAR({dist:.1f}p)")
            elif direction == "SELL" and 'BEAR' in fvg_type.upper():
                score += 6
                confluence.append(f"FVG_BEAR_NEAR({dist:.1f}p)")

    # ── Delta divergence alignment (bonus) ─────────────
    if market_report:
        rolling_delta = market_report.get('rolling_delta', {})
        delta_bias = rolling_delta.get('bias', 'NEUTRAL')

        if direction == "BUY" and delta_bias == "BULLISH":
            score += 8
            confluence.append("DELTA_BULL_ALIGNED")
        elif direction == "SELL" and delta_bias == "BEARISH":
            score += 8
            confluence.append("DELTA_BEAR_ALIGNED")

    # ── Volume surge at reversal (bonus) ────────────────
    if market_report:
        surge = market_report.get('volume_surge', {})
        if surge.get('surge_detected', False):
            score += 8
            confluence.append("VOLUME_SURGE")

    # ── Order flow imbalance (bonus) ────────────────────
    if market_report:
        of_imb = market_report.get('order_flow_imbalance', {})
        imb = of_imb.get('imbalance', 0)
        imb_strength = of_imb.get('strength', 'NONE')

        if direction == "BUY" and imb > 0.15:
            score += 8
            confluence.append(f"OF_BULL_{imb:+.2f}")
        elif direction == "SELL" and imb < -0.15:
            score += 8
            confluence.append(f"OF_BEAR_{imb:+.2f}")

    # ── StochRSI at extremes (bonus) ────────────────────
    if df_m15 is not None and len(df_m15) >= 3:
        stoch_k = float(df_m15.iloc[-1].get('stoch_rsi_k', 50))
        prev_k = float(df_m15.iloc[-2].get('stoch_rsi_k', 50))

        if direction == "SELL" and stoch_k > 70:
            score += 8
            confluence.append("STOCHRSI_OVERBOUGHT")
            if prev_k > stoch_k:
                score += 4
                confluence.append("STOCHRSI_TURNING_DOWN")
        elif direction == "BUY" and stoch_k < 30:
            score += 8
            confluence.append("STOCHRSI_OVERSOLD")
            if prev_k < stoch_k:
                score += 4
                confluence.append("STOCHRSI_TURNING_UP")

    # ── Premium/Discount zone (bonus) ───────────────────
    if smc_report:
        pd_info = smc_report.get('premium_discount', {})
        pd_zone = pd_info.get('zone', '')

        if direction == "SELL" and "PREMIUM" in pd_zone:
            score += 8
            confluence.append("PREMIUM_ZONE_SELL")
        elif direction == "BUY" and "DISCOUNT" in pd_zone:
            score += 8
            confluence.append("DISCOUNT_ZONE_BUY")

    # ── Choppy market penalty ───────────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        if momentum.get('is_choppy', False):
            score -= 15
            confluence.append("CHOPPY_PENALTY")

    # ── Score threshold ─────────────────────────────────
    min_confluence = 3 if relaxed else 5
    min_score = (MIN_SCORE - 18) if relaxed else MIN_SCORE  # Relaxed: 50 vs 68

    if len(confluence) < min_confluence:
        return None

    if score < min_score:
        return None

    # ── Calculate SL/TP ─────────────────────────────────
    entry = close_price

    sl_pips = round(atr_pips * 1.2, 1)
    sl_pips = max(sl_pips, 3.0)

    tp1_pips = round(sl_pips * 2.0, 1)
    tp2_pips = round(sl_pips * 3.5, 1)

    if direction == "BUY":
        sl_price  = round(entry - sl_pips * pip_size, 5)
        tp1_price = round(entry + tp1_pips * pip_size, 5)
        tp2_price = round(entry + tp2_pips * pip_size, 5)
    else:
        sl_price  = round(entry + sl_pips * pip_size, 5)
        tp1_price = round(entry - tp1_pips * pip_size, 5)
        tp2_price = round(entry - tp2_pips * pip_size, 5)

    log.info(f"[{STRATEGY_NAME} v{VERSION}] {direction} {symbol}"
             f" entry={entry:.5f} Score:{score} | "
             f"{', '.join(confluence)}")

    return {
        "direction":   direction,
        "entry_price": entry,
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
        "spread":      0,
    }
