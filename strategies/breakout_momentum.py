# =============================================================
# strategies/breakout_momentum.py  v1.0
# Strategy 14: Breakout Momentum (SMC_STRUCTURE group)
#
# Purpose: Catches expansion moves — enters on breakout from
# consolidation with institutional confirmation. Complements
# existing SMC strategies which are mostly reversion-based.
# This is EXPANSION-based: it thrives when price LEAVES a zone.
#
# Entry logic:
#   1. H1 consolidation detection (ADX < 20 or Bollinger squeeze)
#   2. M15 breakout candle closes outside range
#   3. Retest of broken level (smart money entry, not retail chase)
#   4. Delta confirms breakout direction
#   5. Volume surge confirms institutional move
#   6. ATR expansion after breakout
#
# Win rate target: 40-50% (compensated by 3:1+ R:R)
# Best session: LONDON_OPEN, LONDON_SESSION, NY_LONDON_OVERLAP
# Best state:  BREAKOUT_ACCEPTED, TRENDING_STRONG
# =============================================================

import pandas as pd
import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "BREAKOUT_MOMENTUM"
MIN_SCORE     = 70
VERSION       = "1.0"

# --- Parameters ---
RANGE_LOOKBACK     = 24     # H1 bars to define consolidation range
BREAKOUT_BUFFER    = 0.3    # ATR multiplier for breakout confirmation
RETEST_TOLERANCE   = 3.0    # Pips tolerance for retest detection
MIN_ATR_EXPANSION  = 1.2    # ATR must expand by this ratio post-breakout
MIN_RANGE_PIPS     = 8.0    # Minimum consolidation range in pips to trade


def _get_pip_size(price: float) -> float:
    if price > 500:     return 1.0
    elif price > 50:    return 0.01
    else:               return 0.0001


def _detect_consolidation(df_h1: pd.DataFrame, lookback: int = RANGE_LOOKBACK) -> dict:
    """
    Detect consolidation range on H1.
    Returns dict with range info or None if not consolidating.
    """
    if df_h1 is None or len(df_h1) < lookback + 5:
        return {"consolidating": False}

    recent = df_h1.tail(lookback).copy()
    current_close = float(df_h1.iloc[-1]['close'])
    pip_size = _get_pip_size(current_close)

    # Range = highest high and lowest low in lookback
    range_high = float(recent['high'].max())
    range_low  = float(recent['low'].min())
    range_pips = (range_high - range_low) / pip_size

    # ADX check — consolidation = low ADX
    adx = float(df_h1.iloc[-1].get('adx', 0))

    is_tight = range_pips < 30  # Tight range
    is_wide  = range_pips < 50  # Wider range still valid

    consolidating = False
    consolidation_type = "NONE"

    if adx < 20 and is_tight:
        consolidating = True
        consolidation_type = "TIGHT_LOW_ADX"
    elif adx < 25 and is_wide:
        consolidating = True
        consolidation_type = "WIDE_LOW_ADX"
    elif is_tight and range_pips < 20:
        consolidating = True
        consolidation_type = "TIGHT_RANGE"

    return {
        "consolidating": consolidating,
        "type": consolidation_type,
        "range_high": range_high,
        "range_low": range_low,
        "range_pips": round(range_pips, 1),
        "adx": adx,
        "pip_size": pip_size,
    }


def _detect_breakout(df_m15: pd.DataFrame,
                     range_high: float, range_low: float,
                     pip_size: float) -> dict | None:
    """
    Detect if price has broken out of the consolidation range
    on M15 and is now retesting the broken level.
    """
    if df_m15 is None or len(df_m15) < 10:
        return None

    current_close = float(df_m15.iloc[-1]['close'])
    current_low   = float(df_m15.iloc[-1]['low'])
    current_high  = float(df_m15.iloc[-1]['high'])

    prev_close = float(df_m15.iloc[-2]['close'])
    prev_low   = float(df_m15.iloc[-2]['low'])
    prev_high  = float(df_m15.iloc[-2]['high'])

    # Check for bullish breakout (price was below range, now above)
    # Breakout = candle closed above range high
    # Retest = current price is near or at the broken level
    bullish_breakout = False
    bearish_breakout = False

    # Look at recent 3 candles for breakout
    for i in range(max(0, len(df_m15) - 5), len(df_m15)):
        c = df_m15.iloc[i]
        c_close = float(c['close'])
        c_high  = float(c['high'])
        c_low   = float(c['low'])

        # Bullish breakout: candle closes above range high
        if c_close > range_high and c_low > range_low:
            bullish_breakout = True

        # Bearish breakout: candle closes below range low
        if c_close < range_low and c_high < range_high:
            bearish_breakout = True

    if not bullish_breakout and not bearish_breakout:
        return None

    # Determine direction based on current price position
    if bullish_breakout and current_close > range_high:
        direction = "BUY"
        broken_level = range_high
        # Check for retest: price pulled back near broken level
        dist_to_level = (current_close - range_high) / pip_size
        retest = 0 <= dist_to_level <= RETEST_TOLERANCE
    elif bearish_breakout and current_close < range_low:
        direction = "SELL"
        broken_level = range_low
        dist_to_level = (range_low - current_close) / pip_size
        retest = 0 <= dist_to_level <= RETEST_TOLERANCE
    else:
        return None

    range_size = (range_high - range_low) / pip_size
    if range_size < MIN_RANGE_PIPS:
        return None

    return {
        "direction": direction,
        "broken_level": broken_level,
        "retest": retest,
        "dist_to_level_pips": round(dist_to_level, 1),
        "range_pips": round(range_size, 1),
    }


def evaluate(symbol: str,
             df_m1: pd.DataFrame = None,
             df_m5: pd.DataFrame = None,
             df_m15: pd.DataFrame = None,
             df_h1: pd.DataFrame = None,
             smc_report: dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None) -> dict | None:
    """
    Breakout Momentum Strategy:
    Enters after price breaks out of consolidation with retest entry.
    """
    if df_m15 is None or df_h1 is None:
        return None
    if len(df_m15) < 30 or len(df_h1) < 30:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 2.0:
        return None

    # ── Step 1: Detect consolidation on H1 (mandatory) ─
    consol = _detect_consolidation(df_h1)

    if not consol['consolidating']:
        return None

    score = 0
    confluence = []

    score += 10
    confluence.append(f"CONSOL_{consol['type']}")
    confluence.append(f"RANGE_{consol['range_pips']}p_ADX_{consol['adx']:.0f}")

    # ── Step 2: Detect breakout + retest on M15 ─────────
    breakout = _detect_breakout(df_m15, consol['range_high'],
                                consol['range_low'], pip_size)

    if breakout is None:
        return None

    direction = breakout['direction']
    range_pips = breakout['range_pips']
    retest = breakout['retest']

    score += 15
    confluence.append(f"BREAKOUT_{direction}")

    if retest:
        score += 12
        confluence.append("RETEST_CONFIRMED")
    else:
        # Not at retest yet — score lower, might be chasing
        score -= 5
        confluence.append("NO_RETEST_CHASE")

    # ── Step 3: Delta confirms breakout direction ───────
    rolling_delta = market_report.get('rolling_delta', {})
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')
    of_imb = market_report.get('order_flow_imbalance', {})
    imb = of_imb.get('imbalance', 0)
    of_strength = of_imb.get('strength', 'NONE')

    delta_confirms = False
    if direction == "BUY":
        if delta_bias == "BULLISH":
            delta_confirms = True
            score += 12
            confluence.append("DELTA_BULL")
        elif imb > 0.1 and of_strength in ('STRONG', 'MODERATE', 'EXTREME'):
            delta_confirms = True
            score += 10
            confluence.append(f"OF_BULL_{imb:+.2f}")
    elif direction == "SELL":
        if delta_bias == "BEARISH":
            delta_confirms = True
            score += 12
            confluence.append("DELTA_BEAR")
        elif imb < -0.1 and of_strength in ('STRONG', 'MODERATE', 'EXTREME'):
            delta_confirms = True
            score += 10
            confluence.append(f"OF_BEAR_{imb:+.2f}")

    if not delta_confirms:
        return None  # Must have flow confirmation for breakout

    # ── Step 4: Volume surge confirmation ───────────────
    surge = market_report.get('volume_surge', {})
    if surge.get('surge_detected', False):
        score += 10
        confluence.append(f"VOL_SURGE_{surge.get('surge_ratio', 0):.1f}x")
    else:
        # Volume is important for breakouts — penalty if missing
        score -= 8
        confluence.append("NO_VOLUME_SURGE_PENALTY")

    # ── Step 5: ATR expansion check ─────────────────────
    if len(df_m15) >= 20:
        atr_now = float(df_m15.iloc[-1].get('atr', 0))
        atr_prev = float(df_m15.iloc[-20].get('atr', 0))
        if atr_prev > 0:
            atr_ratio = atr_now / atr_prev
            if atr_ratio >= MIN_ATR_EXPANSION:
                score += 8
                confluence.append(f"ATR_EXPAND_{atr_ratio:.1f}x")
            else:
                score -= 3
                confluence.append("ATR_NOT_EXPANDING")

    # ── Step 6: H4 trend alignment (bonus) ─────────────
    if df_h4 is not None and len(df_h4) >= 20:
        h4 = df_h4.iloc[-1]
        h4_ema9  = float(h4.get('ema_9', 0))
        h4_ema21 = float(h4.get('ema_21', 0))
        h4_st = int(h4.get('supertrend_dir', 0))

        if direction == "BUY" and h4_ema9 > h4_ema21:
            score += 8
            confluence.append("H4_BULL_ALIGN")
        elif direction == "SELL" and h4_ema9 < h4_ema21:
            score += 8
            confluence.append("H4_BEAR_ALIGN")

        if (direction == "BUY" and h4_st == 1) or \
           (direction == "SELL" and h4_st == -1):
            score += 5
            confluence.append("H4_SUPERTREND")

    # ── Step 7: M5 momentum candle (bonus) ─────────────
    if df_m5 is not None and len(df_m5) >= 3:
        m5_last = df_m5.iloc[-1]
        m5_body = m5_last['close'] - m5_last['open']
        if (direction == "BUY" and m5_body > 0) or \
           (direction == "SELL" and m5_body < 0):
            score += 5
            confluence.append("M5_MOMENTUM")

    # ── Step 8: SMC structure confirmation (bonus) ──────
    if smc_report:
        _bos = smc_report.get('structure', {}).get('bos')
        bos_list = [_bos] if _bos and isinstance(_bos, dict) else []
        for bos in bos_list:
            bos_type = bos.get('type', '').upper()
            if direction == "BUY" and 'BULL' in bos_type:
                score += 8
                confluence.append("BOS_BULL_BREAKOUT")
                break
            elif direction == "SELL" and 'BEAR' in bos_type:
                score += 8
                confluence.append("BOS_BEAR_BREAKOUT")
                break

    # ── Choppy market penalty ───────────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        if momentum.get('is_choppy', False):
            score -= 15
            confluence.append("CHOPPY_PENALTY")

    if len(confluence) < 5:
        return None

    # ── Score threshold ─────────────────────────────────
    if score < MIN_SCORE:
        return None

    # ── Calculate SL/TP ─────────────────────────────────
    entry = close_price

    # SL: inside the consolidation range (if retest, SL on other side)
    if direction == "BUY":
        sl_pips = max(5.0, round((entry - consol['range_low']) / pip_size + 2, 1))
        tp1_pips = round(range_pips * 1.0, 1)   # Measured move
        tp2_pips = round(range_pips * 1.5, 1)
    else:
        sl_pips = max(5.0, round((consol['range_high'] - entry) / pip_size + 2, 1))
        tp1_pips = round(range_pips * 1.0, 1)
        tp2_pips = round(range_pips * 1.5, 1)

    # Ensure minimum RR
    if tp1_pips / sl_pips < 1.5:
        tp1_pips = round(sl_pips * 2.0, 1)
        tp2_pips = round(sl_pips * 3.0, 1)

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
