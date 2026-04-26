# =============================================================
# ai_engine/breakout_strategy_model.py  v1.0 — Layer 1 Breakout Model
#
# PURPOSE: Specialized Layer 1 model for BREAKOUT_MOMENTUM strategy.
# Replaces the hard-coded gates in strategies/breakout_momentum.py
# with a learned model that knows which breakout signals actually work.
#
# BREAKOUT-SPECIFIC FEATURES (16 internal + 4 general + 5 cross-strategy = 25):
#   - consol_type (TIGHT_LOW_ADX / WIDE_LOW_ADX / TIGHT_RANGE / RELAXED_WIDE)
#   - range_pips (consolidation width)
#   - adx (at breakout time)
#   - atr_pips (M15 ATR in pips)
#   - atr_ratio (ATR expansion: current/previous)
#   - retest (price retested broken level)
#   - delta_confirms (flow direction matches breakout)
#   - of_imbalance (order flow imbalance value)
#   - of_strength (order flow strength level)
#   - vol_surge (volume surge detected)
#   - vol_surge_ratio (surge magnitude)
#   - h4_trend_aligned (H4 EMA alignment)
#   - h4_supertrend (H4 supertrend aligned)
#   - m5_momentum (M5 momentum candle aligned)
#   - bos_aligned (SMC break-of-structure aligned)
#   - is_choppy (market is choppy)
#
# WHY BREAKOUT SECOND:
#   - Breakout has moderate trade count (more data than VWAP)
#   - Different signal profile: expansion-based, not reversion-based
#   - Hard-coded delta confirmation gate may be too strict
#   - Model can learn optimal retest tolerance + volume requirements
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy BREAKOUT_MOMENTUM
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds breakout-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "BREAKOUT_MOMENTUM"
STRATEGY_KEY = "breakout"

# Consolidation type encoding
CONSOL_TYPE_MAP = {
    'TIGHT_LOW_ADX':  1.0,
    'WIDE_LOW_ADX':   0.5,
    'TIGHT_RANGE':    0.75,
    'RELAXED_WIDE':   0.25,
    'NONE':           0.0,
}

# OF strength encoding
OF_STRENGTH_MAP = {
    'EXTREME':  1.0,
    'STRONG':   0.75,
    'MODERATE': 0.5,
    'WEAK':     0.25,
    'NONE':     0.0,
}


# ════════════════════════════════════════════════════════════════
# BREAKOUT NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    Breakout evaluation WITHOUT hard-coded gates.

    Generates ALL potential breakout signals (buy/sell setups)
    regardless of gate conditions. The Layer 1 model will then
    decide which ones to PASS.

    Removed gates vs original breakout_momentum.py v1.0:
      - ATR filter (atr_pips < 2.0) -> REMOVED
      - Consolidation type filter -> REMOVED (all types accepted)
      - Delta confirmation (strict) -> REMOVED (becomes scoring)
      - Score threshold (70/55) -> REMOVED
      - Confluence minimum (5/3) -> REMOVED
      - R:R gate (tp1 >= sl * 1.5) -> KEPT (prevents nonsensical trades)

    Returns: signal dict or None
    """
    if df_m15 is None or df_h1 is None or market_report is None:
        return None
    if len(df_m15) < 30 or len(df_h1) < 30:
        return None

    from strategies.breakout_momentum import (
        _get_pip_size, _detect_consolidation, _detect_breakout,
        RANGE_LOOKBACK, RETEST_TOLERANCE, MIN_RANGE_PIPS,
    )

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    # ── Step 1: Detect consolidation (always use lagged to avoid lookahead) ──
    consol = _detect_consolidation(df_h1, use_lagged=True,
                                    lag_bars=RANGE_LOOKBACK, symbol=symbol)

    if not consol.get('consolidating'):
        # Relaxed: accept wider ranges too
        if consol.get('range_pips', 0) < 80 and consol.get('range_pips', 0) >= 5:
            consol = {
                'consolidating': True,
                'type': 'RELAXED_WIDE',
                'range_high': consol.get('range_high', 0),
                'range_low': consol.get('range_low', 0),
                'range_pips': consol.get('range_pips', 0),
                'adx': consol.get('adx', 0),
                'pip_size': pip_size,
            }
        else:
            return None

    # ── Step 2: Detect breakout + retest ──
    breakout = _detect_breakout(df_m15, consol['range_high'],
                                consol['range_low'], pip_size,
                                retest_tolerance=8.0)  # wider tolerance for data collection

    if breakout is None:
        return None

    direction = breakout['direction']
    range_pips = breakout['range_pips']
    retest = breakout['retest']

    # ── Collect ALL features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 10
    confluence.append(f"CONSOL_{consol['type']}")
    confluence.append(f"RANGE_{consol['range_pips']}p_ADX_{consol['adx']:.0f}")

    score += 15
    confluence.append(f"BREAKOUT_{direction}")

    if retest:
        score += 12
        confluence.append("RETEST_CONFIRMED")
    else:
        score -= 5
        confluence.append("NO_RETEST_CHASE")

    # Delta / order flow (scoring, not blocking)
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
        else:
            score -= 12
            confluence.append("NO_DELTA_CONFIRM")
    elif direction == "SELL":
        if delta_bias == "BEARISH":
            delta_confirms = True
            score += 12
            confluence.append("DELTA_BEAR")
        elif imb < -0.1 and of_strength in ('STRONG', 'MODERATE', 'EXTREME'):
            delta_confirms = True
            score += 10
            confluence.append(f"OF_BEAR_{imb:+.2f}")
        else:
            score -= 12
            confluence.append("NO_DELTA_CONFIRM")

    # Volume surge (scoring)
    surge = market_report.get('volume_surge', {})
    vol_surge_detected = surge.get('surge_detected', False)
    vol_surge_ratio = surge.get('surge_ratio', 1.0)
    if vol_surge_detected:
        score += 10
        confluence.append(f"VOL_SURGE_{vol_surge_ratio:.1f}x")
    else:
        score -= 8
        confluence.append("NO_VOLUME_SURGE")

    # ATR expansion (scoring)
    atr_ratio = 1.0
    if len(df_m15) >= 20:
        atr_now = float(df_m15.iloc[-1].get('atr', 0))
        atr_prev = float(df_m15.iloc[-20].get('atr', 0))
        if atr_prev > 0:
            atr_ratio = atr_now / atr_prev
            if atr_ratio >= 1.2:
                score += 8
                confluence.append(f"ATR_EXPAND_{atr_ratio:.1f}x")
            else:
                score -= 3
                confluence.append("ATR_NOT_EXPANDING")

    # H4 trend alignment (scoring)
    h4_trend_aligned = False
    h4_supertrend = False
    if df_h4 is not None and len(df_h4) >= 20:
        h4 = df_h4.iloc[-1]
        h4_ema9 = float(h4.get('ema_9', 0))
        h4_ema21 = float(h4.get('ema_21', 0))
        h4_st = int(h4.get('supertrend_dir', 0))

        if (direction == "BUY" and h4_ema9 > h4_ema21) or \
           (direction == "SELL" and h4_ema9 < h4_ema21):
            h4_trend_aligned = True
            score += 8
            confluence.append("H4_TREND_ALIGN")
        if (direction == "BUY" and h4_st == 1) or \
           (direction == "SELL" and h4_st == -1):
            h4_supertrend = True
            score += 5
            confluence.append("H4_SUPERTREND")

    # M5 momentum (scoring)
    m5_momentum = False
    if df_m5 is not None and len(df_m5) >= 3:
        m5_body = df_m5.iloc[-1]['close'] - df_m5.iloc[-1]['open']
        if (direction == "BUY" and m5_body > 0) or \
           (direction == "SELL" and m5_body < 0):
            m5_momentum = True
            score += 5
            confluence.append("M5_MOMENTUM")

    # SMC BOS (scoring)
    bos_aligned = False
    if smc_report:
        _bos = smc_report.get('structure', {}).get('bos')
        bos_list = [_bos] if _bos and isinstance(_bos, dict) else []
        for bos in bos_list:
            bos_type = bos.get('type', '').upper()
            if direction == "BUY" and 'BULL' in bos_type:
                bos_aligned = True
                score += 8
                confluence.append("BOS_BULL_BREAKOUT")
                break
            elif direction == "SELL" and 'BEAR' in bos_type:
                bos_aligned = True
                score += 8
                confluence.append("BOS_BEAR_BREAKOUT")
                break

    # Choppy penalty (scoring)
    is_choppy = False
    if master_report:
        momentum = master_report.get('momentum', {})
        if momentum.get('is_choppy', False):
            is_choppy = True
            score -= 15
            confluence.append("CHOPPY")

    # ── SL/TP calculation ──
    entry = close_price
    if direction == "BUY":
        sl_pips = max(5.0, round((entry - consol['range_low']) / pip_size + 2, 1))
        tp1_pips = round(range_pips * 1.0, 1)
        tp2_pips = round(range_pips * 1.5, 1)
    else:
        sl_pips = max(5.0, round((consol['range_high'] - entry) / pip_size + 2, 1))
        tp1_pips = round(range_pips * 1.0, 1)
        tp2_pips = round(range_pips * 1.5, 1)

    # Keep minimum R:R check
    if tp1_pips / sl_pips < 1.5:
        tp1_pips = round(sl_pips * 2.0, 1)
        tp2_pips = round(sl_pips * 3.0, 1)

    if direction == "BUY":
        sl_price = round(entry - sl_pips * pip_size, 5)
        tp1_price = round(entry + tp1_pips * pip_size, 5)
        tp2_price = round(entry + tp2_pips * pip_size, 5)
    else:
        sl_price = round(entry + sl_pips * pip_size, 5)
        tp1_price = round(entry - tp1_pips * pip_size, 5)
        tp2_price = round(entry - tp2_pips * pip_size, 5)

    log.info(f"[{STRATEGY_NAME}:NOGATE] {direction} {symbol} Score:{score} | {', '.join(confluence)}")

    return {
        "direction": direction, "entry_price": entry,
        "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
        "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
        "strategy": STRATEGY_NAME, "version": "2.0-no-gate",
        "score": score, "confluence": confluence,
        "_breakout_features": {
            'consol_type': consol['type'],
            'range_pips': consol['range_pips'],
            'adx': consol['adx'],
            'atr_pips': atr_pips,
            'atr_ratio': atr_ratio,
            'retest': 1 if retest else 0,
            'dist_to_level': breakout.get('dist_to_level_pips', 0),
            'delta_confirms': 1 if delta_confirms else 0,
            'of_imbalance': imb,
            'of_strength': of_strength,
            'vol_surge': 1 if vol_surge_detected else 0,
            'vol_surge_ratio': vol_surge_ratio,
            'h4_trend_aligned': 1 if h4_trend_aligned else 0,
            'h4_supertrend': 1 if h4_supertrend else 0,
            'm5_momentum': 1 if m5_momentum else 0,
            'bos_aligned': 1 if bos_aligned else 0,
            'is_choppy': 1 if is_choppy else 0,
        }
    }


# ════════════════════════════════════════════════════════════════
# BREAKOUT FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_breakout_features_from_db(row: dict) -> dict:
    """
    Extract breakout-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_breakout_features).

    Falls back to computed defaults for trades without breakout feature rows.
    """
    # If the row has breakout features (from JOIN), use them directly
    if row.get('consol_type') is not None and row.get('consol_type') != '':
        bf = {
            'consol_type': str(row.get('consol_type', 'NONE')),
            'range_pips': float(row.get('range_pips', 0) or 0),
            'adx': float(row.get('adx', 25) or 25),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
            'atr_ratio': float(row.get('atr_ratio', 1.0) or 1.0),
            'retest': int(row.get('retest', 0) or 0),
            'dist_to_level': float(row.get('dist_to_level', 0) or 0),
            'delta_confirms': int(row.get('delta_confirms', 0) or 0),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'vol_surge': int(row.get('vol_surge', 0) or 0),
            'vol_surge_ratio': float(row.get('vol_surge_ratio', 1.0) or 1.0),
            'h4_trend_aligned': int(row.get('h4_trend_aligned', 0) or 0),
            'h4_supertrend': int(row.get('h4_supertrend', 0) or 0),
            'm5_momentum': int(row.get('m5_momentum', 0) or 0),
            'bos_aligned': int(row.get('bos_aligned', 0) or 0),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
        }
    else:
        # Fallback for older trades without backtest_breakout_features rows
        bf = {
            'consol_type': 'NONE',
            'range_pips': float(row.get('va_width_pips', 20) or 20),
            'adx': 25.0,
            'atr_pips': float(row.get('atr', 0) or 0),
            'atr_ratio': 1.0,
            'retest': 0,
            'dist_to_level': 0,
            'delta_confirms': 0,
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'vol_surge': int(row.get('vol_surge_detected', 0) or 0),
            'vol_surge_ratio': float(row.get('vol_surge_ratio', 1.0) or 1.0),
            'h4_trend_aligned': 0,
            'h4_supertrend': 0,
            'm5_momentum': 0,
            'bos_aligned': 0,
            'is_choppy': int(row.get('is_choppy', 0) or 0),
        }
    return bf


def build_breakout_feature_vector(row: dict) -> list:
    """Build breakout model feature vector from a DB row (with JOINed breakout features).

    Returns a list of numeric features for the Breakout Layer 1 model.
    Uses real breakout features when available, falls back to derived values.
    """
    bf = extract_breakout_features_from_db(row)

    features = [
        # ── Breakout-specific internal features (16) ──
        CONSOL_TYPE_MAP.get(bf.get('consol_type', 'NONE'), 0.0),
        bf.get('range_pips', 20) / 80.0,
        bf.get('adx', 25) / 60.0,
        bf.get('atr_pips', 0) / 30.0,
        min(bf.get('atr_ratio', 1.0), 3.0) / 3.0,
        float(bf.get('retest', 0)),
        bf.get('dist_to_level', 0) / 20.0,
        float(bf.get('delta_confirms', 0)),
        bf.get('of_imbalance', 0) / 1.0,  # can be negative
        OF_STRENGTH_MAP.get(str(bf.get('of_strength', 'NONE')), 0.0),
        float(bf.get('vol_surge', 0)),
        min(bf.get('vol_surge_ratio', 1.0), 5.0) / 5.0,
        float(bf.get('h4_trend_aligned', 0)),
        float(bf.get('h4_supertrend', 0)),
        float(bf.get('m5_momentum', 0)),
        float(bf.get('bos_aligned', 0)),
        float(bf.get('is_choppy', 0)),

        # ── General features from backtest_trades (4) ──
        float(row.get('atr', 0) or 0) / 30.0,
        abs(float(row.get('pip_from_vwap', 0) or 0)) / 50.0,
        abs(float(row.get('pip_to_poc', 0) or 0)) / 50.0,
        float(row.get('va_width_pips', 20) or 20) / 50.0,

        # ── Cross-strategy confluence (5) from strategy score columns ──
        float(row.get('ss_smc_ob', 0) or 0) / 100.0,
        float(row.get('ss_ema_cross', 0) or 0) / 100.0,
        float(row.get('ss_vwap_reversion', 0) or 0) / 100.0,
        float(row.get('ss_fvg_reversion', 0) or 0) / 100.0,
        float(row.get('ss_trend_continuation', 0) or 0) / 100.0,
    ]
    return features
