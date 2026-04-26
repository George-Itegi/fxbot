# =============================================================
# ai_engine/ema_cross_strategy_model.py  v1.0 — Layer 1 EMA Cross Model
#
# PURPOSE: Specialized Layer 1 model for EMA_CROSS_MOMENTUM strategy.
# Replaces the hard-coded gates in strategies/ema_cross_momentum.py
# with a learned model that knows which EMA cross signals work.
#
# EMA_CROSS-SPECIFIC FEATURES (14 internal + 4 general + 5 cross-strategy = 23):
#   - h4_cross_bars_ago (bars since H4 EMA 9/21 cross)
#   - h4_cross_strength (strength of the cross: 0-100)
#   - h4_alignment_score (H4 EMA alignment score: 0-100)
#   - h1_rsi (H1 RSI value)
#   - m15_adx (M15 ADX value)
#   - delta_bias (BULLISH / BEARISH / NEUTRAL)
#   - of_imbalance (order flow imbalance value)
#   - of_strength (order flow strength level)
#   - h1_supertrend_dir (H1 supertrend direction)
#   - h4_supertrend_dir (H4 supertrend direction)
#   - h4_ema_spread_9_21 (distance between H4 EMA 9 and EMA 21 in pips)
#   - is_choppy (market is choppy)
#   - vol_surge (volume surge detected)
#   - atr_pips (M15 ATR in pips)
#
# WHY EMA_CROSS:
#   - EMA crosses are classic momentum signals
#   - Post-cross timing and confirmation gates may be suboptimal
#   - Model can learn optimal entry delay and confirmation levels
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy EMA_CROSS_MOMENTUM
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds ema_cross-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "EMA_CROSS_MOMENTUM"
STRATEGY_KEY = "ema_cross"

# OF strength encoding
OF_STRENGTH_MAP = {
    'EXTREME':  1.0,
    'STRONG':   0.75,
    'MODERATE': 0.5,
    'WEAK':     0.25,
    'NONE':     0.0,
}


# ════════════════════════════════════════════════════════════════
# EMA CROSS NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    EMA cross evaluation WITHOUT hard-coded gates.

    Generates ALL potential EMA cross momentum signals regardless of
    gate conditions. The Layer 1 model will decide which to PASS.

    Simplified detection: look for H4 EMA 9/21 crossover, score
    momentum continuation probability with multi-timeframe data.

    Returns: signal dict with _ema_cross_features or None
    """
    if df_h4 is None or df_h1 is None or df_m15 is None:
        return None
    if len(df_h4) < 30:
        return None

    h4 = df_h4.iloc[-1]
    h1 = df_h1.iloc[-1]
    m15 = df_m15.iloc[-1]
    close_price = float(m15['close'])

    from core.pip_utils import get_pip_size as _gps
    pip_size = _gps(symbol, close_price)

    atr_pips = float(m15.get('atr', 0)) / pip_size if pip_size > 0 else 0
    h1_rsi = float(h1.get('rsi', 50))
    m15_adx = float(m15.get('adx', 25))

    if atr_pips < 1.0:
        return None

    # ── Detect H4 EMA 9/21 cross ──
    h4_ema9 = df_h4['ema_9'].astype(float).values
    h4_ema21 = df_h4['ema_21'].astype(float).values
    h4_closes = df_h4['close'].astype(float).values

    if len(h4_ema9) < 10:
        return None

    # Find most recent cross
    cross_bars_ago = None
    cross_direction = None  # 'bullish' or 'bearish'

    for i in range(len(h4_ema9) - 2, max(len(h4_ema9) - 20, -1), -1):
        prev_diff = h4_ema9[i] - h4_ema21[i]
        curr_diff = h4_ema9[i + 1] - h4_ema21[i + 1]
        if prev_diff <= 0 and curr_diff > 0:
            cross_bars_ago = len(h4_ema9) - 1 - (i + 1)
            cross_direction = 'bullish'
            break
        elif prev_diff >= 0 and curr_diff < 0:
            cross_bars_ago = len(h4_ema9) - 1 - (i + 1)
            cross_direction = 'bearish'
            break

    if cross_bars_ago is None or cross_bars_ago > 10:
        # Relaxed: also accept if EMAs are spreading (momentum building)
        curr_spread = h4_ema9[-1] - h4_ema21[-1]
        prev_spread = h4_ema9[-3] - h4_ema21[-3]
        if abs(curr_spread) > abs(prev_spread) and abs(curr_spread / pip_size) > 3:
            cross_bars_ago = 0
            cross_direction = 'bullish' if curr_spread > 0 else 'bearish'
        else:
            return None

    direction = "BUY" if cross_direction == 'bullish' else "SELL"

    # Cross strength: how much separation has developed
    curr_ema9 = h4_ema9[-1]
    curr_ema21 = h4_ema21[-1]
    h4_ema_spread_9_21 = round(abs(curr_ema9 - curr_ema21) / pip_size, 2)
    h4_cross_strength = min(100, int(h4_ema_spread_9_21 / atr_pips * 20))

    # H4 alignment score: how many H4 EMAs are aligned
    h4_ema50 = float(h4.get('ema_50', 0))
    h4_alignment_score = 0
    if direction == "BUY":
        if curr_ema9 > curr_ema21:
            h4_alignment_score += 33
        if curr_ema21 > h4_ema50:
            h4_alignment_score += 33
        if float(h4['close']) > curr_ema9:
            h4_alignment_score += 34
    else:
        if curr_ema9 < curr_ema21:
            h4_alignment_score += 33
        if curr_ema21 < h4_ema50:
            h4_alignment_score += 33
        if float(h4['close']) < curr_ema9:
            h4_alignment_score += 34

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"EMA_CROSS_{direction.upper()}_{cross_bars_ago}bars")

    if cross_bars_ago <= 3:
        score += 10
        confluence.append("FRESH_CROSS")
    elif cross_bars_ago > 5:
        score -= 5
        confluence.append("STALE_CROSS")

    if h4_cross_strength > 60:
        score += 10
        confluence.append(f"STRONG_CROSS_{h4_cross_strength}")
    elif h4_cross_strength > 30:
        score += 5
        confluence.append(f"MOD_CROSS_{h4_cross_strength}")

    if h4_alignment_score >= 80:
        score += 12
        confluence.append(f"H4_ALIGNED_{h4_alignment_score}")
    elif h4_alignment_score >= 50:
        score += 5
        confluence.append(f"H4_PARTIAL_{h4_alignment_score}")

    # RSI (scoring)
    if direction == "BUY":
        if 40 < h1_rsi < 65:
            score += 8
            confluence.append(f"RSI_OK_{h1_rsi:.0f}")
        elif h1_rsi > 75:
            score -= 8
            confluence.append(f"RSI_OVERBOUGHT_{h1_rsi:.0f}")
    else:
        if 35 < h1_rsi < 60:
            score += 8
            confluence.append(f"RSI_OK_{h1_rsi:.0f}")
        elif h1_rsi < 25:
            score -= 8
            confluence.append(f"RSI_OVERSOLD_{h1_rsi:.0f}")

    # ADX (scoring)
    if m15_adx > 20:
        score += 5
        confluence.append(f"ADX_TRENDING_{m15_adx:.0f}")
    else:
        score -= 5
        confluence.append(f"ADX_LOW_{m15_adx:.0f}")

    # Delta (scoring)
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')
    if (direction == "BUY" and delta_bias == "BULLISH") or \
       (direction == "SELL" and delta_bias == "BEARISH"):
        score += 12
        confluence.append("DELTA_CONFIRMS")
    else:
        score -= 8
        confluence.append("NO_DELTA")

    # Order flow (scoring)
    of_imb = market_report.get('order_flow_imbalance', {}) if market_report else {}
    of_imbalance = float(of_imb.get('imbalance', 0))
    of_strength = str(of_imb.get('strength', 'NONE'))

    if (direction == "BUY" and of_imbalance > 0.05) or \
       (direction == "SELL" and of_imbalance < -0.05):
        score += 8
        confluence.append(f"OF_ALIGN_{of_imbalance:+.2f}")

    # Supertrend (scoring)
    h1_supertrend_dir = int(h1.get('supertrend_dir', 0))
    h4_supertrend_dir = int(h4.get('supertrend_dir', 0))
    if (direction == "BUY" and h1_supertrend_dir == 1) or \
       (direction == "SELL" and h1_supertrend_dir == -1):
        score += 8
        confluence.append("H1_ST_ALIGN")
    if (direction == "BUY" and h4_supertrend_dir == 1) or \
       (direction == "SELL" and h4_supertrend_dir == -1):
        score += 5
        confluence.append("H4_ST_ALIGN")

    # Choppy (scoring)
    is_choppy = 0
    if master_report:
        momentum = master_report.get('momentum', {})
        is_choppy = 1 if momentum.get('is_choppy', False) else 0
        if is_choppy:
            score -= 15
            confluence.append("CHOPPY")

    # Volume surge (scoring)
    vol_surge = 0
    surge = market_report.get('volume_surge', {}) if market_report else {}
    if surge.get('surge_detected', False):
        vol_surge = 1
        score += 8
        confluence.append(f"VOL_SURGE_{surge.get('surge_ratio', 1.0):.1f}x")

    # ── SL/TP calculation ──
    sl_pips = max(5.0, round(atr_pips * 1.5, 1))
    tp1_pips = round(h4_ema_spread_9_21 * 1.5 if h4_ema_spread_9_21 > 0 else sl_pips * 2, 1)
    tp1_pips = max(tp1_pips, round(sl_pips * 2, 1))
    tp2_pips = round(tp1_pips * 1.5, 1)

    if direction == "BUY":
        sl_price = round(close_price - sl_pips * pip_size, 5)
        tp1_price = round(close_price + tp1_pips * pip_size, 5)
        tp2_price = round(close_price + tp2_pips * pip_size, 5)
    else:
        sl_price = round(close_price + sl_pips * pip_size, 5)
        tp1_price = round(close_price - tp1_pips * pip_size, 5)
        tp2_price = round(close_price - tp2_pips * pip_size, 5)

    # Keep minimum R:R check
    if tp1_pips < sl_pips * 1.5:
        tp1_pips = round(sl_pips * 2.0, 1)
        tp2_pips = round(sl_pips * 3.0, 1)

    log.info(f"[{STRATEGY_NAME}:NOGATE] {direction} {symbol} Score:{score} | {', '.join(confluence)}")

    return {
        "direction": direction, "entry_price": close_price,
        "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
        "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
        "strategy": STRATEGY_NAME, "version": "2.0-no-gate",
        "score": score, "confluence": confluence,
        "_ema_cross_features": {
            'h4_cross_bars_ago': cross_bars_ago,
            'h4_cross_strength': h4_cross_strength,
            'h4_alignment_score': h4_alignment_score,
            'h1_rsi': h1_rsi,
            'm15_adx': m15_adx,
            'delta_bias': delta_bias,
            'of_imbalance': of_imbalance,
            'of_strength': of_strength,
            'h1_supertrend_dir': h1_supertrend_dir,
            'h4_supertrend_dir': h4_supertrend_dir,
            'h4_ema_spread_9_21': h4_ema_spread_9_21,
            'is_choppy': is_choppy,
            'vol_surge': vol_surge,
            'atr_pips': atr_pips,
        }
    }


# ════════════════════════════════════════════════════════════════
# EMA CROSS FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_ema_cross_features_from_db(row: dict) -> dict:
    """
    Extract EMA cross-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_ema_cross_features).

    Falls back to computed defaults for trades without ema_cross feature rows.
    """
    if row.get('h4_cross_bars_ago') is not None and row.get('h4_cross_bars_ago') != 0:
        ef = {
            'h4_cross_bars_ago': int(row.get('h4_cross_bars_ago', 0) or 0),
            'h4_cross_strength': int(row.get('h4_cross_strength', 50) or 50),
            'h4_alignment_score': int(row.get('h4_alignment_score', 50) or 50),
            'h1_rsi': float(row.get('h1_rsi', 50) or 50),
            'm15_adx': float(row.get('m15_adx', 25) or 25),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'h1_supertrend_dir': int(row.get('h1_supertrend_dir', 0) or 0),
            'h4_supertrend_dir': int(row.get('h4_supertrend_dir', 0) or 0),
            'h4_ema_spread_9_21': float(row.get('h4_ema_spread_9_21', 0) or 0),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'vol_surge': int(row.get('vol_surge', 0) or 0),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
        }
    else:
        ef = {
            'h4_cross_bars_ago': 0,
            'h4_cross_strength': int(row.get('score', 50) or 50),
            'h4_alignment_score': int(row.get('htf_score', 50) or 50),
            'h1_rsi': 50.0,
            'm15_adx': 25.0,
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'h1_supertrend_dir': 0,
            'h4_supertrend_dir': 0,
            'h4_ema_spread_9_21': float(row.get('va_width_pips', 0) or 0),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'vol_surge': int(row.get('vol_surge_detected', 0) or 0),
            'atr_pips': float(row.get('atr', 0) or 0),
        }
    return ef


def build_ema_cross_feature_vector(row: dict) -> list:
    """Build EMA cross model feature vector from a DB row (with JOINed ema_cross features).

    Returns a list of numeric features for the EMA Cross Layer 1 model.
    Uses real ema_cross features when available, falls back to derived values.
    """
    ef = extract_ema_cross_features_from_db(row)

    features = [
        # ── EMA Cross-specific internal features (14) ──
        min(ef.get('h4_cross_bars_ago', 0), 10) / 10.0,
        ef.get('h4_cross_strength', 50) / 100.0,
        ef.get('h4_alignment_score', 50) / 100.0,
        ef.get('h1_rsi', 50) / 100.0,
        ef.get('m15_adx', 25) / 60.0,
        1.0 if str(ef.get('delta_bias', 'NEUTRAL')) == 'BULLISH' else (-1.0 if str(ef.get('delta_bias', 'NEUTRAL')) == 'BEARISH' else 0.0),
        ef.get('of_imbalance', 0) / 1.0,  # can be negative
        OF_STRENGTH_MAP.get(str(ef.get('of_strength', 'NONE')), 0.0),
        float(ef.get('h1_supertrend_dir', 0)) / 1.0,
        float(ef.get('h4_supertrend_dir', 0)) / 1.0,
        ef.get('h4_ema_spread_9_21', 0) / 50.0,
        float(ef.get('is_choppy', 0)),
        float(ef.get('vol_surge', 0)),
        ef.get('atr_pips', 0) / 30.0,

        # ── General features from backtest_trades (4) ──
        float(row.get('atr', 0) or 0) / 30.0,
        abs(float(row.get('pip_from_vwap', 0) or 0)) / 50.0,
        abs(float(row.get('pip_to_poc', 0) or 0)) / 50.0,
        float(row.get('va_width_pips', 20) or 20) / 50.0,

        # ── Cross-strategy confluence (5) from strategy score columns ──
        float(row.get('ss_smc_ob', 0) or 0) / 100.0,
        float(row.get('ss_breakout_momentum', 0) or 0) / 100.0,
        float(row.get('ss_vwap_reversion', 0) or 0) / 100.0,
        float(row.get('ss_fvg_reversion', 0) or 0) / 100.0,
        float(row.get('ss_delta_divergence', 0) or 0) / 100.0,
    ]
    return features
