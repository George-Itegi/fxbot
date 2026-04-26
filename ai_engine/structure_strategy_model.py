# =============================================================
# ai_engine/structure_strategy_model.py  v1.0 — Layer 1 Structure Model
#
# PURPOSE: Specialized Layer 1 model for STRUCTURE_ALIGNMENT strategy.
# Replaces the hard-coded gates in strategies/structure_alignment.py
# with a learned model that knows which structure alignment signals work.
#
# STRUCTURE-SPECIFIC FEATURES (15 internal + 4 general + 5 cross-strategy = 24):
#   - bos_direction (BULLISH / BEARISH — direction of latest BOS)
#   - bos_count (number of consecutive BOS in same direction)
#   - h1_ema_aligned (H1 EMA 9 > EMA 21 aligned with BOS)
#   - h1_full_ema_aligned (H1 all EMAs stacked: 9 > 21 > 50)
#   - h1_supertrend_dir (H1 supertrend direction)
#   - delta_value (delta/CVD value at signal time)
#   - delta_bias (BULLISH / BEARISH / NEUTRAL)
#   - of_imbalance (order flow imbalance value)
#   - of_strength (order flow strength level)
#   - has_opposing_fvg (FVG exists against BOS direction)
#   - pd_zone (PREMIUM / DISCOUNT / EQUILIBRIUM)
#   - h4_trend_aligned (H4 trend direction matches BOS)
#   - vol_surge (volume surge detected)
#   - is_choppy (market is choppy)
#   - atr_pips (M15 ATR in pips)
#
# WHY STRUCTURE:
#   - Structure alignment (BOS + trend + delta) is a high-probability
#     core SMC strategy
#   - Confluence gates (all must align) may be too restrictive
#   - Model can learn which confluence levels actually matter
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy STRUCTURE_ALIGNMENT
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds structure-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "STRUCTURE_ALIGNMENT"
STRATEGY_KEY = "structure"

# BOS direction encoding
BOS_DIR_MAP = {
    'BULLISH':  1.0,
    'BEARISH': -1.0,
    'NONE':     0.0,
}

# OF strength encoding
OF_STRENGTH_MAP = {
    'EXTREME':  1.0,
    'STRONG':   0.75,
    'MODERATE': 0.5,
    'WEAK':     0.25,
    'NONE':     0.0,
}

# PD zone encoding
PD_ZONE_MAP = {
    'PREMIUM':     1.0,
    'DISCOUNT':   -1.0,
    'EQUILIBRIUM': 0.0,
    'NEUTRAL':     0.0,
    'NONE':        0.0,
}


# ════════════════════════════════════════════════════════════════
# STRUCTURE ALIGNMENT NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    Structure alignment evaluation WITHOUT hard-coded gates.

    Generates ALL potential structure alignment signals regardless of
    gate conditions. The Layer 1 model will decide which to PASS.

    Simplified detection: find BOS from SMC report, check multi-
    timeframe alignment, score continuation probability.

    Returns: signal dict with _structure_features or None
    """
    if df_h1 is None or df_h4 is None or df_m15 is None or smc_report is None:
        return None

    h1 = df_h1.iloc[-1]
    h4 = df_h4.iloc[-1]
    m15 = df_m15.iloc[-1]
    close_price = float(m15['close'])

    from core.pip_utils import get_pip_size as _gps
    pip_size = _gps(symbol, close_price)

    atr_pips = float(m15.get('atr', 0)) / pip_size if pip_size > 0 else 0

    if atr_pips < 1.0:
        return None

    # ── Find latest BOS from SMC report ──
    structure = smc_report.get('structure', {})
    bos_list = structure.get('bos', [])
    if not bos_list:
        # Try older format
        bos_list = structure.get('break_of_structure', [])
    if not bos_list:
        return None

    latest_bos = bos_list[0] if isinstance(bos_list, list) else bos_list
    bos_direction = str(latest_bos.get('type', '')).upper()

    if 'BULL' in bos_direction:
        bos_direction = 'BULLISH'
        direction = "BUY"
    elif 'BEAR' in bos_direction:
        bos_direction = 'BEARISH'
        direction = "SELL"
    else:
        return None

    # Count consecutive BOS in same direction
    bos_count = 0
    for bos in bos_list if isinstance(bos_list, list) else [bos_list]:
        bt = str(bos.get('type', '')).upper()
        if (bos_direction == 'BULLISH' and 'BULL' in bt) or \
           (bos_direction == 'BEARISH' and 'BEAR' in bt):
            bos_count += 1
        else:
            break

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"BOS_{bos_direction}_{bos_count}x")

    if bos_count >= 2:
        score += 10
        confluence.append("MULTIPLE_BOS")
    elif bos_count == 1:
        score += 5
        confluence.append("SINGLE_BOS")

    # H1 EMA alignment (scoring)
    h1_ema9 = float(h1.get('ema_9', 0))
    h1_ema21 = float(h1.get('ema_21', 0))
    h1_ema50 = float(h1.get('ema_50', 0))
    h1_close = float(h1['close'])

    h1_ema_aligned = 0
    h1_full_ema_aligned = 0

    if direction == "BUY":
        if h1_ema9 > h1_ema21:
            h1_ema_aligned = 1
            score += 8
            confluence.append("H1_EMA_BULL")
        if h1_ema9 > h1_ema21 > h1_ema50:
            h1_full_ema_aligned = 1
            score += 10
            confluence.append("H1_FULL_EMA_BULL")
    else:
        if h1_ema9 < h1_ema21:
            h1_ema_aligned = 1
            score += 8
            confluence.append("H1_EMA_BEAR")
        if h1_ema9 < h1_ema21 < h1_ema50:
            h1_full_ema_aligned = 1
            score += 10
            confluence.append("H1_FULL_EMA_BEAR")

    # H1 Supertrend (scoring)
    h1_supertrend_dir = int(h1.get('supertrend_dir', 0))
    if (direction == "BUY" and h1_supertrend_dir == 1) or \
       (direction == "SELL" and h1_supertrend_dir == -1):
        score += 8
        confluence.append("H1_ST_ALIGN")
    elif h1_supertrend_dir != 0:
        score -= 5
        confluence.append("H1_ST_AGAINST")

    # H4 trend alignment (scoring)
    h4_ema9 = float(h4.get('ema_9', 0))
    h4_ema21 = float(h4.get('ema_21', 0))
    h4_st = int(h4.get('supertrend_dir', 0))

    h4_trend_aligned = 0
    if (direction == "BUY" and h4_ema9 > h4_ema21) or \
       (direction == "SELL" and h4_ema9 < h4_ema21):
        h4_trend_aligned = 1
        score += 10
        confluence.append("H4_TREND_ALIGN")

    if (direction == "BUY" and h4_st == 1) or \
       (direction == "SELL" and h4_st == -1):
        score += 5
        confluence.append("H4_ST_ALIGN")

    # Delta (scoring)
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_data = rolling_delta.get('delta', {})
    delta_value = float(delta_data.get('value', 0)) if delta_data else 0
    delta_bias = str(rolling_delta.get('bias', 'NEUTRAL'))

    if (direction == "BUY" and delta_bias == "BULLISH") or \
       (direction == "SELL" and delta_bias == "BEARISH"):
        score += 12
        confluence.append("DELTA_CONFIRMS")
    elif (direction == "BUY" and delta_bias == "BEARISH") or \
         (direction == "SELL" and delta_bias == "BULLISH"):
        score -= 10
        confluence.append("DELTA_AGAINST")

    # Order flow (scoring)
    of_imb = market_report.get('order_flow_imbalance', {}) if market_report else {}
    of_imbalance = float(of_imb.get('imbalance', 0))
    of_strength = str(of_imb.get('strength', 'NONE'))

    if (direction == "BUY" and of_imbalance > 0.05) or \
       (direction == "SELL" and of_imbalance < -0.05):
        score += 10
        confluence.append(f"OF_ALIGN_{of_imbalance:+.2f}")
    elif (direction == "BUY" and of_imbalance < -0.1) or \
         (direction == "SELL" and of_imbalance > 0.1):
        score -= 10
        confluence.append(f"OF_AGAINST_{of_imbalance:+.2f}")

    # Opposing FVG (scoring)
    has_opposing_fvg = 0
    fvgs = smc_report.get('fvgs', [])
    if not fvgs:
        fvgs = smc_report.get('fair_value_gaps', [])
    for fvg in fvgs:
        fvg_type = str(fvg.get('type', ''))
        if (direction == "BUY" and 'BEAR' in fvg_type.upper()) or \
           (direction == "SELL" and 'BULL' in fvg_type.upper()):
            has_opposing_fvg = 1
            score -= 10
            confluence.append("OPPOSING_FVG")
            break

    # PD zone (scoring)
    pd_zone = 'NEUTRAL'
    if smc_report:
        pd_info = smc_report.get('premium_discount', {})
        pd_zone = pd_info.get('zone', 'NEUTRAL')
        if direction == "BUY" and 'DISCOUNT' in pd_zone:
            score += 8
            confluence.append("DISCOUNT_ZONE")
        elif direction == "SELL" and 'PREMIUM' in pd_zone:
            score += 8
            confluence.append("PREMIUM_ZONE")
        elif direction == "BUY" and 'PREMIUM' in pd_zone:
            score -= 5
            confluence.append("PREMIUM_ZONE_AGAINST")
        elif direction == "SELL" and 'DISCOUNT' in pd_zone:
            score -= 5
            confluence.append("DISCOUNT_ZONE_AGAINST")

    # Volume surge (scoring)
    vol_surge = 0
    surge = market_report.get('volume_surge', {}) if market_report else {}
    if surge.get('surge_detected', False):
        vol_surge = 1
        score += 8
        confluence.append(f"VOL_SURGE_{surge.get('surge_ratio', 1.0):.1f}x")

    # Choppy (scoring)
    is_choppy = 0
    if master_report:
        momentum = master_report.get('momentum', {})
        is_choppy = 1 if momentum.get('is_choppy', False) else 0
        if is_choppy:
            score -= 15
            confluence.append("CHOPPY")

    # ── SL/TP calculation ──
    sl_pips = max(5.0, round(atr_pips * 1.5, 1))
    tp1_pips = round(sl_pips * 2.0, 1)
    tp2_pips = round(sl_pips * 3.0, 1)

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
        "_structure_features": {
            'bos_direction': bos_direction,
            'bos_count': bos_count,
            'h1_ema_aligned': h1_ema_aligned,
            'h1_full_ema_aligned': h1_full_ema_aligned,
            'h1_supertrend_dir': h1_supertrend_dir,
            'delta_value': delta_value,
            'delta_bias': delta_bias,
            'of_imbalance': of_imbalance,
            'of_strength': of_strength,
            'has_opposing_fvg': has_opposing_fvg,
            'pd_zone': pd_zone,
            'h4_trend_aligned': h4_trend_aligned,
            'vol_surge': vol_surge,
            'is_choppy': is_choppy,
            'atr_pips': atr_pips,
        }
    }


# ════════════════════════════════════════════════════════════════
# STRUCTURE FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_structure_features_from_db(row: dict) -> dict:
    """
    Extract structure-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_structure_features).

    Falls back to computed defaults for trades without structure feature rows.
    """
    if row.get('bos_direction') is not None and row.get('bos_direction') != '':
        sf = {
            'bos_direction': str(row.get('bos_direction', 'NONE')),
            'bos_count': int(row.get('bos_count', 1) or 1),
            'h1_ema_aligned': int(row.get('h1_ema_aligned', 0) or 0),
            'h1_full_ema_aligned': int(row.get('h1_full_ema_aligned', 0) or 0),
            'h1_supertrend_dir': int(row.get('h1_supertrend_dir', 0) or 0),
            'delta_value': float(row.get('delta_value', 0) or 0),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'has_opposing_fvg': int(row.get('has_opposing_fvg', 0) or 0),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'h4_trend_aligned': int(row.get('h4_trend_aligned', 0) or 0),
            'vol_surge': int(row.get('vol_surge', 0) or 0),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
        }
    else:
        sf = {
            'bos_direction': 'BULLISH' if row.get('direction') == 'BUY' else 'BEARISH',
            'bos_count': 1,
            'h1_ema_aligned': 0,
            'h1_full_ema_aligned': 0,
            'h1_supertrend_dir': 0,
            'delta_value': float(row.get('delta', 0) or 0),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'has_opposing_fvg': 0,
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'h4_trend_aligned': int(1 if row.get('htf_approved') else 0),
            'vol_surge': int(row.get('vol_surge_detected', 0) or 0),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'atr_pips': float(row.get('atr', 0) or 0),
        }
    return sf


def build_structure_feature_vector(row: dict) -> list:
    """Build structure alignment model feature vector from a DB row (with JOINed structure features).

    Returns a list of numeric features for the Structure Layer 1 model.
    Uses real structure features when available, falls back to derived values.
    """
    sf = extract_structure_features_from_db(row)

    features = [
        # ── Structure-specific internal features (15) ──
        BOS_DIR_MAP.get(str(sf.get('bos_direction', 'NONE')), 0.0),
        min(sf.get('bos_count', 1), 5) / 5.0,
        float(sf.get('h1_ema_aligned', 0)),
        float(sf.get('h1_full_ema_aligned', 0)),
        float(sf.get('h1_supertrend_dir', 0)) / 1.0,
        min(abs(sf.get('delta_value', 0)), 500) / 500.0,
        1.0 if str(sf.get('delta_bias', 'NEUTRAL')) == 'BULLISH' else (-1.0 if str(sf.get('delta_bias', 'NEUTRAL')) == 'BEARISH' else 0.0),
        sf.get('of_imbalance', 0) / 1.0,  # can be negative
        OF_STRENGTH_MAP.get(str(sf.get('of_strength', 'NONE')), 0.0),
        float(sf.get('has_opposing_fvg', 0)),
        PD_ZONE_MAP.get(str(sf.get('pd_zone', 'NEUTRAL')), 0.0),
        float(sf.get('h4_trend_aligned', 0)),
        float(sf.get('vol_surge', 0)),
        float(sf.get('is_choppy', 0)),
        sf.get('atr_pips', 0) / 30.0,

        # ── General features from backtest_trades (4) ──
        float(row.get('atr', 0) or 0) / 30.0,
        abs(float(row.get('pip_from_vwap', 0) or 0)) / 50.0,
        abs(float(row.get('pip_to_poc', 0) or 0)) / 50.0,
        float(row.get('va_width_pips', 20) or 20) / 50.0,

        # ── Cross-strategy confluence (5) from strategy score columns ──
        float(row.get('ss_smc_ob', 0) or 0) / 100.0,
        float(row.get('ss_breakout_momentum', 0) or 0) / 100.0,
        float(row.get('ss_vwap_reversion', 0) or 0) / 100.0,
        float(row.get('ss_ema_cross', 0) or 0) / 100.0,
        float(row.get('ss_trend_continuation', 0) or 0) / 100.0,
    ]
    return features
