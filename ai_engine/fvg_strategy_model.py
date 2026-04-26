# =============================================================
# ai_engine/fvg_strategy_model.py  v1.0 — Layer 1 FVG Model
#
# PURPOSE: Specialized Layer 1 model for FVG_REVERSION strategy.
# Replaces the hard-coded gates in strategies/fvg_reversion.py
# with a learned model that knows which FVG reversion signals work.
#
# FVG-SPECIFIC FEATURES (15 internal + 4 general + 5 cross-strategy = 24):
#   - fvg_type (BULLISH_FVG / BEARISH_FVG)
#   - fvg_quality_score (0-100 composite quality rating)
#   - fvg_gap_pips (size of the FVG gap in pips)
#   - fvg_distance_pips (distance from price to FVG)
#   - of_imbalance (order flow imbalance value)
#   - of_strength (order flow strength level)
#   - vol_surge (volume surge detected)
#   - vol_surge_ratio (volume surge magnitude)
#   - stoch_rsi_k (M15 StochRSI K value)
#   - stoch_rsi_turning (StochRSI is turning toward FVG direction)
#   - m5_wick_rejection (M5 wick rejected at FVG boundary)
#   - ob_fvg_confluence (OB and FVG overlap / confluence)
#   - ob_fvg_distance (distance between overlapping OB and FVG)
#   - pd_zone (PREMIUM / DISCOUNT / EQUILIBRIUM)
#   - atr_pips (M15 ATR in pips)
#
# WHY FVG:
#   - Fair Value Gaps are a core ICT/SMC concept
#   - Many FVGs fill, but only some lead to reversals
#   - Model can learn which FVG quality/size/distance combinations work
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy FVG_REVERSION
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds fvg-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "FVG_REVERSION"
STRATEGY_KEY = "fvg"

# FVG type encoding
FVG_TYPE_MAP = {
    'BULLISH_FVG':  1.0,
    'BEARISH_FVG': -1.0,
    'NONE':         0.0,
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
# FVG NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    FVG reversion evaluation WITHOUT hard-coded gates.

    Generates ALL potential FVG reversion signals regardless of
    gate conditions. The Layer 1 model will decide which to PASS.

    Simplified detection: find recent 3-candle FVG patterns in
    M15 data, check if price is near the FVG, score reversion.

    Returns: signal dict with _fvg_features or None
    """
    if df_m15 is None or df_h1 is None:
        return None
    if len(df_m15) < 30:
        return None

    m15 = df_m15.iloc[-1]
    h1 = df_h1.iloc[-1]
    close_price = float(m15['close'])

    from core.pip_utils import get_pip_size as _gps
    pip_size = _gps(symbol, close_price)

    atr_pips = float(m15.get('atr', 0)) / pip_size if pip_size > 0 else 0
    stoch_rsi_k = float(m15.get('stoch_rsi_k', 50))

    if atr_pips < 1.0:
        return None

    # ── Detect Fair Value Gaps from M15 data ──
    # FVG: gap between candle[i-2] high and candle[i] low (bullish)
    #      or gap between candle[i-2] low and candle[i] high (bearish)
    df_recent = df_m15.tail(30)
    m15_highs = df_recent['high'].astype(float).values
    m15_lows = df_recent['low'].astype(float).values
    m15_closes = df_recent['close'].astype(float).values

    nearest_fvg = None
    min_dist = float('inf')

    for i in range(2, len(m15_highs) - 1):
        # Bullish FVG: candle[i-2] high < candle[i] low (gap up)
        bull_fvg_top = m15_lows[i]
        bull_fvg_bottom = m15_highs[i - 2]
        bull_gap = bull_fvg_top - bull_fvg_bottom

        if bull_gap > 0:
            fvg_mid = (bull_fvg_top + bull_fvg_bottom) / 2
            dist = abs(close_price - fvg_mid) / pip_size
            if dist < min_dist:
                min_dist = dist
                nearest_fvg = {
                    'type': 'BULLISH_FVG',
                    'top': bull_fvg_top,
                    'bottom': bull_fvg_bottom,
                    'mid': fvg_mid,
                    'gap_pips': round(bull_gap / pip_size, 2),
                    'dist_pips': round(dist, 2),
                    'bar_ago': len(m15_highs) - 1 - i,
                }

        # Bearish FVG: candle[i-2] low > candle[i] high (gap down)
        bear_fvg_top = m15_highs[i - 2]
        bear_fvg_bottom = m15_lows[i]
        bear_gap = bear_fvg_top - bear_fvg_bottom

        if bear_gap > 0:
            fvg_mid = (bear_fvg_top + bear_fvg_bottom) / 2
            dist = abs(close_price - fvg_mid) / pip_size
            if dist < min_dist:
                min_dist = dist
                nearest_fvg = {
                    'type': 'BEARISH_FVG',
                    'top': bear_fvg_top,
                    'bottom': bear_fvg_bottom,
                    'mid': fvg_mid,
                    'gap_pips': round(bear_gap / pip_size, 2),
                    'dist_pips': round(dist, 2),
                    'bar_ago': len(m15_highs) - 1 - i,
                }

    if nearest_fvg is None:
        return None

    fvg_type = nearest_fvg['type']
    fvg_gap_pips = nearest_fvg['gap_pips']
    fvg_distance_pips = nearest_fvg['dist_pips']

    # Only consider FVGs within reasonable distance
    max_dist = atr_pips * 4.0
    if fvg_distance_pips > max_dist:
        return None

    # Determine direction: trade back into the FVG
    if fvg_type == 'BULLISH_FVG':
        # Price is above a bullish FVG — sell back into it
        if close_price > nearest_fvg['top']:
            direction = "SELL"
        elif close_price < nearest_fvg['bottom']:
            # Price below bullish FVG — buy to fill it
            direction = "BUY"
        else:
            return None
    else:
        # BEARISH_FVG
        if close_price < nearest_fvg['bottom']:
            direction = "BUY"
        elif close_price > nearest_fvg['top']:
            direction = "SELL"
        else:
            return None

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"FVG_{fvg_type}")

    # Quality score based on gap size, distance, and recency
    fvg_quality_score = 50
    if fvg_gap_pips > atr_pips * 0.5 and fvg_gap_pips < atr_pips * 2:
        fvg_quality_score += 20
        score += 8
        confluence.append(f"GOOD_GAP_{fvg_gap_pips:.1f}p")
    elif fvg_gap_pips >= atr_pips * 2:
        fvg_quality_score -= 10
        score -= 5
        confluence.append(f"OVERSIZED_GAP_{fvg_gap_pips:.1f}p")

    if fvg_distance_pips < atr_pips * 2:
        fvg_quality_score += 15
        score += 8
        confluence.append(f"NEAR_FVG_{fvg_distance_pips:.1f}p")

    if nearest_fvg.get('bar_ago', 10) <= 5:
        fvg_quality_score += 10
        score += 5
        confluence.append(f"RECENT_FVG_{nearest_fvg['bar_ago']}bars")
    elif nearest_fvg.get('bar_ago', 10) > 15:
        fvg_quality_score -= 15
        score -= 8
        confluence.append(f"OLD_FVG_{nearest_fvg['bar_ago']}bars")

    fvg_quality_score = max(0, min(100, fvg_quality_score))

    # Order flow (scoring)
    of_imb = market_report.get('order_flow_imbalance', {}) if market_report else {}
    of_imbalance = float(of_imb.get('imbalance', 0))
    of_strength = str(of_imb.get('strength', 'NONE'))

    if (direction == "BUY" and of_imbalance > 0.05) or \
       (direction == "SELL" and of_imbalance < -0.05):
        score += 8
        confluence.append(f"OF_ALIGN_{of_imbalance:+.2f}")
    else:
        score -= 5
        confluence.append(f"OF_CONFLICT_{of_imbalance:+.2f}")

    # Volume surge (scoring)
    vol_surge = 0
    vol_surge_ratio = 1.0
    surge = market_report.get('volume_surge', {}) if market_report else {}
    if surge.get('surge_detected', False):
        vol_surge = 1
        vol_surge_ratio = float(surge.get('surge_ratio', 1.0))
        score += 8
        confluence.append(f"VOL_SURGE_{vol_surge_ratio:.1f}x")
    else:
        score -= 3
        confluence.append("NO_VOL_SURGE")

    # StochRSI (scoring)
    stoch_rsi_turning = 0
    if direction == "BUY" and stoch_rsi_k < 40:
        score += 10
        confluence.append("STOCHRSI_LOW")
        if stoch_rsi_k < 30:
            stoch_rsi_turning = 1
    elif direction == "SELL" and stoch_rsi_k > 60:
        score += 10
        confluence.append("STOCHRSI_HIGH")
        if stoch_rsi_k > 70:
            stoch_rsi_turning = 1

    # M5 wick rejection at FVG boundary (scoring)
    m5_wick_rejection = 0
    if df_m5 is not None and len(df_m5) >= 2:
        m5_last = df_m5.iloc[-1]
        m5_high = float(m5_last['high'])
        m5_low = float(m5_last['low'])
        m5_open = float(m5_last['open'])
        m5_close = float(m5_last['close'])
        m5_upper_wick = m5_high - max(m5_close, m5_open)
        m5_lower_wick = min(m5_close, m5_open) - m5_low

        if direction == "BUY" and fvg_type == 'BULLISH_FVG':
            if m5_low <= nearest_fvg['bottom'] and m5_close > nearest_fvg['bottom']:
                m5_wick_rejection = 1
                score += 10
                confluence.append("M5_WICK_REJECTION")
        elif direction == "SELL" and fvg_type == 'BEARISH_FVG':
            if m5_high >= nearest_fvg['top'] and m5_close < nearest_fvg['top']:
                m5_wick_rejection = 1
                score += 10
                confluence.append("M5_WICK_REJECTION")

    # OB-FVG confluence (scoring)
    ob_fvg_confluence = 0
    ob_fvg_distance = 0.0
    if smc_report:
        obs = smc_report.get('order_blocks', [])
        for ob in obs:
            if ob.get('mitigated', False):
                continue
            ob_high = float(ob.get('high', 0))
            ob_low = float(ob.get('low', 0))
            fvg_top = nearest_fvg['top']
            fvg_bottom = nearest_fvg['bottom']
            # Check overlap
            if ob_high > fvg_bottom and ob_low < fvg_top:
                ob_fvg_confluence = 1
                ob_mid = (ob_high + ob_low) / 2
                fvg_mid = (fvg_top + fvg_bottom) / 2
                ob_fvg_distance = round(abs(ob_mid - fvg_mid) / pip_size, 2)
                score += 12
                confluence.append(f"OB_FVG_CONFLUENCE_{ob_fvg_distance:.1f}p")
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

    # ── SL/TP calculation ──
    sl_pips = max(5.0, round(atr_pips * 1.5, 1))
    if direction == "BUY" and fvg_type == 'BULLISH_FVG':
        tp1_pips = round((nearest_fvg['top'] - close_price) / pip_size, 1)
    elif direction == "SELL" and fvg_type == 'BEARISH_FVG':
        tp1_pips = round((close_price - nearest_fvg['bottom']) / pip_size, 1)
    else:
        tp1_pips = round(fvg_gap_pips, 1)

    tp1_pips = max(tp1_pips, round(sl_pips * 2, 1))
    tp2_pips = round(tp1_pips * 1.5, 1)

    if tp1_pips < sl_pips * 1.5:
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

    log.info(f"[{STRATEGY_NAME}:NOGATE] {direction} {symbol} Score:{score} | {', '.join(confluence)}")

    return {
        "direction": direction, "entry_price": close_price,
        "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
        "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
        "strategy": STRATEGY_NAME, "version": "2.0-no-gate",
        "score": score, "confluence": confluence,
        "_fvg_features": {
            'fvg_type': fvg_type,
            'fvg_quality_score': fvg_quality_score,
            'fvg_gap_pips': fvg_gap_pips,
            'fvg_distance_pips': fvg_distance_pips,
            'of_imbalance': of_imbalance,
            'of_strength': of_strength,
            'vol_surge': vol_surge,
            'vol_surge_ratio': vol_surge_ratio,
            'stoch_rsi_k': stoch_rsi_k,
            'stoch_rsi_turning': stoch_rsi_turning,
            'm5_wick_rejection': m5_wick_rejection,
            'ob_fvg_confluence': ob_fvg_confluence,
            'ob_fvg_distance': ob_fvg_distance,
            'pd_zone': pd_zone,
            'atr_pips': atr_pips,
        }
    }


# ════════════════════════════════════════════════════════════════
# FVG FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_fvg_features_from_db(row: dict) -> dict:
    """
    Extract FVG-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_fvg_features).

    Falls back to computed defaults for trades without FVG feature rows.
    """
    if row.get('fvg_type') is not None and row.get('fvg_type') != '':
        ff = {
            'fvg_type': str(row.get('fvg_type', 'NONE')),
            'fvg_quality_score': int(row.get('fvg_quality_score', 50) or 50),
            'fvg_gap_pips': float(row.get('fvg_gap_pips', 0) or 0),
            'fvg_distance_pips': float(row.get('fvg_distance_pips', 0) or 0),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'vol_surge': int(row.get('vol_surge', 0) or 0),
            'vol_surge_ratio': float(row.get('vol_surge_ratio', 1.0) or 1.0),
            'stoch_rsi_k': float(row.get('stoch_rsi_k', 50) or 50),
            'stoch_rsi_turning': int(row.get('stoch_rsi_turning', 0) or 0),
            'm5_wick_rejection': int(row.get('m5_wick_rejection', 0) or 0),
            'ob_fvg_confluence': int(row.get('ob_fvg_confluence', 0) or 0),
            'ob_fvg_distance': float(row.get('ob_fvg_distance', 0) or 0),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
        }
    else:
        ff = {
            'fvg_type': 'BULLISH_FVG' if row.get('direction') == 'SELL' else 'BEARISH_FVG',
            'fvg_quality_score': 50,
            'fvg_gap_pips': float(row.get('va_width_pips', 10) or 10) * 0.3,
            'fvg_distance_pips': abs(float(row.get('pip_from_vwap', 0) or 0)),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'vol_surge': int(row.get('vol_surge_detected', 0) or 0),
            'vol_surge_ratio': float(row.get('vol_surge_ratio', 1.0) or 1.0),
            'stoch_rsi_k': 50.0,
            'stoch_rsi_turning': 0,
            'm5_wick_rejection': 0,
            'ob_fvg_confluence': 0,
            'ob_fvg_distance': 0.0,
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'atr_pips': float(row.get('atr', 0) or 0),
        }
    return ff


def build_fvg_feature_vector(row: dict) -> list:
    """Build FVG model feature vector from a DB row (with JOINed FVG features).

    Returns a list of numeric features for the FVG Layer 1 model.
    Uses real FVG features when available, falls back to derived values.
    """
    ff = extract_fvg_features_from_db(row)

    features = [
        # ── FVG-specific internal features (15) ──
        FVG_TYPE_MAP.get(str(ff.get('fvg_type', 'NONE')), 0.0),
        ff.get('fvg_quality_score', 50) / 100.0,
        ff.get('fvg_gap_pips', 0) / 20.0,
        ff.get('fvg_distance_pips', 0) / 50.0,
        ff.get('of_imbalance', 0) / 1.0,  # can be negative
        OF_STRENGTH_MAP.get(str(ff.get('of_strength', 'NONE')), 0.0),
        float(ff.get('vol_surge', 0)),
        min(ff.get('vol_surge_ratio', 1.0), 5.0) / 5.0,
        ff.get('stoch_rsi_k', 50) / 100.0,
        float(ff.get('stoch_rsi_turning', 0)),
        float(ff.get('m5_wick_rejection', 0)),
        float(ff.get('ob_fvg_confluence', 0)),
        ff.get('ob_fvg_distance', 0) / 30.0,
        PD_ZONE_MAP.get(str(ff.get('pd_zone', 'NEUTRAL')), 0.0),
        ff.get('atr_pips', 0) / 30.0,

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
        float(row.get('ss_liquidity_sweep', 0) or 0) / 100.0,
    ]
    return features
