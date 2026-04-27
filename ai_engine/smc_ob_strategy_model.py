# =============================================================
# ai_engine/smc_ob_strategy_model.py  v1.0 — Layer 1 SMC OB Model
#
# PURPOSE: Specialized Layer 1 model for SMC_OB_REVERSAL strategy.
# Replaces the hard-coded gates in strategies/smc_ob_reversal.py
# with a learned model that knows which OB reversal signals work.
#
# SMC_OB-SPECIFIC FEATURES (15 internal + 4 general + 5 cross-strategy = 24):
#   - ob_type (BULLISH_OB / BEARISH_OB)
#   - ob_dist_pips (distance from price to OB zone)
#   - price_at_ob (price is inside the OB zone)
#   - trend (BULLISH / BEARISH / RANGING)
#   - delta_bias (BULLISH / BEARISH / NEUTRAL)
#   - delta_strength (STRONG / MODERATE / WEAK)
#   - of_imbalance (order flow imbalance value)
#   - of_strength (order flow strength level)
#   - stoch_rsi_k (M15 StochRSI K value)
#   - supertrend_dir_h1 (H1 supertrend direction)
#   - htf_ok (HTF alignment approved)
#   - smc_bias (BULLISH / BEARISH / NEUTRAL)
#   - pd_zone (PREMIUM / DISCOUNT / EQUILIBRIUM)
#   - atr_pips (M15 ATR in pips)
#   - has_bos (BOS confirms OB direction)
#
# WHY SMC_OB:
#   - Core SMC strategy — order block reversals are high-probability
#   - Many gates (HTF, trend, delta) may be too restrictive
#   - Model can learn which OB types and conditions actually profit
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy SMC_OB_REVERSAL
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds SMC_OB-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "SMC_OB_REVERSAL"
STRATEGY_KEY = "smc_ob"

# Order block type encoding
OB_TYPE_MAP = {
    'BULLISH_OB':  1.0,
    'BEARISH_OB': -1.0,
    'NONE':        0.0,
}

# Trend encoding
TREND_MAP = {
    'BULLISH':  1.0,
    'BEARISH': -1.0,
    'RANGING':  0.0,
    'NONE':     0.0,
}

# Delta strength encoding
DELTA_STRENGTH_MAP = {
    'STRONG':   1.0,
    'MODERATE': 0.5,
    'WEAK':     0.25,
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

# SMC bias encoding
SMC_BIAS_MAP = {
    'BULLISH':  1.0,
    'BEARISH': -1.0,
    'NEUTRAL':  0.0,
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
# SMC OB NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    SMC Order Block evaluation WITHOUT hard-coded gates.

    Generates ALL potential OB reversal signals (buy/sell setups)
    regardless of gate conditions. The Layer 1 model will then
    decide which ones to PASS.

    Removed gates vs original smc_ob_reversal.py:
      - HTF alignment filter -> REMOVED (becomes scoring)
      - Trend direction filter -> REMOVED (becomes scoring)
      - Delta confirmation gate -> REMOVED (becomes scoring)
      - Score threshold -> REMOVED
      - Confluence minimum -> REMOVED
      - R:R gate (tp1 >= sl * 1.5) -> KEPT (prevents nonsensical trades)

    Simplified detection: find nearest unmitigated OB from smc_report,
    check price proximity, score everything.

    Returns: signal dict with _smc_ob_features or None
    """
    if df_m15 is None or df_h1 is None or smc_report is None:
        return None

    m15 = df_m15.iloc[-1]
    h1 = df_h1.iloc[-1]
    close_price = float(m15['close'])

    from core.pip_utils import get_pip_size as _gps
    pip_size = _gps(symbol, close_price)

    atr_pips = float(m15.get('atr', 0)) / pip_size if pip_size > 0 else 0
    stoch_rsi_k = float(m15.get('stoch_rsi_k', 50))
    supertrend_dir_h1 = int(h1.get('supertrend_dir', 0))

    # ── Find nearest unmitigated OB from SMC report ──
    order_blocks = smc_report.get('order_blocks', [])
    if not order_blocks:
        return None

    nearest_ob = None
    min_dist = float('inf')

    for ob in order_blocks:
        if ob.get('mitigated', False):
            continue
        ob_type = str(ob.get('type', ''))
        ob_high = float(ob.get('high', 0))
        ob_low = float(ob.get('low', 0))
        ob_mid = (ob_high + ob_low) / 2.0

        dist_pips = abs(close_price - ob_mid) / pip_size if pip_size > 0 else float('inf')

        # Check if price is near the OB (within 2x ATR)
        max_dist = atr_pips * 3.0
        if dist_pips < min_dist and dist_pips <= max_dist:
            min_dist = dist_pips
            nearest_ob = ob

    if nearest_ob is None:
        return None

    ob_type = str(nearest_ob.get('type', 'BULLISH_OB'))
    ob_high = float(nearest_ob.get('high', 0))
    ob_low = float(nearest_ob.get('low', 0))
    ob_mid = (ob_high + ob_low) / 2.0
    ob_dist_pips = abs(close_price - ob_mid) / pip_size
    price_at_ob = 1 if ob_low <= close_price <= ob_high else 0

    # Determine direction based on OB type
    if 'BULL' in ob_type.upper():
        direction = "BUY"
    elif 'BEAR' in ob_type.upper():
        direction = "SELL"
    else:
        return None

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"OB_{ob_type}")

    if price_at_ob:
        score += 12
        confluence.append("PRICE_AT_OB")
    elif ob_dist_pips < atr_pips:
        score += 8
        confluence.append(f"NEAR_OB_{ob_dist_pips:.1f}p")
    else:
        score -= 3
        confluence.append(f"OB_DIST_{ob_dist_pips:.1f}p")

    # Trend (scoring)
    trend = 'RANGING'
    if df_h4 is not None and len(df_h4) >= 20:
        h4 = df_h4.iloc[-1]
        h4_ema9 = float(h4.get('ema_9', 0))
        h4_ema21 = float(h4.get('ema_21', 0))
        if h4_ema9 > h4_ema21 * 1.001:
            trend = 'BULLISH'
            if direction == "BUY":
                score += 10
                confluence.append("TREND_BULL_ALIGN")
            else:
                score -= 8
                confluence.append("TREND_BULL_COUNTER")
        elif h4_ema9 < h4_ema21 * 0.999:
            trend = 'BEARISH'
            if direction == "SELL":
                score += 10
                confluence.append("TREND_BEAR_ALIGN")
            else:
                score -= 8
                confluence.append("TREND_BEAR_COUNTER")

    # Delta / order flow (scoring)
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')
    delta_data = rolling_delta.get('delta', {})
    delta_abs = abs(float(delta_data.get('value', 0))) if delta_data else 0
    delta_strength = 'STRONG' if delta_abs > 100 else ('MODERATE' if delta_abs > 40 else 'WEAK')

    of_imb = market_report.get('order_flow_imbalance', {}) if market_report else {}
    of_imbalance = float(of_imb.get('imbalance', 0))
    of_strength = str(of_imb.get('strength', 'NONE'))

    if direction == "BUY":
        if delta_bias == "BULLISH":
            score += 12
            confluence.append("DELTA_BULL")
        elif delta_bias == "BEARISH":
            score -= 10
            confluence.append("DELTA_BEAR_CONFLICT")
    else:
        if delta_bias == "BEARISH":
            score += 12
            confluence.append("DELTA_BEAR")
        elif delta_bias == "BULLISH":
            score -= 10
            confluence.append("DELTA_BULL_CONFLICT")

    if of_imbalance != 0:
        if (direction == "BUY" and of_imbalance > 0.05) or \
           (direction == "SELL" and of_imbalance < -0.05):
            score += 8
            confluence.append(f"OF_ALIGN_{of_imbalance:+.2f}")
        else:
            score -= 5
            confluence.append(f"OF_CONFLICT_{of_imbalance:+.2f}")

    # StochRSI (scoring)
    if direction == "BUY" and stoch_rsi_k < 35:
        score += 10
        confluence.append("STOCHRSI_OVERSOLD")
    elif direction == "SELL" and stoch_rsi_k > 65:
        score += 10
        confluence.append("STOCHRSI_OVERBOUGHT")
    elif direction == "BUY" and stoch_rsi_k > 70:
        score -= 8
        confluence.append("STOCHRSI_NOT_OVERSOLD")
    elif direction == "SELL" and stoch_rsi_k < 30:
        score -= 8
        confluence.append("STOCHRSI_NOT_OVERBOUGHT")

    # Supertrend H1 (scoring)
    if direction == "BUY" and supertrend_dir_h1 == 1:
        score += 8
        confluence.append("ST_BULL")
    elif direction == "SELL" and supertrend_dir_h1 == -1:
        score += 8
        confluence.append("ST_BEAR")

    # HTF alignment (scoring)
    htf_ok = 1
    if smc_report:
        htf_align = smc_report.get('htf_alignment', {})
        htf_ok = 1 if htf_align.get('approved', True) else 0
        if not htf_ok:
            score -= 10
            confluence.append("HTF_REJECTED")

    # SMC bias (scoring)
    smc_bias = smc_report.get('bias', 'NEUTRAL') if smc_report else 'NEUTRAL'
    if (direction == "BUY" and smc_bias == "BULLISH") or \
       (direction == "SELL" and smc_bias == "BEARISH"):
        score += 10
        confluence.append(f"SMC_BIAS_{smc_bias}")

    # PD zone (scoring)
    pd_zone = smc_report.get('premium_discount', {}).get('zone', 'NEUTRAL') if smc_report else 'NEUTRAL'
    if direction == "BUY" and 'DISCOUNT' in pd_zone:
        score += 8
        confluence.append("DISCOUNT_ZONE")
    elif direction == "SELL" and 'PREMIUM' in pd_zone:
        score += 8
        confluence.append("PREMIUM_ZONE")

    # BOS confirmation (scoring)
    has_bos = 0
    structure = smc_report.get('structure', {}) if smc_report else {}
    bos_list = structure.get('bos', [])
    if bos_list:
        for bos in bos_list:
            bt = str(bos.get('type', ''))
            if (direction == "BUY" and 'BULL' in bt.upper()) or \
               (direction == "SELL" and 'BEAR' in bt.upper()):
                has_bos = 1
                score += 8
                confluence.append("BOS_CONFIRMED")
                break

    # ── SL/TP calculation ──
    sl_pips = max(5.0, round(atr_pips * 1.5, 1))
    if direction == "BUY":
        sl_price = round(close_price - sl_pips * pip_size, 5)
        tp1_pips = round(ob_dist_pips * 1.5 if ob_dist_pips > 0 else sl_pips * 2, 1)
        tp1_price = round(close_price + tp1_pips * pip_size, 5)
        tp2_pips = round(tp1_pips * 1.5, 1)
        tp2_price = round(close_price + tp2_pips * pip_size, 5)
    else:
        sl_price = round(close_price + sl_pips * pip_size, 5)
        tp1_pips = round(ob_dist_pips * 1.5 if ob_dist_pips > 0 else sl_pips * 2, 1)
        tp1_price = round(close_price - tp1_pips * pip_size, 5)
        tp2_pips = round(tp1_pips * 1.5, 1)
        tp2_price = round(close_price - tp2_pips * pip_size, 5)

    # Keep minimum R:R check
    if tp1_pips < sl_pips * 1.5:
        tp1_pips = round(sl_pips * 2.0, 1)
        tp2_pips = round(sl_pips * 3.0, 1)
        if direction == "BUY":
            tp1_price = round(close_price + tp1_pips * pip_size, 5)
            tp2_price = round(close_price + tp2_pips * pip_size, 5)
        else:
            tp1_price = round(close_price - tp1_pips * pip_size, 5)
            tp2_price = round(close_price - tp2_pips * pip_size, 5)

    log.info(f"[{STRATEGY_NAME}:NOGATE] {direction} {symbol} Score:{score} | {', '.join(confluence)}")

    return {
        "direction": direction, "entry_price": close_price,
        "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
        "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
        "strategy": STRATEGY_NAME, "version": "2.0-no-gate",
        "score": score, "confluence": confluence,
        "_smc_ob_features": {
            'ob_type': ob_type,
            'ob_dist_pips': round(ob_dist_pips, 2),
            'price_at_ob': price_at_ob,
            'trend': trend,
            'delta_bias': delta_bias,
            'delta_strength': delta_strength,
            'of_imbalance': of_imbalance,
            'of_strength': of_strength,
            'stoch_rsi_k': stoch_rsi_k,
            'supertrend_dir_h1': supertrend_dir_h1,
            'htf_ok': htf_ok,
            'smc_bias': smc_bias,
            'pd_zone': pd_zone,
            'atr_pips': atr_pips,
            'has_bos': has_bos,
        }
    }


# ════════════════════════════════════════════════════════════════
# SMC OB FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_smc_ob_features_from_db(row: dict) -> dict:
    """
    Extract SMC OB-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_smc_ob_features).

    Falls back to computed defaults for trades without SMC OB feature rows.
    """
    if row.get('ob_type') is not None and row.get('ob_type') != '':
        sf = {
            'ob_type': str(row.get('ob_type', 'NONE')),
            'ob_dist_pips': float(row.get('ob_dist_pips', 0) or 0),
            'price_at_ob': int(row.get('price_at_ob', 0) or 0),
            'trend': str(row.get('trend', 'RANGING')),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'delta_strength': str(row.get('delta_strength', 'NONE')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'stoch_rsi_k': float(row.get('stoch_rsi_k', 50) or 50),
            'supertrend_dir_h1': int(row.get('supertrend_dir_h1', 0) or 0),
            'htf_ok': int(row.get('htf_ok', 1) or 1),
            'smc_bias': str(row.get('smc_bias', 'NEUTRAL')),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
            'has_bos': int(row.get('has_bos', 0) or 0),
        }
    else:
        sf = {
            'ob_type': str(row.get('ob_type', 'NONE')),
            'ob_dist_pips': abs(float(row.get('pip_from_vwap', 0) or 0)),
            'price_at_ob': 0,
            'trend': str(row.get('structure_trend', 'RANGING')),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'delta_strength': str(row.get('rd_bias', 'NONE')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'stoch_rsi_k': 50.0,
            'supertrend_dir_h1': 0,
            'htf_ok': int(1 if row.get('htf_approved') else 0),
            'smc_bias': str(row.get('smc_bias', 'NEUTRAL')),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'atr_pips': float(row.get('atr', 0) or 0),
            'has_bos': 0,
        }
    return sf


def build_smc_ob_feature_vector(row: dict) -> list:
    """Build SMC OB model feature vector from a DB row (with JOINed SMC OB features).

    Returns a list of numeric features for the SMC OB Layer 1 model.
    Uses real SMC OB features when available, falls back to derived values.
    """
    sf = extract_smc_ob_features_from_db(row)

    features = [
        # ── SMC OB-specific internal features (15) ──
        OB_TYPE_MAP.get(str(sf.get('ob_type', 'NONE')), 0.0),
        sf.get('ob_dist_pips', 0) / 50.0,
        float(sf.get('price_at_ob', 0)),
        TREND_MAP.get(str(sf.get('trend', 'RANGING')), 0.0),
        SMC_BIAS_MAP.get(str(sf.get('delta_bias', 'NEUTRAL')), 0.0),
        DELTA_STRENGTH_MAP.get(str(sf.get('delta_strength', 'NONE')), 0.0),
        sf.get('of_imbalance', 0) / 1.0,  # can be negative
        OF_STRENGTH_MAP.get(str(sf.get('of_strength', 'NONE')), 0.0),
        sf.get('stoch_rsi_k', 50) / 100.0,
        float(sf.get('supertrend_dir_h1', 0)) / 1.0,
        float(sf.get('htf_ok', 1)),
        SMC_BIAS_MAP.get(str(sf.get('smc_bias', 'NEUTRAL')), 0.0),
        PD_ZONE_MAP.get(str(sf.get('pd_zone', 'NEUTRAL')), 0.0),
        sf.get('atr_pips', 0) / 30.0,
        float(sf.get('has_bos', 0)),

        # ── General features from backtest_trades (4) ──
        float(row.get('atr', 0) or 0) / 30.0,
        abs(float(row.get('pip_from_vwap', 0) or 0)) / 50.0,
        abs(float(row.get('pip_to_poc', 0) or 0)) / 50.0,
        float(row.get('va_width_pips', 20) or 20) / 50.0,

        # ── Cross-strategy confluence (5) from strategy score columns ──
        float(row.get('ss_liquidity_sweep', 0) or 0) / 100.0,
        float(row.get('ss_breakout_momentum', 0) or 0) / 100.0,
        float(row.get('ss_vwap_reversion', 0) or 0) / 100.0,
        float(row.get('ss_ema_cross', 0) or 0) / 100.0,
        float(row.get('ss_delta_divergence', 0) or 0) / 100.0,
    ]
    return features
