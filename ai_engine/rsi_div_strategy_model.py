# =============================================================
# ai_engine/rsi_div_strategy_model.py  v1.0 — Layer 1 RSI Div Model
#
# PURPOSE: Specialized Layer 1 model for RSI_DIVERGENCE_SMC strategy.
# Replaces the hard-coded gates in strategies/rsi_divergence_smc.py
# with a learned model that knows which RSI divergence signals work.
#
# RSI_DIV-SPECIFIC FEATURES (16 internal + 4 general + 5 cross-strategy = 25):
#   - div_type (BULLISH_DIVERGENCE / BEARISH_DIVERGENCE)
#   - div_strength (STRONG / MODERATE / WEAK)
#   - rsi_diff (RSI difference between divergence swing points)
#   - curr_rsi (current RSI value)
#   - prev_rsi (previous swing RSI value)
#   - price_range_pips (price range of divergence swings)
#   - smc_confirmed (SMC structure confirms divergence)
#   - smc_bias (BULLISH / BEARISH / NEUTRAL)
#   - ob_distance_pips (distance to nearest order block)
#   - fvg_distance_pips (distance to nearest FVG)
#   - delta_bias (BULLISH / BEARISH / NEUTRAL)
#   - of_imbalance (order flow imbalance value)
#   - stoch_rsi_k (M15 StochRSI K value)
#   - pd_zone (PREMIUM / DISCOUNT / EQUILIBRIUM)
#   - atr_pips (M15 ATR in pips)
#
# WHY RSI_DIVERGENCE:
#   - RSI divergence is a classic reversal signal
#   - Combined with SMC concepts (OB, FVG, structure) for confluence
#   - Model can learn when RSI divergence is reliable vs noise
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy RSI_DIVERGENCE_SMC
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds rsi_div-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "RSI_DIVERGENCE_SMC"
STRATEGY_KEY = "rsi_div"

# Divergence type encoding
DIV_TYPE_MAP = {
    'BULLISH_DIVERGENCE':  1.0,
    'BEARISH_DIVERGENCE': -1.0,
    'NONE':                0.0,
}

# Divergence strength encoding
DIV_STRENGTH_MAP = {
    'STRONG':   1.0,
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
# RSI DIVERGENCE NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    RSI divergence evaluation WITHOUT hard-coded gates.

    Generates ALL potential RSI divergence signals regardless of
    gate conditions. The Layer 1 model will decide which to PASS.

    Simplified detection: find RSI peaks/troughs and compare with
    price peaks/troughs. If they disagree, it's a divergence.

    Returns: signal dict with _rsi_div_features or None
    """
    if df_h1 is None or df_m15 is None:
        return None
    if len(df_h1) < 40:
        return None

    h1 = df_h1.iloc[-1]
    m15 = df_m15.iloc[-1]
    close_price = float(m15['close'])

    from core.pip_utils import get_pip_size as _gps
    pip_size = _gps(symbol, close_price)

    atr_pips = float(m15.get('atr', 0)) / pip_size if pip_size > 0 else 0
    stoch_rsi_k = float(m15.get('stoch_rsi_k', 50))

    if atr_pips < 1.0:
        return None

    # ── Detect RSI divergence on H1 ──
    h1_closes = df_h1['close'].astype(float).values
    h1_rsi_vals = df_h1['rsi'].astype(float).values if 'rsi' in df_h1.columns else None

    if h1_rsi_vals is None or len(h1_rsi_vals) < 30:
        # Fallback: compute RSI from H1 closes
        delta = np.diff(h1_closes)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
        avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        h1_rsi_vals = 100.0 - (100.0 / (1.0 + rs))

    if len(h1_rsi_vals) < 20:
        return None

    # Find peaks and troughs in RSI
    lookback = min(30, len(h1_rsi_vals) - 1)
    recent_rsi = h1_rsi_vals[-lookback:]
    recent_closes = h1_closes[-lookback:]

    direction = None
    div_type = 'NONE'
    div_strength = 'WEAK'
    rsi_diff = 0.0
    curr_rsi = float(recent_rsi[-1])
    prev_rsi = 50.0
    price_range_pips = 0.0

    mid = len(recent_rsi) // 2

    # Find RSI peak and trough in each half
    early_peak_rsi = float(np.max(recent_rsi[:mid]))
    early_trough_rsi = float(np.min(recent_rsi[:mid]))
    late_peak_rsi = float(np.max(recent_rsi[mid:]))
    late_trough_rsi = float(np.min(recent_rsi[mid:]))

    early_peak_idx = int(np.argmax(recent_rsi[:mid]))
    early_trough_idx = int(np.argmin(recent_rsi[:mid]))
    late_peak_idx = mid + int(np.argmax(recent_rsi[mid:]))
    late_trough_idx = mid + int(np.argmin(recent_rsi[mid:]))

    # Bearish divergence: price makes higher high but RSI makes lower high
    early_price_high = float(np.max(recent_closes[:mid]))
    late_price_high = float(np.max(recent_closes[mid:]))
    if late_price_high > early_price_high and late_peak_rsi < early_peak_rsi:
        direction = "SELL"
        div_type = "BEARISH_DIVERGENCE"
        rsi_diff = round(early_peak_rsi - late_peak_rsi, 2)
        prev_rsi = round(early_peak_rsi, 2)
        price_range_pips = round((late_price_high - early_price_high) / pip_size, 2)
        if rsi_diff > 10:
            div_strength = 'STRONG'
        elif rsi_diff > 5:
            div_strength = 'MODERATE'

    # Bullish divergence: price makes lower low but RSI makes higher low
    if direction is None:
        early_price_low = float(np.min(recent_closes[:mid]))
        late_price_low = float(np.min(recent_closes[mid:]))
        if late_price_low < early_price_low and late_trough_rsi > early_trough_rsi:
            direction = "BUY"
            div_type = "BULLISH_DIVERGENCE"
            rsi_diff = round(late_trough_rsi - early_trough_rsi, 2)
            prev_rsi = round(early_trough_rsi, 2)
            price_range_pips = round((early_price_low - late_price_low) / pip_size, 2)
            if rsi_diff > 10:
                div_strength = 'STRONG'
            elif rsi_diff > 5:
                div_strength = 'MODERATE'

    if direction is None:
        return None

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"RSI_DIV_{div_type}")

    score += 8
    confluence.append(f"STR_{div_strength}")

    if rsi_diff > 8:
        score += 8
        confluence.append(f"STRONG_DIFF_{rsi_diff:.1f}")

    # StochRSI confirmation (scoring)
    if direction == "BUY" and stoch_rsi_k < 35:
        score += 10
        confluence.append("STOCHRSI_OVERSOLD")
    elif direction == "SELL" and stoch_rsi_k > 65:
        score += 10
        confluence.append("STOCHRSI_OVERBOUGHT")

    # SMC confirmation (scoring)
    smc_confirmed = 0
    smc_bias = 'NEUTRAL'
    ob_distance_pips = 0.0
    fvg_distance_pips = 0.0

    if smc_report:
        smc_bias = smc_report.get('bias', 'NEUTRAL')
        if (direction == "BUY" and smc_bias == "BULLISH") or \
           (direction == "SELL" and smc_bias == "BEARISH"):
            smc_confirmed = 1
            score += 12
            confluence.append("SMC_CONFIRMS")

        # Find nearest OB
        obs = smc_report.get('order_blocks', [])
        min_ob_dist = float('inf')
        for ob in obs:
            if ob.get('mitigated', False):
                continue
            ob_mid = (float(ob.get('high', 0)) + float(ob.get('low', 0))) / 2
            dist = abs(close_price - ob_mid) / pip_size
            if dist < min_ob_dist:
                min_ob_dist = dist
        ob_distance_pips = round(min_ob_dist, 2) if min_ob_dist < float('inf') else 0

        # Find FVG zones (from structure)
        fvgs = smc_report.get('fvgs', [])
        if not fvgs:
            fvgs = smc_report.get('fair_value_gaps', [])
        min_fvg_dist = float('inf')
        for fvg in fvgs:
            fvg_top = float(fvg.get('high', fvg.get('top', 0)))
            fvg_bot = float(fvg.get('low', fvg.get('bottom', 0)))
            fvg_mid = (fvg_top + fvg_bot) / 2
            dist = abs(close_price - fvg_mid) / pip_size
            if dist < min_fvg_dist:
                min_fvg_dist = dist
        fvg_distance_pips = round(min_fvg_dist, 2) if min_fvg_dist < float('inf') else 0

        if ob_distance_pips < atr_pips * 2:
            score += 8
            confluence.append(f"NEAR_OB_{ob_distance_pips:.1f}p")
        if fvg_distance_pips < atr_pips * 2:
            score += 5
            confluence.append(f"NEAR_FVG_{fvg_distance_pips:.1f}p")

    # Delta (scoring)
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_bias = str(rolling_delta.get('bias', 'NEUTRAL'))
    if (direction == "BUY" and delta_bias == "BULLISH") or \
       (direction == "SELL" and delta_bias == "BEARISH"):
        score += 10
        confluence.append("DELTA_CONFIRMS")
    else:
        score -= 5
        confluence.append("NO_DELTA")

    # Order flow (scoring)
    of_imb = market_report.get('order_flow_imbalance', {}) if market_report else {}
    of_imbalance = float(of_imb.get('imbalance', 0))
    if (direction == "BUY" and of_imbalance > 0.05) or \
       (direction == "SELL" and of_imbalance < -0.05):
        score += 8
        confluence.append(f"OF_ALIGN_{of_imbalance:+.2f}")

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

    # Choppy penalty (scoring)
    is_choppy = 0
    if master_report:
        momentum = master_report.get('momentum', {})
        is_choppy = 1 if momentum.get('is_choppy', False) else 0
        if is_choppy:
            score -= 15
            confluence.append("CHOPPY")

    # ── SL/TP calculation ──
    sl_pips = max(5.0, round(atr_pips * 1.5, 1))
    tp1_pips = round(price_range_pips * 0.5 if price_range_pips > 0 else sl_pips * 2, 1)
    tp2_pips = round(price_range_pips * 0.8 if price_range_pips > 0 else sl_pips * 3, 1)

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
        "_rsi_div_features": {
            'div_type': div_type,
            'div_strength': div_strength,
            'rsi_diff': rsi_diff,
            'curr_rsi': round(curr_rsi, 2),
            'prev_rsi': prev_rsi,
            'price_range_pips': price_range_pips,
            'smc_confirmed': smc_confirmed,
            'smc_bias': smc_bias,
            'ob_distance_pips': ob_distance_pips,
            'fvg_distance_pips': fvg_distance_pips,
            'delta_bias': delta_bias,
            'of_imbalance': of_imbalance,
            'stoch_rsi_k': stoch_rsi_k,
            'pd_zone': pd_zone,
            'is_choppy': is_choppy,
            'atr_pips': atr_pips,
        }
    }


# ════════════════════════════════════════════════════════════════
# RSI DIV FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_rsi_div_features_from_db(row: dict) -> dict:
    """
    Extract RSI divergence-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_rsi_div_features).

    Falls back to computed defaults for trades without rsi_div feature rows.
    """
    if row.get('div_type') is not None and row.get('div_type') != '':
        rf = {
            'div_type': str(row.get('div_type', 'NONE')),
            'div_strength': str(row.get('div_strength', 'NONE')),
            'rsi_diff': float(row.get('rsi_diff', 0) or 0),
            'curr_rsi': float(row.get('curr_rsi', 50) or 50),
            'prev_rsi': float(row.get('prev_rsi', 50) or 50),
            'price_range_pips': float(row.get('price_range_pips', 0) or 0),
            'smc_confirmed': int(row.get('smc_confirmed', 0) or 0),
            'smc_bias': str(row.get('smc_bias', 'NEUTRAL')),
            'ob_distance_pips': float(row.get('ob_distance_pips', 0) or 0),
            'fvg_distance_pips': float(row.get('fvg_distance_pips', 0) or 0),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'stoch_rsi_k': float(row.get('stoch_rsi_k', 50) or 50),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
        }
    else:
        rf = {
            'div_type': 'BULLISH_DIVERGENCE' if row.get('direction') == 'BUY' else 'BEARISH_DIVERGENCE',
            'div_strength': 'MODERATE',
            'rsi_diff': 5.0,
            'curr_rsi': 50.0,
            'prev_rsi': 55.0,
            'price_range_pips': float(row.get('va_width_pips', 20) or 20),
            'smc_confirmed': 1 if str(row.get('smc_bias', '')) in ('BULLISH', 'BEARISH') else 0,
            'smc_bias': str(row.get('smc_bias', 'NEUTRAL')),
            'ob_distance_pips': abs(float(row.get('pip_from_vwap', 0) or 0)),
            'fvg_distance_pips': abs(float(row.get('pip_to_poc', 0) or 0)),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'stoch_rsi_k': 50.0,
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'atr_pips': float(row.get('atr', 0) or 0),
        }
    return rf


def build_rsi_div_feature_vector(row: dict) -> list:
    """Build RSI divergence model feature vector from a DB row (with JOINed rsi_div features).

    Returns a list of numeric features for the RSI Div Layer 1 model.
    Uses real rsi_div features when available, falls back to derived values.
    """
    rf = extract_rsi_div_features_from_db(row)

    features = [
        # ── RSI Div-specific internal features (15) ──
        DIV_TYPE_MAP.get(str(rf.get('div_type', 'NONE')), 0.0),
        DIV_STRENGTH_MAP.get(str(rf.get('div_strength', 'NONE')), 0.0),
        rf.get('rsi_diff', 0) / 30.0,
        rf.get('curr_rsi', 50) / 100.0,
        rf.get('prev_rsi', 50) / 100.0,
        rf.get('price_range_pips', 0) / 100.0,
        float(rf.get('smc_confirmed', 0)),
        SMC_BIAS_MAP.get(str(rf.get('smc_bias', 'NEUTRAL')), 0.0),
        rf.get('ob_distance_pips', 0) / 50.0,
        rf.get('fvg_distance_pips', 0) / 50.0,
        1.0 if str(rf.get('delta_bias', 'NEUTRAL')) == 'BULLISH' else (-1.0 if str(rf.get('delta_bias', 'NEUTRAL')) == 'BEARISH' else 0.0),
        rf.get('of_imbalance', 0) / 1.0,  # can be negative
        rf.get('stoch_rsi_k', 50) / 100.0,
        PD_ZONE_MAP.get(str(rf.get('pd_zone', 'NEUTRAL')), 0.0),
        float(rf.get('is_choppy', 0)),
        rf.get('atr_pips', 0) / 30.0,

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
        float(row.get('ss_delta_divergence', 0) or 0) / 100.0,
    ]
    return features
