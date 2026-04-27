# =============================================================
# ai_engine/delta_div_strategy_model.py  v1.0 — Layer 1 Delta Div Model
#
# PURPOSE: Specialized Layer 1 model for DELTA_DIVERGENCE strategy.
# Replaces the hard-coded gates in strategies/delta_divergence.py
# with a learned model that knows which delta divergence signals work.
#
# DELTA_DIV-SPECIFIC FEATURES (15 internal + 4 general + 5 cross-strategy = 24):
#   - div_type (BULLISH_DIVERGENCE / BEARISH_DIVERGENCE)
#   - div_strength (STRONG / MODERATE / WEAK)
#   - swing_range_pips (price range of the divergence swing)
#   - delta_value (delta CVD value at divergence point)
#   - delta_bias (BULLISH / BEARISH / NEUTRAL)
#   - of_imbalance (order flow imbalance value)
#   - of_strength (order flow strength level)
#   - vol_surge (volume surge detected)
#   - surge_ratio (volume surge magnitude)
#   - surge_absorption (surge was absorbed — potential reversal)
#   - stoch_rsi_k (M15 StochRSI K value)
#   - stoch_rsi_turning (StochRSI is turning in divergence direction)
#   - pd_zone (PREMIUM / DISCOUNT / EQUILIBRIUM)
#   - m5_body_ratio (M5 candle body/wick ratio)
#   - atr_pips (M15 ATR in pips)
#
# WHY DELTA_DIVERGENCE:
#   - Delta divergence between price and CVD is a powerful signal
#   - Current gates may be filtering out valid divergences
#   - Model can learn optimal divergence strength thresholds
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy DELTA_DIVERGENCE
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds delta_div-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "DELTA_DIVERGENCE"
STRATEGY_KEY = "delta_div"

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
# DELTA DIVERGENCE NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    Delta divergence evaluation WITHOUT hard-coded gates.

    Generates ALL potential divergence signals regardless of gate
    conditions. The Layer 1 model will decide which to PASS.

    Simplified detection: compare recent price swing direction with
    delta/CVD direction. If they disagree, it's a divergence.

    Returns: signal dict with _delta_div_features or None
    """
    if df_m15 is None or df_h1 is None or market_report is None:
        return None
    if len(df_m15) < 30 or len(df_h1) < 30:
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

    # ── Detect price swing ──
    lookback = 20
    h1_closes = df_h1['close'].astype(float).values[-(lookback + 1):]
    h1_highs = df_h1['high'].astype(float).values[-(lookback + 1):]
    h1_lows = df_h1['low'].astype(float).values[-(lookback + 1):]

    if len(h1_closes) < lookback:
        return None

    price_start = h1_closes[0]
    price_end = h1_closes[-1]
    price_change = price_end - price_start

    swing_high = float(np.max(h1_highs))
    swing_low = float(np.min(h1_lows))
    swing_range_pips = round((swing_high - swing_low) / pip_size, 2)

    if swing_range_pips < atr_pips * 0.5:
        return None

    # ── Detect delta direction ──
    rolling_delta = market_report.get('rolling_delta', {})
    delta_data = rolling_delta.get('delta', {})
    delta_value = float(delta_data.get('value', 0)) if delta_data else 0
    delta_bias = str(rolling_delta.get('bias', 'NEUTRAL'))

    # Detect divergence: price made new high but delta didn't (bearish div),
    # or price made new low but delta didn't (bullish div)
    direction = None
    div_type = 'NONE'
    div_strength = 'WEAK'

    # Find peaks/troughs in price and delta
    mid = len(h1_closes) // 2
    early_high = float(np.max(h1_closes[:mid]))
    late_high = float(np.max(h1_closes[mid:]))
    early_low = float(np.min(h1_closes[:mid]))
    late_low = float(np.min(h1_closes[mid:]))

    # Bearish divergence: price made higher high, but delta suggests selling
    if late_high > early_high and delta_bias in ('BEARISH', 'NEUTRAL'):
        direction = "SELL"
        div_type = "BEARISH_DIVERGENCE"
        price_diff = late_high - early_high
        if price_diff / pip_size > atr_pips * 2:
            div_strength = 'STRONG'
        elif price_diff / pip_size > atr_pips:
            div_strength = 'MODERATE'

    # Bullish divergence: price made lower low, but delta suggests buying
    elif late_low < early_low and delta_bias in ('BULLISH', 'NEUTRAL'):
        direction = "BUY"
        div_type = "BULLISH_DIVERGENCE"
        price_diff = early_low - late_low
        if price_diff / pip_size > atr_pips * 2:
            div_strength = 'STRONG'
        elif price_diff / pip_size > atr_pips:
            div_strength = 'MODERATE'

    if direction is None:
        return None

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"DIV_{div_type}")

    score += 8
    confluence.append(f"STR_{div_strength}")

    if swing_range_pips > atr_pips * 3:
        score += 8
        confluence.append(f"GOOD_SWING_{swing_range_pips:.0f}p")
    else:
        score -= 3
        confluence.append(f"SMALL_SWING_{swing_range_pips:.0f}p")

    # Order flow (scoring)
    of_imb = market_report.get('order_flow_imbalance', {})
    of_imbalance = float(of_imb.get('imbalance', 0))
    of_strength = str(of_imb.get('strength', 'NONE'))

    if (direction == "BUY" and of_imbalance > 0.05) or \
       (direction == "SELL" and of_imbalance < -0.05):
        score += 10
        confluence.append(f"OF_ALIGN_{of_imbalance:+.2f}")
    else:
        score -= 5
        confluence.append(f"OF_CONFLICT_{of_imbalance:+.2f}")

    # Volume surge (scoring)
    vol_surge = 0
    surge_ratio = 1.0
    surge_absorption = 0
    surge = market_report.get('volume_surge', {})
    if surge.get('surge_detected', False):
        vol_surge = 1
        surge_ratio = float(surge.get('surge_ratio', 1.0))
        score += 8
        confluence.append(f"VOL_SURGE_{surge_ratio:.1f}x")
        # Check absorption: surge but price didn't move much = absorption
        m5 = df_m5.iloc[-1] if df_m5 is not None and len(df_m5) > 0 else None
        if m5 is not None:
            m5_range = float(m5['high']) - float(m5['low'])
            m5_body = abs(float(m5['close']) - float(m5['open']))
            if m5_range > 0 and m5_body / m5_range < 0.4:
                surge_absorption = 1
                score += 10
                confluence.append("SURGE_ABSORBED")
    else:
        score -= 5
        confluence.append("NO_VOL_SURGE")

    # StochRSI (scoring)
    stoch_rsi_turning = 0
    if direction == "BUY" and stoch_rsi_k < 40:
        score += 10
        confluence.append("STOCHRSI_LOW")
        if stoch_rsi_k < 25:
            stoch_rsi_turning = 1
            score += 5
            confluence.append("STOCHRSI_TURNING_UP")
    elif direction == "SELL" and stoch_rsi_k > 60:
        score += 10
        confluence.append("STOCHRSI_HIGH")
        if stoch_rsi_k > 75:
            stoch_rsi_turning = 1
            score += 5
            confluence.append("STOCHRSI_TURNING_DOWN")

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

    # M5 body ratio (scoring)
    m5_body_ratio = 0.5
    if df_m5 is not None and len(df_m5) >= 2:
        m5_cur = df_m5.iloc[-1]
        m5_range = float(m5_cur['high']) - float(m5_cur['low'])
        m5_body = abs(float(m5_cur['close']) - float(m5_cur['open']))
        m5_body_ratio = round(m5_body / m5_range, 3) if m5_range > 0 else 0.5
        if m5_body_ratio > 0.6:
            body_dir = "BULL" if float(m5_cur['close']) > float(m5_cur['open']) else "BEAR"
            if (direction == "BUY" and body_dir == "BULL") or \
               (direction == "SELL" and body_dir == "BEAR"):
                score += 8
                confluence.append(f"M5_BODY_ALIGN_{m5_body_ratio:.2f}")

    # ── SL/TP calculation ──
    sl_pips = max(5.0, round(atr_pips * 1.5, 1))
    tp1_pips = round(swing_range_pips * 0.5, 1)
    tp2_pips = round(swing_range_pips * 0.8, 1)

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
        "_delta_div_features": {
            'div_type': div_type,
            'div_strength': div_strength,
            'swing_range_pips': swing_range_pips,
            'delta_value': delta_value,
            'delta_bias': delta_bias,
            'of_imbalance': of_imbalance,
            'of_strength': of_strength,
            'vol_surge': vol_surge,
            'surge_ratio': surge_ratio,
            'surge_absorption': surge_absorption,
            'stoch_rsi_k': stoch_rsi_k,
            'stoch_rsi_turning': stoch_rsi_turning,
            'pd_zone': pd_zone,
            'm5_body_ratio': m5_body_ratio,
            'atr_pips': atr_pips,
        }
    }


# ════════════════════════════════════════════════════════════════
# DELTA DIV FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_delta_div_features_from_db(row: dict) -> dict:
    """
    Extract delta divergence-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_delta_div_features).

    Falls back to computed defaults for trades without delta_div feature rows.
    """
    if row.get('div_type') is not None and row.get('div_type') != '':
        df = {
            'div_type': str(row.get('div_type', 'NONE')),
            'div_strength': str(row.get('div_strength', 'NONE')),
            'swing_range_pips': float(row.get('swing_range_pips', 0) or 0),
            'delta_value': float(row.get('delta_value', 0) or 0),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'vol_surge': int(row.get('vol_surge', 0) or 0),
            'surge_ratio': float(row.get('surge_ratio', 1.0) or 1.0),
            'surge_absorption': int(row.get('surge_absorption', 0) or 0),
            'stoch_rsi_k': float(row.get('stoch_rsi_k', 50) or 50),
            'stoch_rsi_turning': int(row.get('stoch_rsi_turning', 0) or 0),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'm5_body_ratio': float(row.get('m5_body_ratio', 0.5) or 0.5),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
        }
    else:
        df = {
            'div_type': 'BULLISH_DIVERGENCE' if row.get('direction') == 'BUY' else 'BEARISH_DIVERGENCE',
            'div_strength': 'MODERATE',
            'swing_range_pips': float(row.get('va_width_pips', 20) or 20),
            'delta_value': float(row.get('delta', 0) or 0),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'vol_surge': int(row.get('vol_surge_detected', 0) or 0),
            'surge_ratio': float(row.get('vol_surge_ratio', 1.0) or 1.0),
            'surge_absorption': 0,
            'stoch_rsi_k': 50.0,
            'stoch_rsi_turning': 0,
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'm5_body_ratio': 0.5,
            'atr_pips': float(row.get('atr', 0) or 0),
        }
    return df


def build_delta_div_feature_vector(row: dict) -> list:
    """Build delta divergence model feature vector from a DB row (with JOINed delta_div features).

    Returns a list of numeric features for the Delta Div Layer 1 model.
    Uses real delta_div features when available, falls back to derived values.
    """
    df = extract_delta_div_features_from_db(row)

    features = [
        # ── Delta Div-specific internal features (15) ──
        DIV_TYPE_MAP.get(str(df.get('div_type', 'NONE')), 0.0),
        DIV_STRENGTH_MAP.get(str(df.get('div_strength', 'NONE')), 0.0),
        df.get('swing_range_pips', 0) / 100.0,
        min(abs(df.get('delta_value', 0)), 500) / 500.0,
        1.0 if str(df.get('delta_bias', 'NEUTRAL')) == 'BULLISH' else (-1.0 if str(df.get('delta_bias', 'NEUTRAL')) == 'BEARISH' else 0.0),
        df.get('of_imbalance', 0) / 1.0,  # can be negative
        OF_STRENGTH_MAP.get(str(df.get('of_strength', 'NONE')), 0.0),
        float(df.get('vol_surge', 0)),
        min(df.get('surge_ratio', 1.0), 5.0) / 5.0,
        float(df.get('surge_absorption', 0)),
        df.get('stoch_rsi_k', 50) / 100.0,
        float(df.get('stoch_rsi_turning', 0)),
        PD_ZONE_MAP.get(str(df.get('pd_zone', 'NEUTRAL')), 0.0),
        df.get('m5_body_ratio', 0.5),  # already 0-1
        df.get('atr_pips', 0) / 30.0,

        # ── General features from backtest_trades (4) ──
        float(row.get('atr', 0) or 0) / 30.0,
        abs(float(row.get('pip_from_vwap', 0) or 0)) / 50.0,
        abs(float(row.get('pip_to_poc', 0) or 0)) / 50.0,
        float(row.get('va_width_pips', 20) or 20) / 50.0,

        # ── Cross-strategy confluence (5) from strategy score columns ──
        float(row.get('ss_smc_ob', 0) or 0) / 100.0,
        float(row.get('ss_liquidity_sweep', 0) or 0) / 100.0,
        float(row.get('ss_vwap_reversion', 0) or 0) / 100.0,
        float(row.get('ss_ema_cross', 0) or 0) / 100.0,
        float(row.get('ss_breakout_momentum', 0) or 0) / 100.0,
    ]
    return features
