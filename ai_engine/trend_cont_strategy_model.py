# =============================================================
# ai_engine/trend_cont_strategy_model.py  v1.0 — Layer 1 Trend Cont Model
#
# PURPOSE: Specialized Layer 1 model for TREND_CONTINUATION strategy.
# Replaces the hard-coded gates in strategies/trend_continuation.py
# with a learned model that knows which trend continuation signals work.
#
# TREND_CONT-SPECIFIC FEATURES (15 internal + 4 general + 5 cross-strategy = 24):
#   - h4_trend_score (H4 trend strength score 0-100)
#   - pullback_ema_type (EMA_9 / EMA_21 / BOTH — which EMA was tested)
#   - pullback_dist_pips (distance from EMA at pullback)
#   - h1_ema_aligned (H1 EMAs aligned with H4 trend)
#   - h1_supertrend_dir (H1 supertrend direction)
#   - of_imbalance (order flow imbalance value)
#   - of_strength (order flow strength level)
#   - delta_confirms (delta direction matches trend)
#   - rejection_type (PIN_BAR / ENGULFING / HAMMER / NONE)
#   - velocity_pips (price velocity in pips per bar)
#   - velocity_dir (UP / DOWN / FLAT)
#   - is_scalpable (trade meets scalp criteria)
#   - market_state (TRENDING / RANGING / VOLATILE)
#   - is_choppy (market is choppy)
#   - atr_pips (M15 ATR in pips)
#
# WHY TREND_CONTINUATION:
#   - Trend continuation is highest-probability when it works
#   - Pullback depth and EMA type gates may be too strict
#   - Model can learn optimal pullback levels per market state
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy TREND_CONTINUATION
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds trend_cont-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "TREND_CONTINUATION"
STRATEGY_KEY = "trend_cont"

# Pullback EMA type encoding
PULLBACK_EMA_MAP = {
    'EMA_9':  1.0,
    'EMA_21': 0.5,
    'BOTH':   0.75,
    'NONE':   0.0,
}

# OF strength encoding
OF_STRENGTH_MAP = {
    'EXTREME':  1.0,
    'STRONG':   0.75,
    'MODERATE': 0.5,
    'WEAK':     0.25,
    'NONE':     0.0,
}

# Rejection type encoding
REJECTION_MAP = {
    'PIN_BAR':   1.0,
    'ENGULFING': 0.75,
    'HAMMER':    0.5,
    'NONE':      0.0,
}

# Velocity direction encoding
VELOCITY_DIR_MAP = {
    'UP':    1.0,
    'DOWN': -1.0,
    'FLAT':  0.0,
    'NONE':  0.0,
}

# Market state encoding
MARKET_STATE_MAP = {
    'TRENDING':  1.0,
    'RANGING':  -0.5,
    'VOLATILE':   0.5,
    'NONE':       0.0,
}


# ════════════════════════════════════════════════════════════════
# TREND CONTINUATION NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    Trend continuation evaluation WITHOUT hard-coded gates.

    Generates ALL potential trend continuation signals regardless of
    gate conditions. The Layer 1 model will decide which to PASS.

    Simplified detection: identify H4 trend direction, look for
    H1 pullback to EMAs, score continuation probability.

    Returns: signal dict with _trend_cont_features or None
    """
    if df_h4 is None or df_h1 is None or df_m15 is None:
        return None
    if len(df_h4) < 30 or len(df_h1) < 20:
        return None

    h4 = df_h4.iloc[-1]
    h1 = df_h1.iloc[-1]
    m15 = df_m15.iloc[-1]
    close_price = float(m15['close'])

    from core.pip_utils import get_pip_size as _gps
    pip_size = _gps(symbol, close_price)

    atr_pips = float(m15.get('atr', 0)) / pip_size if pip_size > 0 else 0

    if atr_pips < 1.0:
        return None

    # ── Step 1: Detect H4 trend ──
    h4_ema9 = float(h4.get('ema_9', 0))
    h4_ema21 = float(h4.get('ema_21', 0))
    h4_ema50 = float(h4.get('ema_50', 0))
    h4_close = float(h4['close'])
    h4_st = int(h4.get('supertrend_dir', 0))

    h4_trend_score = 0
    direction = None

    # Bullish trend scoring
    bull_score = 0
    if h4_ema9 > h4_ema21:
        bull_score += 25
    if h4_ema21 > h4_ema50:
        bull_score += 25
    if h4_close > h4_ema9:
        bull_score += 25
    if h4_st == 1:
        bull_score += 25

    # Bearish trend scoring
    bear_score = 0
    if h4_ema9 < h4_ema21:
        bear_score += 25
    if h4_ema21 < h4_ema50:
        bear_score += 25
    if h4_close < h4_ema9:
        bear_score += 25
    if h4_st == -1:
        bear_score += 25

    if bull_score >= 50:
        direction = "BUY"
        h4_trend_score = bull_score
    elif bear_score >= 50:
        direction = "SELL"
        h4_trend_score = bear_score
    else:
        return None

    # ── Step 2: Detect H1 pullback ──
    h1_ema9 = float(h1.get('ema_9', 0))
    h1_ema21 = float(h1.get('ema_21', 0))
    h1_close = float(h1['close'])
    h1_low = float(h1['low'])
    h1_high = float(h1['high'])
    h1_supertrend_dir = int(h1.get('supertrend_dir', 0))

    pullback_ema_type = 'NONE'
    pullback_dist_pips = 0.0

    if direction == "BUY":
        # Price pulled back to EMA zone
        if h1_low <= h1_ema9 and h1_close > h1_ema9:
            pullback_ema_type = 'EMA_9'
            pullback_dist_pips = round((h1_ema9 - h1_low) / pip_size, 2)
        elif h1_low <= h1_ema21 and h1_close > h1_ema21:
            pullback_ema_type = 'EMA_21'
            pullback_dist_pips = round((h1_ema21 - h1_low) / pip_size, 2)
        elif h1_low <= h1_ema9 and h1_low <= h1_ema21 and h1_close > h1_ema9:
            pullback_ema_type = 'BOTH'
            pullback_dist_pips = round((h1_ema9 - h1_low) / pip_size, 2)
    else:
        if h1_high >= h1_ema9 and h1_close < h1_ema9:
            pullback_ema_type = 'EMA_9'
            pullback_dist_pips = round((h1_high - h1_ema9) / pip_size, 2)
        elif h1_high >= h1_ema21 and h1_close < h1_ema21:
            pullback_ema_type = 'EMA_21'
            pullback_dist_pips = round((h1_high - h1_ema21) / pip_size, 2)
        elif h1_high >= h1_ema9 and h1_high >= h1_ema21 and h1_close < h1_ema9:
            pullback_ema_type = 'BOTH'
            pullback_dist_pips = round((h1_high - h1_ema9) / pip_size, 2)

    if pullback_ema_type == 'NONE':
        # Relaxed: accept any bar in trend direction even without pullback
        if direction == "BUY" and h1_close > h1_ema9:
            pullback_ema_type = 'EMA_9'
            pullback_dist_pips = 0
        elif direction == "SELL" and h1_close < h1_ema9:
            pullback_ema_type = 'EMA_9'
            pullback_dist_pips = 0
        else:
            return None

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"H4_TREND_{direction}_{h4_trend_score}")

    # H1 EMA alignment (scoring)
    h1_ema_aligned = 0
    if direction == "BUY" and h1_ema9 > h1_ema21:
        h1_ema_aligned = 1
        score += 10
        confluence.append("H1_EMA_BULL")
    elif direction == "SELL" and h1_ema9 < h1_ema21:
        h1_ema_aligned = 1
        score += 10
        confluence.append("H1_EMA_BEAR")

    # Supertrend H1 (scoring)
    if direction == "BUY" and h1_supertrend_dir == 1:
        score += 8
        confluence.append("ST_BULL")
    elif direction == "SELL" and h1_supertrend_dir == -1:
        score += 8
        confluence.append("ST_BEAR")

    # Pullback quality (scoring)
    if pullback_dist_pips > 0 and pullback_dist_pips < atr_pips * 2:
        score += 8
        confluence.append(f"GOOD_PULLBACK_{pullback_ema_type}_{pullback_dist_pips:.1f}p")
    elif pullback_dist_pips == 0:
        score -= 5
        confluence.append("NO_PULLBACK")
    else:
        score -= 3
        confluence.append(f"DEEP_PULLBACK_{pullback_dist_pips:.1f}p")

    # Delta confirmation (scoring)
    delta_confirms = 0
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')
    if (direction == "BUY" and delta_bias == "BULLISH") or \
       (direction == "SELL" and delta_bias == "BEARISH"):
        delta_confirms = 1
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

    # Rejection candle (scoring)
    rejection_type = 'NONE'
    h1_open = float(h1['open'])
    h1_body = abs(h1_close - h1_open)
    h1_range = h1_high - h1_low
    h1_upper_wick = h1_high - max(h1_close, h1_open)
    h1_lower_wick = min(h1_close, h1_open) - h1_low

    if h1_range > 0:
        if h1_upper_wick / h1_range > 0.6 and h1_body / h1_range < 0.3:
            rejection_type = 'PIN_BAR'
            score += 10
            confluence.append("PIN_BAR_REJECTION")
        elif h1_lower_wick / h1_range > 0.6 and h1_body / h1_range < 0.3:
            rejection_type = 'HAMMER'
            score += 10
            confluence.append("HAMMER_REJECTION")
        elif h1_body / h1_range > 0.7:
            rejection_type = 'ENGULFING'
            score += 8
            confluence.append("ENGULFING")

    # Velocity (scoring)
    velocity_pips = 0
    velocity_dir = 'FLAT'
    if len(df_m15) >= 5:
        price_change = float(df_m15.iloc[-1]['close']) - float(df_m15.iloc[-5]['close'])
        velocity_pips = round(abs(price_change) / pip_size / 5, 2)
        velocity_dir = 'UP' if price_change > 0 else ('DOWN' if price_change < 0 else 'FLAT')
        if (direction == "BUY" and velocity_dir == 'UP') or \
           (direction == "SELL" and velocity_dir == 'DOWN'):
            score += 5
            confluence.append(f"VELOCITY_ALIGN_{velocity_pips:.1f}p/bar")

    # Scalpable (scoring)
    is_scalpable = 1 if atr_pips < 15 else 0

    # Market state (scoring)
    market_state = 'NONE'
    if master_report:
        momentum = master_report.get('momentum', {})
        market_state = str(momentum.get('market_state', 'NONE'))

    is_choppy = 0
    if master_report:
        momentum = master_report.get('momentum', {})
        is_choppy = 1 if momentum.get('is_choppy', False) else 0
        if is_choppy:
            score -= 15
            confluence.append("CHOPPY")
        elif market_state == 'TRENDING':
            score += 10
            confluence.append("TRENDING_MARKET")

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
        "_trend_cont_features": {
            'h4_trend_score': h4_trend_score,
            'pullback_ema_type': pullback_ema_type,
            'pullback_dist_pips': pullback_dist_pips,
            'h1_ema_aligned': h1_ema_aligned,
            'h1_supertrend_dir': h1_supertrend_dir,
            'of_imbalance': of_imbalance,
            'of_strength': of_strength,
            'delta_confirms': delta_confirms,
            'rejection_type': rejection_type,
            'velocity_pips': velocity_pips,
            'velocity_dir': velocity_dir,
            'is_scalpable': is_scalpable,
            'market_state': market_state,
            'is_choppy': is_choppy,
            'atr_pips': atr_pips,
        }
    }


# ════════════════════════════════════════════════════════════════
# TREND CONT FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_trend_cont_features_from_db(row: dict) -> dict:
    """
    Extract trend continuation-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_trend_cont_features).

    Falls back to computed defaults for trades without trend_cont feature rows.
    """
    if row.get('h4_trend_score') is not None and row.get('h4_trend_score') != 0:
        tf = {
            'h4_trend_score': int(row.get('h4_trend_score', 0) or 0),
            'pullback_ema_type': str(row.get('pullback_ema_type', 'NONE')),
            'pullback_dist_pips': float(row.get('pullback_dist_pips', 0) or 0),
            'h1_ema_aligned': int(row.get('h1_ema_aligned', 0) or 0),
            'h1_supertrend_dir': int(row.get('h1_supertrend_dir', 0) or 0),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'delta_confirms': int(row.get('delta_confirms', 0) or 0),
            'rejection_type': str(row.get('rejection_type', 'NONE')),
            'velocity_pips': float(row.get('velocity_pips', 0) or 0),
            'velocity_dir': str(row.get('velocity_dir', 'FLAT')),
            'is_scalpable': int(row.get('is_scalpable', 0) or 0),
            'market_state': str(row.get('market_state', 'NONE')),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
        }
    else:
        tf = {
            'h4_trend_score': int(row.get('score', 0) or 0),
            'pullback_ema_type': 'NONE',
            'pullback_dist_pips': abs(float(row.get('pip_from_vwap', 0) or 0)),
            'h1_ema_aligned': 0,
            'h1_supertrend_dir': 0,
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'of_strength': str(row.get('of_strength', 'NONE')),
            'delta_confirms': 1 if str(row.get('delta_bias', '')) == 'BULLISH' or str(row.get('delta_bias', '')) == 'BEARISH' else 0,
            'rejection_type': 'NONE',
            'velocity_pips': abs(float(row.get('momentum_velocity', 0) or 0)),
            'velocity_dir': 'UP' if float(row.get('momentum_velocity', 0) or 0) > 0 else 'DOWN',
            'is_scalpable': 0,
            'market_state': str(row.get('market_state', 'NONE')),
            'is_choppy': int(row.get('is_choppy', 0) or 0),
            'atr_pips': float(row.get('atr', 0) or 0),
        }
    return tf


def build_trend_cont_feature_vector(row: dict) -> list:
    """Build trend continuation model feature vector from a DB row (with JOINed trend_cont features).

    Returns a list of numeric features for the Trend Cont Layer 1 model.
    Uses real trend_cont features when available, falls back to derived values.
    """
    tf = extract_trend_cont_features_from_db(row)

    features = [
        # ── Trend Cont-specific internal features (15) ──
        tf.get('h4_trend_score', 0) / 100.0,
        PULLBACK_EMA_MAP.get(str(tf.get('pullback_ema_type', 'NONE')), 0.0),
        tf.get('pullback_dist_pips', 0) / 30.0,
        float(tf.get('h1_ema_aligned', 0)),
        float(tf.get('h1_supertrend_dir', 0)) / 1.0,
        tf.get('of_imbalance', 0) / 1.0,  # can be negative
        OF_STRENGTH_MAP.get(str(tf.get('of_strength', 'NONE')), 0.0),
        float(tf.get('delta_confirms', 0)),
        REJECTION_MAP.get(str(tf.get('rejection_type', 'NONE')), 0.0),
        tf.get('velocity_pips', 0) / 10.0,
        VELOCITY_DIR_MAP.get(str(tf.get('velocity_dir', 'FLAT')), 0.0),
        float(tf.get('is_scalpable', 0)),
        MARKET_STATE_MAP.get(str(tf.get('market_state', 'NONE')), 0.0),
        float(tf.get('is_choppy', 0)),
        tf.get('atr_pips', 0) / 30.0,

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
