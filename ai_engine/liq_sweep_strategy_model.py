# =============================================================
# ai_engine/liq_sweep_strategy_model.py  v1.0 — Layer 1 Liq Sweep Model
#
# PURPOSE: Specialized Layer 1 model for LIQUIDITY_SWEEP_ENTRY strategy.
# Replaces the hard-coded gates in strategies/liquidity_sweep_entry.py
# with a learned model that knows which liquidity sweep signals work.
#
# LIQ_SWEEP-SPECIFIC FEATURES (15 internal + 4 general + 5 cross-strategy = 24):
#   - sweep_bias (BULLISH / BEARISH — direction of expected reversal)
#   - reversal_pips (potential reversal distance in pips)
#   - swept_level_dist (distance from swept level in pips)
#   - delta_bias (BULLISH / BEARISH / NEUTRAL)
#   - delta_strength (STRONG / MODERATE / WEAK)
#   - has_bos (BOS confirms reversal direction)
#   - bos_type (BULLISH_BOS / BEARISH_BOS)
#   - stoch_rsi_k (M15 StochRSI K value)
#   - supertrend_dir_h1 (H1 supertrend direction)
#   - htf_ok (HTF alignment approved)
#   - smc_bias (BULLISH / BEARISH / NEUTRAL)
#   - pd_zone (PREMIUM / DISCOUNT / EQUILIBRIUM)
#   - vol_surge (volume surge detected)
#   - of_imbalance (order flow imbalance value)
#   - atr_pips (M15 ATR in pips)
#
# WHY LIQUIDITY SWEEP:
#   - Sweep-and-reverse is a core SMC concept
#   - Many false sweeps — model can learn true vs fake sweeps
#   - Volume and delta confirmation gates may be too strict
#
# TRAINING:
#   python -m backtest.run --train-strategy-model --strategy LIQUIDITY_SWEEP_ENTRY
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds liq_sweep-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "LIQUIDITY_SWEEP_ENTRY"
STRATEGY_KEY = "liq_sweep"

# Sweep bias encoding
SWEEP_BIAS_MAP = {
    'BULLISH':  1.0,
    'BEARISH': -1.0,
    'NONE':     0.0,
}

# Delta strength encoding
DELTA_STRENGTH_MAP = {
    'STRONG':   1.0,
    'MODERATE': 0.5,
    'WEAK':     0.25,
    'NONE':     0.0,
}

# BOS type encoding
BOS_TYPE_MAP = {
    'BULLISH_BOS':  1.0,
    'BEARISH_BOS': -1.0,
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
# LIQUIDITY SWEEP NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    Liquidity sweep evaluation WITHOUT hard-coded gates.

    Generates ALL potential sweep reversal signals regardless of
    gate conditions. The Layer 1 model will decide which to PASS.

    Simplified detection: look for price wicks that sweep beyond
    recent swing highs/lows, then score reversal probability.

    Returns: signal dict with _liq_sweep_features or None
    """
    if df_m15 is None or df_h1 is None:
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
    supertrend_dir_h1 = int(h1.get('supertrend_dir', 0))

    if atr_pips < 1.0:
        return None

    # ── Detect swing highs/lows from H1 data ──
    h1_highs = df_h1['high'].astype(float).values[-30:]
    h1_lows = df_h1['low'].astype(float).values[-30:]
    h1_closes = df_h1['close'].astype(float).values[-30:]

    # Recent swing high (highest high in lookback minus last 3 bars)
    swing_high = float(np.max(h1_highs[-20:-3])) if len(h1_highs) > 3 else 0
    swing_low = float(np.min(h1_lows[-20:-3])) if len(h1_lows) > 3 else 0

    if swing_high == 0 or swing_low == 0:
        return None

    # Current bar wick analysis
    cur_high = float(m15['high'])
    cur_low = float(m15['low'])
    cur_body_high = max(float(m15['open']), float(m15['close']))
    cur_body_low = min(float(m15['open']), float(m15['close']))
    upper_wick = cur_high - cur_body_high
    lower_wick = cur_body_low - cur_low

    direction = None
    swept_level_dist = 0.0

    # Detect bearish sweep: price swept above swing high then reversed down
    if cur_high > swing_high and cur_body_high < swing_high:
        upper_wick_pips = upper_wick / pip_size if pip_size > 0 else 0
        if upper_wick_pips >= atr_pips * 0.3:
            direction = "SELL"
            swept_level_dist = round((cur_high - swing_high) / pip_size, 2)
            reversal_pips = round((cur_body_low - swing_low) / pip_size, 2)

    # Detect bullish sweep: price swept below swing low then reversed up
    if direction is None and cur_low < swing_low and cur_body_low > swing_low:
        lower_wick_pips = lower_wick / pip_size if pip_size > 0 else 0
        if lower_wick_pips >= atr_pips * 0.3:
            direction = "BUY"
            swept_level_dist = round((swing_low - cur_low) / pip_size, 2)
            reversal_pips = round((swing_high - cur_body_high) / pip_size, 2)

    if direction is None:
        return None

    sweep_bias = "BULLISH" if direction == "BUY" else "BEARISH"
    reversal_pips = abs(reversal_pips) if reversal_pips else 0

    # ── Collect all features as scoring (no hard blocks) ──
    score = 0
    confluence = []

    score += 15
    confluence.append(f"SWEEP_{direction}")

    if swept_level_dist < atr_pips:
        score += 10
        confluence.append(f"TIGHT_SWEEP_{swept_level_dist:.1f}p")
    else:
        score -= 5
        confluence.append(f"WIDE_SWEEP_{swept_level_dist:.1f}p")

    if reversal_pips > sl_pips * 2 if (sl_pips := atr_pips * 1.5) else False:
        score += 10
        confluence.append(f"GOOD_REVERSAL_{reversal_pips:.0f}p")

    # Delta (scoring)
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')
    delta_data = rolling_delta.get('delta', {})
    delta_abs = abs(float(delta_data.get('value', 0))) if delta_data else 0
    delta_strength = 'STRONG' if delta_abs > 100 else ('MODERATE' if delta_abs > 40 else 'WEAK')

    if direction == "BUY" and delta_bias == "BULLISH":
        score += 12
        confluence.append("DELTA_BULL")
    elif direction == "SELL" and delta_bias == "BEARISH":
        score += 12
        confluence.append("DELTA_BEAR")
    else:
        score -= 8
        confluence.append("NO_DELTA_CONFIRM")

    # Order flow (scoring)
    of_imb = market_report.get('order_flow_imbalance', {}) if market_report else {}
    of_imbalance = float(of_imb.get('imbalance', 0))
    of_strength = str(of_imb.get('strength', 'NONE'))

    if (direction == "BUY" and of_imbalance > 0.05) or \
       (direction == "SELL" and of_imbalance < -0.05):
        score += 8
        confluence.append(f"OF_ALIGN_{of_imbalance:+.2f}")

    # BOS (scoring)
    has_bos = 0
    bos_type = 'NONE'
    if smc_report:
        structure = smc_report.get('structure', {})
        bos_list = structure.get('bos', [])
        if bos_list:
            for bos in bos_list:
                bt = str(bos.get('type', ''))
                if (direction == "BUY" and 'BULL' in bt.upper()) or \
                   (direction == "SELL" and 'BEAR' in bt.upper()):
                    has_bos = 1
                    bos_type = bt
                    score += 8
                    confluence.append("BOS_CONFIRMED")
                    break

    # StochRSI (scoring)
    if direction == "BUY" and stoch_rsi_k < 35:
        score += 10
        confluence.append("STOCHRSI_OVERSOLD")
    elif direction == "SELL" and stoch_rsi_k > 65:
        score += 10
        confluence.append("STOCHRSI_OVERBOUGHT")

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

    # Volume surge (scoring)
    vol_surge = 0
    surge = market_report.get('volume_surge', {}) if market_report else {}
    if surge.get('surge_detected', False):
        vol_surge = 1
        score += 10
        confluence.append(f"VOL_SURGE_{surge.get('surge_ratio', 1.0):.1f}x")
    else:
        score -= 5
        confluence.append("NO_VOL_SURGE")

    # ── SL/TP calculation ──
    sl_pips = max(5.0, round(atr_pips * 1.5, 1))
    tp1_pips = round(swept_level_dist * 2.0 if swept_level_dist > 0 else sl_pips * 2, 1)
    tp2_pips = round(tp1_pips * 1.5, 1)

    # Keep minimum R:R check
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
        "_liq_sweep_features": {
            'sweep_bias': sweep_bias,
            'reversal_pips': reversal_pips,
            'swept_level_dist': swept_level_dist,
            'delta_bias': delta_bias,
            'delta_strength': delta_strength,
            'has_bos': has_bos,
            'bos_type': bos_type,
            'stoch_rsi_k': stoch_rsi_k,
            'supertrend_dir_h1': supertrend_dir_h1,
            'htf_ok': htf_ok,
            'smc_bias': smc_bias,
            'pd_zone': pd_zone,
            'vol_surge': vol_surge,
            'of_imbalance': of_imbalance,
            'atr_pips': atr_pips,
        }
    }


# ════════════════════════════════════════════════════════════════
# LIQ SWEEP FEATURE EXTRACTION (from DB rows with JOINed features)
# ════════════════════════════════════════════════════════════════

def extract_liq_sweep_features_from_db(row: dict) -> dict:
    """
    Extract liquidity sweep-specific features from a backtest_trades DB row
    (with LEFT JOIN on backtest_liq_sweep_features).

    Falls back to computed defaults for trades without liq_sweep feature rows.
    """
    if row.get('sweep_bias') is not None and row.get('sweep_bias') != '':
        lf = {
            'sweep_bias': str(row.get('sweep_bias', 'NONE')),
            'reversal_pips': float(row.get('reversal_pips', 0) or 0),
            'swept_level_dist': float(row.get('swept_level_dist', 0) or 0),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'delta_strength': str(row.get('delta_strength', 'NONE')),
            'has_bos': int(row.get('has_bos', 0) or 0),
            'bos_type': str(row.get('bos_type', 'NONE')),
            'stoch_rsi_k': float(row.get('stoch_rsi_k', 50) or 50),
            'supertrend_dir_h1': int(row.get('supertrend_dir_h1', 0) or 0),
            'htf_ok': int(row.get('htf_ok', 1) or 1),
            'smc_bias': str(row.get('smc_bias', 'NEUTRAL')),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'vol_surge': int(row.get('vol_surge', 0) or 0),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'atr_pips': float(row.get('atr_pips', 0) or 0),
        }
    else:
        lf = {
            'sweep_bias': str(row.get('direction', 'BUY') if row.get('direction') == 'BUY' else 'SELL').upper(),
            'reversal_pips': float(row.get('tp_pips', 0) or 0),
            'swept_level_dist': float(row.get('sl_pips', 0) or 0),
            'delta_bias': str(row.get('delta_bias', 'NEUTRAL')),
            'delta_strength': str(row.get('rd_bias', 'NONE')),
            'has_bos': 0,
            'bos_type': 'NONE',
            'stoch_rsi_k': 50.0,
            'supertrend_dir_h1': 0,
            'htf_ok': int(1 if row.get('htf_approved') else 0),
            'smc_bias': str(row.get('smc_bias', 'NEUTRAL')),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'vol_surge': int(row.get('vol_surge_detected', 0) or 0),
            'of_imbalance': float(row.get('of_imbalance', 0) or 0),
            'atr_pips': float(row.get('atr', 0) or 0),
        }
    return lf


def build_liq_sweep_feature_vector(row: dict) -> list:
    """Build liquidity sweep model feature vector from a DB row (with JOINed liq_sweep features).

    Returns a list of numeric features for the Liq Sweep Layer 1 model.
    Uses real liq_sweep features when available, falls back to derived values.
    """
    lf = extract_liq_sweep_features_from_db(row)

    features = [
        # ── Liq Sweep-specific internal features (15) ──
        SWEEP_BIAS_MAP.get(str(lf.get('sweep_bias', 'NONE')), 0.0),
        lf.get('reversal_pips', 0) / 50.0,
        lf.get('swept_level_dist', 0) / 20.0,
        SMC_BIAS_MAP.get(str(lf.get('delta_bias', 'NEUTRAL')), 0.0),
        DELTA_STRENGTH_MAP.get(str(lf.get('delta_strength', 'NONE')), 0.0),
        float(lf.get('has_bos', 0)),
        BOS_TYPE_MAP.get(str(lf.get('bos_type', 'NONE')), 0.0),
        lf.get('stoch_rsi_k', 50) / 100.0,
        float(lf.get('supertrend_dir_h1', 0)) / 1.0,
        float(lf.get('htf_ok', 1)),
        SMC_BIAS_MAP.get(str(lf.get('smc_bias', 'NEUTRAL')), 0.0),
        PD_ZONE_MAP.get(str(lf.get('pd_zone', 'NEUTRAL')), 0.0),
        float(lf.get('vol_surge', 0)),
        lf.get('of_imbalance', 0) / 1.0,  # can be negative
        lf.get('atr_pips', 0) / 30.0,

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
