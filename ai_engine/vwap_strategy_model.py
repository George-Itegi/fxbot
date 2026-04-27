# =============================================================
# ai_engine/vwap_strategy_model.py  v1.0 — Layer 1 VWAP Model
#
# PURPOSE: Specialized Layer 1 model for VWAP_MEAN_REVERSION strategy.
# Replaces the hard-coded gates in strategies/vwap_mean_reversion.py
# with a learned model that knows which VWAP signals actually work.
#
# VWAP-SPECIFIC FEATURES (added on top of the base 63):
#   - pip_from_vwap (already in base, but heavily weighted for VWAP)
#   - vwap_distance_category (CLOSE/NORMAL/FAR)
#   - stoch_rsi_zone (OVERSOLD/NEUTRAL/OVERBOUGHT)
#   - atr_category (LOW/NORMAL/HIGH)
#   - fear_greed_category (EXTREME_FEAR/FEAR/NEUTRAL/GREED/EXTREME_GREED)
#
# WHY VWAP FIRST:
#   - 0% win rate — can't do worse than 0%
#   - Current gates aren't helping (blocking good trades or letting bad ones through)
#   - ML model will learn from VWAP's own trade history
#   - Zero downside risk: if model fails, keep gates
#
# TRAINING:
#   python -m ai_engine.train_strategy_model --strategy VWAP_MEAN_REVERSION
#
# The base framework (strategy_model.py) handles the actual training.
# This module adds VWAP-specific feature engineering and the
# no-gate signal generator for data collection.
# =============================================================

import numpy as np
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "VWAP_MEAN_REVERSION"
STRATEGY_KEY = "vwap"


# ════════════════════════════════════════════════════════════════
# VWAP NO-GATE SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════

def evaluate_no_gates(symbol, df_m1=None, df_m5=None, df_m15=None, df_h1=None,
                      market_report=None, smc_report=None, external_data=None,
                      master_report=None, df_h4=None):
    """
    VWAP evaluation WITHOUT hard-coded gates.

    This generates ALL potential VWAP signals (buy/sell setups)
    regardless of the current gate conditions. The Layer 1 model
    will then decide which ones to PASS.

    Removed gates vs original vwap_mean_reversion.py v1.1:
      - ATR filter (atr_pips < 2.0) → REMOVED
      - ADX extreme trending (adx > 45) → REMOVED
      - VIX > 25 → REMOVED
      - Master bias filter → REMOVED
      - Premium/Discount filter → REMOVED
      - Confluence minimum (5 items) → REMOVED
      - Minimum score (60) → REMOVED
      - R:R gate (tp1 >= sl * 1.2) → KEPT (prevents nonsensical trades)

    Instead, all these conditions become SCORING features that the
    model can learn to weight appropriately. A signal is generated
    if it has AT LEAST ONE valid entry condition (price vs VWAP).

    Returns: signal dict (same format as original) or None
    """
    if df_m15 is None or df_h1 is None or market_report is None:
        return None

    m15 = df_m15.iloc[-1]
    h1 = df_h1.iloc[-1]
    close_price = float(m15['close'])

    from core.pip_utils import get_pip_size as _gps
    pip_size = _gps(symbol, close_price)

    atr_pips = float(m15['atr']) / pip_size
    adx = float(m15.get('adx', 30))

    # VWAP data
    vwap_data = market_report.get('vwap', {})
    vwap = float(vwap_data.get('vwap', 0))
    vwap_pos = vwap_data.get('position', '')
    pip_from_v = float(vwap_data.get('pip_from_vwap', 0))

    if vwap == 0:
        return None

    # Profile data
    prof = market_report.get('profile', {})
    poc = float(prof.get('poc', 0))
    vah = float(prof.get('vah', 0))
    val = float(prof.get('val', 0))
    va_pos = prof.get('price_position', '')

    # SMC / HTF
    pd_zone = ''
    htf_ok = True
    if smc_report:
        pd_zone = smc_report.get('premium_discount', {}).get('zone', '')
        htf_ok = smc_report.get('htf_alignment', {}).get('approved', True)

    # External data
    vix = float(external_data.get('vix', 20)) if external_data else 20.0
    fg_score = float(external_data.get('fear_greed', 50)) if external_data else 50.0
    master_bias = master_report.get('combined_bias', '') if master_report else ''

    MIN_RR = 1.2

    # ── BUY SETUP (no gates, just scoring) ──
    if 'BELOW' in vwap_pos or va_pos == 'BELOW_VAL':
        score = 0
        confluence = []

        # Entry condition (always present for VWAP mean reversion BUY)
        score += 20; confluence.append("PRICE_BELOW_VWAP")

        # Scoring features (no hard blocks, just points)
        if 5 <= abs(pip_from_v) <= 30:
            score += 15; confluence.append(f"VWAP_DIST_{abs(pip_from_v):.0f}p")

        if poc > close_price:
            poc_dist = (poc - close_price) / pip_size
            if poc_dist <= 40:
                score += 15; confluence.append(f"POC_ABOVE_{poc_dist:.0f}p")

        if int(h1.get('supertrend_dir', 0)) == 1:
            score += 15; confluence.append("SUPERTREND_BULL")

        stoch_k = float(m15.get('stoch_rsi_k', 50))
        stoch_d = float(m15.get('stoch_rsi_d', 50))
        if stoch_k < 30 and stoch_k > stoch_d:
            score += 20; confluence.append("STOCHRSI_CROSS_UP")
        elif stoch_k < 40:
            score += 10; confluence.append("STOCHRSI_OVERSOLD")

        if fg_score <= 30:
            score += 10; confluence.append(f"FEAR_{fg_score:.0f}")

        if 'BAND' in vwap_pos:
            score += 10; confluence.append("AT_VWAP_BAND")

        if not htf_ok:
            score -= 15; confluence.append("HTF_REJECTED")

        # Gate conditions as NEGATIVE scoring (not hard blocks):
        # Low volatility — the model can learn if this matters
        if atr_pips < 2.0:
            score -= 10; confluence.append("LOW_VOL")
        # Extreme trending
        if adx > 45:
            score -= 15; confluence.append("EXTREME_TREND")
        # VIX high
        if vix > 25:
            score -= 10; confluence.append("HIGH_VIX")
        # Master bias against
        if master_bias == 'BEARISH':
            score -= 10; confluence.append("MASTER_BEARISH")
        # Premium zone
        if 'PREMIUM' in pd_zone and 'DISCOUNT' not in pd_zone:
            score -= 10; confluence.append("PREMIUM_ZONE")

        # Keep minimum R:R check (prevents garbage trades)
        sl_price = round(close_price - atr_pips * 1.2 * pip_size, 5)
        tp1_price = round(vwap, 5)
        tp2_price = round(poc if poc > close_price else vah, 5)
        sl_pips = round((close_price - sl_price) / pip_size, 1)
        tp1_pips = round((tp1_price - close_price) / pip_size, 1)
        tp2_pips = round((tp2_price - close_price) / pip_size, 1)

        if sl_pips <= 0:
            return None

        # R:R check — kept as the only hard filter
        if tp1_pips < sl_pips * MIN_RR:
            return None

        # Add VWAP-specific features to the signal dict
        log.info(f"[{STRATEGY_NAME}:NOGATE] BUY {symbol} Score:{score} | {', '.join(confluence)}")
        return {
            "direction": "BUY", "entry_price": close_price,
            "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "strategy": STRATEGY_NAME, "version": "2.0-no-gate",
            "score": score, "confluence": confluence,
            # VWAP-specific features for the model
            "_vwap_features": {
                'atr_pips': atr_pips,
                'adx': adx,
                'vix': vix,
                'fg_score': fg_score,
                'pip_from_vwap': abs(pip_from_v),
                'vwap_pos': vwap_pos,
                'va_pos': va_pos,
                'pd_zone': pd_zone,
                'htf_ok': htf_ok,
                'master_bias': master_bias,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'poc_dist': (poc - close_price) / pip_size if poc > 0 else 0,
                'supertrend_dir': int(h1.get('supertrend_dir', 0)),
                'poc_above': 1 if poc > close_price else 0,
                'vah_above': 1 if vah > close_price else 0,
            }
        }

    # ── SELL SETUP (mirror) ──
    if 'ABOVE' in vwap_pos or va_pos == 'ABOVE_VAH':
        score = 0
        confluence = []

        score += 20; confluence.append("PRICE_ABOVE_VWAP")

        if 5 <= abs(pip_from_v) <= 30:
            score += 15; confluence.append(f"VWAP_DIST_{abs(pip_from_v):.0f}p")

        if poc < close_price:
            poc_dist = (close_price - poc) / pip_size
            if poc_dist <= 40:
                score += 15; confluence.append(f"POC_BELOW_{poc_dist:.0f}p")

        if int(h1.get('supertrend_dir', 0)) == -1:
            score += 15; confluence.append("SUPERTREND_BEAR")

        stoch_k = float(m15.get('stoch_rsi_k', 50))
        stoch_d = float(m15.get('stoch_rsi_d', 50))
        if stoch_k > 70 and stoch_k < stoch_d:
            score += 20; confluence.append("STOCHRSI_CROSS_DOWN")
        elif stoch_k > 60:
            score += 10; confluence.append("STOCHRSI_OVERBOUGHT")

        if fg_score >= 70:
            score += 10; confluence.append(f"GREED_{fg_score:.0f}")

        if 'BAND' in vwap_pos:
            score += 10; confluence.append("AT_VWAP_BAND")

        if not htf_ok:
            score -= 15; confluence.append("HTF_REJECTED")

        # Gate conditions as negative scoring
        if atr_pips < 2.0:
            score -= 10; confluence.append("LOW_VOL")
        if adx > 45:
            score -= 15; confluence.append("EXTREME_TREND")
        if vix > 25:
            score -= 10; confluence.append("HIGH_VIX")
        if master_bias == 'BULLISH':
            score -= 10; confluence.append("MASTER_BULLISH")
        if 'DISCOUNT' in pd_zone and 'PREMIUM' not in pd_zone:
            score -= 10; confluence.append("DISCOUNT_ZONE")

        sl_price = round(close_price + atr_pips * 1.2 * pip_size, 5)
        tp1_price = round(vwap, 5)
        tp2_price = round(poc if poc < close_price else val, 5)
        sl_pips = round((sl_price - close_price) / pip_size, 1)
        tp1_pips = round((close_price - tp1_price) / pip_size, 1)
        tp2_pips = round((close_price - tp2_price) / pip_size, 1)

        if sl_pips <= 0:
            return None

        if tp1_pips < sl_pips * MIN_RR:
            return None

        log.info(f"[{STRATEGY_NAME}:NOGATE] SELL {symbol} Score:{score} | {', '.join(confluence)}")
        return {
            "direction": "SELL", "entry_price": close_price,
            "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "strategy": STRATEGY_NAME, "version": "2.0-no-gate",
            "score": score, "confluence": confluence,
            "_vwap_features": {
                'atr_pips': atr_pips,
                'adx': adx,
                'vix': vix,
                'fg_score': fg_score,
                'pip_from_vwap': abs(pip_from_v),
                'vwap_pos': vwap_pos,
                'va_pos': va_pos,
                'pd_zone': pd_zone,
                'htf_ok': htf_ok,
                'master_bias': master_bias,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'poc_dist': (close_price - poc) / pip_size if poc > 0 else 0,
                'supertrend_dir': int(h1.get('supertrend_dir', 0)),
                'poc_below': 1 if poc < close_price else 0,
                'val_below': 1 if val < close_price else 0,
            }
        }

    return None


# ════════════════════════════════════════════════════════════════
# VWAP FEATURE EXTRACTION (from no-gate signal data)
# ════════════════════════════════════════════════════════════════

def extract_vwap_features_from_db(row: dict) -> dict:
    """
    Extract VWAP-specific features from a backtest_trades DB row.

    Joins backtest_trades with backtest_vwap_features to get real VWAP
    internal features (ADX, VIX, StochRSI, etc.). Falls back to
    computed defaults for trades without VWAP feature rows (older data).

    Returns a dict of VWAP-specific features for model training.
    """
    # If the row has VWAP features (from JOIN), use them directly
    if row.get('atr_pips') is not None and row.get('atr_pips') != 0:
        vwap_features = {
            'atr_pips': float(row.get('atr_pips', 0)),
            'adx': float(row.get('adx', 50) or 50),
            'vix': float(row.get('vix', 20) or 20),
            'fg_score': float(row.get('fg_score', 50) or 50),
            'pip_from_vwap': abs(float(row.get('pip_from_vwap', 0) or 0)),
            'vwap_pos': str(row.get('vwap_pos', 'neutral')),
            'va_pos': str(row.get('va_pos', 'neutral')),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'htf_ok': bool(row.get('htf_ok', 0)),
            'master_bias': str(row.get('master_bias', '')),
            'stoch_k': float(row.get('stoch_k', 50) or 50),
            'stoch_d': float(row.get('stoch_d', 50) or 50),
            'poc_dist': abs(float(row.get('pip_to_poc', 0) or 0)),
            'supertrend_dir': int(row.get('supertrend_dir', 0) or 0),
            'poc_above': int(row.get('poc_above', 0) or 0),
            'val_below': int(row.get('val_below', 0) or 0),
        }
    else:
        # Fallback for older trades without backtest_vwap_features rows
        vwap_features = {
            'pip_from_vwap': float(row.get('pip_from_vwap', 0) or 0),
            'pip_to_poc': float(row.get('pip_to_poc', 50) or 50),
            'atr': float(row.get('atr', 0) or 0),
            'price_position': str(row.get('price_position', 'INSIDE_VA')),
            'pd_zone': str(row.get('pd_zone', 'NEUTRAL')),
            'htf_approved': 1 if row.get('htf_approved') else 0,
            'combined_bias': str(row.get('combined_bias', 'NEUTRAL')),
            'score': float(row.get('score', 0) or 0),
            'sl_pips': float(row.get('sl_pips', 10) or 10),
            'tp_pips': float(row.get('tp_pips', 15) or 15),
        }
    return vwap_features


def build_vwap_feature_vector(row: dict) -> list:
    """Build VWAP model feature vector from a DB row (with JOINed VWAP features).

    Returns a list of numeric features for the VWAP Layer 1 model.
    Uses real VWAP features when available, falls back to derived values.
    """
    vf = extract_vwap_features_from_db(row)

    # Normalize values
    atr_val = vf.get('atr_pips', vf.get('atr', 0)) or 0
    features = [
        # ── VWAP-specific internal features (16) ──
        atr_val / max(atr_val, 1),  # self-normalized
        vf.get('adx', 50) / 100.0,
        vf.get('vix', 20) / 50.0,
        vf.get('fg_score', 50) / 100.0,
        # Encode vwap_pos
        1.0 if 'ABOVE' in str(vf.get('vwap_pos', '')) else (-1.0 if 'BELOW' in str(vf.get('vwap_pos', '')) else 0.0),
        # Encode va_pos
        1.0 if 'ABOVE' in str(vf.get('va_pos', '')) else (-1.0 if 'BELOW' in str(vf.get('va_pos', '')) else 0.0),
        # Encode pd_zone
        1.0 if 'PREMIUM' in str(vf.get('pd_zone', '')) else (-1.0 if 'DISCOUNT' in str(vf.get('pd_zone', '')) else 0.0),
        1.0 if vf.get('htf_ok') or vf.get('htf_approved') else 0.0,
        # Encode master_bias
        1.0 if str(vf.get('master_bias', '') or vf.get('combined_bias', '')) == 'BULLISH' else (-1.0 if str(vf.get('master_bias', '') or vf.get('combined_bias', '')) == 'BEARISH' else 0.0),
        vf.get('stoch_k', 50) / 100.0,
        vf.get('stoch_d', 50) / 100.0,
        abs(vf.get('poc_dist', vf.get('pip_to_poc', 0))) / 50.0,
        float(vf.get('supertrend_dir', 0)),
        float(vf.get('poc_above', 0)),
        float(vf.get('val_below', 0)),

        # ── General VWAP features from backtest_trades (4) ──
        abs(float(row.get('pip_from_vwap', 0) or 0)) / 50.0,
        _encode_price_position(str(row.get('price_position', 'INSIDE_VA'))),
        abs(float(row.get('pip_to_poc', 0) or 0)) / 50.0,
        float(row.get('va_width_pips', 20) or 20) / 50.0,

        # ── Cross-strategy confluence (5) from strategy score columns ──
        float(row.get('ss_smc_ob', 0) or 0) / 100.0,
        float(row.get('ss_ema_cross', 0) or 0) / 100.0,
        float(row.get('ss_breakout_momentum', 0) or 0) / 100.0,
        float(row.get('ss_fvg_reversion', 0) or 0) / 100.0,
        float(row.get('ss_trend_continuation', 0) or 0) / 100.0,
    ]
    return features


def _encode_price_position(pp: str) -> float:
    """Encode price_position into numeric value."""
    pp_map = {'ABOVE_VAH': 2.0, 'ABOVE_VA': 1.0, 'INSIDE_VA': 0.0,
              'BELOW_VA': -1.0, 'BELOW_VAL': -2.0}
    return pp_map.get(pp.upper(), 0.0)
