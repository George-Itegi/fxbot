# =============================================================
# strategies/smc_ob_reversal.py
# Strategy 2: SMC Order Block Reversal
# Price pulls back into institutional OB zone then reverses.
# This is the highest probability SMC setup for day trading.
# Best state : TRENDING_STRONG with pullback
# Best session: LONDON_KILLZONE, NY_LONDON_OVERLAP
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "SMC_OB_REVERSAL"
MIN_SCORE     = 65
VERSION       = "1.0"


def evaluate(symbol: str,
             df_m1: pd.DataFrame,
             df_m5: pd.DataFrame,
             df_m15:  pd.DataFrame,
             df_h1:   pd.DataFrame,
             smc_report:   dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None) -> dict | None:
    """
    Fires when price returns to an unmitigated Order Block
    and shows reversal confirmation (StochRSI + delta + candle).
    """
    if df_m15 is None or df_h1 is None:
        return None
    if smc_report is None:
        return None

    m15 = df_m15.iloc[-1]
    h1  = df_h1.iloc[-1]

    close_price = float(m15['close'])
    pip_size    = 0.01 if close_price > 50 else 0.0001
    atr_pips    = float(m15['atr']) / pip_size

    if atr_pips < 3.0:
        return None

    # Get SMC data
    structure  = smc_report.get('structure', {})
    trend      = structure.get('trend', 'RANGING')
    nearest_ob = smc_report.get('nearest_ob')
    pd_zone    = smc_report.get('premium_discount', {}).get('zone', '')
    htf_ok     = smc_report.get('htf_alignment', {}).get('approved', True)
    smc_bias   = smc_report.get('smc_bias', 'NEUTRAL')
    last_sweep = smc_report.get('last_sweep')

    # Need a clear trend + active OB
    if trend == 'RANGING' or nearest_ob is None:
        return None
    if nearest_ob.get('mitigated', True):
        return None

    ob_type   = nearest_ob.get('type', '')
    ob_top    = float(nearest_ob.get('top', 0))
    ob_bottom = float(nearest_ob.get('bottom', 0))
    ob_mid    = float(nearest_ob.get('mid', 0))

    # Distance from price to OB — must be nearby
    dist_pips = abs(close_price - ob_mid) / pip_size
    if dist_pips > 30:
        return None  # OB too far away

    # Get delta from market report
    rolling_delta = market_report.get('rolling_delta', {}) if market_report else {}
    delta_bias    = rolling_delta.get('bias', 'NEUTRAL')

    score      = 0
    confluence = []

    # ── BULLISH OB REVERSAL (BUY) ─────────────────────────
    if ob_type == 'BULLISH_OB' and trend == 'BULLISH':

        # Price must be AT or inside the OB zone
        price_at_ob = (ob_bottom * 0.9998 <= close_price <= ob_top * 1.0005)
        if not price_at_ob:
            return None

        score += 25; confluence.append("PRICE_AT_BULLISH_OB")

        # Supertrend still bullish on H1
        if int(h1.get('supertrend_dir', 0)) == 1:
            score += 15; confluence.append("SUPERTREND_BULL_H1")

        # StochRSI oversold — buyers stepping in
        stoch_k = float(m15.get('stoch_rsi_k', 50))
        if stoch_k < 25:
            score += 20; confluence.append("STOCHRSI_OVERSOLD")
        elif stoch_k < 35:
            score += 10; confluence.append("STOCHRSI_LOW")

        # Delta confirming buyers
        if delta_bias == 'BULLISH':
            score += 15; confluence.append("DELTA_BULL_CONFIRM")

        # HTF alignment
        if htf_ok and smc_bias == 'BULLISH':
            score += 15; confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 8; confluence.append("HTF_APPROVED")

        # Recent bullish sweep = extra confidence
        if last_sweep and last_sweep.get('bias') == 'BULLISH':
            score += 10; confluence.append("RECENT_BULL_SWEEP")

        # Penalize if in extreme premium
        if 'EXTREME_PREMIUM' in pd_zone:
            score -= 15; confluence.append("PD_PREMIUM_PENALTY")

        if score >= MIN_SCORE:
            sl_price  = round(ob_bottom - atr_pips * 0.2 * pip_size, 5)
            tp1_price = round(close_price + atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(close_price + atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((close_price - sl_price) / pip_size, 1)
            tp1_pips  = round((tp1_price - close_price) / pip_size, 1)
            tp2_pips  = round((tp2_price - close_price) / pip_size, 1)
            log.info(f"[{STRATEGY_NAME}] BUY {symbol} @ OB"
                     f" Score:{score} | {', '.join(confluence)}")
            return {
                "direction":   "BUY",
                "entry_price": close_price,
                "sl_price":    sl_price,
                "tp1_price":   tp1_price,
                "tp2_price":   tp2_price,
                "sl_pips":     sl_pips,
                "tp1_pips":    tp1_pips,
                "tp2_pips":    tp2_pips,
                "strategy":    STRATEGY_NAME,
                "version":     VERSION,
                "score":       score,
                "confluence":  confluence,
                "ob_zone":     f"{ob_bottom}—{ob_top}",
            }

    # ── BEARISH OB REVERSAL (SELL) ────────────────────────
    score      = 0
    confluence = []

    if ob_type == 'BEARISH_OB' and trend == 'BEARISH':

        price_at_ob = (ob_bottom * 0.9995 <= close_price <= ob_top * 1.0002)
        if not price_at_ob:
            return None

        score += 25; confluence.append("PRICE_AT_BEARISH_OB")

        if int(h1.get('supertrend_dir', 0)) == -1:
            score += 15; confluence.append("SUPERTREND_BEAR_H1")

        stoch_k = float(m15.get('stoch_rsi_k', 50))
        if stoch_k > 75:
            score += 20; confluence.append("STOCHRSI_OVERBOUGHT")
        elif stoch_k > 65:
            score += 10; confluence.append("STOCHRSI_HIGH")

        if delta_bias == 'BEARISH':
            score += 15; confluence.append("DELTA_BEAR_CONFIRM")

        if htf_ok and smc_bias == 'BEARISH':
            score += 15; confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 8; confluence.append("HTF_APPROVED")

        if last_sweep and last_sweep.get('bias') == 'BEARISH':
            score += 10; confluence.append("RECENT_BEAR_SWEEP")

        if 'EXTREME_DISCOUNT' in pd_zone:
            score -= 15; confluence.append("PD_DISCOUNT_PENALTY")

        if score >= MIN_SCORE:
            sl_price  = round(ob_top + atr_pips * 0.2 * pip_size, 5)
            tp1_price = round(close_price - atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(close_price - atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((sl_price - close_price) / pip_size, 1)
            tp1_pips  = round((close_price - tp1_price) / pip_size, 1)
            tp2_pips  = round((close_price - tp2_price) / pip_size, 1)
            log.info(f"[{STRATEGY_NAME}] SELL {symbol} @ OB"
                     f" Score:{score} | {', '.join(confluence)}")
            return {
                "direction":   "SELL",
                "entry_price": close_price,
                "sl_price":    sl_price,
                "tp1_price":   tp1_price,
                "tp2_price":   tp2_price,
                "sl_pips":     sl_pips,
                "tp1_pips":    tp1_pips,
                "tp2_pips":    tp2_pips,
                "strategy":    STRATEGY_NAME,
                "version":     VERSION,
                "score":       score,
                "confluence":  confluence,
                "ob_zone":     f"{ob_bottom}—{ob_top}",
            }

    return None
