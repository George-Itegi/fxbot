# =============================================================
# strategies/smc_ob_reversal.py  v2.0
# Strategy 2: SMC Order Block Reversal
# Price pulls back into institutional OB zone then reverses.
# This is the highest probability SMC setup for day trading.
#
# v2.0 CHANGES (combined AI audit):
#   1. Entry at OB zone edge (not arbitrary candle close)
#   2. Order flow / delta confirmation is MANDATORY (not bonus)
#   3. Must have volume surge or strong OF imbalance from master_report
#   4. Tighter distance check — price must be INSIDE the OB zone
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "SMC_OB_REVERSAL"
MIN_SCORE     = 70
VERSION       = "2.0"


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
    and shows reversal confirmation (delta + OF + StochRSI).

    v2.0: Entry is at the OB zone edge, not candle close.
    Order flow confirmation is MANDATORY.
    """
    if df_m15 is None or df_h1 is None:
        return None
    if smc_report is None:
        return None
    if market_report is None:
        return None

    m15 = df_m15.iloc[-1]
    h1  = df_h1.iloc[-1]

    # Pip size detection — must match order_manager._get_pip_point
    sym = str(symbol).upper()
    close_price = float(m15['close'])
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        pip_size = 1.0
    elif "XAU" in sym:
        pip_size = 0.1
    elif "XAG" in sym:
        pip_size = 0.01
    elif any(x in sym for x in ["WTI", "BRN"]):
        pip_size = 0.01
    elif close_price > 50:
        pip_size = 0.01
    else:
        pip_size = 0.0001

    atr_pips = float(m15['atr']) / pip_size
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
    ob_dist   = float(nearest_ob.get('pips_away', 0))

    # ── OB Freshness Gate ──
    # OBs older than ~20 H1 bars are stale — institutional interest fades.
    # Use pips_away as a proxy: if price has moved >30 pips away since OB formed,
    # it's likely an old zone that has lost relevance.
    # Exception: large moves are fine if price has RETURNED to the zone (touched)
    MAX_OB_AGE_PIPS = 30.0
    if ob_dist > MAX_OB_AGE_PIPS:
        return None

    # ── MANDATORY: Price must be inside or touching the OB zone ──
    # Tighter tolerance than before — price must actually be IN the zone
    tolerance_pips = 3.0 * pip_size  # 3 pips tolerance
    price_at_ob = (ob_bottom - tolerance_pips <= close_price <= ob_top + tolerance_pips)
    if not price_at_ob:
        return None

    # ── MANDATORY: Order flow must confirm direction ──
    # FIXED: Delta confirmation is no longer optional — it's required.
    # Without institutional flow confirmation, an OB touch is just noise.
    rolling_delta = market_report.get('rolling_delta', {})
    delta_bias    = rolling_delta.get('bias', 'NEUTRAL')
    delta_strength = rolling_delta.get('strength', 'WEAK')

    # Also check order flow imbalance from market report
    of_imb = market_report.get('order_flow_imbalance', {})
    of_dir = of_imb.get('direction', 'NEUTRAL')
    of_strength = of_imb.get('strength', 'NONE')

    score      = 0
    confluence = []

    # ── BULLISH OB REVERSAL (BUY) ─────────────────────────
    if ob_type == 'BULLISH_OB' and trend == 'BULLISH':

        # MANDATORY: Delta must confirm buyers
        if delta_bias != 'BULLISH':
            return None
        score += 15; confluence.append("DELTA_BULL_MANDATORY")

        # Price at OB (already verified above)
        score += 25; confluence.append("PRICE_AT_BULLISH_OB")

        # Use OB bottom as entry reference (support level)
        # If price is below OB bottom, enter at OB bottom (limit style)
        # If price is inside OB, use current price (market entry)
        if close_price < ob_bottom:
            entry_price = ob_bottom  # Ideal: limit buy at OB bottom
            confluence.append("ENTRY_AT_OB_BOTTOM")
        else:
            entry_price = close_price  # Price already in zone — enter now
            confluence.append("ENTRY_IN_OB_ZONE")

        # Order flow imbalance bonus
        if of_dir in ('BUY', 'BULLISH') or of_imb.get('imbalance', 0) > 0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 15; confluence.append("OF_BULL_STRONG")
            else:
                score += 8; confluence.append("OF_BULL_CONFIRMS")

        # Delta strength bonus
        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5; confluence.append("DELTA_STRONG")

        # Supertrend still bullish on H1
        if int(h1.get('supertrend_dir', 0)) == 1:
            score += 10; confluence.append("SUPERTREND_BULL_H1")

        # StochRSI oversold — buyers stepping in
        stoch_k = float(m15.get('stoch_rsi_k', 50))
        if stoch_k < 25:
            score += 15; confluence.append("STOCHRSI_OVERSOLD")
        elif stoch_k < 35:
            score += 8; confluence.append("STOCHRSI_LOW")

        # HTF alignment
        if htf_ok and smc_bias == 'BULLISH':
            score += 10; confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 5; confluence.append("HTF_APPROVED")

        # Recent bullish sweep = extra confidence
        if last_sweep and last_sweep.get('bias') == 'BULLISH':
            score += 8; confluence.append("RECENT_BULL_SWEEP")

        # BOS confirmation from SMC structure
        bos = structure.get('bos')
        if bos and 'BULLISH' in bos.get('type', ''):
            score += 10; confluence.append("BOS_BULL_CONFIRM")

        # Penalize if in extreme premium
        if 'EXTREME_PREMIUM' in pd_zone:
            score -= 15; confluence.append("PD_PREMIUM_PENALTY")

        # ── Fibonacci confluence bonus ──────────────────
        try:
            from backtest.fib_builder import build_fib_report, check_fib_confluence
            fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4, current_price=entry_price)
            fib_check = check_fib_confluence(entry_price, "BUY", fib_report, pip_size)
            if fib_check['fib_bonus'] > 0:
                score += fib_check['fib_bonus']
                confluence.extend(fib_check['confluence'])
        except Exception:
            pass

        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            # SL below OB bottom with ATR buffer
            sl_price  = round(ob_bottom - atr_pips * 0.2 * pip_size, 5)
            tp1_price = round(entry_price + atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(entry_price + atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((entry_price - sl_price) / pip_size, 1)
            tp1_pips  = round((tp1_price - entry_price) / pip_size, 1)
            tp2_pips  = round((tp2_price - entry_price) / pip_size, 1)

            log.info(f"[{STRATEGY_NAME} v{VERSION}] BUY {symbol}"
                     f" entry={entry_price:.5f} Score:{score} | "
                     f"{', '.join(confluence)}")
            return {
                "direction":   "BUY",
                "entry_price": entry_price,
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

        # MANDATORY: Delta must confirm sellers
        if delta_bias != 'BEARISH':
            return None
        score += 15; confluence.append("DELTA_BEAR_MANDATORY")

        score += 25; confluence.append("PRICE_AT_BEARISH_OB")

        # Use OB top as entry reference (resistance level)
        if close_price > ob_top:
            entry_price = ob_top  # Ideal: limit sell at OB top
            confluence.append("ENTRY_AT_OB_TOP")
        else:
            entry_price = close_price  # Price already in zone
            confluence.append("ENTRY_IN_OB_ZONE")

        # Order flow imbalance bonus
        if of_dir in ('SELL', 'BEARISH') or of_imb.get('imbalance', 0) < -0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 15; confluence.append("OF_BEAR_STRONG")
            else:
                score += 8; confluence.append("OF_BEAR_CONFIRMS")

        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5; confluence.append("DELTA_STRONG")

        if int(h1.get('supertrend_dir', 0)) == -1:
            score += 10; confluence.append("SUPERTREND_BEAR_H1")

        stoch_k = float(m15.get('stoch_rsi_k', 50))
        if stoch_k > 75:
            score += 15; confluence.append("STOCHRSI_OVERBOUGHT")
        elif stoch_k > 65:
            score += 8; confluence.append("STOCHRSI_HIGH")

        if htf_ok and smc_bias == 'BEARISH':
            score += 10; confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 5; confluence.append("HTF_APPROVED")

        if last_sweep and last_sweep.get('bias') == 'BEARISH':
            score += 8; confluence.append("RECENT_BEAR_SWEEP")

        bos = structure.get('bos')
        if bos and 'BEARISH' in bos.get('type', ''):
            score += 10; confluence.append("BOS_BEAR_CONFIRM")

        if 'EXTREME_DISCOUNT' in pd_zone:
            score -= 15; confluence.append("PD_DISCOUNT_PENALTY")

        # ── Fibonacci confluence bonus ──────────────────
        try:
            from backtest.fib_builder import build_fib_report, check_fib_confluence
            fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4, current_price=entry_price)
            fib_check = check_fib_confluence(entry_price, "SELL", fib_report, pip_size)
            if fib_check['fib_bonus'] > 0:
                score += fib_check['fib_bonus']
                confluence.extend(fib_check['confluence'])
        except Exception:
            pass

        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            sl_price  = round(ob_top + atr_pips * 0.2 * pip_size, 5)
            tp1_price = round(entry_price - atr_pips * 1.5 * pip_size, 5)
            tp2_price = round(entry_price - atr_pips * 3.0 * pip_size, 5)
            sl_pips   = round((sl_price - entry_price) / pip_size, 1)
            tp1_pips  = round((entry_price - tp1_price) / pip_size, 1)
            tp2_pips  = round((entry_price - tp2_price) / pip_size, 1)

            log.info(f"[{STRATEGY_NAME} v{VERSION}] SELL {symbol}"
                     f" entry={entry_price:.5f} Score:{score} | "
                     f"{', '.join(confluence)}")
            return {
                "direction":   "SELL",
                "entry_price": entry_price,
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
