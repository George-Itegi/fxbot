# =============================================================
# strategies/optimal_trade_entry_fib.py  v1.0
# Strategy: ICT Optimal Trade Entry (OTE) Fibonacci
#
# After a strong trending move, price tends to retrace to the
# 61.8%-78.6% Fibonacci level — the "OTE zone" / "golden zone" —
# before continuing.  This is the institutional re-entry sweet spot
# where smart money steps back in with full position size.
#
# Entry: Price inside the OTE zone + delta confirmation +
#        H4 supertrend alignment + discretionary confluence.
#
# Group: TREND_FOLLOWING
# Best sessions: London Open, NY-London Overlap
# Best states: TRENDING_STRONG, BREAKOUT_ACCEPTED
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "OPTIMAL_TRADE_ENTRY_FIB"
MIN_SCORE     = 70
VERSION       = "1.0"


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _check_displacement(df_m15: pd.DataFrame, direction: str,
                        pip_size: float) -> dict:
    """
    Check if the latest M15 candle shows displacement (a strong body
    > 5 pips in the trade direction).  Displacement inside the OTE
    zone is a high-confidence signal that institutions are re-entering.

    Returns dict:
        has_displacement: bool
        displacement_pips: float  (positive value in pips)
    """
    result = {'has_displacement': False, 'displacement_pips': 0.0}

    if df_m15 is None or len(df_m15) < 1:
        return result

    candle = df_m15.iloc[-1]
    body = float(candle['close']) - float(candle['open'])
    body_pips = abs(body) / pip_size

    if body_pips < 5.0:
        return result

    if direction == 'BUY' and body > 0:
        result['has_displacement'] = True
        result['displacement_pips'] = body_pips
    elif direction == 'SELL' and body < 0:
        result['has_displacement'] = True
        result['displacement_pips'] = body_pips

    return result


def evaluate(symbol: str,
             df_m1: pd.DataFrame = None,
             df_m5: pd.DataFrame = None,
             df_m15: pd.DataFrame = None,
             df_h1: pd.DataFrame = None,
             smc_report: dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None,
             relaxed: bool = False) -> dict | None:
    """
    Fires when price retraces into the ICT Optimal Trade Entry
    (OTE / golden zone: 61.8%-78.6% Fibonacci retracement) with
    delta, trend, and structural confirmation.
    """
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_h1) < 50 or len(df_h4) < 20:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 3.0:
        return None

    # ── Build Fibonacci report ──
    from backtest.fib_builder import build_fib_report, check_fib_confluence
    fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4,
                                  current_price=close_price)

    fib_direction = fib_report.get('direction', 'NEUTRAL')
    in_gz = fib_report.get('in_golden_zone', False)
    fib_bias = fib_report.get('fib_bias', 'NEUTRAL')
    golden_zone = fib_report.get('golden_zone', {})
    retracement_levels = fib_report.get('retracement_levels', {})
    extension_levels = fib_report.get('extension_levels', {})
    swing_high_price = fib_report.get('swing_high', {}).get('price', 0)
    swing_low_price = fib_report.get('swing_low', {}).get('price', 0)
    confluence_score = fib_report.get('confluence_score', 0)

    if fib_direction == 'NEUTRAL' or not retracement_levels:
        return None

    # ── Get SMC data ──
    structure = (smc_report or {}).get('structure', {})
    trend     = structure.get('trend', 'RANGING')
    smc_bias  = (smc_report or {}).get('smc_bias', 'NEUTRAL')
    htf_ok    = (smc_report or {}).get('htf_alignment', {}).get('approved', True)
    pd_zone   = (smc_report or {}).get('premium_discount', {}).get('zone', '')

    # Delta data (MANDATORY)
    rolling_delta = market_report.get('rolling_delta', {})
    delta_bias    = rolling_delta.get('bias', 'NEUTRAL')
    delta_strength = rolling_delta.get('strength', 'WEAK')

    # Order flow data
    of_imb      = market_report.get('order_flow_imbalance', {})
    of_dir      = of_imb.get('direction', 'NEUTRAL')
    of_strength = of_imb.get('strength', 'NONE')
    of_imb_value = of_imb.get('imbalance', 0)

    # Volume data
    volume_surge     = market_report.get('volume_surge', {})
    vol_surge_active = volume_surge.get('surge_detected', False)
    surge_ratio      = volume_surge.get('surge_ratio', 1.0)

    # Supertrend & StochRSI
    h1  = df_h1.iloc[-1]
    m15 = df_m15.iloc[-1]
    h4  = df_h4.iloc[-1]
    supertrend_dir_h1 = int(h1.get('supertrend_dir', 0))
    supertrend_dir_h4 = int(h4.get('supertrend_dir', 0))
    stoch_k = float(m15.get('stoch_rsi_k', 50))

    # Pre-compute displacement
    displacement_buy = _check_displacement(df_m15, 'BUY', pip_size)
    displacement_sell = _check_displacement(df_m15, 'SELL', pip_size)

    # BOS from SMC
    bos = structure.get('bos')
    has_bull_bos = bos and 'BULLISH' in bos.get('type', '')
    has_bear_bos = bos and 'BEARISH' in bos.get('type', '')

    score      = 0
    confluence = []

    # ═══════════════════════════════════════════════════════════
    # BULLISH (BUY) — fib direction is 'UP'
    # ═══════════════════════════════════════════════════════════
    if fib_direction == 'UP':
        # MANDATORY: Price must be inside the golden zone (OTE)
        if not in_gz:
            return None
        score += 25
        confluence.append("PRICE_IN_OTE_ZONE")

        # MANDATORY: Delta must be bullish (institutions buying)
        if delta_bias != 'BULLISH':
            return None
        score += 15
        confluence.append("DELTA_BULL_MANDATORY")

        # H4 supertrend must be bullish — confirms trend is intact
        if supertrend_dir_h4 == 1:
            score += 12
            confluence.append("H4_SUPERTREND_BULL")

        # H1 supertrend confirmation
        if supertrend_dir_h1 == 1:
            score += 8
            confluence.append("H1_SUPERTREND_BULL")

        # StochRSI oversold — buyers stepping in at the zone
        if stoch_k < 30:
            score += 10
            confluence.append("STOCHRSI_OVERSOLD")

        # Order flow imbalance confirming buyers
        if of_dir in ('BUY', 'BULLISH') or of_imb_value > 0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 12
                confluence.append("OF_BULL_STRONG")
            else:
                score += 6
                confluence.append("OF_BULL_CONFIRMS")

        # Volume surge
        if vol_surge_active:
            score += 8
            confluence.append("VOLUME_SURGE")

        # Delta strength
        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5
            confluence.append("DELTA_STRONG")

        # HTF alignment
        if htf_ok and smc_bias == 'BULLISH':
            score += 8
            confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 3
            confluence.append("HTF_APPROVED")

        # M15 displacement candle in the zone
        if displacement_buy['has_displacement']:
            score += 10
            confluence.append(
                f"M15_DISPLACEMENT_IN_ZONE_{displacement_buy['displacement_pips']:.0f}p")

        # BOS confirmation
        if has_bull_bos:
            score += 8
            confluence.append("BOS_BULL_CONFIRM")

        # PD zone discount bonus
        if 'DISCOUNT' in pd_zone:
            score += 5
            confluence.append("PD_DISCOUNT_BONUS")

        # PD zone premium penalty
        if 'PREMIUM' in pd_zone:
            score -= 12
            confluence.append("PD_PREMIUM_PENALTY")

        # ── Minimum confluence ──
        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            # ── SL / TP calculation ──
            gz_low  = golden_zone.get('low', 0)   # 78.6% level for UP
            gz_high = golden_zone.get('high', 0)   # 61.8% level for UP
            entry_price = close_price

            # SL: below the 78.6% level (deeper boundary of OTE zone),
            #     with ATR * 0.5 buffer — whichever is wider
            sl_below_zone = gz_low - atr_pips * 0.5 * pip_size
            sl_atr = entry_price - atr_pips * 0.5 * pip_size
            sl_price = round(min(sl_below_zone, sl_atr), 5)

            # Ensure SL is below current price
            if sl_price >= entry_price:
                sl_price = round(entry_price - atr_pips * 0.5 * pip_size, 5)

            # TP1: 1.618 extension, or ATR * 2.5 from entry
            tp1_ext = extension_levels.get('1.618')
            if tp1_ext and tp1_ext > entry_price:
                tp1_price = round(tp1_ext, 5)
            else:
                tp1_price = round(entry_price + atr_pips * 2.5 * pip_size, 5)

            # TP2: 2.618 extension, or ATR * 4.0 from entry
            tp2_ext = extension_levels.get('2.618')
            if tp2_ext and tp2_ext > entry_price:
                tp2_price = round(tp2_ext, 5)
            else:
                tp2_price = round(entry_price + atr_pips * 4.0 * pip_size, 5)

            sl_pips  = round((entry_price - sl_price) / pip_size, 1)
            tp1_pips = round((tp1_price - entry_price) / pip_size, 1)
            tp2_pips = round((tp2_price - entry_price) / pip_size, 1)

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
                "ote_zone":    f"{gz_low:.5f}—{gz_high:.5f}",
                "_ote_fib_features": {
                    'fib_direction': fib_direction,
                    'ote_zone_low': round(gz_low, 5),
                    'ote_zone_high': round(gz_high, 5),
                    'in_gz': 1,
                    'fib_bias': fib_bias,
                    'fib_confluence_score': confluence_score,
                    'delta_bias': delta_bias,
                    'delta_strength': delta_strength,
                    'supertrend_dir_h1': supertrend_dir_h1,
                    'supertrend_dir_h4': supertrend_dir_h4,
                    'stoch_rsi_k': round(stoch_k, 1),
                    'of_imbalance': of_imb_value,
                    'of_strength': of_strength,
                    'of_direction': of_dir,
                    'vol_surge': 1 if vol_surge_active else 0,
                    'htf_ok': 1 if htf_ok else 0,
                    'smc_bias': smc_bias,
                    'pd_zone': pd_zone,
                    'has_bull_bos': 1 if has_bull_bos else 0,
                    'displacement_pips': displacement_buy['displacement_pips'],
                    'has_displacement': 1 if displacement_buy['has_displacement'] else 0,
                    'swing_high': round(swing_high_price, 5),
                    'swing_low': round(swing_low_price, 5),
                    'atr_pips': round(atr_pips, 1),
                },
            }

    # ═══════════════════════════════════════════════════════════
    # BEARISH (SELL) — fib direction is 'DOWN'
    # ═══════════════════════════════════════════════════════════
    score      = 0
    confluence = []

    if fib_direction == 'DOWN':
        # MANDATORY: Price must be inside the golden zone (OTE)
        if not in_gz:
            return None
        score += 25
        confluence.append("PRICE_IN_OTE_ZONE")

        # MANDATORY: Delta must be bearish (institutions selling)
        if delta_bias != 'BEARISH':
            return None
        score += 15
        confluence.append("DELTA_BEAR_MANDATORY")

        # H4 supertrend must be bearish — confirms trend is intact
        if supertrend_dir_h4 == -1:
            score += 12
            confluence.append("H4_SUPERTREND_BEAR")

        # H1 supertrend confirmation
        if supertrend_dir_h1 == -1:
            score += 8
            confluence.append("H1_SUPERTREND_BEAR")

        # StochRSI overbought — sellers stepping in at the zone
        if stoch_k > 70:
            score += 10
            confluence.append("STOCHRSI_OVERBOUGHT")

        # Order flow imbalance confirming sellers
        if of_dir in ('SELL', 'BEARISH') or of_imb_value < -0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 12
                confluence.append("OF_BEAR_STRONG")
            else:
                score += 6
                confluence.append("OF_BEAR_CONFIRMS")

        # Volume surge
        if vol_surge_active:
            score += 8
            confluence.append("VOLUME_SURGE")

        # Delta strength
        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5
            confluence.append("DELTA_STRONG")

        # HTF alignment
        if htf_ok and smc_bias == 'BEARISH':
            score += 8
            confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 3
            confluence.append("HTF_APPROVED")

        # M15 displacement candle in the zone
        if displacement_sell['has_displacement']:
            score += 10
            confluence.append(
                f"M15_DISPLACEMENT_IN_ZONE_{displacement_sell['displacement_pips']:.0f}p")

        # BOS confirmation
        if has_bear_bos:
            score += 8
            confluence.append("BOS_BEAR_CONFIRM")

        # PD zone premium bonus
        if 'PREMIUM' in pd_zone:
            score += 5
            confluence.append("PD_PREMIUM_BONUS")

        # PD zone discount penalty
        if 'DISCOUNT' in pd_zone:
            score -= 12
            confluence.append("PD_DISCOUNT_PENALTY")

        # ── Minimum confluence ──
        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            # ── SL / TP calculation ──
            gz_low  = golden_zone.get('low', 0)   # 61.8% level for DOWN
            gz_high = golden_zone.get('high', 0)   # 78.6% level for DOWN
            entry_price = close_price

            # SL: above the 78.6% level (deeper boundary of OTE zone),
            #     with ATR * 0.5 buffer — whichever is wider
            sl_above_zone = gz_high + atr_pips * 0.5 * pip_size
            sl_atr = entry_price + atr_pips * 0.5 * pip_size
            sl_price = round(max(sl_above_zone, sl_atr), 5)

            # Ensure SL is above current price
            if sl_price <= entry_price:
                sl_price = round(entry_price + atr_pips * 0.5 * pip_size, 5)

            # TP1: 1.618 extension (below for DOWN), or ATR * 2.5 from entry
            tp1_ext = extension_levels.get('1.618')
            if tp1_ext and tp1_ext < entry_price:
                tp1_price = round(tp1_ext, 5)
            else:
                tp1_price = round(entry_price - atr_pips * 2.5 * pip_size, 5)

            # TP2: 2.618 extension, or ATR * 4.0 from entry
            tp2_ext = extension_levels.get('2.618')
            if tp2_ext and tp2_ext < entry_price:
                tp2_price = round(tp2_ext, 5)
            else:
                tp2_price = round(entry_price - atr_pips * 4.0 * pip_size, 5)

            sl_pips  = round((sl_price - entry_price) / pip_size, 1)
            tp1_pips = round((entry_price - tp1_price) / pip_size, 1)
            tp2_pips = round((entry_price - tp2_price) / pip_size, 1)

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
                "ote_zone":    f"{gz_low:.5f}—{gz_high:.5f}",
                "_ote_fib_features": {
                    'fib_direction': fib_direction,
                    'ote_zone_low': round(gz_low, 5),
                    'ote_zone_high': round(gz_high, 5),
                    'in_gz': 1,
                    'fib_bias': fib_bias,
                    'fib_confluence_score': confluence_score,
                    'delta_bias': delta_bias,
                    'delta_strength': delta_strength,
                    'supertrend_dir_h1': supertrend_dir_h1,
                    'supertrend_dir_h4': supertrend_dir_h4,
                    'stoch_rsi_k': round(stoch_k, 1),
                    'of_imbalance': of_imb_value,
                    'of_strength': of_strength,
                    'of_direction': of_dir,
                    'vol_surge': 1 if vol_surge_active else 0,
                    'htf_ok': 1 if htf_ok else 0,
                    'smc_bias': smc_bias,
                    'pd_zone': pd_zone,
                    'has_bear_bos': 1 if has_bear_bos else 0,
                    'displacement_pips': displacement_sell['displacement_pips'],
                    'has_displacement': 1 if displacement_sell['has_displacement'] else 0,
                    'swing_high': round(swing_high_price, 5),
                    'swing_low': round(swing_low_price, 5),
                    'atr_pips': round(atr_pips, 1),
                },
            }

    return None
