# =============================================================
# strategies/supply_demand_zone.py  v1.0
# Strategy: Supply/Demand Zone Entry
#
# Institutional S&D zones represent areas where banks placed
# large orders before a strong displacement. Unlike Order Blocks
# (3-candle micro patterns), S&D zones capture the broader
# consolidation before aggressive moves — the institutional intent.
#
# Entry: Price returns to a fresh H1 S&D zone with order flow
# confirmation (delta + optional OF imbalance). Zone freshness
# and displacement quality are the primary edge.
#
# Group: SMC_STRUCTURE
# Best sessions: London Open, London Session, NY-London Overlap
# Best states: TRENDING_STRONG, BREAKOUT_ACCEPTED, TRENDING_EXTENDED
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "SUPPLY_DEMAND_ZONE_ENTRY"
MIN_SCORE     = 70
VERSION       = "1.0"


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _find_demand_zone(df_h1: pd.DataFrame, pip_size: float) -> dict | None:
    """
    Identify the most recent demand zone (bullish) on H1.
    
    A demand zone is a consolidation area before an upward displacement.
    We look for:
      1. At least 3 candles of consolidation (small bodies, range-bound)
      2. Followed by a strong displacement (2+ large bullish candles)
      3. Price has since pulled back toward or into the zone
    
    Returns dict with zone_top, zone_bottom, displacement_pips, age_bars,
    or None if no valid zone found.
    """
    if df_h1 is None or len(df_h1) < 20:
        return None

    lookback = min(40, len(df_h1) - 1)
    
    for i in range(len(df_h1) - lookback, len(df_h1) - 3):
        # Check for consolidation: 3+ candles with small bodies
        consolidation_high = float(df_h1.iloc[i]['high'])
        consolidation_low  = float(df_h1.iloc[i]['low'])
        consolidation_bodies = 0
        
        for j in range(i, min(i + 4, len(df_h1))):
            candle = df_h1.iloc[j]
            body = abs(float(candle['close']) - float(candle['open']))
            range_val = float(candle['high']) - float(candle['low'])
            if range_val > 0 and body / range_val < 0.4:  # Small body = doji/consolidation
                consolidation_bodies += 1
            consolidation_high = max(consolidation_high, float(candle['high']))
            consolidation_low  = min(consolidation_low, float(candle['low']))
        
        if consolidation_bodies < 2:
            continue
        
        zone_range_pips = (consolidation_high - consolidation_low) / pip_size
        if zone_range_pips > 20 or zone_range_pips < 2:
            continue  # Zone too wide or too narrow
        
        # Check for displacement after consolidation
        disp_idx = i + 4 if i + 4 < len(df_h1) else len(df_h1) - 1
        if disp_idx >= len(df_h1) - 2:
            continue
        
        disp_candles = 0
        disp_pips = 0.0
        for j in range(disp_idx, min(disp_idx + 3, len(df_h1))):
            candle = df_h1.iloc[j]
            body = float(candle['close']) - float(candle['open'])
            if body > 0:
                disp_pips += body / pip_size
                if body / pip_size > 3:  # Strong bullish candle
                    disp_candles += 1
        
        if disp_candles < 1 or disp_pips < 8:
            continue  # No meaningful displacement
        
        # Zone freshness: count bars since zone formed
        age_bars = len(df_h1) - 1 - disp_idx
        if age_bars > 25:
            continue  # Zone is stale (institutional interest faded)
        
        return {
            'type': 'DEMAND',
            'zone_top': consolidation_high,
            'zone_bottom': consolidation_low,
            'zone_mid': (consolidation_high + consolidation_low) / 2,
            'displacement_pips': disp_pips,
            'age_bars': age_bars,
            'zone_range_pips': zone_range_pips,
        }
    
    return None


def _find_supply_zone(df_h1: pd.DataFrame, pip_size: float) -> dict | None:
    """
    Identify the most recent supply zone (bearish) on H1.
    Mirror of _find_demand_zone for bearish setups.
    """
    if df_h1 is None or len(df_h1) < 20:
        return None

    lookback = min(40, len(df_h1) - 1)
    
    for i in range(len(df_h1) - lookback, len(df_h1) - 3):
        consolidation_high = float(df_h1.iloc[i]['high'])
        consolidation_low  = float(df_h1.iloc[i]['low'])
        consolidation_bodies = 0
        
        for j in range(i, min(i + 4, len(df_h1))):
            candle = df_h1.iloc[j]
            body = abs(float(candle['close']) - float(candle['open']))
            range_val = float(candle['high']) - float(candle['low'])
            if range_val > 0 and body / range_val < 0.4:
                consolidation_bodies += 1
            consolidation_high = max(consolidation_high, float(candle['high']))
            consolidation_low  = min(consolidation_low, float(candle['low']))
        
        if consolidation_bodies < 2:
            continue
        
        zone_range_pips = (consolidation_high - consolidation_low) / pip_size
        if zone_range_pips > 20 or zone_range_pips < 2:
            continue
        
        disp_idx = i + 4 if i + 4 < len(df_h1) else len(df_h1) - 1
        if disp_idx >= len(df_h1) - 2:
            continue
        
        disp_candles = 0
        disp_pips = 0.0
        for j in range(disp_idx, min(disp_idx + 3, len(df_h1))):
            candle = df_h1.iloc[j]
            body = float(candle['open']) - float(candle['close'])
            if body > 0:
                disp_pips += body / pip_size
                if body / pip_size > 3:
                    disp_candles += 1
        
        if disp_candles < 1 or disp_pips < 8:
            continue
        
        age_bars = len(df_h1) - 1 - disp_idx
        if age_bars > 25:
            continue
        
        return {
            'type': 'SUPPLY',
            'zone_top': consolidation_high,
            'zone_bottom': consolidation_low,
            'zone_mid': (consolidation_high + consolidation_low) / 2,
            'displacement_pips': disp_pips,
            'age_bars': age_bars,
            'zone_range_pips': zone_range_pips,
        }
    
    return None


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
    Fires when price returns to a fresh supply/demand zone
    with institutional confirmation (delta + volume + structure).
    """
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_h1) < 20 or len(df_h4) < 20:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 3.0:
        return None

    # Get SMC data
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

    h1  = df_h1.iloc[-1]
    m15 = df_m15.iloc[-1]
    h4  = df_h4.iloc[-1]
    supertrend_dir_h1 = int(h1.get('supertrend_dir', 0))
    supertrend_dir_h4 = int(h4.get('supertrend_dir', 0))
    stoch_k = float(m15.get('stoch_rsi_k', 50))

    # ── Find S&D zones ──
    demand_zone = _find_demand_zone(df_h1, pip_size)
    supply_zone = _find_supply_zone(df_h1, pip_size)

    score      = 0
    confluence = []

    # ═══════════════════════════════════════════════════════════
    # BULLISH: Price returns to Demand Zone (BUY)
    # ═══════════════════════════════════════════════════════════
    if demand_zone is not None:
        zone_top    = demand_zone['zone_top']
        zone_bottom = demand_zone['zone_bottom']
        zone_range  = demand_zone['zone_range_pips']
        disp_pips   = demand_zone['displacement_pips']
        age_bars    = demand_zone['age_bars']

        # MANDATORY: Price must be inside or touching the demand zone
        tolerance = 3.0 * pip_size
        price_in_zone = (zone_bottom - tolerance <= close_price <= zone_top + tolerance)
        if not price_in_zone:
            demand_zone = None  # Clear — price not at zone
        else:
            score += 25
            confluence.append("PRICE_AT_DEMAND_ZONE")

    if demand_zone is not None:
        zone_top    = demand_zone['zone_top']
        zone_bottom = demand_zone['zone_bottom']
        disp_pips   = demand_zone['displacement_pips']
        age_bars    = demand_zone['age_bars']

        # MANDATORY: Delta must confirm buyers
        if delta_bias != 'BULLISH':
            return None
        score += 15
        confluence.append("DELTA_BULL_MANDATORY")

        # H4 structure alignment
        if supertrend_dir_h4 == 1:
            score += 15
            confluence.append("H4_SUPERTREND_BULL")
        elif trend == 'BULLISH':
            score += 10
            confluence.append("H4_STRUCTURE_BULL")

        # H1 supertrend confirmation
        if supertrend_dir_h1 == 1:
            score += 10
            confluence.append("H1_SUPERTREND_BULL")

        # Zone freshness bonus
        if age_bars <= 10:
            score += 10
            confluence.append("FRESH_ZONE")
        elif age_bars <= 15:
            score += 5
            confluence.append("RECENT_ZONE")

        # Displacement quality — stronger displacement = better zone
        if disp_pips > 20:
            score += 10
            confluence.append(f"STRONG_DISPLACEMENT_{disp_pips:.0f}p")
        elif disp_pips > 12:
            score += 5
            confluence.append(f"GOOD_DISPLACEMENT_{disp_pips:.0f}p")

        # Order flow imbalance bonus
        if of_dir in ('BUY', 'BULLISH') or of_imb_value > 0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 15
                confluence.append("OF_BULL_STRONG")
            else:
                score += 8
                confluence.append("OF_BULL_CONFIRMS")

        # Delta strength bonus
        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5
            confluence.append("DELTA_STRONG")

        # StochRSI oversold — buyers stepping in
        if stoch_k < 25:
            score += 10
            confluence.append("STOCHRSI_OVERSOLD")
        elif stoch_k < 35:
            score += 5
            confluence.append("STOCHRSI_LOW")

        # Volume surge bonus
        if vol_surge_active:
            score += 8
            confluence.append("VOLUME_SURGE")

        # HTF alignment
        if htf_ok and smc_bias == 'BULLISH':
            score += 8
            confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 3
            confluence.append("HTF_APPROVED")

        # BOS confirmation
        bos = structure.get('bos')
        has_bos = False
        if bos and 'BULLISH' in bos.get('type', ''):
            score += 10
            confluence.append("BOS_BULL_CONFIRM")
            has_bos = True

        # Premium zone penalty
        if 'EXTREME_PREMIUM' in pd_zone:
            score -= 15
            confluence.append("PD_PREMIUM_PENALTY")

        # Fibonacci confluence bonus
        try:
            from backtest.fib_builder import build_fib_report, check_fib_confluence
            fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4, current_price=close_price)
            fib_check = check_fib_confluence(close_price, "BUY", fib_report, pip_size)
            if fib_check['fib_bonus'] > 0:
                score += fib_check['fib_bonus']
                confluence.extend(fib_check['confluence'])
        except Exception:
            pass

        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            # Entry at zone bottom (support) or current price if inside zone
            entry_price = max(close_price, zone_bottom)
            sl_price  = round(zone_bottom - atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(entry_price + atr_pips * 2.0 * pip_size, 5)
            tp2_price = round(entry_price + atr_pips * 3.5 * pip_size, 5)
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
                "sd_zone":     f"{zone_bottom:.5f}—{zone_top:.5f}",
                "_sd_zone_features": {
                    'zone_type': 'DEMAND',
                    'zone_range_pips': demand_zone['zone_range_pips'],
                    'price_at_zone': 1,
                    'displacement_pips': disp_pips,
                    'age_bars': age_bars,
                    'trend': trend,
                    'delta_bias': delta_bias,
                    'delta_strength': delta_strength,
                    'of_imbalance': of_imb_value,
                    'of_strength': of_strength,
                    'stoch_rsi_k': stoch_k,
                    'supertrend_dir_h1': supertrend_dir_h1,
                    'supertrend_dir_h4': supertrend_dir_h4,
                    'htf_ok': 1 if htf_ok else 0,
                    'smc_bias': smc_bias,
                    'pd_zone': pd_zone,
                    'vol_surge': 1 if vol_surge_active else 0,
                    'has_bos': 1 if has_bos else 0,
                    'atr_pips': atr_pips,
                },
            }

    # ═══════════════════════════════════════════════════════════
    # BEARISH: Price returns to Supply Zone (SELL)
    # ═══════════════════════════════════════════════════════════
    score      = 0
    confluence = []

    if supply_zone is not None:
        zone_top    = supply_zone['zone_top']
        zone_bottom = supply_zone['zone_bottom']
        zone_range  = supply_zone['zone_range_pips']
        disp_pips   = supply_zone['displacement_pips']
        age_bars    = supply_zone['age_bars']

        tolerance = 3.0 * pip_size
        price_in_zone = (zone_bottom - tolerance <= close_price <= zone_top + tolerance)
        if not price_in_zone:
            supply_zone = None
        else:
            score += 25
            confluence.append("PRICE_AT_SUPPLY_ZONE")

    if supply_zone is not None:
        zone_top    = supply_zone['zone_top']
        zone_bottom = supply_zone['zone_bottom']
        disp_pips   = supply_zone['displacement_pips']
        age_bars    = supply_zone['age_bars']

        # MANDATORY: Delta must confirm sellers
        if delta_bias != 'BEARISH':
            return None
        score += 15
        confluence.append("DELTA_BEAR_MANDATORY")

        # H4 structure alignment
        if supertrend_dir_h4 == -1:
            score += 15
            confluence.append("H4_SUPERTREND_BEAR")
        elif trend == 'BEARISH':
            score += 10
            confluence.append("H4_STRUCTURE_BEAR")

        # H1 supertrend confirmation
        if supertrend_dir_h1 == -1:
            score += 10
            confluence.append("H1_SUPERTREND_BEAR")

        # Zone freshness bonus
        if age_bars <= 10:
            score += 10
            confluence.append("FRESH_ZONE")
        elif age_bars <= 15:
            score += 5
            confluence.append("RECENT_ZONE")

        # Displacement quality
        if disp_pips > 20:
            score += 10
            confluence.append(f"STRONG_DISPLACEMENT_{disp_pips:.0f}p")
        elif disp_pips > 12:
            score += 5
            confluence.append(f"GOOD_DISPLACEMENT_{disp_pips:.0f}p")

        # Order flow imbalance bonus
        if of_dir in ('SELL', 'BEARISH') or of_imb_value < -0.2:
            if of_strength in ('STRONG', 'EXTREME'):
                score += 15
                confluence.append("OF_BEAR_STRONG")
            else:
                score += 8
                confluence.append("OF_BEAR_CONFIRMS")

        # Delta strength bonus
        if delta_strength in ('STRONG', 'MODERATE'):
            score += 5
            confluence.append("DELTA_STRONG")

        # StochRSI overbought
        if stoch_k > 75:
            score += 10
            confluence.append("STOCHRSI_OVERBOUGHT")
        elif stoch_k > 65:
            score += 5
            confluence.append("STOCHRSI_HIGH")

        # Volume surge bonus
        if vol_surge_active:
            score += 8
            confluence.append("VOLUME_SURGE")

        # HTF alignment
        if htf_ok and smc_bias == 'BEARISH':
            score += 8
            confluence.append("HTF_SMC_ALIGNED")
        elif htf_ok:
            score += 3
            confluence.append("HTF_APPROVED")

        # BOS confirmation
        bos = structure.get('bos')
        has_bos = False
        if bos and 'BEARISH' in bos.get('type', ''):
            score += 10
            confluence.append("BOS_BEAR_CONFIRM")
            has_bos = True

        # Discount zone penalty
        if 'EXTREME_DISCOUNT' in pd_zone:
            score -= 15
            confluence.append("PD_DISCOUNT_PENALTY")

        # Fibonacci confluence bonus
        try:
            from backtest.fib_builder import build_fib_report, check_fib_confluence
            fib_report = build_fib_report(df_h1=df_h1, df_h4=df_h4, current_price=close_price)
            fib_check = check_fib_confluence(close_price, "SELL", fib_report, pip_size)
            if fib_check['fib_bonus'] > 0:
                score += fib_check['fib_bonus']
                confluence.extend(fib_check['confluence'])
        except Exception:
            pass

        if len(confluence) < 5:
            return None

        if score >= MIN_SCORE:
            entry_price = min(close_price, zone_top)
            sl_price  = round(zone_top + atr_pips * 0.3 * pip_size, 5)
            tp1_price = round(entry_price - atr_pips * 2.0 * pip_size, 5)
            tp2_price = round(entry_price - atr_pips * 3.5 * pip_size, 5)
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
                "sd_zone":     f"{zone_bottom:.5f}—{zone_top:.5f}",
                "_sd_zone_features": {
                    'zone_type': 'SUPPLY',
                    'zone_range_pips': supply_zone['zone_range_pips'],
                    'price_at_zone': 1,
                    'displacement_pips': disp_pips,
                    'age_bars': age_bars,
                    'trend': trend,
                    'delta_bias': delta_bias,
                    'delta_strength': delta_strength,
                    'of_imbalance': of_imb_value,
                    'of_strength': of_strength,
                    'stoch_rsi_k': stoch_k,
                    'supertrend_dir_h1': supertrend_dir_h1,
                    'supertrend_dir_h4': supertrend_dir_h4,
                    'htf_ok': 1 if htf_ok else 0,
                    'smc_bias': smc_bias,
                    'pd_zone': pd_zone,
                    'vol_surge': 1 if vol_surge_active else 0,
                    'has_bos': 1 if has_bos else 0,
                    'atr_pips': atr_pips,
                },
            }

    return None
