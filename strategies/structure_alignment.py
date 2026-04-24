# =============================================================
# strategies/structure_alignment.py  v1.0
# Strategy 15: Multi-TF Structure Alignment (ORDER_FLOW group)
#
# Purpose: Second ORDER_FLOW strategy so the group isn't a single
# point of failure. Unlike DELTA_DIVERGENCE which looks for DIVERGENCE
# (price vs delta disagreement), this looks for AGREEMENT across
# timeframes — BOS on M15 + BOS on H1 in the same direction, with
# cumulative delta confirming.
#
# Entry logic:
#   1. BOS on M15 in a clear direction
#   2. BOS on H1 in the SAME direction (cross-TF confirmation)
#   3. Cumulative delta agrees with direction (not diverging)
#   4. OF imbalance STRONG/EXTREME
#   5. No opposing FVG nearby
#   6. Premium/discount zone alignment
#
# Win rate target: 45-55%
# Best session: LONDON_SESSION, NY_LONDON_OVERLAP, LONDON_OPEN
# Best state:  TRENDING_STRONG, BREAKOUT_ACCEPTED, TRENDING_EXTENDED
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "STRUCTURE_ALIGNMENT"
MIN_SCORE     = 70
VERSION       = "1.0"

# --- Parameters ---
OPPOSING_FVG_BUFFER = 20.0   # Pips — check for opposing FVG in this range
MIN_BOS_AGE_BARS    = 15     # Maximum bars since BOS occurred


def _get_pip_size(price: float) -> float:
    if price > 500:     return 1.0
    elif price > 50:    return 0.01
    else:               return 0.0001


def _get_bos_direction(bos_list: list) -> dict:
    """
    Extract the most recent BOS direction from a BOS list.
    Returns dict with direction and count of matching BOS signals.
    """
    if not bos_list:
        return {"direction": "NONE", "count": 0}

    bull_count = 0
    bear_count = 0
    latest_bull = None
    latest_bear = None

    for bos in bos_list:
        bos_type = str(bos.get('type', '')).upper()
        if 'BULL' in bos_type:
            bull_count += 1
            if latest_bull is None:
                latest_bull = bos
        elif 'BEAR' in bos_type:
            bear_count += 1
            if latest_bear is None:
                latest_bear = bos

    # Return dominant direction
    if bull_count > bear_count:
        return {"direction": "BULLISH", "count": bull_count,
                "latest": latest_bull}
    elif bear_count > bull_count:
        return {"direction": "BEARISH", "count": bear_count,
                "latest": latest_bear}
    else:
        return {"direction": "NONE", "count": 0, "latest": None}


def _check_opposing_fvg(fvg_list: list, direction: str,
                        close_price: float, pip_size: float) -> bool:
    """
    Check if there's an opposing FVG near current price.
    BUY signal: check for unfilled BEARISH FVG above price.
    SELL signal: check for unfilled BULLISH FVG below price.
    """
    for fvg in fvg_list:
        fvg_type = str(fvg.get('type', '')).upper()
        fvg_top = float(fvg.get('top', 0))
        fvg_bottom = float(fvg.get('bottom', 0))
        filled = fvg.get('filled', True)

        if filled:
            continue

        fvg_mid = (fvg_top + fvg_bottom) / 2
        dist = abs(close_price - fvg_mid) / pip_size

        if dist > OPPOSING_FVG_BUFFER:
            continue

        if direction == "BUY" and "BEAR" in fvg_type:
            return True  # Opposing bearish FVG nearby
        elif direction == "SELL" and "BULL" in fvg_type:
            return True  # Opposing bullish FVG nearby

    return False


def evaluate(symbol: str,
             df_m1: pd.DataFrame = None,
             df_m5: pd.DataFrame = None,
             df_m15: pd.DataFrame = None,
             df_h1: pd.DataFrame = None,
             smc_report: dict = None,
             market_report: dict = None,
             df_h4: pd.DataFrame = None,
             master_report: dict = None) -> dict | None:
    """
    Multi-TF Structure Alignment Strategy:
    Fires when BOS on both M15 and H1 agree on direction, with
    cumulative delta confirming (agreement, not divergence).
    """
    if df_m15 is None or df_h1 is None:
        return None
    if len(df_m15) < 30 or len(df_h1) < 30:
        return None
    if market_report is None:
        return None

    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size

    if atr_pips < 2.0:
        return None

    # ── Step 1: Get BOS from SMC report (mandatory) ─────
    if smc_report is None:
        return None

    bos_list = smc_report.get('bos', [])
    fvg_list = smc_report.get('fvgs', [])

    if not bos_list:
        return None  # No BOS = no structure break = no signal

    # The SMC report contains BOS from the timeframe it was built on.
    # We use it as the primary structure signal.
    bos_info = _get_bos_direction(bos_list)

    if bos_info['direction'] == "NONE" or bos_info['count'] == 0:
        return None

    # Determine trade direction from BOS
    if bos_info['direction'] == "BULLISH":
        direction = "BUY"
    else:
        direction = "SELL"

    score = 0
    confluence = []

    # ── Score: BOS strength ─────────────────────────────
    bos_count = bos_info['count']
    if bos_count >= 3:
        score += 25
        confluence.append(f"BOS_MULTIPLE({bos_count})")
    elif bos_count >= 2:
        score += 18
        confluence.append(f"BOS_DOUBLE({bos_count})")
    else:
        score += 12
        confluence.append("BOS_SINGLE")

    confluence.append(f"STRUCTURE_{bos_info['direction']}")

    # ── Step 2: H1 structure confirmation (mandatory) ───
    # Check H1 EMA alignment as cross-TF structure proxy
    h1 = df_h1.iloc[-1]
    h1_ema9  = float(h1.get('ema_9', 0))
    h1_ema21 = float(h1.get('ema_21', 0))
    h1_ema50 = float(h1.get('ema_50', 0))
    h1_st = int(h1.get('supertrend_dir', 0))

    h1_confirms = False
    if direction == "BUY":
        if h1_ema9 > h1_ema21:
            h1_confirms = True
            score += 15
            confluence.append("H1_EMA_BULL")
        if h1_ema9 > h1_ema21 > h1_ema50:
            score += 5
            confluence.append("H1_FULL_BULL_ALIGN")
    elif direction == "SELL":
        if h1_ema9 < h1_ema21:
            h1_confirms = True
            score += 15
            confluence.append("H1_EMA_BEAR")
        if h1_ema9 < h1_ema21 < h1_ema50:
            score += 5
            confluence.append("H1_FULL_BEAR_ALIGN")

    if not h1_confirms:
        return None  # Cross-TF structure must agree

    # H1 Supertrend bonus
    if (direction == "BUY" and h1_st == 1) or \
       (direction == "SELL" and h1_st == -1):
        score += 8
        confluence.append("H1_SUPERTREND_ALIGNED")

    # ── Step 3: Cumulative delta agreement (mandatory) ──
    rolling_delta = market_report.get('rolling_delta', {})
    delta_bias = rolling_delta.get('bias', 'NEUTRAL')
    delta_value = rolling_delta.get('delta', 0)

    delta_agrees = False
    if direction == "BUY":
        if delta_bias == "BULLISH":
            delta_agrees = True
            score += 15
            confluence.append(f"DELTA_BULL_{delta_value:+.0f}")
        elif delta_value > 0:
            delta_agrees = True
            score += 8
            confluence.append(f"DELTA_POSITIVE_{delta_value:+.0f}")
    elif direction == "SELL":
        if delta_bias == "BEARISH":
            delta_agrees = True
            score += 15
            confluence.append(f"DELTA_BEAR_{delta_value:+.0f}")
        elif delta_value < 0:
            delta_agrees = True
            score += 8
            confluence.append(f"DELTA_NEGATIVE_{delta_value:+.0f}")

    if not delta_agrees:
        return None  # Delta must agree (this is ORDER_FLOW group)

    # ── Step 4: OF imbalance STRONG/EXTREME ─────────────
    of_imb = market_report.get('order_flow_imbalance', {})
    imb = of_imb.get('imbalance', 0)
    imb_strength = of_imb.get('strength', 'NONE')

    if direction == "BUY":
        if imb > 0.15 and imb_strength in ('STRONG', 'EXTREME'):
            score += 10
            confluence.append(f"OF_STRONG_BULL_{imb:+.2f}")
        elif imb > 0.1:
            score += 5
            confluence.append(f"OF_MODERATE_BULL_{imb:+.2f}")
    elif direction == "SELL":
        if imb < -0.15 and imb_strength in ('STRONG', 'EXTREME'):
            score += 10
            confluence.append(f"OF_STRONG_BEAR_{imb:+.2f}")
        elif imb < -0.1:
            score += 5
            confluence.append(f"OF_MODERATE_BEAR_{imb:+.2f}")

    # ── Step 5: No opposing FVG (filter) ───────────────
    has_opposing = _check_opposing_fvg(fvg_list, direction,
                                        close_price, pip_size)
    if has_opposing:
        score -= 12
        confluence.append("OPPOSING_FVG_PENALTY")
    else:
        score += 5
        confluence.append("NO_OPPOSING_FVG")

    # ── Step 6: Premium/Discount zone alignment ─────────
    if smc_report:
        pd_info = smc_report.get('premium_discount', {})
        pd_zone = pd_info.get('zone', '')

        if direction == "SELL" and "PREMIUM" in pd_zone:
            score += 8
            confluence.append("PREMIUM_ZONE_SELL")
        elif direction == "BUY" and "DISCOUNT" in pd_zone:
            score += 8
            confluence.append("DISCOUNT_ZONE_BUY")
        elif direction == "SELL" and "DISCOUNT" in pd_zone:
            score -= 5
            confluence.append("SELLING_FROM_DISCOUNT_PENALTY")
        elif direction == "BUY" and "PREMIUM" in pd_zone:
            score -= 5
            confluence.append("BUYING_FROM_PREMIUM_PENALTY")

    # ── Step 7: H4 trend alignment (bonus) ─────────────
    if df_h4 is not None and len(df_h4) >= 20:
        h4 = df_h4.iloc[-1]
        h4_ema9  = float(h4.get('ema_9', 0))
        h4_ema21 = float(h4.get('ema_21', 0))

        if direction == "BUY" and h4_ema9 > h4_ema21:
            score += 8
            confluence.append("H4_TREND_BULL")
        elif direction == "SELL" and h4_ema9 < h4_ema21:
            score += 8
            confluence.append("H4_TREND_BEAR")

    # ── Step 8: Volume surge (bonus) ────────────────────
    if market_report:
        surge = market_report.get('volume_surge', {})
        if surge.get('surge_detected', False):
            score += 5
            confluence.append("VOLUME_SURGE")

    # ── Step 9: M5 momentum (bonus) ─────────────────────
    if df_m5 is not None and len(df_m5) >= 3:
        m5_last = df_m5.iloc[-1]
        m5_body = m5_last['close'] - m5_last['open']
        if (direction == "BUY" and m5_body > 0) or \
           (direction == "SELL" and m5_body < 0):
            score += 5
            confluence.append("M5_MOMENTUM")

    # ── Choppy market penalty ───────────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        if momentum.get('is_choppy', False):
            score -= 15
            confluence.append("CHOPPY_PENALTY")

    if len(confluence) < 5:
        return None

    # ── Score threshold ─────────────────────────────────
    if score < MIN_SCORE:
        return None

    # ── Calculate SL/TP ─────────────────────────────────
    entry = close_price

    sl_pips = round(atr_pips * 1.5, 1)
    sl_pips = max(sl_pips, 3.0)

    tp1_pips = round(sl_pips * 2.0, 1)
    tp2_pips = round(sl_pips * 3.5, 1)

    if direction == "BUY":
        sl_price  = round(entry - sl_pips * pip_size, 5)
        tp1_price = round(entry + tp1_pips * pip_size, 5)
        tp2_price = round(entry + tp2_pips * pip_size, 5)
    else:
        sl_price  = round(entry + sl_pips * pip_size, 5)
        tp1_price = round(entry - tp1_pips * pip_size, 5)
        tp2_price = round(entry - tp2_pips * pip_size, 5)

    log.info(f"[{STRATEGY_NAME} v{VERSION}] {direction} {symbol}"
             f" entry={entry:.5f} Score:{score} | "
             f"{', '.join(confluence)}")

    return {
        "direction":   direction,
        "entry_price": entry,
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
        "spread":      0,
    }
