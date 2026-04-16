# =============================================================
# strategies/trend_continuation.py
# Strategy 9: Trend Continuation (Ride the London/NY Expansion)
# 
# PURPOSE: Catch strong trend continuation moves after a brief
# pullback to a key level (EMA, order block, or supply/demand zone).
# Designed for the Expansion phase of London/NY sessions.
#
# Entry criteria:
#   1. H4 trend is STRONG (EMA stack + price above/below all EMAs)
#   2. H1 structure confirms trend (HH/HL for bull, LH/LL for bear)
#   3. M15 pullback to EMA 21 or 50 (profit-taking pullback)
#   4. M5 shows rejection candle at the level (hammer, pin bar, engulf)
#   5. M1 momentum resuming in trend direction (velocity aligned)
#   6. Order flow confirms (delta + imbalance aligned with trend)
#
# SL: 1.5x M15 ATR (below pullback low for BUY, above high for SELL)
# TP: 3x M15 ATR (ride the expansion) — trailing after 1.5x ATR profit
#
# Best session: LONDON_SESSION, NY_LONDON_OVERLAP
# Best state:   TRENDING_STRONG, BREAKOUT_ACCEPTED
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "TREND_CONTINUATION"
MIN_SCORE     = 60
VERSION       = "1.0"


def _get_pip_size(price: float) -> float:
    """Return pip size for a symbol based on its price."""
    if price > 500:     # Indices
        return 1.0
    elif price > 50:    # JPY pairs, Gold
        return 0.01
    else:               # Standard forex
        return 0.0001


def _detect_rejection_candle(df_m5: pd.DataFrame, direction: str) -> str:
    """
    Detect a rejection candle at a key level on M5.
    Returns: 'REJECTION_BULL' or 'REJECTION_BEAR' or 'NONE'
    
    Bull rejection: small upper wick, long lower wick (hammer)
    Bear rejection: small lower wick, long upper wick (shooting star)
    """
    if df_m5 is None or len(df_m5) < 3:
        return "NONE"
    
    last = df_m5.iloc[-1]
    body = abs(last['close'] - last['open'])
    full_range = last['high'] - last['low']
    
    if full_range <= 0:
        return "NONE"
    
    upper_wick = last['high'] - max(last['close'], last['open'])
    lower_wick = min(last['close'], last['open']) - last['low']
    
    body_ratio = body / full_range
    
    # Bullish rejection: long lower wick, small body at top
    if lower_wick > body * 2.0 and upper_wick < body * 0.5 and body_ratio < 0.4:
        if last['close'] > last['open']:
            return "REJECTION_BULL"
    
    # Bearish rejection: long upper wick, small body at bottom
    if upper_wick > body * 2.0 and lower_wick < body * 0.5 and body_ratio < 0.4:
        if last['close'] < last['open']:
            return "REJECTION_BEAR"
    
    # Also check for strong bullish/bearish engulfing
    if df_m5 is not None and len(df_m5) >= 2:
        prev = df_m5.iloc[-2]
        prev_body = prev['close'] - prev['open']
        curr_body = last['close'] - last['open']
        
        # Bull engulfing at pullback
        if prev_body < 0 and curr_body > 0 and curr_body > abs(prev_body) * 1.2:
            return "REJECTION_BULL"
        # Bear engulfing at pullback
        if prev_body > 0 and curr_body < 0 and abs(curr_body) > prev_body * 1.2:
            return "REJECTION_BEAR"
    
    return "NONE"


def _is_pullback_to_ema(df_m15: pd.DataFrame, direction: str) -> dict:
    """
    Check if M15 price has pulled back to EMA 21 or EMA 50.
    Returns dict with pullback info.
    """
    if df_m15 is None or len(df_m15) < 20:
        return {"is_pullback": False, "level": 0, "ema_type": ""}
    
    last = df_m15.iloc[-1]
    ema21 = float(last.get('ema_21', 0))
    ema50 = float(last.get('ema_50', 0))
    close = float(last['close'])
    low = float(last['low'])
    high = float(last['high'])
    pip_size = _get_pip_size(close)
    
    tolerance = 2.0  # pips tolerance for "at" the EMA
    
    if direction == "BUY":
        # Price pulled back to EMA21 or EMA50 and is now bouncing
        dist_ema21 = (close - ema21) / pip_size
        dist_ema50 = (close - ema50) / pip_size
        
        if abs(dist_ema21) <= tolerance and dist_ema21 >= -tolerance:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21",
                    "dist_pips": dist_ema21}
        if abs(dist_ema50) <= tolerance * 1.5 and dist_ema50 >= -tolerance:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50",
                    "dist_pips": dist_ema50}
        # Check if low touched EMA but closed above
        if low <= ema21 * 1.001 and close > ema21:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21_TOUCH",
                    "dist_pips": dist_ema21}
        if low <= ema50 * 1.001 and close > ema50:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50_TOUCH",
                    "dist_pips": dist_ema50}
    
    elif direction == "SELL":
        dist_ema21 = (ema21 - close) / pip_size
        dist_ema50 = (ema50 - close) / pip_size
        
        if abs(dist_ema21) <= tolerance and dist_ema21 >= -tolerance:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21",
                    "dist_pips": dist_ema21}
        if abs(dist_ema50) <= tolerance * 1.5 and dist_ema50 >= -tolerance:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50",
                    "dist_pips": dist_ema50}
        # Check if high touched EMA but closed below
        if high >= ema21 * 0.999 and close < ema21:
            return {"is_pullback": True, "level": ema21, "ema_type": "EMA21_TOUCH",
                    "dist_pips": dist_ema21}
        if high >= ema50 * 0.999 and close < ema50:
            return {"is_pullback": True, "level": ema50, "ema_type": "EMA50_TOUCH",
                    "dist_pips": dist_ema50}
    
    return {"is_pullback": False, "level": 0, "ema_type": ""}


def _check_h4_trend(df_h4: pd.DataFrame, direction: str) -> dict:
    """Check H4 trend strength and alignment."""
    if df_h4 is None or len(df_h4) < 20:
        return {"aligned": False, "strength": 0}
    
    last = df_h4.iloc[-1]
    ema9 = float(last.get('ema_9', 0))
    ema21 = float(last.get('ema_21', 0))
    ema50 = float(last.get('ema_50', 0))
    st_dir = int(last.get('supertrend_dir', 0))
    close = float(last['close'])
    
    score = 0
    
    if direction == "BUY":
        # Perfect EMA stack: EMA9 > EMA21 > EMA50 and price above all
        if ema9 > ema21 > ema50 and close > ema9:
            score += 30
        elif ema9 > ema21 and close > ema21:
            score += 20
        elif close > ema21:
            score += 10
        
        if st_dir == 1:
            score += 15
    
    elif direction == "SELL":
        if ema9 < ema21 < ema50 and close < ema9:
            score += 30
        elif ema9 < ema21 and close < ema21:
            score += 20
        elif close < ema21:
            score += 10
        
        if st_dir == -1:
            score += 15
    
    return {"aligned": score >= 20, "strength": score}


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
    Trend Continuation Strategy:
    Enters on pullbacks within a strong H4 trend during London/NY expansion.
    """
    # ── Data validation ─────────────────────────────────────
    if df_m15 is None or df_h1 is None or df_h4 is None:
        return None
    if len(df_m15) < 30 or len(df_h1) < 20 or len(df_h4) < 20:
        return None
    if df_m5 is None or len(df_m5) < 20:
        return None
    
    close_price = float(df_m15.iloc[-1]['close'])
    pip_size = _get_pip_size(close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size
    
    if atr_pips < 2.0:
        return None  # Too quiet
    
    score = 0
    confluence = []
    
    # ── Step 1: Determine H4 trend direction ────────────────
    h4_bull = _check_h4_trend(df_h4, "BUY")
    h4_bear = _check_h4_trend(df_h4, "SELL")
    
    # Pick the dominant H4 trend
    if h4_bull['strength'] > h4_bear['strength'] and h4_bull['aligned']:
        direction = "BUY"
        score += h4_bull['strength']
        confluence.append(f"H4_BULL_TREND({h4_bull['strength']}pts)")
    elif h4_bear['strength'] > h4_bull['strength'] and h4_bear['aligned']:
        direction = "SELL"
        score += h4_bear['strength']
        confluence.append(f"H4_BEAR_TREND({h4_bear['strength']}pts)")
    else:
        return None  # No clear H4 trend
    
    # ── Step 2: H1 structure confirmation ───────────────────
    h1 = df_h1.iloc[-1]
    h1_ema_bull = float(h1.get('ema_9', 0)) > float(h1.get('ema_21', 0))
    h1_ema_bear = float(h1.get('ema_9', 0)) < float(h1.get('ema_21', 0))
    h1_st = int(h1.get('supertrend_dir', 0))
    
    if direction == "BUY" and h1_ema_bull:
        score += 15
        confluence.append("H1_STRUCTURE_BULL")
    elif direction == "SELL" and h1_ema_bear:
        score += 15
        confluence.append("H1_STRUCTURE_BEAR")
    else:
        return None  # H1 must agree with H4
    
    if (direction == "BUY" and h1_st == 1) or (direction == "SELL" and h1_st == -1):
        score += 5
        confluence.append("H1_SUPERTREND_ALIGNED")
    
    # ── Step 3: M15 pullback to key EMA ─────────────────────
    pullback = _is_pullback_to_ema(df_m15, direction)
    if pullback['is_pullback']:
        score += 20
        confluence.append(f"M15_PULLBACK_{pullback['ema_type']}")
    else:
        # Soft check: is price at least near the trend?
        m15_last = df_m15.iloc[-1]
        m15_st = int(m15_last.get('supertrend_dir', 0))
        if (direction == "BUY" and m15_st == 1) or (direction == "SELL" and m15_st == -1):
            score += 5
            confluence.append("M15_SUPERTREND_OK")
        else:
            return None  # No pullback and no trend confirmation
    
    # ── Step 4: M5 rejection candle ─────────────────────────
    rejection = _detect_rejection_candle(df_m5, direction)
    if (direction == "BUY" and rejection == "REJECTION_BULL") or \
       (direction == "SELL" and rejection == "REJECTION_BEAR"):
        score += 15
        confluence.append("M5_REJECTION_CANDLE")
    else:
        # Check if M5 is at least showing directional momentum
        if df_m5 is not None and len(df_m5) >= 3:
            m5_last = df_m5.iloc[-1]
            m5_body = m5_last['close'] - m5_last['open']
            if (direction == "BUY" and m5_body > 0) or \
               (direction == "SELL" and m5_body < 0):
                score += 5
                confluence.append("M5_DIRECTIONAL")
    
    # ── Step 5: M1 momentum resumption ─────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        velocity = momentum.get('velocity_pips_min', 0)
        vel_dir = momentum.get('velocity_direction', 'FLAT')
        is_scalpable = momentum.get('is_scalpable', False)
        
        if is_scalpable:
            if (direction == "BUY" and vel_dir == "UP") or \
               (direction == "SELL" and vel_dir == "DOWN"):
                score += 10
                confluence.append(f"M1_VELOCITY_{velocity}p/m")
            else:
                score += 3
                confluence.append("M1_VELOCITY_EXISTS")
    
    # ── Step 6: Order flow confirmation ─────────────────────
    if master_report:
        of_imb = master_report.get('order_flow_imbalance', {})
        imb = of_imb.get('imbalance', 0)
        
        if direction == "BUY" and imb > 0.15:
            score += 10
            confluence.append(f"OF_BULL_{imb:+.2f}")
        elif direction == "SELL" and imb < -0.15:
            score += 10
            confluence.append(f"OF_BEAR_{imb:+.2f}")
    
    # ── Step 7: SMC / HTF confirmation (bonus) ──────────────
    if smc_report:
        htf = smc_report.get('htf_alignment', {})
        if htf.get('approved', False):
            h4_bias = htf.get('h4_bias', '')
            if (direction == "BUY" and h4_bias == "BULLISH") or \
               (direction == "SELL" and h4_bias == "BEARISH"):
                score += 10
                confluence.append("HTF_ALIGNED")
    
    # ── Choppy market penalty ───────────────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        if momentum.get('is_choppy', False):
            score -= 15
            confluence.append("CHOPPY_PENALTY")
    
    # ── Score threshold ─────────────────────────────────────
    if score < MIN_SCORE:
        return None
    
    # ── Calculate SL/TP (ATR-based) ────────────────────────
    sl_pips = round(atr_pips * 1.5, 1)
    sl_pips = max(sl_pips, 3.0)  # Minimum 3 pips
    
    # TP: 2.5-3x SL for trend continuation (let winners run)
    tp1_pips = round(sl_pips * 2.5, 1)
    tp2_pips = round(sl_pips * 4.0, 1)
    
    if direction == "BUY":
        entry = float(df_m1.iloc[-1]['close']) if df_m1 is not None and len(df_m1) > 0 else close_price
        sl_price = round(entry - sl_pips * pip_size, 5)
        tp1_price = round(entry + tp1_pips * pip_size, 5)
        tp2_price = round(entry + tp2_pips * pip_size, 5)
    else:
        entry = float(df_m1.iloc[-1]['close']) if df_m1 is not None and len(df_m1) > 0 else close_price
        sl_price = round(entry + sl_pips * pip_size, 5)
        tp1_price = round(entry - tp1_pips * pip_size, 5)
        tp2_price = round(entry - tp2_pips * pip_size, 5)
    
    log.info(f"[{STRATEGY_NAME}] {direction} {symbol}"
             f" Score:{score} | {', '.join(confluence)}")
    
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
