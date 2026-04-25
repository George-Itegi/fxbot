# =============================================================
# strategies/m1_momentum_scalp.py
# Strategy 6: M1 Momentum Scalp
# Combines M15 directional bias + M5 sweep confirmation +
# M1 engulfing pattern + volume spike trigger.
# Fast scalp entries with tight SL. TP 8-15 pips, SL 5 pips.
# Win rate target: 65%
# Best session: LONDON_SESSION, NY_LONDON_OVERLAP, NY_SESSION
# Best state:  TRENDING_STRONG, BREAKOUT_ACCEPTED
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "M1_MOMENTUM_SCALP"
MIN_SCORE     = 75
VERSION       = "2.0"  # v2.0: ATR-based SL instead of fixed 5p

# --- Scalping parameters ---
SL_ATR_MULTIPLIER   = 1.0    # SL = M5 ATR * this multiplier (was fixed 5.0 pips)
SL_MIN_PIPS        = 3.0    # Minimum SL pips (safety floor)
SL_MAX_PIPS        = 12.0   # Maximum SL pips (safety cap)
TP_MIN_RR          = 1.5    # Minimum TP/SL ratio
TP_RR_MULTIPLIER   = 2.5    # TP = SL * this (let winners run)


def _get_pip_size(symbol: str, price: float) -> float:
    from core.pip_utils import get_pip_size as _gps
    return _gps(symbol, price)


def _detect_m1_engulfing(df_m1: pd.DataFrame) -> str:
    """
    Detect engulfing pattern on the last 2 M1 candles.
    Returns: 'BULLISH_ENGULF', 'BEARISH_ENGULF', or 'NONE'
    
    A bullish engulfing = previous candle is bearish (red), current is bullish (green)
    and current body fully engulfs the previous body.
    """
    if df_m1 is None or len(df_m1) < 3:
        return "NONE"
    
    prev = df_m1.iloc[-2]
    curr = df_m1.iloc[-1]
    
    prev_body = prev['close'] - prev['open']
    curr_body = curr['close'] - curr['open']
    
    # Bullish engulfing: prev red, curr green, curr body > prev body
    if prev_body < 0 and curr_body > 0:
        if curr_body > abs(prev_body):
            # Check current high engulfs prev high
            if curr['high'] >= prev['high']:
                return "BULLISH_ENGULF"
    
    # Bearish engulfing: prev green, curr red, curr body > prev body
    elif prev_body > 0 and curr_body < 0:
        if abs(curr_body) > prev_body:
            if curr['low'] <= prev['low']:
                return "BEARISH_ENGULF"
    
    return "NONE"


def _check_volume_spike(df_m1: pd.DataFrame, multiplier: float = 1.5) -> bool:
    """
    Check if the current M1 candle has a volume spike.
    Current volume must be >= multiplier * average of last 20 candles.
    """
    if df_m1 is None or len(df_m1) < 21:
        return False
    
    current_vol = float(df_m1.iloc[-1].get('tick_volume', 0))
    avg_vol = float(df_m1.iloc[-21:-1]['tick_volume'].mean())
    
    if avg_vol <= 0:
        return False
    
    return current_vol >= avg_vol * multiplier


def _get_m15_bias(df_m15: pd.DataFrame) -> str:
    """
    Determine M15 directional bias using EMA alignment + supertrend.
    Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
    """
    if df_m15 is None or len(df_m15) < 20:
        return "NEUTRAL"
    
    last = df_m15.iloc[-1]
    ema9  = last.get('ema_9', 0)
    ema21 = last.get('ema_21', 0)
    ema50 = last.get('ema_50', 0)
    st_dir = int(last.get('supertrend_dir', 0))
    
    bull_ema = ema9 > ema21 > ema50
    bear_ema = ema9 < ema21 < ema50
    
    if bull_ema and st_dir == 1:
        return "BULLISH"
    elif bear_ema and st_dir == -1:
        return "BEARISH"
    elif ema9 > ema21:
        return "BULLISH"
    elif ema9 < ema21:
        return "BEARISH"
    
    return "NEUTRAL"


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
    Fires when M15 bias + M5 sweep + M1 engulfing + volume spike
    all align for a high-probability momentum scalp.
    
    Entry criteria:
      1. M15 directional bias confirmed (EMA + supertrend)
      2. M5 shows recent sweep (liquidity grab) aligned with bias
      3. M1 shows engulfing candle pattern in bias direction
      4. Current M1 candle has volume spike (>= 1.5x average)
      5. Momentum velocity is sufficient (from master_report)
      6. Order flow imbalance supports direction
    """
    # ── Data validation ─────────────────────────────────────
    if df_m1 is None or df_m5 is None or df_m15 is None:
        return None
    if len(df_m1) < 21 or len(df_m5) < 50 or len(df_m15) < 30:
        return None
    
    close_price = float(df_m1.iloc[-1]['close'])
    pip_size = _get_pip_size(symbol, close_price)
    atr_pips = float(df_m15.iloc[-1].get('atr', 0)) / pip_size
    
    # Need minimum ATR for scalp to be worthwhile
    if atr_pips < 2.0:
        return None
    
    score = 0
    confluence = []
    
    # ── Gate 1: M15 Directional Bias ────────────────────────
    m15_bias = _get_m15_bias(df_m15)
    if m15_bias == "NEUTRAL":
        return None  # No clear bias = no scalp
    
    direction = "BUY" if m15_bias == "BULLISH" else "SELL"
    score += 15
    confluence.append(f"M15_BIAS_{m15_bias}")
    
    # Extra if all 3 EMAs aligned perfectly
    m15_last = df_m15.iloc[-1]
    if direction == "BULLISH" and m15_last.get('ema_9', 0) > m15_last.get('ema_21', 0) > m15_last.get('ema_50', 0):
        score += 5
        confluence.append("M15_EMA_STACKED")
    elif direction == "BEARISH" and m15_last.get('ema_9', 0) < m15_last.get('ema_21', 0) < m15_last.get('ema_50', 0):
        score += 5
        confluence.append("M15_EMA_STACKED")
    
    # ── Gate 2: M5 Sweep / Structure Confirmation ────────────
    sweep_confirmed = False
    if smc_report:
        last_sweep = smc_report.get('last_sweep')
        if last_sweep:
            sweep_bias = last_sweep.get('bias', '')
            reversal = last_sweep.get('reversal_pips', 0)
            
            # Sweep aligns with M15 bias and shows reversal
            if direction == "BULLISH" and sweep_bias == "BULLISH" and reversal > 2:
                score += 20
                confluence.append("M5_BULL_SWEEP_REVERSED")
                sweep_confirmed = True
            elif direction == "BEARISH" and sweep_bias == "BEARISH" and reversal > 2:
                score += 20
                confluence.append("M5_BEAR_SWEEP_REVERSED")
                sweep_confirmed = True
    
    # Fallback: M5 EMA + supertrend alignment (if no sweep)
    if not sweep_confirmed and df_m5 is not None and len(df_m5) >= 20:
        m5_last = df_m5.iloc[-1]
        m5_ema_ok = (direction == "BULLISH" and m5_last.get('ema_9', 0) > m5_last.get('ema_21', 0)) or \
                    (direction == "BEARISH" and m5_last.get('ema_9', 0) < m5_last.get('ema_21', 0))
        m5_st_ok = (direction == "BULLISH" and int(m5_last.get('supertrend_dir', 0)) == 1) or \
                   (direction == "BEARISH" and int(m5_last.get('supertrend_dir', 0)) == -1)
        
        if m5_ema_ok and m5_st_ok:
            score += 10
            confluence.append("M5_STRUCTURE_ALIGNED")
    
    # ── Gate 3: M1 Engulfing Pattern ────────────────────────
    engulfing = _detect_m1_engulfing(df_m1)
    
    if direction == "BULLISH" and engulfing == "BULLISH_ENGULF":
        score += 25
        confluence.append("M1_BULL_ENGULF")
    elif direction == "BEARISH" and engulfing == "BEARISH_ENGULF":
        score += 25
        confluence.append("M1_BEAR_ENGULF")
    elif engulfing == "NONE":
        # Soft fallback: check if M1 candle is strong in direction
        m1_last = df_m1.iloc[-1]
        m1_body = m1_last['close'] - m1_last['open']
        m1_range = m1_last['high'] - m1_last['low']
        if m1_range > 0:
            body_ratio = abs(m1_body) / m1_range
            if body_ratio > 0.6:
                aligned = (direction == "BULLISH" and m1_body > 0) or \
                          (direction == "BEARISH" and m1_body < 0)
                if aligned:
                    score += 10
                    confluence.append("M1_STRONG_CANDLE")
        # No M1 candle confirmation at all — don't fire
        if score < MIN_SCORE + 10:  # Need at least one candle signal
            return None
    
    # ── Gate 4: Volume Spike ────────────────────────────────
    vol_spike = _check_volume_spike(df_m1, multiplier=1.5)
    if vol_spike:
        score += 15
        confluence.append("M1_VOL_SPIKE")
    else:
        # Try tick volume surge from market report
        if market_report:
            surge = market_report.get('volume_surge', {})
            if surge.get('surge_detected', False):
                score += 10
                confluence.append("TICK_VOL_SURGE")
    
    # ── Gate 5: Momentum Velocity (from master_report) ──────
    if master_report:
        momentum = master_report.get('momentum', {})
        velocity = momentum.get('velocity_pips_min', 0)
        vel_dir = momentum.get('velocity_direction', 'FLAT')
        is_scalpable = momentum.get('is_scalpable', False)
        
        if is_scalpable:
            if (direction == "BULLISH" and vel_dir == "UP") or \
               (direction == "BEARISH" and vel_dir == "DOWN"):
                score += 15
                confluence.append(f"VELOCITY_ALIGNED_{velocity}p/m")
            else:
                score += 5
                confluence.append(f"VELOCITY_OK_{velocity}p/m")
    
    # ── Gate 6: Order Flow Imbalance ────────────────────────
    if market_report:
        of_imb = market_report.get('order_flow_imbalance', {})
        imb = of_imb.get('imbalance', 0)
        imb_dir = of_imb.get('direction', 'NEUTRAL')
        
        if direction == "BULLISH" and imb > 0.2:
            score += 10
            confluence.append(f"OF_BULL_{imb:+.2f}")
        elif direction == "BEARISH" and imb < -0.2:
            score += 10
            confluence.append(f"OF_BEAR_{imb:+.2f}")
    
    # ── HTF Confirmation (bonus) ────────────────────────────
    if smc_report:
        htf = smc_report.get('htf_alignment', {})
        if htf.get('approved', False):
            h4_bias = htf.get('h4_bias', '')
            if (direction == "BULLISH" and h4_bias == "BULLISH") or \
               (direction == "BEARISH" and h4_bias == "BEARISH"):
                score += 10
                confluence.append("HTF_H4_ALIGNED")
    
    # ── Score threshold ─────────────────────────────────────
    if len(confluence) < 5:
        return None
    if score < MIN_SCORE:
        return None
    
    # ── Calculate SL/TP (ATR-based, not fixed) ──────────────
    # SL based on current M5 ATR — adapts to volatility
    if df_m5 is not None and len(df_m5) >= 20:
        m5_atr_raw = float(df_m5.iloc[-1].get('atr', 0))
        if m5_atr_raw > 0:
            sl_pips_raw = round(m5_atr_raw / pip_size * SL_ATR_MULTIPLIER, 1)
            sl_pips = max(SL_MIN_PIPS, min(SL_MAX_PIPS, sl_pips_raw))
        else:
            sl_pips = SL_MIN_PIPS
    else:
        sl_pips = SL_MIN_PIPS

    # TP based on SL with minimum R:R
    tp_pips = round(sl_pips * TP_RR_MULTIPLIER, 1)
    tp_pips = max(sl_pips * TP_MIN_RR, tp_pips)  # At least 1.5:1 R:R
    
    if direction == "BUY":
        sl_price  = round(close_price - sl_pips * pip_size, 5)
        tp1_price = round(close_price + tp_pips * pip_size, 5)
        tp2_price = round(close_price + tp_pips * 2.0 * pip_size, 5)
    else:  # SELL
        sl_price  = round(close_price + sl_pips * pip_size, 5)
        tp1_price = round(close_price - tp_pips * pip_size, 5)
        tp2_price = round(close_price - tp_pips * 2.0 * pip_size, 5)
    
    tp1_pips = tp_pips
    tp2_pips = round(tp_pips * 2.0, 1)

    log.info(f"[{STRATEGY_NAME}] {direction} {symbol}"
             f" Score:{score} | SL:{sl_pips}p TP:{tp_pips}p"
             f" | {', '.join(confluence)}")
    
    return {
        "direction":   direction,
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
        "spread":      0,
    }
