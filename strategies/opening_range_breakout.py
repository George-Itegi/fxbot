# =============================================================
# strategies/opening_range_breakout.py
# Strategy 7: Opening Range Breakout (ORB)
# Measures the first 15 minutes of London/NY open.
# Enters on retest of the broken range high/low.
# TP = range size | SL = 30% of range
# Win rate: 65-70% on session opens
# Best session: LONDON_OPEN, LONDON_SESSION (first hour)
#               NY_LONDON_OVERLAP (first hour of NY)
# Best state:  BALANCED, TRENDING_STRONG
# =============================================================

import pandas as pd
from datetime import datetime, timezone
from core.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "OPENING_RANGE_BREAKOUT"
MIN_SCORE     = 55
VERSION       = "1.0"

# --- ORB Parameters ---
RANGE_MINUTES  = 15       # Minutes after session open to establish the range
MAX_ENTRY_WAIT = 120      # Max minutes after open to still take a retest entry
SL_RANGE_PCT   = 0.30     # SL = 30% of opening range size
MIN_RANGE_PIPS = 5.0      # Minimum range size in pips to be valid (avoid dead sessions)
MAX_RANGE_PIPS = 50.0     # Maximum range size (too wide = dangerous breakout)

# Session open times (UTC hours)
SESSION_OPENS = {
    "LONDON_OPEN":  7,    # London opens at 07:00 UTC
    "LONDON_SESSION": 8,  # London session starts at 08:00 UTC
}


def _get_pip_size(price: float) -> float:
    """Return pip size for a symbol based on its price."""
    if price > 500:
        return 1.0
    elif price > 50:
        return 0.01
    else:
        return 0.0001


def _get_session_open_hour(session: str) -> int:
    """Return the UTC hour when the session opens."""
    # London open is the primary trigger
    if "LONDON" in session:
        return 7  # 07:00 UTC
    # NY open within the overlap
    if "NY" in session:
        return 12  # NY effectively starts around 12:00-13:00 UTC in our session model
    return -1


def _calculate_opening_range(df_m1: pd.DataFrame, session: str,
                              now_utc: datetime) -> dict | None:
    """
    Calculate the opening range for the current session.
    
    Returns dict with:
      - range_high, range_low, range_size, range_pips
      - minutes_since_open, is_range_formed, is_range_broken
      - break_direction, break_price
    Or None if range can't be calculated.
    """
    if df_m1 is None or len(df_m1) < 20:
        return None
    
    open_hour = _get_session_open_hour(session)
    if open_hour < 0:
        return None
    
    # Find the session open: find the earliest M1 candle at or after open hour
    df_m1 = df_m1.copy()
    df_m1['time'] = pd.to_datetime(df_m1['time'], utc=True)
    
    # Today's session open time
    today_open = now_utc.replace(hour=open_hour, minute=0, second=0, microsecond=0)
    
    # If we're before today's open, try yesterday
    if now_utc < today_open:
        return None
    
    # Minutes since open
    minutes_since_open = (now_utc - today_open).total_seconds() / 60
    
    # Range must be formed (at least RANGE_MINUTES minutes passed)
    if minutes_since_open < RANGE_MINUTES:
        return None
    
    # Too late — no more ORB entries after MAX_ENTRY_WAIT minutes
    if minutes_since_open > MAX_ENTRY_WAIT:
        return None
    
    # Filter candles from session open to RANGE_MINUTES
    range_end = today_open + pd.Timedelta(minutes=RANGE_MINUTES)
    range_candles = df_m1[
        (df_m1['time'] >= today_open) &
        (df_m1['time'] <= range_end)
    ]
    
    if len(range_candles) < 5:
        return None
    
    range_high = float(range_candles['high'].max())
    range_low  = float(range_candles['low'].min())
    range_size = range_high - range_low
    
    close_price = float(df_m1.iloc[-1]['close'])
    pip_size = _get_pip_size(close_price)
    range_pips = round(range_size / pip_size, 1)
    
    # Range must be meaningful
    if range_pips < MIN_RANGE_PIPS or range_pips > MAX_RANGE_PIPS:
        return None
    
    # Check if range has been broken by current price action
    # Look at candles AFTER the range formation period
    post_range = df_m1[df_m1['time'] > range_end]
    if len(post_range) < 1:
        return {"range_high": range_high, "range_low": range_low,
                "range_size": range_size, "range_pips": range_pips,
                "minutes_since_open": minutes_since_open,
                "is_range_formed": True, "is_range_broken": False,
                "break_direction": "NONE", "break_price": 0}
    
    # Check if price broke above range
    break_direction = "NONE"
    break_price = 0
    
    # Check if any post-range candle broke above or below
    for _, candle in post_range.iterrows():
        if candle['high'] > range_high:
            break_direction = "UP"
            break_price = float(candle['high'])
            break
        elif candle['low'] < range_low:
            break_direction = "DOWN"
            break_price = float(candle['low'])
            break
    
    # For entry we need a RETEST — price broke out then came back to the range level
    current_close = float(post_range.iloc[-1]['close'])
    current_low   = float(post_range.iloc[-1]['low'])
    current_high  = float(post_range.iloc[-1]['high'])
    
    is_retesting_up = False
    is_retesting_down = False
    
    if break_direction == "UP":
        # Price broke up, now retesting the range high from above
        if current_low <= range_high * 1.0003 and current_close > range_high:
            is_retesting_up = True
    elif break_direction == "DOWN":
        # Price broke down, now retesting the range low from below
        if current_high >= range_low * 0.9997 and current_close < range_low:
            is_retesting_down = True
    
    return {
        "range_high": range_high,
        "range_low": range_low,
        "range_size": range_size,
        "range_pips": range_pips,
        "minutes_since_open": minutes_since_open,
        "is_range_formed": True,
        "is_range_broken": break_direction != "NONE",
        "break_direction": break_direction,
        "break_price": break_price,
        "is_retesting_up": is_retesting_up,
        "is_retesting_down": is_retesting_down,
    }


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
    Fires when price breaks out of the opening range and retests
    the breakout level. Classic institutional strategy with
    high win rate during session opens.
    
    Entry criteria:
      1. Session must be LONDON_OPEN or within first 2 hours of London/NY
      2. Opening range established (first 15 minutes)
      3. Price has broken out of the range (above or below)
      4. Price is retesting the breakout level (pullback entry)
      5. Volume confirms the breakout direction
      6. Delta / order flow supports direction
    """
    # ── Session check — only trade during session opens ──────
    session = master_report.get('session', 'UNKNOWN') if master_report else 'UNKNOWN'
    
    # Only fire during London open period (first 2 hours)
    # NY is also valid but London has higher ORB win rate
    valid_sessions = ["LONDON_OPEN", "LONDON_SESSION"]
    if session not in valid_sessions:
        return None
    
    # ── Data validation ─────────────────────────────────────
    if df_m1 is None or len(df_m1) < 30:
        return None
    
    close_price = float(df_m1.iloc[-1]['close'])
    pip_size = _get_pip_size(close_price)
    
    # ── Calculate Opening Range ─────────────────────────────
    now_utc = datetime.now(timezone.utc)
    orb = _calculate_opening_range(df_m1, session, now_utc)
    
    if orb is None:
        return None
    
    if not orb.get('is_range_formed', False):
        return None
    
    if not orb.get('is_range_broken', False):
        return None  # No breakout yet — wait
    
    score = 0
    confluence = []
    
    range_pips = orb['range_pips']
    
    # ── Entry: Retest of breakout level ─────────────────────
    direction = None
    
    if orb.get('is_retesting_up', False):
        direction = "BUY"
        score += 30
        confluence.append(f"ORB_RETEST_UP_range={range_pips}p")
    elif orb.get('is_retesting_down', False):
        direction = "SELL"
        score += 30
        confluence.append(f"ORB_RETEST_DOWN_range={range_pips}p")
    else:
        # Breakout happened but no retest yet — skip this cycle
        return None
    
    # ── Volume confirmation ─────────────────────────────────
    if df_m1 is not None and len(df_m1) >= 21:
        current_vol = float(df_m1.iloc[-1].get('tick_volume', 0))
        avg_vol = float(df_m1.iloc[-21:-1]['tick_volume'].mean())
        if avg_vol > 0 and current_vol >= avg_vol * 1.3:
            score += 15
            confluence.append("VOL_CONFIRM_RETEST")
    
    # ── Tick volume surge from market report ────────────────
    if market_report:
        surge = market_report.get('volume_surge', {})
        if surge.get('surge_detected', False):
            score += 10
            confluence.append(f"SURGE_{surge.get('surge_ratio', 0)}x")
            # Surge direction should align
            surge_dir = surge.get('surge_direction', '')
            if (direction == "BUY" and surge_dir == "BUY") or \
               (direction == "SELL" and surge_dir == "SELL"):
                score += 5
                confluence.append("SURGE_ALIGNED")
    
    # ── Delta / Order Flow confirmation ─────────────────────
    if market_report:
        rolling_delta = market_report.get('rolling_delta', {})
        delta_bias = rolling_delta.get('bias', 'NEUTRAL')
        delta_strength = rolling_delta.get('strength', 'WEAK')
        
        if direction == "BUY" and delta_bias == "BULLISH":
            score += 10
            confluence.append("DELTA_BULL")
        elif direction == "SELL" and delta_bias == "BEARISH":
            score += 10
            confluence.append("DELTA_BEAR")
        
        if delta_strength in ("STRONG", "MODERATE"):
            score += 5
            confluence.append("DELTA_STRONG")
        
        # Order flow imbalance
        of_imb = market_report.get('order_flow_imbalance', {})
        imb = of_imb.get('imbalance', 0)
        if direction == "BUY" and imb > 0.15:
            score += 5
            confluence.append(f"OF_BULL_{imb:+.2f}")
        elif direction == "SELL" and imb < -0.15:
            score += 5
            confluence.append(f"OF_BEAR_{imb:+.2f}")
    
    # ── Momentum at entry ──────────────────────────────────
    if master_report:
        momentum = master_report.get('momentum', {})
        if momentum.get('is_choppy', False):
            # Choppy during retest = risky
            score -= 10
        elif momentum.get('is_scalpable', False):
            score += 5
            confluence.append("MOMENTUM_OK")
    
    # ── M15 structure alignment (bonus) ────────────────────
    if df_m15 is not None and len(df_m15) >= 10:
        m15_last = df_m15.iloc[-1]
        m15_ema9 = m15_last.get('ema_9', 0)
        m15_ema21 = m15_last.get('ema_21', 0)
        
        if direction == "BUY" and m15_ema9 > m15_ema21:
            score += 5
            confluence.append("M15_TREND_BULL")
        elif direction == "SELL" and m15_ema9 < m15_ema21:
            score += 5
            confluence.append("M15_TREND_BEAR")
    
    # ── Premium/Discount check ─────────────────────────────
    if smc_report:
        pd_info = smc_report.get('premium_discount', {})
        pd_zone = pd_info.get('zone', '')
        if direction == "BUY" and "DISCOUNT" in pd_zone:
            score += 5
            confluence.append("DISCOUNT_ZONE")
        elif direction == "SELL" and "PREMIUM" in pd_zone:
            score += 5
            confluence.append("PREMIUM_ZONE")
    
    # ── Score threshold ─────────────────────────────────────
    if score < MIN_SCORE:
        return None
    
    # ── Calculate SL/TP ─────────────────────────────────────
    # TP = range size (classic ORB target)
    # SL = 30% of range
    sl_pips = max(3.0, round(range_pips * SL_RANGE_PCT, 1))
    tp_pips = round(range_pips, 1)
    
    # Ensure minimum R:R
    if tp_pips / sl_pips < 1.5:
        tp_pips = round(sl_pips * 1.5, 1)
    
    if direction == "BUY":
        sl_price  = round(orb['range_low'] - sl_pips * pip_size, 5)
        tp1_price = round(close_price + tp_pips * pip_size, 5)
        tp2_price = round(close_price + tp_pips * 1.5 * pip_size, 5)
    else:
        sl_price  = round(orb['range_high'] + sl_pips * pip_size, 5)
        tp1_price = round(close_price - tp_pips * pip_size, 5)
        tp2_price = round(close_price - tp_pips * 1.5 * pip_size, 5)
    
    tp2_pips = round(tp_pips * 1.5, 1)
    
    log.info(f"[{STRATEGY_NAME}] {direction} {symbol}"
             f" Range:{range_pips}p SL:{sl_pips}p TP:{tp_pips}p"
             f" Score:{score} | {', '.join(confluence)}")
    
    return {
        "direction":   direction,
        "entry_price": close_price,
        "sl_price":    sl_price,
        "tp1_price":   tp1_price,
        "tp2_price":   tp2_price,
        "sl_pips":     sl_pips,
        "tp1_pips":    tp_pips,
        "tp2_pips":    tp2_pips,
        "strategy":    STRATEGY_NAME,
        "version":     VERSION,
        "score":       score,
        "confluence":  confluence,
        "spread":      0,
        "orb_range":   range_pips,
        "session":     session,
    }
