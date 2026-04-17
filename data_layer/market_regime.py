# =============================================================
# data_layer/market_regime.py
# Detects what kind of market we're in RIGHT NOW.
# Each strategy only activates in its optimal regime.
# Regimes: TRENDING_UP | TRENDING_DOWN | RANGING | VOLATILE
#
# FIXED: Session detection now covers all 24 hours with no gaps.
# Aligns with config/settings.py SESSIONS definitions.
# =============================================================

import pandas as pd
from core.logger import get_logger

log = get_logger(__name__)


def detect_regime(df_h1: pd.DataFrame, symbol: str = None) -> str:
    """
    Classify current market regime using ADX, EMA alignment, ATR.
    Returns: 'TRENDING_UP' | 'TRENDING_DOWN' | 'RANGING' | 'VOLATILE'
    """
    if df_h1 is None or len(df_h1) < 50:
        return "UNKNOWN"

    h1 = df_h1.iloc[-1]
    
    # Use H4 for "HTF" filter instead of D1
    ema200_htf = 0
    close_htf  = 0
    if symbol:
        from data_layer.price_feed import get_candles
        df_h4 = get_candles(symbol, 'H4', 50)
        if df_h4 is not None and not df_h4.empty:
            ema200_htf = df_h4['ema_200'].iloc[-1]
            close_htf  = df_h4['close'].iloc[-1]

    adx       = h1.get('adx', 0)
    atr       = h1.get('atr', 0)
    atr_avg   = df_h1['atr'].rolling(20).mean().iloc[-1]
    ema9      = h1.get('ema_9', 0)
    ema21     = h1.get('ema_21', 0)
    ema50     = h1.get('ema_50', 0)

    # Volatile: ATR is much higher than its average
    if atr > atr_avg * 1.8:
        log.info("[REGIME] VOLATILE - ATR spike detected")
        return "VOLATILE"

    # Trending: ADX > 25 confirms a real trend
    if adx > 25:
        bull_structure = ema9 > ema21 > ema50
        bear_structure = ema9 < ema21 < ema50
        
        # HTF Confirm if possible
        htf_bull = close_htf > ema200_htf if ema200_htf > 0 else True
        htf_bear = close_htf < ema200_htf if ema200_htf > 0 else True

        if bull_structure and htf_bull:
            log.info("[REGIME] TRENDING_UP - Strong bull structure")
            return "TRENDING_UP"
        if bear_structure and htf_bear:
            log.info("[REGIME] TRENDING_DOWN - Strong bear structure")
            return "TRENDING_DOWN"

    # Ranging: ADX < 20 = no directional conviction
    if adx < 20:
        log.info("[REGIME] RANGING - Low ADX, choppy market")
        return "RANGING"

    # Weak trend - treat as ranging
    log.info("[REGIME] RANGING - Weak trend, staying cautious")
    return "RANGING"


def get_session() -> str:
    """
    Return the current trading session based on UTC hour.
    Aligned with real institutional forex session behaviors.

    Session breakdown (UTC) — based on EAT (UTC+3) market structure:
      SYDNEY             21:00 - 00:00  (Price Discovery — reacts to weekend news,
                                         thin liquidity, early ranges)
      TOKYO              00:00 - 07:00  (Accumulation — tight ranges, low volatility,
                                         smart money builds positions quietly)
      LONDON_OPEN        07:00 - 08:00  (Manipulation — Judas Swing, false breakouts,
                                         stop hunts, traps retail before real move)
      LONDON_SESSION     08:00 - 12:00  (Expansion — sets the daily trend,
                                         strong directional moves)
      NY_LONDON_OVERLAP  12:00 - 16:00  (Distribution — highest liquidity,
                                         institutions exit, massive volume)
      NY_AFTERNOON       16:00 - 21:00  (Late Distribution — liquidation,
                                         high volatility, reversals or continuation)
    """
    from datetime import datetime, timezone
    hour = datetime.now(timezone.utc).hour

    if 21 <= hour < 24:
        return "SYDNEY"
    elif 0 <= hour < 7:
        return "TOKYO"
    elif 7 <= hour < 8:
        return "LONDON_OPEN"
    elif 8 <= hour < 12:
        return "LONDON_SESSION"
    elif 12 <= hour < 16:
        return "NY_LONDON_OVERLAP"
    else:  # 16-20
        return "NY_AFTERNOON"


def is_preferred_session() -> bool:
    """
    Returns True during high-probability trading windows.
    London + NY sessions (manipulation, expansion, distribution).
    """
    return get_session() in [
        "LONDON_OPEN",
        "LONDON_SESSION",
        "NY_LONDON_OVERLAP",
        "NY_AFTERNOON",
    ]


def is_tradeable_session() -> bool:
    """
    Returns True for ALL sessions — DEAD_ZONE block disabled for testing.
    """
    return True


def get_session_quality() -> float:
    """
    Returns a session quality multiplier (0.0 - 1.0).
    Used to boost or reduce confidence during different sessions.
    Based on real institutional behavior patterns.
    """
    quality_map = {
        "NY_LONDON_OVERLAP": 1.0,   # Distribution — highest liquidity, best window
        "LONDON_SESSION":    0.9,   # Expansion — strong directional moves
        "NY_AFTERNOON":      0.8,   # Late distribution — still good but fading
        "LONDON_OPEN":       0.7,   # Manipulation — volatile, Judas swings
        "TOKYO":             0.4,   # Accumulation — tight ranges, low volatility
        "SYDNEY":            0.3,   # Price discovery — thin liquidity
    }
    return quality_map.get(get_session(), 0.5)
