# =============================================================
# data_layer/market_regime.py
# Detects what kind of market we're in RIGHT NOW.
# Each strategy only activates in its optimal regime.
# Regimes: TRENDING_UP | TRENDING_DOWN | RANGING | VOLATILE
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
        log.info("[REGIME] 🌪️ VOLATILE — ATR spike detected")
        return "VOLATILE"

    # Trending: ADX > 25 confirms a real trend
    if adx > 25:
        bull_structure = ema9 > ema21 > ema50
        bear_structure = ema9 < ema21 < ema50
        
        # HTF Confirm if possible
        htf_bull = close_htf > ema200_htf if ema200_htf > 0 else True
        htf_bear = close_htf < ema200_htf if ema200_htf > 0 else True

        if bull_structure and htf_bull:
            log.info("[REGIME] 📈 TRENDING_UP — Strong bull structure")
            return "TRENDING_UP"
        if bear_structure and htf_bear:
            log.info("[REGIME] 📉 TRENDING_DOWN — Strong bear structure")
            return "TRENDING_DOWN"

    # Ranging: ADX < 20 = no directional conviction

    # Ranging: ADX < 20 = no directional conviction
    if adx < 20:
        log.info("[REGIME] ↔️ RANGING — Low ADX, choppy market")
        return "RANGING"

    # Weak trend — treat as ranging
    log.info("[REGIME] ↔️ RANGING — Weak trend, staying cautious")
    return "RANGING"


def get_session() -> str:
    """Return the current trading session based on UTC hour."""
    from datetime import datetime, timezone
    hour = datetime.now(timezone.utc).hour

    if 8 <= hour < 11:
        return "LONDON_KILLZONE"
    elif 12 <= hour < 16:
        return "NY_LONDON_OVERLAP"
    elif 13 <= hour < 17:
        return "NY_KILLZONE"
    elif 0 <= hour < 7:
        return "ASIAN"
    elif 20 <= hour < 24:
        return "DEAD_ZONE"
    else:
        return "TRANSITION"


def is_preferred_session() -> bool:
    """Returns True only during high-probability trading windows."""
    return get_session() in [
        "LONDON_KILLZONE", "NY_LONDON_OVERLAP", "NY_KILLZONE"
    ]
