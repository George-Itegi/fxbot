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
    Returns True ONLY during whitelisted sessions (v2.0).
    NY_AFTERNOON and SYDNEY are HARD BLOCKED — backtest proven negative.
    Tokyo is NOT in whitelist (marginal edge, monitoring).
    """
    from config.settings import SESSION_WHITELIST
    return get_session() in SESSION_WHITELIST


def is_tradeable_session() -> bool:
    """
    Returns True ONLY during whitelisted sessions (v2.0).
    Hard blocks: NY_AFTERNOON (-$2,607), SYDNEY (-$3,361).
    This is the master gate — ALL trading paths must check this.
    """
    from config.settings import SESSION_WHITELIST
    return get_session() in SESSION_WHITELIST


def get_session_quality() -> float:
    """
    Returns a session quality multiplier (0.0 - 1.0).
    v2.0: Blocked sessions return 0.0 (absolute block).
    Only whitelisted sessions get positive quality.
    """
    from config.settings import SESSION_WHITELIST, SESSION_BLACKLIST

    session = get_session()

    # Hard block: blacklisted sessions = 0.0
    if session in SESSION_BLACKLIST:
        return 0.0

    # Only whitelisted sessions get quality scores
    quality_map = {
        "NY_LONDON_OVERLAP": 1.0,   # Distribution — highest liquidity, best window
        "LONDON_SESSION":    0.9,   # Expansion — strong directional moves
        "LONDON_OPEN":       0.7,   # Manipulation — volatile, Judas swings
    }
    return quality_map.get(session, 0.0)


# ════════════════════════════════════════════════════════════════
# ATR PERCENTILE CALCULATOR
# Where does the current ATR sit vs recent history?
# ════════════════════════════════════════════════════════════════

def calculate_atr_percentile(symbol: str, timeframe: str = 'H1',
                             period: int = 100) -> dict:
    """
    Calculate ATR percentile — where current ATR ranks vs last N periods.

    Fetches candles, computes ATR(14) for the whole series, then ranks
    the current ATR value within the last `period` ATR values.

    Args:
        symbol:    Forex pair symbol (e.g. 'EURJPY')
        timeframe: MT5 timeframe string (e.g. 'H1', 'M15')
        period:    Number of ATR periods to compare against

    Returns:
        dict with:
            - atr_current:  float  (current ATR value)
            - atr_percentile: float (0-100, where current ranks)
            - atr_avg:      float  (average ATR over lookback)
            - atr_ratio:    float  (current / average)
            - volatility_state: str ('EXTREME'|'HIGH'|'NORMAL'|'LOW'|'DEAD')
    """
    # Safe defaults
    _default = {
        'atr_current': 0.0,
        'atr_percentile': 50.0,
        'atr_avg': 0.0,
        'atr_ratio': 1.0,
        'volatility_state': 'NORMAL',
    }

    from data_layer.price_feed import get_candles

    # Fetch enough bars for ATR warmup + period comparison
    bars_needed = period + 60  # 60 extra for ATR(14) warmup
    df = get_candles(symbol, timeframe, bars_needed)

    if df is None or len(df) < period + 20:
        log.warning(f"[ATR_PERCENTILE] Insufficient data for {symbol} {timeframe}")
        return _default

    try:
        # Use the ATR already computed by price_feed._add_indicators()
        atr_series = df['atr'].dropna()

        if len(atr_series) < period:
            log.warning(f"[ATR_PERCENTILE] Only {len(atr_series)} ATR values "
                        f"(need {period}) for {symbol}")
            return _default

        # Take the last `period` values for comparison
        atr_window = atr_series.iloc[-period:]
        atr_current = float(atr_window.iloc[-1])
        atr_avg = float(atr_window.mean())

        # Percentile ranking: what % of values is current ATR above?
        # percentile = (values below current) / total * 100
        below_count = int((atr_window < atr_current).sum())
        atr_percentile = round(below_count / len(atr_window) * 100, 1)

        # Ratio of current to average
        atr_ratio = round(atr_current / atr_avg, 3) if atr_avg > 0 else 1.0

        # Volatility state classification
        if atr_percentile > 80:
            volatility_state = 'EXTREME'
        elif atr_percentile > 60:
            volatility_state = 'HIGH'
        elif atr_percentile > 40:
            volatility_state = 'NORMAL'
        elif atr_percentile > 20:
            volatility_state = 'LOW'
        else:
            volatility_state = 'DEAD'

        result = {
            'atr_current': round(atr_current, 6),
            'atr_percentile': atr_percentile,
            'atr_avg': round(atr_avg, 6),
            'atr_ratio': atr_ratio,
            'volatility_state': volatility_state,
        }

        log.debug(f"[ATR_PERCENTILE] {symbol} {timeframe}: "
                  f"ATR={atr_current:.6f} pctl={atr_percentile:.1f}% "
                  f"ratio={atr_ratio:.3f} state={volatility_state}")

        return result

    except Exception as e:
        log.error(f"[ATR_PERCENTILE] Error for {symbol} {timeframe}: {e}")
        return _default
