# =============================================================
# data_layer/mtf_analysis.py
# Multi-Timeframe RSI & Alignment Module
#
# Computes continuous multi-timeframe confluence scores to replace
# the binary htf_approved flag. Checks EMA alignment, RSI agreement,
# and price vs EMA200 across M15/M30/H1/H4/D1 timeframes.
#
# Two main functions:
#   - calculate_mtf_rsi():      RSI(14) across 6 timeframes
#   - calculate_mtf_continuous_score(): 0-100 confluence score
# =============================================================

from data_layer.price_feed import get_candles
from core.logger import get_logger

log = get_logger(__name__)

# Timeframes used for multi-timeframe analysis (ascending order)
MTF_TIMEFRAMES = ['M15', 'M30', 'H1', 'H4', 'D1']

# Full set including M5 for RSI
RSI_TIMEFRAMES = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']

# EMA keys available on candles from price_feed
EMA_KEYS = ['ema_9', 'ema_21', 'ema_50']

# Safe default returns
_MTF_RSI_DEFAULT = {
    'm5_rsi': 50.0, 'm15_rsi': 50.0, 'm30_rsi': 50.0,
    'h1_rsi': 50.0, 'h4_rsi': 50.0, 'd1_rsi': 50.0,
    'rsi_alignment': 0.0, 'rsi_avg': 50.0,
}

_MTF_SCORE_DEFAULT = {
    'mtf_score': 50.0,
    'mtf_trend_agreement': 0.0,
    'mtf_rsi_agreement': 0.0,
    'tf_bullish_count': 0,
    'tf_bearish_count': 0,
    'mtf_verdict': 'NEUTRAL',
}


def calculate_mtf_rsi(symbol: str) -> dict:
    """
    Get RSI(14) from multiple timeframes.

    Fetches candles for each timeframe and extracts the latest RSI(14)
    value. Computes an alignment metric: how many timeframes agree
    on direction (>50 bullish, <50 bearish).

    Args:
        symbol: Forex pair symbol (e.g. 'EURJPY')

    Returns:
        dict with:
            - m5_rsi, m15_rsi, m30_rsi, h1_rsi, h4_rsi, d1_rsi: float (0-100)
            - rsi_alignment: float (-1 to +1, +1 = all bullish, -1 = all bearish)
            - rsi_avg: float (average RSI across all timeframes)
    """
    result = dict(_MTF_RSI_DEFAULT)
    rsi_values = []

    # Mapping of timeframe strings to result keys
    tf_key_map = {
        'M5': 'm5_rsi', 'M15': 'm15_rsi', 'M30': 'm30_rsi',
        'H1': 'h1_rsi', 'H4': 'h4_rsi', 'D1': 'd1_rsi',
    }

    for tf in RSI_TIMEFRAMES:
        key = tf_key_map.get(tf)
        if not key:
            continue

        # Fetch candles — 50 bars enough for RSI(14) warmup
        bars = 60 if tf != 'M5' else 30
        df = get_candles(symbol, tf, bars)

        if df is not None and not df.empty and 'rsi' in df.columns:
            last_rsi = float(df['rsi'].iloc[-1])
            if not (last_rsi != last_rsi):  # Check for NaN
                result[key] = round(last_rsi, 1)
                rsi_values.append(last_rsi)

    if rsi_values:
        # Average RSI
        result['rsi_avg'] = round(sum(rsi_values) / len(rsi_values), 1)

        # Alignment: +1 = all > 50 (bullish), -1 = all < 50 (bearish)
        bullish_count = sum(1 for r in rsi_values if r > 50)
        bearish_count = sum(1 for r in rsi_values if r < 50)
        total = len(rsi_values)

        # Continuous alignment: (bullish - bearish) / total
        result['rsi_alignment'] = round((bullish_count - bearish_count) / total, 2)

    return result


def calculate_mtf_continuous_score(symbol: str,
                                    entry_bias: str = None) -> dict:
    """
    Continuous multi-timeframe confluence score replacing binary htf_approved.

    For each timeframe (M15/M30/H1/H4/D1), evaluates:
      1. EMA stack direction (EMA9 > EMA21 > EMA50 = bullish)
      2. RSI agreement (>50 bullish, <50 bearish)
      3. Price vs EMA200 (if available, H4/D1 only)

    Produces a continuous 0-100 score based on timeframe agreement.

    Args:
        symbol:      Forex pair symbol (e.g. 'EURJPY')
        entry_bias:  Optional preferred direction ('BUY' or 'SELL').
                     If provided, score is boosted when MTF agrees.

    Returns:
        dict with:
            - mtf_score:           float (0-100, higher = stronger confluence)
            - mtf_trend_agreement: float (0-1, how many TFs agree on trend)
            - mtf_rsi_agreement:   float (-1 to +1, RSI direction alignment)
            - tf_bullish_count:    int (0-5, number of bullish timeframes)
            - tf_bearish_count:    int (0-5, number of bearish timeframes)
            - mtf_verdict:         str ('STRONG_BULL'|'BULL'|'NEUTRAL'|'BEAR'|'STRONG_BEAR')
    """
    result = dict(_MTF_SCORE_DEFAULT)

    tf_scores = []    # Per-TF score: +1 bullish, -1 bearish, 0 neutral
    bullish_count = 0
    bearish_count = 0
    rsi_bullish = 0
    rsi_bearish = 0

    for tf in MTF_TIMEFRAMES:
        # More bars for higher timeframes (need EMA200 on H4/D1)
        bars = 250 if tf in ('H4', 'D1') else 80
        df = get_candles(symbol, tf, bars)

        if df is None or df.empty:
            tf_scores.append(0.0)
            continue

        last = df.iloc[-1]
        tf_score = 0.0

        # --- Check 1: EMA stack alignment ---
        ema9 = float(last.get('ema_9', 0))
        ema21 = float(last.get('ema_21', 0))
        ema50 = float(last.get('ema_50', 0))

        bullish_ema = ema9 > ema21 > ema50
        bearish_ema = ema9 < ema21 < ema50

        # --- Check 2: RSI agreement ---
        rsi = float(last.get('rsi', 50))
        rsi_bull = rsi > 55  # Slightly above 50 for conviction
        rsi_bear = rsi < 45

        if rsi > 50:
            rsi_bullish += 1
        elif rsi < 50:
            rsi_bearish += 1

        # --- Check 3: Price vs EMA200 (H4/D1 only) ---
        htf_bull = True  # Default: no opinion
        if tf in ('H4', 'D1') and 'ema_200' in df.columns:
            ema200 = float(last.get('ema_200', 0))
            close = float(last['close'])
            if ema200 > 0:
                htf_bull = close > ema200

        # --- Combine checks for this TF ---
        # Full bull: EMA stack + RSI bull + above EMA200
        if bullish_ema and rsi_bull and htf_bull:
            tf_score = 1.0
            bullish_count += 1
        elif bullish_ema and rsi_bull:
            tf_score = 0.7
            bullish_count += 1
        elif bearish_ema and not rsi_bear and htf_bull:
            tf_score = 0.0
        # Full bear: EMA stack + RSI bear + below EMA200
        elif bearish_ema and rsi_bear and not htf_bull:
            tf_score = -1.0
            bearish_count += 1
        elif bearish_ema and rsi_bear:
            tf_score = -0.7
            bearish_count += 1
        elif bearish_ema and not rsi_bull and not htf_bull:
            tf_score = 0.0
        else:
            tf_score = 0.0  # Mixed/no clear signal

        tf_scores.append(tf_score)

    # --- Aggregate across all timeframes ---
    total_tfs = len(MTF_TIMEFRAMES)
    if total_tfs == 0:
        return result

    # Raw average score (-1 to +1)
    raw_avg = sum(tf_scores) / total_tfs

    # Trend agreement: fraction of TFs that agree on a direction
    agreeing = max(bullish_count, bearish_count)
    trend_agreement = round(agreeing / total_tfs, 2)

    # RSI agreement
    total_rsi = rsi_bullish + rsi_bearish
    if total_rsi > 0:
        rsi_agreement = round((rsi_bullish - rsi_bearish) / total_rsi, 2)
    else:
        rsi_agreement = 0.0

    # Convert raw_avg (-1 to +1) to score (0 to 100)
    # Center at 50, scale by agreement strength
    mtf_score = round(50 + raw_avg * 50, 1)
    mtf_score = max(0.0, min(100.0, mtf_score))

    # Boost score if entry_bias matches MTF direction
    if entry_bias == 'BUY' and bullish_count >= 3:
        boost = min(10.0, bullish_count * 2.0)
        mtf_score = round(min(100.0, mtf_score + boost), 1)
    elif entry_bias == 'SELL' and bearish_count >= 3:
        boost = min(10.0, bearish_count * 2.0)
        mtf_score = round(max(0.0, mtf_score - boost), 1)

    # Verdict
    if mtf_score >= 80 and bullish_count >= 4:
        verdict = 'STRONG_BULL'
    elif mtf_score >= 60 and bullish_count >= 3:
        verdict = 'BULL'
    elif mtf_score <= 20 and bearish_count >= 4:
        verdict = 'STRONG_BEAR'
    elif mtf_score <= 40 and bearish_count >= 3:
        verdict = 'BEAR'
    else:
        verdict = 'NEUTRAL'

    result = {
        'mtf_score': mtf_score,
        'mtf_trend_agreement': trend_agreement,
        'mtf_rsi_agreement': rsi_agreement,
        'tf_bullish_count': bullish_count,
        'tf_bearish_count': bearish_count,
        'mtf_verdict': verdict,
    }

    log.debug(f"[MTF_ANALYSIS] {symbol}: score={mtf_score:.1f} "
              f"bull={bullish_count}/{total_tfs} bear={bearish_count}/{total_tfs} "
              f"verdict={verdict}")

    return result
