# =============================================================
# rpde/feature_snapshot.py  — Feature Extraction for RPDE Snapshots
#
# PURPOSE: Extract the 93-engineered ML Gate features at each golden
# moment discovered by the scanner. This is the bridge between the
# RPDE reverse-engineering approach and the existing v4.2 feature
# engine.
#
# TWO MODES:
#   1. Historical snapshots (extract_snapshot_at_bar):
#      Given M5 candle data and a bar index, compute what the 93
#      features WOULD HAVE been at that point in time. Uses only
#      OHLCV data — no live MT5 tick connection needed.
#
#   2. Live snapshots (extract_snapshot_from_report):
#      Given a live master/market/SMC report, extract features using
#      the same logic as ai_engine.ml_gate.extract_features. This
#      enables real-time pattern matching against discovered patterns.
#
# FEATURE NAMES MUST match ai_engine/ml_gate.py FEATURE_NAMES exactly.
# Uses the same encoding maps (_BIAS_MAP, _SESSION_MAP, etc.).
# =============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from core.logger import get_logger
from core.pip_utils import get_pip_size

log = get_logger(__name__)

# ── Import feature names from ml_gate for validation ───────────
from ai_engine.ml_gate import FEATURE_NAMES, _SESSION_MAP, _BIAS_MAP, \
    _CONFIDENCE_MAP, _OF_STRENGTH_MAP, _PD_ZONE_MAP, _PRICE_POS_MAP, \
    _STATE_MAP, _TREND_MAP


# ════════════════════════════════════════════════════════════════
# SESSION COMPUTATION
# ════════════════════════════════════════════════════════════════

def compute_session(timestamp) -> str:
    """
    Determine which trading session a timestamp falls in.
    Aligned with config/settings.py SESSIONS definitions (UTC hours).

    Args:
        timestamp: datetime object (tz-aware or naive, treated as UTC)

    Returns:
        Session name string (e.g. 'LONDON_SESSION', 'NY_LONDON_OVERLAP')
    """
    try:
        # Normalize to UTC
        if isinstance(timestamp, pd.Timestamp):
            ts = timestamp
            if ts.tzinfo is not None:
                ts = ts.tz_convert('UTC')
        elif isinstance(timestamp, datetime):
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is not None:
                ts = ts.tz_convert('UTC')
        else:
            ts = pd.Timestamp(timestamp)

        hour = ts.hour

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
        else:
            return "NY_AFTERNOON"
    except Exception:
        return "UNKNOWN"


# ════════════════════════════════════════════════════════════════
# ATR COMPUTATION
# ════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute ATR(14) from the last row of an OHLC DataFrame.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)

    Returns:
        float: ATR value from the last bar, or 0.0 on error
    """
    if df is None or len(df) < period + 1:
        return 0.0

    try:
        h = df['high']
        l = df['low']
        c = df['close']

        hl = h - l
        hc = (h - c.shift()).abs()
        lc = (l - c.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(com=period - 1, min_periods=period).mean()

        return float(atr.iloc[-1])
    except Exception:
        return 0.0


# ════════════════════════════════════════════════════════════════
# TECHNICAL INDICATOR COMPUTATION
# ════════════════════════════════════════════════════════════════

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard technical indicators to M5 DataFrame.
    Mirrors backtest/data_loader.py _add_indicators() exactly.
    """
    c = df['close']
    h = df['high']
    l = df['low']

    # EMAs
    df['ema_9'] = c.ewm(span=9, adjust=False).mean()
    df['ema_21'] = c.ewm(span=21, adjust=False).mean()
    df['ema_50'] = c.ewm(span=50, adjust=False).mean()
    df['ema_200'] = c.ewm(span=200, adjust=False).mean()

    # RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = -delta.clip(upper=0).ewm(com=13, min_periods=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # ATR (14)
    hl_col = h - l
    hc_col = (h - c.shift()).abs()
    lc_col = (l - c.shift()).abs()
    tr = pd.concat([hl_col, hc_col, lc_col], axis=1).max(axis=1)
    df['atr'] = tr.ewm(com=13, min_periods=14).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands (20, 2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_mid'] = sma20

    # Volume MA
    df['vol_ma20'] = df['volume'].rolling(20).mean() if 'volume' in df.columns else 20.0

    # ADX (14)
    df['adx'] = _calc_adx(df, 14)

    return df


def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX calculation — same as price_feed / backtest/data_loader."""
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1
    ).max(axis=1)
    dm_p = (h - h.shift()).clip(lower=0).where(
        (h - h.shift()) > (l.shift() - l), 0)
    dm_m = (l.shift() - l).clip(lower=0).where(
        (l.shift() - l) > (h - h.shift()), 0)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    di_p = 100 * dm_p.ewm(com=period - 1, min_periods=period).mean() / atr
    di_m = 100 * dm_m.ewm(com=period - 1, min_periods=period).mean() / atr
    dx = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan))
    return dx.ewm(com=period - 1, min_periods=period).mean()


# ════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME RESAMPLING
# ════════════════════════════════════════════════════════════════

def _resample_tf(df_m5: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """
    Resample M5 DataFrame to a higher timeframe.

    Args:
        df_m5: M5 DataFrame with time, open, high, low, close, volume
        tf_name: 'M15', 'M30', 'H1', 'H4', or 'D1'

    Returns:
        Resampled DataFrame with OHLCV
    """
    tf_map = {
        'M15': '15min',
        'M30': '30min',
        'H1': '1h',
        'H4': '4h',
        'D1': '1D',
    }
    rule = tf_map.get(tf_name)
    if rule is None:
        return pd.DataFrame()

    try:
        df = df_m5.set_index('time')
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna().reset_index()
        resampled = resampled.rename(columns={'index': 'time'})
        return resampled
    except Exception as ex:
        log.debug(f"[RPDE_SNAPSHOT] Resample to {tf_name} failed: {ex}")
        return pd.DataFrame()


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI(14) from a price series. Returns last value."""
    if len(series) < period + 1:
        return 50.0  # Neutral default
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    return float(rsi.iloc[-1])


# ════════════════════════════════════════════════════════════════
# VWAP COMPUTATION (from candles only)
# ════════════════════════════════════════════════════════════════

def _compute_vwap_features(df: pd.DataFrame, pip_value: float,
                            current_price: float) -> dict:
    """
    Compute VWAP-related features from candle data.
    Mimics data_layer/vwap_calculator logic but uses only OHLCV.

    Returns dict with: pip_from_vwap, position, pip_to_poc, price_position
    """
    try:
        df = df.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['date'] = df['time'].dt.date

        # Daily VWAP
        df['tp_x_vol'] = df['typical_price'] * df['volume']
        df['cum_tp_vol'] = df.groupby('date')['tp_x_vol'].cumsum()
        df['cum_vol'] = df.groupby('date')['volume'].cumsum()

        vwap = np.where(
            df['cum_vol'] > 0,
            df['cum_tp_vol'] / df['cum_vol'],
            df['typical_price']
        )[-1]

        pip_from_vwap = round((current_price - vwap) / pip_value, 1) if pip_value > 0 else 0.0

        # Position relative to VWAP
        if current_price > vwap:
            position = "ABOVE"
        elif current_price < vwap:
            position = "BELOW"
        else:
            position = "AT"

        # POC approximation (price at max volume)
        # Find the bar with highest volume in recent history
        recent = df.tail(50)
        if len(recent) > 0:
            poc_idx = recent['volume'].idxmax()
            poc_price = df.loc[poc_idx, 'typical_price']
            pip_to_poc = round(abs(current_price - poc_price) / pip_value, 1) if pip_value > 0 else 50.0
        else:
            pip_to_poc = 50.0

        # Price position relative to value area (simplified)
        # Use BB bands as a proxy for value area
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            if pd.notna(bb_upper) and pd.notna(bb_lower):
                if current_price > bb_upper:
                    price_position = "ABOVE_VAH"
                elif current_price > (bb_upper + bb_lower) / 2:
                    price_position = "ABOVE_VA"
                elif current_price > (bb_upper + bb_lower) / 2:
                    price_position = "INSIDE_VA"
                elif current_price > bb_lower:
                    price_position = "BELOW_VA"
                else:
                    price_position = "BELOW_VAL"
            else:
                price_position = "INSIDE_VA"
        else:
            price_position = "INSIDE_VA"

        return {
            'pip_from_vwap': pip_from_vwap,
            'position': position,
            'pip_to_poc': pip_to_poc,
            'price_position': price_position,
        }
    except Exception as ex:
        log.debug(f"[RPDE_SNAPSHOT] VWAP computation failed: {ex}")
        return {
            'pip_from_vwap': 0.0,
            'position': 'AT',
            'pip_to_poc': 50.0,
            'price_position': 'INSIDE_VA',
        }


# ════════════════════════════════════════════════════════════════
# MARKET STATE COMPUTATION (from candle indicators)
# ════════════════════════════════════════════════════════════════

def _compute_market_state(df: pd.DataFrame) -> str:
    """
    Determine market state from candle-based indicators.
    Mirrors data_layer/market_regime logic.
    """
    try:
        if len(df) < 50:
            return 'BALANCED'

        last = df.iloc[-1]
        adx = last.get('adx', 0)
        atr = last.get('atr', 0)
        atr_avg = df['atr'].tail(20).mean() if 'atr' in df.columns else atr

        if pd.isna(adx) or pd.isna(atr):
            return 'BALANCED'

        # Volatile
        if atr_avg > 0 and atr > atr_avg * 1.8:
            return 'BREAKOUT_ACCEPTED'  # High volatility

        # Trending
        if adx > 25:
            ema9 = last.get('ema_9', 0)
            ema21 = last.get('ema_21', 0)
            ema50 = last.get('ema_50', 0)
            if pd.notna(ema9) and pd.notna(ema21) and pd.notna(ema50):
                if ema9 > ema21 > ema50:
                    return 'TRENDING_STRONG'
                elif ema9 < ema21 < ema50:
                    return 'REVERSAL_RISK'  # Strong bearish → reversal risk for longs
                elif adx > 35:
                    return 'TRENDING_EXTENDED'
            return 'BREAKOUT_ACCEPTED'

        # Ranging / choppy
        if adx < 20:
            return 'BALANCED'

        return 'BALANCED'
    except Exception:
        return 'BALANCED'


# ════════════════════════════════════════════════════════════════
# SMC STRUCTURE FEATURES (simplified from candles)
# ════════════════════════════════════════════════════════════════

def _compute_smc_features(df: pd.DataFrame, pip_value: float,
                           current_price: float) -> dict:
    """
    Compute simplified SMC structure features from candle data.
    These approximate the full SMC scanner results when tick data
    is unavailable.

    Returns dict with: trend, has_bos, bos_direction, pd_zone,
                       pips_to_eq, smc_bias, has_sweep, sweep_aligned
    """
    try:
        # Trend: EMA alignment
        if len(df) < 50:
            return _default_smc_features()

        last = df.iloc[-1]
        ema9 = last.get('ema_9', 0)
        ema21 = last.get('ema_21', 0)
        ema50 = last.get('ema_50', 0)

        if pd.isna(ema9) or pd.isna(ema21):
            return _default_smc_features()

        # Structure trend
        if ema9 > ema21 and ema21 > ema50:
            trend = 'BULLISH'
        elif ema9 < ema21 and ema21 < ema50:
            trend = 'BEARISH'
        else:
            trend = 'RANGING'

        # BOS detection: price crossing EMA21
        has_bos = False
        bos_direction = 0
        if len(df) >= 3:
            prev_close = df['close'].iloc[-3]
            prev_ema21 = df['ema_21'].iloc[-3] if 'ema_21' in df.columns else ema21
            curr_close = df['close'].iloc[-1]
            if pd.notna(prev_ema21):
                # Cross above EMA21
                if prev_close < prev_ema21 and curr_close > ema21:
                    has_bos = True
                    bos_direction = 1  # BULL
                # Cross below EMA21
                elif prev_close > prev_ema21 and curr_close < ema21:
                    has_bos = True
                    bos_direction = -1  # BEAR

        # Premium/Discount zone relative to EMA50
        pips_to_eq = 0.0
        pd_zone = 'NEUTRAL'
        if pd.notna(ema50) and pip_value > 0:
            pips_to_eq = round((current_price - ema50) / pip_value, 1)
            atr_val = last.get('atr', 0)
            atr_pips = atr_val / pip_value if (pip_value > 0 and atr_val > 0) else 20.0

            if pips_to_eq > atr_pips * 2:
                pd_zone = 'EXTREME_PREMIUM'
            elif pips_to_eq > atr_pips:
                pd_zone = 'PREMIUM'
            elif pips_to_eq < -atr_pips * 2:
                pd_zone = 'EXTREME_DISCOUNT'
            elif pips_to_eq < -atr_pips:
                pd_zone = 'DISCOUNT'
            else:
                pd_zone = 'NEUTRAL'

        # SMC bias from trend
        smc_bias = 'BULLISH' if trend == 'BULLISH' else \
                   'BEARISH' if trend == 'BEARISH' else 'NEUTRAL'

        # Sweep detection: price exceeded recent high/low then reversed
        has_sweep = False
        if len(df) >= 10:
            recent_high = df['high'].tail(10).max()
            recent_low = df['low'].tail(10).min()
            curr_high = df['high'].iloc[-1]
            curr_low = df['low'].iloc[-1]
            curr_close = df['close'].iloc[-1]
            # Sweep up: new high but closed below
            if curr_high > recent_high * 1.0001 and curr_close < recent_high:
                has_sweep = True
            # Sweep down: new low but closed above
            if curr_low < recent_low * 0.9999 and curr_close > recent_low:
                has_sweep = True

        # Sweep alignment
        sweep_aligned = 0
        if has_sweep and trend == 'BEARISH' and pd_zone in ('DISCOUNT', 'EXTREME_DISCOUNT'):
            sweep_aligned = 1  # Bullish sweep in discount zone during bearish = reversal signal
        elif has_sweep and trend == 'BULLISH' and pd_zone in ('PREMIUM', 'EXTREME_PREMIUM'):
            sweep_aligned = 1  # Bearish sweep in premium zone during bullish = reversal signal

        return {
            'trend': trend,
            'has_bos': has_bos,
            'bos_direction': bos_direction,
            'pd_zone': pd_zone,
            'pips_to_eq': pips_to_eq,
            'smc_bias': smc_bias,
            'has_sweep': has_sweep,
            'sweep_aligned': sweep_aligned,
        }
    except Exception as ex:
        log.debug(f"[RPDE_SNAPSHOT] SMC features failed: {ex}")
        return _default_smc_features()


def _default_smc_features() -> dict:
    """Return neutral SMC feature defaults."""
    return {
        'trend': 'RANGING',
        'has_bos': False,
        'bos_direction': 0,
        'pd_zone': 'NEUTRAL',
        'pips_to_eq': 0.0,
        'smc_bias': 'NEUTRAL',
        'has_sweep': False,
        'sweep_aligned': 0,
    }


# ════════════════════════════════════════════════════════════════
# MOMENTUM / CHOPPINESS FEATURES
# ════════════════════════════════════════════════════════════════

def _compute_momentum_features(df: pd.DataFrame, pip_value: float) -> dict:
    """
    Compute momentum velocity and choppiness from candle data.
    Uses the same logic as data_layer/momentum_velocity but from M5 candles.

    Returns dict with:
        - velocity_pips_min: float
        - is_choppy: bool
        - surge_ratio: float (volume surge ratio)
    """
    try:
        if len(df) < 10 or pip_value <= 0:
            return {'velocity_pips_min': 0.0, 'is_choppy': True, 'surge_ratio': 1.0}

        recent = df.tail(10)
        max_price = recent['high'].max()
        min_price = recent['low'].min()
        first_close = recent['close'].iloc[0]
        last_close = recent['close'].iloc[-1]

        price_range_pips = (max_price - min_price) / pip_value
        total_minutes = 10 * 5  # 10 M5 candles = 50 minutes
        velocity_pips_min = round(price_range_pips / total_minutes * 1.0, 2) if total_minutes > 0 else 0.0

        # Choppy: low velocity and range
        is_choppy = velocity_pips_min <= 0.5

        # Volume surge ratio
        if 'volume' in df.columns and 'vol_ma20' in df.columns:
            current_vol = df['volume'].iloc[-1]
            avg_vol = df['vol_ma20'].iloc[-1]
            surge_ratio = round(current_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        else:
            surge_ratio = 1.0

        return {
            'velocity_pips_min': velocity_pips_min,
            'is_choppy': is_choppy,
            'surge_ratio': surge_ratio,
        }
    except Exception:
        return {'velocity_pips_min': 0.0, 'is_choppy': True, 'surge_ratio': 1.0}


# ════════════════════════════════════════════════════════════════
# DELTA COMPUTATION (price direction proxy)
# ════════════════════════════════════════════════════════════════

def _compute_delta_features(df: pd.DataFrame) -> dict:
    """
    Compute delta (price direction) features from candle data.
    This is a CANDLE-BASED approximation of order flow delta.

    Returns dict with:
        - delta: float (cumulative direction score, last 20 bars)
        - rolling_delta: float (rolling delta, last 50 bars)
        - delta_bias: str
        - rd_bias: str
    """
    try:
        if len(df) < 50:
            return {'delta': 0.0, 'rolling_delta': 0.0, 'delta_bias': 'NEUTRAL', 'rd_bias': 'NEUTRAL'}

        # Delta = sum of (close > open ? +1 : -1) weighted by candle body size
        body = df['close'] - df['open']
        delta_short = body.tail(20).sum()
        delta_long = body.tail(50).sum()

        # Normalize by ATR
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else 1.0
        if atr > 0:
            delta_normalized = round(delta_short / atr, 2)
            rd_normalized = round(delta_long / atr, 2)
        else:
            delta_normalized = 0.0
            rd_normalized = 0.0

        # Bias classification
        delta_bias = 'BULLISH' if delta_normalized > 0.5 else \
                     'BEARISH' if delta_normalized < -0.5 else 'NEUTRAL'
        rd_bias = 'BULLISH' if rd_normalized > 1.0 else \
                  'BEARISH' if rd_normalized < -1.0 else 'NEUTRAL'

        return {
            'delta': delta_normalized,
            'rolling_delta': rd_normalized,
            'delta_bias': delta_bias,
            'rd_bias': rd_bias,
        }
    except Exception:
        return {'delta': 0.0, 'rolling_delta': 0.0, 'delta_bias': 'NEUTRAL', 'rd_bias': 'NEUTRAL'}


# ════════════════════════════════════════════════════════════════
# ATR PERCENTILE (from candle data)
# ════════════════════════════════════════════════════════════════

def _compute_atr_percentile(df: pd.DataFrame, period: int = 100) -> dict:
    """
    Compute ATR percentile from candle data.
    Mirrors data_layer/market_regime.calculate_atr_percentile logic.
    """
    try:
        atr_series = df['atr'].dropna() if 'atr' in df.columns else pd.Series()
        if len(atr_series) < period:
            return {'atr_percentile': 50.0, 'atr_ratio': 1.0}

        atr_window = atr_series.iloc[-period:]
        atr_current = float(atr_window.iloc[-1])
        atr_avg = float(atr_window.mean())

        below_count = int((atr_window < atr_current).sum())
        atr_percentile = round(below_count / len(atr_window) * 100, 1)
        atr_ratio = round(atr_current / atr_avg, 3) if atr_avg > 0 else 1.0

        return {'atr_percentile': atr_percentile, 'atr_ratio': atr_ratio}
    except Exception:
        return {'atr_percentile': 50.0, 'atr_ratio': 1.0}


# ════════════════════════════════════════════════════════════════
# MTF RSI COMPUTATION
# ════════════════════════════════════════════════════════════════

def _compute_mtf_rsi(df_m5: pd.DataFrame, bar_idx: int) -> dict:
    """
    Compute RSI across multiple timeframes using the M5 data
    available up to bar_idx.

    Returns dict with m5_rsi, m15_rsi, m30_rsi, h1_rsi, h4_rsi, d1_rsi
    """
    # Slice M5 data up to (and including) bar_idx
    df = df_m5.iloc[:bar_idx + 1].copy()

    result = {}

    # M5 RSI from the M5 data directly
    m5_close = df['close'].tail(50)
    result['m5_rsi'] = round(_compute_rsi(m5_close, 14), 1)

    # Resample to higher timeframes and compute RSI
    for tf_name in ['M15', 'M30', 'H1', 'H4', 'D1']:
        key = tf_name.lower().replace('h', 'h').replace('d', 'd') + '_rsi'
        # Map to expected feature names
        key = 'mr_' + tf_name.lower().replace('m', 'm').replace('h', 'h').replace('d', 'd') + '_rsi'

        try:
            tf_df = _resample_tf(df, tf_name)
            if len(tf_df) >= 20:
                result[key] = round(_compute_rsi(tf_df['close'], 14), 1)
            else:
                result[key] = 50.0
        except Exception:
            result[key] = 50.0

    return result


# ════════════════════════════════════════════════════════════════
# MTF CONTINUOUS SCORE
# ════════════════════════════════════════════════════════════════

def _compute_mtf_score(df_m5: pd.DataFrame, bar_idx: int) -> dict:
    """
    Compute multi-timeframe continuous alignment score.
    Measures how aligned trends are across timeframes.

    Returns dict with mtf_score, trend_agreement, rsi_agreement
    """
    try:
        df = df_m5.iloc[:bar_idx + 1].copy()
        if len(df) < 200:
            return {'mtf_score': 50.0, 'trend_agreement': 0.5, 'rsi_agreement': 0.0}

        # Trend direction per timeframe (using EMA alignment)
        tf_directions = []

        # M5 trend
        df = _add_indicators(df)
        m5 = df.tail(100)
        if len(m5) > 0 and pd.notna(m5['ema_9'].iloc[-1]) and pd.notna(m5['ema_21'].iloc[-1]):
            m5_trend = 1 if m5['ema_9'].iloc[-1] > m5['ema_21'].iloc[-1] else -1
            tf_directions.append(m5_trend)

        # Higher timeframes
        for tf_name in ['M15', 'H1', 'H4']:
            try:
                tf_df = _resample_tf(df, tf_name)
                if len(tf_df) >= 50:
                    tf_df = _add_indicators(tf_df)
                    if pd.notna(tf_df['ema_9'].iloc[-1]) and pd.notna(tf_df['ema_21'].iloc[-1]):
                        direction = 1 if tf_df['ema_9'].iloc[-1] > tf_df['ema_21'].iloc[-1] else -1
                        tf_directions.append(direction)
            except Exception:
                pass

        if len(tf_directions) < 2:
            return {'mtf_score': 50.0, 'trend_agreement': 0.5, 'rsi_agreement': 0.0}

        # Trend agreement: what fraction agree on direction
        bullish_count = sum(1 for d in tf_directions if d > 0)
        bearish_count = sum(1 for d in tf_directions if d < 0)
        dominant_count = max(bullish_count, bearish_count)
        trend_agreement = round(dominant_count / len(tf_directions), 2)

        # MTF score: 0-100 based on agreement
        mtf_score = round(trend_agreement * 100, 1)

        # RSI agreement: direction from RSI values
        mtf_rsi = _compute_mtf_rsi(df_m5, bar_idx)
        rsi_values = list(mtf_rsi.values())
        if len(rsi_values) >= 2:
            rsi_bullish = sum(1 for r in rsi_values if r > 50)
            rsi_bearish = sum(1 for r in rsi_values if r < 50)
            rsi_agreement = round((rsi_bullish - rsi_bearish) / len(rsi_values), 2)
        else:
            rsi_agreement = 0.0

        return {
            'mtf_score': mtf_score,
            'trend_agreement': trend_agreement,
            'rsi_agreement': rsi_agreement,
        }
    except Exception:
        return {'mtf_score': 50.0, 'trend_agreement': 0.5, 'rsi_agreement': 0.0}


# ════════════════════════════════════════════════════════════════
# HTF ALIGNMENT FEATURES
# ════════════════════════════════════════════════════════════════

def _compute_htf_features(df_m5: pd.DataFrame, bar_idx: int,
                           pip_value: float) -> dict:
    """
    Compute HTF alignment features.
    Returns dict with htf_approved (bool), htf_score (float)
    """
    try:
        df = df_m5.iloc[:bar_idx + 1].copy()
        if len(df) < 200:
            return {'htf_approved': False, 'htf_score': 50.0}

        # Check H4 trend alignment
        h4_df = _resample_tf(df, 'H4')
        if len(h4_df) < 20:
            return {'htf_approved': False, 'htf_score': 50.0}

        h4_df = _add_indicators(h4_df)
        last_h4 = h4_df.iloc[-1]

        if pd.notna(last_h4.get('ema_9')) and pd.notna(last_h4.get('ema_200')):
            htf_approved = last_h4['close'] > last_h4['ema_200']
            # Score based on how far above/below EMA200
            if pip_value > 0:
                dist_from_ema200 = (last_h4['close'] - last_h4['ema_200']) / pip_value
                htf_score = min(100, max(0, 50 + dist_from_ema200 * 2))
            else:
                htf_score = 50.0
        else:
            htf_approved = False
            htf_score = 50.0

        return {'htf_approved': htf_approved, 'htf_score': round(htf_score, 1)}
    except Exception:
        return {'htf_approved': False, 'htf_score': 50.0}


# ════════════════════════════════════════════════════════════════
# MARKET QUALITY SCORE (composite from available indicators)
# ════════════════════════════════════════════════════════════════

def _compute_market_quality(df: pd.DataFrame, bar_idx: int,
                             session: str) -> dict:
    """
    Compute a composite market quality score from available indicators.
    This approximates what the master scanner produces.

    Returns dict with:
        - final_score, market_score, smc_score
        - combined_bias, bias_confidence
    """
    try:
        last = df.iloc[bar_idx]

        # Market score components
        adx = last.get('adx', 0)
        rsi = last.get('rsi', 50)
        atr = last.get('atr', 0)
        ema9 = last.get('ema_9', 0)
        ema21 = last.get('ema_21', 0)

        # Simple scoring: ADX strength + trend direction + momentum
        market_score = 0.0
        if pd.notna(adx):
            market_score += min(30, adx)  # ADX up to 30
        if pd.notna(ema9) and pd.notna(ema21):
            # Trend alignment bonus
            if ema9 > ema21:
                market_score += 15
        if pd.notna(rsi):
            # RSI momentum bonus (neutral at 50)
            market_score += max(0, min(20, (rsi - 40) * 0.5))
        if pd.notna(atr):
            # Volatility bonus (more volatility = more opportunity)
            atr_avg = df['atr'].tail(20).mean() if 'atr' in df.columns else atr
            if atr_avg > 0 and pd.notna(atr_avg):
                vol_ratio = atr / atr_avg
                market_score += max(0, min(15, vol_ratio * 10))

        # Session quality bonus
        session_quality = {
            'NY_LONDON_OVERLAP': 15,
            'LONDON_SESSION': 12,
            'LONDON_OPEN': 8,
            'TOKYO': 5,
            'SYDNEY': 3,
            'NY_AFTERNOON': 5,
        }
        market_score += session_quality.get(session, 5)

        market_score = min(100, round(market_score, 1))

        # SMC score (simplified)
        smc_score = market_score * 0.8  # Generally lower than market score

        # Combined bias from EMA direction + RSI
        if pd.notna(ema9) and pd.notna(ema21):
            if ema9 > ema21 and (pd.notna(rsi) and rsi > 55):
                combined_bias = 'BULLISH'
            elif ema9 < ema21 and (pd.notna(rsi) and rsi < 45):
                combined_bias = 'BEARISH'
            else:
                combined_bias = 'NEUTRAL'
        else:
            combined_bias = 'NEUTRAL'

        # Bias confidence from ADX
        if pd.notna(adx):
            bias_confidence = 'HIGH' if adx > 30 else 'MODERATE' if adx > 20 else 'LOW'
        else:
            bias_confidence = 'LOW'

        # Final score = weighted average
        final_score = round(market_score * 0.6 + smc_score * 0.4, 1)

        return {
            'final_score': final_score,
            'market_score': market_score,
            'smc_score': smc_score,
            'combined_bias': combined_bias,
            'bias_confidence': bias_confidence,
        }
    except Exception:
        return {
            'final_score': 50.0,
            'market_score': 50.0,
            'smc_score': 40.0,
            'combined_bias': 'NEUTRAL',
            'bias_confidence': 'LOW',
        }


# ════════════════════════════════════════════════════════════════
# MAIN: Extract Historical Snapshot at Bar
# ════════════════════════════════════════════════════════════════

def extract_snapshot_at_bar(pair: str, bar_timestamp: datetime,
                             df_m5: pd.DataFrame) -> dict:
    """
    Extract all 93 features at a specific historical bar.

    This is the bridge between the RPDE scanner and the existing
    feature engine. It simulates what the feature engine would have
    computed at that point in time.

    IMPORTANT: df_m5 must be the FULL historical M5 data (with
    indicators pre-computed) so that bar_timestamp can be found
    within it. The snapshot will use only data UP TO bar_timestamp
    to prevent look-ahead bias.

    Args:
        pair: Symbol string (e.g. 'EURJPY')
        bar_timestamp: The exact timestamp of the bar to snapshot
        df_m5: Full M5 DataFrame with OHLCV data

    Returns:
        dict with all feature values, including:
            - All 93 ML Gate features (keyed by feature name)
            - '_meta': metadata (direction inferred, session, etc.)
            - '_direction_inferred': BUY/SELL based on bias
    """
    try:
        pip_value = get_pip_size(pair)

        # Find the bar index for bar_timestamp
        # Handle timezone-aware vs naive timestamps
        ts = pd.Timestamp(bar_timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC')

        # Match the timestamp in the DataFrame
        mask = df_m5['time'] == ts
        if mask.sum() == 0:
            # Try without timezone
            mask = df_m5['time'].dt.tz_localize(None) == ts.tz_localize(None)
            if mask.sum() == 0:
                # Try finding the closest bar
                if df_m5['time'].dt.tzinfo is not None:
                    mask = (df_m5['time'] - ts).abs() < pd.Timedelta(minutes=3)
                else:
                    ts_naive = ts.tz_localize(None) if ts.tzinfo else ts
                    mask = (df_m5['time'] - ts_naive).abs() < pd.Timedelta(minutes=3)

        if mask.sum() == 0:
            log.warning(f"[RPDE_SNAPSHOT] Bar not found for {pair} at {bar_timestamp}")
            return {}

        bar_idx = df_m5.index[mask][0]

        # Slice data UP TO (and including) bar_idx for look-ahead protection
        df = df_m5.iloc[:bar_idx + 1].copy()

        # Need at least 200 bars for indicators
        if len(df) < 200:
            log.debug(f"[RPDE_SNAPSHOT] Insufficient history at bar {bar_idx}: "
                      f"{len(df)} bars (need 200)")
            return {}

        # ── Add indicators if not already present ──
        if 'rsi' not in df.columns:
            df = _add_indicators(df)

        # ── Current bar values ──
        current = df.iloc[-1]
        current_price = float(current['close'])
        current_spread = float(current.get('spread', 0))

        # ── Compute session ──
        session = compute_session(current['time'])

        # ── Market quality features ──
        mq = _compute_market_quality(df, len(df) - 1, session)

        # ── Delta (candle-based proxy for order flow) ──
        delta_features = _compute_delta_features(df)

        # ── Order flow imbalance (candle-based proxy) ──
        # Use delta-based approximation since we don't have tick data
        of_imbalance_val = delta_features['delta']
        of_imbalance_strength = 'NONE'
        if abs(of_imbalance_val) > 2.0:
            of_imbalance_strength = 'EXTREME'
        elif abs(of_imbalance_val) > 1.0:
            of_imbalance_strength = 'STRONG'
        elif abs(of_imbalance_val) > 0.5:
            of_imbalance_strength = 'MODERATE'

        # ── VWAP features ──
        vwap_features = _compute_vwap_features(df, pip_value, current_price)

        # ── SMC features ──
        smc_features = _compute_smc_features(df, pip_value, current_price)

        # ── Momentum features ──
        mom_features = _compute_momentum_features(df, pip_value)

        # ── Market state ──
        market_state = _compute_market_state(df)

        # ── HTF features ──
        htf_features = _compute_htf_features(df_m5, bar_idx, pip_value)

        # ── ATR percentile ──
        atr_pct_features = _compute_atr_percentile(df)

        # ── MTF RSI ──
        mtf_rsi_features = _compute_mtf_rsi(df_m5, bar_idx)

        # ── MTF continuous score ──
        mtf_score_features = _compute_mtf_score(df_m5, bar_idx)

        # ── RSI at current bar ──
        current_rsi = float(current.get('rsi', 50)) if pd.notna(current.get('rsi', 50)) else 50.0
        current_atr = float(current.get('atr', 0)) if pd.notna(current.get('atr', 0)) else 0.0

        # ── Symbol type features ──
        sym_upper = pair.upper()
        is_jpy = 1.0 if 'JPY' in sym_upper else 0.0
        is_commodity = 1.0 if ('XAU' in sym_upper or 'XAG' in sym_upper) else 0.0
        is_index = 1.0 if any(x in sym_upper for x in
                               ['US30', 'US500', 'USTEC', 'JP225', 'DE30', 'UK100']) else 0.0

        # ── Session features ──
        sess_enc = _SESSION_MAP.get(session, 1)
        is_london_open = 1.0 if session == 'LONDON_OPEN' else 0.0
        is_overlap = 1.0 if session == 'NY_LONDON_OVERLAP' else 0.0
        is_ny_afternoon = 1.0 if session == 'NY_AFTERNOON' else 0.0
        vol_surge = 1.0 if mom_features.get('surge_ratio', 1.0) > 2.0 else 0.0

        # ── Direction inference (from combined bias) ──
        direction_inferred = 'BUY' if mq['combined_bias'] == 'BULLISH' else \
                             'SELL' if mq['combined_bias'] == 'BEARISH' else None

        # ══════════════════════════════════════════════════════
        # BUILD THE 93-FEATURE DICT
        # Feature names MUST match ai_engine/ml_gate.py FEATURE_NAMES
        # ══════════════════════════════════════════════════════
        features = {
            # ── Group 1: Market quality (7) ─────────────────
            'fq_final_score': mq['final_score'],
            'fq_market_score': mq['market_score'],
            'fq_smc_score': mq['smc_score'],
            'fq_combined_bias': _BIAS_MAP.get(mq['combined_bias'], 0.0),
            'fq_bias_confidence': _CONFIDENCE_MAP.get(mq['bias_confidence'], 1.0),
            'fq_htf_approved': 1.0 if htf_features['htf_approved'] else -1.0,
            'fq_htf_score': htf_features['htf_score'],

            # ── Group 2: Order flow (6) ─────────────────────
            # Candle-based proxy — no tick data available for historical scans
            'of_delta': delta_features['delta'],
            'of_rolling_delta': delta_features['rolling_delta'],
            'of_delta_bias': _BIAS_MAP.get(delta_features['delta_bias'], 0.0),
            'of_rd_bias': _BIAS_MAP.get(delta_features['rd_bias'], 0.0),
            'of_imbalance': of_imbalance_val,
            'of_imb_strength': _OF_STRENGTH_MAP.get(of_imbalance_strength, 0.0),

            # ── Group 3: VWAP (4) ──────────────────────────
            'vw_pip_from_vwap': vwap_features['pip_from_vwap'],
            'vw_position': 1.0 if vwap_features['position'] == 'ABOVE' else
                          -1.0 if vwap_features['position'] == 'BELOW' else 0.0,
            'vw_pip_to_poc': vwap_features['pip_to_poc'],
            'vw_price_position': _PRICE_POS_MAP.get(
                vwap_features['price_position'], 0.0),

            # ── Group 4: SMC structure (8) ─────────────────
            'smc_structure_trend': _TREND_MAP.get(smc_features['trend'], 0.0),
            'smc_has_bos': 1.0 if smc_features['has_bos'] else 0.0,
            'smc_bos_direction': smc_features['bos_direction'],
            'smc_pd_zone': _PD_ZONE_MAP.get(smc_features['pd_zone'], 0.0),
            'smc_pips_to_eq': smc_features['pips_to_eq'],
            'smc_smc_bias': _BIAS_MAP.get(smc_features['smc_bias'], 0.0),
            'smc_has_sweep': 1.0 if smc_features['has_sweep'] else 0.0,
            'smc_sweep_aligned': smc_features['sweep_aligned'],

            # ── Group 5: Trade parameters (5) ──────────────
            # For historical snapshots, use neutral defaults
            # (these are filled in by the strategy engine during live trading)
            'tp_score': 50.0,        # Neutral score
            'tp_sl_pips': 10.0,      # Default SL
            'tp_tp_pips': 20.0,      # Default TP
            'tp_rr_ratio': 2.0,      # Default R:R
            'tp_direction': 1.0 if direction_inferred == 'BUY' else
                          -1.0 if direction_inferred == 'SELL' else 0.0,

            # ── Group 6: Strategy scores (10) ──────────────
            # All zeros for historical snapshots — strategies are not evaluated
            # during pattern discovery (the scanner finds moves, not strategies)
            'ss_smc_ob': 0.0,
            'ss_liquidity_sweep': 0.0,
            'ss_delta_divergence': 0.0,
            'ss_trend_continuation': 0.0,
            'ss_ema_cross': 0.0,
            'ss_rsi_divergence': 0.0,
            'ss_sd_zone': 0.0,
            'ss_bos_momentum': 0.0,
            'ss_ote_fib': 0.0,
            'ss_inst_candles': 0.0,

            # ── Group 7: Consensus features (3) ────────────
            'cs_total_signals': 0.0,
            'cs_groups_agreeing': 0.0,
            'cs_direction_clear': 0.0,

            # ── Group 8: Session/Time (5) ──────────────────
            'st_session': sess_enc,
            'st_is_london_open': is_london_open,
            'st_is_overlap': is_overlap,
            'st_is_ny_afternoon': is_ny_afternoon,
            'st_vol_surge': vol_surge,

            # ── Group 9: Volatility/State (5) ─────────────
            'vs_atr': current_atr,
            'vs_market_state': _STATE_MAP.get(market_state, 0.0),
            'vs_surge_ratio': mom_features.get('surge_ratio', 1.0),
            'vs_momentum_velocity': mom_features.get('velocity_pips_min', 0.0),
            'vs_choppy': 1.0 if mom_features.get('is_choppy') else 0.0,

            # ── Group 10: Symbol type (3) ──────────────────
            'sym_is_jpy': is_jpy,
            'sym_is_commodity': is_commodity,
            'sym_is_index': is_index,

            # ── Group 11: Self-improvement (3) ─────────────
            # Neutral defaults — no live trading history for pattern discovery
            'si_recent_wr': 0.5,
            'si_recent_avg_r': 0.0,
            'si_strategy_wr': 0.5,

            # ── Group 12: Price context (1) ─────────────────
            'fx_spread_pips': current_spread,

            # ── Group 13: Fibonacci confluence (3) ──────────
            # Neutral defaults — no Fib data available for historical bars
            'fib_confluence_score': 0.0,
            'fib_in_golden_zone': 0.0,
            'fib_bias_aligned': 0.0,

            # ── Group 14: Historical pair-strategy perf (8) ──
            # Neutral defaults — no trade history for pattern discovery
            'hist_ps_avg_r_recent': 0.0,
            'hist_ps_wr_recent': 0.5,
            'hist_ps_trades_recent': 0.0,
            'hist_ps_avg_r_all': 0.0,
            'hist_ps_wr_all': 0.5,
            'hist_ps_trades_all': 0.0,
            'hist_ps_avg_r_decay': 0.0,
            'hist_ps_avg_r_trend': 0.0,

            # ── Group 15: Currency Strength (4) ─────────────
            # Neutral defaults — no live currency strength data
            'cs_base_strength': 0.0,
            'cs_quote_strength': 0.0,
            'cs_strength_delta': 0.0,
            'cs_pair_bias': 0.0,

            # ── Group 16: ATR Percentile (2) ───────────────
            'ap_atr_percentile': atr_pct_features['atr_percentile'],
            'ap_atr_ratio': atr_pct_features['atr_ratio'],

            # ── Group 17: MTF RSI (6) ─────────────────────
            'mr_m5_rsi': mtf_rsi_features.get('mr_m5_rsi', current_rsi),
            'mr_m15_rsi': mtf_rsi_features.get('mr_m15_rsi', 50.0),
            'mr_m30_rsi': mtf_rsi_features.get('mr_m30_rsi', 50.0),
            'mr_h1_rsi': mtf_rsi_features.get('mr_h1_rsi', 50.0),
            'mr_h4_rsi': mtf_rsi_features.get('mr_h4_rsi', 50.0),
            'mr_d1_rsi': mtf_rsi_features.get('mr_d1_rsi', 50.0),

            # ── Group 18: MTF Continuous Score (3) ─────────
            'mt_mtf_score': mtf_score_features['mtf_score'],
            'mt_trend_agreement': mtf_score_features['trend_agreement'],
            'mt_rsi_agreement': mtf_score_features['rsi_agreement'],

            # ── Group 19: Intermarket (3) ──────────────────
            # Neutral defaults — no live intermarket data for historical bars
            'im_vix': 20.0,
            'im_dxy_change': 0.0,
            'im_risk_env': 0.0,

            # ── Group 20: Streak (2) ───────────────────────
            # Neutral defaults — no streak data for pattern discovery
            'sk_current_streak': 0.0,
            'sk_recent_wr': 50.0,

            # ── Group 21: Z-Score (1) ──────────────────────
            'zs_signal_zscore': 0.0,

            # ── Group 22: Smart Money Score (1) ─────────────
            # Neutral default — no smart money footprint for historical bars
            'sm_footprint_score': 0.0,
        }

        # ── Validation: ensure we have exactly 93 features ──
        assert len(features) == 93, \
            f"Expected 93 features, got {len(features)}"

        # ── Verify all feature names match FEATURE_NAMES ──
        for fname in FEATURE_NAMES:
            if fname not in features:
                log.error(f"[RPDE_SNAPSHOT] Missing feature: {fname}")

        # ── Attach metadata ──
        features['_meta'] = {
            'pair': pair,
            'bar_time': current['time'],
            'bar_index': bar_idx,
            'session': session,
            'market_state': market_state,
            'atr': current_atr,
            'spread': current_spread,
            'direction_inferred': direction_inferred,
            'pip_value': pip_value,
            'source': 'rpde_historical',
        }

        return features

    except Exception as ex:
        log.error(f"[RPDE_SNAPSHOT] extract_snapshot_at_bar failed "
                  f"for {pair} at {bar_timestamp}: {ex}")
        return {}


def extract_snapshot_at_index(pair: str, bar_idx: int,
                               df_m5: pd.DataFrame) -> dict:
    """
    Extract all 93 features at a specific bar index (optimized for scanner).

    Same as extract_snapshot_at_bar but takes bar_idx directly instead
    of doing timestamp matching. The DataFrame should have indicators
    pre-computed (via _add_indicators) for best performance.

    Args:
        pair: Symbol string (e.g. 'EURJPY')
        bar_idx: The integer index of the bar in df_m5
        df_m5: Full M5 DataFrame (ideally with indicators pre-computed)

    Returns:
        dict with all 93 feature values keyed by feature name.
    """
    try:
        pip_value = get_pip_size(pair)

        if bar_idx < 200 or bar_idx >= len(df_m5):
            return {}

        # Slice data UP TO (and including) bar_idx for look-ahead protection
        df = df_m5.iloc[:bar_idx + 1].copy()

        # Add indicators if not already present
        if 'rsi' not in df.columns:
            df = _add_indicators(df)

        # ── Current bar values ──
        current = df.iloc[-1]
        current_price = float(current['close'])
        current_spread = float(current.get('spread', 0))

        # ── Compute session ──
        session = compute_session(current['time'])

        # ── Market quality features ──
        mq = _compute_market_quality(df, len(df) - 1, session)

        # ── Delta (candle-based proxy for order flow) ──
        delta_features = _compute_delta_features(df)

        # ── Order flow imbalance ──
        of_imbalance_val = delta_features['delta']
        of_imbalance_strength = 'NONE'
        if abs(of_imbalance_val) > 2.0:
            of_imbalance_strength = 'EXTREME'
        elif abs(of_imbalance_val) > 1.0:
            of_imbalance_strength = 'STRONG'
        elif abs(of_imbalance_val) > 0.5:
            of_imbalance_strength = 'MODERATE'

        # ── VWAP features ──
        vwap_features = _compute_vwap_features(df, pip_value, current_price)

        # ── SMC features ──
        smc_features = _compute_smc_features(df, pip_value, current_price)

        # ── Momentum features ──
        mom_features = _compute_momentum_features(df, pip_value)

        # ── Market state ──
        market_state = _compute_market_state(df)

        # ── HTF features (uses full df_m5, not sliced) ──
        htf_features = _compute_htf_features(df_m5, bar_idx, pip_value)

        # ── ATR percentile ──
        atr_pct_features = _compute_atr_percentile(df)

        # ── MTF RSI (uses full df_m5) ──
        mtf_rsi_features = _compute_mtf_rsi(df_m5, bar_idx)

        # ── MTF continuous score (uses full df_m5) ──
        mtf_score_features = _compute_mtf_score(df_m5, bar_idx)

        # ── RSI / ATR at current bar ──
        current_rsi = float(current.get('rsi', 50)) if pd.notna(current.get('rsi', 50)) else 50.0
        current_atr = float(current.get('atr', 0)) if pd.notna(current.get('atr', 0)) else 0.0

        # ── Symbol type features ──
        sym_upper = pair.upper()
        is_jpy = 1.0 if 'JPY' in sym_upper else 0.0
        is_commodity = 1.0 if ('XAU' in sym_upper or 'XAG' in sym_upper) else 0.0
        is_index = 1.0 if any(x in sym_upper for x in
                               ['US30', 'US500', 'USTEC', 'JP225', 'DE30', 'UK100']) else 0.0

        # ── Session features ──
        sess_enc = _SESSION_MAP.get(session, 1)
        is_london_open = 1.0 if session == 'LONDON_OPEN' else 0.0
        is_overlap = 1.0 if session == 'NY_LONDON_OVERLAP' else 0.0
        is_ny_afternoon = 1.0 if session == 'NY_AFTERNOON' else 0.0
        vol_surge = 1.0 if mom_features.get('surge_ratio', 1.0) > 2.0 else 0.0

        # ── Direction inference ──
        direction_inferred = 'BUY' if mq['combined_bias'] == 'BULLISH' else \
                             'SELL' if mq['combined_bias'] == 'BEARISH' else None

        # ═══ BUILD THE 93-FEATURE DICT ═══
        features = {
            # Group 1: Market quality (7)
            'fq_final_score': mq['final_score'],
            'fq_market_score': mq['market_score'],
            'fq_smc_score': mq['smc_score'],
            'fq_combined_bias': _BIAS_MAP.get(mq['combined_bias'], 0.0),
            'fq_bias_confidence': _CONFIDENCE_MAP.get(mq['bias_confidence'], 1.0),
            'fq_htf_approved': 1.0 if htf_features['htf_approved'] else -1.0,
            'fq_htf_score': htf_features['htf_score'],
            # Group 2: Order flow (6)
            'of_delta': delta_features['delta'],
            'of_rolling_delta': delta_features['rolling_delta'],
            'of_delta_bias': _BIAS_MAP.get(delta_features['delta_bias'], 0.0),
            'of_rd_bias': _BIAS_MAP.get(delta_features['rd_bias'], 0.0),
            'of_imbalance': of_imbalance_val,
            'of_imb_strength': _OF_STRENGTH_MAP.get(of_imbalance_strength, 0.0),
            # Group 3: VWAP (4)
            'vw_pip_from_vwap': vwap_features['pip_from_vwap'],
            'vw_position': 1.0 if vwap_features['position'] == 'ABOVE' else
                          -1.0 if vwap_features['position'] == 'BELOW' else 0.0,
            'vw_pip_to_poc': vwap_features['pip_to_poc'],
            'vw_price_position': _PRICE_POS_MAP.get(vwap_features['price_position'], 0.0),
            # Group 4: SMC structure (8)
            'smc_structure_trend': _TREND_MAP.get(smc_features['trend'], 0.0),
            'smc_has_bos': 1.0 if smc_features['has_bos'] else 0.0,
            'smc_bos_direction': smc_features['bos_direction'],
            'smc_pd_zone': _PD_ZONE_MAP.get(smc_features['pd_zone'], 0.0),
            'smc_pips_to_eq': smc_features['pips_to_eq'],
            'smc_smc_bias': _BIAS_MAP.get(smc_features['smc_bias'], 0.0),
            'smc_has_sweep': 1.0 if smc_features['has_sweep'] else 0.0,
            'smc_sweep_aligned': smc_features['sweep_aligned'],
            # Group 5: Trade parameters (5)
            'tp_score': 50.0,
            'tp_sl_pips': 10.0,
            'tp_tp_pips': 20.0,
            'tp_rr_ratio': 2.0,
            'tp_direction': 1.0 if direction_inferred == 'BUY' else
                          -1.0 if direction_inferred == 'SELL' else 0.0,
            # Group 6: Strategy scores (10)
            'ss_smc_ob': 0.0,
            'ss_liquidity_sweep': 0.0,
            'ss_delta_divergence': 0.0,
            'ss_trend_continuation': 0.0,
            'ss_ema_cross': 0.0,
            'ss_rsi_divergence': 0.0,
            'ss_sd_zone': 0.0,
            'ss_bos_momentum': 0.0,
            'ss_ote_fib': 0.0,
            'ss_inst_candles': 0.0,
            # Group 7: Consensus features (3)
            'cs_total_signals': 0.0,
            'cs_groups_agreeing': 0.0,
            'cs_direction_clear': 0.0,
            # Group 8: Session/Time (5)
            'st_session': sess_enc,
            'st_is_london_open': is_london_open,
            'st_is_overlap': is_overlap,
            'st_is_ny_afternoon': is_ny_afternoon,
            'st_vol_surge': vol_surge,
            # Group 9: Volatility/State (5)
            'vs_atr': current_atr,
            'vs_market_state': _STATE_MAP.get(market_state, 0.0),
            'vs_surge_ratio': mom_features.get('surge_ratio', 1.0),
            'vs_momentum_velocity': mom_features.get('velocity_pips_min', 0.0),
            'vs_choppy': 1.0 if mom_features.get('is_choppy') else 0.0,
            # Group 10: Symbol type (3)
            'sym_is_jpy': is_jpy,
            'sym_is_commodity': is_commodity,
            'sym_is_index': is_index,
            # Group 11: Self-improvement (3)
            'si_recent_wr': 0.5,
            'si_recent_avg_r': 0.0,
            'si_strategy_wr': 0.5,
            # Group 12: Price context (1)
            'fx_spread_pips': current_spread,
            # Group 13: Fibonacci confluence (3)
            'fib_confluence_score': 0.0,
            'fib_in_golden_zone': 0.0,
            'fib_bias_aligned': 0.0,
            # Group 14: Historical pair-strategy perf (8)
            'hist_ps_avg_r_recent': 0.0,
            'hist_ps_wr_recent': 0.5,
            'hist_ps_trades_recent': 0.0,
            'hist_ps_avg_r_all': 0.0,
            'hist_ps_wr_all': 0.5,
            'hist_ps_trades_all': 0.0,
            'hist_ps_avg_r_decay': 0.0,
            'hist_ps_avg_r_trend': 0.0,
            # Group 15: Currency Strength (4)
            'cs_base_strength': 0.0,
            'cs_quote_strength': 0.0,
            'cs_strength_delta': 0.0,
            'cs_pair_bias': 0.0,
            # Group 16: ATR Percentile (2)
            'ap_atr_percentile': atr_pct_features['atr_percentile'],
            'ap_atr_ratio': atr_pct_features['atr_ratio'],
            # Group 17: MTF RSI (6)
            'mr_m5_rsi': mtf_rsi_features.get('mr_m5_rsi', current_rsi),
            'mr_m15_rsi': mtf_rsi_features.get('mr_m15_rsi', 50.0),
            'mr_m30_rsi': mtf_rsi_features.get('mr_m30_rsi', 50.0),
            'mr_h1_rsi': mtf_rsi_features.get('mr_h1_rsi', 50.0),
            'mr_h4_rsi': mtf_rsi_features.get('mr_h4_rsi', 50.0),
            'mr_d1_rsi': mtf_rsi_features.get('mr_d1_rsi', 50.0),
            # Group 18: MTF Continuous Score (3)
            'mt_mtf_score': mtf_score_features['mtf_score'],
            'mt_trend_agreement': mtf_score_features['trend_agreement'],
            'mt_rsi_agreement': mtf_score_features['rsi_agreement'],
            # Group 19: Intermarket (3)
            'im_vix': 20.0,
            'im_dxy_change': 0.0,
            'im_risk_env': 0.0,
            # Group 20: Streak (2)
            'sk_current_streak': 0.0,
            'sk_recent_wr': 50.0,
            # Group 21: Z-Score (1)
            'zs_signal_zscore': 0.0,
            # Group 22: Smart Money Score (1)
            'sm_footprint_score': 0.0,
        }

        # Attach metadata
        features['_meta'] = {
            'pair': pair,
            'bar_time': current['time'],
            'bar_index': bar_idx,
            'session': session,
            'market_state': market_state,
            'atr': current_atr,
            'spread': current_spread,
            'direction_inferred': direction_inferred,
            'pip_value': pip_value,
            'source': 'rpde_historical',
        }

        return features

    except Exception as ex:
        log.debug(f"[RPDE_SNAPSHOT] extract_snapshot_at_index failed "
                  f"for {pair} at bar {bar_idx}: {ex}")
        return {}


# ════════════════════════════════════════════════════════════════
# LIVE SNAPSHOT: Extract from Master Report
# ════════════════════════════════════════════════════════════════

def extract_snapshot_from_report(pair: str, master_report: dict,
                                  market_report: dict, smc_report: dict,
                                  flow_data: dict, direction: str = None,
                                  spread_pips: float = 0.0) -> dict:
    """
    Extract features from a live/paper master report (for real-time pattern matching).
    Uses the same extraction logic as ai_engine.ml_gate.extract_features
    but returns a dict keyed by feature name instead of a numpy array.

    This function is the LIVE counterpart to extract_snapshot_at_bar.
    It's used when the bot is running and we want to compare the current
    market state against discovered patterns.

    Args:
        pair: Symbol string (e.g. 'EURJPY')
        master_report: Combined master report dict
        market_report: Market analysis report dict
        smc_report: SMC structure report dict
        flow_data: Order flow data dict
        direction: Optional direction override ('BUY' or 'SELL')
        spread_pips: Current spread in pips

    Returns:
        dict with all 93 feature values keyed by feature name
    """
    try:
        # Use the existing ml_gate extraction to get the feature array
        from ai_engine.ml_gate import extract_features

        # Build a minimal signal dict for extract_features
        signal = {
            'direction': direction or master_report.get('combined_bias', 'NEUTRAL'),
            'score': master_report.get('final_score', 50),
            'sl_pips': 10,
            'tp1_pips': 20,
            'tp_pips': 20,
            'fib_data': {},
        }

        # Extract features as numpy array
        feature_array = extract_features(
            signal=signal,
            master_report=master_report,
            market_report=market_report,
            smc_report=smc_report,
            flow_data=flow_data,
            symbol=pair,
            spread_pips=spread_pips,
        )

        if feature_array is None:
            log.error(f"[RPDE_SNAPSHOT] extract_features returned None for {pair}")
            return {}

        # Flatten to 1D
        flat = feature_array.flatten()

        # Map to feature names
        # FEATURE_NAMES has 93 entries, flat may have 96 (3 unnamed self-calibration)
        features = {}
        for i, fname in enumerate(FEATURE_NAMES):
            if i < len(flat):
                features[fname] = round(float(flat[i]), 4)
            else:
                features[fname] = 0.0

        # Attach metadata
        features['_meta'] = {
            'pair': pair,
            'session': master_report.get('session', 'UNKNOWN'),
            'market_state': master_report.get('market_state', 'BALANCED'),
            'atr': float((market_report or {}).get('atr', 0)),
            'spread': spread_pips,
            'direction': direction,
            'source': 'rpde_live',
        }

        return features

    except Exception as ex:
        log.error(f"[RPDE_SNAPSHOT] extract_snapshot_from_report failed "
                  f"for {pair}: {ex}")
        return {}


# ════════════════════════════════════════════════════════════════
# UTILITY: Convert feature dict to numpy array
# ════════════════════════════════════════════════════════════════

def snapshot_to_array(snapshot: dict) -> np.ndarray:
    """
    Convert a feature snapshot dict to a 1x96 numpy array
    (93 named features + 3 unnamed self-calibration = 96).

    Args:
        snapshot: dict from extract_snapshot_at_bar or extract_snapshot_from_report

    Returns:
        numpy array of shape (1, 96) or None on error
    """
    try:
        if not snapshot:
            return None

        values = []
        for fname in FEATURE_NAMES:
            values.append(float(snapshot.get(fname, 0.0)))

        # Append 3 unnamed self-calibration features (always 0.0 for snapshots)
        values.extend([0.0, 0.0, 0.0])

        return np.array(values, dtype=np.float32).reshape(1, -1)
    except Exception as ex:
        log.error(f"[RPDE_SNAPSHOT] snapshot_to_array failed: {ex}")
        return None
