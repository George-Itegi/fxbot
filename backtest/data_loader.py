# =============================================================
# backtest/data_loader.py
# Downloads historical OHLCV candles from MT5 and caches them.
# Calculates all technical indicators using the same logic as price_feed.py
# =============================================================

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from core.logger import get_logger

log = get_logger(__name__)

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,   "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,  "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,   "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,   "W1":  mt5.TIMEFRAME_W1,
}


def download_candles(symbol: str, timeframe: str,
                     start_date: datetime.datetime,
                     end_date: datetime.datetime) -> pd.DataFrame:
    """
    Download historical OHLCV candles from MT5.
    Returns DataFrame with time, open, high, low, close, tick_volume.
    """
    tf = TF_MAP.get(timeframe)
    if tf is None:
        log.error(f"Unknown timeframe: {timeframe}")
        return pd.DataFrame()

    request_start = start_date.replace(tzinfo=datetime.timezone.utc)
    request_end   = end_date.replace(hour=23, minute=59, second=59,
                                      tzinfo=datetime.timezone.utc)

    rates = mt5.copy_rates_range(symbol, tf, request_start, request_end)

    if rates is None or len(rates) == 0:
        log.warning(f"No data for {symbol} {timeframe} "
                     f"{start_date.date()} to {end_date.date()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df = df.drop_duplicates(subset='time').reset_index(drop=True)

    log.info(f"  Downloaded {symbol} {timeframe}: "
             f"{len(df)} bars ({start_date.date()} to {end_date.date()})")
    return df


def _tag_session(dt) -> str:
    """Tag each candle with the trading session."""
    try:
        hour = dt.hour
        if 21 <= hour < 24: return "SYDNEY"
        if 0  <= hour <  7:  return "TOKYO"
        if 7  <= hour <  8:  return "LONDON_OPEN"
        if 8  <= hour < 12:  return "LONDON_SESSION"
        if 12 <= hour < 16:  return "NY_LONDON_OVERLAP"
        if 16 <= hour < 21:  return "NY_AFTERNOON"
        return "SYDNEY"
    except Exception:
        return "UNKNOWN"


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ALL technical indicators to the DataFrame.
    This is a copy of price_feed._add_indicators() so the backtest
    uses the EXACT same indicator logic as live trading.
    """
    c = df['close']
    h = df['high']
    l = df['low']

    # EMAs
    df['ema_9']  = c.ewm(span=9,  adjust=False).mean()
    df['ema_21'] = c.ewm(span=21, adjust=False).mean()
    df['ema_50'] = c.ewm(span=50, adjust=False).mean()
    df['ema_200']= c.ewm(span=200,adjust=False).mean()

    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = -delta.clip(upper=0).ewm(com=13, min_periods=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # ATR (14)
    hl   = h - l
    hc   = (h - c.shift()).abs()
    lc   = (l - c.shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.ewm(com=13, min_periods=14).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # Bollinger Bands (20, 2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_mid']   = sma20

    # ADX (14)
    df['adx'] = _calc_adx(df, 14)

    # Volume MA
    df['vol_ma20'] = df['tick_volume'].rolling(20).mean()

    # Stochastic RSI (14, 3, 3)
    df = _add_stoch_rsi(df)

    # Supertrend (10, 3.0)
    df = _add_supertrend(df, period=10, multiplier=3.0)

    # Session tag
    df['session'] = df['time'].apply(_tag_session)

    # Drop NaN rows (first ~50 bars won't have all indicators)
    df = df.dropna().reset_index(drop=True)
    return df


def _add_stoch_rsi(df: pd.DataFrame, rsi_period=14,
                   stoch_period=14, smooth_k=3, smooth_d=3):
    """Stochastic RSI — same as price_feed."""
    rsi     = df['rsi']
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
    stoch_raw = 100 * (rsi - rsi_min) / rsi_range
    df['stoch_rsi_k'] = stoch_raw.rolling(smooth_k).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(smooth_d).mean()
    return df


def _add_supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """Supertrend — same as price_feed."""
    h   = df['high']
    l   = df['low']
    c   = df['close']
    hl2 = (h + l) / 2
    atr = df['atr']
    upper_b = hl2 + (multiplier * atr)
    lower_b = hl2 - (multiplier * atr)

    supertrend = pd.Series(np.nan, index=df.index)
    direction  = pd.Series(0,      index=df.index)

    supertrend.iloc[0] = upper_b.iloc[0]
    direction.iloc[0]  = -1

    for i in range(1, len(df)):
        prev_st  = supertrend.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        curr_c   = c.iloc[i]
        curr_ub  = upper_b.iloc[i]
        curr_lb  = lower_b.iloc[i]
        prev_ub  = upper_b.iloc[i - 1]
        prev_lb  = lower_b.iloc[i - 1]

        final_ub = min(curr_ub, prev_ub) if curr_c <= prev_st else curr_ub
        final_lb = max(curr_lb, prev_lb) if curr_c >= prev_st else curr_lb

        if prev_dir == -1:
            if curr_c > prev_st:
                direction.iloc[i]  = 1
                supertrend.iloc[i] = final_lb
            else:
                direction.iloc[i]  = -1
                supertrend.iloc[i] = final_ub
        else:
            if curr_c < prev_st:
                direction.iloc[i]  = -1
                supertrend.iloc[i] = final_ub
            else:
                direction.iloc[i]  = 1
                supertrend.iloc[i] = final_lb

    df['supertrend_val'] = supertrend.round(5)
    df['supertrend_dir'] = direction
    return df


def _calc_adx(df: pd.DataFrame, period=14):
    """ADX calculation — same as price_feed."""
    h, l, c = df['high'], df['low'], df['close']
    tr   = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()],
                     axis=1).max(axis=1)
    dm_p = (h - h.shift()).clip(lower=0).where(
        (h-h.shift()) > (l.shift()-l), 0)
    dm_m = (l.shift() - l).clip(lower=0).where(
        (l.shift()-l) > (h-h.shift()), 0)
    atr  = tr.ewm(com=period-1, min_periods=period).mean()
    di_p = 100 * dm_p.ewm(com=period-1, min_periods=period).mean() / atr
    di_m = 100 * dm_m.ewm(com=period-1, min_periods=period).mean() / atr
    dx   = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan))
    return dx.ewm(com=period-1, min_periods=period).mean()


def load_all_data(symbol: str,
                  start_date: datetime.datetime,
                  end_date: datetime.datetime,
                  cache_dir: str) -> dict:
    """
    Load (or download + cache) all timeframes for one symbol.
    Returns dict: {"M1": df, "M5": df, "M15": df, "H1": df, "H4": df}
    All DataFrames have indicators pre-calculated.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # M1 is the heaviest — always download fresh or use cache
    # H4/H1/M15/M5 are much smaller and quick

    result = {}
    for tf in ["M1", "M5", "M15", "H1", "H4"]:
        cache_file = os.path.join(cache_dir,
                                  f"{symbol}_{tf}_{start_date.date()}_{end_date.date()}.pkl")

        if os.path.exists(cache_file):
            log.info(f"  Loading {symbol} {tf} from cache...")
            with open(cache_file, 'rb') as f:
                result[tf] = pickle.load(f)
        else:
            df = download_candles(symbol, tf, start_date, end_date)
            if df.empty:
                log.error(f"  No data for {symbol} {tf} — cannot backtest")
                return {}
            log.info(f"  Calculating indicators on {symbol} {tf} ({len(df)} bars)...")
            df = _add_indicators(df)
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            result[tf] = df

    return result


def get_candles_at_time(data: dict, current_time: datetime.datetime,
                        count: int = 200) -> dict:
    """
    Slice historical data to only include bars UP TO current_time.
    This simulates what the live bot sees at any given moment.
    Returns dict of DataFrames, each ending before current_time.
    """
    cutoff = pd.Timestamp(current_time, tz='UTC')
    result = {}
    for tf, df in data.items():
        # Only include bars that closed before current_time
        mask = df['time'] <= cutoff
        sliced = df[mask].tail(count).copy().reset_index(drop=True)
        result[tf] = sliced
    return result
