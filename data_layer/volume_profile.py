# =============================================================
# data_layer/volume_profile.py
# PURPOSE: Calculate Volume Profile from MT5 candle data.
# Outputs: POC, VAH, VAL, HVN, LVN, price position context.
# Run this file standalone to test.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()


def connect():
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    if not mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER")
    ):
        print(f"Login failed: {mt5.last_error()}")
        return False
    print("Connected to MT5\n")
    return True


def get_candles(symbol: str, timeframe, count: int) -> pd.DataFrame | None:
    """Fetch OHLCV candles from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def build_volume_profile(df: pd.DataFrame, pip_size: float = 0.0001,
                         bins: int = 100) -> dict:
    """
    Build a volume profile from OHLCV candle data.
    Distributes each candle's tick_volume evenly across its high-low range.

    Returns a full profile object with all key levels.
    """
    if df is None or df.empty:
        return {}

    # Price range for the whole dataset
    price_high = df['high'].max()
    price_low  = df['low'].min()
    price_range = price_high - price_low

    if price_range == 0:
        return {}

    # Create price level buckets
    price_levels = np.linspace(price_low, price_high, bins)
    bucket_size  = price_levels[1] - price_levels[0]
    volume_at_price = np.zeros(bins)

    # Distribute each candle's volume across its high-low range
    for _, candle in df.iterrows():
        vol = candle['tick_volume']
        if vol == 0:
            continue
        # Find which buckets this candle's range covers
        low_idx  = int((candle['low']  - price_low) / bucket_size)
        high_idx = int((candle['high'] - price_low) / bucket_size)
        low_idx  = max(0, min(low_idx,  bins - 1))
        high_idx = max(0, min(high_idx, bins - 1))
        spread   = high_idx - low_idx + 1
        # Distribute volume evenly across the range
        volume_at_price[low_idx:high_idx + 1] += vol / spread

    return price_levels, volume_at_price, bucket_size


def calculate_value_area(price_levels, volume_at_price,
                         value_area_pct: float = 0.70) -> dict:
    """
    Calculate POC, VAH, VAL from a volume profile.
    Value Area = price range containing 70% of total volume.
    """
    total_volume = volume_at_price.sum()
    if total_volume == 0:
        return {}

    # POC = price level with the most volume
    poc_idx = int(np.argmax(volume_at_price))
    poc     = round(float(price_levels[poc_idx]), 5)

    # Value Area — expand outward from POC until 70% of volume captured
    target_volume = total_volume * value_area_pct
    captured      = volume_at_price[poc_idx]
    low_idx       = poc_idx
    high_idx      = poc_idx

    while captured < target_volume:
        can_go_up   = high_idx + 1 < len(price_levels)
        can_go_down = low_idx  - 1 >= 0

        vol_up   = volume_at_price[high_idx + 1] if can_go_up   else 0
        vol_down = volume_at_price[low_idx  - 1] if can_go_down else 0

        if not can_go_up and not can_go_down:
            break
        if vol_up >= vol_down:
            high_idx += 1
            captured += vol_up
        else:
            low_idx  -= 1
            captured += vol_down

    vah = round(float(price_levels[high_idx]), 5)
    val = round(float(price_levels[low_idx]),  5)

    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'poc_idx':  poc_idx,
        'high_idx': high_idx,
        'low_idx':  low_idx,
        'value_area_volume': round(float(captured), 2),
        'total_volume':      round(float(total_volume), 2),
    }


def find_hvn_lvn(price_levels, volume_at_price, top_n: int = 3) -> dict:
    """
    Find High Volume Nodes (HVN) and Low Volume Nodes (LVN).
    HVN = price levels with very high volume  → support/resistance magnets
    LVN = price levels with very low volume   → fast move zones
    """
    mean_vol = np.mean(volume_at_price)
    std_vol  = np.std(volume_at_price)

    hvn_threshold = mean_vol + (std_vol * 0.8)
    lvn_threshold = mean_vol - (std_vol * 0.5)
    lvn_threshold = max(lvn_threshold, 0)

    hvn_list = []
    lvn_list = []

    for i, (price, vol) in enumerate(zip(price_levels, volume_at_price)):
        if vol >= hvn_threshold:
            hvn_list.append(round(float(price), 5))
        elif vol <= lvn_threshold and vol > 0:
            lvn_list.append(round(float(price), 5))

    # Return only the strongest nodes
    hvn_sorted = sorted(hvn_list,
                        key=lambda p: volume_at_price[
                            np.argmin(np.abs(price_levels - p))],
                        reverse=True)[:top_n]
    lvn_sorted = lvn_list[:top_n]

    return {'hvn_list': hvn_sorted, 'lvn_list': lvn_sorted}


def get_price_position(current_price: float, poc: float,
                       vah: float, val: float,
                       pip_size: float = 0.0001) -> dict:
    """
    Tell the bot WHERE price is relative to the value area.
    pip_size passed in so Gold/JPY pairs calculate correctly.
    """
    if current_price > vah:
        position = "ABOVE_VAH"
        bias     = "BULLISH"
        note     = "Price above value — buyers in control"
    elif current_price < val:
        position = "BELOW_VAL"
        bias     = "BEARISH"
        note     = "Price below value — sellers in control"
    else:
        position = "INSIDE_VA"
        bias     = "NEUTRAL"
        note     = "Price inside value area — ranging"

    pip_to_poc = round(abs(current_price - poc) / pip_size, 1)
    pip_to_vah = round(abs(current_price - vah) / pip_size, 1)
    pip_to_val = round(abs(current_price - val) / pip_size, 1)

    return {
        'position':   position,
        'bias':       bias,
        'note':       note,
        'pip_to_poc': pip_to_poc,
        'pip_to_vah': pip_to_vah,
        'pip_to_val': pip_to_val,
    }


def get_full_profile(symbol: str, timeframe=None, candle_count: int = 200,
                     session_type: str = "SESSION", bins: int = 100) -> dict:
    """
    Master function — builds the complete volume profile object.
    This is what the bot calls every scan cycle.
    """
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M15

    df = get_candles(symbol, timeframe, candle_count)
    if df is None:
        return {}

    # Current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {}
    current_price = tick.bid

    # Pip size — 0.0001 for most pairs, 0.01 for JPY pairs
    sym_info = mt5.symbol_info(symbol)
    pip_size = sym_info.point * 10 if sym_info else 0.0001

    # Build profile
    price_levels, volume_at_price, bucket_size = build_volume_profile(
        df, pip_size=pip_size, bins=bins)

    # Key levels
    va      = calculate_value_area(price_levels, volume_at_price)
    nodes   = find_hvn_lvn(price_levels, volume_at_price)
    context = get_price_position(current_price,
                                 va['poc'], va['vah'], va['val'],
                                 pip_size=pip_size)

    va_width_pips = round((va['vah'] - va['val']) / pip_size, 1)

    return {
        'symbol':        symbol,
        'session_type':  session_type,
        'timestamp_range': {
            'from': str(df['time'].iloc[0]),
            'to':   str(df['time'].iloc[-1]),
        },
        'current_price':  round(current_price, 5),
        'poc':            va['poc'],
        'vah':            va['vah'],
        'val':            va['val'],
        'hvn_list':       nodes['hvn_list'],
        'lvn_list':       nodes['lvn_list'],
        'va_width_pips':  va_width_pips,
        'profile_width':  round(float(
            price_levels[-1] - price_levels[0]) / pip_size, 1),
        'value_area_volume': va['value_area_volume'],
        'total_volume':      va['total_volume'],
        'price_position': context['position'],
        'bias':           context['bias'],
        'pip_to_poc':     context['pip_to_poc'],
        'pip_to_vah':     context['pip_to_vah'],
        'pip_to_val':     context['pip_to_val'],
        'note':           context['note'],
    }


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"

    print(f"Building Volume Profile for {SYMBOL}...\n")
    profile = get_full_profile(
        symbol       = SYMBOL,
        timeframe    = mt5.TIMEFRAME_M15,
        candle_count = 200,
        session_type = "SESSION",
        bins         = 100
    )

    if profile:
        print("=" * 50)
        print(f"  VOLUME PROFILE — {profile['symbol']}")
        print(f"  Session : {profile['session_type']}")
        print(f"  Range   : {profile['timestamp_range']['from']}")
        print(f"          → {profile['timestamp_range']['to']}")
        print("=" * 50)
        print(f"  Current Price : {profile['current_price']}")
        print(f"  POC           : {profile['poc']}  ← Most traded price")
        print(f"  VAH           : {profile['vah']}  ← Value Area High")
        print(f"  VAL           : {profile['val']}  ← Value Area Low")
        print(f"  VA Width      : {profile['va_width_pips']} pips")
        print("-" * 50)
        print(f"  HVN (magnets) : {profile['hvn_list']}")
        print(f"  LVN (gaps)    : {profile['lvn_list']}")
        print("-" * 50)
        print(f"  Price Position: {profile['price_position']}")
        print(f"  Bias          : {profile['bias']}")
        print(f"  Note          : {profile['note']}")
        print("-" * 50)
        print(f"  Pips to POC   : {profile['pip_to_poc']}")
        print(f"  Pips to VAH   : {profile['pip_to_vah']}")
        print(f"  Pips to VAL   : {profile['pip_to_val']}")
        print("=" * 50)
    else:
        print("Failed to build profile.")

    mt5.shutdown()
