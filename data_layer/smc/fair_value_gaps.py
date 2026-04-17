# =============================================================
# data_layer/smc/fair_value_gaps.py
# PURPOSE: Detect Fair Value Gaps (FVG) — price imbalances
# where price moved so fast it left a gap between candles.
# Price is magnetically drawn back to fill these gaps.
# Bullish FVG = gap left during fast move up (unfilled demand)
# Bearish FVG = gap left during fast move down (unfilled supply)
# Run standalone to test.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
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
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def detect_fvg(df: pd.DataFrame,
               min_gap_pips: float = 2.0,
               pip_size: float = 0.0001,
               max_gaps: int = 5) -> dict:
    """
    Detect Fair Value Gaps from 3-candle patterns.

    Bullish FVG: Candle[i-1].high < Candle[i+1].low
    → Price gapped up leaving unfilled space below
    → Price will likely return to fill this gap

    Bearish FVG: Candle[i-1].low > Candle[i+1].high
    → Price gapped down leaving unfilled space above
    → Price will likely return to fill this gap

    min_gap_pips: Minimum gap size to qualify as FVG
    """
    df        = df.copy().reset_index(drop=True)
    bull_fvgs = []
    bear_fvgs = []

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt  = df.iloc[i + 1]

        # --- Bullish FVG ---
        # Gap between previous candle high and next candle low
        bull_gap = nxt['low'] - prev['high']
        if bull_gap > 0:
            gap_pips = bull_gap / pip_size
            if gap_pips >= min_gap_pips:
                bull_fvgs.append({
                    'type':      'BULLISH_FVG',
                    'time':      str(curr['time']),
                    'top':       round(float(nxt['low']),  5),
                    'bottom':    round(float(prev['high']), 5),
                    'mid':       round(float((nxt['low'] + prev['high']) / 2), 5),
                    'gap_pips':  round(gap_pips, 1),
                    'filled':    False,
                    'note':      'Unfilled demand zone — price may return to fill',
                })

        # --- Bearish FVG ---
        # Gap between previous candle low and next candle high
        bear_gap = prev['low'] - nxt['high']
        if bear_gap > 0:
            gap_pips = bear_gap / pip_size
            if gap_pips >= min_gap_pips:
                bear_fvgs.append({
                    'type':      'BEARISH_FVG',
                    'time':      str(curr['time']),
                    'top':       round(float(prev['low']),  5),
                    'bottom':    round(float(nxt['high']),  5),
                    'mid':       round(float((prev['low'] + nxt['high']) / 2), 5),
                    'gap_pips':  round(gap_pips, 1),
                    'filled':    False,
                    'note':      'Unfilled supply zone — price may return to fill',
                })

    return {
        'bullish_fvgs': bull_fvgs[-max_gaps:],
        'bearish_fvgs': bear_fvgs[-max_gaps:],
    }


def check_filled(fvgs: list, df: pd.DataFrame) -> list:
    """
    Mark FVGs as filled if price has traded back into the gap zone.
    Filled FVG = gap closed, no longer a magnet.
    Unfilled FVG = still active, price likely to return.
    """
    result     = []
    all_lows   = df['low'].values
    all_highs  = df['high'].values

    for fvg in fvgs:
        fvg = fvg.copy()
        if fvg['type'] == 'BULLISH_FVG':
            # Filled if price came back down into the gap
            fvg['filled'] = any(all_lows <= fvg['top'])
        elif fvg['type'] == 'BEARISH_FVG':
            # Filled if price came back up into the gap
            fvg['filled'] = any(all_highs >= fvg['bottom'])
        result.append(fvg)
    return result


def score_fvg_quality(fvg: dict, df: pd.DataFrame,
                      pip_size: float = 0.0001) -> dict:
    """
    Score FVG quality from 0-100.
    High quality FVG = clean, large, not partially filled.
    Low quality FVG  = tiny, messy, partially mitigated.
    Only high quality FVGs should be used as trade targets.
    """
    fvg   = fvg.copy()
    score = 0
    reasons = []

    # Size score (bigger gap = more imbalance = stronger)
    gap = fvg['gap_pips']
    if gap >= 10:
        score += 40; reasons.append(f"Large gap ({gap}p)")
    elif gap >= 5:
        score += 25; reasons.append(f"Medium gap ({gap}p)")
    else:
        score += 10; reasons.append(f"Small gap ({gap}p)")

    # Freshness — recent FVGs are more relevant
    total    = len(df)
    fvg_time = fvg.get('time', '')
    try:
        fvg_idx = df[df['time'].astype(str).str[:16] ==
                     fvg_time[:16]].index
        if len(fvg_idx) > 0:
            age = total - fvg_idx[0]
            if age <= 10:
                score += 30; reasons.append("Very recent")
            elif age <= 30:
                score += 20; reasons.append("Recent")
            else:
                score += 5;  reasons.append("Old")
    except Exception:
        score += 10

    # Not filled = still valid
    if not fvg.get('filled', True):
        score += 30; reasons.append("Still unfilled")

    fvg['quality_score']   = min(score, 100)
    fvg['quality_reasons'] = reasons
    fvg['high_quality']    = score >= 60
    return fvg


def get_quality_fvgs(fvgs: list, df: pd.DataFrame,
                     pip_size: float = 0.0001,
                     min_score: int = 60) -> list:
    """Return only high quality unfilled FVGs."""
    result = []
    for fvg in fvgs:
        if fvg.get('filled'):
            continue
        scored = score_fvg_quality(fvg, df, pip_size)
        if scored['quality_score'] >= min_score:
            result.append(scored)
    return sorted(result,
                  key=lambda x: x['quality_score'], reverse=True)



def get_nearest_fvg(fvgs: list, current_price: float) -> dict | None:
    """Return the closest unfilled FVG to current price."""
    active = [f for f in fvgs if not f['filled']]
    if not active:
        return None
    return min(active, key=lambda f: abs(current_price - f['mid']))


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"
    print(f"Detecting Fair Value Gaps for {SYMBOL} H1...\n")

    df = get_candles(SYMBOL, mt5.TIMEFRAME_H1, 200)
    if df is None:
        print("No data.")
        mt5.shutdown()
        exit()

    tick          = mt5.symbol_info_tick(SYMBOL)
    current_price = tick.bid

    # Get pip size (JPY and Gold need different value)
    sym_info = mt5.symbol_info(SYMBOL)
    pip_size = sym_info.point * 10 if sym_info else 0.0001

    fvgs = detect_fvg(df, min_gap_pips=2.0, pip_size=pip_size)

    # Check which are filled
    fvgs['bullish_fvgs'] = check_filled(fvgs['bullish_fvgs'], df)
    fvgs['bearish_fvgs'] = check_filled(fvgs['bearish_fvgs'], df)

    nearest = get_nearest_fvg(
        fvgs['bullish_fvgs'] + fvgs['bearish_fvgs'],
        current_price)

    print("=" * 58)
    print(f"  FAIR VALUE GAPS — {SYMBOL} H1")
    print(f"  Current Price : {current_price:.5f}")
    print("=" * 58)

    print(f"\n  BULLISH FVGs (unfilled demand — price may return):")
    if fvgs['bullish_fvgs']:
        for f in fvgs['bullish_fvgs']:
            status = "❌ FILLED" if f['filled'] else "✅ UNFILLED"
            print(f"  {status} | {f['time'][:16]}"
                  f" | Zone: {f['bottom']} — {f['top']}"
                  f" | Gap: {f['gap_pips']}p")
    else:
        print("  None detected.")

    print(f"\n  BEARISH FVGs (unfilled supply — price may return):")
    if fvgs['bearish_fvgs']:
        for f in fvgs['bearish_fvgs']:
            status = "❌ FILLED" if f['filled'] else "✅ UNFILLED"
            print(f"  {status} | {f['time'][:16]}"
                  f" | Zone: {f['bottom']} — {f['top']}"
                  f" | Gap: {f['gap_pips']}p")
    else:
        print("  None detected.")

    if nearest:
        print(f"\n  NEAREST UNFILLED FVG:")
        print(f"  Type   : {nearest['type']}")
        print(f"  Zone   : {nearest['bottom']} — {nearest['top']}")
        print(f"  Mid    : {nearest['mid']}")
        print(f"  Size   : {nearest['gap_pips']} pips")
        print(f"  Note   : {nearest['note']}")
    print("=" * 58)

    mt5.shutdown()
