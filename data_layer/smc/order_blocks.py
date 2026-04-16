# =============================================================
# data_layer/smc/order_blocks.py
# PURPOSE: Detect Order Blocks — price zones where institutions
# placed large orders. These become key support/resistance.
# Bullish OB = last bearish candle before a big bull move.
# Bearish OB = last bullish candle before a big bear move.
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


def detect_order_blocks(df: pd.DataFrame,
                        impulse_factor: float = 1.5,
                        max_blocks: int = 5) -> dict:
    """
    Detect bullish and bearish order blocks.

    Bullish OB: Last bearish (red) candle before price impulsively
                moves up. This is where institutions bought.
    Bearish OB: Last bullish (green) candle before price impulsively
                moves down. This is where institutions sold.

    impulse_factor: How many times larger the impulse candle must be
                    vs the average candle size to qualify.
    """
    df      = df.copy().reset_index(drop=True)
    avg_size = (df['high'] - df['low']).mean()

    bullish_obs = []
    bearish_obs = []

    for i in range(1, len(df) - 1):
        current   = df.iloc[i]
        next_c    = df.iloc[i + 1]
        cur_size  = current['high'] - current['low']
        next_size = next_c['high']  - next_c['low']

        # --- Bullish Order Block ---
        # Current candle is bearish (red), next candle is a big bull impulse
        is_bearish    = current['close'] < current['open']
        next_is_bull  = (next_c['close'] > next_c['open'] and
                         next_size > avg_size * impulse_factor)
        if is_bearish and next_is_bull:
            bullish_obs.append({
                'type':      'BULLISH_OB',
                'time':      str(current['time']),
                'top':       round(float(current['open']), 5),
                'bottom':    round(float(current['low']),  5),
                'mid':       round(float((current['open'] + current['low']) / 2), 5),
                'size_pips': round(float(current['open'] - current['low']) / 0.0001, 1),
                'mitigated': False,
                'note':      'Institution bought here — strong support zone',
            })

        # --- Bearish Order Block ---
        # Current candle is bullish (green), next candle is a big bear impulse
        is_bullish    = current['close'] > current['open']
        next_is_bear  = (next_c['close'] < next_c['open'] and
                         next_size > avg_size * impulse_factor)
        if is_bullish and next_is_bear:
            bearish_obs.append({
                'type':      'BEARISH_OB',
                'time':      str(current['time']),
                'top':       round(float(current['high']), 5),
                'bottom':    round(float(current['close']), 5),
                'mid':       round(float((current['high'] + current['close']) / 2), 5),
                'size_pips': round(float(current['high'] - current['close']) / 0.0001, 1),
                'mitigated': False,
                'note':      'Institution sold here — strong resistance zone',
            })

    return {
        'bullish_obs': bullish_obs[-max_blocks:],
        'bearish_obs': bearish_obs[-max_blocks:],
    }


def check_mitigation(obs: list, current_price: float) -> list:
    """
    Mark order blocks as mitigated if price has traded through them.
    Mitigated OB = already used up, no longer valid.
    Unmitigated OB = still active, still a valid level.
    """
    result = []
    for ob in obs:
        ob = ob.copy()
        if ob['type'] == 'BULLISH_OB':
            ob['mitigated'] = current_price < ob['bottom']
        elif ob['type'] == 'BEARISH_OB':
            ob['mitigated'] = current_price > ob['top']
        result.append(ob)
    return result


def get_nearest_ob(obs: list, current_price: float) -> dict | None:
    """Return the closest unmitigated order block to current price."""
    active = [ob for ob in obs if not ob['mitigated']]
    if not active:
        return None
    return min(active,
               key=lambda ob: abs(current_price - ob['mid']))


def detect_breaker_blocks(obs: list, current_price: float) -> list:
    """
    Detect Breaker Blocks — mitigated OBs that flip role.
    Bullish OB broken through downward → becomes BEARISH breaker.
    Bearish OB broken through upward  → becomes BULLISH breaker.
    These are high-quality S/R levels because institutions
    defended them once, failed, and now they flip role.
    """
    breakers = []
    for ob in obs:
        if ob['type'] == 'BULLISH_OB' and ob['mitigated']:
            # Bullish OB broken — now acts as resistance
            breakers.append({
                'type':    'BEARISH_BREAKER',
                'top':     ob['top'],
                'bottom':  ob['bottom'],
                'mid':     ob['mid'],
                'note':    'Failed bull OB — now resistance. Look for sells here.',
            })
        elif ob['type'] == 'BEARISH_OB' and ob['mitigated']:
            # Bearish OB broken — now acts as support
            breakers.append({
                'type':    'BULLISH_BREAKER',
                'top':     ob['top'],
                'bottom':  ob['bottom'],
                'mid':     ob['mid'],
                'note':    'Failed bear OB — now support. Look for buys here.',
            })
    return breakers


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"
    print(f"Detecting Order Blocks for {SYMBOL} H1...\n")

    df = get_candles(SYMBOL, mt5.TIMEFRAME_H1, 200)
    if df is None:
        print("No data.")
        mt5.shutdown()
        exit()

    tick          = mt5.symbol_info_tick(SYMBOL)
    current_price = tick.bid
    blocks        = detect_order_blocks(df)

    # Check mitigation
    blocks['bullish_obs'] = check_mitigation(
        blocks['bullish_obs'], current_price)
    blocks['bearish_obs'] = check_mitigation(
        blocks['bearish_obs'], current_price)

    nearest = get_nearest_ob(
        blocks['bullish_obs'] + blocks['bearish_obs'], current_price)

    print("=" * 55)
    print(f"  ORDER BLOCKS — {SYMBOL} H1")
    print(f"  Current Price : {current_price:.5f}")
    print("=" * 55)

    print(f"\n  BULLISH Order Blocks (Institutional Buy Zones):")
    for ob in blocks['bullish_obs']:
        status = "❌ MITIGATED" if ob['mitigated'] else "✅ ACTIVE"
        print(f"  {status} | {ob['time'][:16]}"
              f" | Top: {ob['top']} | Bot: {ob['bottom']}"
              f" | {ob['size_pips']}p")

    print(f"\n  BEARISH Order Blocks (Institutional Sell Zones):")
    for ob in blocks['bearish_obs']:
        status = "❌ MITIGATED" if ob['mitigated'] else "✅ ACTIVE"
        print(f"  {status} | {ob['time'][:16]}"
              f" | Top: {ob['top']} | Bot: {ob['bottom']}"
              f" | {ob['size_pips']}p")

    if nearest:
        print(f"\n  NEAREST ACTIVE OB:")
        print(f"  Type   : {nearest['type']}")
        print(f"  Zone   : {nearest['bottom']} — {nearest['top']}")
        print(f"  Mid    : {nearest['mid']}")
        print(f"  Note   : {nearest['note']}")

    # --- Breaker Blocks ---
    breakers = detect_breaker_blocks(
        blocks['bullish_obs'] + blocks['bearish_obs'], current_price)
    print(f"\n  BREAKER BLOCKS (flipped OBs):")
    if breakers:
        for b in breakers:
            print(f"  {b['type']} | Zone: {b['bottom']} — {b['top']}")
            print(f"  Note: {b['note']}")
    else:
        print("  None detected.")
    print("=" * 55)

    mt5.shutdown()
