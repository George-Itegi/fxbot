# =============================================================
# data_layer/smc/premium_discount.py
# PURPOSE: Calculate Premium/Discount zones.
# Is price currently CHEAP (discount) or EXPENSIVE (premium)?
# Institutions BUY in discount, SELL in premium.
# This prevents the bot from buying at the top of a range.
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


def calculate_premium_discount(df: pd.DataFrame,
                               current_price: float,
                               pip_size: float = 0.0001) -> dict:
    """
    Calculate where price sits in the premium/discount framework.

    Range = highest high to lowest low of the lookback period.
    Equilibrium = 50% of that range (fair value midpoint).

    Above 50% = PREMIUM (expensive — institutions sell here).
    Below 50% = DISCOUNT (cheap — institutions buy here).

    Zones:
    75-100% = Extreme Premium  → Strong sell area
    50-75%  = Premium          → Mild sell area
    25-50%  = Discount         → Mild buy area
    0-25%   = Extreme Discount → Strong buy area
    """
    range_high = df['high'].max()
    range_low  = df['low'].min()
    rang       = range_high - range_low

    if rang == 0:
        return {}

    equilibrium  = range_low + (rang * 0.50)
    premium_zone = range_low + (rang * 0.75)
    discount_zone= range_low + (rang * 0.25)

    # Where is current price in the range (0% = bottom, 100% = top)
    position_pct = ((current_price - range_low) / rang) * 100

    # Zone classification
    if position_pct >= 75:
        zone  = 'EXTREME_PREMIUM'
        bias  = 'SELL'
        note  = 'Price extremely expensive — institutions sell here'
    elif position_pct >= 50:
        zone  = 'PREMIUM'
        bias  = 'SELL'
        note  = 'Price above equilibrium — prefer sells'
    elif position_pct >= 25:
        zone  = 'DISCOUNT'
        bias  = 'BUY'
        note  = 'Price below equilibrium — prefer buys'
    else:
        zone  = 'EXTREME_DISCOUNT'
        bias  = 'BUY'
        note  = 'Price extremely cheap — institutions buy here'

    pips_to_eq   = round((current_price - equilibrium) / pip_size, 1)
    range_pips   = round(rang / pip_size, 1)

    return {
        'range_high':    round(float(range_high),  5),
        'range_low':     round(float(range_low),   5),
        'equilibrium':   round(float(equilibrium), 5),
        'premium_zone':  round(float(premium_zone),  5),
        'discount_zone': round(float(discount_zone), 5),
        'current_price': round(current_price, 5),
        'position_pct':  round(position_pct, 1),
        'zone':          zone,
        'bias':          bias,
        'pips_to_eq':    pips_to_eq,
        'range_pips':    range_pips,
        'note':          note,
    }


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    SYMBOL = "EURUSD"
    print(f"Calculating Premium/Discount for {SYMBOL}...\n")

    df_h1 = get_candles(SYMBOL, mt5.TIMEFRAME_H1,  100)
    df_h4 = get_candles(SYMBOL, mt5.TIMEFRAME_H4,  100)

    tick          = mt5.symbol_info_tick(SYMBOL)
    current_price = tick.bid
    sym_info      = mt5.symbol_info(SYMBOL)
    pip_size      = sym_info.point * 10 if sym_info else 0.0001

    for label, df in [("H1 (100 candles)", df_h1),
                      ("H4 (100 candles)", df_h4)]:
        if df is None:
            continue
        pd_result = calculate_premium_discount(df, current_price, pip_size)
        if not pd_result:
            continue

        print("=" * 55)
        print(f"  PREMIUM/DISCOUNT — {SYMBOL} {label}")
        print("=" * 55)
        print(f"  Range High    : {pd_result['range_high']}")
        print(f"  Premium Zone  : {pd_result['premium_zone']}  (75%)")
        print(f"  Equilibrium   : {pd_result['equilibrium']}  (50%) ← fair value")
        print(f"  Discount Zone : {pd_result['discount_zone']}  (25%)")
        print(f"  Range Low     : {pd_result['range_low']}")
        print(f"  Range Width   : {pd_result['range_pips']} pips")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Current Price : {pd_result['current_price']}")
        print(f"  Position      : {pd_result['position_pct']}% of range")
        print(f"  Zone          : {pd_result['zone']}")
        print(f"  Bias          : {pd_result['bias']}")
        print(f"  Pips to EQ    : {pd_result['pips_to_eq']:+.1f}")
        print(f"  Note          : {pd_result['note']}")
        print()

    mt5.shutdown()
