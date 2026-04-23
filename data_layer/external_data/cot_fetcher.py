# =============================================================
# data_layer/external_data/cot_fetcher.py
# PURPOSE: Fetch and parse CFTC Commitment of Traders report.
# Shows what hedge funds and institutions are holding.
# Source: CFTC Financial Futures file — free, updated weekly.
# Run standalone to test.
# =============================================================

import requests
import pandas as pd
import io
from dotenv import load_dotenv

load_dotenv()

# Correct URL — financial futures file (where forex lives)
COT_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"

# Column index mapping (from CFTC layout)
COL_MARKET   = 0    # Market name
COL_DATE     = 2    # Report date
COL_OI       = 7    # Open interest
# Non-commercial (large speculators / hedge funds)
COL_NC_LONG  = 8
COL_NC_SHORT = 9
COL_NC_SPREAD= 10
# Commercial
COL_COM_LONG = 11
COL_COM_SHORT= 12
# Non-reportable (retail)
COL_NR_LONG  = 13
COL_NR_SHORT = 14

# Exact CFTC market names for our symbols
SYMBOL_MAP = {
    "EURUSD": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBPUSD": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
    "USDJPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "AUDUSD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "USDCAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "XAUUSD": "GOLD - COMMODITY EXCHANGE INC.",
}

def fetch_cot_data() -> pd.DataFrame | None:
    """Download and parse the CFTC financial futures COT file."""
    try:
        print("Fetching COT data from CFTC...")
        r = requests.get(COT_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text),
                         low_memory=False, header=None)
        print(f"COT loaded: {len(df)} markets")
        return df
    except Exception as e:
        print(f"COT fetch failed: {e}")
        return None


def _safe_int(val) -> int:
    try:
        return int(str(val).replace(',', '').strip())
    except Exception:
        return 0


def get_cot_for_symbol(symbol: str, df: pd.DataFrame) -> dict | None:
    """Extract and interpret COT for one symbol."""
    market_name = SYMBOL_MAP.get(symbol)
    if not market_name:
        return {'symbol': symbol, 'error': 'Not in COT map'}

    # For gold use disaggregated file — skip for now
    if symbol == 'XAUUSD':
        return {'symbol': symbol, 'error': 'Gold uses separate file'}

    mask = df[COL_MARKET].str.strip() == market_name
    rows = df[mask]
    if rows.empty:
        return {'symbol': symbol,
                'error': f'No data found for {market_name}'}

    row = rows.iloc[0]

    nc_long  = _safe_int(row[COL_NC_LONG])
    nc_short = _safe_int(row[COL_NC_SHORT])
    nc_net   = nc_long - nc_short
    com_long = _safe_int(row[COL_COM_LONG])
    com_short= _safe_int(row[COL_COM_SHORT])
    com_net  = com_long - com_short
    oi       = _safe_int(row[COL_OI])

    # Net as % of open interest (shows how extreme positioning is)
    nc_net_pct = round((nc_net / oi * 100), 1) if oi > 0 else 0

    return {
        'symbol':      symbol,
        'market':      market_name,
        'report_date': str(row[COL_DATE]),
        'oi':          oi,
        'nc_long':     nc_long,
        'nc_short':    nc_short,
        'nc_net':      nc_net,
        'nc_net_pct':  nc_net_pct,
        'com_long':    com_long,
        'com_short':   com_short,
        'com_net':     com_net,
    }


def interpret_cot(cot: dict) -> dict:
    """Turn raw COT numbers into a directional signal."""
    if not cot or 'error' in cot:
        return {'bias': 'NEUTRAL', 'strength': 'UNKNOWN',
                'note': cot.get('error', 'No data') if cot else 'No data'}

    nc_net     = cot['nc_net']
    nc_net_pct = cot.get('nc_net_pct', 0)
    com_net    = cot['com_net']

    # Bias from non-commercials (hedge funds)
    if nc_net > 0:
        bias = 'BULLISH'
    elif nc_net < 0:
        bias = 'BEARISH'
    else:
        bias = 'NEUTRAL'

    # Commercials are contrarian — if they agree = weaker signal
    com_contrarian = (com_net < 0 and bias == 'BULLISH') or \
                     (com_net > 0 and bias == 'BEARISH')

    # Strength based on % of open interest
    abs_pct = abs(nc_net_pct)
    if abs_pct >= 30:
        strength = 'EXTREME'
        note = '⚠️ Extreme positioning — potential reversal risk'
    elif abs_pct >= 15:
        strength = 'STRONG'
        note = 'Strong hedge fund conviction'
    elif abs_pct >= 5:
        strength = 'MODERATE'
        note = 'Moderate institutional positioning'
    else:
        strength = 'WEAK'
        note = 'Weak or neutral positioning'

    return {
        'bias':             bias,
        'strength':         strength,
        'nc_net':           nc_net,
        'nc_net_pct':       nc_net_pct,
        'com_contrarian':   com_contrarian,
        'note':             note,
    }

def get_day_trading_filter(cot_result: dict) -> dict:
    """
    Convert weekly COT data into a day trading session filter.
    For day trading, COT is used as a BIAS GATE only:
    - EXTREME positioning against your trade direction = SKIP
    - STRONG positioning with your trade direction = BONUS CONFIDENCE
    - WEAK/MODERATE = neutral, don't override intraday signals
    """
    bias     = cot_result.get('bias', 'NEUTRAL')
    strength = cot_result.get('strength', 'WEAK')
    nc_pct   = abs(cot_result.get('nc_net_pct', 0))

    # Only block trades when COT is extreme AND against direction
    if strength == 'EXTREME' and nc_pct >= 35:
        return {
            'day_trade_bias':    bias,
            'blocks_longs':      bias == 'BEARISH',
            'blocks_shorts':     bias == 'BULLISH',
            'confidence_boost':  False,
            'note': f'COT EXTREME {bias} — avoid trades against this bias',
        }
    elif strength == 'STRONG':
        return {
            'day_trade_bias':    bias,
            'blocks_longs':      False,
            'blocks_shorts':     False,
            'confidence_boost':  True,
            'note': f'COT STRONG {bias} — adds confidence to aligned trades',
        }
    else:
        return {
            'day_trade_bias':    'NEUTRAL',
            'blocks_longs':      False,
            'blocks_shorts':     False,
            'confidence_boost':  False,
            'note': 'COT weak/moderate — no impact on day trades',
        }


def get_all_cot(symbols: list) -> dict:
    """Fetch COT for all symbols. Returns dict keyed by symbol."""
    df = fetch_cot_data()
    if df is None:
        return {}
    results = {}
    for symbol in symbols:
        cot    = get_cot_for_symbol(symbol, df)
        interp = interpret_cot(cot)
        merged = {**cot, **interp} if cot else interp
        merged['day_filter'] = get_day_trading_filter(merged)
        results[symbol] = merged
    return results


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    WATCHLIST = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    print(f"Fetching COT for {len(WATCHLIST)} symbols...\n")
    results = get_all_cot(WATCHLIST)

    for symbol, data in results.items():
        print("=" * 55)
        print(f"  COT REPORT — {symbol}")
        print("=" * 55)
        if 'error' in data:
            print(f"  ⚠️  {data['error']}")
            continue
        print(f"  Report Date    : {data.get('report_date')}")
        print(f"  Open Interest  : {data.get('oi'):,}")
        print(f"  -" * 25)
        print(f"  Hedge Funds    : Long={data.get('nc_long'):>8,}"
              f"  Short={data.get('nc_short'):>8,}"
              f"  Net={data.get('nc_net'):>+8,}"
              f"  ({data.get('nc_net_pct'):+.1f}% of OI)")
        print(f"  Commercials    : Long={data.get('com_long'):>8,}"
              f"  Short={data.get('com_short'):>8,}"
              f"  Net={data.get('com_net'):>+8,}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Bias           : {data.get('bias')}")
        print(f"  Strength       : {data.get('strength')}")
        print(f"  Com Contrarian : {'✅ YES' if data.get('com_contrarian') else '❌ NO'}")
        print(f"  Note           : {data.get('note')}")
        print()
