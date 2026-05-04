# =============================================================
# data_layer/external_data/external_scanner.py
# PURPOSE: Combines all 4 external data modules into one
# macro report optimized for forex DAY TRADING.
# Includes: session gating, volatility scoring, cross-filtering.
# Run standalone to test.
# =============================================================

from datetime import datetime, timezone
from dotenv import load_dotenv

from data_layer.external_data.cot_fetcher import get_all_cot
from data_layer.external_data.intermarket import fetch_intermarket, interpret_intermarket
from data_layer.external_data.fear_greed import get_fear_greed
from data_layer.external_data.news_sentiment import get_news_sentiment, is_news_blackout

load_dotenv()

_cache = {
    'cot':         {'data': {}, 'fetched_at': None},
    'intermarket': {'data': {}, 'fetched_at': None},
    'fear_greed':  {'data': {}, 'fetched_at': None},
    'news':        {'data': {}, 'fetched_at': None},
}

COT_REFRESH_HOURS         = 168
INTERMARKET_REFRESH_HOURS = 1
FEAR_GREED_REFRESH_HOURS  = 4
NEWS_REFRESH_MINUTES      = 15

SESSION_MULTIPLIERS = {
    'LONDON_OPEN':       1.2,   # Manipulation — high opportunity but deceptive
    'NY_LONDON_OVERLAP': 1.4,   # Distribution — best window, highest liquidity
    'LONDON_SESSION':    1.3,   # Expansion — strong directional moves
    # v2.0: Blocked sessions — 0.0 multiplier (absolute block)
    'NY_AFTERNOON':      0.0,   # BLOCKED: 36 trades, 11.1% WR, -$2,607
    'TOKYO':             0.0,   # BLOCKED: marginal edge, not in whitelist
    'SYDNEY':            0.0,   # BLOCKED: ~67 trades, ~20% WR, -$3,361
}

def _needs_refresh(key: str, max_hours: float = 1.0) -> bool:
    fetched = _cache[key]['fetched_at']
    if fetched is None:
        return True
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600 >= max_hours


def get_external_data(symbols: list, force_refresh: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    if force_refresh or _needs_refresh('cot', COT_REFRESH_HOURS):
        print("[EXTERNAL] Fetching COT (weekly)...")
        _cache['cot']['data']       = get_all_cot(symbols)
        _cache['cot']['fetched_at'] = now
    if force_refresh or _needs_refresh('intermarket', INTERMARKET_REFRESH_HOURS):
        print("[EXTERNAL] Fetching intermarket...")
        raw = fetch_intermarket()
        _cache['intermarket']['data']       = interpret_intermarket(raw)
        _cache['intermarket']['fetched_at'] = now
    if force_refresh or _needs_refresh('fear_greed', FEAR_GREED_REFRESH_HOURS):
        print("[EXTERNAL] Fetching Fear & Greed...")
        _cache['fear_greed']['data']       = get_fear_greed()
        _cache['fear_greed']['fetched_at'] = now
    if force_refresh or _needs_refresh('news', NEWS_REFRESH_MINUTES / 60):
        print("[EXTERNAL] Fetching news...")
        _cache['news']['data']       = get_news_sentiment()
        _cache['news']['fetched_at'] = now
    return _build_report(symbols)

def _calc_cross_filter(symbol: str, direction: str,
                       cot: dict, im: dict, fg: dict, news: dict) -> dict:
    votes = []; agreements = []; conflicts = []

    im_signal = im.get('pair_signals', {}).get(symbol, {})
    im_bias   = im_signal.get('bias', 'NEUTRAL')
    if im_bias == direction:
        votes.append(25); agreements.append(f"Intermarket {im_bias}")
    elif im_bias not in ('NEUTRAL', direction):
        votes.append(-15); conflicts.append(f"Intermarket {im_bias}")

    usd_bias   = im.get('usd_bias', 'NEUTRAL')
    inv_pairs  = ['EURUSD','GBPUSD','AUDUSD','NZDUSD','USDCHF']
    usd_ok = (
        (symbol in inv_pairs and 'BEAR' in usd_bias and direction=='BULLISH') or
        (symbol in inv_pairs and 'BULL' in usd_bias and direction=='BEARISH') or
        (symbol not in inv_pairs and 'BULL' in usd_bias and direction=='BULLISH')
    )
    if usd_ok:
        votes.append(20); agreements.append("USD bias aligned")

    cot_filter = cot.get(symbol, {}).get('day_filter', {})
    cot_bias   = cot_filter.get('day_trade_bias', 'NEUTRAL')
    if cot_bias == direction:
        votes.append(20); agreements.append(f"COT {cot_bias}")
    elif cot_filter.get('blocks_longs') and direction == 'BULLISH':
        votes.append(-20); conflicts.append("COT blocks longs")
    elif cot_filter.get('blocks_shorts') and direction == 'BEARISH':
        votes.append(-20); conflicts.append("COT blocks shorts")

    fg_map  = {'BUY':'BULLISH','SELL':'BEARISH','NEUTRAL':'NEUTRAL'}
    fg_bias = fg_map.get(fg.get('signal','NEUTRAL'), 'NEUTRAL')
    if fg_bias == direction:
        votes.append(15); agreements.append("Fear/Greed aligned")
    elif fg_bias not in ('NEUTRAL', direction):
        votes.append(-10); conflicts.append("Fear/Greed against")

    score = max(0, min(100, 50 + sum(votes)))
    return {'score': score, 'agreements': agreements, 'conflicts': conflicts}

def _build_report(symbols: list) -> dict:
    cot        = _cache['cot']['data']
    im         = _cache['intermarket']['data']
    fg         = _cache['fear_greed']['data']
    news       = _cache['news']['data']
    blackout   = news.get('blackout', {})
    volatility = news.get('volatility', {})
    session    = im.get('session', 'UNKNOWN')
    sess_mult  = SESSION_MULTIPLIERS.get(session, 1.0)

    # ── Master gate ───────────────────────────────────────────
    # ONLY these block trading — COT/IM/FG do NOT block signals
    # v2.0: Session blocking ENFORCED (was disabled for testing)
    day_trade_ok    = True
    blocking_reason = None

    # v2.0: Hard block — no trading outside whitelisted sessions
    from config.settings import SESSION_WHITELIST
    if session not in SESSION_WHITELIST:
        day_trade_ok    = False
        blocking_reason = f"SESSION BLOCKED: {session} not in whitelist"
    if blackout.get('blackout'):
        day_trade_ok    = False
        blocking_reason = f"NEWS BLACKOUT: {blackout.get('reason')}"
    elif volatility.get('volatility_score', 0) >= 70:
        day_trade_ok    = False
        blocking_reason = f"VOLATILITY EXTREME: {volatility.get('advice')}"
    elif im.get('vix', 20) > 40:
        day_trade_ok    = False
        blocking_reason = f"VIX EXTREME: {im.get('vix'):.1f}"

    # COT/IM/FG kept for reference/dashboard only — NOT used to gate signals
    fg_gate        = fg.get('day_gate', {})
    size_reduction = False   # disabled — was causing false blocks

    # Minimal symbol context — no COT/IM bias injection into signal path
    symbol_context = {}
    for sym in symbols:
        symbol_context[sym] = {
            'macro_bias':         'NEUTRAL',   # Not used in signal scoring
            'im_signal':          'NEUTRAL',   # Reference only
            'cot_bias':           'NEUTRAL',   # Reference only
            'cot_blocks':         False,        # Not blocking signals
            'session_multiplier': sess_mult,
        }

    return {
        'timestamp':          datetime.now(timezone.utc).strftime('%H:%M:%S UTC'),
        'session':            session,
        'session_multiplier': sess_mult,
        'day_trade_ok':       day_trade_ok,
        'blocking_reason':    blocking_reason,
        'size_reduction':     size_reduction,
        'risk_env':           im.get('risk_environment','UNKNOWN'),
        'usd_bias':           im.get('usd_bias','NEUTRAL'),
        'vix':                im.get('vix', 0),
        'dxy_change':         im.get('dxy_change', 0),
        'fear_greed':         fg.get('score', 50),
        'fg_zone':            fg.get('zone','NEUTRAL'),
        'fg_gate':            fg_gate,
        'news_sentiment':     news.get('overall','NEUTRAL'),
        'news_blackout':      blackout,
        'volatility':         volatility,
        'symbol_context':     symbol_context,
        'cot_data':           cot,
    }

def print_external_report(r: dict):
    ok   = r['day_trade_ok']
    gate = "✅ TRADING ALLOWED" if ok else "🛑 BLOCKED"
    mult = r['session_multiplier']
    mult_str = f"x{mult}" if mult > 0 else "x0 (blocked)"

    print(f"\n{'█'*57}")
    print(f"  APEX TRADER — EXTERNAL DATA REPORT")
    print(f"  {r['timestamp']}  |  Session: {r['session']} ({mult_str})")
    print(f"{'█'*57}")
    print(f"  DAY TRADE GATE  : {gate}")
    if not ok:
        print(f"  REASON          : {r['blocking_reason']}")
    if r['size_reduction']:
        print(f"  SIZE ALERT      : {r['fg_gate'].get('reason')}")

    vol = r.get('volatility', {})
    print(f"\n  ── VOLATILITY RISK ─────────────────────────")
    print(f"  Score  : {vol.get('volatility_score',0)}/100"
          f"  Level: {vol.get('level','?')}")
    print(f"  Advice : {vol.get('advice','?')}")

    print(f"\n  ── MACRO ENVIRONMENT ───────────────────────")
    print(f"  Risk Env   : {r['risk_env']}")
    print(f"  USD Bias   : {r['usd_bias']}")
    print(f"  VIX        : {r['vix']:.2f}")
    print(f"  DXY Change : {r['dxy_change']:+.3f}%")
    print(f"  Fear/Greed : {r['fear_greed']} ({r['fg_zone']})")
    print(f"  News       : {r['news_sentiment']}")

    if r['news_blackout'].get('upcoming'):
        print(f"\n  UPCOMING HIGH IMPACT NEWS:")
        for e in r['news_blackout']['upcoming']:
            print(f"  ⚡ {e['title']} [{e['country']}]"
                  f" — {e['mins_away']:.0f} mins")

    print(f"\n  ── CROSS-FILTER SCORES (per symbol) ────────")
    print(f"  {'Symbol':<8} {'Macro':<10} {'Bull CF':<10}"
          f" {'Bear CF':<10} Agreements")
    print(f"  {'─'*53}")
    for sym, ctx in r['symbol_context'].items():
        icon   = "📈" if ctx['macro_bias']=='BULLISH' else \
                 "📉" if ctx['macro_bias']=='BEARISH' else "↔️"
        bull_s = ctx['bull_cross_filter']['score']
        bear_s = ctx['bear_cross_filter']['score']
        agrees = ', '.join(ctx['bull_cross_filter']['agreements'][:2])
        block  = " 🛑" if ctx['cot_blocks'] else ""
        print(f"  {sym:<8} {ctx['macro_bias']:<10}{icon}"
              f" {bull_s:<10} {bear_s:<10}{block}")
        if agrees:
            print(f"           {agrees}")
    print(f"{'█'*57}\n")


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    WATCHLIST = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
    print("Running External Data Scanner...\n")
    report = get_external_data(WATCHLIST, force_refresh=True)
    print_external_report(report)
