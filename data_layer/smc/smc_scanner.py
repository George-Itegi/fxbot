# =============================================================
# data_layer/smc/smc_scanner.py
# PURPOSE: Unified SMC scanner — combines ALL SMC modules
# including new: sweeps, premium/discount, breakers, HTF align.
# This is the complete institutional SMC report per symbol.
# Run standalone to test.
# =============================================================

import MetaTrader5 as mt5
from dotenv import load_dotenv
import os

from data_layer.smc.market_structure import (
    get_candles, find_swing_points, get_last_swings, detect_structure
)
from data_layer.smc.order_blocks import (
    detect_order_blocks, check_mitigation,
    get_nearest_ob, detect_breaker_blocks
)
from data_layer.smc.liquidity_pools import (
    detect_liquidity_pools, check_sweeps, get_nearest_pool
)
from data_layer.smc.fair_value_gaps import (
    detect_fvg, check_filled, get_nearest_fvg, get_quality_fvgs
)
from data_layer.smc.liquidity_sweeps import (
    detect_sweeps, get_last_sweep, get_recent_sweeps
)
from data_layer.smc.premium_discount import calculate_premium_discount
from data_layer.smc.htf_alignment import check_htf_alignment
from data_layer.momentum_velocity import get_pip_size

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


def scan_smc(symbol: str, timeframe=None) -> dict | None:
    """
    Run complete SMC scan on one symbol.
    Returns full institutional SMC report or None.
    """
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_H1

    df = get_candles(symbol, timeframe, 200)
    if df is None:
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    current_price = tick.bid

    # v4.1 FIX: Use centralized get_pip_size() instead of naive point*10
    pip_size = get_pip_size(symbol)

    # --- 1. Market Structure ---
    df_swings = find_swing_points(df, swing_length=5)
    structure = detect_structure(df_swings)
    swings    = get_last_swings(df_swings, n=3)

    # --- 2. Order Blocks + Breaker Blocks ---
    blocks = detect_order_blocks(df, impulse_factor=1.5)
    blocks['bullish_obs'] = check_mitigation(
        blocks['bullish_obs'], current_price)
    blocks['bearish_obs'] = check_mitigation(
        blocks['bearish_obs'], current_price)
    nearest_ob  = get_nearest_ob(
        blocks['bullish_obs'] + blocks['bearish_obs'], current_price)
    breakers    = detect_breaker_blocks(
        blocks['bullish_obs'] + blocks['bearish_obs'], current_price)

    # --- 3. Liquidity Pools ---
    pools = detect_liquidity_pools(df, swing_length=5)
    pools['buyside_pools']  = check_sweeps(pools['buyside_pools'],  df)
    pools['sellside_pools'] = check_sweeps(pools['sellside_pools'], df)
    nearest_pool = get_nearest_pool(
        pools['buyside_pools'] + pools['sellside_pools'], current_price)

    # --- 4. Fair Value Gaps (quality filtered) ---
    fvgs = detect_fvg(df, min_gap_pips=2.0, pip_size=pip_size)
    fvgs['bullish_fvgs'] = check_filled(fvgs['bullish_fvgs'], df)
    fvgs['bearish_fvgs'] = check_filled(fvgs['bearish_fvgs'], df)
    nearest_fvg   = get_nearest_fvg(
        fvgs['bullish_fvgs'] + fvgs['bearish_fvgs'], current_price)
    quality_fvgs  = get_quality_fvgs(
        fvgs['bullish_fvgs'] + fvgs['bearish_fvgs'], df,
        pip_size=pip_size, min_score=60)

    # --- 5. Liquidity Sweeps ---
    sweeps      = detect_sweeps(df, swing_length=5,
                                reversal_pips=3.0, pip_size=pip_size)
    last_sweep  = get_last_sweep(sweeps)
    recent_sweeps = get_recent_sweeps(sweeps, n=3)

    # --- 6. Premium / Discount ---
    pd_zone = calculate_premium_discount(df, current_price, pip_size)

    # --- 7. HTF Alignment ---
    entry_bias  = structure.get('trend', 'NEUTRAL')
    htf         = check_htf_alignment(symbol, entry_bias)

    # --- 8. Full SMC Bias Score ---
    smc_bias, smc_score = _calc_smc_bias(
        structure, blocks, pools, fvgs,
        nearest_ob, nearest_pool, nearest_fvg,
        last_sweep, pd_zone, htf,
        current_price, pip_size)

    return {
        'symbol':          symbol,
        'current_price':   round(current_price, 5),
        'structure':       structure,
        'swings':          swings,
        'order_blocks':    blocks,
        'nearest_ob':      nearest_ob,
        'breaker_blocks':  breakers,
        'liquidity':       pools,
        'nearest_pool':    nearest_pool,
        'fvgs':            fvgs,
        'nearest_fvg':     nearest_fvg,
        'quality_fvgs':    quality_fvgs,
        'sweeps':          sweeps,
        'last_sweep':      last_sweep,
        'recent_sweeps':   recent_sweeps,
        'premium_discount':pd_zone,
        'htf_alignment':   htf,
        'smc_bias':        smc_bias,
        'smc_score':       smc_score,
    }

def _calc_smc_bias(structure, blocks, pools, fvgs,
                   nearest_ob, nearest_pool, nearest_fvg,
                   last_sweep, pd_zone, htf,
                   current_price, pip_size) -> tuple:
    """
    Calculate SMC bias and score from ALL modules.
    Returns (bias string, score 0-100).
    """
    score = 0
    bull  = 0
    bear  = 0

    # Structure trend (25 pts)
    trend = structure.get('trend', 'RANGING')
    if trend == 'BULLISH':
        score += 25; bull += 1
    elif trend == 'BEARISH':
        score += 25; bear += 1

    # BOS direction (15 pts)
    bos = structure.get('bos')
    if bos:
        if 'BULLISH' in bos['type']:
            score += 15; bull += 1
        elif 'BEARISH' in bos['type']:
            score += 15; bear += 1

    # HTF alignment (20 pts — most important filter)
    htf_score = htf.get('score', 0)
    if htf.get('approved'):
        score += int(htf_score * 0.20)
        if htf.get('entry_bias') == 'BULLISH':
            bull += 1
        elif htf.get('entry_bias') == 'BEARISH':
            bear += 1

    # Last sweep direction (15 pts — actual trigger)
    if last_sweep:
        if last_sweep['bias'] == 'BULLISH':
            score += 15; bull += 1
        elif last_sweep['bias'] == 'BEARISH':
            score += 15; bear += 1

    # Premium/Discount alignment (15 pts)
    pd_bias = pd_zone.get('bias', '')
    if pd_bias == 'BUY':
        score += 15; bull += 1
    elif pd_bias == 'SELL':
        score += 15; bear += 1

    # Nearest OB type (10 pts)
    if nearest_ob:
        dist = abs(current_price - nearest_ob['mid']) / pip_size
        if dist <= 50:
            if nearest_ob['type'] == 'BULLISH_OB':
                score += 10; bull += 1
            elif nearest_ob['type'] == 'BEARISH_OB':
                score += 10; bear += 1

    bias = 'BULLISH' if bull > bear else \
           'BEARISH' if bear > bull else 'NEUTRAL'
    return bias, min(score, 100)

def print_smc_report(r: dict):
    """Print complete upgraded SMC report."""
    if not r:
        return

    s     = r['structure']
    bias  = r['smc_bias']
    icon  = "📈" if bias == "BULLISH" else "📉" if bias == "BEARISH" else "↔️"
    bos   = s.get('bos')
    choch = s.get('choch')
    nob   = r.get('nearest_ob')
    npool = r.get('nearest_pool')
    nfvg  = r.get('nearest_fvg')
    sw    = r.get('last_sweep')
    pd    = r.get('premium_discount', {})
    htf   = r.get('htf_alignment', {})
    brk   = r.get('breaker_blocks', [])
    qfvg  = r.get('quality_fvgs', [])

    print(f"\n{'═'*55}")
    print(f"  SMC REPORT — {r['symbol']}")
    print(f"  Current Price : {r['current_price']}")
    print(f"  SMC Bias      : {bias} {icon}  |  Score: {r['smc_score']}/100")
    print(f"{'═'*55}")

    print(f"\n  ── MARKET STRUCTURE ────────────────────────")
    print(f"  Trend  : {s['trend']}"
          f"  HH:{s['hh_count']} HL:{s['hl_count']}"
          f" | LH:{s['lh_count']} LL:{s['ll_count']}")
    print(f"  Last SH: {s['last_swing_high']}"
          f"  | Last SL: {s['last_swing_low']}")
    if bos:
        print(f"  BOS    : {bos['type']} @ {bos['level']}"
              f"  ({bos.get('break_pips','?')} pips)")
    if choch:
        print(f"  ⚠️ CHOCH: {choch['type']} @ {choch['level']}")

    print(f"\n  ── HTF ALIGNMENT ───────────────────────────")
    print(f"  H4 Bias   : {htf.get('h4_bias')}")
    print(f"  Alignment : {htf.get('alignment')}"
          f"  | Score: {htf.get('score')}/100"
          f"  | {'✅ APPROVED' if htf.get('approved') else '❌ REJECTED'}")

    print(f"\n  ── PREMIUM / DISCOUNT ──────────────────────")
    print(f"  Zone      : {pd.get('zone')}"
          f"  ({pd.get('position_pct')}% of range)")
    print(f"  Bias      : {pd.get('bias')}"
          f"  | Pips to EQ: {pd.get('pips_to_eq',0):+.1f}")
    print(f"  Note      : {pd.get('note')}")

    print(f"\n  ── LAST LIQUIDITY SWEEP ────────────────────")
    if sw:
        icon2 = "📈" if sw['bias'] == 'BULLISH' else "📉"
        print(f"  {icon2} {sw['type']} @ {sw['swept_level']}"
              f"  | {sw['time'][:16]}")
        print(f"  Reversal  : {sw['reversal_pips']} pips")
        print(f"  Note      : {sw['note']}")
    else:
        print("  No recent sweep detected.")

    print(f"\n  ── KEY LEVELS ──────────────────────────────")
    if nob:
        print(f"  Nearest OB   : {nob['type']}"
              f" | {nob['bottom']} — {nob['top']}")
    if brk:
        print(f"  Breakers     : {len(brk)} detected"
              f" | Nearest: {brk[0]['type']} {brk[0]['bottom']}—{brk[0]['top']}")
    if npool:
        status = "UNSWEPT" if not npool['swept'] else "SWEPT"
        print(f"  Nearest Pool : {npool['type']} @ {npool['level']}"
              f" ({status}, {npool['touches']} touches)")
    if qfvg:
        f = qfvg[0]
        print(f"  Best FVG     : {f['type']}"
              f" | {f['bottom']}—{f['top']}"
              f" | Quality: {f['quality_score']}/100")
    elif nfvg:
        print(f"  Nearest FVG  : {nfvg['type']}"
              f" | {nfvg['bottom']}—{nfvg['top']}")
    print(f"{'─'*55}")

# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    WATCHLIST = ["EURUSD", "GBPUSD", "XAUUSD"]
    print(f"Running UPGRADED SMC scan on {len(WATCHLIST)} symbols...\n")

    for symbol in WATCHLIST:
        print(f"Scanning {symbol}...")
        report = scan_smc(symbol, timeframe=mt5.TIMEFRAME_H1)
        if report:
            print_smc_report(report)
        else:
            print(f"  ⚠️ Could not scan {symbol}")

    print(f"\nSMC Scan complete.")
    mt5.shutdown()
