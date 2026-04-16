# =============================================================
# data_layer/market_scanner.py
# PURPOSE: Unified scanner that combines all data layer modules.
# Runs tick data, delta, volume profile, and VWAP together
# and prints one clean institutional report per symbol.
# This is what the bot will call every scan cycle.
# Run this file standalone to test.
# =============================================================

import MetaTrader5 as mt5
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

from data_layer.tick_fetcher import get_ticks
from data_layer.delta_calculator import calculate_delta, get_rolling_delta, calculate_order_flow_imbalance
from data_layer.volume_profile import get_full_profile
from data_layer.vwap_calculator import get_vwap_context
from data_layer.tick_volume_surge import detect_tick_volume_surge
from data_layer.momentum_velocity import calculate_momentum_velocity, get_pip_size

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


def scan_symbol(symbol: str) -> dict | None:
    """
    Run full institutional scan on one symbol.
    Returns a combined report dict or None if data unavailable.
    """
    # --- 1. Tick Data + Delta ---
    df_ticks = get_ticks(symbol, num_ticks=500)
    if df_ticks is None:
        print(f"[SCANNER] No tick data for {symbol}")
        return None

    full_delta    = calculate_delta(df_ticks)
    rolling_delta = get_rolling_delta(df_ticks, window=100)

    # --- 2. Volume Profile ---
    profile = get_full_profile(
        symbol       = symbol,
        timeframe    = mt5.TIMEFRAME_M15,
        candle_count = 200,
        session_type = "SESSION",
        bins         = 100
    )

    # --- 3. VWAP ---
    vwap = get_vwap_context(
        symbol       = symbol,
        timeframe    = mt5.TIMEFRAME_M15,
        candle_count = 200
    )

    if not profile or not vwap:
        print(f"[SCANNER] Missing profile/vwap data for {symbol}")
        return None

    # --- 4. Order Flow Imbalance ---
    order_flow_imb = calculate_order_flow_imbalance(df_ticks, window=50)

    # --- 5. Tick Volume Surge Detection ---
    volume_surge = detect_tick_volume_surge(df_ticks, surge_multiplier=2.0)

    # --- 6. Momentum Velocity ---
    pip_size = get_pip_size(symbol)
    momentum = calculate_momentum_velocity(df_ticks, pip_size=pip_size, window_seconds=60)

    # --- 7. Combined Bias ---
    bias_votes = []
    if full_delta.get('bias')    == 'BULLISH': bias_votes.append(1)
    elif full_delta.get('bias')  == 'BEARISH': bias_votes.append(-1)
    if rolling_delta.get('bias') == 'BULLISH': bias_votes.append(1)
    elif rolling_delta.get('bias')== 'BEARISH': bias_votes.append(-1)
    if profile.get('bias')       == 'BULLISH': bias_votes.append(1)
    elif profile.get('bias')     == 'BEARISH': bias_votes.append(-1)
    if vwap.get('bias') in ('BULLISH', 'STRONG_BULL'): bias_votes.append(1)
    elif vwap.get('bias') in ('BEARISH', 'STRONG_BEAR'): bias_votes.append(-1)

    vote_sum = sum(bias_votes)
    if vote_sum >= 2:
        combined_bias = "BULLISH"
    elif vote_sum <= -2:
        combined_bias = "BEARISH"
    else:
        combined_bias = "NEUTRAL"

    report = {
        'symbol':         symbol,
        'timestamp':      datetime.now(timezone.utc).strftime('%H:%M:%S'),
        'delta':          full_delta,
        'rolling_delta':  rolling_delta,
        'order_flow_imbalance': order_flow_imb,
        'volume_surge':   volume_surge,
        'momentum':       momentum,
        'profile':        profile,
        'vwap':           vwap,
        'combined_bias':  combined_bias,
        'bias_votes':     vote_sum,
    }

    # Add tradeability score and market state
    trade_score          = calculate_tradeability_score(report)
    report['trade_score']= trade_score['score']
    report['score_reasons'] = trade_score['reasons']
    report['market_state']  = detect_market_state(report, trade_score['score'])

    return report


def print_report(report: dict):
    """Print a clean institutional-style report for one symbol."""
    if not report:
        return

    sym  = report['symbol']
    d    = report['delta']
    rd   = report['rolling_delta']
    p    = report['profile']
    v    = report['vwap']
    bias = report['combined_bias']
    imb  = report.get('order_flow_imbalance', {})
    surge = report.get('volume_surge', {})
    mom  = report.get('momentum', {})

    bias_icon  = "📈" if bias == "BULLISH" else "📉" if bias == "BEARISH" else "↔️"
    state      = report.get('market_state', 'UNKNOWN')
    score      = report.get('trade_score', 0)
    reasons    = report.get('score_reasons', [])

    state_icons = {
        "TRENDING_STRONG":    "🚀",
        "TRENDING_EXTENDED":  "⚠️",
        "BALANCED":           "↔️",
        "REVERSAL_RISK":      "🔄",
        "BREAKOUT_ACCEPTED":  "✅",
        "BREAKOUT_REJECTED":  "❌",
    }
    state_icon = state_icons.get(state, "❓")

    print(f"\n{'═'*55}")
    print(f"  {sym}  |  {report['timestamp']} UTC")
    print(f"  BIAS   : {bias} {bias_icon}  (votes: {report['bias_votes']:+d}/4)")
    print(f"  STATE  : {state} {state_icon}")
    print(f"  SCORE  : {score}/100")
    print(f"{'═'*55}")

    print(f"\n  ── ORDER FLOW ──────────────────────────────")
    print(f"  Full Delta    : {d.get('delta', 0):+d}  "
          f"({d.get('bias','?')} / {d.get('strength','?')})")
    print(f"  Rolling Delta : {rd.get('delta', 0):+d}  "
          f"({rd.get('bias','?')} / {rd.get('strength','?')})  ← last 100 ticks")
    print(f"  Buy Ticks     : {d.get('buy_ticks', 0)}")
    print(f"  Sell Ticks    : {d.get('sell_ticks', 0)}")

    print(f"\n  ── ORDER FLOW IMBALANCE ────────────────────")
    print(f"  Imbalance     : {imb.get('imbalance', 0):+.2f}"
          f"  ({imb.get('direction', '?')} / {imb.get('strength', '?')})")
    print(f"  Can BUY       : {'YES' if imb.get('can_buy') else 'NO'}"
          f"  |  Can SELL: {'YES' if imb.get('can_sell') else 'NO'}")

    print(f"\n  ── VOLUME SURGE ───────────────────────────")
    surge_icon = "!!" if surge.get('surge_detected') else "--"
    print(f"  Surge         : {surge_icon}"
          f"  Ratio: {surge.get('surge_ratio', 0)}x"
          f"  ({surge.get('surge_strength', '?')})"
          f"  Dir: {surge.get('surge_direction', '?')}")

    print(f"\n  ── MOMENTUM VELOCITY ──────────────────────")
    mom_icon = ">>>" if mom.get('is_scalpable') else "..."
    print(f"  Velocity      : {mom_icon} {mom.get('velocity_pips_min', 0)} pips/min"
          f"  ({mom.get('velocity_direction', '?')})"
          f"  | Scalpable: {mom.get('is_scalpable')}"
          f"  | Choppy: {mom.get('is_choppy')}")

    print(f"\n  ── VOLUME PROFILE ──────────────────────────")
    print(f"  Current Price : {p.get('current_price')}")
    print(f"  POC           : {p.get('poc')}  ← Most traded level")
    print(f"  VAH           : {p.get('vah')}  ← Value Area High")
    print(f"  VAL           : {p.get('val')}  ← Value Area Low")
    print(f"  VA Width      : {p.get('va_width_pips')} pips")
    print(f"  HVN Magnets   : {p.get('hvn_list')}")
    print(f"  LVN Gaps      : {p.get('lvn_list')}")
    print(f"  Position      : {p.get('price_position')}  → {p.get('note')}")
    print(f"  Pips to POC   : {p.get('pip_to_poc')}")

    print(f"\n  ── VWAP ────────────────────────────────────")
    print(f"  VWAP          : {v.get('vwap')}  ← Today's fair value")
    print(f"  Upper Band 1  : {v.get('upper_band_1')}")
    print(f"  Lower Band 1  : {v.get('lower_band_1')}")
    print(f"  Pips from VWAP: {v.get('pip_from_vwap'):+.1f}")
    print(f"  Position      : {v.get('position')}")
    print(f"  Note          : {v.get('note')}")
    print(f"\n  ── SCORE BREAKDOWN ─────────────────────────")
    for r in reasons:
        print(f"  ✔ {r}")
    print(f"{'─'*55}")


def calculate_tradeability_score(report: dict) -> dict:
    """
    Score how tradeable a setup is from 0 to 100.
    Higher = better quality trade opportunity.
    Combines: delta strength, VWAP distance, POC distance,
    price position, delta/price agreement,
    + NEW: volume surge, order flow imbalance, momentum velocity.
    """
    score = 0
    reasons = []

    d  = report.get('delta', {})
    rd = report.get('rolling_delta', {})
    p  = report.get('profile', {})
    v  = report.get('vwap', {})
    imb = report.get('order_flow_imbalance', {})
    surge = report.get('volume_surge', {})
    mom = report.get('momentum', {})

    # --- Delta direction (15 pts) ---
    if d.get('bias') == rd.get('bias') and d.get('bias') != 'NEUTRAL':
        score += 15
        reasons.append(f"Delta aligned ({d.get('bias')})")
    elif d.get('bias') != 'NEUTRAL':
        score += 8
        reasons.append("Delta partially aligned")

    # --- Delta strength (10 pts) ---
    strength = rd.get('strength', 'WEAK')
    if strength == 'STRONG':
        score += 10
        reasons.append("Strong rolling delta")
    elif strength == 'MODERATE':
        score += 5
        reasons.append("Moderate rolling delta")

    # --- Order Flow Imbalance (15 pts, was 20 — deduplicated) ---
    imbalance = imb.get('imbalance', 0)
    imb_strength = imb.get('strength', 'NONE')
    if imb_strength == 'EXTREME':
        score += 15
        reasons.append(f"Strong OF imbalance ({imbalance:+.2f})")
    elif imb_strength == 'STRONG':
        score += 12
        reasons.append(f"Strong OF imbalance ({imbalance:+.2f})")
    elif imb_strength == 'MODERATE':
        score += 7
        reasons.append(f"Moderate OF imbalance ({imbalance:+.2f})")
    elif imb_strength == 'WEAK':
        score += 2

    # --- Volume Surge Detection (10 pts, was 15 — deduplicated) ---
    if surge.get('surge_detected', False):
        surge_str = surge.get('surge_strength', 'NONE')
        if surge_str in ('EXTREME', 'STRONG'):
            score += 10
            reasons.append(f"Volume surge ({surge.get('surge_ratio')}x)")
        else:
            score += 6
            reasons.append(f"Volume surge ({surge.get('surge_ratio')}x)")
    else:
        # Small penalty for no institutional activity
        score -= 2

    # --- Momentum Velocity (8 pts, was 10 — deduplicated) ---
    # NOTE: Momentum and volume surge often co-occur.
    # We track them separately but cap the combined contribution
    # to avoid double-counting the same market energy.
    if mom.get('is_scalpable', False):
        score += 8
        reasons.append(f"Momentum active ({mom.get('velocity_pips_min')} pips/min)")
    elif mom.get('is_choppy', True):
        score -= 5
        reasons.append(f"Choppy market ({mom.get('velocity_pips_min')} pips/min)")
    else:
        score += 3
        reasons.append(f"Moderate momentum ({mom.get('velocity_pips_min')} pips/min)")

    # --- Deduplication cap: OF + Volume + Momentum max combined 25 pts ---
    of_contribution = 0
    if imb_strength in ('EXTREME', 'STRONG'):
        of_contribution = 15
    elif imb_strength == 'MODERATE':
        of_contribution = 7
    else:
        of_contribution = 2
    
    vol_contribution = 10 if surge.get('surge_detected', False) else -2
    mom_contribution = 8 if mom.get('is_scalpable', False) else -5
    
    combined_flow = of_contribution + vol_contribution + mom_contribution
    if combined_flow > 25:
        # Cap to prevent score inflation from correlated metrics
        overflow = combined_flow - 25
        score -= overflow
        reasons.append(f"Flow dedup cap (-{overflow}pts)")

    # --- Price vs VWAP distance (15 pts) ---
    pip_vwap = abs(v.get('pip_from_vwap', 999))
    vwap_pos = v.get('position', '')
    if pip_vwap <= 10:
        score += 15
        reasons.append("Price very close to VWAP (high value)")
    elif pip_vwap <= 20:
        score += 10
        reasons.append("Price near VWAP")
    elif pip_vwap <= 35:
        score += 5
        reasons.append("Price moderately extended from VWAP")
    else:
        score += 0
        reasons.append("Price far from VWAP (extended/risky)")

    # --- Price vs POC distance (10 pts) ---
    pip_poc = abs(p.get('pip_to_poc', 999))
    if pip_poc <= 10:
        score += 10
        reasons.append("Price at POC (highest value zone)")
    elif pip_poc <= 30:
        score += 7
        reasons.append("Price near POC")
    elif pip_poc <= 60:
        score += 3
        reasons.append("Price moderately far from POC")

    # --- Price position in value area (10 pts) ---
    pos = p.get('price_position', '')
    if pos == 'INSIDE_VA':
        score += 10
        reasons.append("Price inside value area (balanced)")
    elif pos in ('ABOVE_VAH', 'BELOW_VAL'):
        score += 5
        reasons.append("Price outside value area (breakout)")

    # --- Delta agrees with price position (5 pts) ---
    bias    = report.get('combined_bias', 'NEUTRAL')
    d_bias  = d.get('bias', 'NEUTRAL')
    if bias == 'BULLISH' and d_bias == 'BULLISH':
        score += 5
        reasons.append("Delta confirms bullish bias")
    elif bias == 'BEARISH' and d_bias == 'BEARISH':
        score += 5
        reasons.append("Delta confirms bearish bias")

    return {
        'score':   min(max(score, 0), 100),
        'reasons': reasons,
    }


def detect_market_state(report: dict, score: int) -> str:
    """
    Classify the market into one of 6 institutional states.
    This tells the bot exactly what kind of market it is in.
    Now includes momentum velocity and volume surge for better state detection.
    """
    p       = report.get('profile', {})
    v       = report.get('vwap', {})
    rd      = report.get('rolling_delta', {})
    d       = report.get('delta', {})
    bias    = report.get('combined_bias', 'NEUTRAL')
    pos     = p.get('price_position', '')
    pip_vwap= abs(v.get('pip_from_vwap', 0))
    votes   = report.get('bias_votes', 0)
    mom     = report.get('momentum', {})
    surge   = report.get('volume_surge', {})

    # CHOPPY: low momentum velocity, no surge, price inside VA
    if mom.get('is_choppy', False) and not surge.get('surge_detected', False):
        if pos == 'INSIDE_VA':
            return "BALANCED"

    # REVERSAL_RISK: price extended, delta turning opposite
    if pip_vwap > 30 and d.get('bias') != rd.get('bias'):
        return "REVERSAL_RISK"

    # TRENDING_EXTENDED: strong trend but too far from value
    if votes >= 3 and pip_vwap > 25:
        return "TRENDING_EXTENDED"

    # BREAKOUT_ACCEPTED: outside value area, delta confirms + surge
    if pos in ('ABOVE_VAH', 'BELOW_VAL'):
        delta_confirms = d.get('bias') == bias and rd.get('bias') == bias
        surge_confirms = surge.get('surge_detected', False)
        if delta_confirms and surge_confirms:
            return "BREAKOUT_ACCEPTED"
        elif delta_confirms:
            return "BREAKOUT_ACCEPTED"
        else:
            return "BREAKOUT_REJECTED"

    # TRENDING_STRONG: all factors aligned + momentum active
    if votes >= 3 and pip_vwap <= 20 and mom.get('is_scalpable', False):
        return "TRENDING_STRONG"

    # TRENDING_STRONG (legacy): all factors aligned, not extended
    if votes >= 3 and pip_vwap <= 20:
        return "TRENDING_STRONG"

    # BALANCED: price inside value, delta weak
    if pos == 'INSIDE_VA' and abs(d.get('delta', 0)) < 20:
        return "BALANCED"

    return "BALANCED"


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    if not connect():
        exit()

    # Test on these symbols one by one
    WATCHLIST = ["EURUSD", "GBPUSD", "XAUUSD"]

    print(f"Running institutional scan on {len(WATCHLIST)} symbols...\n")

    for symbol in WATCHLIST:
        print(f"Scanning {symbol}...")
        report = scan_symbol(symbol)
        if report:
            print_report(report)
        else:
            print(f"  ⚠️  Could not scan {symbol}\n")

    print(f"\nScan complete.")
    mt5.shutdown()
