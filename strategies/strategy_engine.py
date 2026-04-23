# =============================================================
# strategies/strategy_engine.py  v5.0
# MERGED FIX: Combines both AI audit findings.
#
# Key changes:
#  1. Hard state gates — wrong market state = SKIP (not just log)
#  2. Mandatory institutional gate — order flow OR volume surge
#     must confirm BEFORE any strategy score is trusted
#  3. Minimum 2-strategy consensus required for execution
#  4. No session blocking (trade any session per user preference)
#  5. Correlated strategies penalized to reduce false consensus
# =============================================================

import MetaTrader5 as mt5
from datetime import datetime, timezone
from core.logger import get_logger
from data_layer.price_feed import get_candles
from strategies.strategy_registry import get_active_strategies, REGISTRY
from strategies.smc_ob_reversal import evaluate as ob_evaluate
from strategies.liquidity_sweep_entry import evaluate as sweep_evaluate
from strategies.vwap_mean_reversion import evaluate as vwap_evaluate
from strategies.delta_divergence import evaluate as delta_div_evaluate
from strategies.trend_continuation import evaluate as trend_cont_evaluate
from strategies.fvg_reversion import evaluate as fvg_evaluate

log = get_logger(__name__)

# ── Minimum score each strategy must hit to be considered ────
# Raised because scoring is inflated — lagging indicators score
# too easily. These minimums require real institutional confluence.
STRATEGY_MIN_SCORES = {
    "SMC_OB_REVERSAL":       70,   # OB + delta + sweep confirmation
    "LIQUIDITY_SWEEP_ENTRY": 70,   # Sweep + BOS + delta
    "VWAP_MEAN_REVERSION":   65,   # VWAP distance + structure
    "DELTA_DIVERGENCE":      70,   # Price vs delta divergence
    "TREND_CONTINUATION":    72,   # Multi-TF trend + pullback
    "FVG_REVERSION":         68,   # FVG gap fill entry
}

# ── Strategies grouped by what they fundamentally measure ────
# Each group represents a DIFFERENT data source / market approach.
# Consensus gate requires 2+ DIFFERENT groups to agree — this
# prevents correlated strategies from echoing the same signal.
#
# Independence: 4 groups, 4 genuinely different data sources
#   SMC_STRUCTURE:    Market structure (BOS, CHoCH, OB, sweep levels)
#   TREND_FOLLOWING:  Multi-timeframe trend + pullback to EMA21
#   MEAN_REVERSION:   Volume-weighted price (VWAP, POC, Value Area)
#   ORDER_FLOW:       Cumulative delta divergence (tick direction flow)
STRATEGY_GROUPS = {
    "SMC_STRUCTURE": [
        "SMC_OB_REVERSAL", "LIQUIDITY_SWEEP_ENTRY", "FVG_REVERSION"],
    "TREND_FOLLOWING": [
        "TREND_CONTINUATION"],
    "MEAN_REVERSION": [
        "VWAP_MEAN_REVERSION"],
    "ORDER_FLOW": [
        "DELTA_DIVERGENCE"],
}


def run_strategies(symbol: str,
                   master_report: dict,
                   external_data: dict = None) -> dict | None:
    """
    Run strategies on one symbol and return the best confirmed signal.

    Signal is only returned when:
      1. Master score >= 45 (market conditions acceptable)
      2. At least one INSTITUTIONAL gate passes (OF imbalance or volume surge)
      3. At least 2 strategies from DIFFERENT groups agree on direction
      4. Best signal score >= its own strategy minimum threshold
      5. Direction matches master combined_bias (if bias is clear)
    """
    if master_report is None:
        return None

    market_report = master_report.get('market_report', {})
    smc_report    = master_report.get('smc_report', {})
    market_state  = master_report.get('market_state', 'BALANCED')
    session       = master_report.get('session', 'UNKNOWN')
    final_score   = master_report.get('final_score', 0)
    combined_bias = master_report.get('combined_bias', 'NEUTRAL')

    # Gate 1: Market score minimum
    if final_score < 45:
        log.debug(f"[ENGINE] {symbol} — final_score {final_score} < 45, skip")
        return None

    # Gate 2: Institutional confirmation check (STRICT)
    # FIX: Momentum alone (is_scalpable) does NOT confirm institutional activity.
    # We need ORDER FLOW or VOLUME SURGE — these prove institutions are moving.
    # Momentum can happen on retail noise. This was the #1 false signal source.
    of_imb    = master_report.get('order_flow_imbalance', {})
    surge     = master_report.get('volume_surge', {})
    momentum  = master_report.get('momentum', {})

    imb_value    = of_imb.get('imbalance', 0)
    imb_strength = of_imb.get('strength', 'NONE')
    surge_active = surge.get('surge_detected', False)
    is_scalpable = momentum.get('is_scalpable', False)
    is_choppy    = momentum.get('is_choppy', True)

    # FIXED: Require order flow OR volume surge (not just momentum)
    has_order_flow = imb_strength in ('STRONG', 'EXTREME')
    has_volume = surge_active

    institutional_confirmed = has_order_flow or has_volume

    # Momentum is a BONUS confirm, not a standalone gate.
    # But if both OF and volume are present, that's the strongest setup.
    if not institutional_confirmed:
        log.debug(f"[ENGINE] {symbol} — no institutional activity "
                  f"(imb={imb_value:+.2f}/{imb_strength}, surge={has_volume}, "
                  f"scalpable={is_scalpable}) — need OF or volume")
        return None

    # Hard gate: never trade choppy markets
    if is_choppy and not surge_active:
        log.debug(f"[ENGINE] {symbol} — choppy + no surge, skip")
        return None

    # Fetch candles
    df_m1  = get_candles(symbol, 'M1',  100)
    df_m5  = get_candles(symbol, 'M5',  200)
    df_m15 = get_candles(symbol, 'M15', 200)
    df_h1  = get_candles(symbol, 'H1',  200)
    df_h4  = get_candles(symbol, 'H4',  100)

    if df_m15 is None or df_h1 is None:
        log.warning(f"[ENGINE] {symbol} — missing candle data")
        return None

    active  = get_active_strategies()
    signals = []

    for strategy_name in active:
        try:
            signal = _run_one_strategy(
                strategy_name, symbol,
                df_m1, df_m5, df_m15, df_h1, df_h4,
                smc_report, market_report,
                market_state, session, master_report)

            if signal is None:
                continue

            direction = str(signal.get('direction', ''))
            score     = signal.get('score', 0)

            # Apply per-strategy minimum score
            min_score = STRATEGY_MIN_SCORES.get(strategy_name, 70)
            if score < min_score:
                log.debug(f"[ENGINE] {symbol} {strategy_name} "
                          f"score {score} < {min_score}, skip")
                continue

            signal['symbol']       = symbol
            signal['market_state'] = market_state
            signal['session']      = session
            signal['timestamp']    = datetime.now(
                timezone.utc).strftime('%H:%M:%S UTC')
            signal['direction']    = direction
            signal['group']        = _get_strategy_group(strategy_name)
            signals.append(signal)

            log.info(f"[ENGINE] {symbol} {strategy_name} "
                     f"{direction} score={score} "
                     f"[{signal['group']}]")

        except Exception as e:
            log.error(f"[ENGINE] Error in {strategy_name}: {e}")

    if not signals:
        return None

    # Gate 3: Bias direction filter
    if combined_bias == 'BULLISH':
        signals = [s for s in signals if s['direction'] == 'BUY']
    elif combined_bias == 'BEARISH':
        signals = [s for s in signals if s['direction'] == 'SELL']

    if not signals:
        log.debug(f"[ENGINE] {symbol} — all signals filtered by bias")
        return None

    # Gate 4: Multi-group consensus
    # Need signals from at least 2 DIFFERENT strategy groups
    # to avoid false consensus from correlated strategies
    buy_groups  = set(s['group'] for s in signals if s['direction'] == 'BUY')
    sell_groups = set(s['group'] for s in signals if s['direction'] == 'SELL')

    buy_signals  = [s for s in signals if s['direction'] == 'BUY']
    sell_signals = [s for s in signals if s['direction'] == 'SELL']

    # FIXED: STRICT multi-group consensus — no fallback.
    # The old code allowed 3+ signals from the SAME group to bypass
    # the cross-validation rule. This defeated the purpose — correlated
    # strategies echoing each other is NOT real consensus.
    # Require 2+ DIFFERENT groups always.
    if len(buy_groups) >= 2 and len(buy_groups) >= len(sell_groups):
        final_signals = buy_signals
        final_groups  = buy_groups
    elif len(sell_groups) >= 2:
        final_signals = sell_signals
        final_groups  = sell_groups
    else:
        log.info(f"[ENGINE] {symbol} — insufficient multi-group consensus "
                 f"(BUY groups:{buy_groups}, SELL groups:{sell_groups})")
        return None

    best = max(final_signals, key=lambda s: s['score'])
    log.info(f"[ENGINE] {symbol} CONFIRMED: {best['strategy']} "
             f"{best['direction']} score={best['score']} "
             f"groups={final_groups} ({len(final_signals)} signals)")
    return best


def _get_strategy_group(name: str) -> str:
    """Return the group category for a strategy."""
    for group, members in STRATEGY_GROUPS.items():
        if name in members:
            return group
    return "OTHER"


def _run_one_strategy(name, symbol,
                      df_m1, df_m5, df_m15, df_h1, df_h4,
                      smc_report, market_report,
                      market_state, session,
                      master_report=None) -> dict | None:
    """
    Route to the correct strategy evaluate function.
    FIXED: hard state gates now actually block execution.
    """
    info          = REGISTRY.get(name, {})
    best_states   = info.get('best_state', [])

    # ── HARD state gates ───────────────────────────────────────
    # Only fire strategies in market states where their edge exists.
    # This prevents strategies firing in wrong conditions.
    HARD_STATE_GATES = {
        "VWAP_MEAN_REVERSION":   ["BALANCED", "REVERSAL_RISK"],
        "DELTA_DIVERGENCE":      ["REVERSAL_RISK", "BREAKOUT_REJECTED",
                                  "BALANCED"],
    }

    if name in HARD_STATE_GATES:
        allowed_states = HARD_STATE_GATES[name]
        if market_state not in allowed_states:
            log.debug(f"[ENGINE] {name} blocked — state {market_state} "
                      f"not in {allowed_states}")
            return None

    # Route to strategy evaluate function
    try:
        if name == "LIQUIDITY_SWEEP_ENTRY":
            return sweep_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report, market_report=market_report)

        elif name == "SMC_OB_REVERSAL":
            return ob_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report, market_report=market_report)

        elif name == "TREND_CONTINUATION":
            return trend_cont_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report,
                market_report=market_report,
                df_h4=df_h4, master_report=master_report)

        elif name == "VWAP_MEAN_REVERSION":
            return vwap_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                market_report=market_report,
                smc_report=smc_report, master_report=master_report)

        elif name == "DELTA_DIVERGENCE":
            return delta_div_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report,
                market_report=market_report,
                master_report=master_report)

        elif name == "FVG_REVERSION":
            return fvg_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report,
                market_report=market_report,
                master_report=master_report)

        else:
            log.debug(f"[ENGINE] {name} is retired or unknown, skip")
            return None

    except Exception as e:
        log.error(f"[ENGINE] {name} on {symbol}: {e}")

    return None
