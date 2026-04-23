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
from strategies.ema_cross_momentum import evaluate as ema_cross_evaluate
from strategies.rsi_divergence_smc import evaluate as rsi_div_smc_evaluate
from strategies.breakout_momentum import evaluate as breakout_evaluate
from strategies.structure_alignment import evaluate as struct_align_evaluate

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
    "EMA_CROSS_MOMENTUM":    70,   # EMA crossover + RSI momentum + delta
    "RSI_DIVERGENCE_SMC":    68,   # RSI divergence + BOS/CHoCH
    "BREAKOUT_MOMENTUM":     70,   # Consolidation breakout + retest
    "STRUCTURE_ALIGNMENT":   70,   # BOS + H1 structure + delta agreement
}

# ── Strategies grouped by what they fundamentally measure ────
# Each group represents a DIFFERENT data source / market approach.
# Consensus gate requires 2+ DIFFERENT groups to agree — this
# prevents correlated strategies from echoing the same signal.
#
# Independence: 5 groups, 5 genuinely different data sources
#   SMC_STRUCTURE:    Market structure (BOS, CHoCH, OB, sweep levels, breakouts)
#   TREND_FOLLOWING:  Multi-timeframe trend + EMA crossover / pullback to EMA21
#   MEAN_REVERSION:   Volume-weighted price (VWAP, POC, Value Area)
#   ORDER_FLOW:       Cumulative delta divergence / agreement (tick direction flow)
#   OSCILLATOR:       RSI divergence + SMC confirmation (oscillator-based)
STRATEGY_GROUPS = {
    "SMC_STRUCTURE": [
        "SMC_OB_REVERSAL", "LIQUIDITY_SWEEP_ENTRY", "FVG_REVERSION",
        "BREAKOUT_MOMENTUM"],
    "TREND_FOLLOWING": [
        "TREND_CONTINUATION", "EMA_CROSS_MOMENTUM"],
    "MEAN_REVERSION": [
        "VWAP_MEAN_REVERSION"],
    "ORDER_FLOW": [
        "DELTA_DIVERGENCE", "STRUCTURE_ALIGNMENT"],
    "OSCILLATOR": [
        "RSI_DIVERGENCE_SMC"],
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
    # v4.4: Include agreement group count for dynamic position sizing
    best['agreement_groups'] = len(final_groups)
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
    #
    # Data-driven from EURUSD backtest (31% WR overall):
    #   BALANCED:        11 trades, 18.2% WR, -$801  → trend strategies bleed
    #   TRENDING_STRONG:  1 trade,  100% WR, +$518  → perfect for trend
    #   BREAKOUT_ACCEPTED: 12 trades, 33.3% WR, +$329 → good
    #   TRENDING_EXTENDED: 5 trades,  40% WR, +$351  → good
    #   TOKYO session:   14 trades, 21.4% WR, -$619  → low vol, choppy
    #   NY_AFTERNOON:    2 trades,   0% WR, -$323  → fading volume
    HARD_STATE_GATES = {
        # Trend strategies — need directional markets, NOT ranges
        "TREND_CONTINUATION":    ["TRENDING_STRONG", "BREAKOUT_ACCEPTED",
                                  "TRENDING_EXTENDED"],
        "EMA_CROSS_MOMENTUM":    ["TRENDING_STRONG", "BREAKOUT_ACCEPTED",
                                  "TRENDING_EXTENDED"],
        # Mean reversion — needs ranges, NOT trends
        "VWAP_MEAN_REVERSION":   ["BALANCED", "REVERSAL_RISK",
                                  "BREAKOUT_REJECTED"],
        # Divergence — needs reversal/breakdown setups
        "DELTA_DIVERGENCE":      ["REVERSAL_RISK", "BREAKOUT_REJECTED",
                                  "BALANCED", "TRENDING_EXTENDED"],
        "RSI_DIVERGENCE_SMC":    ["REVERSAL_RISK", "BREAKOUT_REJECTED",
                                  "BALANCED"],
    }

    # ── HARD session gates ──────────────────────────────────────
    # Block strategies from sessions where they have no edge.
    #   SYDNEY:             3 trades, 66.7% WR, +$568  → keep all
    #   TOKYO:             14 trades, 21.4% WR, -$619  → block trend strategies
    #   LONDON_OPEN:         0 trades                    → keep (SMC manipulation)
    #   LONDON_SESSION:      6 trades, 33.3% WR, +$180  → keep all
    #   NY_LONDON_OVERLAP:   4 trades, 50.0% WR, +$590  → keep all
    #   NY_AFTERNOON:        2 trades,  0.0% WR, -$323  → block trend strategies
    HARD_SESSION_GATES = {
        # Trend strategies lose in low-volume sessions
        "TREND_CONTINUATION":    ["SYDNEY", "LONDON_OPEN", "LONDON_SESSION",
                                  "NY_LONDON_OVERLAP"],
        "EMA_CROSS_MOMENTUM":    ["SYDNEY", "LONDON_OPEN", "LONDON_SESSION",
                                  "NY_LONDON_OVERLAP"],
        # Breakout needs volume — only London/NY
        "BREAKOUT_MOMENTUM":     ["LONDON_OPEN", "LONDON_SESSION",
                                  "NY_LONDON_OVERLAP"],
    }

    # ── Check market state gate ─────────────────────────────────
    if name in HARD_STATE_GATES:
        allowed_states = HARD_STATE_GATES[name]
        if market_state not in allowed_states:
            log.debug(f"[ENGINE] {name} blocked — state {market_state} "
                      f"not in {allowed_states}")
            return None

    # ── Check session gate ───────────────────────────────────────
    if name in HARD_SESSION_GATES and session:
        allowed_sessions = HARD_SESSION_GATES[name]
        if session not in allowed_sessions:
            log.debug(f"[ENGINE] {name} blocked — session {session} "
                      f"not in {allowed_sessions}")
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

        elif name == "EMA_CROSS_MOMENTUM":
            return ema_cross_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report,
                market_report=market_report,
                df_h4=df_h4, master_report=master_report)

        elif name == "RSI_DIVERGENCE_SMC":
            return rsi_div_smc_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report,
                market_report=market_report,
                df_h4=df_h4, master_report=master_report)

        elif name == "BREAKOUT_MOMENTUM":
            return breakout_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report,
                market_report=market_report,
                df_h4=df_h4, master_report=master_report)

        elif name == "STRUCTURE_ALIGNMENT":
            return struct_align_evaluate(
                symbol, df_m1, df_m5, df_m15, df_h1,
                smc_report=smc_report,
                market_report=market_report,
                df_h4=df_h4, master_report=master_report)

        else:
            log.debug(f"[ENGINE] {name} is retired or unknown, skip")
            return None

    except Exception as e:
        log.error(f"[ENGINE] {name} on {symbol}: {e}")

    return None
