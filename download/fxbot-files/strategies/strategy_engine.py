# =============================================================
# strategies/strategy_engine.py
# Coordinates all strategies. Called every scan cycle.
# Runs all active strategies, returns highest-scoring signal.
# Logs every signal (traded or not) to database.
# =============================================================

import MetaTrader5 as mt5
from datetime import datetime, timezone
from core.logger import get_logger
from data_layer.price_feed import get_candles
from strategies.strategy_registry import get_active_strategies, REGISTRY
from strategies.ema_trend import evaluate as ema_evaluate
from strategies.smc_ob_reversal import evaluate as ob_evaluate
from strategies.liquidity_sweep_entry import evaluate as sweep_evaluate
from strategies.vwap_mean_reversion import evaluate as vwap_evaluate
from strategies.order_flow_exhaustion import evaluate as exhaustion_evaluate
from strategies.m1_momentum_scalp import evaluate as m1_scalp_evaluate
from strategies.opening_range_breakout import evaluate as orb_evaluate
from strategies.delta_divergence import evaluate as delta_div_evaluate
from strategies.trend_continuation import evaluate as trend_cont_evaluate
from strategies.smart_money_footprint import evaluate as smf_evaluate

log = get_logger(__name__)

# Map strategy names to their evaluate functions
STRATEGY_FUNCTIONS = {
    "EMA_TREND_MTF":        ema_evaluate,
    "SMC_OB_REVERSAL":      ob_evaluate,
    "LIQUIDITY_SWEEP_ENTRY":sweep_evaluate,
    "VWAP_MEAN_REVERSION":  vwap_evaluate,
    "ORDER_FLOW_EXHAUSTION":exhaustion_evaluate,
    "M1_MOMENTUM_SCALP":    m1_scalp_evaluate,
    "OPENING_RANGE_BREAKOUT": orb_evaluate,
    "DELTA_DIVERGENCE":     delta_div_evaluate,
    "TREND_CONTINUATION":  trend_cont_evaluate,
    "SMART_MONEY_FOOTPRINT": smf_evaluate,
}

def run_strategies(symbol: str,
                   master_report: dict,
                   external_data: dict = None) -> dict | None:
    """
    Run all active strategies on one symbol.
    Returns the highest-scoring signal or None.
    external_data used only for context — NOT for signal scoring.
    """
    if master_report is None:
        return None

    market_report = master_report.get('market_report', {})
    smc_report    = master_report.get('smc_report', {})
    market_state  = master_report.get('market_state', 'BALANCED')
    session       = master_report.get('session', 'UNKNOWN')
    final_score   = master_report.get('final_score', 0)
    day_trade_ok  = master_report.get('day_trade_ok', True)

    # Hard gate — session/news blocked
    if not day_trade_ok:
        log.info(f"[ENGINE] {symbol} — blocked: "
                 f"{master_report.get('block_reason')}")
        return None

    if final_score < 50:
        log.info(f"[ENGINE] {symbol} — score too low ({final_score})")
        return None

    # Fetch candle data (M1 + M5 + M15 + H1 + H4 for hybrid intraday + scalping)
    df_m1  = get_candles(symbol, 'M1',  100)
    df_m5  = get_candles(symbol, 'M5',  200)
    df_m15 = get_candles(symbol, 'M15', 200)
    df_h1  = get_candles(symbol, 'H1',  200)
    df_h4  = get_candles(symbol, 'H4',  100)

    if df_m15 is None or df_h1 is None:
        log.warning(f"[ENGINE] {symbol} — Missing candle data")
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

            if signal:
                signal['symbol']       = symbol
                signal['market_state'] = market_state
                signal['session']      = session
                signal['timestamp']    = datetime.now(
                    timezone.utc).strftime('%H:%M:%S UTC')
                # Sanitize direction to prevent numpy float leaking
                if 'direction' in signal:
                    signal['direction'] = str(signal['direction'])
                signals.append(signal)
                log.info(f"[ENGINE] {symbol} signal: {strategy_name}"
                         f" {signal['direction']} score={signal['score']}")
        except Exception as e:
            log.error(f"[ENGINE] Error in {strategy_name}: {e}")

    if not signals:
        return None

    best = max(signals, key=lambda s: s['score'])
    log.info(f"[ENGINE] {symbol} BEST: {best['strategy']}"
             f" {best['direction']} score={best['score']}")
    return best

def _run_one_strategy(name, symbol,
                      df_m1, df_m5, df_m15, df_h1, df_h4,
                      smc_report, market_report,
                      market_state, session,
                      master_report=None) -> dict | None:
    """Route to the correct strategy evaluate function."""

    info = REGISTRY.get(name, {})
    best_states   = info.get('best_state', [])
    best_sessions = info.get('best_session', [])

    # ── HARD GATES: Block strategies outside their optimal conditions ──
    # v4.7 FIX: These were soft (log-only) — now they BLOCK the strategy.
    # This prevents a range strategy from firing in a trending market,
    # and a London strategy from firing during Tokyo.

    # Hard gate: market state must match
    if best_states and market_state not in best_states:
        log.info(f"[ENGINE] BLOCKED {name} — state {market_state}"
                 f" not in best_states {best_states}")
        return None

    # Hard gate: session must match (or be a preferred session)
    from config.settings import PREFERRED_SESSIONS
    is_preferred_session = session in PREFERRED_SESSIONS

    if best_sessions and session not in best_sessions and not is_preferred_session:
        log.info(f"[ENGINE] BLOCKED {name} — session {session}"
                 f" not in best_sessions {best_sessions}")
        return None

    if name == "EMA_TREND_MTF":
        return ema_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1, df_h4,
            smc_report=smc_report,
            master_report=market_report)

    elif name == "SMC_OB_REVERSAL":
        return ob_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report)

    elif name == "LIQUIDITY_SWEEP_ENTRY":
        return sweep_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report)

    elif name == "VWAP_MEAN_REVERSION":
        return vwap_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            market_report=market_report,
            smc_report=smc_report,
            master_report=master_report)

    elif name == "ORDER_FLOW_EXHAUSTION":
        return exhaustion_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report)

    elif name == "M1_MOMENTUM_SCALP":
        return m1_scalp_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report,
            master_report=master_report)

    elif name == "OPENING_RANGE_BREAKOUT":
        return orb_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report,
            master_report=master_report)

    elif name == "DELTA_DIVERGENCE":
        return delta_div_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report,
            master_report=master_report)

    elif name == "TREND_CONTINUATION":
        return trend_cont_evaluate(
            symbol, df_m1, df_m5, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report,
            df_h4=df_h4,
            master_report=master_report)

    elif name == "SMART_MONEY_FOOTPRINT":
        return smf_evaluate(
            symbol,
            df_m1=df_m1, df_m5=df_m5, df_m15=df_m15, df_h1=df_h1,
            smc_report=smc_report,
            market_report=market_report,
            master_report=master_report)

    return None
