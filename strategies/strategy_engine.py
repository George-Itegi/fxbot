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

log = get_logger(__name__)

# Map strategy names to their evaluate functions
STRATEGY_FUNCTIONS = {
    "EMA_TREND_MTF":        ema_evaluate,
    "SMC_OB_REVERSAL":      ob_evaluate,
    "LIQUIDITY_SWEEP_ENTRY":sweep_evaluate,
    "VWAP_MEAN_REVERSION":  vwap_evaluate,
    "ORDER_FLOW_EXHAUSTION":exhaustion_evaluate,
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

    if final_score < 30:
        log.info(f"[ENGINE] {symbol} — score too low ({final_score})")
        return None

    # Fetch candle data
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
                df_m15, df_h1, df_h4,
                smc_report, market_report,
                market_state, session, master_report)

            if signal:
                signal['symbol']       = symbol
                signal['market_state'] = market_state
                signal['session']      = session
                signal['timestamp']    = datetime.now(
                    timezone.utc).strftime('%H:%M:%S UTC')
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
                      df_m15, df_h1, df_h4,
                      smc_report, market_report,
                      market_state, session,
                      master_report=None) -> dict | None:
    """Route to the correct strategy evaluate function."""

    info = REGISTRY.get(name, {})
    best_states   = info.get('best_state', [])
    best_sessions = info.get('best_session', [])

    # Soft filter — warn but still run if outside best conditions
    if best_states and market_state not in best_states:
        log.info(f"[ENGINE] {name} — state {market_state}"
                 f" not ideal (best: {best_states})")

    if name == "EMA_TREND_MTF":
        return ema_evaluate(
            symbol, df_m15, df_h1, df_h4,
            smc_report=smc_report,
            master_report=market_report)

    elif name == "SMC_OB_REVERSAL":
        return ob_evaluate(
            symbol, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report)

    elif name == "LIQUIDITY_SWEEP_ENTRY":
        return sweep_evaluate(
            symbol, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report)

    elif name == "VWAP_MEAN_REVERSION":
        return vwap_evaluate(
            symbol, df_m15, df_h1,
            market_report=market_report,
            smc_report=smc_report,
            master_report=master_report)

    elif name == "ORDER_FLOW_EXHAUSTION":
        return exhaustion_evaluate(
            symbol, df_m15, df_h1,
            smc_report=smc_report,
            market_report=market_report)

    return None
