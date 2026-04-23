# =============================================================
# backtest/engine.py  v2.0
# The main backtest loop — upgraded to match live system.
#
# Key upgrades vs v1.0:
#  1. Builds proper master_report (matches live master_scanner.py schema)
#  2. Updates feature_store (needed by LIQUIDITY_SWEEP_ENTRY)
#  3. Passes master_report to all strategies
#  4. Final score gate (≥ 45) from market + SMC scores
#  5. Dynamic position sizing with conviction levels
#  6. Partial TP + ATR trailing + dynamic TP extension
#  7. Confluence minimum = 6 (matches live)
#  8. Proper pip value per symbol (not hardcoded $1)
# =============================================================

import datetime
import time as time_mod
from dataclasses import dataclass
from core.logger import get_logger
from backtest.data_loader import load_all_data, get_candles_at_time
from backtest.tick_simulator import simulate_order_flow_from_full
from backtest.smc_builder import build_smc_report
from backtest.market_builder import build_market_report
from backtest.trade_tracker import TradeTracker
from backtest.config import (
    SYMBOLS, START_DATE, END_DATE, CACHE_DIR,
    AVG_SPREAD_PIPS, SLIPPAGE_PIPS, PIP_VALUE_PER_LOT,
    STARTING_BALANCE, SCAN_EVERY_N_BARS, STRATEGIES_FILTER,
    PARTIAL_TP_ENABLED, ATR_TRAIL_ENABLED, DYNAMIC_TP_EXTENSION_ENABLED,
    DYNAMIC_SIZING_ENABLED, BASE_RISK_PERCENT, MIN_CONFLUENCE,
    MIN_RR_RATIO, MASTER_MIN_SCORE, MAX_OPEN_TRADES, MAX_PER_SYMBOL,
    CONVICTION_LOW_SCORE_MAX, CONVICTION_MED_SCORE_MAX,
)

# Relaxed mode overrides
RELAXED_MIN_SCORE = 35
RELAXED_MIN_CONFLUENCE = 4
RELAXED_MIN_RR_RATIO = 1.5
RELAXED_CONSENSUS_GROUPS = 1  # Only 1 strategy group needed

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    symbol: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    scan_every_n_bars: int = 15   # Scan every 15 M1 bars = M15 close
    max_trades: int = 9999        # Max trades to execute (0 = unlimited)
    strategies_filter: list = None  # Empty = all
    relaxed_mode: bool = False     # Lower gates for data collection
    store_db: bool = False         # Store trades/signals in MySQL
    run_id: str = 'default'       # Run identifier for DB grouping
    use_model: bool = False        # Use trained XGBoost model as additional gate


def get_pip_size(symbol: str) -> float:
    """Calculate pip size for a symbol."""
    sym = symbol.upper()
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        return 1.0
    elif "XAU" in sym:
        return 0.1
    elif "XAG" in sym:
        return 0.01
    elif any(x in sym for x in ["JPY"]):
        return 0.01
    else:
        return 0.0001


def _build_master_report(symbol: str,
                         market_report: dict,
                         smc_report: dict,
                         flow_data: dict) -> dict:
    """
    Build a master_report dict that matches the live master_scanner.py schema.
    This ensures all strategies receive the same data structure as in live trading.
    """
    market_score = market_report.get('trade_score', 0)
    smc_score = smc_report.get('smc_score', 0)

    # HTF approved?
    htf = smc_report.get('htf_alignment', {})
    htf_approved = htf.get('approved', True)
    htf_penalty = 0 if htf_approved else 30

    # Premium/discount penalty
    pd = smc_report.get('premium_discount', {})
    pd_bias = str(pd.get('bias', ''))
    market_bias = str(market_report.get('combined_bias', 'NEUTRAL'))
    smc_bias = str(smc_report.get('smc_bias', 'NEUTRAL'))
    pd_penalty = 15 if (
        (market_bias == "BULLISH" and pd_bias == "SELL") or
        (market_bias == "BEARISH" and pd_bias == "BUY")
    ) else 0

    # Final score (same formula as live)
    base_score = (market_score * 0.50) + (smc_score * 0.50)
    final_score = max(0, round(base_score - htf_penalty - pd_penalty))

    # Combined bias
    if market_bias == smc_bias and market_bias != "NEUTRAL":
        combined_bias = market_bias
        bias_confidence = "HIGH"
    elif market_bias == "NEUTRAL" or smc_bias == "NEUTRAL":
        combined_bias = market_bias if market_bias != "NEUTRAL" else smc_bias
        bias_confidence = "MODERATE"
    else:
        combined_bias = "CONFLICTED"
        bias_confidence = "LOW"

    # Sweep alignment
    last_sweep = smc_report.get('last_sweep', {})
    sweep_aligned = last_sweep.get('bias') == combined_bias if last_sweep else False

    # Order flow / volume / momentum shortcuts
    of_imb = flow_data.get('order_flow_imbalance', {})
    volume_surge = flow_data.get('volume_surge', {})
    momentum = flow_data.get('momentum', {})

    # Session
    session = market_report.get('session', 'UNKNOWN')

    return {
        "symbol": symbol,
        "timestamp": "",  # filled later
        "session": session,
        "session_quality": 1.0,
        "day_trade_ok": True,
        "block_reason": "",
        "market_report": market_report,
        "smc_report": smc_report,
        "fractal_alignment": {"score": 0, "passed": True},
        "market_score": market_score,
        "smc_score": smc_score,
        "final_score": final_score,
        "market_bias": market_bias,
        "smc_bias": smc_bias,
        "combined_bias": combined_bias,
        "bias_confidence": bias_confidence,
        "market_state": market_report.get('market_state', 'BALANCED'),
        "htf_approved": htf_approved,
        "pd_penalty": pd_penalty,
        "sweep_aligned": sweep_aligned,
        "recommendation": {"action": "TRADE", "reason": "backtest"},
        "order_flow_imbalance": of_imb,
        "volume_surge": volume_surge,
        "momentum": momentum,
        "scalping_signal": {"status": "INSUFFICIENT", "gates_passed": 0},
        "volume_profile": market_report.get('profile', {}),
        "vwap_context": market_report.get('vwap', {}),
    }


def run_backtest(config: BacktestConfig) -> dict:
    """
    Run a full backtest on one symbol.
    Returns performance summary dict.
    """
    symbol = config.symbol
    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"  BACKTEST v2.0: {symbol}")
    log.info(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
    log.info(f"  Features: PartialTP={PARTIAL_TP_ENABLED} "
             f"Trail={ATR_TRAIL_ENABLED} "
             f"ExtTP={DYNAMIC_TP_EXTENSION_ENABLED} "
             f"DynamicSize={DYNAMIC_SIZING_ENABLED}")
    log.info(f"{'='*60}")

    start_time = time_mod.time()

    # ── Load historical data ──────────────────────────────
    data = load_all_data(symbol, config.start_date, config.end_date, CACHE_DIR)
    if not data:
        log.error(f"No data loaded for {symbol}")
        return {}

    df_m1 = data['M1']
    df_m5 = data['M5']
    df_m15 = data['M15']
    df_h1 = data['H1']
    df_h4 = data['H4']

    log.info(f"  M1: {len(df_m1)} bars, H1: {len(df_h1)} bars, H4: {len(df_h4)} bars")

    # ── Pip size and pip value ────────────────────────────
    pip_size = get_pip_size(symbol)
    pip_value = PIP_VALUE_PER_LOT.get(symbol, PIP_VALUE_PER_LOT['DEFAULT'])

    spread = AVG_SPREAD_PIPS.get(symbol, AVG_SPREAD_PIPS['DEFAULT'])
    total_slippage = spread + SLIPPAGE_PIPS  # Total cost in pips

    log.info(f"  Pip size: {pip_size}, Pip value: ${pip_value}/lot/pip")
    log.info(f"  Spread: {spread}p, Slippage: {SLIPPAGE_PIPS}p, Total cost: {total_slippage}p")

    # ── Initialize trade tracker ──────────────────────────
    tracker = TradeTracker(
        starting_balance=STARTING_BALANCE,
        pip_value_per_lot=pip_value,
        max_open=MAX_OPEN_TRADES,
        max_per_symbol=MAX_PER_SYMBOL,
        partial_tp_enabled=PARTIAL_TP_ENABLED,
        atr_trail_enabled=ATR_TRAIL_ENABLED,
        dynamic_tp_enabled=DYNAMIC_TP_EXTENSION_ENABLED,
        dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
        base_risk_percent=BASE_RISK_PERCENT,
    )

    # ── Import strategy engine ───────────────────────────
    from strategies.strategy_engine import _run_one_strategy, _get_strategy_group
    from strategies.strategy_registry import get_active_strategies
    from data_layer.feature_store import store

    # Determine which strategies to run
    if config.strategies_filter:
        active_strategies = [s for s in config.strategies_filter
                             if s in get_active_strategies()]
    else:
        active_strategies = get_active_strategies()

    log.info(f"  Strategies: {', '.join(active_strategies)}")
    log.info(f"")

    # ── Main backtest loop ────────────────────────────────
    total_m1_bars = len(df_m1)
    signals_found = 0
    signals_blocked_consensus = 0
    signals_blocked_gate = 0
    signals_blocked_score = 0
    trades_executed = 0

    # ── Store feature snapshots per trade (for DB) ─────
    trade_reports = {}  # ticket -> {master_report, market_report, smc_report, flow}

    # ── Load XGBoost model if --use-model ────────────────
    xgb_model = None
    model_blocked_count = 0
    if config.use_model:
        try:
            from ai_engine.xgboost_classifier import is_model_trained
            if is_model_trained():
                import joblib
                from ai_engine.xgboost_classifier import MODEL_PATH, extract_features
                xgb_model = joblib.load(MODEL_PATH)
                log.info(f"  [MODEL] XGBoost model loaded — will filter trades")
            else:
                log.warning(f"  [MODEL] --use-model but no trained model found")
        except Exception as e:
            log.warning(f"  [MODEL] Failed to load model: {e}")

    scan_bar = config.scan_every_n_bars

    for bar_idx in range(scan_bar, total_m1_bars):
        current_bar = df_m1.iloc[bar_idx]
        current_time = current_bar['time']

        # Only scan every N bars (M15 frequency)
        if bar_idx % config.scan_every_n_bars != 0:
            # But still check exits on every bar
            tracker.check_exits(
                current_time,
                float(current_bar['high']),
                float(current_bar['low']),
                float(current_bar['close']),
                pip_size,
                pip_value,
            )
            continue

        # ── Skip if we already have too many trades ───────
        if config.max_trades > 0 and trades_executed >= config.max_trades:
            break

        # ── Get candles UP TO this point (simulates live) ──
        sliced = get_candles_at_time(data, current_time, count=200)

        s_m1  = sliced['M1']
        s_m5  = sliced['M5']
        s_m15 = sliced['M15']
        s_h1  = sliced['H1']
        s_h4  = sliced['H4']

        if len(s_m15) < 30 or len(s_h1) < 30 or len(s_h4) < 10:
            continue

        # ── Build order flow from M1 data ─────────────────
        flow = simulate_order_flow_from_full(s_m1, pip_size)

        # ── Build SMC report ──────────────────────────────
        smc_report = build_smc_report(s_h1, s_h4, current_time)

        # ── Build market report ───────────────────────────
        market_report = build_market_report(s_m15, flow, smc_report, symbol)

        # ── Build master_report (matches live schema) ─────
        master_report = _build_master_report(symbol, market_report, smc_report, flow)
        master_report['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # ── Update feature_store (needed by LIQUIDITY_SWEEP) ──
        market_data_for_store = {
            'current_price': float(current_bar['close']),
            'atr': float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0,
            'delta': flow.get('delta', {}),
            'rolling_delta': flow.get('rolling_delta', {}),
            'order_flow_imbalance': flow.get('order_flow_imbalance', {}),
            'volume_surge': flow.get('volume_surge', {}),
            'momentum': flow.get('momentum', {}),
            'vwap': market_report.get('vwap', {}),
            'profile': market_report.get('profile', {}),
        }
        store.update_symbol_features(symbol, market_data_for_store, smc_report)

        # ── Gate 0: Master score minimum ─────────────────
        final_score = master_report.get('final_score', 0)
        score_gate = RELAXED_MIN_SCORE if config.relaxed_mode else MASTER_MIN_SCORE
        if final_score < score_gate:
            if config.store_db:
                signals_blocked_score += 1
                if signals_blocked_score % 50 == 1:  # Store every ~50th to avoid flooding DB
                    try:
                        from backtest.db_store import store_blocked_signal
                        store_blocked_signal(
                            symbol=symbol, direction='NONE',
                            strategy='PRE_GATE', score=final_score,
                            confluence=[], master_report=master_report,
                            market_report=market_report, smc_report=smc_report,
                            flow_data=flow, was_traded=False,
                            skip_reason=f'score_below_{score_gate}',
                            run_id=config.run_id,
                        )
                    except Exception:
                        pass
            continue

        # ── Gate 1: Institutional confirmation ────────────
        imb_strength = flow.get('order_flow_imbalance', {}).get('strength', 'NONE')
        surge_active = flow.get('volume_surge', {}).get('surge_detected', False)
        has_institutional = imb_strength in ('STRONG', 'EXTREME') or surge_active

        if not has_institutional:
            signals_blocked_gate += 1
            # Store gate-blocked signals for ML (every ~50th to manage volume)
            if config.store_db and signals_blocked_gate % 50 == 1:
                try:
                    from backtest.db_store import store_blocked_signal
                    store_blocked_signal(
                        symbol=symbol, direction='NONE',
                        strategy='PRE_GATE', score=final_score,
                        confluence=[], master_report=master_report,
                        market_report=market_report, smc_report=smc_report,
                        flow_data=flow, was_traded=False,
                        skip_reason='no_institutional_flow',
                        run_id=config.run_id,
                    )
                except Exception:
                    pass
            continue

        # ── Gate 2: Choppy market ─────────────────────────
        is_choppy = flow.get('momentum', {}).get('is_choppy', True)
        if is_choppy and not surge_active:
            if config.store_db:
                try:
                    from backtest.db_store import store_blocked_signal
                    store_blocked_signal(
                        symbol=symbol, direction='NONE',
                        strategy='PRE_GATE', score=final_score,
                        confluence=[], master_report=master_report,
                        market_report=market_report, smc_report=smc_report,
                        flow_data=flow, was_traded=False,
                        skip_reason='choppy_market',
                        run_id=config.run_id,
                    )
                except Exception:
                    pass
            continue

        # ── Run strategies ────────────────────────────────
        market_state = master_report.get('market_state', 'BALANCED')
        session = master_report.get('session', 'UNKNOWN')
        combined_bias = master_report.get('combined_bias', 'NEUTRAL')

        signals = []
        for strategy_name in active_strategies:
            try:
                signal = _run_one_strategy(
                    strategy_name, symbol,
                    s_m1, s_m5, s_m15, s_h1, s_h4,
                    smc_report, market_report,
                    market_state, session, master_report)

                if signal is None:
                    continue

                direction = str(signal.get('direction', ''))
                score = signal.get('score', 0)

                # Apply per-strategy minimum score
                from strategies.strategy_engine import STRATEGY_MIN_SCORES
                min_score = STRATEGY_MIN_SCORES.get(strategy_name, 70)
                if score < min_score:
                    continue

                signal['symbol'] = symbol
                signal['group'] = _get_strategy_group(strategy_name)
                signals.append(signal)

            except Exception as e:
                # Log but don't crash the backtest
                log.debug(f"  [{symbol}] {strategy_name} error: {e}")
                continue

        if not signals:
            continue

        signals_found += 1

        # ── Gate 3: Bias direction filter ─────────────────
        # In relaxed mode, skip this filter to collect more data
        if not config.relaxed_mode:
            if combined_bias == 'BULLISH':
                signals = [s for s in signals if s['direction'] == 'BUY']
            elif combined_bias == 'BEARISH':
                signals = [s for s in signals if s['direction'] == 'SELL']
            if not signals:
                continue

        # ── Gate 4: Multi-group consensus ──────────────────
        buy_groups  = set(s['group'] for s in signals if s['direction'] == 'BUY')
        sell_groups = set(s['group'] for s in signals if s['direction'] == 'SELL')

        min_groups = RELAXED_CONSENSUS_GROUPS if config.relaxed_mode else 2

        if len(buy_groups) >= min_groups and len(buy_groups) >= len(sell_groups):
            final_signals = [s for s in signals if s['direction'] == 'BUY']
            final_groups = buy_groups
        elif len(sell_groups) >= min_groups:
            final_signals = [s for s in signals if s['direction'] == 'SELL']
            final_groups = sell_groups
        else:
            signals_blocked_consensus += 1
            # In relaxed mode, still store the blocked signal for ML
            if config.store_db:
                try:
                    from backtest.db_store import store_blocked_signal
                    for sig in signals:
                        store_blocked_signal(
                            symbol=symbol, direction=sig['direction'],
                            strategy=sig.get('strategy', 'UNKNOWN'),
                            score=sig.get('score', 0),
                            confluence=sig.get('confluence', []),
                            master_report=master_report, market_report=market_report,
                            smc_report=smc_report, flow_data=flow,
                            was_traded=False,
                            skip_reason=f'consensus_blocked(buy={len(buy_groups)},sell={len(sell_groups)})',
                            run_id=config.run_id,
                        )
                except Exception:
                    pass
            continue

        # ── Best signal ──────────────────────────────────
        best = max(final_signals, key=lambda s: s['score'])

        # ── Gate 5: Confluence check ────────────────────
        confluence = best.get('confluence', [])
        min_conv = RELAXED_MIN_CONFLUENCE if config.relaxed_mode else MIN_CONFLUENCE
        if len(confluence) < min_conv:
            continue

        # ── Gate 6: XGBoost model filter (if --use-model) ──
        if xgb_model is not None:
            try:
                from ai_engine.xgboost_classifier import extract_features
                features = extract_features(best, market_report, smc_report)
                if features is not None:
                    win_prob = float(xgb_model.predict_proba(features)[0][1])
                    if win_prob < 0.45:
                        # Model says SKIP — block this trade
                        model_blocked_count += 1
                        if config.store_db:
                            try:
                                from backtest.db_store import store_blocked_signal
                                store_blocked_signal(
                                    symbol=symbol, direction=best['direction'],
                                    strategy=best.get('strategy', 'UNKNOWN'),
                                    score=best.get('score', 0),
                                    confluence=best.get('confluence', []),
                                    master_report=master_report, market_report=market_report,
                                    smc_report=smc_report, flow_data=flow,
                                    was_traded=False,
                                    skip_reason=f'model_skip(prob={win_prob:.2f})',
                                    run_id=config.run_id,
                                )
                            except Exception:
                                pass
                        if model_blocked_count <= 5 or model_blocked_count % 10 == 0:
                            log.info(f"  [MODEL] Blocked {best.get('strategy','')} "
                                     f"{best['direction']} win_prob={win_prob:.2f}")
                        continue
                    else:
                        best['model_probability'] = win_prob
            except Exception as e:
                log.debug(f"  [MODEL] Prediction error: {e}")

        # ── Check if we can open a trade ─────────────────
        if not tracker.can_open(symbol):
            continue

        # ── Execute trade ────────────────────────────────
        entry_price = float(current_bar['close'])

        # Apply spread + slippage
        if best['direction'] == 'BUY':
            entry_price += total_slippage * pip_size
        else:
            entry_price -= total_slippage * pip_size

        sl_pips = best.get('sl_pips', 0)
        tp1_pips = best.get('tp1_pips', 0)

        # Also check tp_pips (some strategies use this field)
        tp_pips = tp1_pips or best.get('tp_pips', 0)

        # Validate SL/TP
        if sl_pips <= 0 or tp_pips <= 0:
            continue

        min_rr = RELAXED_MIN_RR_RATIO if config.relaxed_mode else MIN_RR_RATIO
        if tp_pips / sl_pips < min_rr:
            continue

        # Set actual SL/TP based on entry price
        if best['direction'] == 'BUY':
            sl_price = entry_price - sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size
        else:
            sl_price = entry_price + sl_pips * pip_size
            tp_price = entry_price - tp_pips * pip_size

        # Get ATR for trailing (from M15)
        atr_value = float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0.0

        # Agreement groups count for dynamic sizing
        agreement_groups = len(final_groups)

        tracker.open_trade(
            symbol=symbol,
            direction=best['direction'],
            strategy=best.get('strategy', 'UNKNOWN'),
            entry_time=current_time,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            score=best.get('score', 0),
            confluence=confluence,
            session=market_report.get('session', 'UNKNOWN'),
            market_state=market_state,
            agreement_groups=agreement_groups,
            atr_value=atr_value,
        )
        trades_executed += 1

        # ── Save feature snapshot at trade entry (for DB storage) ─
        trade_reports[tracker.ticket_counter] = {
            'master_report': master_report,
            'market_report': market_report,
            'smc_report': smc_report,
            'flow': flow,
        }

        # ── Store signal + link to trade ticket in DB ────────
        if config.store_db:
            try:
                from backtest.db_store import store_blocked_signal
                # Store the executed signal with trade_ticket link
                store_blocked_signal(
                    symbol=symbol, direction=best['direction'],
                    strategy=best.get('strategy', 'UNKNOWN'),
                    score=best.get('score', 0),
                    confluence=best.get('confluence', []),
                    master_report=master_report, market_report=market_report,
                    smc_report=smc_report, flow_data=flow,
                    was_traded=True, skip_reason='EXECUTED',
                    run_id=config.run_id,
                    trade_ticket=tracker.ticket_counter,
                )
            except Exception:
                pass

        if trades_executed % 5 == 0:
            log.info(f"  [{symbol}] Trade #{trades_executed} | "
                     f"{best['direction']} {best.get('strategy','')} "
                     f"score={best.get('score',0)} "
                     f"groups={final_groups} "
                     f"conviction={tracker.determine_conviction(best.get('score',0), agreement_groups)} "
                     f"| Balance: ${tracker.balance:.2f}")

    # ── Check remaining open trades at end ───────────────
    if tracker.open_trades:
        last_bar = df_m1.iloc[-1]
        tracker.close_remaining_at_end(
            last_bar['time'], float(last_bar['close']), pip_size, pip_value)

    elapsed = time_mod.time() - start_time
    summary = tracker.get_summary()
    summary['symbol'] = symbol
    summary['elapsed_seconds'] = round(elapsed, 1)
    summary['signals_found'] = signals_found
    summary['signals_blocked_consensus'] = signals_blocked_consensus
    summary['signals_blocked_gate'] = signals_blocked_gate
    summary['signals_blocked_score'] = signals_blocked_score
    summary['trades_executed'] = trades_executed
    summary['final_score_avg'] = final_score if trades_executed > 0 else 0
    summary['relaxed_mode'] = config.relaxed_mode
    summary['run_id'] = config.run_id
    summary['model_blocked'] = model_blocked_count

    # ── Store all completed trades + update signal outcomes ─
    if config.store_db:
        try:
            from backtest.db_store import store_trade, update_signal_outcome
            spread = AVG_SPREAD_PIPS.get(symbol, AVG_SPREAD_PIPS['DEFAULT'])
            stored = 0
            for trade in tracker.closed_trades:
                # Use per-trade feature snapshot (captured at entry time)
                reports = trade_reports.get(trade.ticket, {})
                store_trade(
                    trade=trade,
                    master_report=reports.get('master_report'),
                    market_report=reports.get('market_report'),
                    smc_report=reports.get('smc_report'),
                    flow_data=reports.get('flow'),
                    run_id=config.run_id,
                    spread_pips=spread,
                    slippage_pips=SLIPPAGE_PIPS,
                )
                # Also backfill outcome + profit_r into backtest_signals
                update_signal_outcome(trade, run_id=config.run_id)
                stored += 1
            log.info(f"  [DB] Stored {stored} trades in MySQL + updated signal outcomes")
        except Exception as e:
            log.warning(f"  [DB] Could not store trades: {e}")

    return summary


# =============================================================
# PARALLEL MULTI-SYMBOL BACKTEST — processes all pairs on the
# same M1 timeline, exactly like live trading where all pairs
# are scanned simultaneously every M15 bar.
# =============================================================

def run_parallel_backtest(symbols: list, start_date, end_date,
                          scan_every: int = 15, relaxed_mode: bool = False,
                          store_db: bool = False, run_id: str = 'default',
                          max_trades_per_symbol: int = 9999,
                          use_model: bool = False) -> list:
    """
    Run all symbols in parallel on the same M1 timeline.
    Each symbol gets its own TradeTracker, strategies scan independently,
    but all pairs share the same clock — like live trading.
    
    Returns list of per-symbol summary dicts.
    """
    import time as time_mod
    from backtest.data_loader import load_all_data, get_candles_at_time
    from backtest.tick_simulator import simulate_order_flow_from_full
    from backtest.smc_builder import build_smc_report
    from backtest.market_builder import build_market_report
    from backtest.trade_tracker import TradeTracker
    from strategies.strategy_engine import _run_one_strategy, _get_strategy_group
    from strategies.strategy_registry import get_active_strategies

    start_time = time_mod.time()

    active_strategies = get_active_strategies()
    log.info(f"")
    log.info(f"{'='*65}")
    log.info(f"  PARALLEL BACKTEST v3.0 — {len(symbols)} symbols")
    log.info(f"  Period: {start_date.date()} to {end_date.date()}")
    log.info(f"  Strategies: {', '.join(active_strategies)}")
    log.info(f"  Mode: {'RELAXED' if relaxed_mode else 'STRICT'}")
    log.info(f"{'='*65}")

    # ── Load data for all symbols ──────────────────────────
    symbol_data = {}   # symbol -> {M1, M5, M15, H1, H4}
    symbol_trackers = {}  # symbol -> TradeTracker
    symbol_reports = {}   # symbol -> {ticket -> snapshot}
    symbol_pip = {}       # symbol -> pip_size
    symbol_pipval = {}    # symbol -> pip_value
    symbol_spread = {}    # symbol -> total_slippage in pips
    symbol_stats = {}     # symbol -> {signals_found, blocked_gate, blocked_consensus, blocked_score, executed, model_blocked}

    # ── Load XGBoost model if --use-model ────────────────
    xgb_model = None
    if use_model:
        try:
            from ai_engine.xgboost_classifier import is_model_trained
            if is_model_trained():
                import joblib
                from ai_engine.xgboost_classifier import MODEL_PATH, extract_features
                xgb_model = joblib.load(MODEL_PATH)
                log.info(f"  [MODEL] XGBoost model loaded — will filter trades")
            else:
                log.warning(f"  [MODEL] --use-model but no trained model found")
        except Exception as e:
            log.warning(f"  [MODEL] Failed to load model: {e}")

    for sym in symbols:
        data = load_all_data(sym, start_date, end_date, CACHE_DIR)
        if not data:
            log.warning(f"  No data for {sym}, skipping")
            continue

        symbol_data[sym] = data
        pip_size = get_pip_size(sym)
        pip_value = PIP_VALUE_PER_LOT.get(sym, PIP_VALUE_PER_LOT['DEFAULT'])
        spread = AVG_SPREAD_PIPS.get(sym, AVG_SPREAD_PIPS['DEFAULT'])

        symbol_pip[sym] = pip_size
        symbol_pipval[sym] = pip_value
        symbol_spread[sym] = spread + SLIPPAGE_PIPS

        symbol_trackers[sym] = TradeTracker(
            starting_balance=STARTING_BALANCE,
            pip_value_per_lot=pip_value,
            max_open=MAX_OPEN_TRADES,
            max_per_symbol=MAX_PER_SYMBOL,
            partial_tp_enabled=False if relaxed_mode else PARTIAL_TP_ENABLED,
            atr_trail_enabled=False if relaxed_mode else ATR_TRAIL_ENABLED,
            dynamic_tp_enabled=False if relaxed_mode else DYNAMIC_TP_EXTENSION_ENABLED,
            dynamic_sizing_enabled=DYNAMIC_SIZING_ENABLED,
            base_risk_percent=BASE_RISK_PERCENT,
        )
        symbol_reports[sym] = {}
        symbol_stats[sym] = {
            'signals_found': 0, 'blocked_gate': 0,
            'blocked_consensus': 0, 'blocked_score': 0,
            'executed': 0, 'model_blocked': 0,
        }

        log.info(f"  Loaded {sym}: M1={len(data['M1'])} bars, H1={len(data['H1'])} bars")

    if not symbol_data:
        return []

    # ── Find the shortest M1 series (determines loop length) ─
    min_len = min(len(d['M1']) for d in symbol_data.values())
    log.info(f"  Timeline: {min_len} M1 bars (~{min_len // 1440} days)")

    # ── Main parallel loop ──────────────────────────────────
    for bar_idx in range(scan_every, min_len):
        # Get the current time from the first symbol (all share same UTC clock)
        ref_sym = list(symbol_data.keys())[0]
        current_time = symbol_data[ref_sym]['M1'].iloc[bar_idx]['time']

        # ── Check exits on EVERY bar for ALL symbols ───────
        for sym in list(symbol_data.keys()):
            data = symbol_data[sym]
            if bar_idx >= len(data['M1']):
                continue
            bar = data['M1'].iloc[bar_idx]
            symbol_trackers[sym].check_exits(
                current_time,
                float(bar['high']),
                float(bar['low']),
                float(bar['close']),
                symbol_pip[sym],
                symbol_pipval[sym],
            )

        # ── Only run strategy scan every N bars ────────────
        if bar_idx % scan_every != 0:
            continue

        # ── Scan ALL symbols at this timestamp ─────────────
        for sym in list(symbol_data.keys()):
            stats = symbol_stats[sym]
            tracker = symbol_trackers[sym]
            data = symbol_data[sym]

            # Skip if max trades reached
            if max_trades_per_symbol > 0 and stats['executed'] >= max_trades_per_symbol:
                continue

            # Skip if already have open trade for this symbol
            if not tracker.can_open(sym):
                continue

            # Get sliced data up to current time
            if bar_idx >= len(data['M1']):
                continue

            sliced = get_candles_at_time(data, current_time, count=200)
            s_m1, s_m5, s_m15, s_h1, s_h4 = (
                sliced['M1'], sliced['M5'], sliced['M15'],
                sliced['H1'], sliced['H4']
            )

            if len(s_m15) < 30 or len(s_h1) < 30 or len(s_h4) < 10:
                continue

            current_bar = s_m1.iloc[-1]

            # Build reports
            flow = simulate_order_flow_from_full(s_m1, symbol_pip[sym])
            smc_report = build_smc_report(s_h1, s_h4, current_time)
            market_report = build_market_report(s_m15, flow, smc_report, sym)
            master_report = _build_master_report(sym, market_report, smc_report, flow)
            master_report['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

            # Update feature store
            market_data_for_store = {
                'current_price': float(current_bar['close']),
                'atr': float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0,
                'delta': flow.get('delta', {}),
                'rolling_delta': flow.get('rolling_delta', {}),
                'order_flow_imbalance': flow.get('order_flow_imbalance', {}),
                'volume_surge': flow.get('volume_surge', {}),
                'momentum': flow.get('momentum', {}),
                'vwap': market_report.get('vwap', {}),
                'profile': market_report.get('profile', {}),
            }
            store.update_symbol_features(sym, market_data_for_store, smc_report)

            # Gate 0: Score
            final_score = master_report.get('final_score', 0)
            score_gate = RELAXED_MIN_SCORE if relaxed_mode else MASTER_MIN_SCORE
            if final_score < score_gate:
                stats['blocked_score'] += 1
                if store_db and stats['blocked_score'] % 50 == 1:
                    try:
                        from backtest.db_store import store_blocked_signal
                        store_blocked_signal(
                            symbol=sym, direction='NONE',
                            strategy='PRE_GATE', score=final_score,
                            confluence=[], master_report=master_report,
                            market_report=market_report, smc_report=smc_report,
                            flow_data=flow, was_traded=False,
                            skip_reason=f'score_below_{score_gate}',
                            run_id=run_id,
                        )
                    except Exception:
                        pass
                continue

            # Gate 1: Institutional
            imb_strength = flow.get('order_flow_imbalance', {}).get('strength', 'NONE')
            surge_active = flow.get('volume_surge', {}).get('surge_detected', False)
            has_institutional = imb_strength in ('STRONG', 'EXTREME') or surge_active

            if not has_institutional:
                stats['blocked_gate'] += 1
                if store_db and stats['blocked_gate'] % 50 == 1:
                    try:
                        from backtest.db_store import store_blocked_signal
                        store_blocked_signal(
                            symbol=sym, direction='NONE',
                            strategy='PRE_GATE', score=final_score,
                            confluence=[], master_report=master_report,
                            market_report=market_report, smc_report=smc_report,
                            flow_data=flow, was_traded=False,
                            skip_reason='no_institutional_flow',
                            run_id=run_id,
                        )
                    except Exception:
                        pass
                continue

            # Gate 2: Choppy
            is_choppy = flow.get('momentum', {}).get('is_choppy', True)
            if is_choppy and not surge_active:
                if store_db:
                    try:
                        from backtest.db_store import store_blocked_signal
                        store_blocked_signal(
                            symbol=sym, direction='NONE',
                            strategy='PRE_GATE', score=final_score,
                            confluence=[], master_report=master_report,
                            market_report=market_report, smc_report=smc_report,
                            flow_data=flow, was_traded=False,
                            skip_reason='choppy_market',
                            run_id=run_id,
                        )
                    except Exception:
                        pass
                continue

            # Run strategies
            market_state = master_report.get('market_state', 'BALANCED')
            session = master_report.get('session', 'UNKNOWN')
            combined_bias = master_report.get('combined_bias', 'NEUTRAL')

            signals = []
            for strategy_name in active_strategies:
                try:
                    signal = _run_one_strategy(
                        strategy_name, sym,
                        s_m1, s_m5, s_m15, s_h1, s_h4,
                        smc_report, market_report,
                        market_state, session, master_report)
                    if signal is None:
                        continue
                    direction = str(signal.get('direction', ''))
                    score = signal.get('score', 0)
                    from strategies.strategy_engine import STRATEGY_MIN_SCORES
                    min_score = STRATEGY_MIN_SCORES.get(strategy_name, 70)
                    if score < min_score:
                        continue
                    signal['symbol'] = sym
                    signal['group'] = _get_strategy_group(strategy_name)
                    signals.append(signal)
                except Exception:
                    continue

            if not signals:
                continue

            stats['signals_found'] += 1

            # Gate 3: Bias filter (skip in relaxed)
            if not relaxed_mode:
                if combined_bias == 'BULLISH':
                    signals = [s for s in signals if s['direction'] == 'BUY']
                elif combined_bias == 'BEARISH':
                    signals = [s for s in signals if s['direction'] == 'SELL']
                if not signals:
                    continue

            # Gate 4: Consensus
            buy_groups = set(s['group'] for s in signals if s['direction'] == 'BUY')
            sell_groups = set(s['group'] for s in signals if s['direction'] == 'SELL')
            min_groups = RELAXED_CONSENSUS_GROUPS if relaxed_mode else 2

            if len(buy_groups) >= min_groups and len(buy_groups) >= len(sell_groups):
                final_signals = [s for s in signals if s['direction'] == 'BUY']
                final_groups = buy_groups
            elif len(sell_groups) >= min_groups:
                final_signals = [s for s in signals if s['direction'] == 'SELL']
                final_groups = sell_groups
            else:
                stats['blocked_consensus'] += 1
                if store_db:
                    try:
                        from backtest.db_store import store_blocked_signal
                        for sig in signals:
                            store_blocked_signal(
                                symbol=sym, direction=sig['direction'],
                                strategy=sig.get('strategy', 'UNKNOWN'),
                                score=sig.get('score', 0),
                                confluence=sig.get('confluence', []),
                                master_report=master_report, market_report=market_report,
                                smc_report=smc_report, flow_data=flow,
                                was_traded=False,
                                skip_reason=f'consensus_blocked(buy={len(buy_groups)},sell={len(sell_groups)})',
                                run_id=run_id,
                            )
                    except Exception:
                        pass
                continue

            # Best signal
            best = max(final_signals, key=lambda s: s['score'])

            # Gate 5: Confluence
            confluence = best.get('confluence', [])
            min_conv = RELAXED_MIN_CONFLUENCE if relaxed_mode else MIN_CONFLUENCE
            if len(confluence) < min_conv:
                continue

            # Gate 6: XGBoost model filter (if --use-model)
            if xgb_model is not None:
                try:
                    from ai_engine.xgboost_classifier import extract_features
                    features = extract_features(best, market_report, smc_report)
                    if features is not None:
                        win_prob = float(xgb_model.predict_proba(features)[0][1])
                        if win_prob < 0.45:
                            stats['model_blocked'] += 1
                            if store_db:
                                try:
                                    from backtest.db_store import store_blocked_signal
                                    store_blocked_signal(
                                        symbol=sym, direction=best['direction'],
                                        strategy=best.get('strategy', 'UNKNOWN'),
                                        score=best.get('score', 0),
                                        confluence=best.get('confluence', []),
                                        master_report=master_report, market_report=market_report,
                                        smc_report=smc_report, flow_data=flow,
                                        was_traded=False,
                                        skip_reason=f'model_skip(prob={win_prob:.2f})',
                                        run_id=run_id,
                                    )
                                except Exception:
                                    pass
                            continue
                        else:
                            best['model_probability'] = win_prob
                except Exception:
                    pass

            # Execute trade
            entry_price = float(current_bar['close'])
            total_slip = symbol_spread[sym]
            if best['direction'] == 'BUY':
                entry_price += total_slip * symbol_pip[sym]
            else:
                entry_price -= total_slip * symbol_pip[sym]

            sl_pips = best.get('sl_pips', 0)
            tp_pips = best.get('tp1_pips', 0) or best.get('tp_pips', 0)

            if sl_pips <= 0 or tp_pips <= 0:
                continue

            min_rr = RELAXED_MIN_RR_RATIO if relaxed_mode else MIN_RR_RATIO
            if tp_pips / sl_pips < min_rr:
                continue

            if best['direction'] == 'BUY':
                sl_price = entry_price - sl_pips * symbol_pip[sym]
                tp_price = entry_price + tp_pips * symbol_pip[sym]
            else:
                sl_price = entry_price + sl_pips * symbol_pip[sym]
                tp_price = entry_price - tp_pips * symbol_pip[sym]

            atr_value = float(s_m15.iloc[-1].get('atr', 0)) if len(s_m15) > 0 else 0.0
            agreement_groups = len(final_groups)

            tracker.open_trade(
                symbol=sym,
                direction=best['direction'],
                strategy=best.get('strategy', 'UNKNOWN'),
                entry_time=current_time,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                score=best.get('score', 0),
                confluence=confluence,
                session=market_report.get('session', 'UNKNOWN'),
                market_state=market_state,
                agreement_groups=agreement_groups,
                atr_value=atr_value,
            )
            stats['executed'] += 1

            # Save snapshot
            symbol_reports[sym][tracker.ticket_counter] = {
                'master_report': master_report,
                'market_report': market_report,
                'smc_report': smc_report,
                'flow': flow,
            }

            # Store executed signal
            if store_db:
                try:
                    from backtest.db_store import store_blocked_signal
                    store_blocked_signal(
                        symbol=sym, direction=best['direction'],
                        strategy=best.get('strategy', 'UNKNOWN'),
                        score=best.get('score', 0),
                        confluence=best.get('confluence', []),
                        master_report=master_report, market_report=market_report,
                        smc_report=smc_report, flow_data=flow,
                        was_traded=True, skip_reason='EXECUTED',
                        run_id=run_id,
                        trade_ticket=tracker.ticket_counter,
                    )
                except Exception:
                    pass

            log.info(f"  [{sym}] Trade #{stats['executed']} | "
                     f"{best['direction']} {best.get('strategy','')} "
                     f"score={best.get('score',0)} | "
                     f"Balance: ${tracker.balance:.2f}")

    # ── Close remaining open trades ────────────────────────
    for sym in list(symbol_data.keys()):
        tracker = symbol_trackers[sym]
        if tracker.open_trades:
            data = symbol_data[sym]
            last_bar = data['M1'].iloc[-1]
            tracker.close_remaining_at_end(
                last_bar['time'], float(last_bar['close']),
                symbol_pip[sym], symbol_pipval[sym])

    # ── Build per-symbol summaries + store to DB ────────────
    all_results = []
    elapsed = time_mod.time() - start_time

    for sym in list(symbol_data.keys()):
        tracker = symbol_trackers[sym]
        stats = symbol_stats[sym]

        summary = tracker.get_summary()
        summary['symbol'] = sym
        summary['elapsed_seconds'] = round(elapsed, 1)
        summary['signals_found'] = stats['signals_found']
        summary['signals_blocked_consensus'] = stats['blocked_consensus']
        summary['signals_blocked_gate'] = stats['blocked_gate']
        summary['signals_blocked_score'] = stats['blocked_score']
        summary['trades_executed'] = stats['executed']
        summary['final_score_avg'] = 0
        summary['relaxed_mode'] = relaxed_mode
        summary['run_id'] = run_id
        summary['model_blocked'] = stats.get('model_blocked', 0)

        # Store trades to DB
        if store_db:
            try:
                from backtest.db_store import store_trade, update_signal_outcome
                spread = AVG_SPREAD_PIPS.get(sym, AVG_SPREAD_PIPS['DEFAULT'])
                stored = 0
                for trade in tracker.closed_trades:
                    reports = symbol_reports.get(sym, {}).get(trade.ticket, {})
                    store_trade(
                        trade=trade,
                        master_report=reports.get('master_report'),
                        market_report=reports.get('market_report'),
                        smc_report=reports.get('smc_report'),
                        flow_data=reports.get('flow'),
                        run_id=run_id,
                        spread_pips=spread,
                        slippage_pips=SLIPPAGE_PIPS,
                    )
                    update_signal_outcome(trade, run_id=run_id)
                    stored += 1
                log.info(f"  [DB] {sym}: Stored {stored} trades in MySQL")
            except Exception as e:
                log.warning(f"  [DB] {sym}: Could not store trades: {e}")

        all_results.append(summary)

    log.info(f"\n  Parallel backtest completed in {elapsed:.1f}s")
    return all_results
