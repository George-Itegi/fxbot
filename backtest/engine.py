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
            continue

        # ── Gate 1: Institutional confirmation ────────────
        imb_strength = flow.get('order_flow_imbalance', {}).get('strength', 'NONE')
        surge_active = flow.get('volume_surge', {}).get('surge_detected', False)
        has_institutional = imb_strength in ('STRONG', 'EXTREME') or surge_active

        if not has_institutional:
            signals_blocked_gate += 1
            continue

        # ── Gate 2: Choppy market ─────────────────────────
        is_choppy = flow.get('momentum', {}).get('is_choppy', True)
        if is_choppy and not surge_active:
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

        # ── Store signal + mark as executed in DB ──────────
        if config.store_db:
            try:
                from backtest.db_store import store_blocked_signal, mark_signal_executed
                # Store the executed signal
                store_blocked_signal(
                    symbol=symbol, direction=best['direction'],
                    strategy=best.get('strategy', 'UNKNOWN'),
                    score=best.get('score', 0),
                    confluence=best.get('confluence', []),
                    master_report=master_report, market_report=market_report,
                    smc_report=smc_report, flow_data=flow,
                    was_traded=True, skip_reason='EXECUTED',
                    run_id=config.run_id,
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
