# =============================================================
# backtest/engine.py
# The main backtest loop. Slides time forward through historical data,
# builds market reports, runs strategies, and tracks trades.
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
from backtest.config import (SYMBOLS, START_DATE, END_DATE, CACHE_DIR,
                              AVG_SPREAD_PIPS, SLIPPAGE_PIPS,
                              STARTING_BALANCE, RISK_PERCENT_PER_TRADE,
                              SCAN_EVERY_N_BARS, STRATEGIES_FILTER)

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


def run_backtest(config: BacktestConfig) -> dict:
    """
    Run a full backtest on one symbol.
    Returns performance summary dict.
    """
    symbol = config.symbol
    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"  BACKTEST: {symbol}")
    log.info(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
    log.info(f"{'='*60}")

    start_time = time_mod.time()

    # ── Load historical data ──────────────────────────────
    data = load_all_data(symbol, config.start_date, config.end_date,
                         CACHE_DIR)
    if not data:
        log.error(f"No data loaded for {symbol}")
        return {}

    df_m1 = data['M1']
    df_m5 = data['M5']
    df_m15 = data['M15']
    df_h1 = data['H1']
    df_h4 = data['H4']

    log.info(f"  M1: {len(df_m1)} bars, H1: {len(df_h1)} bars, H4: {len(df_h4)} bars")

    # ── Pip size ─────────────────────────────────────────
    sym = symbol.upper()
    if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
        pip_size = 1.0
    elif "XAU" in sym:
        pip_size = 0.1
    elif "XAG" in sym:
        pip_size = 0.01
    elif sym_info_digits_match(symbol):
        pip_size = 0.01
    else:
        pip_size = 0.0001

    spread = AVG_SPREAD_PIPS.get(symbol, AVG_SPREAD_PIPS['DEFAULT'])
    total_slippage = spread + SLIPPAGE_PIPS  # Total cost in pips

    # ── Initialize trade tracker ──────────────────────────
    tracker = TradeTracker(
        starting_balance=STARTING_BALANCE
    )

    # ── Import strategy engine ───────────────────────────
    from strategies.strategy_engine import _run_one_strategy, _get_strategy_group
    from strategies.strategy_registry import get_active_strategies

    # Determine which strategies to run
    if config.strategies_filter:
        active_strategies = [s for s in config.strategies_filter
                             if s in get_active_strategies()]
    else:
        active_strategies = get_active_strategies()

    log.info(f"  Strategies: {', '.join(active_strategies)}")
    log.info(f"  Spread: {spread}p, Slippage: {SLIPPAGE_PIPS}p")
    log.info(f"")

    # ── Main backtest loop ────────────────────────────────
    # We scan at every M15 candle close (every 15 M1 bars)
    total_m1_bars = len(df_m1)
    signals_found = 0
    signals_blocked_consensus = 0
    signals_blocked_gate = 0
    trades_executed = 0

    scan_bar = config.scan_every_n_bars  # Start scanning after enough bars for indicators

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
                10.0  # Approximate pip value
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
        market_report = build_market_report(
            s_m15, flow, smc_report, symbol)

        # ── Run strategies ────────────────────────────────
        signals = []
        for strategy_name in active_strategies:
            try:
                signal = _run_one_strategy(
                    strategy_name, symbol,
                    s_m1, s_m5, s_m15, s_h1, s_h4,
                    smc_report, market_report,
                    market_report.get('market_state', 'BALANCED'),
                    market_report.get('session', 'UNKNOWN'),
                    master_report=None  # Simplified
                )

                if signal is None:
                    continue

                direction = str(signal.get('direction', ''))
                score = signal.get('score', 0)

                # Apply per-strategy minimum score (from strategy_engine)
                from strategies.strategy_engine import STRATEGY_MIN_SCORES
                min_score = STRATEGY_MIN_SCORES.get(strategy_name, 70)
                if score < min_score:
                    continue

                signal['symbol'] = symbol
                signal['group'] = _get_strategy_group(strategy_name)
                signals.append(signal)

            except Exception as e:
                # Silently skip strategy errors during backtest
                continue

        if not signals:
            continue

        signals_found += 1

        # ── Bias direction filter ─────────────────────────
        combined_bias = market_report.get('combined_bias', 'NEUTRAL')
        if combined_bias == 'BULLISH':
            signals = [s for s in signals if s['direction'] == 'BUY']
        elif combined_bias == 'BEARISH':
            signals = [s for s in signals if s['direction'] == 'SELL']
        if not signals:
            continue

        # ── Multi-group consensus gate ────────────────────
        buy_groups  = set(s['group'] for s in signals if s['direction'] == 'BUY')
        sell_groups = set(s['group'] for s in signals if s['direction'] == 'SELL')

        if len(buy_groups) >= 2 and len(buy_groups) >= len(sell_groups):
            final_signals = [s for s in signals if s['direction'] == 'BUY']
            final_groups = buy_groups
        elif len(sell_groups) >= 2:
            final_signals = [s for s in signals if s['direction'] == 'SELL']
            final_groups = sell_groups
        else:
            signals_blocked_consensus += 1
            continue

        # ── Institutional confirmation gate ───────────────
        imb_strength = flow.get('order_flow_imbalance', {}).get('strength', 'NONE')
        surge_active = flow.get('volume_surge', {}).get('surge_detected', False)
        has_institutional = imb_strength in ('STRONG', 'EXTREME') or surge_active

        if not has_institutional:
            signals_blocked_gate += 1
            continue

        # ── Best signal ──────────────────────────────────
        best = max(final_signals, key=lambda s: s['score'])

        # ── Confluence check ─────────────────────────────
        confluence = best.get('confluence', [])
        if len(confluence) < 4:  # Slightly relaxed for backtest (live=6)
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

        sl_price = best.get('sl_price', 0)
        tp_price = best.get('tp_price', 0)
        sl_pips  = best.get('sl_pips', 0)
        tp_pips  = best.get('tp1_pips', 0)

        # Validate SL/TP
        if sl_pips <= 0 or tp_pips <= 0:
            continue

        if tp_pips / sl_pips < 2.0:  # Minimum R:R
            continue

        # Set actual SL/TP based on entry price
        if best['direction'] == 'BUY':
            sl_price = entry_price - sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size
        else:
            sl_price = entry_price + sl_pips * pip_size
            tp_price = entry_price - tp_pips * pip_size

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
            market_state=market_report.get('market_state', 'BALANCED'),
        )
        trades_executed += 1

        if trades_executed % 5 == 0:
            log.info(f"  [{symbol}] Trade #{trades_executed} | "
                     f"{best['direction']} {best.get('strategy','')} "
                     f"score={best.get('score',0)} "
                     f"groups={final_groups} | "
                     f"Balance: ${tracker.balance:.2f}")

    # ── Check remaining open trades at end ───────────────
    if tracker.open_trades:
        last_bar = df_m1.iloc[-1]
        for trade in tracker.open_trades:
            trade.exit_time = last_bar['time']
            trade.exit_price = float(last_bar['close'])
            if trade.direction == 'BUY':
                trade.profit_pips = (trade.exit_price - trade.entry_price) / pip_size
            else:
                trade.profit_pips = (trade.entry_price - trade.exit_price) / pip_size
            trade.profit_usd = trade.profit_pips * 10.0 * 0.01
            trade.outcome = 'WIN' if trade.profit_pips > 0 else 'LOSS'
            tracker.balance += trade.profit_usd
            tracker.closed_trades.append(trade)
        tracker.open_trades = []

    elapsed = time_mod.time() - start_time
    summary = tracker.get_summary()
    summary['symbol'] = symbol
    summary['elapsed_seconds'] = round(elapsed, 1)
    summary['signals_found'] = signals_found
    summary['signals_blocked_consensus'] = signals_blocked_consensus
    summary['signals_blocked_gate'] = signals_blocked_gate
    summary['trades_executed'] = trades_executed

    return summary


def sym_info_digits_match(symbol: str) -> bool:
    """Check if symbol is likely a JPY pair."""
    jpy_currencies = ['JPY', 'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY']
    sym = symbol.upper()
    return any(x in sym for x in jpy_currencies)
