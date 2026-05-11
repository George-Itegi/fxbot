"""
Deriv Over/Under Bot — Multi-Market Orchestrator v5
=====================================================
Multiple MarketWorkers -> MarketSelector (bandit) -> shared RiskManager -> Execute -> Learn

Usage:
  python main.py                                    # Trade all default markets
  python main.py --markets 1HZ100V 1HZ50V R_100    # Specific markets
  python main.py --symbol 1HZ100V                  # Single market (backward compatible)
"""

import argparse
import asyncio
import signal as sig
import sys
import time
from typing import Optional

from config import (DEFAULT_SYMBOL, INITIAL_BANKROLL, DEFAULT_MARKETS,
                    OVER_BARRIER, UNDER_BARRIER,
                    MODEL_SNAPSHOT_INTERVAL, MODEL_TYPE, MIN_STAKE,
                    MAX_OPEN_POSITIONS, MIN_TRADE_INTERVAL_SEC,
                    get_symbol_decimals, get_symbol_category,
                    supports_digit_contracts, SYMBOLS, VALID_MULTI_MARKET_SYMBOLS)
from data.deriv_ws import DerivWS
from trading.market_worker import MarketWorker
from trading.market_selector import MarketSelector
from trading.risk_manager import RiskManager
from trading.stake_manager import StakeManager
from trading.trade_logger import TradeLogger
from utils.logger import setup_logger
from utils.metrics import PerformanceTracker, TradeRecord

logger = setup_logger("main")


class DerivBot:
    """Multi-market bot orchestrator — LIVE trading on demo account."""

    def __init__(self, markets: list = None, model_type: str = None,
                 bankroll: float = None):
        self.markets = markets or DEFAULT_MARKETS
        self.model_type = model_type or MODEL_TYPE

        for m in self.markets:
            if not supports_digit_contracts(m):
                logger.warning(f"  {m} may NOT support Digit Over/Under contracts!")

        self.ws = DerivWS()
        effective_bankroll = bankroll or INITIAL_BANKROLL
        self.risk_mgr = RiskManager(effective_bankroll)
        self.stake_mgr = StakeManager(initial_bankroll=effective_bankroll)
        self.trade_logger = TradeLogger()
        self.perf_tracker = PerformanceTracker()
        self.selector = MarketSelector()

        self.workers: dict[str, MarketWorker] = {}

        self.running = False
        self._shutdown_done = False
        self._active_trade_symbol: Optional[str] = None
        self._contract_to_symbol: dict[int, str] = {}
        self._real_balance: float = 0.0
        self._global_trade_counter = 0
        self._last_trade_time: float = 0.0

        self.ws.on_tick = self._on_tick
        self.ws.on_trade_result = self._on_trade_result
        self.ws.on_balance = self._on_balance
        self.ws.on_error = self._on_error

        logger.info(
            f"DerivBot v5 initialized: {len(self.markets)} markets, "
            f"LIVE mode, model={self.model_type}"
        )
        logger.info(f"  Markets: {', '.join(self.markets)}")

    async def start(self):
        logger.info("=" * 65)
        logger.info("  DERIV OVER/UNDER BOT v5 — MULTI-MARKET LIVE TRADING")
        logger.info(f"  Markets:    {', '.join(self.markets)}")
        logger.info(f"  Mode:       LIVE (demo account)")
        logger.info(f"  Model:      {self.model_type}")
        logger.info(f"  Selector:   Market bandit (epsilon=10%)")
        logger.info(f"  Bankroll:   ${self.risk_mgr.bankroll:.2f}")
        logger.info(f"  Barrier:    Over {OVER_BARRIER} / Under {UNDER_BARRIER}")
        logger.info(f"  Duration:   1-10t (dynamic per market)")
        logger.info("")
        logger.info("  Circuit breaker: 10 losses -> 30s cooldown")
        logger.info("  Dynamic stakes: confidence + agreement + drawdown recovery")
        logger.info("  No daily trade limit (demo training mode)")
        logger.info("  No session time limit (demo training mode)")
        logger.info("  ALL TRADES ARE REAL on your Deriv demo account")
        logger.info("=" * 65)

        self.running = True

        connected = await self.ws.connect()
        if not connected:
            logger.error("Failed to connect to Deriv. Exiting.")
            return

        logger.info("Fetching symbol metadata from Deriv API...")
        try:
            decimal_map = await self.ws.get_active_symbols()
            for symbol in self.markets:
                if symbol in decimal_map:
                    logger.info(f"  {symbol}: {decimal_map[symbol]} decimal places (API)")
        except Exception as e:
            logger.warning(f"Failed to fetch symbol metadata: {e}")

        await self._fetch_balance()
        await self.ws.subscribe_to_balance()

        for symbol in self.markets:
            worker = MarketWorker(symbol, self.ws, model_type=self.model_type)
            worker._bankroll = self.risk_mgr.bankroll
            self.workers[symbol] = worker

            logger.info(f"Warming up {symbol}...")
            try:
                history = await self.ws.get_tick_history(symbol, count=1000)
                await worker.warmup(history)
            except Exception as e:
                logger.error(f"{symbol} warmup failed: {e}")

        for symbol in self.markets:
            await self.ws.subscribe_ticks(symbol)

        logger.info(f"All {len(self.markets)} markets subscribed and ready!")

        try:
            while self.running:
                await asyncio.sleep(1)

                if (self._global_trade_counter > 0 and
                        self._global_trade_counter % MODEL_SNAPSHOT_INTERVAL == 0):
                    await self._save_all_states()

                today = time.strftime("%Y-%m-%d")
                if today != self.risk_mgr._last_day:
                    self.risk_mgr.reset_daily()
                    for worker in self.workers.values():
                        worker.signal_gen.reset_daily()

                for symbol, worker in self.workers.items():
                    if worker.live_tick_count % 500 == 0 and worker.live_tick_count > 0:
                        await worker._fetch_payout_rate()

                total_ticks = sum(w.live_tick_count for w in self.workers.values())
                if total_ticks % 300 == 0 and total_ticks > 0:
                    await self._fetch_balance()

        except asyncio.CancelledError:
            logger.info("Bot shutdown requested")
        finally:
            await self.shutdown()

    async def _fetch_balance(self):
        if not self.ws or not self.ws.is_connected:
            return
        try:
            response = await self.ws._send({"balance": 1}, timeout=10)
            balance_data = response.get("balance", {})
            balance = float(balance_data.get("balance", 0))
            currency = balance_data.get("currency", "USD")

            if balance > 0:
                self._real_balance = balance
                self.risk_mgr.update_bankroll(balance)
                if self.risk_mgr.initial_bankroll == INITIAL_BANKROLL:
                    self.risk_mgr.initial_bankroll = balance
                logger.info(f"Real balance: ${balance:.2f} {currency}")
        except Exception as e:
            logger.debug(f"Balance fetch failed: {e}")

    def _on_balance(self, balance_data: dict):
        balance = float(balance_data.get("balance", 0))
        if balance > 0:
            self._real_balance = balance
            self.risk_mgr.update_bankroll(balance)
            # Update stake manager's drawdown tracking
            self.stake_mgr._update_drawdown(balance)
            # Push bankroll to all workers so they can calculate dynamic stakes
            for worker in self.workers.values():
                worker._bankroll = balance

    async def _on_tick(self, symbol: str, tick_data: dict):
        try:
            worker = self.workers.get(symbol)
            if worker is None:
                return

            signal = worker.process_tick(tick_data)

            if signal is not None and self._active_trade_symbol is None:
                if time.time() - self._last_trade_time < MIN_TRADE_INTERVAL_SEC:
                    return
                # CRITICAL: Fire as separate task to avoid async deadlock.
                # If we await here, the recv loop is blocked and can't read
                # the proposal response → 10s timeout.
                asyncio.create_task(self._evaluate_and_trade())

        except Exception as e:
            logger.error(f"Error in _on_tick ({symbol}): {e}", exc_info=True)

    async def _evaluate_and_trade(self):
        if self._active_trade_symbol is not None:
            return

        best_symbol = self.selector.select_market(self.workers)
        if best_symbol is None:
            return

        worker = self.workers[best_symbol]
        signal = worker.get_fresh_signal()
        if signal is None:
            return

        # Dynamic stake sizing via StakeManager
        # Uses confidence, agreement, drawdown recovery, and win streaks
        dynamic_stake = self.stake_mgr.calculate_stake(
            signal=signal,
            bankroll=self.risk_mgr.bankroll,
            payout=worker.current_payout,
        )
        signal.stake = dynamic_stake

        # Log stake breakdown for transparency
        breakdown = self.stake_mgr.state.last_stake_breakdown
        if breakdown and len(breakdown) > 2:
            logger.info(
                f"STAKE: ${dynamic_stake:.2f} = "
                f"base=${breakdown.get('base_stake', 0):.2f} × "
                f"conf={breakdown.get('confidence_mult', 1):.1f} × "
                f"agree={breakdown.get('agreement_mult', 1):.1f} × "
                f"dd={breakdown.get('drawdown_factor', 1):.2f} × "
                f"streak={breakdown.get('streak_factor', 1):.1f} × "
                f"ev={breakdown.get('ev_factor', 1):.1f} "
                f"{'[RECOVERY]' if breakdown.get('recovery_mode') else ''}"
            )

        risk_decision = self.risk_mgr.can_trade(signal)
        self.trade_logger.log_risk_decision(signal, risk_decision)

        if not risk_decision.approved:
            worker.clear_signal()
            return

        self._active_trade_symbol = best_symbol
        self.risk_mgr.open_positions += 1
        self.trade_logger.log_signal(signal)

        try:
            result = await worker.execute_trade(signal, risk_decision.adjusted_stake)
            self.trade_logger.log_execution(signal, result)

            if result.success:
                if result.contract_id:
                    self._contract_to_symbol[result.contract_id] = best_symbol

                if result.payout > 0:
                    real_payout_rate = result.payout / result.stake
                    if 0.5 < real_payout_rate < 1.5:
                        worker.current_payout = real_payout_rate

                logger.info(
                    f"TRADE OPEN: {best_symbol} {signal.direction} "
                    f"barrier={signal.barrier} stake=${result.stake:.2f} "
                    f"payout=${result.payout:.2f} dur={signal.contract_duration}t "
                    f"contract={result.contract_id}"
                )
            else:
                logger.warning(f"Trade FAILED ({best_symbol}): {result.error}")
                self.risk_mgr.open_positions -= 1
                self._active_trade_symbol = None

        except Exception as e:
            logger.error(f"Execute error ({best_symbol}): {e}", exc_info=True)
            self.risk_mgr.open_positions -= 1
            self._active_trade_symbol = None

    async def _on_trade_result(self, contract_data: dict):
        contract_id = contract_data.get("contract_id")
        symbol = self._contract_to_symbol.pop(contract_id, None)

        if symbol is None:
            for sym, worker in self.workers.items():
                if contract_id in worker.executor.pending_contracts:
                    symbol = sym
                    break

        if symbol is None:
            return

        worker = self.workers.get(symbol)
        if worker is None:
            return

        result = worker.handle_trade_result(contract_data)
        if result is None:
            return

        signal, won, payout = result
        stake = signal.stake

        self.risk_mgr.record_outcome(won, stake, payout)
        self.stake_mgr.record_outcome(won, stake, payout, self.risk_mgr.bankroll)
        self._global_trade_counter += 1
        worker.trade_counter += 1

        pnl = payout - stake if won else -stake
        self.selector.record_outcome(symbol, won, pnl)

        trade = TradeRecord(
            trade_id=self._global_trade_counter,
            timestamp=signal.timestamp,
            symbol=symbol,
            direction=signal.direction,
            barrier=signal.barrier,
            confidence=signal.confidence,
            expected_value=signal.expected_value,
            stake=stake,
            payout=payout - stake if won else 0,
            won=won,
            balance_after=self.risk_mgr.bankroll,
            notes=f"contract={contract_id}",
        )
        self.perf_tracker.record_trade(trade)
        self.trade_logger.log_outcome(signal, won, payout, self.risk_mgr.bankroll)

        drift_event = worker.learn_from_outcome(signal, won, payout)

        if drift_event:
            self.risk_mgr.set_drift_state(drift_event.severity == "critical")
        else:
            self.risk_mgr.set_drift_state(False)

        self._active_trade_symbol = None
        self._last_trade_time = time.time()

        if self._global_trade_counter % 10 == 0:
            self._log_status()

        # Log profitability check periodically
        if self._global_trade_counter % 20 == 0:
            profit_check = self.stake_mgr.is_profitable_at_current_win_rate(worker.current_payout)
            logger.info(
                f"PROFIT CHECK: wr={profit_check['win_rate']:.1%} "
                f"breakeven={profit_check['breakeven_wr']:.1%} "
                f"{'PROFITABLE' if profit_check['is_profitable'] else 'LOSING'} "
                f"gap={profit_check['gap']:+.1%} "
                f"ev=${profit_check['ev_per_dollar']:+.4f}/$1"
            )

    def _log_status(self):
        selector_summary = self.selector.summary()
        risk_summary = self.risk_mgr.summary()
        stake_summary = self.stake_mgr.summary()

        logger.info("=" * 65)
        logger.info(f"  TRADE #{self._global_trade_counter} | "
                     f"Balance: ${risk_summary['bankroll']:.2f} | "
                     f"P&L: ${risk_summary['total_pnl']:.2f} | "
                     f"Consec.L: {risk_summary['consecutive_losses']}")
        logger.info(
            f"  Stake Mgr: wr={stake_summary['recent_win_rate']:.1%} "
            f"dd={stake_summary['current_drawdown_pct']}% "
            f"streak=W{stake_summary['consecutive_wins']}/L{stake_summary['consecutive_losses']} "
            f"{'[RECOVERY]' if stake_summary['recovery_mode'] else ''}"
        )
        logger.info(f"  Market Selector: last={selector_summary['last_selected']}")

        for symbol, worker in self.workers.items():
            w_summary = worker.summary()
            logger.info(
                f"  {symbol}: ticks={w_summary['live_ticks']} "
                f"trades={w_summary['trade_count']} "
                f"acc={w_summary['model_accuracy']:.1f}% "
                f"dur={w_summary['best_duration']}{w_summary['duration_unit']} "
                f"payout={w_summary['payout_rate']:.2%} "
                f"drift={w_summary['drift_active']}"
            )

        for symbol, mstats in selector_summary.get("markets", {}).items():
            logger.info(
                f"  Selector {symbol}: wr={mstats['win_rate']:.1%} "
                f"trades={mstats['total_trades']} pnl=${mstats['pnl']:.2f}"
            )
        logger.info("=" * 65)

    def _on_error(self, error_msg: str):
        logger.error(f"WebSocket error: {error_msg}")

    async def _save_all_states(self):
        for symbol, worker in self.workers.items():
            try:
                await worker.save_state()
            except Exception as e:
                logger.error(f"Failed to save state for {symbol}: {e}")

    async def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True

        self.running = False
        logger.info("Shutting down...")

        await self._save_all_states()

        risk_summary = self.risk_mgr.summary()
        selector_summary = self.selector.summary()
        stake_summary = self.stake_mgr.summary()

        logger.info("=" * 65)
        logger.info("  SESSION SUMMARY")
        logger.info(f"  Bankroll:       ${risk_summary['bankroll']:.2f}")
        logger.info(f"  Total P&L:      ${risk_summary['total_pnl']:.2f}")
        logger.info(f"  ROI:            {risk_summary['roi']:.1f}%")
        logger.info(f"  Total Trades:   {risk_summary['total_trades']}")
        logger.info(f"  Consec. Losses: {risk_summary['consecutive_losses']}")
        logger.info(
            f"  Stake Mgr:      wr={stake_summary['recent_win_rate']:.1%} "
            f"dd={stake_summary['current_drawdown_pct']}% "
            f"peak=${stake_summary['peak_bankroll']:.2f} "
            f"{'[RECOVERY]' if stake_summary['recovery_mode'] else ''}"
        )
        logger.info("")
        logger.info("  Per-Market:")

        for symbol, worker in self.workers.items():
            model_summary = worker.model.summary()
            logger.info(f"    {symbol}:")
            logger.info(f"      Accuracy: {model_summary['accuracy']:.1f}%")
            logger.info(f"      Updates:  {model_summary['total_updates']}")
            logger.info(f"      Trades:   {worker.trade_counter}")

        logger.info("")
        logger.info("  Market Selector:")
        for symbol, mstats in selector_summary.get("markets", {}).items():
            logger.info(f"    {symbol}: wr={mstats['win_rate']:.1%}, "
                         f"trades={mstats['total_trades']}, pnl=${mstats['pnl']:.2f}")
        logger.info("=" * 65)

        await self.ws.disconnect()
        logger.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Deriv Over/Under Bot v5 — Multi-Market Live Demo Trading"
    )
    parser.add_argument(
        "--markets", nargs="+", default=DEFAULT_MARKETS,
        help=f"Markets to trade (default: {' '.join(DEFAULT_MARKETS)})"
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Single symbol (backward compatible, overrides --markets)"
    )
    parser.add_argument(
        "--bankroll", type=float, default=None,
        help="Starting bankroll override"
    )
    parser.add_argument(
        "--model", choices=["ensemble", "logistic", "hoeffding", "srp"],
        default=MODEL_TYPE,
        help=f"Model type (default: {MODEL_TYPE})"
    )

    args = parser.parse_args()

    markets = args.markets
    if args.symbol:
        markets = [args.symbol]

    for m in markets:
        if m not in SYMBOLS:
            logger.error(f"Unknown symbol: {m}. Valid: {VALID_MULTI_MARKET_SYMBOLS}")
            sys.exit(1)

    bot = DerivBot(
        markets=markets,
        bankroll=args.bankroll,
        model_type=args.model,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        if not bot._shutdown_done:
            logger.info("Ctrl+C received, shutting down...")
            asyncio.create_task(bot.shutdown())

    try:
        sig.signal(sig.SIGINT, lambda s, f: signal_handler())
    except (ValueError, OSError):
        pass

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        if not bot._shutdown_done:
            loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
