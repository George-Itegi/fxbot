"""
Deriv Over/Under Bot — Multi-Market Orchestrator v8.1
=====================================================
Structured Quality Trading — mirrors the manual trading process:
1. Trend check (3-window agreement, 3-sigma threshold)
2. Digit frequency Over/Under analysis (PRIMARY direction signal)
3. Observation phase (20-30s watching digit movement -> duration)
4. Execute few high-quality trades, then stop that market
5. Profit target per market session ($50)
6. Single Logistic Regression as ML CONFIRMATION

v8.1 Changes from v8:
- Martingale MUST recover on the SAME market (no switching markets)
- ML disagreement now BLOCKS trades (if setup is strong, ML should agree)
- Multiple trades per market: keep trading a market while setup is good
- Raised setup quality threshold from 0.60 to 0.70
- Removed ML confidence gate during martingale (was blocking ALL recovery)
- After a win, continue trading the same market (don't switch)
- Shorter loss cooldown on the martingale market (2s, not 15-30s)

Usage:
  python main.py                                    # Trade all default markets
  python main.py --markets 1HZ100V 1HZ50V R_100    # Specific markets
  python main.py --symbol 1HZ100V                  # Single market
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
                    ALLOW_MULTIPLE_TRADES, MAX_CONCURRENT_TRADES,
                    TICK_LEARN_ENABLED, TICK_LEARN_INTERVAL,
                    get_symbol_decimals, get_symbol_category,
                    supports_digit_contracts, SYMBOLS, VALID_MULTI_MARKET_SYMBOLS,
                    PROFIT_TARGET_PER_MARKET, MARTINGALE_MIN_CONFIDENCE,
                    MARTINGALE_MIN_SETUP_SCORE, MARTINGALE_SAME_MARKET,
                    MARKET_STICKY_AFTER_TRADE,
                    MIN_SETUP_SCORE, OBSERVATION_PERIOD_SEC,
                    ML_DISAGREEMENT_BLOCKS,
                    DYNAMIC_BARRIERS, CONTRACT_TYPE_OVER, CONTRACT_TYPE_UNDER)
import config as config
from data.deriv_ws import DerivWS
from trading.market_worker import MarketWorker
from trading.market_selector import MarketSelector
from trading.setup_detector import SetupDetector
from trading.risk_manager import RiskManager
from trading.stake_manager import StakeManager
from trading.trade_logger import TradeLogger
from utils.logger import setup_logger
from utils.metrics import PerformanceTracker, TradeRecord

logger = setup_logger("main")


class DerivBot:
    """Multi-market bot orchestrator — v8.1 Structured Quality Trading."""

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
        
        # SHARED setup detector across all workers
        self.setup_detector = SetupDetector()
        self.selector = MarketSelector()

        self.workers: dict[str, MarketWorker] = {}

        self.running = False
        self._shutdown_done = False
        self._active_trades: dict[str, float] = {}
        self._contract_to_symbol: dict[int, str] = {}
        self._real_balance: float = 0.0
        self._global_trade_counter = 0
        self._last_trade_time: float = 0.0
        self._symbol_cooldowns: dict[str, float] = {}
        
        # v8.1: Track the "active market" — we keep trading this market
        # until setup breaks or profit target reached
        self._active_market: Optional[str] = None
        self._active_market_since: float = 0.0
        self._active_market_trades: int = 0

        self._trade_timeout_sec = 60.0

        self.ws.on_tick = self._on_tick
        self.ws.on_trade_result = self._on_trade_result
        self.ws.on_balance = self._on_balance
        self.ws.on_error = self._on_error

        logger.info(
            f"DerivBot v8.1 initialized: {len(self.markets)} markets, "
            f"LIVE mode, model={self.model_type}"
        )
        logger.info(f"  Markets: {', '.join(self.markets)}")

    async def start(self):
        logger.info("=" * 65)
        logger.info("  DERIV OVER/UNDER BOT v10 — CONSERVATIVE EDGE DETECTION")
        logger.info(f"  Markets:    {len(self.markets)} markets")
        for i, m in enumerate(self.markets):
            logger.info(f"    [{i+1}] {m} — {SYMBOLS.get(m, {}).get('name', m)}")
        logger.info(f"  Mode:       LIVE (demo account)")
        logger.info(f"  Model:      {self.model_type} (ML CONFIRMATION — blocks on disagreement)")
        logger.info(f"  Barriers:   MODERATE ONLY — Over 3-6, Under 4-7 (no more Over 7/8 lottery)")
        logger.info(f"  Bankroll:   ${self.risk_mgr.bankroll:.2f}")
        logger.info("")
        logger.info("  v10 Key Changes from v9:")
        logger.info("    - Bayesian shrinkage: observed prob blended with natural (33% natural weight)")
        logger.info("    - z-score 3.0 minimum (was 1.3 — caught noise as edges)")
        logger.info("    - ALL 3 windows must agree (was 2/3)")
        logger.info("    - ML disagreement BLOCKS trades (was 20% reduction)")
        logger.info("    - Fixed 5-tick duration (was chaotic 2t/3t/5t)")
        logger.info("    - 35% min confidence (was 12%!), 8% min EV (was 5%)")
        logger.info("    - Martingale: SAME market + SAME barrier")
        logger.info("")
        logger.info(f"  Confidence:  BAYESIAN-ADJUSTED win probability (not raw observed)")
        logger.info(f"  EV minimum:  8% to trade")
        logger.info(f"  Trend:       SOFT BIAS (boost/penalty, not required)")
        logger.info(f"  Freq edge:   3%+ absolute, 15%+ relative (z-score > 3.0)")
        logger.info(f"  Profit target: ${PROFIT_TARGET_PER_MARKET:.0f} per market session")
        logger.info(f"  Martingale:  2.35x on loss (max 2 steps, SAME market + barrier)")
        logger.info(f"  Trade interval: {MIN_TRADE_INTERVAL_SEC}s minimum")
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
            worker = MarketWorker(
                symbol, self.ws, model_type=self.model_type,
                setup_detector=self.setup_detector,
            )
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

                # Stuck trade detection
                for sym, open_time in list(self._active_trades.items()):
                    elapsed = time.time() - open_time
                    if elapsed > self._trade_timeout_sec:
                        logger.warning(
                            f"STUCK TRADE DETECTED: {sym} open for {elapsed:.0f}s. Auto-clearing."
                        )
                        worker = self.workers.get(sym)
                        if worker:
                            for cid in list(worker.executor.pending_contracts.keys()):
                                self._contract_to_symbol.pop(cid, None)
                            worker.executor.pending_contracts.clear()
                        self._active_trades.pop(sym, None)
                        self.risk_mgr.open_positions = len(self._active_trades)
                        self._last_trade_time = time.time()

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
            self.stake_mgr._update_drawdown(balance)
            for worker in self.workers.values():
                worker._bankroll = balance

    async def _on_tick(self, symbol: str, tick_data: dict):
        try:
            worker = self.workers.get(symbol)
            if worker is None:
                return

            signal = worker.process_tick(tick_data)

            max_positions = MAX_CONCURRENT_TRADES if ALLOW_MULTIPLE_TRADES else 1
            if signal is not None and len(self._active_trades) < max_positions:
                if time.time() - self._last_trade_time < MIN_TRADE_INTERVAL_SEC:
                    return
                if symbol in self._symbol_cooldowns and time.time() < self._symbol_cooldowns[symbol]:
                    return
                asyncio.create_task(self._evaluate_and_trade())

        except Exception as e:
            logger.error(f"Error in _on_tick ({symbol}): {e}", exc_info=True)

    def _should_stay_on_market(self, symbol: str) -> bool:
        """
        v8.1: Check if we should stay on the current active market.
        
        We stay if:
        1. The market still has a valid setup
        2. Profit target not reached
        3. Setup not broken
        """
        if self._active_market != symbol:
            return False
        
        worker = self.workers.get(symbol)
        if worker is None:
            return False
        
        setup = worker._current_setup
        if setup is None or not setup.active:
            return False
        
        session = self.setup_detector.get_session(symbol)
        if session and session.profit_target_reached:
            return False
        
        return True

    async def _evaluate_and_trade(self):
        max_positions = MAX_CONCURRENT_TRADES if ALLOW_MULTIPLE_TRADES else 1
        if len(self._active_trades) >= max_positions:
            return

        best_symbol = None  # v10: Initialize — was missing, caused market hopping bug

        # Push martingale state to all workers
        is_martingale = self.stake_mgr.state.martingale_step > 0
        martingale_dir = self.stake_mgr.state.martingale_direction
        martingale_market = self.stake_mgr.state.martingale_market
        martingale_barrier = self.stake_mgr.state.martingale_barrier  # v9
        
        for worker in self.workers.values():
            worker._is_martingale_active = is_martingale
            worker._martingale_direction = martingale_dir
            worker._martingale_barrier = martingale_barrier  # v9

        # ─── v8.1: MANDATORY MARTINGALE SAME MARKET ───
        # If martingale is active, we MUST recover on the SAME market.
        # No switching markets during recovery — the setup was good on THAT market.
        if is_martingale and MARTINGALE_SAME_MARKET and martingale_market:
            # Check if the martingale market is available (not at profit target, has setup)
            worker = self.workers.get(martingale_market)
            if worker is None:
                logger.warning(f"MARTINGALE MARKET {martingale_market} not found — resetting martingale")
                self.stake_mgr.state.martingale_step = 0
                self.stake_mgr.state.martingale_direction = None
                self.stake_mgr.state.martingale_market = None
                self.stake_mgr.state.martingale_barrier = None  # v9
                return
            
            # Check if the martingale market is tradable
            if not self.setup_detector.is_market_tradable(martingale_market):
                logger.info(
                    f"MARTINGALE MARKET {martingale_market} not tradable "
                    f"(profit target reached or setup broken) — resetting martingale"
                )
                self.stake_mgr.state.martingale_step = 0
                self.stake_mgr.state.martingale_direction = None
                self.stake_mgr.state.martingale_market = None
                self.stake_mgr.state.martingale_barrier = None  # v9
                return
            
            # Check if martingale market has a setup that matches our direction AND barrier
            # v10: Both direction AND barrier must match for martingale recovery
            # If we lost on Over 5, we should recover on Over 5 (same barrier)
            setup = worker._current_setup
            barrier_ok = True
            if martingale_barrier is not None and setup and setup.barrier != martingale_barrier:
                barrier_ok = False
            
            if setup and setup.active and setup.direction == martingale_dir and barrier_ok:
                # Good — we can recover on this market
                signal = worker.get_fresh_signal()
                if signal is not None and signal.direction == martingale_dir:
                    best_symbol = martingale_market
                else:
                    # Setup is active but no fresh signal yet — wait
                    return
            else:
                # Setup on martingale market doesn't match — 
                # the direction or barrier changed, which means our setup broke.
                # Reset martingale (take the loss) rather than chase a different setup
                reason = "direction"
                if not barrier_ok:
                    reason = f"barrier (expected {martingale_barrier}, got {setup.barrier if setup else 'NONE'})"
                logger.info(
                    f"MARTINGALE: Setup on {martingale_market} changed {reason} "
                    f"— resetting martingale, taking the loss"
                )
                self.stake_mgr.state.martingale_step = 0
                self.stake_mgr.state.martingale_direction = None
                self.stake_mgr.state.martingale_market = None
                self.stake_mgr.state.martingale_barrier = None  # v9
                # Fall through to normal market selection below
                is_martingale = False
                martingale_market = None

        # ─── v8.1: MARKET PERSISTENCE ───
        # If we have an active market (recently traded), stay on it if:
        # 1. Setup is still valid
        # 2. Profit target not reached
        # 3. Not too long since last trade
        if not is_martingale and self._active_market and MARKET_STICKY_AFTER_TRADE:
            if self._should_stay_on_market(self._active_market):
                # Stay on the active market — check for a fresh signal
                worker = self.workers.get(self._active_market)
                if worker:
                    signal = worker.get_fresh_signal()
                    if signal is not None:
                        best_symbol = self._active_market
                    else:
                        # No signal yet on active market — wait
                        return
                else:
                    self._active_market = None
            else:
                # Active market's setup broke or profit target reached — find a new one
                old_market = self._active_market
                self._active_market = None
                self._active_market_trades = 0
                logger.info(
                    f"MARKET PERSISTENCE: Leaving {old_market} "
                    f"(setup broken or profit target reached)"
                )

        # If we didn't pick a market from persistence or martingale, use the selector
        # v10: Fixed bug where hasattr checked self.best_symbol (local var, not attribute)
        # This was causing the bot to always go to the selector and hop between markets
        if not is_martingale and best_symbol is None:
            best_symbol = self.selector.select_market(self.workers)
            
            if best_symbol is None:
                return
            
            # New market selected — update active market tracking
            if best_symbol != self._active_market:
                if self._active_market is not None:
                    logger.info(
                        f"MARKET SWITCH: {self._active_market} -> {best_symbol} "
                        f"(new setup found)"
                    )
                self._active_market = best_symbol
                self._active_market_since = time.time()
                self._active_market_trades = 0

        if best_symbol is None:
            return

        if best_symbol in self._active_trades:
            return

        worker = self.workers[best_symbol]
        signal = worker.get_fresh_signal()
        if signal is None:
            return

        dynamic_stake = self.stake_mgr.calculate_stake(
            signal=signal,
            bankroll=self.risk_mgr.bankroll,
            payout=worker.current_payout,
        )
        signal.stake = dynamic_stake

        breakdown = self.stake_mgr.state.last_stake_breakdown
        signal.is_martingale = breakdown.get("mode") == "martingale_recovery" if breakdown else False

        if breakdown and len(breakdown) > 2:
            recovery_tag = " [MARTINGALE RECOVERY]" if signal.is_martingale else ""
            market_tag = f" [{best_symbol}]" if best_symbol != self.selector._last_selected_market else ""
            logger.info(
                f"STAKE: ${dynamic_stake:.2f} = "
                f"base=${breakdown.get('base_stake', 0):.2f} x "
                f"conf={breakdown.get('confidence_mult', 1):.1f} x "
                f"setup={breakdown.get('setup_mult', 1):.1f} x "
                f"streak={breakdown.get('streak_factor', 1):.1f} x "
                f"ev={breakdown.get('ev_factor', 1):.1f}"
                f"{recovery_tag}{market_tag}"
            )

        risk_decision = self.risk_mgr.can_trade(signal)
        self.trade_logger.log_risk_decision(signal, risk_decision)

        if not risk_decision.approved:
            worker.clear_signal()
            return

        self._active_trades[best_symbol] = time.time()
        self.risk_mgr.open_positions = len(self._active_trades)
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
                    f"setup={signal.setup_score:.2f} "
                    f"contract={result.contract_id}"
                )
            else:
                logger.warning(f"Trade FAILED ({best_symbol}): {result.error}")
                self._active_trades.pop(best_symbol, None)
                self.risk_mgr.open_positions = len(self._active_trades)

        except Exception as e:
            logger.error(f"Execute error ({best_symbol}): {e}", exc_info=True)
            self._active_trades.pop(best_symbol, None)
            self.risk_mgr.open_positions = len(self._active_trades)

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
        
        # v9: Pass symbol AND barrier to stake_mgr for martingale tracking
        self.stake_mgr.record_outcome(
            won, stake, payout, self.risk_mgr.bankroll,
            direction=signal.direction, symbol=symbol, barrier=signal.barrier
        )
        
        self._global_trade_counter += 1
        worker.trade_counter += 1
        self._active_market_trades += 1

        # ─── v8.1: Loss cooldown — shorter for martingale market ───
        # On the martingale market, we want to recover quickly (2s cooldown).
        # On other markets, normal cooldown applies.
        is_martingale_market = (
            self.stake_mgr.state.martingale_step > 0 and
            self.stake_mgr.state.martingale_market == symbol
        )
        
        if not won:
            if is_martingale_market:
                # Quick cooldown on the martingale market — we need to recover
                cooldown_time = 2  # 2 seconds between martingale recovery trades
                logger.info(
                    f"LOSS COOLDOWN: {cooldown_time}s on {symbol} (martingale recovery)"
                )
            else:
                consec = self.risk_mgr.consecutive_losses
                if consec >= 3:
                    cooldown_time = 30
                elif consec >= 2:
                    cooldown_time = 15
                else:
                    cooldown_time = 5

            if ALLOW_MULTIPLE_TRADES:
                self._symbol_cooldowns[symbol] = time.time() + cooldown_time
            else:
                self._last_trade_time = time.time() + cooldown_time
        else:
            # Win — very short cooldown, keep trading this market
            self._last_trade_time = time.time()  # No extra delay on win
            logger.info(
                f"WIN on {symbol} — continuing to trade this market "
                f"(session trade #{self._active_market_trades})"
            )

        pnl = payout - stake if won else -stake
        self.selector.record_outcome(symbol, won, pnl)

        # Record in market session
        self.setup_detector.record_session_trade(symbol, won, pnl)

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
            duration=signal.contract_duration,
            notes=f"setup_score={signal.setup_score:.2f} contract={contract_id}",
        )
        self.perf_tracker.record_trade(trade)
        self.trade_logger.log_outcome(signal, won, payout, self.risk_mgr.bankroll)

        drift_event = worker.learn_from_outcome(signal, won, payout)

        if drift_event:
            self.risk_mgr.set_drift_state(drift_event.severity == "critical")
        else:
            self.risk_mgr.set_drift_state(False)

        self._active_trades.pop(symbol, None)

        # ─── v8.1: Market session management ───
        # Check if we should stop trading this market
        session = self.setup_detector.get_session(symbol)
        if session:
            logger.info(
                f"[{symbol}] Session: PnL=${session.session_pnl:.2f} "
                f"trades={session.session_trades} wins={session.session_wins} "
                f"{'TARGET REACHED' if session.profit_target_reached else ''}"
            )
            
            # If profit target reached, clear active market
            if session.profit_target_reached:
                if self._active_market == symbol:
                    logger.info(
                        f"MARKET PERSISTENCE: Leaving {symbol} (profit target ${session.session_pnl:.2f} reached)"
                    )
                    self._active_market = None
                    self._active_market_trades = 0

        # Check if martingale was reset (max steps or direction changed)
        if self.stake_mgr.state.martingale_step == 0:
            # Martingale done (either won or max steps) — 
            # if we just took a max-steps loss on the active market, leave it
            if not won and self._active_market == symbol and self._active_market_trades <= 1:
                # Only 1 trade and it was a total loss (martingale exhausted) — 
                # the setup might not be as good as we thought. Leave.
                self._active_market = None
                self._active_market_trades = 0

        if self._global_trade_counter % 10 == 0:
            self._log_status()

    def _log_status(self):
        selector_summary = self.selector.summary()
        risk_summary = self.risk_mgr.summary()
        stake_summary = self.stake_mgr.summary()

        logger.info("=" * 65)
        active_count = len(self._active_trades)
        active_syms = ", ".join(self._active_trades.keys()) if self._active_trades else "none"
        logger.info(f"  TRADE #{self._global_trade_counter} | "
                     f"Balance: ${risk_summary['bankroll']:.2f} | "
                     f"P&L: ${risk_summary['total_pnl']:.2f} | "
                     f"Consec.L: {risk_summary['consecutive_losses']} | "
                     f"Active: {active_count} [{active_syms}]")
        logger.info(
            f"  Stake Mgr: wr={stake_summary['recent_win_rate']:.1%} "
            f"dd={stake_summary['current_drawdown_pct']}% "
            f"martingale_step={stake_summary['martingale_step']}"
        )
        mart_market = stake_summary.get('martingale_market', '')
        if mart_market:
            logger.info(f"  Martingale market: {mart_market} direction={stake_summary['martingale_direction']}")
        logger.info(f"  Active market: {self._active_market} (trades={self._active_market_trades})")
        logger.info(f"  Market Selector: last={selector_summary['last_selected']}")

        for symbol, worker in self.workers.items():
            w_summary = worker.summary()
            session = self.setup_detector.get_session(symbol)
            session_pnl = f"session=${session.session_pnl:.2f}" if session else ""
            target = " TARGET!" if w_summary.get('profit_target_reached') else ""
            active_tag = " [ACTIVE]" if symbol == self._active_market else ""
            logger.info(
                f"  {symbol}{active_tag}: ticks={w_summary['live_ticks']} "
                f"trades={w_summary['trade_count']} "
                f"setup={w_summary['setup_score']:.2f}({w_summary['setup_direction']}) "
                f"dur={w_summary['best_duration']}{w_summary['duration_unit']} "
                f"{session_pnl}{target}"
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
        stake_summary = self.stake_mgr.summary()

        logger.info("=" * 65)
        logger.info("  SESSION SUMMARY")
        logger.info(f"  Bankroll:       ${risk_summary['bankroll']:.2f}")
        logger.info(f"  Total P&L:      ${risk_summary['total_pnl']:.2f}")
        logger.info(f"  ROI:            {risk_summary['roi']:.1f}%")
        logger.info(f"  Total Trades:   {risk_summary['total_trades']}")
        logger.info("")

        for symbol, worker in self.workers.items():
            model_summary = worker.model.summary()
            session = self.setup_detector.get_session(symbol)
            session_info = ""
            if session:
                session_info = f"session_pnl=${session.session_pnl:.2f} trades={session.session_trades}"
            logger.info(f"    {symbol}: accuracy={model_summary['accuracy']:.1f}% {session_info}")

        logger.info("=" * 65)

        await self.ws.disconnect()
        logger.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Deriv Over/Under Bot v8.1 — Structured Quality Trading"
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
        "--model", choices=["logistic", "ensemble"],
        default=MODEL_TYPE,
        help=f"Model type (default: {MODEL_TYPE} — logistic is recommended for v8)"
    )
    parser.add_argument(
        "--multi-trade", action="store_true", default=False,
        help="Allow multiple markets to trade simultaneously"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Max simultaneous open trades"
    )
    parser.add_argument(
        "--tick-learn", action="store_true", default=False,
        help="Enable per-tick model learning during live trading"
    )
    parser.add_argument(
        "--tick-learn-interval", type=int, default=None,
        help="Learn every Nth tick when --tick-learn is on"
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

    if args.multi_trade:
        config.ALLOW_MULTIPLE_TRADES = True
    if args.max_concurrent is not None:
        config.MAX_CONCURRENT_TRADES = args.max_concurrent
        config.MAX_OPEN_POSITIONS = args.max_concurrent
    elif args.multi_trade and args.max_concurrent is None:
        config.MAX_CONCURRENT_TRADES = 5
        config.MAX_OPEN_POSITIONS = 5

    if args.tick_learn:
        config.TICK_LEARN_ENABLED = True
    if args.tick_learn_interval is not None:
        config.TICK_LEARN_INTERVAL = args.tick_learn_interval

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
