"""
Deriv Over/Under Bot — Main Orchestrator
=========================================
Ties all components together:
  Tick Stream → Aggregator → Feature Engine → Model → Signal → Risk → Execute → Learn

This is the main entry point. Run with:
  python main.py                  # Start bot (paper mode by default)
  python main.py --mode live      # Start in live mode (REAL MONEY)
  python main.py --symbol R_75    # Trade specific symbol
"""

import argparse
import asyncio
import signal as sig
import sys
import time
from typing import Optional

from config import (DEFAULT_SYMBOL, TRADING_MODE, INITIAL_BANKROLL,
                    TICK_WINDOWS, OVER_BARRIER, CONTRACT_DURATION,
                    MODEL_SNAPSHOT_INTERVAL)
from data.deriv_ws import DerivWS
from data.tick_aggregator import TickAggregator
from data.feature_engine import FeatureEngine
from models.online_learner import OverUnderModel
from models.drift_detector import DriftDetector
from models.model_persistence import ModelPersistence
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager
from trading.execution_engine import ExecutionEngine
from trading.trade_logger import TradeLogger
from utils.logger import setup_logger
from utils.metrics import PerformanceTracker, TradeRecord

logger = setup_logger("main")


class DerivBot:
    """
    Main bot orchestrator.
    
    Lifecycle:
    1. Connect to Deriv WebSocket
    2. Stream ticks into aggregator
    3. Compute features every tick
    4. Run model prediction
    5. Generate signal if edge detected
    6. Check risk rules
    7. Execute trade (paper or live)
    8. Feed outcome back to model (online learning)
    9. Repeat
    """
    
    def __init__(self, symbol: str = DEFAULT_SYMBOL, mode: str = None):
        self.symbol = symbol
        self.mode = mode or TRADING_MODE
        
        # ─── Initialize Components ───
        self.ws = DerivWS()
        self.aggregator = TickAggregator(symbol)
        self.feature_engine = FeatureEngine(self.aggregator)
        self.model = OverUnderModel(model_type="logistic")
        self.drift_detector = DriftDetector()
        self.persistence = ModelPersistence()
        self.signal_gen = SignalGenerator()
        self.risk_mgr = RiskManager(INITIAL_BANKROLL)
        self.executor = ExecutionEngine(self.ws)
        self.trade_logger = TradeLogger()
        self.perf_tracker = PerformanceTracker()
        
        # ─── State ───
        self.running = False
        self._pending_signal: Optional = None  # Signal awaiting outcome
        self._current_payout = 0.85  # Default estimate, updated from proposals
        self._ticks_since_last_trade = 0
        self._trade_counter = 0
        self._warmup_done = False
        
        # ─── Wire WebSocket callbacks ───
        self.ws.on_tick = self._on_tick
        self.ws.on_trade_result = self._on_trade_result
        self.ws.on_error = self._on_error
        
        logger.info(f"DerivBot initialized: symbol={symbol}, mode={self.mode}")
    
    async def start(self):
        """Start the bot."""
        logger.info("=" * 60)
        logger.info("  DERIV OVER/UNDER BOT STARTING")
        logger.info(f"  Symbol:    {self.symbol}")
        logger.info(f"  Mode:      {self.mode.upper()}")
        logger.info(f"  Barrier:   Over {OVER_BARRIER} / Under {OVER_BARRIER + 1}")
        logger.info(f"  Duration:  {CONTRACT_DURATION} ticks")
        logger.info(f"  Bankroll:  ${INITIAL_BANKROLL:.2f}")
        logger.info("=" * 60)
        
        self.running = True
        
        # Connect to Deriv
        connected = await self.ws.connect()
        if not connected:
            logger.error("Failed to connect to Deriv. Exiting.")
            return
        
        # Try to load existing model
        snapshots = self.persistence.list_snapshots()
        if snapshots:
            logger.info(f"Found {len(snapshots)} model snapshots")
            if self.persistence.load_snapshot(self.model, snapshots[0]["path"]):
                logger.info("Model loaded from snapshot")
        
        # Fetch historical ticks for warmup
        logger.info("Fetching historical ticks for warmup...")
        history = await self.ws.get_tick_history(self.symbol, count=1000)
        if history:
            for tick in history:
                self.aggregator.add_tick(tick["epoch"], tick["quote"])
            logger.info(f"Warmup: {len(history)} historical ticks loaded")
        
        # Subscribe to live ticks
        await self.ws.subscribe_ticks(self.symbol)
        
        # Run main loop
        try:
            while self.running:
                await asyncio.sleep(1)
                
                # Periodic tasks
                if self._trade_counter > 0 and self._trade_counter % MODEL_SNAPSHOT_INTERVAL == 0:
                    self.persistence.save_snapshot(self.model)
                
                # Reset daily counters
                today = time.strftime("%Y-%m-%d")
                if today != self.risk_mgr._last_day:
                    self.risk_mgr.reset_daily()
                    self.signal_gen.reset_daily()
        
        except asyncio.CancelledError:
            logger.info("Bot shutdown requested")
        finally:
            await self.shutdown()
    
    async def _on_tick(self, symbol: str, tick_data: dict):
        """
        Main tick handler — called for every new tick.
        This is the hot path — keep it fast.
        """
        epoch = tick_data.get("epoch", time.time())
        quote = tick_data.get("quote", 0)
        
        # Add to aggregator
        tick = self.aggregator.add_tick(epoch, quote)
        self._ticks_since_last_trade += 1
        
        # Need enough data before we can compute features
        if not self.aggregator.is_warm("short"):
            return
        
        # Compute features
        features = self.feature_engine.compute_features()
        if features is None:
            return
        
        # ─── Main Trading Logic (runs every tick) ───
        
        # Check cooldown
        from config import COOLDOWN_AFTER_LOSS_TICKS, MIN_TRADE_INTERVAL_SEC
        time_ok = (time.time() - self.signal_gen._last_signal_time) >= MIN_TRADE_INTERVAL_SEC
        
        if not time_ok or self._pending_signal is not None:
            return
        
        # Get prediction
        prediction = self.model.predict(features)
        
        # Only attempt to trade if model is confident enough
        if not prediction.is_tradeable:
            return
        
        # Generate signal
        signal = self.signal_gen.generate(
            prediction=prediction,
            features=features,
            payout=self._current_payout,
            bankroll=self.risk_mgr.bankroll,
            model_in_drift=self.drift_detector.drift_active,
        )
        
        if signal is None:
            return
        
        # Check risk
        risk_decision = self.risk_mgr.can_trade(signal)
        self.trade_logger.log_risk_decision(signal, risk_decision)
        
        if not risk_decision.approved:
            return
        
        # Execute trade
        self.risk_mgr.open_positions += 1
        self._pending_signal = signal
        
        result = await self.executor.execute(signal, risk_decision)
        self.trade_logger.log_signal(signal)
        self.trade_logger.log_execution(signal, result)
        
        if result.success:
            if result.is_paper:
                # Paper trade — immediate outcome
                await self._handle_paper_outcome(signal, result)
            else:
                # Live trade — outcome comes via WebSocket callback
                pass
    
    async def _handle_paper_outcome(self, signal: Signal, result: OrderResult):
        """Handle paper trade outcome immediately."""
        won = result.paper_outcome
        payout = result.payout if won else 0
        
        # Record in risk manager
        self.risk_mgr.record_outcome(won, signal.stake, payout + signal.stake if won else 0)
        
        # Record in performance tracker
        self._trade_counter += 1
        trade = TradeRecord(
            trade_id=self._trade_counter,
            timestamp=signal.timestamp,
            symbol=self.symbol,
            direction=signal.direction,
            barrier=signal.barrier,
            confidence=signal.confidence,
            expected_value=signal.expected_value,
            stake=signal.stake,
            payout=payout,
            won=won,
            balance_after=self.risk_mgr.bankroll,
        )
        self.perf_tracker.record_trade(trade)
        self.trade_logger.log_outcome(signal, won, payout, self.risk_mgr.bankroll)
        
        # Update model (online learning)
        outcome = 1 if won else 0
        self.model.learn_one(signal.features_snapshot, outcome)
        
        # Check drift
        drift_event = self.drift_detector.update(outcome)
        if drift_event:
            self.risk_mgr.set_drift_state(drift_event.severity == "critical")
        
        self._pending_signal = None
        self._ticks_since_last_trade = 0
    
    async def _on_trade_result(self, contract_data: dict):
        """Handle live contract result from WebSocket."""
        result = self.executor.handle_contract_result(contract_data)
        if result is None:
            return
        
        signal, won, payout = result
        
        # Record in risk manager
        self.risk_mgr.record_outcome(won, signal.stake, payout)
        
        # Record in performance tracker
        self._trade_counter += 1
        trade = TradeRecord(
            trade_id=self._trade_counter,
            timestamp=signal.timestamp,
            symbol=self.symbol,
            direction=signal.direction,
            barrier=signal.barrier,
            confidence=signal.confidence,
            expected_value=signal.expected_value,
            stake=signal.stake,
            payout=payout,
            won=won,
            balance_after=self.risk_mgr.bankroll,
            notes=f"contract={contract_data.get('contract_id')}",
        )
        self.perf_tracker.record_trade(trade)
        self.trade_logger.log_outcome(signal, won, payout, self.risk_mgr.bankroll)
        
        # Update model
        outcome = 1 if won else 0
        self.model.learn_with_prediction(
            signal.features_snapshot, outcome,
            # We'd need the prediction from signal time — approximate
            self.model.predict(signal.features_snapshot),
        )
        
        # Check drift
        drift_event = self.drift_detector.update(outcome)
        if drift_event:
            self.risk_mgr.set_drift_state(drift_event.severity == "critical")
        
        self._pending_signal = None
        self._ticks_since_last_trade = 0
    
    def _on_error(self, error_msg: str):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error_msg}")
    
    async def shutdown(self):
        """Graceful shutdown."""
        self.running = False
        logger.info("Shutting down...")
        
        # Save model snapshot
        self.persistence.save_snapshot(self.model)
        
        # Save performance summary
        summary = self.risk_mgr.summary()
        model_summary = self.model.summary()
        
        logger.info("=" * 60)
        logger.info("  SESSION SUMMARY")
        logger.info(f"  Bankroll:     ${summary['bankroll']:.2f}")
        logger.info(f"  Total P&L:    ${summary['total_pnl']:.2f}")
        logger.info(f"  ROI:          {summary['roi']:.1f}%")
        logger.info(f"  Total Trades: {summary['total_trades']}")
        logger.info(f"  Model Acc:    {model_summary['accuracy']:.1f}%")
        logger.info(f"  Drift Events: {model_summary['drift_events']}")
        logger.info("=" * 60)
        
        await self.ws.disconnect()
        logger.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Deriv Over/Under Bot")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL,
                        help=f"Trading symbol (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--mode", choices=["paper", "live"], default=TRADING_MODE,
                        help="Trading mode (default: paper)")
    parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL,
                        help=f"Starting bankroll (default: ${INITIAL_BANKROLL})")
    
    args = parser.parse_args()
    
    # Override config if args provided
    if args.mode:
        import config
        config.TRADING_MODE = args.mode
    if args.bankroll:
        import config
        config.INITIAL_BANKROLL = args.bankroll
    
    bot = DerivBot(symbol=args.symbol, mode=args.mode)
    
    # Handle Ctrl+C
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Ctrl+C received, shutting down...")
        asyncio.create_task(bot.shutdown())
    
    try:
        sig.signal(sig.SIGINT, lambda s, f: signal_handler())
    except (ValueError, OSError):
        pass  # Signal not available on Windows in some contexts
    
    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
