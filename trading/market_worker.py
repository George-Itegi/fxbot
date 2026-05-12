"""
Market Worker — Per-Market State Encapsulation (v8)
====================================================
Integrates SetupDetector for structured quality trading.
Each worker now tracks its own market session (profit target, setup state).
"""

import time
from typing import Optional

from config import (OVER_BARRIER, UNDER_BARRIER, get_symbol_decimals, MIN_STAKE,
                    TICK_LEARN_ENABLED, TICK_LEARN_INTERVAL,
                    DRIFT_RETRAIN_ENABLED, DRIFT_RETRAIN_COOLDOWN,
                    CONTRACT_TYPE_OVER, CONTRACT_TYPE_UNDER,
                    OBSERVATION_PERIOD_SEC,
                    DYNAMIC_BARRIERS)
from data.tick_aggregator import TickAggregator
from data.feature_engine import FeatureEngine
from models.online_learner import OverUnderModel
from models.duration_optimizer import DurationOptimizer
from models.drift_detector import DriftDetector
from models.model_persistence import ModelPersistence
from trading.signal_generator import SignalGenerator, Signal
from trading.setup_detector import SetupDetector, Setup
from trading.execution_engine import ExecutionEngine
from trading.risk_manager import RiskDecision
from utils.logger import setup_logger

logger = setup_logger("trading.market_worker")


class MarketWorker:
    """
    Encapsulates all per-market components for one trading symbol.
    
    v8: Now integrates SetupDetector for trend + digit frequency analysis.
    The setup detector determines IF we should trade and in what direction.
    The ML model (Logistic Regression) is a CONFIRMATION signal.
    """

    def __init__(self, symbol: str, deriv_ws, model_type: str = "logistic",
                 setup_detector: SetupDetector = None):
        self.symbol = symbol
        self.ws = deriv_ws
        self.model_type = model_type
        self._duration_unit = "t"

        configured_dp = get_symbol_decimals(symbol)
        self.aggregator = TickAggregator(symbol, decimal_places=configured_dp)
        self.feature_engine = FeatureEngine(self.aggregator)
        self.model = OverUnderModel(model_type=model_type)
        self.duration_optimizer = DurationOptimizer()
        self.drift_detector = DriftDetector()
        self.persistence = ModelPersistence(symbol=symbol)
        self.signal_gen = SignalGenerator(dynamic_barriers=True)
        self.executor = ExecutionEngine(
            self.ws, symbol=self.symbol, duration_unit=self._duration_unit,
        )

        # Setup detector (shared across workers for session management)
        self.setup_detector = setup_detector or SetupDetector()

        # Current setup state
        self._current_setup: Optional[Setup] = None
        self._observation_started: bool = False
        self._observation_complete: bool = False

        self.current_payout = 0.85
        self.trade_counter = 0
        self.live_tick_count = 0
        self._warmup_done = False
        self._bankroll = 0.0

        self._latest_signal: Optional[Signal] = None
        self._signal_time: float = 0.0
        self._signal_freshness_sec = 5.0

        # Per-tick live learning
        self._tick_learn_enabled = TICK_LEARN_ENABLED
        self._tick_learn_interval = TICK_LEARN_INTERVAL
        self._last_drift_retrain_time: float = 0.0

        # Martingale state (updated from StakeManager via main.py)
        self._is_martingale_active: bool = False
        self._martingale_direction: Optional[str] = None
        self._martingale_barrier: Optional[int] = None  # v9: Barrier from martingale origin

        # Direction Cooldown
        self._blocked_direction: Optional[str] = None
        self._blocked_until: float = 0.0
        self._consecutive_same_dir_losses: int = 0
        self._loss_streak_cooldown_until: float = 0.0

        logger.info(f"MarketWorker created: {symbol}, model={model_type}")

    async def warmup(self, history: list):
        """Process historical ticks for warmup training."""
        if not history:
            logger.warning(f"{self.symbol}: No historical ticks for warmup")
            return

        dp = self.aggregator.decimal_places or get_symbol_decimals(self.symbol)

        # Load model snapshot
        snapshots = self.persistence.list_snapshots(symbol=self.symbol)
        if snapshots:
            loaded = self.persistence.load_state(
                self.model,
                duration_optimizer=self.duration_optimizer,
                feature_engine=self.feature_engine,
                filepath=snapshots[0]["path"],
            )
            if loaded:
                try:
                    import pickle
                    with open(snapshots[0]["path"], "rb") as f:
                        state = pickle.load(f)
                    if state.get("version", 1) >= 2:
                        self.current_payout = state.get("payout_rate", 0.85)
                        self.trade_counter = state.get("trade_counter", 0)
                        self.live_tick_count = state.get("live_tick_count", 0)
                except Exception:
                    pass
                logger.info(f"{self.symbol}: Model loaded from snapshot")

        for tick in history:
            t = self.aggregator.add_tick(tick["epoch"], tick["quote"], decimal_places=dp)
            self.feature_engine.update_streaks(t.digit)
            self.feature_engine.update_markov(t.digit)

        total_transitions = sum(self.feature_engine._transition_total.values())
        logger.info(f"{self.symbol} warmup: {len(history)} ticks, Markov: {total_transitions}")

        if self.aggregator.is_warm("short"):
            trained = 0
            all_ticks = list(self.aggregator.windows.get("long", []))
            for i, tick in enumerate(all_ticks):
                if i < 50:
                    continue
                try:
                    features = self.feature_engine.compute_features()
                    if features is None:
                        continue
                    label = 1 if tick.digit > OVER_BARRIER else 0
                    self.model.learn_one(features, label)
                    trained += 1
                except Exception:
                    pass
            logger.info(f"{self.symbol}: Model trained on {trained} historical samples")

        await self._fetch_payout_rate()
        self._warmup_done = True
        logger.info(f"{self.symbol}: Warmup complete")

    def process_tick(self, tick_data: dict) -> Optional[Signal]:
        """Process a new tick. Returns a Signal if one is generated."""
        try:
            epoch = tick_data.get("epoch", time.time())
            quote = tick_data.get("quote", 0)
            decimal_places = tick_data.get("decimal_places")

            tick = self.aggregator.add_tick(epoch, quote, decimal_places=decimal_places)
            self.live_tick_count += 1

            self.feature_engine.update_streaks(tick.digit)
            self.feature_engine.update_markov(tick.digit)

            if not self.aggregator.is_warm("short"):
                return None

            features = self.feature_engine.compute_features()
            if features is None:
                return None

            # Per-tick live learning
            if self._tick_learn_enabled and self.live_tick_count % self._tick_learn_interval == 0:
                label = 1 if tick.digit > OVER_BARRIER else 0
                self.model.learn_one(features, label)

            # ─── SETUP DETECTION (v8: PRIMARY DECISION) ───
            setup = self.setup_detector.evaluate(self.symbol, features)
            self._current_setup = setup

            # ─── OBSERVATION PHASE ───
            # If a new setup is detected and we haven't started observation yet
            if setup.active and not self._observation_started:
                # v9: Use the barrier from the setup for observation
                obs_direction = 1 if setup.direction == CONTRACT_TYPE_OVER else -1
                self.setup_detector.start_observation(self.symbol, obs_direction, setup.barrier)
                self._observation_started = True
                self._observation_complete = False
            
            # If setup is active and we're observing, feed ticks to observer
            if setup.active and self._observation_started and not self._observation_complete:
                complete = self.setup_detector.observe_tick(self.symbol, tick.digit)
                if complete:
                    self._observation_complete = True
            
            # If setup is no longer active, reset observation
            if not setup.active:
                self._observation_started = False
                self._observation_complete = False
                self.setup_detector.clear_observation(self.symbol)

            # ─── Check market session tradability ───
            if not self.setup_detector.is_market_tradable(self.symbol):
                self._latest_signal = None
                return None

            # ML prediction (confirmation signal)
            prediction = self.model.predict(features)

            # Determine duration from observation phase
            if self._observation_complete:
                duration = self.setup_detector.get_observed_duration(self.symbol)
            else:
                duration = 5  # Default while observing

            # Periodic logging
            if self.live_tick_count % 100 == 0:
                model_summary = self.model.summary()
                setup_score = setup.setup_score if setup.active else 0
                setup_dir = setup.direction.replace("DIGIT", "") if setup.active else "NONE"
                barrier_str = f"{setup_dir}{setup.barrier}" if setup.active else "NONE"
                obs_status = "OBSERVING" if self._observation_started and not self._observation_complete else \
                            "READY" if self._observation_complete else "NO_SETUP"
                
                if setup.active:
                    logger.info(
                        f"[{self.symbol} {self.live_tick_count}t] "
                        f"setup={setup_score:.2f}({barrier_str}) "
                        f"obs={setup.observed_prob:.1%} nat={setup.natural_prob:.0%} "
                        f"z={setup.z_score:.1f} EV={setup.best_barrier_eval.ev:+.1% if setup.best_barrier_eval else 0} "
                        f"pay={setup.payout_rate:.1%} "
                        f"dur={duration}t obs={obs_status}"
                    )
                else:
                    logger.info(
                        f"[{self.symbol} {self.live_tick_count}t] "
                        f"setup={setup_score:.2f}(NONE) "
                        f"dur={duration}t obs={obs_status} "
                        f"payout={self.current_payout:.2%}"
                    )

            # Direction block check
            if self._blocked_direction and time.time() < self._blocked_until:
                if setup.active and setup.direction == self._blocked_direction:
                    self._latest_signal = None
                    return None
            elif time.time() >= self._blocked_until and self._blocked_direction:
                self._blocked_direction = None

            # Generate signal (setup + ML confirmation)
            # v9: Use dynamic payout from setup (varies by barrier)
            dynamic_payout = setup.payout_rate if setup.active else self.current_payout
            signal = self.signal_gen.generate(
                setup=setup,
                prediction=prediction,
                features=features,
                payout=dynamic_payout,
                bankroll=self._bankroll,
                duration=duration,
                model_in_drift=self.drift_detector.drift_active,
                is_martingale=self._is_martingale_active,
                martingale_direction=self._martingale_direction,
                martingale_barrier=self._martingale_barrier,
            )

            if signal is not None:
                self._latest_signal = signal
                self._signal_time = time.time()
                return signal

            self._latest_signal = None
            return None

        except Exception as e:
            logger.error(f"{self.symbol} tick error: {e}", exc_info=True)
            return None

    def get_fresh_signal(self) -> Optional[Signal]:
        if self._latest_signal is None:
            return None
        if time.time() - self._signal_time > self._signal_freshness_sec:
            self._latest_signal = None
            return None
        return self._latest_signal

    def clear_signal(self):
        self._latest_signal = None

    async def execute_trade(self, signal: Signal, stake: float) -> dict:
        risk_decision = RiskDecision(
            approved=True, reason="Setup-based trade approved",
            adjusted_stake=stake, checks={"setup_score": signal.setup_score},
        )
        result = await self.executor.execute(signal, risk_decision)
        return result

    def handle_trade_result(self, contract_data: dict) -> Optional[tuple]:
        return self.executor.handle_contract_result(contract_data)

    def learn_from_outcome(self, signal: Signal, won: bool, payout: float):
        stake = signal.stake
        outcome = 1 if won else 0

        self.model.learn_with_prediction(
            signal.features_snapshot, outcome,
            self.model.predict(signal.features_snapshot),
        )

        self.duration_optimizer.record_result(
            duration=signal.contract_duration, won=won, payout=payout, stake=stake,
        )

        # Direction Cooldown Logic
        if not won:
            if signal.direction == self._blocked_direction or self._blocked_direction is None:
                self._consecutive_same_dir_losses += 1
            else:
                self._consecutive_same_dir_losses = 1

            if self._consecutive_same_dir_losses >= 2:
                self._blocked_direction = signal.direction
                if self._consecutive_same_dir_losses == 2:
                    block_sec = 30
                elif self._consecutive_same_dir_losses == 3:
                    block_sec = 60
                else:
                    block_sec = 120
                self._blocked_until = time.time() + block_sec
                logger.warning(
                    f"DIRECTION BLOCK: {signal.direction} blocked for {block_sec}s "
                    f"({self._consecutive_same_dir_losses} consecutive losses)"
                )

            if self._consecutive_same_dir_losses >= 2:
                cooldown = min(60, 15 * (self._consecutive_same_dir_losses - 1))
                self._loss_streak_cooldown_until = time.time() + cooldown
        else:
            if not (self._blocked_direction and signal.direction != self._blocked_direction):
                self._consecutive_same_dir_losses = 0
                self._blocked_direction = None
                self._blocked_until = 0.0

        drift_event = self.drift_detector.update(outcome)
        if drift_event:
            self.duration_optimizer.on_drift_detected()

            if (DRIFT_RETRAIN_ENABLED
                    and drift_event.severity == "critical"
                    and time.time() - self._last_drift_retrain_time > DRIFT_RETRAIN_COOLDOWN):
                buffer_size = len(self.model.replay_buffer)
                if buffer_size >= 100:
                    logger.warning(
                        f"CRITICAL DRIFT on {self.symbol}! "
                        f"Retraining from {buffer_size} buffer samples..."
                    )
                    self.model.retrain_from_buffer()
                    self.drift_detector.reset()
                    self._last_drift_retrain_time = time.time()

            return drift_event
        return None

    async def save_state(self):
        self.persistence.save_state(
            model=self.model,
            duration_optimizer=self.duration_optimizer,
            feature_engine=self.feature_engine,
            payout_rate=self.current_payout,
            trade_counter=self.trade_counter,
            live_tick_count=self.live_tick_count,
        )

    async def _fetch_payout_rate(self):
        if not self.ws or not self.ws.is_connected:
            return
        try:
            duration = self.duration_optimizer.select_duration()
            proposal = await self.ws.get_proposal(
                symbol=self.symbol, contract_type="DIGITOVER",
                barrier=OVER_BARRIER, stake=MIN_STAKE,
                duration=duration, duration_unit=self._duration_unit,
            )
            if proposal and proposal.get("payout", 0) > 0:
                new_payout = proposal["payout"] / proposal["stake"]
                if 0.5 < new_payout < 1.5:
                    old = self.current_payout
                    self.current_payout = new_payout
                    if abs(old - new_payout) > 0.01:
                        logger.info(f"{self.symbol} payout: {old:.2%} -> {new_payout:.2%}")
        except Exception as e:
            logger.debug(f"{self.symbol} payout fetch failed: {e}")

    def get_signal_score(self) -> float:
        """Used by MarketSelector to rank markets."""
        signal = self.get_fresh_signal()
        if signal is None:
            return 0.0
        # v9: EV is the primary quality metric (not confidence * payout)
        # A signal with 15% confidence but 50% EV is better than 80% confidence with 2% EV
        return signal.expected_value * signal.setup_score * max(0.5, signal.z_score)

    def summary(self) -> dict:
        model_summary = self.model.summary()
        dur_summary = self.duration_optimizer.summary()
        drift_summary = self.drift_detector.summary()
        setup = self._current_setup
        session = self.setup_detector.get_session(self.symbol)
        
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "live_ticks": self.live_tick_count,
            "trade_count": self.trade_counter,
            "payout_rate": self.current_payout,
            "warmup_done": self._warmup_done,
            "has_signal": self._latest_signal is not None,
            "model_accuracy": model_summary.get("accuracy", 0),
            "model_updates": model_summary.get("total_updates", 0),
            "drift_active": drift_summary.get("drift_active", False),
            "best_duration": dur_summary.get("best_duration", 5),
            "duration_unit": self._duration_unit,
            "setup_score": setup.setup_score if setup else 0,
            "setup_active": setup.active if setup else False,
            "setup_direction": setup.direction.replace("DIGIT","") if setup and setup.active else "NONE",
            "session_pnl": session.session_pnl if session else 0,
            "session_trades": session.session_trades if session else 0,
            "profit_target_reached": session.profit_target_reached if session else False,
        }
