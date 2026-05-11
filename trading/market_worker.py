"""
Market Worker — Per-Market State Encapsulation
================================================
Each MarketWorker encapsulates all components for ONE trading symbol.
"""

import time
from typing import Optional

from config import (OVER_BARRIER, UNDER_BARRIER, get_symbol_decimals, MIN_STAKE)
from data.tick_aggregator import TickAggregator
from data.feature_engine import FeatureEngine
from models.online_learner import OverUnderModel
from models.duration_optimizer import DurationOptimizer
from models.drift_detector import DriftDetector
from models.model_persistence import ModelPersistence
from trading.signal_generator import SignalGenerator, Signal
from trading.execution_engine import ExecutionEngine
from trading.risk_manager import RiskDecision
from utils.logger import setup_logger

logger = setup_logger("trading.market_worker")


class MarketWorker:
    """
    Encapsulates all per-market components for one trading symbol.
    """

    def __init__(self, symbol: str, deriv_ws, model_type: str = "ensemble"):
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
        self.signal_gen = SignalGenerator(
            duration_optimizer=self.duration_optimizer,
            dynamic_barriers=True,
            min_model_agreement=0.67,
        )
        self.executor = ExecutionEngine(
            self.ws, symbol=self.symbol, duration_unit=self._duration_unit,
        )

        self.current_payout = 0.85
        self.trade_counter = 0
        self.live_tick_count = 0
        self._warmup_done = False
        self._bankroll = 0.0  # Updated by main.py from RiskManager

        self._latest_signal: Optional[Signal] = None
        self._signal_time: float = 0.0
        self._signal_freshness_sec = 5.0

        logger.info(f"MarketWorker created: {symbol}, model={model_type}")

    async def warmup(self, history: list):
        """Process historical ticks for warmup training."""
        if not history:
            logger.warning(f"{self.symbol}: No historical ticks for warmup")
            return

        dp = self.aggregator.decimal_places or get_symbol_decimals(self.symbol)

        # Load model snapshot first
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

            prediction = self.model.predict(features)

            if self.live_tick_count % 100 == 0:
                model_summary = self.model.summary()
                dur_rec = self.duration_optimizer.get_recommendation()
                logger.info(
                    f"[{self.symbol} {self.live_tick_count}t] "
                    f"prob_over={prediction.prob_over:.3f} "
                    f"conf={prediction.confidence:.3f} "
                    f"agree={prediction.model_agreement:.0%} "
                    f"acc={model_summary['accuracy']:.1f}% "
                    f"dur={dur_rec['recommended_duration']}{self._duration_unit} "
                    f"payout={self.current_payout:.2%}"
                )

            if not prediction.is_tradeable:
                self._latest_signal = None
                return None

            signal = self.signal_gen.generate(
                prediction=prediction,
                features=features,
                payout=self.current_payout,
                bankroll=self._bankroll,
                model_in_drift=self.drift_detector.drift_active,
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
            approved=True, reason="Meta-selector approved",
            adjusted_stake=stake, checks={"meta_selector": True},
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

        drift_event = self.drift_detector.update(outcome)
        if drift_event:
            self.duration_optimizer.on_drift_detected()
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
        signal = self.get_fresh_signal()
        if signal is None:
            return 0.0
        model_acc = self.model.stats.accuracy if self.model.stats.total_updates > 0 else 0.5
        model_acc_factor = max(0.1, model_acc)
        agreement_factor = signal.model_agreement
        score = signal.confidence * signal.expected_value * self.current_payout * model_acc_factor * agreement_factor
        return score

    def summary(self) -> dict:
        model_summary = self.model.summary()
        dur_summary = self.duration_optimizer.summary()
        drift_summary = self.drift_detector.summary()
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
        }
