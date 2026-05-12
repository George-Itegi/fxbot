"""
Model Persistence v2 — Per-Symbol Full State
==============================================
Saves ALL bot state per market symbol, not just the ML model.
Fixes bugs from v1 (wrong attribute names, missing _scaler_warmed, etc.).
"""

import collections
import pickle
import time
from pathlib import Path
from typing import Optional

from config import MODEL_DIR
from utils.logger import setup_logger

logger = setup_logger("models.persistence")


class ModelPersistence:
    """
    Save/load FULL per-market state snapshots.
    
    v2 saves everything needed to resume a market exactly where it left off:
    - ML model (sub_models, scaler, stats, replay_buffer, calibration)
    - DurationOptimizer (all per-duration stats, epsilon, drift mode)
    - FeatureEngine (Markov matrix, digit pairs, last digit)
    - Market metadata (payout rate, trade count, tick count)
    """

    def __init__(self, model_dir: str = str(MODEL_DIR), symbol: str = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol

    def save_state(self, model, duration_optimizer=None, feature_engine=None,
                   payout_rate: float = 0.85, trade_counter: int = 0,
                   live_tick_count: int = 0, extra: dict = None) -> Optional[str]:
        symbol_prefix = f"{self.symbol}_" if self.symbol else ""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        version = model.stats.model_version if hasattr(model, 'stats') else 0
        snapshot_name = f"{symbol_prefix}state_v{version}_{timestamp}"
        filepath = self.model_dir / f"{snapshot_name}.pkl"

        model_data = {
            "model_type": model.model_type,
            "is_ensemble": model._is_ensemble,
            "scaler": model.scaler,
            "scaler_warmed": model._scaler_warmed,
            "stats": model.stats,
            "replay_buffer": list(model.replay_buffer),
            "confidence_bins": dict(model._confidence_bins),
            "correct": model._correct,
            "total": model._total,
            "saved_at": time.time(),
        }

        if model._is_ensemble:
            model_data["sub_models"] = model._sub_models
            model_data["sub_stats"] = dict(model._sub_stats)
        else:
            model_data["single_model"] = model.model

        dur_data = None
        if duration_optimizer is not None:
            dur_data = {
                "epsilon": duration_optimizer.epsilon,
                "current_best": duration_optimizer._current_best,
                "total_selections": duration_optimizer._total_selections,
                "exploration_count": duration_optimizer._exploration_count,
                "drift_mode": duration_optimizer._drift_mode,
                "last_drift_time": duration_optimizer._last_drift_time,
                "stats": {},
            }
            for d, s in duration_optimizer.stats.items():
                dur_data["stats"][d] = {
                    "duration": s.duration,
                    "wins": s.wins,
                    "losses": s.losses,
                    "total_payout": s.total_payout,
                    "total_stake": s.total_stake,
                    "recent_results": list(s.recent_results),
                }

        fe_data = None
        if feature_engine is not None:
            fe_data = {
                "transition_counts": {
                    k: dict(v) for k, v in feature_engine._transition_counts.items()
                },
                "transition_total": dict(feature_engine._transition_total),
                "last_digit": feature_engine._last_digit,
                "digit_pairs": dict(feature_engine._digit_pairs),
                "pair_total": feature_engine._pair_total,
            }

        state_data = {
            "version": 2,
            "symbol": self.symbol,
            "model": model_data,
            "duration_optimizer": dur_data,
            "feature_engine": fe_data,
            "payout_rate": payout_rate,
            "trade_counter": trade_counter,
            "live_tick_count": live_tick_count,
            "extra": extra or {},
        }

        try:
            with open(filepath, "wb") as f:
                pickle.dump(state_data, f)
            logger.info(f"State saved: {filepath.name} "
                        f"(model v{version}, {model.stats.total_updates} updates)")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return None

    def load_state(self, model, duration_optimizer=None, feature_engine=None,
                   filepath: str = None) -> bool:
        if filepath is None:
            snapshot = self._find_latest_snapshot()
            if snapshot is None:
                logger.info("No snapshot found to load")
                return False
            filepath = snapshot

        path = Path(filepath)
        if not path.exists():
            logger.error(f"Snapshot not found: {filepath}")
            return False

        try:
            with open(path, "rb") as f:
                state_data = pickle.load(f)

            version = state_data.get("version", 1)
            if version == 1:
                return self._load_v1(state_data, model)
            else:
                return self._load_v2(state_data, model, duration_optimizer,
                                     feature_engine)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def _load_v2(self, state_data, model, duration_optimizer, feature_engine):
        model_data = state_data.get("model", {})

        if model_data.get("is_ensemble"):
            if "sub_models" in model_data:
                model._sub_models = model_data["sub_models"]
            if "sub_stats" in model_data:
                model._sub_stats = model_data["sub_stats"]
                for name in model._sub_models:
                    if name not in model._sub_stats:
                        model._sub_stats[name] = {"correct": 0, "total": 0, "weight": 1.0}
        else:
            if "single_model" in model_data:
                model.model = model_data["single_model"]

        model.scaler = model_data.get("scaler", model.scaler)
        model._scaler_warmed = model_data.get("scaler_warmed", False)
        model.stats = model_data.get("stats", model.stats)
        model._correct = model_data.get("correct", 0)
        model._total = model_data.get("total", 0)

        rb_data = model_data.get("replay_buffer", [])
        model.replay_buffer = collections.deque(rb_data, maxlen=model.replay_buffer.maxlen)

        cb_data = model_data.get("confidence_bins", {})
        model._confidence_bins = collections.defaultdict(
            lambda: {"correct": 0, "total": 0}, cb_data
        )

        logger.info(
            f"Model loaded: v{model.stats.model_version}, "
            f"{model.stats.total_updates} updates, "
            f"scaler_warmed={model._scaler_warmed}"
        )

        dur_data = state_data.get("duration_optimizer")
        if dur_data and duration_optimizer is not None:
            duration_optimizer.epsilon = dur_data.get("epsilon", duration_optimizer.epsilon)
            duration_optimizer._current_best = dur_data.get("current_best", duration_optimizer.default_duration)
            duration_optimizer._total_selections = dur_data.get("total_selections", 0)
            duration_optimizer._exploration_count = dur_data.get("exploration_count", 0)
            duration_optimizer._drift_mode = dur_data.get("drift_mode", False)
            duration_optimizer._last_drift_time = dur_data.get("last_drift_time", 0.0)

            for d_str, s_data in dur_data.get("stats", {}).items():
                d = int(d_str) if isinstance(d_str, str) else d_str
                if d in duration_optimizer.stats:
                    stats = duration_optimizer.stats[d]
                    stats.wins = s_data.get("wins", 0)
                    stats.losses = s_data.get("losses", 0)
                    stats.total_payout = s_data.get("total_payout", 0.0)
                    stats.total_stake = s_data.get("total_stake", 0.0)
                    stats.recent_results = collections.deque(
                        s_data.get("recent_results", []), maxlen=50
                    )

            logger.info(
                f"DurationOptimizer loaded: best={duration_optimizer._current_best}t, "
                f"epsilon={duration_optimizer.epsilon:.3f}, "
                f"selections={duration_optimizer._total_selections}"
            )

        fe_data = state_data.get("feature_engine")
        if fe_data and feature_engine is not None:
            tc = fe_data.get("transition_counts", {})
            for k, v in tc.items():
                feature_engine._transition_counts[int(k)] = {int(k2): v2 for k2, v2 in v.items()}

            tt = fe_data.get("transition_total", {})
            feature_engine._transition_total = {int(k): v for k, v in tt.items()}

            feature_engine._last_digit = fe_data.get("last_digit")
            feature_engine._digit_pairs = fe_data.get("digit_pairs", {})
            feature_engine._pair_total = fe_data.get("pair_total", 0)

            total_transitions = sum(feature_engine._transition_total.values())
            logger.info(
                f"FeatureEngine loaded: {total_transitions} Markov transitions, "
                f"{len(feature_engine._digit_pairs)} digit pairs"
            )

        return True

    def _load_v1(self, snapshot_data, model):
        try:
            if "sub_models" in snapshot_data:
                model._sub_models = snapshot_data["sub_models"]
                model._sub_stats = snapshot_data.get("sub_stats", {})
            elif "models" in snapshot_data:
                model._sub_models = snapshot_data["models"]
                old_accuracies = snapshot_data.get("model_accuracies", {})
                old_weights = snapshot_data.get("model_weights", {})
                model._sub_stats = {}
                for name in model._sub_models:
                    model._sub_stats[name] = {
                        "correct": 0, "total": 0,
                        "weight": old_weights.get(name, 1.0),
                    }
            else:
                old_model = snapshot_data.get("model")
                if old_model is not None:
                    model._sub_models = {"logistic": old_model}
                    model._sub_stats = {"logistic": {"correct": 0, "total": 0, "weight": 1.0}}

            model.scaler = snapshot_data.get("scaler", model.scaler)
            model._scaler_warmed = True
            model.stats = snapshot_data.get("stats", model.stats)

            rb_data = snapshot_data.get("replay_buffer", [])
            model.replay_buffer = collections.deque(
                rb_data, maxlen=model.replay_buffer.maxlen
            )

            cb_data = snapshot_data.get("confidence_bins", {})
            model._confidence_bins = collections.defaultdict(
                lambda: {"correct": 0, "total": 0}, cb_data
            )

            logger.info(f"V1 snapshot loaded: v{model.stats.model_version}, "
                         f"{model.stats.total_updates} updates")
            return True
        except Exception as e:
            logger.error(f"Failed to load v1 snapshot: {e}")
            return False

    def list_snapshots(self, symbol: str = None) -> list:
        sym = symbol or self.symbol
        snapshots = []
        for f in sorted(self.model_dir.glob("*.pkl"), reverse=True):
            if sym and not f.name.startswith(f"{sym}_"):
                continue
            snapshots.append({
                "name": f.name,
                "path": str(f),
                "size_kb": f.stat().st_size / 1024,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(f.stat().st_mtime)),
            })
        return snapshots

    def _find_latest_snapshot(self) -> Optional[str]:
        snapshots = self.list_snapshots()
        if snapshots:
            return snapshots[0]["path"]
        return None

    def cleanup_old_snapshots(self, keep_last: int = 5, symbol: str = None):
        sym = symbol or self.symbol
        snapshots = self.list_snapshots(symbol=sym)
        if len(snapshots) > keep_last:
            for old_snap in snapshots[keep_last:]:
                Path(old_snap["path"]).unlink()
                logger.info(f"Deleted old snapshot: {old_snap['name']}")

    # ─── Legacy compatibility ───
    def save_snapshot(self, model, snapshot_name: Optional[str] = None):
        return self.save_state(model)

    def load_snapshot(self, model, filepath: str) -> bool:
        return self.load_state(model, filepath=filepath)
