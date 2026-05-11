"""
Online Learner — Over/Under Prediction Model (v2 — Ensemble)
==============================================================
Uses River library for online/incremental learning.
Supports multiple model types with hot-swapping + ENSEMBLE voting.

Models learn ONE sample at a time — no full batch retraining needed.
This is ideal for streaming tick data where data arrives continuously.

Model Architecture Guide:
=========================
- "logistic":   Online Logistic Regression — fast, interpretable, good baseline
- "hoeffding":  Hoeffding Adaptive Tree — non-linear, adapts to drift automatically
- "srp":        Streaming Random Patches — BEST single model, robust to drift
- "ensemble":   MULTI-MODEL VOTING — combines logistic + tree + SRP for max accuracy

RECOMMENDED: "ensemble" — uses 3 diverse models that vote together.
- Each model sees the same data but learns differently
- Majority voting smooths out individual model errors
- If one model drifts, the others compensate
- Automatically re-weights models based on recent accuracy
"""

import collections
import time
from dataclasses import dataclass, field
from typing import Optional

from config import (LEARNING_RATE, L2_REGULARIZATION, REPLAY_BUFFER_SIZE,
                    DRIFT_DETECTION_SENSITIVITY, MIN_CONFIDENCE)
from utils.logger import setup_logger

logger = setup_logger("models.online_learner")


@dataclass
class Prediction:
    """Model prediction result."""
    prob_over: float        # P(over hits)
    prob_under: float       # P(under hits)
    predicted_class: int    # 1=over, 0=under
    confidence: float       # max(prob_over, prob_under)
    is_tradeable: bool      # meets minimum confidence threshold
    model_version: int      # which model version made this prediction
    model_agreement: float  # fraction of sub-models that agree (ensemble only)


@dataclass
class ModelStats:
    """Model performance statistics."""
    total_updates: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_loss: float = 0.0
    drift_events: int = 0
    last_drift_time: float = 0.0
    model_version: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class OverUnderModel:
    """
    Online learning model for Over/Under binary classification.
    
    Architecture options (set via model_type param or MODEL_TYPE in config):
    - "logistic":   Online Logistic Regression (fast, interpretable)
    - "hoeffding":  Hoeffding Adaptive Tree (non-linear, auto-drift-adapt)
    - "srp":        Streaming Random Patches (best single model)
    - "ensemble":   Multi-model voting (RECOMMENDED — max accuracy + robustness)
    
    The ensemble mode runs 3 diverse models in parallel and uses
    weighted majority voting to produce a single prediction.
    Each sub-model tracks its own accuracy, and weights are adjusted
    so better-performing models get more influence.
    """
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self._is_ensemble = (model_type == "ensemble")
        
        # Scaler — shared across all models (learn before transform!)
        self.scaler = self._create_scaler()
        
        # Create model(s)
        if self._is_ensemble:
            self._sub_models = self._create_ensemble()
            self.model = None  # Not used in ensemble mode
            self._sub_stats = {name: {"correct": 0, "total": 0, "weight": 1.0}
                              for name in self._sub_models}
            logger.info(f"Ensemble model: {list(self._sub_models.keys())}")
        else:
            self.model = self._create_single_model(model_type)
            self._sub_models = {}
            self._sub_stats = {}
        
        # Metrics tracking
        self._correct = 0
        self._total = 0
        self._loss_sum = 0.0
        
        # Concept drift detection
        self._drift_detector = None
        
        # Replay buffer for periodic retraining
        self.replay_buffer = collections.deque(maxlen=REPLAY_BUFFER_SIZE)
        
        # Model stats
        self.stats = ModelStats()
        
        # Trade confidence tracking (for calibration)
        self._confidence_bins = collections.defaultdict(lambda: {"correct": 0, "total": 0})
        
        # Warmup tracking — before scaler has stats, use learn_one first
        self._scaler_warmed = False
        
        logger.info(f"OverUnderModel initialized: {model_type}")
    
    # ─── Model Creation ───
    
    def _create_ensemble(self) -> dict:
        """
        Create the 3-model ensemble.
        Diversity is KEY — each model type learns differently:
          1. Logistic:  Linear decision boundary, fast, good for stable regimes
          2. HAT:       Non-linear tree, adapts to drift via ADWIN at each node
          3. SRP:       Random patches + bagging, most robust to noise
        """
        try:
            from river import linear_model, ensemble, tree, optim, preprocessing
        except ImportError:
            logger.warning("River not installed. Using fallback mock model.")
            return {"mock": _MockModel()}
        
        models = {}
        
        # Model 1: Logistic Regression with Adam optimizer
        # Adam adapts learning rate per-feature — converges faster than SGD
        models["logistic"] = linear_model.LogisticRegression(
            optimizer=optim.Adam(lr=LEARNING_RATE),
        )
        
        # Model 2: Hoeffding Adaptive Tree (HAT)
        # Each node monitors drift with ADWIN — replaces subtrees when drift detected
        # This is the BEST single tree for non-stationary data
        models["hat"] = tree.HoeffdingAdaptiveTreeClassifier(
            max_depth=12,
            grace_period=50,         # Wait for 50 samples before splitting
            delta=1e-7,              # Split confidence (lower = more conservative splits)
            seed=42,
        )
        
        # Model 3: Streaming Random Patches (SRP)
        # Combines random subspaces + bagging — most robust single model
        # Each base learner sees a RANDOM SUBSET of features → diversity
        models["srp"] = ensemble.SRPClassifier(
            model=tree.HoeffdingTreeClassifier(max_depth=10, grace_period=50, delta=1e-7),
            n_models=6,              # 6 sub-learners
            seed=42,
        )
        
        return models
    
    def _create_single_model(self, model_type: str):
        """Create a single River model."""
        try:
            from river import linear_model, ensemble, tree, optim, preprocessing
        except ImportError:
            logger.warning("River not installed. Using fallback mock model.")
            return _MockModel()
        
        if model_type == "logistic":
            return linear_model.LogisticRegression(
                optimizer=optim.Adam(lr=LEARNING_RATE),
            )
        elif model_type == "hoeffding":
            return tree.HoeffdingAdaptiveTreeClassifier(
                max_depth=12,
                grace_period=50,
                delta=1e-7,
                seed=42,
            )
        elif model_type == "srp":
            return ensemble.SRPClassifier(
                model=tree.HoeffdingTreeClassifier(max_depth=10, grace_period=50, delta=1e-7),
                n_models=6,
                seed=42,
            )
        elif model_type == "forest":
            # Legacy alias — maps to SRP (better than old RandomForestClassifier)
            return ensemble.SRPClassifier(
                model=tree.HoeffdingTreeClassifier(max_depth=10, delta=1e-7),
                n_models=8,
                seed=42,
            )
        elif model_type == "boosting":
            return ensemble.AdaBoostClassifier(
                model=tree.HoeffdingTreeClassifier(max_depth=8, delta=1e-7),
                n_models=10,
                seed=42,
            )
        else:
            logger.warning(f"Unknown model type: {model_type}, using logistic")
            return linear_model.LogisticRegression(
                optimizer=optim.Adam(lr=LEARNING_RATE),
            )
    
    def _create_scaler(self):
        """Create online feature scaler."""
        try:
            from river import preprocessing
            return preprocessing.StandardScaler()
        except ImportError:
            return _MockScaler()
    
    # ─── Prediction ───
    
    def predict(self, features: dict) -> Prediction:
        """
        Get prediction for current features.
        
        In ensemble mode: runs all sub-models, combines via weighted voting.
        In single mode: runs just the one model.
        
        Args:
            features: dict of feature_name -> value
        
        Returns:
            Prediction with probabilities, confidence, and agreement score
        """
        try:
            # ALWAYS learn the scaler BEFORE transforming
            # This fixes the predict-before-learn bug
            if not self._scaler_warmed:
                # Scaler has no stats yet — return neutral prediction
                return Prediction(
                    prob_over=0.5, prob_under=0.5, predicted_class=0,
                    confidence=0.5, is_tradeable=False,
                    model_version=self.stats.model_version,
                    model_agreement=0.0,
                )
            
            scaled = self.scaler.transform_one(features)
            
            if self._is_ensemble:
                return self._predict_ensemble(scaled)
            else:
                return self._predict_single(scaled)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return Prediction(
                prob_over=0.5, prob_under=0.5, predicted_class=0,
                confidence=0.5, is_tradeable=False,
                model_version=self.stats.model_version,
                model_agreement=0.0,
            )
    
    @staticmethod
    def _normalize_proba(proba: dict) -> tuple:
        """
        Normalize probability dict to (prob_over, prob_under).
        
        River models return different key formats:
        - LogisticRegression: {True: 0.6, False: 0.4}  (boolean keys)
        - HAT/SRP:            {1: 0.6, 0: 0.4}          (integer keys)
        - Untrained models:   {} or {0: 0.5, 1: 0.5}    (empty or uniform)
        
        Returns (prob_class_1, prob_class_0) both as floats.
        """
        if not proba:
            return 0.5, 0.5
        
        # Try integer keys first (HAT, SRP, most tree-based)
        p_over = proba.get(1)
        p_under = proba.get(0)
        
        # If not found, try boolean keys (LogisticRegression)
        if p_over is None:
            p_over = proba.get(True)
        if p_under is None:
            p_under = proba.get(False)
        
        # Fallback: if only one key, the other is 1 - it
        if p_over is not None and p_under is None:
            p_under = 1.0 - p_over
        elif p_under is not None and p_over is None:
            p_over = 1.0 - p_under
        elif p_over is None and p_under is None:
            # Unknown key format — use first value as prob_over
            values = list(proba.values())
            if values:
                p_over = max(values)
                p_under = 1.0 - p_over
            else:
                p_over = 0.5
                p_under = 0.5
        
        return float(p_over), float(p_under)
    
    def _predict_ensemble(self, scaled_features: dict) -> Prediction:
        """
        Run all sub-models and combine via weighted voting.
        
        Weighted voting:
        - Each model's vote is weighted by its recent accuracy
        - If model A has 60% accuracy and model B has 50%, A's vote counts more
        - Probability = weighted average of individual probabilities
        """
        prob_over_sum = 0.0
        prob_under_sum = 0.0
        total_weight = 0.0
        votes_over = 0
        votes_under = 0
        
        for name, model in self._sub_models.items():
            try:
                proba = model.predict_proba_one(scaled_features)
                p_over, p_under = self._normalize_proba(proba)
                
                # Weight by recent accuracy
                weight = self._sub_stats[name]["weight"]
                
                prob_over_sum += p_over * weight
                prob_under_sum += p_under * weight
                total_weight += weight
                
                # Count votes
                if p_over >= p_under:
                    votes_over += 1
                else:
                    votes_under += 1
                    
            except Exception as e:
                logger.debug(f"Sub-model {name} predict error: {e}")
                # Default to 0.5 on error
                prob_over_sum += 0.5
                prob_under_sum += 0.5
                total_weight += 1.0
        
        if total_weight == 0:
            total_weight = 1.0
        
        prob_over = prob_over_sum / total_weight
        prob_under = prob_under_sum / total_weight
        
        predicted_class = 1 if prob_over >= prob_under else 0
        confidence = max(prob_over, prob_under)
        agreement = max(votes_over, votes_under) / len(self._sub_models) if self._sub_models else 0.5
        
        return Prediction(
            prob_over=round(prob_over, 4),
            prob_under=round(prob_under, 4),
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            is_tradeable=confidence >= MIN_CONFIDENCE,
            model_version=self.stats.model_version,
            model_agreement=round(agreement, 4),
        )
    
    def _predict_single(self, scaled_features: dict) -> Prediction:
        """Single model prediction."""
        proba = self.model.predict_proba_one(scaled_features)
        prob_over, prob_under = self._normalize_proba(proba)
        
        predicted_class = 1 if prob_over >= prob_under else 0
        confidence = max(prob_over, prob_under)
        
        return Prediction(
            prob_over=round(prob_over, 4),
            prob_under=round(prob_under, 4),
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            is_tradeable=confidence >= MIN_CONFIDENCE,
            model_version=self.stats.model_version,
            model_agreement=1.0,
        )
    
    # ─── Learning ───
    
    def learn_one(self, features: dict, outcome: int):
        """
        Update model with one observation.
        
        CRITICAL FIX: Scaler now learns BEFORE transform, so the scaler
        always has up-to-date statistics when features are transformed.
        
        Args:
            features: dict of feature_name -> value (from FeatureEngine)
            outcome: 1 if Over condition was met, 0 otherwise
        """
        try:
            # FIX: Learn scaler FIRST so it has stats for transform
            self.scaler.learn_one(features)
            self._scaler_warmed = True
            
            # NOW transform — scaler has valid statistics
            scaled = self.scaler.transform_one(features)
            
            if self._is_ensemble:
                self._learn_ensemble(scaled, features, outcome)
            else:
                self._learn_single(scaled, features, outcome)
            
            # Add to replay buffer
            self.replay_buffer.append((features, outcome))
            
            # Update stats
            self.stats.total_updates += 1
            self.stats.accuracy = self._correct / self._total if self._total > 0 else 0.0
            self.stats.last_updated = time.time()
            
            # Check for drift
            self._check_drift(outcome)
            
        except Exception as e:
            logger.error(f"Learning error: {e}")
    
    def _learn_ensemble(self, scaled: dict, raw_features: dict, outcome: int):
        """Learn on all sub-models and update their weights."""
        for name, model in self._sub_models.items():
            try:
                pred = model.predict_one(scaled)
                model.learn_one(scaled, outcome)
                
                # Track per-model accuracy
                self._sub_stats[name]["total"] += 1
                if pred == outcome:
                    self._sub_stats[name]["correct"] += 1
                
                # Update weight based on recent accuracy
                # Use exponential moving average of accuracy
                stats = self._sub_stats[name]
                if stats["total"] > 0:
                    acc = stats["correct"] / stats["total"]
                    # Weight = accuracy raised to power of 2 (amplify differences)
                    # A model with 55% acc gets weight 0.3025, 60% gets 0.36, 50% gets 0.25
                    stats["weight"] = max(0.1, acc ** 2)
                    
            except Exception as e:
                logger.debug(f"Sub-model {name} learn error: {e}")
        
        # Track overall accuracy (use ensemble vote)
        # Re-predict after learning to get the ensemble prediction
        try:
            ens_pred = self._predict_ensemble(scaled)
            self._total += 1
            if ens_pred.predicted_class == outcome:
                self._correct += 1
        except Exception:
            pass
    
    def _learn_single(self, scaled: dict, raw_features: dict, outcome: int):
        """Learn on single model."""
        pred = self.model.predict_one(scaled)
        self.model.learn_one(scaled, outcome)
        
        self._total += 1
        if pred == outcome:
            self._correct += 1
    
    def learn_with_prediction(self, features: dict, outcome: int, 
                               prediction: Prediction):
        """
        Update model AND track calibration of confidence bins.
        Use this when you have the prediction that was made before the outcome.
        """
        # Standard learning
        self.learn_one(features, outcome)
        
        # Track calibration
        conf_bin = round(prediction.confidence, 1)
        self._confidence_bins[conf_bin]["total"] += 1
        if prediction.predicted_class == outcome:
            self._confidence_bins[conf_bin]["correct"] += 1
    
    def warmup(self, feature_list: list[dict], label_list: list[int]):
        """
        Batch warmup from historical data.
        Call this before going live to give the model a starting point.
        """
        logger.info(f"Warmup: {len(feature_list)} samples...")
        
        correct = 0
        for features, label in zip(feature_list, label_list):
            try:
                # FIX: learn scaler first
                self.scaler.learn_one(features)
                self._scaler_warmed = True
                
                scaled = self.scaler.transform_one(features)
                
                if self._is_ensemble:
                    for name, model in self._sub_models.items():
                        try:
                            pred = model.predict_one(scaled)
                            model.learn_one(scaled, label)
                            self._sub_stats[name]["total"] += 1
                            if pred == label:
                                self._sub_stats[name]["correct"] += 1
                        except Exception:
                            pass
                    # Track overall
                    ens_pred = self._predict_ensemble(scaled)
                    if ens_pred.predicted_class == label:
                        correct += 1
                else:
                    pred = self.model.predict_one(scaled)
                    if pred == label:
                        correct += 1
                    self.model.learn_one(scaled, label)
                
                self.stats.total_updates += 1
            except Exception as e:
                logger.debug(f"Warmup skip: {e}")
        
        warmup_acc = correct / len(feature_list) if feature_list else 0
        logger.info(f"Warmup complete: {warmup_acc:.1%} accuracy on training data")
        self.stats.last_updated = time.time()
    
    def retrain_from_buffer(self):
        """
        Retrain on replay buffer (useful after drift event).
        Creates a new model version.
        """
        if len(self.replay_buffer) < 100:
            logger.warning("Replay buffer too small for retraining, skipping")
            return
        
        logger.info(f"Retraining from {len(self.replay_buffer)} buffer samples...")
        
        # Create fresh model(s) + scaler
        self.scaler = self._create_scaler()
        self._scaler_warmed = False
        
        if self._is_ensemble:
            self._sub_models = self._create_ensemble()
            self._sub_stats = {name: {"correct": 0, "total": 0, "weight": 1.0}
                              for name in self._sub_models}
        else:
            self.model = self._create_single_model(self.model_type)
        
        self._correct = 0
        self._total = 0
        
        for features, label in self.replay_buffer:
            try:
                self.scaler.learn_one(features)
                self._scaler_warmed = True
                scaled = self.scaler.transform_one(features)
                
                if self._is_ensemble:
                    for name, model in self._sub_models.items():
                        try:
                            model.learn_one(scaled, label)
                        except Exception:
                            pass
                else:
                    self.model.learn_one(scaled, label)
                
                self._total += 1
            except Exception as e:
                logger.debug(f"Retrain skip: {e}")
        
        self.stats.model_version += 1
        self.stats.accuracy = self._correct / self._total if self._total > 0 else 0
        logger.info(f"Retrain complete: v{self.stats.model_version}, "
                     f"accuracy={self.stats.accuracy:.1%}")
    
    # ─── Drift Detection ───
    
    def _check_drift(self, outcome: int):
        """Check for concept drift using ADWIN."""
        try:
            from river import drift
            
            if self._drift_detector is None:
                self._drift_detector = drift.ADWIN(delta=DRIFT_DETECTION_SENSITIVITY)
            
            self._drift_detector.update(outcome)
            
            if self._drift_detector.detected_change_flag:
                self.stats.drift_events += 1
                self.stats.last_drift_time = time.time()
                logger.warning(
                    f"CONCEPT DRIFT DETECTED! (event #{self.stats.drift_events})"
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Drift check error: {e}")
    
    # ─── Model Info ───
    
    def get_feature_importance(self) -> Optional[dict]:
        """
        Get feature importance.
        For ensemble: returns logistic model weights (most interpretable).
        For logistic: returns weights directly.
        """
        target_model = None
        
        if self._is_ensemble and "logistic" in self._sub_models:
            target_model = self._sub_models["logistic"]
        elif self.model_type == "logistic" and self.model is not None:
            target_model = self.model
        
        if target_model is None:
            return None
        
        try:
            if hasattr(target_model, 'weights'):
                return dict(target_model.weights)
        except Exception as e:
            logger.debug(f"Feature importance unavailable: {e}")
        return None
    
    def get_sub_model_accuracy(self) -> dict:
        """Get accuracy of each sub-model (ensemble mode only)."""
        result = {}
        for name, stats in self._sub_stats.items():
            if stats["total"] > 0:
                result[name] = {
                    "accuracy": round(stats["correct"] / stats["total"] * 100, 1),
                    "weight": round(stats["weight"], 4),
                    "samples": stats["total"],
                }
            else:
                result[name] = {"accuracy": 0, "weight": 1.0, "samples": 0}
        return result
    
    def get_calibration(self) -> dict:
        """
        Get confidence calibration data.
        Shows if model confidence matches actual win rate.
        """
        calibration = {}
        for bin_val, counts in sorted(self._confidence_bins.items()):
            total = counts["total"]
            correct = counts["correct"]
            if total > 0:
                calibration[bin_val] = {
                    "confidence": bin_val,
                    "actual_win_rate": correct / total,
                    "sample_count": total,
                }
        return calibration
    
    def summary(self) -> dict:
        """Get model summary."""
        result = {
            "model_type": self.model_type,
            "version": self.stats.model_version,
            "total_updates": self.stats.total_updates,
            "accuracy": round(self.stats.accuracy * 100, 1),
            "drift_events": self.stats.drift_events,
            "replay_buffer_size": len(self.replay_buffer),
            "is_trained": self.stats.total_updates > 0,
            "scaler_warmed": self._scaler_warmed,
        }
        
        if self._is_ensemble:
            result["sub_models"] = list(self._sub_models.keys())
            result["sub_model_accuracy"] = self.get_sub_model_accuracy()
        
        return result


class _MockModel:
    """Fallback mock model when River is not installed."""
    
    def predict_one(self, x):
        return 0
    
    def predict_proba_one(self, x):
        return {0: 0.5, 1: 0.5}
    
    def learn_one(self, x, y):
        pass


class _MockScaler:
    """Fallback mock scaler."""
    
    def transform_one(self, x):
        return x
    
    def learn_one(self, x):
        pass
