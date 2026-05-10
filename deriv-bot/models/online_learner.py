"""
Online Learner — Over/Under Prediction Model
==============================================
Uses River library for online/incremental learning.
Supports multiple model types with hot-swapping.

Models learn ONE sample at a time — no full batch retraining needed.
This is ideal for streaming tick data where data arrives continuously.
"""

import collections
import time
from dataclasses import dataclass, field
from typing import Optional

from config import (LEARNING_RATE, L2_REGULARIZATION, REPLAY_BUFFER_SIZE,
                    DRIFT_DETECTION_SENSITIVITY)
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
    
    Architecture options (set via MODEL_TYPE in config):
    - "logistic": Online Logistic Regression (fast, interpretable)
    - "forest":   Online Random Forest (non-linear, better accuracy)
    - "boosting": Online Gradient Boosting (best accuracy, slower)
    
    Default: Logistic Regression — best starting point.
    """
    
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.model = self._create_model()
        self.scaler = self._create_scaler()
        
        # Metrics tracking
        self._correct = 0
        self._total = 0
        self._loss_sum = 0.0
        
        # Concept drift detection
        self._drift_detector = None  # Set up after first prediction
        
        # Replay buffer for periodic retraining
        self.replay_buffer = collections.deque(maxlen=REPLAY_BUFFER_SIZE)
        
        # Model stats
        self.stats = ModelStats()
        
        # Trade confidence tracking (for calibration)
        self._confidence_bins = collections.defaultdict(lambda: {"correct": 0, "total": 0})
        
        logger.info(f"OverUnderModel initialized: {model_type}")
    
    def _create_model(self):
        """Create the River online learning model."""
        try:
            from river import linear_model, ensemble, tree
        except ImportError:
            logger.warning("River not installed. Using fallback mock model.")
            return _MockModel()
        
        if self.model_type == "logistic":
            return linear_model.LogisticRegression(
                l2=L2_REGULARIZATION,
                optimizer="sgd",
                lr=LEARNING_RATE,
            )
        elif self.model_type == "forest":
            return ensemble.RandomForestClassifier(
                n_models=10,
                max_depth=10,
                seed=42,
            )
        elif self.model_type == "boosting":
            return ensemble.AdaboostClassifier(
                model=tree.HoeffdingTreeClassifier(max_depth=8),
                n_models=10,
                seed=42,
            )
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using logistic")
            return linear_model.LogisticRegression(
                l2=L2_REGULARIZATION,
                optimizer="sgd",
                lr=LEARNING_RATE,
            )
    
    def _create_scaler(self):
        """Create online feature scaler."""
        try:
            from river import preprocessing
            return preprocessing.StandardScaler()
        except ImportError:
            return _MockScaler()
    
    def predict(self, features: dict) -> Prediction:
        """
        Get prediction for current features.
        
        Args:
            features: dict of feature_name -> value
        
        Returns:
            Prediction with probabilities and confidence
        """
        try:
            # Scale features
            scaled = self.scaler.transform_one(features)
            
            # Get probability of class 1 (Over hits)
            proba = self.model.predict_proba_one(scaled)
            
            prob_over = proba.get(1, 0.5)
            prob_under = proba.get(0, 0.5)
            
            predicted_class = 1 if prob_over >= prob_under else 0
            confidence = max(prob_over, prob_under)
            
            return Prediction(
                prob_over=round(prob_over, 4),
                prob_under=round(prob_under, 4),
                predicted_class=predicted_class,
                confidence=round(confidence, 4),
                is_tradeable=confidence >= 0.56,  # Minimum tradeable threshold
                model_version=self.stats.model_version,
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return Prediction(
                prob_over=0.5, prob_under=0.5, predicted_class=0,
                confidence=0.5, is_tradeable=False,
                model_version=self.stats.model_version,
            )
    
    def learn_one(self, features: dict, outcome: int):
        """
        Update model with one observation.
        
        Args:
            features: dict of feature_name -> value (from FeatureEngine)
            outcome: 1 if Over condition was met, 0 otherwise
        """
        try:
            # Scale features
            scaled = self.scaler.transform_one(features)
            
            # Get prediction before learning (for accuracy tracking)
            pred = self.model.predict_one(scaled)
            
            # Learn
            self.model.learn_one(scaled, outcome)
            self.scaler.learn_one(features)  # Update scaler statistics
            
            # Track accuracy
            self._total += 1
            if pred == outcome:
                self._correct += 1
            
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
                scaled = self.scaler.transform_one(features)
                pred = self.model.predict_one(scaled)
                if pred == label:
                    correct += 1
                self.model.learn_one(scaled, label)
                self.scaler.learn_one(features)
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
        
        # Create fresh model + scaler
        old_model = self.model
        old_scaler = self.scaler
        self.model = self._create_model()
        self.scaler = self._create_scaler()
        self._correct = 0
        self._total = 0
        
        for features, label in self.replay_buffer:
            try:
                scaled = self.scaler.transform_one(features)
                pred = self.model.predict_one(scaled)
                if pred == label:
                    self._correct += 1
                self._total += 1
                self.model.learn_one(scaled, label)
                self.scaler.learn_one(features)
            except Exception as e:
                logger.debug(f"Retrain skip: {e}")
        
        self.stats.model_version += 1
        self.stats.accuracy = self._correct / self._total if self._total > 0 else 0
        logger.info(f"Retrain complete: v{self.stats.model_version}, "
                     f"accuracy={self.stats.accuracy:.1%}")
    
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
    
    def get_feature_importance(self) -> Optional[dict]:
        """
        Get feature importance (only available for logistic regression).
        Returns dict of {feature_name: weight}.
        """
        if self.model_type != "logistic":
            return None
        
        try:
            if hasattr(self.model, 'weights'):
                weights = self.model.weights
                return dict(weights)
        except Exception as e:
            logger.debug(f"Feature importance unavailable: {e}")
        return None
    
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
        return {
            "model_type": self.model_type,
            "version": self.stats.model_version,
            "total_updates": self.stats.total_updates,
            "accuracy": round(self.stats.accuracy * 100, 1),
            "drift_events": self.stats.drift_events,
            "replay_buffer_size": len(self.replay_buffer),
            "is_trained": self.stats.total_updates > 0,
        }


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
