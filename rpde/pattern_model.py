# =============================================================
# rpde/pattern_model.py  —  RPDE L1: Per-Pair XGBoost Pattern Model
#
# PURPOSE: Learn to recognize pattern-favorable market conditions
# for a specific currency pair. Instead of 10 strategy-specific
# models (v4.2), RPDE uses ONE model per pair that learns from
# ALL discovered patterns (golden moments) for that pair.
#
# ARCHITECTURE:
#   v4.2: 10 strategy models (one per strategy, trained on strategy-specific trades)
#   v5.0: 1 pattern model per pair (trained on ALL golden moments for that pair)
#
# Key differences from v4.2 ML Gate:
#   - Trained on golden moments (forward_return target), not trade outcomes
#   - Uses 93 ML Gate features (same feature space, reusable knowledge)
#   - Per-pair specialization (captures pair-specific microstructure)
#   - Time-based train/val split (no look-ahead bias)
#   - Incremental training via warm-start
#   - Optional Experience Replay Buffer integration
#
# TARGET:
#   forward_return = move_pips / atr_pips (R-multiple of the forward move)
#   Clipped to [-2.0, 5.0] to prevent outlier distortion
# =============================================================

import os
import json
import time
import numpy as np
import joblib
from datetime import datetime, timezone
from typing import Optional, List, Tuple

from core.logger import get_logger
from ai_engine.ml_gate import FEATURE_NAMES

log = get_logger(__name__)

# ── Regression-specific constants (aligned with v4.2 ML Gate) ──
R_CLIP_MIN = -2.0
R_CLIP_MAX = 5.0
TAKE_THRESHOLD = 0.3   # predicted_r >= 0.3 → TAKE
CAUTION_THRESHOLD = 0.0  # predicted_r < 0.0 → SKIP


class PatternModel:
    """Per-pair XGBoost pattern recognition model.

    Learns to predict forward_return (R-multiple of the expected move)
    from 93 ML Gate features. Trained on golden moments discovered by
    the RPDE Big Move Scanner.

    Usage:
        model = PatternModel("EURJPY")
        model.train(golden_moments)
        result = model.predict(features)
    """

    def __init__(self, pair: str):
        """
        Args:
            pair: Currency pair string (e.g. 'EURJPY')
        """
        self.pair = pair.upper()
        self.model = None
        self.meta = {}
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.model_path = os.path.join(
            self.model_dir, f'pattern_{self.pair.lower()}.pkl')
        self.meta_path = os.path.join(
            self.model_dir, f'pattern_{self.pair.lower()}_meta.json')
        self.is_loaded = False
        self.n_features = len(FEATURE_NAMES)  # 93

        # Try loading existing model on init
        self.load()

    # ════════════════════════════════════════════════════════════════
    # MODEL EXISTENCE CHECK
    # ════════════════════════════════════════════════════════════════

    def is_trained(self) -> bool:
        """Check if a trained model exists for this pair."""
        return self.model is not None and self.is_loaded

    # ════════════════════════════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════════════════════════════

    def train(self, golden_moments: list, incremental: bool = False,
              use_replay: bool = False) -> dict:
        """
        Train the pattern model for this pair.

        Args:
            golden_moments: List of golden moment dicts, each with
                            'features' (dict) and 'forward_return' (float).
                            Features are the 93 ML Gate features keyed by name.
            incremental: If True, warm-start from existing model.
            use_replay: If True, use Experience Replay Buffer to augment data.

        Returns:
            Training results dict with metrics.
        """
        from rpde.config import XGB_PARAMS, MIN_TRAINING_SAMPLES, NEGATIVE_SAMPLE_RATIO, NEGATIVE_MAX_SAMPLES_PER_PAIR
        from sklearn.model_selection import TimeSeriesSplit

        t0 = time.time()
        log.info(f"[RPDE_MODEL] Training pattern model for {self.pair}: "
                 f"{len(golden_moments)} golden moments, "
                 f"incremental={incremental}, replay={use_replay}")

        # ── Step 1: Load features from golden_moments ──
        X_list = []
        y_list = []
        skipped = 0

        for moment in golden_moments:
            features = moment.get('features', {})
            if not features:
                skipped += 1
                continue

            # Build feature vector aligned to FEATURE_NAMES
            row = []
            valid = True
            for feat_name in FEATURE_NAMES:
                val = features.get(feat_name)
                if val is None:
                    valid = False
                    break
                row.append(float(val))

            if not valid:
                skipped += 1
                continue

            # Target: forward_return
            forward_r = moment.get('forward_return')
            if forward_r is None:
                forward_r = moment.get('move_pips', 0)
                # Approximate R-multiple if we have ATR info
                atr_val = features.get('vs_atr', 0)
                if atr_val and atr_val > 0:
                    # Convert ATR from price units — need pip size
                    from core.pip_utils import get_pip_size
                    pip_value = get_pip_size(self.pair)
                    if pip_value > 0:
                        atr_pips = atr_val / pip_value
                        if atr_pips > 0:
                            forward_r = forward_r / atr_pips
                        else:
                            forward_r = 0.0
                    else:
                        forward_r = 0.0
                else:
                    forward_r = 0.0

            try:
                forward_r = float(forward_r)
            except (TypeError, ValueError):
                skipped += 1
                continue

            # Clip target to prevent outlier distortion
            forward_r = max(R_CLIP_MIN, min(R_CLIP_MAX, forward_r))

            X_list.append(row)
            y_list.append(forward_r)

        # ── Step 2: Add negative samples ──
        # Use TRUE negative samples from DB (bars that passed regime filter
        # but did NOT produce a big move). These have direction='NONE' and
        # near-zero forward_returns, teaching the model when NOT to trade.
        #
        # OLD (BROKEN): loaded all golden moments which are all winners,
        # giving the model zero negative signal.
        negative_count = 0
        try:
            from rpde.database import load_negative_samples as _load_neg
            true_negatives = _load_neg(pair=self.pair)
            
            if true_negatives:
                import random
                target_negatives = min(
                    len(X_list) * NEGATIVE_SAMPLE_RATIO,
                    len(true_negatives),
                    NEGATIVE_MAX_SAMPLES_PER_PAIR
                )
                
                if target_negatives > 0:
                    sampled_negatives = random.sample(
                        true_negatives, 
                        min(target_negatives, len(true_negatives))
                    )
                    
                    for moment in sampled_negatives:
                        features = moment.get('features') or moment.get('feature_snapshot') or {}
                        if not features:
                            continue
                        
                        row = []
                        valid = True
                        for feat_name in FEATURE_NAMES:
                            val = features.get(feat_name)
                            if val is None:
                                valid = False
                                break
                            row.append(float(val))
                        
                        if not valid:
                            continue
                        
                        # Use ACTUAL forward_return (near zero for non-golden bars)
                        forward_r = moment.get('forward_return', 0.0)
                        if forward_r is None:
                            forward_r = 0.0
                        try:
                            forward_r = float(forward_r)
                        except (TypeError, ValueError):
                            continue
                        
                        # Clip to same range
                        forward_r = max(R_CLIP_MIN, min(R_CLIP_MAX, forward_r))
                        
                        X_list.append(row)
                        y_list.append(forward_r)
                        negative_count += 1
                    
                    if negative_count > 0:
                        log.info(f"[RPDE_MODEL] Added {negative_count} TRUE negative samples "
                                 f"(direction=NONE from DB, "
                                 f"ratio 1:{negative_count/max(len(X_list)-negative_count,1):.1f}, "
                                 f"total now: {len(X_list)})")
                else:
                    log.info(f"[RPDE_MODEL] True negatives found ({len(true_negatives)}) "
                             f"but target_negatives=0, skipping")
            else:
                log.info(f"[RPDE_MODEL] No true negative samples found in DB for {pair} — "
                         f"run scanner first to collect direction='NONE' samples")
        except Exception as ex:
            log.debug(f"[RPDE_MODEL] True negative sampling skipped: {ex}")

        # ── Step 3: Augment with Replay Buffer if requested ──
        if use_replay:
            replay_X, replay_y = self._load_replay_buffer_data()
            if replay_X is not None and len(replay_X) > 0:
                X_list.extend(replay_X)
                y_list.extend(replay_y)
                log.info(f"[RPDE_MODEL] Added {len(replay_y)} samples from "
                         f"replay buffer (total now: {len(X_list)})")

        if len(X_list) < MIN_TRAINING_SAMPLES:
            log.warning(
                f"[RPDE_MODEL] Insufficient samples for {self.pair}: "
                f"{len(X_list)} valid (need >= {MIN_TRAINING_SAMPLES}), "
                f"{skipped} skipped")
            return {
                'pair': self.pair,
                'trained': False,
                'samples': len(X_list),
                'skipped': skipped,
                'reason': f'Need >= {MIN_TRAINING_SAMPLES} samples, got {len(X_list)}',
            }

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # ── Step 4: Time-based train/val split (80/20) ──
        # Use last 20% as validation (most recent data for realistic evaluation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_val) < 10:
            log.warning(
                f"[RPDE_MODEL] Validation set too small for {self.pair}: "
                f"{len(X_val)} samples. Using TimeSeriesSplit instead.")
            # Fallback: use TimeSeriesSplit with 3 folds
            tscv = TimeSeriesSplit(n_splits=3)
            train_idx, val_idx = list(tscv.split(X))[-1]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        log.info(f"[RPDE_MODEL] {self.pair} split: "
                 f"train={len(X_train)}, val={len(X_val)}")

        # ── Step 5: Train XGBoost ──
        import xgboost as xgb

        xgb_params = dict(XGB_PARAMS)

        # XGBoost 2.1+ API: eval_metric AND early_stopping_rounds
        # must BOTH be in the constructor. Do NOT pass them to .fit()
        # They are already in XGB_PARAMS from config.py, so they
        # flow straight through to the constructor via **xgb_params.

        # Build model
        warm_start_model = None
        if incremental and self.model is not None:
            warm_start_model = self.model
            log.info(f"[RPDE_MODEL] Incremental training: warming up from "
                     f"existing model ({self.pair})")

        model = xgb.XGBRegressor(
            **xgb_params,
            n_jobs=-1,
            random_state=42,
        )

        # For incremental training with warm_start, disable early
        # stopping and reduce tree count for the incremental update.
        if warm_start_model is not None:
            model.set_params(xgb_model=warm_start_model.get_booster())
            model.set_params(n_estimators=min(200, xgb_params.get('n_estimators', 500)))
            model.set_params(early_stopping_rounds=None)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        self.model = model

        # ── Step 6: Evaluate ──
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        # Clip predictions to same range as target
        val_preds_clipped = np.clip(val_preds, R_CLIP_MIN, R_CLIP_MAX)

        train_mae = float(np.mean(np.abs(train_preds - y_train)))
        val_mae = float(np.mean(np.abs(val_preds_clipped - y_val)))

        # R-squared
        ss_res = float(np.sum((y_val - val_preds_clipped) ** 2))
        ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
        val_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Correlation
        if np.std(val_preds_clipped) > 1e-8 and np.std(y_val) > 1e-8:
            val_corr = float(np.corrcoef(val_preds_clipped, y_val)[0, 1])
        else:
            val_corr = 0.0

        # Quintile calibration
        quintile_calibration = self._quintile_calibration(
            val_preds_clipped, y_val)

        # Feature importance
        top_features = self._get_top_features(model, top_n=15)

        # ── Step 7: Save model and metadata ──
        self.meta = {
            'pair': self.pair,
            'version': '5.0',
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'training_samples': len(X_list),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'skipped_moments': skipped,
            'n_features': self.n_features,
            'incremental': incremental,
            'use_replay': use_replay,
            'train_mae': round(train_mae, 6),
            'val_mae': round(val_mae, 6),
            'val_r2': round(val_r2, 6),
            'val_corr': round(val_corr, 6),
            'top_features': top_features,
            'quintile_calibration': quintile_calibration,
            'y_mean': round(float(np.mean(y)), 4),
            'y_std': round(float(np.std(y)), 4),
            'y_positive_pct': round(float(np.mean(y > 0)), 4),
            'n_negative_samples': negative_count,
            'total_positive_samples': len(X_list) - negative_count - skipped,
        }

        self.save()

        duration = round(time.time() - t0, 2)
        log.info(
            f"[RPDE_MODEL] {self.pair} training complete ({duration}s): "
            f"samples={len(X_list)}, train_mae={train_mae:.4f}, "
            f"val_mae={val_mae:.4f}, val_r2={val_r2:.4f}, "
            f"val_corr={val_corr:.4f}")
        log.info(f"[RPDE_MODEL] {self.pair} quintile calibration:")
        for q, (pred_r, actual_r, wr, n) in quintile_calibration.items():
            log.info(f"  Q{q}: pred_R={pred_r:+.3f}, actual_R={actual_r:+.3f}, "
                     f"WR={wr:.0%}, n={n}")

        return {
            'pair': self.pair,
            'trained': True,
            'samples': len(X_list),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'skipped': skipped,
            'train_mae': round(train_mae, 6),
            'val_mae': round(val_mae, 6),
            'val_r2': round(val_r2, 6),
            'val_corr': round(val_corr, 6),
            'top_features': top_features,
            'quintile_calibration': quintile_calibration,
            'duration_seconds': duration,
        }

    # ════════════════════════════════════════════════════════════════
    # PREDICTION
    # ════════════════════════════════════════════════════════════════

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict pattern match quality for current features.

        Args:
            features: numpy array of shape (93,) or (1, 93) containing
                      the 93 ML Gate feature values in FEATURE_NAMES order.

        Returns:
            dict with predicted_r, confidence, direction, is_pattern.
        """
        if not self.is_trained():
            return {
                'predicted_r': 0.0,
                'confidence': 0.0,
                'direction': None,
                'is_pattern': False,
                'model_loaded': False,
            }

        try:
            # Ensure 2D shape
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Validate feature count
            if features.shape[1] != self.n_features:
                log.warning(
                    f"[RPDE_MODEL] Feature mismatch for {self.pair}: "
                    f"expected {self.n_features}, got {features.shape[1]}")
                return {
                    'predicted_r': 0.0,
                    'confidence': 0.0,
                    'direction': None,
                    'is_pattern': False,
                    'model_loaded': True,
                    'error': f'Feature count mismatch: {features.shape[1]} vs {self.n_features}',
                }

            # Replace NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            predicted_r = float(self.model.predict(features)[0])
            predicted_r = max(R_CLIP_MIN, min(R_CLIP_MAX, predicted_r))
            predicted_r = round(predicted_r, 4)

            # Confidence: sigmoid-like mapping from predicted_r magnitude
            # Higher predicted_r → higher confidence
            # Range: 0.0 to 1.0
            confidence = min(1.0, max(0.0, predicted_r / 2.0))

            # Direction from predicted_r sign
            if predicted_r > 0.05:
                direction = 'BUY'
            elif predicted_r < -0.05:
                direction = 'SELL'
            else:
                direction = None

            # Pattern detection: is this a pattern-favorable condition?
            is_pattern = predicted_r >= TAKE_THRESHOLD

            return {
                'predicted_r': predicted_r,
                'confidence': round(confidence, 4),
                'direction': direction,
                'is_pattern': is_pattern,
                'model_loaded': True,
            }

        except Exception as e:
            log.error(f"[RPDE_MODEL] Prediction failed for {self.pair}: {e}")
            return {
                'predicted_r': 0.0,
                'confidence': 0.0,
                'direction': None,
                'is_pattern': False,
                'model_loaded': True,
                'error': str(e),
            }

    # ════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ════════════════════════════════════════════════════════════════

    def load(self) -> bool:
        """Load model from disk.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        if not os.path.exists(self.model_path):
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True

            # Load metadata
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)

            model_size_kb = round(os.path.getsize(self.model_path) / 1024, 1)
            log.info(f"[RPDE_MODEL] Loaded model for {self.pair}: "
                     f"{model_size_kb}KB, "
                     f"trained={self.meta.get('trained_at', 'unknown')}, "
                     f"samples={self.meta.get('training_samples', '?')}")
            return True

        except Exception as e:
            log.error(f"[RPDE_MODEL] Failed to load model for {self.pair}: {e}")
            self.model = None
            self.is_loaded = False
            return False

    def save(self):
        """Save model and metadata to disk."""
        try:
            os.makedirs(self.model_dir, exist_ok=True)

            # Save model
            joblib.dump(self.model, self.model_path, compress=3)

            # Save metadata
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f, indent=2, default=_json_serializer)

            model_size_kb = round(os.path.getsize(self.model_path) / 1024, 1)
            log.info(f"[RPDE_MODEL] Saved model for {self.pair}: "
                     f"{model_size_kb}KB → {self.model_path}")

        except Exception as e:
            log.error(f"[RPDE_MODEL] Failed to save model for {self.pair}: {e}")

    # ════════════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE
    # ════════════════════════════════════════════════════════════════

    def get_feature_importance(self) -> list:
        """
        Get top feature importances from the trained model.

        Returns:
            List of (feature_name, importance) tuples, sorted descending.
            Empty list if model is not trained.
        """
        if not self.is_trained():
            return []

        return self._get_top_features(self.model, top_n=20)

    # ════════════════════════════════════════════════════════════════
    # MODEL INFO
    # ════════════════════════════════════════════════════════════════

    def get_info(self) -> dict:
        """Get model metadata and status."""
        info = {
            'pair': self.pair,
            'trained': self.is_trained(),
            'model_path': self.model_path,
            'meta_path': self.meta_path,
            'n_features': self.n_features,
        }

        if os.path.exists(self.model_path):
            info['size_kb'] = round(os.path.getsize(self.model_path) / 1024, 1)
            info['age_hours'] = round(
                (time.time() - os.path.getmtime(self.model_path)) / 3600, 1)

        info.update(self.meta)
        return info

    # ════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_top_features(model, top_n: int = 15) -> list:
        """Extract top N feature importances from XGBoost model.

        Returns:
            List of (feature_name, importance) tuples.
        """
        try:
            importance = model.feature_importances_
            # Ensure alignment with FEATURE_NAMES
            n_feats = min(len(importance), len(FEATURE_NAMES))
            feat_imp = [(FEATURE_NAMES[i], round(float(importance[i]), 6))
                        for i in range(n_feats)]
            # Sort by importance descending
            feat_imp.sort(key=lambda x: x[1], reverse=True)
            return feat_imp[:top_n]
        except Exception as e:
            log.debug(f"[RPDE_MODEL] Feature importance extraction failed: {e}")
            return []

    @staticmethod
    def _quintile_calibration(preds: np.ndarray, actuals: np.ndarray) -> dict:
        """
        Compute quintile calibration: for each predicted R quintile,
        what is the actual average R and win rate?

        This measures if the model's predictions are well-calibrated:
        - Top quintile should have highest actual R and WR
        - Bottom quintile should have lowest

        Returns:
            Dict mapping quintile label to (pred_r_avg, actual_r_avg, wr, n).
        """
        result = {}
        n = len(preds)
        if n < 20:
            return result

        quintile_size = n // 5
        labels = ['1_BOTTOM', '2', '3', '4', '5_TOP']

        for i, label in enumerate(labels):
            start = i * quintile_size
            if i < 4:
                end = (i + 1) * quintile_size
            else:
                end = n  # Last quintile gets remainder

            q_preds = preds[start:end]
            q_actuals = actuals[start:end]

            if len(q_actuals) == 0:
                continue

            pred_avg = round(float(np.mean(q_preds)), 4)
            actual_avg = round(float(np.mean(q_actuals)), 4)
            wr = float(np.mean(q_actuals > 0))
            count = int(len(q_actuals))

            result[label] = (pred_avg, actual_avg, round(wr, 4), count)

        return result

    def _load_replay_buffer_data(self) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Load training data from the Experience Replay Buffer.

        Returns:
            Tuple of (X_array, y_list) or (None, None) if unavailable.
        """
        try:
            from ai_engine.replay_buffer import get_replay_buffer
            from rpde.config import REPLAY_BUFFER_SAMPLE_SIZE

            buf = get_replay_buffer()
            if buf.is_empty():
                log.debug("[RPDE_MODEL] Replay buffer is empty, skipping")
                return None, None

            samples = buf.sample(n=REPLAY_BUFFER_SAMPLE_SIZE)
            if not samples:
                return None, None

            X_rows = []
            y_rows = []
            used = 0

            for trade in samples:
                # Extract features from trade dict
                # The replay buffer contains trades with feature columns
                row = []
                valid = True
                for feat_name in FEATURE_NAMES:
                    val = trade.get(feat_name)
                    if val is None:
                        # Map old column names to feature names where possible
                        val = _map_replay_column(trade, feat_name)
                        if val is None:
                            valid = False
                            break
                    row.append(float(val))

                if not valid:
                    continue

                profit_r = trade.get('profit_r')
                if profit_r is None:
                    continue

                try:
                    profit_r = float(profit_r)
                    profit_r = max(R_CLIP_MIN, min(R_CLIP_MAX, profit_r))
                except (TypeError, ValueError):
                    continue

                X_rows.append(row)
                y_rows.append(profit_r)
                used += 1

            if not X_rows:
                return None, None

            log.info(f"[RPDE_MODEL] Loaded {used} samples from replay buffer "
                     f"(of {len(samples)} drawn)")
            return np.array(X_rows, dtype=np.float32), y_rows

        except ImportError:
            log.debug("[RPDE_MODEL] Replay buffer not available")
            return None, None
        except Exception as e:
            log.debug(f"[RPDE_MODEL] Replay buffer loading failed: {e}")
            return None, None


# ════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ════════════════════════════════════════════════════════════════

def _json_serializer(obj):
    """JSON serializer fallback for numpy types."""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, '__float__'):
        return float(obj)
    if hasattr(obj, '__int__'):
        return int(obj)
    return str(obj)


def _map_replay_column(trade: dict, feature_name: str):
    """
    Map a feature name to the corresponding column name in the
    replay buffer's trade dict (which uses DB column names from
    backtest_trades table).

    Returns float value or None if not found.
    """
    # Mapping from FEATURE_NAMES to DB column names
    _COL_MAP = {
        'fq_final_score': 'final_score',
        'fq_market_score': 'market_score',
        'fq_smc_score': 'smc_score',
        'fq_combined_bias': 'combined_bias',
        'fq_bias_confidence': None,  # Not in DB
        'fq_htf_approved': 'htf_approved',
        'fq_htf_score': 'htf_score',
        'of_delta': 'delta',
        'of_rolling_delta': 'rolling_delta',
        'of_delta_bias': 'delta_bias',
        'of_rd_bias': 'rd_bias',
        'of_imbalance': 'of_imbalance',
        'of_imb_strength': 'of_strength',
        'vw_pip_from_vwap': 'pip_from_vwap',
        'vw_position': None,  # Encoded in price_position
        'vw_pip_to_poc': 'pip_to_poc',
        'vw_price_position': 'price_position',
        'smc_structure_trend': 'structure_trend',
        'smc_has_bos': None,
        'smc_bos_direction': None,
        'smc_pd_zone': 'pd_zone',
        'smc_pips_to_eq': 'pips_to_eq',
        'smc_smc_bias': 'smc_bias',
        'smc_has_sweep': None,
        'smc_sweep_aligned': None,
        'tp_score': 'score',
        'tp_sl_pips': 'sl_pips',
        'tp_tp_pips': 'tp_pips',
        'tp_rr_ratio': None,  # Computed
        'tp_direction': 'direction',
        'ss_smc_ob': 'ss_smc_ob',
        'ss_liquidity_sweep': 'ss_liquidity_sweep',
        'ss_delta_divergence': 'ss_delta_divergence',
        'ss_trend_continuation': 'ss_trend_continuation',
        'ss_ema_cross': 'ss_ema_cross',
        'ss_rsi_divergence': 'ss_rsi_divergence',
        'ss_sd_zone': 'ss_supply_demand',
        'ss_bos_momentum': 'ss_bos_momentum',
        'ss_ote_fib': 'ss_optimal_trade',
        'ss_inst_candles': 'ss_institutional',
        'cs_total_signals': None,
        'cs_groups_agreeing': 'agreement_groups',
        'cs_direction_clear': None,
        'st_session': 'session',
        'st_is_london_open': None,
        'st_is_overlap': None,
        'st_is_ny_afternoon': None,
        'st_vol_surge': 'vol_surge_detected',
        'vs_atr': 'atr',
        'vs_market_state': 'market_state',
        'vs_surge_ratio': 'vol_surge_ratio',
        'vs_momentum_velocity': 'momentum_velocity',
        'vs_choppy': 'is_choppy',
        'sym_is_jpy': None,  # Computed from symbol
        'sym_is_commodity': None,
        'sym_is_index': None,
        'si_recent_wr': None,
        'si_recent_avg_r': None,
        'si_strategy_wr': None,
        'fx_spread_pips': 'spread_pips',
        'fib_confluence_score': 'fib_confluence_score',
        'fib_in_golden_zone': 'fib_in_golden_zone',
        'fib_bias_aligned': 'fib_bias_aligned',
        'hist_ps_avg_r_recent': 'hist_ps_avg_r_recent',
        'hist_ps_wr_recent': 'hist_ps_wr_recent',
        'hist_ps_trades_recent': 'hist_ps_trades_recent',
        'hist_ps_avg_r_all': 'hist_ps_avg_r_all',
        'hist_ps_wr_all': 'hist_ps_wr_all',
        'hist_ps_trades_all': 'hist_ps_trades_all',
        'hist_ps_avg_r_decay': 'hist_ps_avg_r_decay',
        'hist_ps_avg_r_trend': 'hist_ps_avg_r_trend',
        'cs_base_strength': 'cs_base_strength',
        'cs_quote_strength': 'cs_quote_strength',
        'cs_strength_delta': 'cs_strength_delta',
        'cs_pair_bias': 'cs_pair_bias',
        'ap_atr_percentile': 'ap_atr_percentile',
        'ap_atr_ratio': 'ap_atr_ratio',
        'mr_m5_rsi': 'mr_m5_rsi',
        'mr_m15_rsi': 'mr_m15_rsi',
        'mr_m30_rsi': 'mr_m30_rsi',
        'mr_h1_rsi': 'mr_h1_rsi',
        'mr_h4_rsi': 'mr_h4_rsi',
        'mr_d1_rsi': 'mr_d1_rsi',
        'mt_mtf_score': 'mt_mtf_score',
        'mt_trend_agreement': 'mt_trend_agreement',
        'mt_rsi_agreement': 'mt_rsi_agreement',
        'im_vix': 'im_vix',
        'im_dxy_change': 'im_dxy_change',
        'im_risk_env': 'im_risk_env',
        'sk_current_streak': 'sk_current_streak',
        'sk_recent_wr': 'sk_recent_wr',
        'zs_signal_zscore': 'zs_signal_zscore',
        'sm_footprint_score': 'sm_footprint_score',
    }

    # Direct match in feature name
    if feature_name in trade:
        return trade[feature_name]

    # Map through column name lookup
    col_name = _COL_MAP.get(feature_name)
    if col_name is not None and col_name in trade:
        val = trade[col_name]
        if val is None:
            return None
        return val

    # Encoded fields that need special handling
    if feature_name == 'tp_direction':
        d = trade.get('direction', '')
        return 1.0 if str(d) == 'BUY' else (-1.0 if str(d) == 'SELL' else 0.0)

    if feature_name == 'vs_vol_surge':
        return 1.0 if trade.get('vol_surge_detected') else 0.0

    if feature_name == 'vs_choppy':
        return 1.0 if trade.get('is_choppy') else 0.0

    if feature_name == 'sym_is_jpy':
        return 1.0 if 'JPY' in str(trade.get('symbol', '')).upper() else 0.0

    if feature_name == 'sym_is_commodity':
        return 1.0 if any(x in str(trade.get('symbol', '')).upper()
                          for x in ['XAU', 'XAG']) else 0.0

    # Try to find by partial match (some columns may have been stored
    # under slightly different names)
    # For fields that have no direct mapping, return None
    return None


# ════════════════════════════════════════════════════════════════
# BATCH OPERATIONS
# ════════════════════════════════════════════════════════════════

def train_all_pairs(pairs: list = None, incremental: bool = False,
                    use_replay: bool = False) -> dict:
    """
    Train pattern models for all specified pairs.

    Args:
        pairs: List of pair strings. If None, uses PAIR_WHITELIST.
        incremental: Pass through to individual model training.
        use_replay: Pass through to individual model training.

    Returns:
        Dict mapping pair -> training result.
    """
    if pairs is None:
        from config.settings import PAIR_WHITELIST
        pairs = PAIR_WHITELIST

    from rpde.database import load_golden_moments

    results = {}
    for pair in pairs:
        try:
            # Load golden moments for this pair from DB
            moments = load_golden_moments(pair=pair)

            if not moments:
                log.info(f"[RPDE_MODEL] No golden moments for {pair}, skipping")
                results[pair] = {
                    'pair': pair,
                    'trained': False,
                    'samples': 0,
                    'reason': 'No golden moments found',
                }
                continue

            model = PatternModel(pair)
            result = model.train(moments, incremental=incremental,
                                 use_replay=use_replay)
            results[pair] = result

        except Exception as e:
            log.error(f"[RPDE_MODEL] Failed to train {pair}: {e}")
            results[pair] = {
                'pair': pair,
                'trained': False,
                'error': str(e),
            }

    # Summary
    trained = sum(1 for r in results.values() if r.get('trained'))
    log.info(f"[RPDE_MODEL] Batch training complete: {trained}/{len(pairs)} "
             f"models trained successfully")

    return results
