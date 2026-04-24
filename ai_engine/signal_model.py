# =============================================================
# ai_engine/signal_model.py
# PURPOSE: Strategy-Informed XGBoost — the brain of the bot.
#
# WHAT IT DOES:
#   - Takes 60 features (market + ALL strategy scores)
#   - Predicts: probability this trade will WIN
#   - Replaces multi-group consensus as the final gate
#   - Retrains automatically every 50 completed trades
#
# SELF-IMPROVEMENT LOOP:
#   1. Every trade is recorded with its 60 features + outcome
#   2. Every 50 trades → model retrains on ALL history
#   3. Model gets smarter every week without any manual work
#
# DECISION THRESHOLD:
#   ≥ 0.65 (65%) → TAKE the trade
#   0.50-0.65    → CAUTION (reduce size)
#   < 0.50       → SKIP
# =============================================================

import os
import numpy as np
from core.logger import get_logger
from ai_engine.feature_extractor import (
    extract_features, FEATURE_NAMES, FEATURE_COUNT)

log = get_logger(__name__)

MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'models', 'signal_model_v2.pkl')
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'models', 'training_history.pkl')
SCALER_PATH  = os.path.join(os.path.dirname(__file__), 'models', 'feature_scaler.pkl')

MIN_TRADES_TO_TRAIN = 50     # Need at least 50 completed trades
RETRAIN_EVERY       = 50     # Retrain every N new trades
WIN_PROB_THRESHOLD  = 0.62   # Must exceed this to take trade
CAUTION_THRESHOLD   = 0.50   # Below this = skip entirely


class SignalModel:
    """
    Strategy-Informed XGBoost model.
    Trained on completed trade outcomes.
    Predicts WIN probability from 60 features.
    """

    def __init__(self):
        self._model   = None
        self._trained = False
        self._history = []           # list of (features_array, label)
        self._trades_since_retrain = 0
        self._performance_cache = {}  # recent win rates by symbol/strategy/session
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self._load()

    # ── Public API ────────────────────────────────────────────

    def predict(self,
                signal: dict,
                master_report: dict,
                market_report: dict,
                smc_report: dict,
                all_signals: list = None,
                symbol: str = None) -> dict:
        """
        Predict WIN probability for a signal.
        Returns dict with probability, decision, and confidence band.
        """
        features = extract_features(
            signal, master_report, market_report, smc_report,
            all_signals=all_signals, symbol=symbol,
            performance_cache=self._performance_cache)

        if features is None:
            return self._neutral_result('feature_extraction_failed')

        if not self._trained or self._model is None:
            return self._neutral_result('model_not_trained_yet')

        try:
            win_prob  = float(self._model.predict_proba(features)[0][1])
            loss_prob = 1.0 - win_prob

            if win_prob >= WIN_PROB_THRESHOLD:
                decision = 'TAKE'
                # Scale position size with confidence
                # 65% → 0.7x size, 80% → 1.0x size, 90% → 1.2x size
                size_mult = round(min(1.3, max(0.5,
                    (win_prob - WIN_PROB_THRESHOLD) / (1 - WIN_PROB_THRESHOLD) * 1.3)), 2)
            elif win_prob >= CAUTION_THRESHOLD:
                decision  = 'CAUTION'
                size_mult = 0.5
            else:
                decision  = 'SKIP'
                size_mult = 0.0

            return {
                'win_probability':  round(win_prob, 4),
                'loss_probability': round(loss_prob, 4),
                'decision':         decision,
                'size_multiplier':  size_mult,
                'trained':          True,
                'features_used':    FEATURE_COUNT,
            }

        except Exception as e:
            log.error(f"[MODEL] Prediction error: {e}")
            return self._neutral_result(f'prediction_error:{e}')

    def record_outcome(self,
                       signal: dict,
                       master_report: dict,
                       market_report: dict,
                       smc_report: dict,
                       outcome: str,
                       profit_pips: float,
                       all_signals: list = None,
                       symbol: str = None):
        """
        Record a completed trade outcome for future retraining.
        Call this when a trade closes (WIN or LOSS).

        outcome: 'WIN' or 'LOSS'
        profit_pips: actual pips gained (positive=win, negative=loss)
        """
        features = extract_features(
            signal, master_report, market_report, smc_report,
            all_signals=all_signals, symbol=symbol,
            performance_cache=self._performance_cache)

        if features is None:
            log.warning("[MODEL] Could not record outcome — feature extraction failed")
            return

        label = 1 if outcome == 'WIN' else 0
        self._history.append((features, label))
        self._trades_since_retrain += 1

        # Update performance cache
        self._update_performance_cache(signal, symbol, outcome)

        log.debug(f"[MODEL] Recorded {outcome} ({profit_pips:+.1f}p) "
                  f"— history: {len(self._history)} trades "
                  f"({self._trades_since_retrain} since last retrain)")

        # Auto-retrain every N trades
        if self._trades_since_retrain >= RETRAIN_EVERY:
            log.info(f"[MODEL] Auto-retraining after "
                     f"{self._trades_since_retrain} new trades...")
            self.retrain()

        self._save_history()

    def retrain(self) -> dict:
        """
        Retrain XGBoost on ALL recorded history.
        Called automatically every 50 trades, or manually.
        Returns training results dict.
        """
        if len(self._history) < MIN_TRADES_TO_TRAIN:
            log.info(f"[MODEL] Only {len(self._history)} trades — "
                     f"need {MIN_TRADES_TO_TRAIN} to retrain")
            return {'status': 'skipped',
                    'reason': f'{len(self._history)} < {MIN_TRADES_TO_TRAIN}'}

        try:
            import xgboost as xgb
            from sklearn.model_selection import StratifiedKFold, cross_val_score

            X = np.vstack([f for f, _ in self._history])
            y = np.array([l for _, l in self._history], dtype=np.int32)

            win_count  = int(y.sum())
            loss_count = len(y) - win_count
            win_rate   = round(win_count / len(y) * 100, 1)

            log.info(f"[MODEL] Training on {len(y)} trades "
                     f"({win_count}W/{loss_count}L = {win_rate}% WR)")

            # Handle class imbalance
            scale_pw = (loss_count / win_count
                        if win_count > 0 and win_count < loss_count
                        else 1.0)

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pw,
                use_label_encoder=False,
                eval_metric='auc',
                random_state=42,
                n_jobs=-1,
            )

            # Cross-validation for honest accuracy estimate
            cv_scores = cross_val_score(
                model, X, y, cv=StratifiedKFold(n_splits=5),
                scoring='roc_auc')
            cv_auc = round(float(cv_scores.mean()), 4)

            # Train on full data for production
            model.fit(X, y)

            # Feature importance — which features matter most
            importances = model.feature_importances_
            top10 = sorted(zip(FEATURE_NAMES, importances),
                           key=lambda x: x[1], reverse=True)[:10]

            self._model   = model
            self._trained = True
            self._trades_since_retrain = 0
            self._save_model()

            log.info(f"[MODEL] Retrained ✅  CV-AUC={cv_auc:.3f}  "
                     f"Top feature: {top10[0][0]} ({top10[0][1]:.3f})")

            return {
                'status':     'retrained',
                'trades':     len(y),
                'win_rate':   win_rate,
                'cv_auc':     cv_auc,
                'top_features': [(f, round(i, 4)) for f, i in top10],
            }

        except Exception as e:
            log.error(f"[MODEL] Retraining failed: {e}")
            return {'status': 'error', 'reason': str(e)}

    def train_from_backtest_db(self) -> dict:
        """
        Seed the model using your existing backtest database.
        Call this ONCE to bootstrap — then live trades take over.
        This is how you get the model working before 50 live trades.
        """
        try:
            from database.db_manager import get_connection
            conn   = get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("""
                SELECT * FROM backtest_trades
                WHERE outcome IS NOT NULL
                  AND outcome != ''
                  AND source = 'BACKTEST'
                LIMIT 5000
            """)
            rows = cursor.fetchall()
            conn.close()

            if len(rows) < MIN_TRADES_TO_TRAIN:
                return {'status': 'skipped',
                        'reason': f'Only {len(rows)} backtest trades'}

            # Convert DB rows to feature vectors
            converted = 0
            for row in rows:
                try:
                    # Reconstruct minimal signal dict from DB row
                    signal = {
                        'strategy':    row.get('strategy', ''),
                        'direction':   row.get('direction', 'BUY'),
                        'score':       float(row.get('score', 0) or 0),
                        'sl_pips':     float(row.get('sl_pips', 10) or 10),
                        'tp1_pips':    float(row.get('tp_pips', 20) or 20),
                        'confluence':  [],
                    }
                    # Build minimal market/smc reports from stored columns
                    market_report = {
                        'delta': {'delta': float(row.get('delta', 0) or 0),
                                  'bias': _decode_bias(row.get('delta_bias', 0))},
                        'rolling_delta': {'delta': float(row.get('rolling_delta', 0) or 0),
                                          'bias': _decode_bias(row.get('rd_bias', 0))},
                        'order_flow_imbalance': {
                            'imbalance': float(row.get('of_imbalance', 0) or 0),
                            'strength':  row.get('of_strength', 'NONE') or 'NONE'},
                        'volume_surge': {'surge_ratio': float(row.get('surge_ratio', 1) or 1),
                                         'surge_detected': bool(row.get('surge_detected', 0))},
                        'momentum': {'is_scalpable': bool(row.get('is_scalpable', 0)),
                                     'is_choppy': bool(row.get('is_choppy', 1))},
                        'vwap': {'pip_from_vwap': float(row.get('pip_from_vwap', 0) or 0),
                                 'position': row.get('vwap_position', 'AT_VWAP') or 'AT_VWAP'},
                        'profile': {'pip_to_poc': float(row.get('pip_to_poc', 0) or 0),
                                    'va_width_pips': float(row.get('va_width_pips', 20) or 20),
                                    'price_position': row.get('price_position', 'INSIDE_VA') or 'INSIDE_VA',
                                    'current_price': 1.0},
                        'final_score': float(row.get('final_score', 50) or 50),
                        'smc_score':   float(row.get('smc_score', 50) or 50),
                        'market_state': row.get('market_state', 'BALANCED') or 'BALANCED',
                        'session': row.get('session', 'UNKNOWN') or 'UNKNOWN',
                    }
                    master_report = {
                        'final_score':   market_report['final_score'],
                        'market_score':  float(row.get('market_score', 50) or 50),
                        'smc_score':     market_report['smc_score'],
                        'combined_bias': row.get('combined_bias', 'NEUTRAL') or 'NEUTRAL',
                        'market_state':  market_report['market_state'],
                        'session':       market_report['session'],
                        'order_flow_imbalance': market_report['order_flow_imbalance'],
                        'volume_surge':  market_report['volume_surge'],
                        'momentum':      market_report['momentum'],
                    }
                    smc_report = {
                        'premium_discount': {'zone': row.get('pd_zone', 'NEUTRAL') or 'NEUTRAL',
                                             'pips_to_eq': float(row.get('pips_to_eq', 0) or 0)},
                        'htf_alignment': {'approved': bool(row.get('htf_approved', 0)),
                                          'score': float(row.get('htf_score', 50) or 50),
                                          'h4_bias': ''},
                        'structure': {},
                        'nearest_ob': None,
                        'nearest_fvg': None,
                        'last_sweep': {},
                    }

                    features = extract_features(
                        signal, master_report, market_report, smc_report,
                        symbol=row.get('symbol', ''))

                    if features is not None:
                        label = 1 if row.get('win') else 0
                        self._history.append((features, label))
                        converted += 1

                except Exception:
                    continue

            log.info(f"[MODEL] Loaded {converted}/{len(rows)} backtest trades into history")
            self._save_history()

            # Now train on the loaded data
            return self.retrain()

        except Exception as e:
            log.error(f"[MODEL] Backtest seed failed: {e}")
            return {'status': 'error', 'reason': str(e)}

    def get_status(self) -> dict:
        """Return model status for dashboard display."""
        return {
            'trained':              self._trained,
            'total_history':        len(self._history),
            'trades_since_retrain': self._trades_since_retrain,
            'next_retrain_in':      max(0, RETRAIN_EVERY - self._trades_since_retrain),
            'win_prob_threshold':   WIN_PROB_THRESHOLD,
            'feature_count':        FEATURE_COUNT,
            'model_path':           MODEL_PATH,
            'model_exists':         os.path.exists(MODEL_PATH),
        }

    # ── Private helpers ───────────────────────────────────────

    def _update_performance_cache(self, signal, symbol, outcome):
        """Track recent win rates by symbol/strategy/session."""
        is_win = 1 if outcome == 'WIN' else 0
        alpha  = 0.1  # Exponential moving average — recent trades matter more

        for key in [
            f'symbol_{(symbol or "").upper()}',
            f'strategy_{signal.get("strategy", "")}',
            f'session_{signal.get("session", "")}',
        ]:
            prev = self._performance_cache.get(key, 0.5)
            self._performance_cache[key] = round(
                prev * (1 - alpha) + is_win * alpha, 4)

    def _neutral_result(self, reason: str) -> dict:
        return {
            'win_probability':  0.5,
            'loss_probability': 0.5,
            'decision':         'NEUTRAL',
            'size_multiplier':  1.0,
            'trained':          self._trained,
            'reason':           reason,
        }

    def _save_model(self):
        try:
            import joblib
            joblib.dump(self._model, MODEL_PATH)
        except Exception as e:
            log.error(f"[MODEL] Save failed: {e}")

    def _save_history(self):
        try:
            import joblib
            joblib.dump((self._history, self._performance_cache), HISTORY_PATH)
        except Exception as e:
            log.error(f"[MODEL] History save failed: {e}")

    def _load(self):
        """Load model and history from disk on startup."""
        try:
            import joblib
            if os.path.exists(MODEL_PATH):
                self._model   = joblib.load(MODEL_PATH)
                self._trained = True
                log.info(f"[MODEL] Loaded trained model from {MODEL_PATH}")
            if os.path.exists(HISTORY_PATH):
                loaded = joblib.load(HISTORY_PATH)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    self._history, self._performance_cache = loaded
                else:
                    self._history = loaded
                log.info(f"[MODEL] Loaded {len(self._history)} trade history entries")
        except Exception as e:
            log.warning(f"[MODEL] Could not load from disk: {e}")


def _decode_bias(val) -> str:
    """Convert numeric delta_bias DB column back to string."""
    v = float(val or 0)
    if v > 0:   return 'BULLISH'
    if v < 0:   return 'BEARISH'
    return 'NEUTRAL'


# ── Singleton instance — shared across the bot ───────────────
_model_instance = None

def get_model() -> SignalModel:
    """Get or create the singleton SignalModel instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SignalModel()
    return _model_instance
