# =============================================================
# ai_engine/strategy_model.py  v1.0 — Layer 1 Strategy Model
#
# PURPOSE: Per-strategy XGBoost models that replace hard-coded
# gates inside individual strategy files. Each model learns
# "Is this a good [STRATEGY_NAME] signal?" from historical trades.
#
# ARCHITECTURE (Two-Layer MoE):
#   Layer 1 — Strategy Model (THIS FILE):
#     One XGBoost per strategy, trained ONLY on that strategy's trades.
#     Input:  63 features (same as ML Gate) + strategy-specific features
#     Output: PASS (R >= threshold) or REJECT (R < threshold)
#     Replaces: Hard-coded gates inside each strategy's evaluate()
#
#   Layer 2 — Meta-Model (ai_engine/ml_gate.py v3.3):
#     Current ML Gate, evaluates ALL signals that pass Layer 1.
#     Trained on ALL trades (real + shadow, all strategies).
#
# WHY TWO LAYERS:
#   One model at both stages = conflict of interest (judging itself twice).
#   Layer 1: "Is this a good VWAP signal?" — narrow, specialized
#   Layer 2: "Of all passed signals, which should I execute?" — broad, meta
#
# DATA FLOW:
#   Signal → Layer 1 (PASS/REJECT)
#     REJECT → L1 Shadow (simulate outcome, feed back to L1 training)
#     PASS → Layer 2 ML Gate (TAKE/CAUTION/SKIP)
#       TAKE → Real trade
#       CAUTION/SKIP → L2 Shadow (simulate outcome, feed back to L2 training)
#
# TRAINING DATA:
#   - Source: backtest_trades WHERE strategy = '{strategy_name}'
#   - Includes: BACKTEST + SHADOW trades
#   - Target: profit_r (R-multiple, continuous)
#   - Minimum: 80 trades before training
#   - Conservative params: max_depth=3, lr=0.03 (small dataset)
#
# MODEL FILES:
#   ai_engine/models/{strategy_key}_strategy_model.pkl
#   ai_engine/models/{strategy_key}_strategy_model_meta.json
#
# USAGE:
#   from ai_engine.strategy_model import StrategyModelManager
#   mgr = StrategyModelManager()
#   if mgr.has_model('VWAP_MEAN_REVERSION'):
#       verdict = mgr.evaluate_signal('VWAP_MEAN_REVERSION', features)
#       # verdict = {'pass': True, 'predicted_r': 0.8, ...}
# =============================================================

import os
import json
import time
import numpy as np
from datetime import datetime, timezone
from core.logger import get_logger

log = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ── Thresholds ──
MIN_TRADES_TO_TRAIN = 50
RETRAIN_EVERY_N_TRADES = 50

# ── Regression-specific constants ──
R_CLIP_MIN = -2.0
R_CLIP_MAX = 5.0

# ── Layer 1 verdict thresholds ──
# PASS: predicted R >= this value → signal passes to Layer 2
# REJECT: predicted R < this value → signal rejected, shadow-simulated
PASS_THRESHOLD = 0.2  # Default — lower than Layer 2's 0.5, L1 is permissive

# ── Per-strategy threshold overrides ──
# Strategies with weaker edge need higher thresholds to avoid marginal trades.
STRATEGY_PASS_THRESHOLDS = {
    'EMA_CROSS_MOMENTUM': 0.3,      # Raised from 0.2 — trim marginal trades
    'FVG_REVERSION': 0.5,           # High threshold — only pass if model predicts strong edge
    'RSI_DIVERGENCE_SMC': 0.4,      # Moderate threshold — let model try, generates few signals
    # All other strategies use PASS_THRESHOLD (0.2)
}

# ── Soft-disabled strategies ──
# Empty list = all strategies active (using per-strategy thresholds above).
# Add strategy names here to hard-block them (no signals, no shadow trades).
# DISABLED_STRATEGIES = ['FVG_REVERSION', 'RSI_DIVERGENCE_SMC']  # Uncomment to re-disable
DISABLED_STRATEGIES = []


class StrategyModel:
    """
    A single strategy's Layer 1 model.

    Handles:
    - Model loading/saving
    - Feature extraction from DB rows (reuses ML Gate features)
    - Training from backtest_trades WHERE strategy = self.strategy_name
    - Prediction (PASS/REJECT with predicted R)
    - Self-calibration features (source, predicted_r, prediction_error)
    """

    def __init__(self, strategy_name: str, strategy_key: str = None):
        """
        Args:
            strategy_name: Full strategy name (e.g., 'VWAP_MEAN_REVERSION')
            strategy_key: Short key for model file (e.g., 'vwap'). If None, derived from name.
        """
        self.strategy_name = strategy_name
        self.strategy_key = strategy_key or self._derive_key(strategy_name)
        self.model_path = os.path.join(MODELS_DIR, f'{self.strategy_key}_strategy_model.pkl')
        self.meta_path = os.path.join(MODELS_DIR, f'{self.strategy_key}_strategy_model_meta.json')
        self.model = None
        self.meta = None
        self._loaded = False

    @staticmethod
    def _derive_key(name: str) -> str:
        """Derive a short key from strategy name. VWAP_MEAN_REVERSION -> vwap."""
        # Simple mapping for known strategies
        known = {
            'VWAP_MEAN_REVERSION': 'vwap',
            'BREAKOUT_MOMENTUM': 'breakout',
            'FVG_REVERSION': 'fvg',
            'RSI_DIVERGENCE_SMC': 'rsi_div',
            'SUPERTREND_BOUNCE': 'supertrend',
            'STRUCTURE_ALIGNMENT': 'structure',
            'SMC_OB_REVERSAL': 'smc_ob',
            'ENGULFING': 'engulfing',
            'TREND_CONTINUATION': 'trend_cont',
            'EMA_CROSS_MOMENTUM': 'ema_cross',
            'LIQUIDITY_SWEEP_ENTRY': 'liq_sweep',
            'DELTA_DIVERGENCE': 'delta_div',
        }
        return known.get(name, name.lower().replace('_', '_')[:15])

    def is_trained(self) -> bool:
        """Check if a trained model exists on disk."""
        return os.path.exists(self.model_path)

    def load(self):
        """Load model from disk. Returns True on success."""
        if self._loaded:
            return True
        if not os.path.exists(self.model_path):
            return False
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)
            self._loaded = True
            return True
        except Exception as e:
            log.error(f"[STRAT_MODEL:{self.strategy_key}] Load failed: {e}")
            return False

    def get_info(self) -> dict:
        """Get model metadata."""
        info = {
            'strategy': self.strategy_name,
            'key': self.strategy_key,
            'trained': self.is_trained(),
            'path': self.model_path,
        }
        if self.is_trained():
            info['size_kb'] = round(os.path.getsize(self.model_path) / 1024, 1)
            info['age_hours'] = round(
                (time.time() - os.path.getmtime(self.model_path)) / 3600, 1)
        if self.meta:
            info.update(self.meta)
        return info

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict R-multiple for a signal.

        Returns:
            dict with:
                - pass: bool (predicted_r >= PASS_THRESHOLD)
                - predicted_r: float
                - verdict: 'PASS' or 'REJECT'
                - trained: bool
        """
        if not self.load():
            return {
                'pass': True,  # No model = let it through to existing gates
                'predicted_r': 0.0,
                'verdict': 'NO_MODEL',
                'trained': False,
            }

        try:
            predicted_r = float(self.model.predict(features)[0])
            predicted_r = max(R_CLIP_MIN, min(R_CLIP_MAX, predicted_r))
            # Use per-strategy threshold if defined, else default
            threshold = STRATEGY_PASS_THRESHOLDS.get(self.strategy_name, PASS_THRESHOLD)
            is_pass = predicted_r >= threshold

            return {
                'pass': is_pass,
                'predicted_r': round(predicted_r, 4),
                'verdict': 'PASS' if is_pass else 'REJECT',
                'trained': True,
            }
        except Exception as e:
            log.error(f"[STRAT_MODEL:{self.strategy_key}] Prediction failed: {e}")
            return {
                'pass': True,  # Error = don't block
                'predicted_r': 0.0,
                'verdict': 'ERROR',
                'trained': True,
            }

    def train(self, rows: list) -> dict:
        """
        Train the strategy model from a list of DB row dicts.

        Args:
            rows: List of dicts from backtest_trades WHERE strategy = self.strategy_name

        Returns:
            Training result dict with metrics.
        """
        try:
            import xgboost as xgb
            import joblib
            from sklearn.model_selection import train_test_split

            if len(rows) < MIN_TRADES_TO_TRAIN:
                return {
                    'status': 'skipped',
                    'reason': f'Only {len(rows)} trades (need {MIN_TRADES_TO_TRAIN})',
                    'strategy': self.strategy_name,
                }

            # ── Build feature matrix from DB rows ──
            # Reuse ML Gate's extract_features_from_db for feature consistency
            from ai_engine.ml_gate import extract_features_from_db
            from ai_engine.pair_strategy_features import compute_hist_ps_features_for_training

            # ── Pre-compute historical pair-strategy features (v3.4) ──
            log.info(f"[STRAT_MODEL:{self.strategy_key}] Pre-computing pair-strategy "
                     f"features for {len(rows)} rows...")
            hist_ps_map = compute_hist_ps_features_for_training(rows)
            log.info(f"[STRAT_MODEL:{self.strategy_key}] Pair-strategy features "
                     f"computed for {len(hist_ps_map)} rows")

            X = []
            y = []
            backtest_count = 0
            shadow_count = 0

            for idx, row in enumerate(rows):
                try:
                    hist_ps = hist_ps_map.get(idx)
                    features = extract_features_from_db(row, hist_ps_features=hist_ps)
                    if features is None:
                        continue
                    r_multiple = float(row.get('profit_r', 0) or 0)
                    X.append(features)
                    y.append(r_multiple)
                    if str(row.get('source', 'BACKTEST')) == 'SHADOW':
                        shadow_count += 1
                    else:
                        backtest_count += 1
                except Exception:
                    continue

            if len(X) < MIN_TRADES_TO_TRAIN:
                return {
                    'status': 'skipped',
                    'reason': f'Only {len(X)} valid rows after encoding',
                    'strategy': self.strategy_name,
                }

            X = np.array(X, dtype=np.float32)
            y_raw = np.array(y, dtype=np.float32)
            y = np.clip(y_raw, R_CLIP_MIN, R_CLIP_MAX)

            # ── Target statistics ──
            win_count = int((y_raw > 0).sum())
            loss_count = int((y_raw <= 0).sum())
            wr = win_count / len(y) * 100
            mean_r = float(np.mean(y_raw))
            median_r = float(np.median(y_raw))

            log.info(f"[STRAT_MODEL:{self.strategy_key}] Training: {len(y)} trades "
                     f"({backtest_count} real + {shadow_count} shadow) "
                     f"({win_count}W/{loss_count}L = {wr:.1f}% WR, "
                     f"mean_R={mean_r:.3f})")

            # ── Train/val split ──
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # ── Adaptive XGBoost params by dataset size ──
            n_trades = len(y)
            if n_trades < 500:
                depth, eta, child_w = 3, 0.03, 10
                alpha, lam, cols = 1.0, 5.0, 0.5
            elif n_trades < 1000:
                depth, eta, child_w = 4, 0.04, 7
                alpha, lam, cols = 0.5, 3.0, 0.6
            else:
                depth, eta, child_w = 5, 0.05, 5
                alpha, lam, cols = 0.1, 1.0, 0.7

            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=depth,
                learning_rate=eta,
                objective='reg:squarederror',
                random_state=42,
                min_child_weight=child_w,
                subsample=0.8,
                colsample_bytree=cols,
                reg_alpha=alpha,
                reg_lambda=lam,
                early_stopping_rounds=50,
            )

            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

            # ── Regression metrics ──
            val_preds = model.predict(X_val)
            train_preds = model.predict(X_train)

            train_mae = float(np.mean(np.abs(train_preds - y_train)))
            val_mae = float(np.mean(np.abs(val_preds - y_val)))
            val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))

            ss_res = np.sum((y_val - val_preds) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            val_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            if len(y_val) > 2 and np.std(y_val) > 0 and np.std(val_preds) > 0:
                val_corr = float(np.corrcoef(y_val, val_preds)[0, 1])
            else:
                val_corr = 0.0

            # ── Feature importance (top 15) ──
            from ai_engine.ml_gate import FEATURE_NAMES
            importance = model.feature_importances_
            top_features = sorted(zip(FEATURE_NAMES, importance),
                                  key=lambda x: x[1], reverse=True)[:15]

            # ── Quintile calibration ──
            calibration = _check_calibration(val_preds, y_val)

            # ── Retrain on full data for production ──
            best_n = getattr(model, 'best_iteration', 500)
            model_final = xgb.XGBRegressor(
                n_estimators=best_n,
                max_depth=depth,
                learning_rate=eta,
                objective='reg:squarederror',
                random_state=42,
                min_child_weight=child_w,
                subsample=0.8,
                colsample_bytree=cols,
                reg_alpha=alpha,
                reg_lambda=lam,
            )
            model_final.fit(X, y, verbose=False)

            # ── Save model + metadata ──
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(model_final, self.model_path)

            # Count how many trades PASS vs REJECT with this model
            all_preds = model_final.predict(X)
            threshold = STRATEGY_PASS_THRESHOLDS.get(self.strategy_name, PASS_THRESHOLD)
            pass_count = int((all_preds >= threshold).sum())
            reject_count = len(all_preds) - pass_count
            pass_wr = 0.0
            reject_wr = 0.0
            if pass_count > 0:
                pass_wr = float((y_raw[all_preds >= threshold] > 0).sum() / pass_count * 100)
            if reject_count > 0:
                reject_wr = float((y_raw[all_preds < threshold] > 0).sum() / reject_count * 100)

            meta = {
                'status': 'trained',
                'version': '1.0-strategy',
                'model_type': 'XGBRegressor',
                'target': 'profit_r (R-multiple)',
                'strategy': self.strategy_name,
                'strategy_key': self.strategy_key,
                'layer': 1,
                'n_features': len(FEATURE_NAMES),
                'total_trades': int(len(y)),
                'wins': int(win_count),
                'losses': int(loss_count),
                'win_rate': float(round(wr, 1)),
                'mean_r': round(mean_r, 3),
                'median_r': round(median_r, 3),
                # Regression metrics
                'train_mae': round(train_mae, 4),
                'val_mae': round(val_mae, 4),
                'val_rmse': round(val_rmse, 4),
                'val_r2': round(val_r2, 4),
                'val_correlation': round(val_corr, 4),
                # Thresholds
                'pass_threshold': threshold,
                'best_iteration': int(best_n),
                # Selection stats (model's own filtering)
                'pass_count': pass_count,
                'reject_count': reject_count,
                'pass_wr': round(pass_wr, 1),
                'reject_wr': round(reject_wr, 1),
                # Feature importance
                'top_features': [(f, float(round(i, 4))) for f, i in top_features],
                'calibration': calibration,
                'model_size_kb': float(round(os.path.getsize(self.model_path) / 1024, 1)),
                'trained_at': datetime.now(timezone.utc).isoformat(),
            }
            with open(self.meta_path, 'w') as f:
                json.dump(meta, f, indent=2, default=_json_default)

            self.model = model_final
            self.meta = meta
            self._loaded = True

            log.info(f"[STRAT_MODEL:{self.strategy_key}] Trained: {len(y)} trades "
                     f"(MAE={val_mae:.3f}, R2={val_r2:.3f}, corr={val_corr:.3f}) "
                     f"PASS={pass_count} ({pass_wr:.1f}% WR) "
                     f"REJECT={reject_count} ({reject_wr:.1f}% WR)")

            return meta

        except ImportError as e:
            return {'status': 'error', 'reason': f'Missing dependency: {e}',
                    'strategy': self.strategy_name}
        except Exception as e:
            log.error(f"[STRAT_MODEL:{self.strategy_key}] Training failed: {e}")
            return {'status': 'error', 'reason': str(e),
                    'strategy': self.strategy_name}


class StrategyModelManager:
    """
    Manages all Layer 1 strategy models.

    Responsibilities:
    - Load available strategy models
    - Route signals through their respective model
    - Track which strategies have active models
    - Provide shadow data for L1 rejections
    """

    # All 10 strategies with their model keys
    STRATEGY_REGISTRY = {
        'VWAP_MEAN_REVERSION': 'vwap',
        'BREAKOUT_MOMENTUM': 'breakout',
        'FVG_REVERSION': 'fvg',
        'RSI_DIVERGENCE_SMC': 'rsi_div',
        'SUPERTREND_BOUNCE': 'supertrend',
        'STRUCTURE_ALIGNMENT': 'structure',
        'SMC_OB_REVERSAL': 'smc_ob',
        'ENGULFING': 'engulfing',
        'TREND_CONTINUATION': 'trend_cont',
        'EMA_CROSS_MOMENTUM': 'ema_cross',
        'LIQUIDITY_SWEEP_ENTRY': 'liq_sweep',
        'DELTA_DIVERGENCE': 'delta_div',
    }

    def __init__(self):
        self._models = {}  # strategy_name -> StrategyModel instance
        self._active = set()  # set of strategy names with loaded models
        self._load_available_models()

    def _load_available_models(self):
        """Scan the models directory and load any existing strategy models."""
        for strategy_name, strategy_key in self.STRATEGY_REGISTRY.items():
            # Skip soft-disabled strategies — don't load their models
            if strategy_name in DISABLED_STRATEGIES:
                log.info(f"[STRAT_MODEL] Skipping DISABLED strategy: {strategy_name}")
                continue
            model = StrategyModel(strategy_name, strategy_key)
            if model.is_trained():
                if model.load():
                    self._models[strategy_name] = model
                    self._active.add(strategy_name)
                    log.info(f"[STRAT_MODEL] Loaded model for {strategy_name} "
                             f"({strategy_key})")

    def has_model(self, strategy_name: str) -> bool:
        """Check if a strategy has an active Layer 1 model."""
        return strategy_name in self._active

    def get_model(self, strategy_name: str) -> StrategyModel:
        """Get the StrategyModel instance for a strategy (or None)."""
        return self._models.get(strategy_name)

    def evaluate_signal(self, strategy_name: str, features: np.ndarray) -> dict:
        """
        Evaluate a signal through its Layer 1 model.

        Args:
            strategy_name: The strategy that generated the signal
            features: Feature array (same 66 features as ML Gate)

        Returns:
            dict with:
                - has_model: bool
                - pass: bool (True = let signal through to Layer 2)
                - predicted_r: float
                - verdict: 'PASS', 'REJECT', 'NO_MODEL', or 'ERROR'
        """
        model = self._models.get(strategy_name)
        if model is None:
            # No model for this strategy — pass through to existing gates
            return {
                'has_model': False,
                'pass': True,
                'predicted_r': 0.0,
                'verdict': 'NO_MODEL',
            }

        # Soft-disable check: reject signals from disabled strategies
        if strategy_name in DISABLED_STRATEGIES:
            return {
                'has_model': True,
                'pass': False,
                'predicted_r': -1.0,
                'verdict': 'DISABLED',
            }

        result = model.predict(features)
        result['has_model'] = True
        return result

    def train_strategy(self, strategy_name: str, rows: list = None) -> dict:
        """
        Train a specific strategy model.

        Args:
            strategy_name: Strategy to train
            rows: Optional pre-fetched DB rows. If None, will query DB.

        Returns:
            Training result dict.
        """
        strategy_key = self.STRATEGY_REGISTRY.get(strategy_name)
        if not strategy_key:
            return {
                'status': 'error',
                'reason': f'Unknown strategy: {strategy_name}',
            }

        model = StrategyModel(strategy_name, strategy_key)

        # Fetch from DB if rows not provided
        if rows is None:
            rows = self._fetch_strategy_trades(strategy_name)

        result = model.train(rows)

        if result.get('status') == 'trained':
            self._models[strategy_name] = model
            self._active.add(strategy_name)

        return result

    # Strategy-specific feature table JOINs for _fetch_strategy_trades()
    # NOTE: Columns that conflict with backtest_trades columns use AS aliases.
    #       The extraction functions (e.g., extract_delta_div_features_from_db) check
    #       for 'div_type' to detect if the JOIN succeeded, then read aliased cols.
    _FEATURE_TABLE_JOINS = {
        'DELTA_DIVERGENCE': {
            'table': 'backtest_delta_div_features',
            'alias': 'ddf',
            'columns': [
                'ddf.div_type',
                'ddf.div_strength',
                'ddf.swing_range_pips',
                'ddf.delta_value',
                'ddf.delta_bias AS ddf_delta_bias',
                'ddf.of_imbalance AS ddf_of_imbalance',
                'ddf.of_strength AS ddf_of_strength',
                'ddf.vol_surge',
                'ddf.surge_ratio',
                'ddf.surge_absorption',
                'ddf.stoch_rsi_k',
                'ddf.stoch_rsi_turning',
                'ddf.pd_zone AS ddf_pd_zone',
                'ddf.m5_body_ratio',
                'ddf.atr_pips',
            ],
        },
        'RSI_DIVERGENCE_SMC': {
            'table': 'backtest_rsi_div_features',
            'alias': 'rdf',
            'columns': [
                'rdf.div_type',
                'rdf.div_strength',
                'rdf.rsi_diff',
                'rdf.curr_rsi',
                'rdf.prev_rsi',
                'rdf.price_range_pips',
                'rdf.smc_confirmed',
                'rdf.smc_bias AS rdf_smc_bias',
                'rdf.ob_distance_pips',
                'rdf.fvg_distance_pips',
                'rdf.delta_bias AS rdf_delta_bias',
                'rdf.of_imbalance AS rdf_of_imbalance',
                'rdf.stoch_rsi_k',
                'rdf.pd_zone AS rdf_pd_zone',
                'rdf.is_choppy AS rdf_is_choppy',
                'rdf.atr_pips',
            ],
        },
        'LIQUIDITY_SWEEP_ENTRY': {
            'table': 'backtest_liq_sweep_features',
            'alias': 'lsf',
            'columns': [
                'lsf.sweep_bias',
                'lsf.reversal_pips',
                'lsf.swept_level_dist',
                'lsf.delta_strength',
                'lsf.has_bos',
                'lsf.bos_type',
                'lsf.stoch_rsi_k',
                'lsf.supertrend_dir_h1',
                'lsf.htf_ok',
                'lsf.smc_bias AS lsf_smc_bias',
                'lsf.pd_zone AS lsf_pd_zone',
                'lsf.vol_surge',
                'lsf.of_imbalance AS lsf_of_imbalance',
                'lsf.atr_pips',
            ],
        },
        'EMA_CROSS_MOMENTUM': {
            'table': 'backtest_ema_cross_features',
            'alias': 'ecf',
            'columns': [
                'ecf.h4_cross_bars_ago',
                'ecf.h4_cross_strength',
                'ecf.h4_alignment_score',
                'ecf.h1_rsi',
                'ecf.m15_adx',
                'ecf.delta_bias AS ecf_delta_bias',
                'ecf.of_imbalance AS ecf_of_imbalance',
                'ecf.of_strength AS ecf_of_strength',
                'ecf.h1_supertrend_dir',
                'ecf.h4_supertrend_dir',
                'ecf.h4_ema_spread_9_21',
                'ecf.is_choppy AS ecf_is_choppy',
                'ecf.vol_surge',
                'ecf.atr_pips',
            ],
        },
    }

    def _fetch_strategy_trades(self, strategy_name: str) -> list:
        """Fetch all trades for a strategy from the database.
        
        Includes LEFT JOIN on strategy-specific feature tables when available,
        so that strategy-specific feature builders (e.g., build_delta_div_feature_vector)
        can use real feature data instead of falling back to defaults.
        """
        try:
            from database.db_manager import get_connection
            from backtest.db_store import _ensure_tables

            conn = get_connection()
            try:
                _ensure_tables(conn)
            except Exception:
                pass

            cursor = conn.cursor(dictionary=True)

            # Base columns from backtest_trades
            base_columns = """
                t.symbol, t.direction, t.strategy, t.session, t.market_state,
                t.score, t.sl_pips, t.tp_pips, t.confluence_count,
                t.delta, t.rolling_delta, t.delta_bias, t.rd_bias,
                t.of_imbalance, t.of_strength, t.vol_surge_detected, t.vol_surge_ratio,
                t.momentum_velocity, t.is_choppy,
                t.smc_bias, t.pd_zone, t.pips_to_eq, t.structure_trend,
                t.atr, t.pip_from_vwap, t.pip_to_poc, t.va_width_pips, t.price_position,
                t.final_score, t.market_score, t.smc_score, t.htf_approved, t.htf_score,
                t.combined_bias, t.agreement_groups,
                t.spread_pips, t.slippage_pips,
                t.ss_smc_ob, t.ss_liquidity_sweep, t.ss_vwap_reversion,
                t.ss_delta_divergence, t.ss_trend_continuation,
                t.ss_fvg_reversion, t.ss_ema_cross, t.ss_rsi_divergence,
                t.ss_breakout_momentum, t.ss_structure_align,
                t.profit_pips, t.profit_r, t.win,
                t.source, t.model_predicted_r, t.entry_time
            """

            # Check if this strategy has a feature table JOIN
            join_info = self._FEATURE_TABLE_JOINS.get(strategy_name)
            if join_info:
                extra_cols = ',\n                '.join(join_info['columns'])
                join_clause = (
                    f" LEFT JOIN {join_info['table']} {join_info['alias']} "
                    f"ON {join_info['alias']}.trade_id = t.id "
                )
                query = f"""
                    SELECT
                        {base_columns},
                        {extra_cols}
                    FROM backtest_trades t
                    {join_clause}
                    WHERE t.strategy = %s
                      AND t.source IN ('BACKTEST', 'SHADOW')
                      AND t.outcome IS NOT NULL
                      AND t.outcome != ''
                      AND t.win IS NOT NULL
                      AND t.profit_r IS NOT NULL
                    ORDER BY t.entry_time ASC
                """
            else:
                query = f"""
                    SELECT
                        {base_columns}
                    FROM backtest_trades t
                    WHERE t.strategy = %s
                      AND t.source IN ('BACKTEST', 'SHADOW')
                      AND t.outcome IS NOT NULL
                      AND t.outcome != ''
                      AND t.win IS NOT NULL
                      AND t.profit_r IS NOT NULL
                    ORDER BY t.entry_time ASC
                """

            cursor.execute(query, (strategy_name,))
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception as e:
            log.error(f"[STRAT_MODEL] Failed to fetch trades for {strategy_name}: {e}")
            return []

    def get_status(self) -> dict:
        """Get status of all strategy models."""
        status = {
            'active_models': list(self._active),
            'total_active': len(self._active),
            'strategies': {},
        }
        for name, key in self.STRATEGY_REGISTRY.items():
            if name in self._active:
                status['strategies'][name] = self._models[name].get_info()
            else:
                # Check if model file exists but wasn't loaded
                model = StrategyModel(name, key)
                status['strategies'][name] = model.get_info()
        return status


def _check_calibration(preds: np.ndarray, actuals: np.ndarray,
                      n_buckets: int = 5) -> list:
    """Check model calibration by splitting predictions into quintiles."""
    try:
        if len(preds) < n_buckets * 3:
            return []

        sorted_indices = np.argsort(preds)
        bucket_size = len(sorted_indices) // n_buckets
        if bucket_size < 3:
            return []

        calibration = []
        labels = ['Q1 (worst)', 'Q2', 'Q3 (middle)', 'Q4', 'Q5 (best)']

        for i in range(n_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < n_buckets - 1 else len(sorted_indices)
            indices = sorted_indices[start:end]

            bucket_preds = preds[indices]
            bucket_actuals = actuals[indices]

            calibration.append({
                labels[i]: {
                    'predicted_r': round(float(np.mean(bucket_preds)), 3),
                    'actual_mean_r': round(float(np.mean(bucket_actuals)), 3),
                    'win_rate_pct': round(float((bucket_actuals > 0).sum() / len(bucket_actuals) * 100), 1),
                    'count': int(len(bucket_actuals)),
                }
            })

        return calibration
    except Exception:
        return []


def _json_default(obj):
    """Fallback serializer for numpy types."""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, '__float__'):
        return float(obj)
    return str(obj)


# ── Module-level singleton ──
_manager_instance = None


def get_strategy_model_manager() -> StrategyModelManager:
    """Get or create the global StrategyModelManager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = StrategyModelManager()
    return _manager_instance
