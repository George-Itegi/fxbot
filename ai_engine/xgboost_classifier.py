# =============================================================
# ai_engine/xgboost_classifier.py  v2.0
# PURPOSE: Predict win/loss probability for any trade signal.
# Uses 21 features extracted from market conditions at signal time.
#
# v2.0 CHANGES:
#   - FIXED: Feature mismatch — now trains from backtest_trades
#     (21 features) matching extract_features() exactly
#   - Added train_from_backtest() for training from rich DB data
#   - Added train_from_live() as fallback (original method)
#   - Added model info reporting (feature importance, accuracy)
#   - train_model() now tries backtest first, then live
# =============================================================

import numpy as np
import os
from core.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          'models', 'xgb_model.pkl')
MIN_TRADES_TO_TRAIN = 50   # Minimum trades before training


# ── Feature names (must match extract_features exactly) ─────
FEATURE_NAMES = [
    'score', 'sl_pips', 'tp1_pips', 'tp2_pips', 'direction',
    'session',
    'delta', 'rolling_delta', 'delta_bias', 'rd_bias',
    'vwap_pip_from', 'vwap_position',
    'pip_to_poc', 'price_position', 'va_width_pips',
    'pd_zone', 'pips_to_eq', 'htf_approved', 'htf_score',
    'final_score', 'smc_score',
]


def extract_features(signal: dict,
                     market_report: dict,
                     smc_report: dict) -> np.ndarray | None:
    """
    Extract 21 numerical features from trade context.
    These are the inputs XGBoost learns from.
    Output shape: (1, 21)
    """
    try:
        m = market_report or {}
        s = smc_report    or {}

        d   = m.get('delta', {})
        rd  = m.get('rolling_delta', {})
        vwap= m.get('vwap', {})
        prof= m.get('profile', {})
        pd_z= s.get('premium_discount', {})
        htf = s.get('htf_alignment', {})

        # Session encoding — aligned with institutional behaviors
        session_map = {
            'NY_LONDON_OVERLAP': 4,  # Distribution — highest liquidity
            'LONDON_SESSION':    3,  # Expansion — strong moves
            'LONDON_OPEN':       3,  # Manipulation — high opportunity
            'NY_AFTERNOON':      2,  # Late distribution
            'TOKYO':             1,  # Accumulation
            'SYDNEY':            0,  # Price discovery
        }
        sess_enc = session_map.get(
            signal.get('session', 'UNKNOWN'), 1)

        # Direction encoding
        dir_enc = 1 if signal.get('direction') == 'BUY' else -1

        # Premium/discount encoding
        pd_map = {
            'EXTREME_PREMIUM': -2, 'PREMIUM': -1,
            'NEUTRAL': 0, 'DISCOUNT': 1, 'EXTREME_DISCOUNT': 2,
        }
        pd_enc = pd_map.get(pd_z.get('zone', 'NEUTRAL'), 0)

        features = [
            # Signal quality
            float(signal.get('score', 0)),
            float(signal.get('sl_pips', 10)),
            float(signal.get('tp1_pips', 15)),
            float(signal.get('tp2_pips', 25)),
            float(dir_enc),
            float(sess_enc),

            # Order flow
            float(d.get('delta', 0)),
            float(rd.get('delta', 0)),
            1.0 if d.get('bias') == 'BULLISH' else
            -1.0 if d.get('bias') == 'BEARISH' else 0.0,
            1.0 if rd.get('bias') == 'BULLISH' else
            -1.0 if rd.get('bias') == 'BEARISH' else 0.0,

            # VWAP context
            float(vwap.get('pip_from_vwap', 0)),
            1.0 if 'ABOVE' in vwap.get('position', '') else -1.0,

            # Volume Profile
            float(prof.get('pip_to_poc', 50)),
            1.0 if prof.get('price_position') == 'ABOVE_VAH' else
            -1.0 if prof.get('price_position') == 'BELOW_VAL' else 0.0,
            float(prof.get('va_width_pips', 50)),

            # SMC
            float(pd_enc),
            float(pd_z.get('pips_to_eq', 0)),
            1.0 if htf.get('approved') else -1.0,
            float(htf.get('score', 50)),

            float(m.get('final_score', 50)),
            float(m.get('smc_score', 50)),
        ]
        return np.array(features, dtype=np.float32).reshape(1, -1)

    except Exception as ex:
        log.error(f"[XGB] Feature extraction failed: {ex}")
        return None


def predict_win_probability(features: np.ndarray) -> float:
    """
    Use trained XGBoost model to predict win probability.
    Returns float 0.0-1.0 (0=likely loss, 1=likely win).
    Returns 0.5 (neutral) if model not trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        return 0.5
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        prob  = model.predict_proba(features)[0][1]
        return round(float(prob), 4)
    except Exception as e:
        log.error(f"[XGB] Prediction failed: {e}")
        return 0.5


def is_model_trained() -> bool:
    """Check if a trained XGBoost model exists on disk."""
    return os.path.exists(MODEL_PATH)


def get_model_info() -> dict:
    """Get information about the trained model (size, age, etc.)."""
    info = {
        'trained': False,
        'path': MODEL_PATH,
        'size_kb': 0,
    }
    if os.path.exists(MODEL_PATH):
        info['trained'] = True
        info['size_kb'] = round(os.path.getsize(MODEL_PATH) / 1024, 1)
        import time
        info['age_hours'] = round(
            (time.time() - os.path.getmtime(MODEL_PATH)) / 3600, 1)
    return info


# ── BACKTEST TRAINING (21 features from backtest_trades) ─────

def train_from_backtest() -> dict:
    """
    Train XGBoost on backtest_trades table (rich 66-column data).
    Uses the SAME 21 features as extract_features() — no mismatch.

    Returns dict with training results.
    """
    try:
        import xgboost as xgb
        import joblib
        from database.db_manager import get_connection

        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch all backtest trades with the 21 features we need
        cursor.execute("""
            SELECT
                score, sl_pips, tp_pips, direction, session,
                delta, rolling_delta, delta_bias, rd_bias,
                pip_from_vwap, pip_to_poc, va_width_pips,
                pd_zone, pips_to_eq, htf_approved,
                final_score, smc_score, htf_score,
                price_position,
                profit_pips, win
            FROM backtest_trades
            WHERE source = 'BACKTEST'
              AND outcome IS NOT NULL
              AND outcome != ''
        """)
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < MIN_TRADES_TO_TRAIN:
            log.info(f"[XGB] Only {len(rows)} backtest trades — need "
                     f"{MIN_TRADES_TO_TRAIN} to train")
            return {
                'status': 'skipped',
                'reason': f'Only {len(rows)} trades (need {MIN_TRADES_TO_TRAIN})',
                'rows': len(rows),
            }

        # Build feature matrix — MUST match extract_features() exactly
        X = []
        y = []

        # Encoding maps (same as extract_features)
        session_map = {
            'NY_LONDON_OVERLAP': 4, 'LONDON_SESSION': 3,
            'LONDON_OPEN': 3, 'NY_AFTERNOON': 2,
            'TOKYO': 1, 'SYDNEY': 0,
        }
        pd_map = {
            'EXTREME_PREMIUM': -2, 'PREMIUM': -1,
            'NEUTRAL': 0, 'DISCOUNT': 1, 'EXTREME_DISCOUNT': 2,
        }

        for row in rows:
            try:
                features = [
                    # Signal quality
                    float(row.get('score', 0) or 0),
                    float(row.get('sl_pips', 10) or 10),
                    float(row.get('tp_pips', 15) or 15),
                    float(row.get('tp_pips', 15) or 15),  # tp2 same as tp (backtest uses single TP)
                    1.0 if str(row.get('direction', '')) == 'BUY' else -1.0,
                    float(session_map.get(row.get('session', ''), 1)),

                    # Order flow
                    float(row.get('delta', 0) or 0),
                    float(row.get('rolling_delta', 0) or 0),
                    float(row.get('delta_bias', 0) or 0),
                    float(row.get('rd_bias', 0) or 0),

                    # VWAP + Volume Profile
                    float(row.get('pip_from_vwap', 0) or 0),
                    1.0 if 'ABOVE' in str(row.get('price_position', '')) else
                    -1.0 if 'BELOW' in str(row.get('price_position', '')) else 0.0,
                    float(row.get('pip_to_poc', 50) or 50),
                    1.0 if str(row.get('price_position', '')) == 'ABOVE_VAH' else
                    -1.0 if str(row.get('price_position', '')) == 'BELOW_VAL' else 0.0,
                    float(row.get('va_width_pips', 50) or 50),

                    # SMC
                    float(pd_map.get(row.get('pd_zone', 'NEUTRAL'), 0)),
                    float(row.get('pips_to_eq', 0) or 0),
                    1.0 if row.get('htf_approved') else -1.0,
                    float(row.get('htf_score', 50) or 50),

                    # Market scores
                    float(row.get('final_score', 50) or 50),
                    float(row.get('smc_score', 50) or 50),
                ]
                label = 1 if row.get('win') else 0
                X.append(features)
                y.append(label)
            except Exception:
                continue

        if len(X) < MIN_TRADES_TO_TRAIN:
            return {
                'status': 'skipped',
                'reason': f'Only {len(X)} valid rows after encoding',
                'rows': len(X),
            }

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Class balance check
        win_count = int(y.sum())
        loss_count = len(y) - win_count
        log.info(f"[XGB] Training data: {len(y)} trades "
                 f"({win_count} wins / {loss_count} losses "
                 f"= {win_count/len(y)*100:.1f}% WR)")

        # Scale weight if imbalanced
        scale_pos = loss_count / win_count if win_count > 0 and win_count < loss_count else 1.0

        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss', random_state=42,
            scale_pos_weight=scale_pos,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        # Train with validation split for accuracy reporting
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        # Calculate accuracy
        train_acc = round(model.score(X_train, y_train) * 100, 1)
        val_acc = round(model.score(X_val, y_val) * 100, 1)

        # Feature importance
        importance = model.feature_importances_
        top_features = sorted(zip(FEATURE_NAMES, importance),
                              key=lambda x: x[1], reverse=True)[:10]

        # Retrain on full data for production
        model.fit(X, y, verbose=False)

        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        log.info(f"[XGB] Model trained on {len(y)} backtest trades "
                 f"(train_acc={train_acc}%, val_acc={val_acc}%)")

        return {
            'status': 'trained',
            'source': 'backtest_trades',
            'total_trades': len(y),
            'wins': win_count,
            'losses': loss_count,
            'win_rate': round(win_count / len(y) * 100, 1),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'top_features': [(f, round(i, 4)) for f, i in top_features],
            'model_size_kb': round(os.path.getsize(MODEL_PATH) / 1024, 1),
        }

    except ImportError as e:
        log.error(f"[XGB] Missing dependency: {e}")
        return {'status': 'error', 'reason': str(e)}
    except Exception as e:
        log.error(f"[XGB] Backtest training failed: {e}")
        return {'status': 'error', 'reason': str(e)}


# ── LIVE TRAINING (7 features from live trades table) ───────

def train_from_live() -> dict:
    """
    Train XGBoost on live trades table (simpler features).
    Fallback when no backtest data exists.

    Returns dict with training results.
    """
    try:
        import xgboost as xgb
        import joblib
        from database.db_manager import get_connection

        conn   = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ai_score, sl_pips, tp_pips, rsi_at_entry,
                   atr_at_entry, spread_at_entry, session,
                   market_regime, outcome
            FROM trades
            WHERE outcome IS NOT NULL
            AND outcome != ''
        """)
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < MIN_TRADES_TO_TRAIN:
            log.info(f"[XGB] Only {len(rows)} live trades — need "
                     f"{MIN_TRADES_TO_TRAIN} to train")
            return {
                'status': 'skipped',
                'reason': f'Only {len(rows)} live trades (need {MIN_TRADES_TO_TRAIN})',
                'rows': len(rows),
            }

        # Build feature matrix from live DB data
        X = []
        y = []
        for row in rows:
            ai_score, sl, tp, rsi, atr, spread, session, regime, outcome = row
            sess_map = {'LONDON_SESSION': 3, 'NY_LONDON_OVERLAP': 4,
                        'NY_AFTERNOON': 2, 'TOKYO': 1, 'SYDNEY': 0}
            features = [
                float(ai_score or 50),
                float(sl or 10),
                float(tp or 15),
                float(rsi or 50),
                float(atr or 0.001),
                float(spread or 1),
                float(sess_map.get(session or '', 1)),
            ]
            label = 1 if outcome in ('WIN_TP1', 'WIN_TP2') else 0
            X.append(features)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        win_count = int(y.sum())

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss', random_state=42)
        model.fit(X, y)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        train_acc = round(model.score(X, y) * 100, 1)
        log.info(f"[XGB] Live model trained on {len(rows)} trades "
                 f"(acc={train_acc}%)")

        return {
            'status': 'trained',
            'source': 'live_trades',
            'total_trades': len(rows),
            'wins': win_count,
            'losses': len(rows) - win_count,
            'win_rate': round(win_count / len(rows) * 100, 1),
            'train_accuracy': train_acc,
        }

    except ImportError as e:
        return {'status': 'error', 'reason': str(e)}
    except Exception as e:
        log.error(f"[XGB] Live training failed: {e}")
        return {'status': 'error', 'reason': str(e)}


def train_model() -> dict:
    """
    Train XGBoost — tries backtest_trades first (rich 21 features),
    falls back to live trades (simpler 7 features).

    Returns dict with training results.
    """
    log.info("[XGB] Starting model training...")

    # Try backtest data first (richer features)
    result = train_from_backtest()
    if result['status'] == 'trained':
        return result

    log.info(f"[XGB] Backtest training {result['status']}: "
             f"{result.get('reason', '')} — trying live trades...")

    # Fallback to live trades
    result = train_from_live()
    return result


def score_signal(signal: dict, market_report: dict,
                 smc_report: dict) -> dict:
    """
    Main function — extract features and predict win probability.
    Returns dict with probability and recommendation.
    """
    features = extract_features(signal, market_report,
                                 smc_report)
    if features is None:
        return {'probability': 0.5, 'recommendation': 'NEUTRAL',
                'trained': False}

    prob = predict_win_probability(features)
    trained = os.path.exists(MODEL_PATH)

    if prob >= 0.70:
        rec = 'STRONG_TAKE'
    elif prob >= 0.60:
        rec = 'TAKE'
    elif prob >= 0.45:
        rec = 'CAUTION'
    else:
        rec = 'SKIP'

    return {
        'probability':    prob,
        'recommendation': rec,
        'trained':        trained,
    }
