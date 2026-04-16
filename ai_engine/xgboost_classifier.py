# =============================================================
# ai_engine/xgboost_classifier.py
# PURPOSE: Predict win/loss probability for any trade signal.
# Uses 25 features extracted from market conditions at signal time.
# Trained on historical trade results from database.
# Gets more accurate with more trades — reliable after 200+.
# =============================================================

import numpy as np
import os
from core.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          'models', 'xgb_model.pkl')
MIN_TRADES_TO_TRAIN = 50   # Minimum trades before training


def extract_features(signal: dict,
                     market_report: dict,
                     smc_report: dict) -> np.ndarray | None:
    """
    Extract 21 numerical features from trade context. (External data removed)
    These are the inputs XGBoost learns from.
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

        # Session encoding
        session_map = {
            'LONDON_KILLZONE': 3, 'NY_LONDON_OVERLAP': 4,
            'NY_SESSION': 2, 'LONDON_OPEN': 3,
            'ASIAN_SESSION': 1, 'DEAD_ZONE': 0,
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
        log.info("[XGB] Model not trained yet — returning neutral 0.5")
        return 0.5
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        prob  = model.predict_proba(features)[0][1]
        return round(float(prob), 4)
    except Exception as e:
        log.error(f"[XGB] Prediction failed: {e}")
        return 0.5


def train_model() -> bool:
    """
    Train XGBoost on all completed trades in database.
    Called automatically after every 50 new trades.
    Returns True if training succeeded.
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
            log.info(f"[XGB] Only {len(rows)} trades — need "
                     f"{MIN_TRADES_TO_TRAIN} to train")
            return False

        # Build simple feature matrix from DB data
        X = []
        y = []
        for row in rows:
            ai_score, sl, tp, rsi, atr, spread, session, regime, outcome = row
            sess_map = {'LONDON_KILLZONE': 3, 'NY_LONDON_OVERLAP': 4,
                        'NY_SESSION': 2, 'ASIAN_SESSION': 1}
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

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss', random_state=42)
        model.fit(X, y)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        log.info(f"[XGB] Model trained on {len(rows)} trades ✅")
        return True

    except Exception as e:
        log.error(f"[XGB] Training failed: {e}")
        return False


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
