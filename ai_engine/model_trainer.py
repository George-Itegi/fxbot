# =============================================================
# ai_engine/model_trainer.py
# PURPOSE: Orchestrates training of both XGBoost and LSTM.
# Called automatically after every 50 new trades.
# Also provides the combined AI score for any signal.
# =============================================================

import os
from datetime import datetime, timezone
from core.logger import get_logger
from ai_engine.xgboost_classifier import (
    train_model as train_xgb, score_signal)
from ai_engine.lstm_predictor import (
    train_lstm, predict_direction, align_signal)

log = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def train_all_models(df_candles=None) -> dict:
    """
    Train both XGBoost and LSTM models.
    Call after every 50 new completed trades.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    log.info("[TRAINER] Starting model training...")

    # Train XGBoost from database
    log.info("[TRAINER] Training XGBoost...")
    xgb_ok = train_xgb()
    results['xgboost'] = 'trained' if xgb_ok else 'skipped'

    # Train LSTM from candle data
    if df_candles is not None and len(df_candles) > 200:
        log.info("[TRAINER] Training LSTM...")
        lstm_ok = train_lstm(df_candles)
        results['lstm'] = 'trained' if lstm_ok else 'skipped'
    else:
        results['lstm'] = 'no_data'

    results['timestamp'] = datetime.now(timezone.utc).isoformat()
    log.info(f"[TRAINER] Complete: {results}")
    return results


def get_ai_score(signal: dict,
                 market_report: dict,
                 smc_report: dict,
                 df_candles=None) -> dict:
    """
    Get combined AI score for a trade signal.
    Combines XGBoost win probability + LSTM direction alignment.

    Returns:
        ai_score       : 0-100 combined score
        xgb_probability: XGBoost win probability
        lstm_direction : LSTM predicted direction
        lstm_aligned   : Whether LSTM agrees
        recommendation : STRONG_TAKE / TAKE / CAUTION / SKIP
        trained        : Whether models are trained
    """
    # XGBoost score
    xgb = score_signal(signal, market_report,
                        smc_report)
    xgb_prob = xgb['probability']
    xgb_trained = xgb['trained']

    # LSTM score
    lstm_result = {'direction': 'NEUTRAL',
                   'confidence': 0.5, 'trained': False}
    lstm_align  = {'aligned': None, 'boost': 0,
                   'note': 'LSTM not available'}

    if df_candles is not None:
        lstm_result = predict_direction(df_candles)
        lstm_align  = align_signal(signal, lstm_result)

    # Combined score
    # Base: XGBoost probability * 100
    # Adjust: LSTM alignment boost/penalty
    base_score  = xgb_prob * 100
    lstm_boost  = lstm_align.get('boost', 0)
    ai_score    = max(0, min(100, round(base_score + lstm_boost)))

    # Final recommendation
    if not xgb_trained:
        recommendation = 'NEUTRAL'
        note = 'Models not trained — using rule-based score only'
    elif ai_score >= 70:
        recommendation = 'STRONG_TAKE'
        note = f'AI strongly recommends: {ai_score}/100'
    elif ai_score >= 60:
        recommendation = 'TAKE'
        note = f'AI recommends: {ai_score}/100'
    elif ai_score >= 45:
        recommendation = 'CAUTION'
        note = f'AI uncertain: {ai_score}/100'
    else:
        recommendation = 'SKIP'
        note = f'AI recommends skip: {ai_score}/100'

    return {
        'ai_score':         ai_score,
        'xgb_probability':  xgb_prob,
        'xgb_trained':      xgb_trained,
        'lstm_direction':   lstm_result.get('direction', 'NEUTRAL'),
        'lstm_confidence':  lstm_result.get('confidence', 0.5),
        'lstm_trained':     lstm_result.get('trained', False),
        'lstm_aligned':     lstm_align.get('aligned'),
        'lstm_note':        lstm_align.get('note', ''),
        'recommendation':   recommendation,
        'note':             note,
    }
