# =============================================================
# ai_engine/model_trainer.py  v2.0
# PURPOSE: Orchestrates training of both XGBoost and LSTM.
#
# v2.0 CHANGES:
#   - train_model() returns detailed results dict
#   - Added train_xgb_from_backtest() direct function
#   - Added get_model_status() for CLI reporting
#   - train_all_models() accepts source parameter
# =============================================================

import os
from datetime import datetime, timezone
from core.logger import get_logger
from ai_engine.xgboost_classifier import (
    train_model as train_xgb,
    train_from_backtest as train_xgb_backtest,
    score_signal,
    is_model_trained,
    get_model_info as xgb_model_info,
    FEATURE_NAMES,
)
from ai_engine.lstm_predictor import (
    train_lstm, predict_direction, align_signal)

log = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def train_all_models(df_candles=None, source: str = 'auto') -> dict:
    """
    Train both XGBoost and LSTM models.

    Args:
        df_candles: Candle DataFrame for LSTM training (optional)
        source: 'backtest' | 'live' | 'auto'
            - backtest: train XGBoost from backtest_trades only
            - live: train XGBoost from live trades only
            - auto: try backtest first, fallback to live

    Returns dict with training results for each model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    log.info("[TRAINER] Starting model training...")

    # Train XGBoost
    log.info("[TRAINER] Training XGBoost...")
    if source == 'backtest':
        xgb_result = train_xgb_backtest()
    elif source == 'live':
        from ai_engine.xgboost_classifier import train_from_live
        xgb_result = train_from_live()
    else:  # auto
        xgb_result = train_xgb()

    results['xgboost'] = xgb_result

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


def train_xgboost(source: str = 'backtest') -> dict:
    """
    Train only XGBoost model. Convenience function for CLI.
    Returns detailed training results.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if source == 'backtest':
        return train_xgb_backtest()
    elif source == 'live':
        from ai_engine.xgboost_classifier import train_from_live
        return train_from_live()
    else:
        return train_xgb()


def get_model_status() -> dict:
    """
    Get status of all trained models. For CLI reporting.
    """
    xgb_info = xgb_model_info()

    lstm_path = os.path.join(MODELS_DIR, 'lstm_model.keras')
    lstm_trained = os.path.exists(lstm_path)
    lstm_info = {
        'trained': lstm_trained,
        'path': lstm_path,
        'size_kb': round(os.path.getsize(lstm_path) / 1024, 1) if lstm_trained else 0,
    }
    if lstm_trained:
        import time
        lstm_info['age_hours'] = round(
            (time.time() - os.path.getmtime(lstm_path)) / 3600, 1)

    return {
        'xgboost': xgb_info,
        'lstm': lstm_info,
        'models_dir': MODELS_DIR,
    }


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
