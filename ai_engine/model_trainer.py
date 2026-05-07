# =============================================================
# ai_engine/model_trainer.py  v3.0
# PURPOSE: Orchestrates training of ML gate model.
#
# v3.0 CHANGES:
#   - Now wraps ml_gate.py (Strategy-Informed ML v3.0)
#   - Backward compatible with old xgboost_classifier.py
#   - train_model() delegates to ml_gate.train_model()
#   - get_model_status() reports both old and new model info
#   - get_ai_score() uses ml_gate.score_signal()
# =============================================================

import os
from datetime import datetime, timezone
from core.logger import get_logger

log = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def train_all_models(df_candles=None, source: str = 'auto',
                     incremental: bool = False,
                     use_replay: bool = False) -> dict:
    """
    Train the ML gate model (primary) + legacy XGBoost (fallback).

    Args:
        df_candles: Unused (kept for backward compatibility)
        source: 'backtest' | 'live' | 'auto'
        incremental: If True, uses hybrid incremental training for ML Gate
        use_replay: If True, uses Experience Replay Buffer
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    mode_parts = []
    if incremental:
        mode_parts.append('INCREMENTAL')
    if use_replay:
        mode_parts.append('REPLAY')
    mode_str = ' + '.join(mode_parts) if mode_parts else 'FROM SCRATCH'
    log.info(f"[TRAINER] Starting ML Gate v3.0 training... mode={mode_str}")

    # Train ML Gate v3.0 (Strategy-Informed)
    try:
        from ai_engine.ml_gate import train_model as train_ml_gate
        ml_result = train_ml_gate(source=source, incremental=incremental,
                                  use_replay=use_replay)
        results['ml_gate'] = ml_result
    except Exception as e:
        results['ml_gate'] = {'status': 'error', 'reason': str(e)}

    # Also train legacy XGBoost (for backward compatibility)
    # Legacy model does not support incremental — always from scratch
    try:
        from ai_engine.xgboost_classifier import train_model as train_xgb
        xgb_result = train_xgb()
        results['xgboost_legacy'] = xgb_result
    except Exception as e:
        results['xgboost_legacy'] = {'status': 'error', 'reason': str(e)}

    results['timestamp'] = datetime.now(timezone.utc).isoformat()
    results['training_mode'] = mode_str
    log.info(f"[TRAINER] Complete: {results}")
    return results


def train_xgboost(source: str = 'backtest') -> dict:
    """
    Train ML Gate model. Convenience function for CLI.
    Delegates to ml_gate.train_model().
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        from ai_engine.ml_gate import train_model
        return train_model(source=source)
    except Exception as e:
        return {'status': 'error', 'reason': str(e)}


def get_model_status() -> dict:
    """Get status of all trained models."""
    # ML Gate v3.0
    try:
        from ai_engine.ml_gate import get_model_info
        ml_gate_info = get_model_info()
    except Exception:
        ml_gate_info = {'trained': False}

    # Legacy XGBoost
    try:
        from ai_engine.xgboost_classifier import get_model_info as xgb_info
        xgb_info = xgb_info()
    except Exception:
        xgb_info = {'trained': False}

    return {
        'ml_gate_v3': ml_gate_info,
        'xgboost_legacy': xgb_info,
        'models_dir': MODELS_DIR,
    }


def get_ai_score(signal: dict,
                 market_report: dict,
                 smc_report: dict,
                 df_candles=None,
                 flow_data: dict = None,
                 all_strategy_scores: dict = None,
                 symbol: str = '') -> dict:
    """
    Get ML score for a trade signal.
    Uses ml_gate v3.0 if available, falls back to legacy XGBoost.
    """
    # Try ML Gate v3.0 first
    try:
        from ai_engine.ml_gate import is_model_trained, score_signal
        if is_model_trained():
            result = score_signal(
                signal, {'market_report': market_report, 'smc_report': smc_report},
                market_report, smc_report,
                flow_data or {},
                all_strategy_scores=all_strategy_scores,
                symbol=symbol,
            )
            if result.get('trained'):
                result['model_version'] = 'ml_gate_v3r'
                return result
    except Exception:
        pass

    # Fallback to legacy XGBoost
    try:
        from ai_engine.xgboost_classifier import score_signal
        xgb = score_signal(signal, market_report, smc_report)
        xgb['model_version'] = 'xgboost_legacy'
        return xgb
    except Exception:
        return {
            'ai_score': 50,
            'predicted_r': 0.0,
            'recommendation': 'NEUTRAL',
            'trained': False,
            'model_version': 'none',
        }
