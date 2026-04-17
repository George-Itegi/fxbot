# =============================================================
# ai_engine/lstm_predictor.py
# PURPOSE: Predict price direction using LSTM deep learning.
# Learns temporal patterns from sequences of candles.
# Outputs: UP / DOWN / NEUTRAL + confidence score.
# Reliable after 500+ trades of retraining.
# =============================================================

import numpy as np
import os
from core.logger import get_logger

log = get_logger(__name__)

MODEL_PATH   = os.path.join(os.path.dirname(__file__),
                            'models', 'lstm_model.keras')
SCALER_PATH  = os.path.join(os.path.dirname(__file__),
                            'models', 'lstm_scaler.pkl')
SEQUENCE_LEN = 50    # Look at last 50 candles
MIN_TRADES   = 100   # Minimum before training


def prepare_sequence(df_candles) -> np.ndarray | None:
    """
    Prepare a 50-candle sequence for LSTM prediction.
    Features per candle: close, rsi, macd_hist, atr,
                         stoch_rsi_k, supertrend_dir, volume_ratio
    """
    try:
        import pandas as pd
        required = ['close', 'rsi', 'macd_hist', 'atr',
                    'stoch_rsi_k', 'supertrend_dir', 'tick_volume',
                    'vol_ma20']
        for col in required:
            if col not in df_candles.columns:
                return None

        df = df_candles.tail(SEQUENCE_LEN).copy()
        if len(df) < SEQUENCE_LEN:
            return None

        # Normalize each feature 0-1
        seq = []
        for _, row in df.iterrows():
            vol_ratio = (float(row['tick_volume']) /
                         float(row['vol_ma20'])
                         if float(row['vol_ma20']) > 0 else 1.0)
            seq.append([
                float(row['close']),
                float(row['rsi']) / 100.0,
                float(row['macd_hist']),
                float(row['atr']),
                float(row.get('stoch_rsi_k', 50)) / 100.0,
                float(row.get('supertrend_dir', 0)),
                min(vol_ratio, 3.0) / 3.0,
            ])

        arr = np.array(seq, dtype=np.float32)

        # Scale close prices to returns
        closes = arr[:, 0]
        returns = np.diff(closes) / (closes[:-1] + 1e-8)
        arr[1:, 0] = returns
        arr[0,  0] = 0.0

        return arr.reshape(1, SEQUENCE_LEN, arr.shape[1])

    except Exception as e:
        log.error(f"[LSTM] Sequence prep failed: {e}")
        return None


def predict_direction(df_candles) -> dict:
    """
    Predict price direction for next 5-15 candles.
    Returns direction (UP/DOWN/NEUTRAL) and confidence.
    Returns NEUTRAL if model not trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        log.info("[LSTM] Model not trained yet — returning NEUTRAL")
        return {'direction': 'NEUTRAL', 'confidence': 0.5,
                'trained': False}

    sequence = prepare_sequence(df_candles)
    if sequence is None:
        return {'direction': 'NEUTRAL', 'confidence': 0.5,
                'trained': False}

    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model(MODEL_PATH)
        output = model.predict(sequence, verbose=0)[0]

        # output = [prob_down, prob_neutral, prob_up]
        prob_up   = float(output[2])
        prob_down = float(output[0])
        prob_neu  = float(output[1])

        if prob_up > prob_down and prob_up > 0.45:
            direction  = 'UP'
            confidence = prob_up
        elif prob_down > prob_up and prob_down > 0.45:
            direction  = 'DOWN'
            confidence = prob_down
        else:
            direction  = 'NEUTRAL'
            confidence = prob_neu

        return {
            'direction':  direction,
            'confidence': round(confidence, 4),
            'prob_up':    round(prob_up, 4),
            'prob_down':  round(prob_down, 4),
            'prob_neutral': round(prob_neu, 4),
            'trained':    True,
        }

    except Exception as e:
        log.error(f"[LSTM] Prediction failed: {e}")
        return {'direction': 'NEUTRAL', 'confidence': 0.5,
                'trained': False}

def train_lstm(df_candles) -> bool:
    """
    Train LSTM on historical candle sequences.
    Uses future price movement as labels:
    - UP    if price rose > 0.3% in next 10 candles
    - DOWN  if price fell > 0.3% in next 10 candles
    - NEUTRAL otherwise
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        n      = len(df_candles)
        if n < SEQUENCE_LEN + 100:
            log.info(f"[LSTM] Not enough data: {n} candles")
            return False

        X = []; y = []
        closes = df_candles['close'].values
        forward = 10  # Look 10 candles ahead
        threshold = 0.003  # 0.3% move

        for i in range(SEQUENCE_LEN, n - forward):
            seq = prepare_sequence(
                df_candles.iloc[i-SEQUENCE_LEN:i])
            if seq is None:
                continue

            entry = closes[i]
            future_max = closes[i:i+forward].max()
            future_min = closes[i:i+forward].min()

            up_move   = (future_max - entry) / entry
            down_move = (entry - future_min) / entry

            if up_move > threshold and up_move > down_move:
                label = 2   # UP
            elif down_move > threshold and down_move > up_move:
                label = 0   # DOWN
            else:
                label = 1   # NEUTRAL

            X.append(seq[0])
            y.append(label)

        if len(X) < MIN_TRADES:
            log.info(f"[LSTM] Only {len(X)} samples — need {MIN_TRADES}")
            return False

        X = np.array(X, dtype=np.float32)
        y = tf.keras.utils.to_categorical(y, num_classes=3)

        # Build LSTM model
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True,
                              input_shape=(SEQUENCE_LEN, 7)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(3, activation='softmax'),
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X, y,
                  epochs=50, batch_size=32,
                  validation_split=0.2,
                  callbacks=[early_stop],
                  verbose=0)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        log.info(f"[LSTM] Model trained on {len(X)} sequences ✅")
        return True

    except Exception as e:
        log.error(f"[LSTM] Training failed: {e}")
        return False


def align_signal(signal: dict, lstm_result: dict) -> dict:
    """
    Check if LSTM prediction aligns with strategy signal.
    Returns alignment info to boost or reduce confidence.
    """
    direction   = signal.get('direction', '')
    lstm_dir    = lstm_result.get('direction', 'NEUTRAL')
    confidence  = lstm_result.get('confidence', 0.5)

    if lstm_dir == 'NEUTRAL' or not lstm_result.get('trained'):
        return {'aligned': None, 'boost': 0,
                'note': 'LSTM not trained yet'}

    aligned = ((direction == 'BUY'  and lstm_dir == 'UP') or
               (direction == 'SELL' and lstm_dir == 'DOWN'))

    if aligned and confidence >= 0.65:
        boost = 10
        note  = f"LSTM confirms {lstm_dir} ({confidence:.0%})"
    elif aligned:
        boost = 5
        note  = f"LSTM weakly confirms {lstm_dir}"
    else:
        boost = -10
        note  = f"LSTM conflicts: predicts {lstm_dir}"

    return {'aligned': aligned, 'boost': boost, 'note': note}
