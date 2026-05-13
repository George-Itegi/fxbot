"""
train_model.py — Train XGBoost model for GoldScalper and export to ONNX
==========================================================================

PREREQUISITES:
    pip install xgboost pandas numpy scikit-learn onnxmltools onnxconverter-common

USAGE:
    1. Run ExportTickData.mq5 in MT5 to generate gold_features.csv
    2. Copy gold_features.csv from MQL5/Files/ to this directory
    3. Run: python train_model.py
    4. Copy GoldScalper.onnx from this directory to MQL5/Files/
    5. In GoldScalper config: INP_USE_ML=true
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

try:
    import onnxmltools
    from onnxconverter_common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: onnxmltools not installed. ONNX export will be skipped.")
    print("Install with: pip install onnxmltools onnxconverter-common")


# ─── Configuration ───
INPUT_FILE = "gold_features.csv"
OUTPUT_MODEL = "GoldScalper.onnx"
OUTPUT_XGBOOST = "GoldScalper.json"

# Feature columns (must match FeatureBuilder.mqh order)
FEATURE_COLUMNS = [
    "hurst_approx", "adx", "adx_slope", "atr_ratio", "volume_zscore", "price_position",
    "mom_5", "mom_20", "rsi_14", "price_vs_vwap", "bb_position", "range_5", "range_20",
    "wick_ratio", "consec_dir", "bar_body_ratio",
    "session_id", "is_overlap", "is_friday", "is_monday",
    "spread", "spread_avg", "spread_ratio",
    "vol_ratio", "tick_intensity", "buy_pressure",
    "regime_x_vol", "momentum_align", "time_sin",
]
FEATURE_COUNT = len(FEATURE_COLUMNS)

# Training parameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
}

WALK_FORWARD_SPLITS = 5
MIN_SAMPLES = 1000


def load_data():
    """Load and preprocess the feature CSV."""
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found!")
        print(f"  1. Run ExportTickData.mq5 in MT5
2. Copy gold_features.csv from MQL5/Files/ to this directory
3. Run: python train_model.py
4. Copy GoldScalper.onnx to MQL5/Files/
5. Enable ML in GoldScalper: INP_USE_ML=true

TEST_SIZE = 0.2         # Last 20% for final test
MIN_SAMPLES = 500       # Minimum data rows needed


def load_data(filepath):
    """Load and validate the CSV data from MT5 export."""
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found!")
        print("Run ExportTickData.mq5 in MT5 first, then copy the CSV here.")
        return None, None
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    if len(df) < MIN_SAMPLES:
        print(f"ERROR: Need at least {MIN_SAMPLES} rows, got {len(df)}")
        print("Export more bars in MT5 (increase INP_EXPORT_BARS)")
        return None, None
    
    return df


def prepare_features_labels(df):
    """Extract features and create binary labels."""
    # Check feature columns exist
    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing:
        print(f"WARNING: Missing feature columns: {missing}")
        available = [f for f in FEATURE_COLUMNS if f in df.columns]
    else:
        available = FEATURE_COLUMNS
    
    X = df[available].values.astype(np.float32)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Label: 1 = price went up, 0 = price went down or neutral
    y = (df['label'] > 0).astype(int).values
    
    # Class balance
    pos = y.sum()
    neg = len(y) - pos
    print(f"Labels: {pos} up ({pos/len(y)*100:.1f}%), {neg} down ({neg/len(y)*100:.1f}%)")
    
    return X, y, available


def train_model(X, y):
    """Train XGBoost with walk-forward validation."""
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_model = None
    best_score = 0
    scores = []
    
    print("\n─── Walk-Forward Validation ───")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Calculate scale_pos_weight for class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale = neg_count / max(pos_count, 1)
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            scale_pos_weight=scale,
            eval_metric='logloss',
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        scores.append(acc)
        
        print(f"  Fold {fold+1}: accuracy={acc:.4f} "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        
        if acc > best_score:
            best_score = acc
            best_model = model
    
    avg_score = np.mean(scores)
    print(f"\n  Average accuracy: {avg_score:.4f}")
    print(f"  Best fold: {best_score:.4f}")
    
    return best_model, avg_score


def final_test(model, X, y):
    """Evaluate on the last 20% of data (unseen)."""
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n─── Final Test (last {TEST_SIZE*100:.0f}% of data) ───")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    # Feature importance
    print("─── Top 10 Features ───")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {FEATURE_COLUMNS[idx]}: {importances[idx]:.4f}")
    
    return acc


def export_onnx(model, feature_count, output_path="GoldScalper.onnx"):
    """Export XGBoost model to ONNX format for MQ5."""
    if not ONNX_AVAILABLE:
        print("\nONNX export skipped (onnxmltools not installed)")
        print("Install: pip install onnxmltools onnxconverter-common")
        return False
    
    try:
        initial_type = [('float_input', FloatTensorType([None, feature_count]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types)
        
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        file_size = os.path.getsize(output_path) / 1024
        print(f"\n✅ ONNX model saved: {output_path} ({file_size:.1f} KB)")
        print(f"   Features: {feature_count}")
        print(f"   Copy to: <MT5_Data>/MQL5/Files/{os.path.basename(output_path)}")
        return True
    
    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        return False


def main():
    print("═══════════════════════════════════════════")
    print("  GoldScalper ML Model Training Pipeline")
    print("═══════════════════════════════════════════")
    
    # 1. Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    # 2. Prepare features and labels
    X, y, feature_names = prepare_features_labels(df)
    
    # 3. Train with walk-forward validation
    model, avg_score = train_model(X, y)
    
    if model is None:
        print("Training failed!")
        return
    
    # 4. Final test on unseen data
    test_acc = final_test(model, X, y)
    
    # 5. Export to ONNX
    export_onnx(model, len(feature_names), "GoldScalper.onnx")
    
    # 6. Summary
    print("\n═══════════════════════════════════════════")
    print("  TRAINING COMPLETE")
    print(f"  Walk-forward accuracy: {avg_score:.4f}")
    print(f"  Final test accuracy:   {test_acc:.4f}")
    print(f"  Features used:         {len(feature_names)}")
    print("")
    print("  NEXT STEPS:")
    print("  1. Copy GoldScalper.onnx to MT5's MQL5/Files/ directory")
    print("  2. In GoldScalper config: INP_USE_ML=true")
    print("  3. INP_FEATURE_COUNT = " + str(len(feature_names)))
    print("  4. Re-attach EA to chart")
    print("═══════════════════════════════════════════")


if __name__ == "__main__":
    main()
