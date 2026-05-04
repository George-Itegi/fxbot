"""
The REAL test: not AUC of L1 alone, but how much does
l1_predicted_r add to L2?

If L1 uses the same 63 features as L2:
  L2 already knows everything L1 knows → l1_r adds nothing

If L1 uses only strategy-specific features:
  L1 knows things L2 cannot see → l1_r is genuinely new info

We test this directly: train L2 with and without l1_r,
comparing the AUC improvement from adding l1_r
under both Option A and Option B.
"""
import numpy as np
import sys
sys.path.insert(0, 'D:/forexbot')
import warnings
warnings.filterwarnings('ignore')

from database.db_manager import get_connection
from collections import defaultdict

conn = get_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("""
    SELECT * FROM backtest_trades
    WHERE strategy = 'EMA_CROSS_MOMENTUM'
      AND profit_r IS NOT NULL AND outcome IS NOT NULL
    ORDER BY entry_time
""")
rows = cursor.fetchall()
conn.close()

print(f"EMA_CROSS_MOMENTUM: {len(rows)} trades, WR={sum(1 for r in rows if r.get('win'))/len(rows)*100:.1f}%")

from ai_engine.ml_gate import extract_features_from_db
from ai_engine.ema_cross_strategy_model import build_ema_cross_feature_vector
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np

def get_cv_auc(X, y, label, n_splits=5):
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, val in kf.split(X):
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=1.0, reg_lambda=3.0, use_label_encoder=False,
            eval_metric='logloss', random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[val])[:, 1]
        aucs.append(roc_auc_score(y[val], prob))
    auc = np.mean(aucs)
    std = np.std(aucs)
    print(f"  {label:55s} AUC={auc:.4f} +/-{std:.4f} ({X.shape[1]} features)")
    return auc, m, X, y

# Build all feature sets
X_generic, X_specific, X_combined = [], [], []
y_all = []

for row in rows:
    try:
        gen = extract_features_from_db(row)
        spec = np.array(build_ema_cross_feature_vector(row), dtype=np.float32)
        if gen is None: continue
        r = float(row.get('profit_r', 0) or 0)
        X_generic.append(gen)
        X_specific.append(spec)
        X_combined.append(np.concatenate([gen, spec]))
        y_all.append(1 if r > 0 else 0)
    except: continue

y = np.array(y_all)

print("\n=== STEP 1: Train L1 under Option A (generic) and Option B (specific) ===\n")

# Option A: L1 uses generic 63 features
_, m_A, _, _ = get_cv_auc(X_generic, y, "Option A L1 (63 generic features)")

# Option B: L1 uses specific 23 features only
_, m_B, _, _ = get_cv_auc(X_specific, y, "Option B L1 (23 specific features only)")

print("\n=== STEP 2: Generate l1_predicted_r from each approach ===\n")

# Use full dataset to train each L1, predict l1_r for all samples
# (simplified - in real system this would be out-of-fold)
X_gen_arr = np.array(X_generic, dtype=np.float32)
X_spec_arr = np.array(X_specific, dtype=np.float32)
X_comb_arr = np.array(X_combined, dtype=np.float32)

from sklearn.model_selection import cross_val_predict

def get_oof_probs(X, y, label):
    """Get out-of-fold predictions to avoid leakage."""
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, val in kf.split(X):
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=1.0, reg_lambda=3.0, use_label_encoder=False,
            eval_metric='logloss', random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr])
        oof[val] = m.predict_proba(X[val])[:, 1]
    print(f"  {label}: OOF AUC = {roc_auc_score(y, oof):.4f}")
    return oof

print("Generating out-of-fold L1 predictions (avoids data leakage)...")
l1_probs_A = get_oof_probs(X_gen_arr,  y, "Option A L1 (generic 63)")
l1_probs_B = get_oof_probs(X_spec_arr, y, "Option B L1 (specific 23)")

print("\n=== STEP 3: How much does l1_r add to L2? ===\n")
print("L2 base = 63 generic features. We test adding l1_r from each approach.")

# L2 baseline: generic 63 only
auc_base, _, _, _ = get_cv_auc(
    X_gen_arr, y,
    "L2 baseline (63 generic, no l1_r)")

# L2 + Option A l1_r (L1 was trained on same 63 features)
X_l2_A = np.column_stack([X_gen_arr, l1_probs_A])
auc_A, _, _, _ = get_cv_auc(
    X_l2_A, y,
    "L2 + Option A l1_r (L1=generic, CORRELATED with L2)")

# L2 + Option B l1_r (L1 was trained on DIFFERENT 23 specific features)
X_l2_B = np.column_stack([X_gen_arr, l1_probs_B])
auc_B, _, _, _ = get_cv_auc(
    X_l2_B, y,
    "L2 + Option B l1_r (L1=specific, INDEPENDENT of L2)")

# Bonus: L2 + combined L1 (86 features)
X_l2_comb = np.column_stack([X_gen_arr, get_oof_probs(X_comb_arr, y, "Combined L1 (86)")])
auc_comb, _, _, _ = get_cv_auc(
    X_l2_comb, y,
    "L2 + Combined L1 l1_r (L1=86 features)")

print(f"""
=== FINAL VERDICT ===
  L2 baseline (no l1_r):           AUC = {auc_base:.4f}
  L2 + Option A l1_r (correlated): AUC = {auc_A:.4f}  delta = {auc_A-auc_base:+.4f}
  L2 + Option B l1_r (independent):AUC = {auc_B:.4f}  delta = {auc_B-auc_base:+.4f}
  L2 + Combined l1_r:              AUC = {auc_comb:.4f}  delta = {auc_comb-auc_base:+.4f}

  Option A improvement over baseline: {(auc_A-auc_base)*100:+.2f}% AUC points
  Option B improvement over baseline: {(auc_B-auc_base)*100:+.2f}% AUC points
  Combined improvement over baseline: {(auc_comb-auc_base)*100:+.2f}% AUC points
""")
print("DONE")
