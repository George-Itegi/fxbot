import numpy as np
import sys
sys.path.insert(0, 'D:/forexbot')
from database.db_manager import get_connection
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

conn = get_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("""
    SELECT * FROM backtest_trades
    WHERE strategy IN ('EMA_CROSS_MOMENTUM', 'TREND_CONTINUATION')
      AND profit_r IS NOT NULL AND outcome IS NOT NULL
    ORDER BY strategy, entry_time
""")
rows = cursor.fetchall()
conn.close()

by_strategy = defaultdict(list)
for row in rows:
    by_strategy[row['strategy']].append(row)

print(f"EMA_CROSS: {len(by_strategy['EMA_CROSS_MOMENTUM'])} trades")
print(f"TREND_CONT: {len(by_strategy['TREND_CONTINUATION'])} trades")

from ai_engine.ml_gate import extract_features_from_db
from ai_engine.ema_cross_strategy_model import build_ema_cross_feature_vector
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold

def run_cv(X, y_raw, label):
    X = np.array(X, dtype=np.float32)
    y = (np.array(y_raw) > 0).astype(int)
    if len(set(y)) < 2 or len(y) < 30:
        print(f"  {label}: insufficient data")
        return
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
        reg_alpha=1.0, reg_lambda=3.0, use_label_encoder=False,
        eval_metric='logloss', random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"  {label}: n={len(y)} features={X.shape[1]} AUC={scores.mean():.4f} +/-{scores.std():.4f} WR={y.mean()*100:.1f}%")

print("\n=== EMA_CROSS_MOMENTUM ===")
X_combined, X_specific, y = [], [], []
for row in by_strategy['EMA_CROSS_MOMENTUM']:
    try:
        gen = extract_features_from_db(row)
        spec = build_ema_cross_feature_vector(row)
        if gen is None: continue
        r = float(row.get('profit_r', 0) or 0)
        X_combined.append(np.concatenate([gen, np.array(spec, dtype=np.float32)]))
        X_specific.append(np.array(spec, dtype=np.float32))
        y.append(r)
    except: continue

run_cv(X_combined, y, "Combined 63+23=86 features")
run_cv(X_specific, y, "Specific only 23 features ")

print("\n=== TREND_CONTINUATION ===")
X_gen, X_ext, y2 = [], [], []
for row in by_strategy['TREND_CONTINUATION']:
    try:
        gen = extract_features_from_db(row)
        if gen is None: continue
        h4 = float(row.get('htf_score', 50) or 50) / 100
        st = float({'BULLISH':1.0,'BEARISH':-1.0,'RANGING':0.0}.get(str(row.get('structure_trend','RANGING')),0.0))
        cf = float(row.get('confluence_count', 0) or 0) / 10
        sc = float(row.get('score', 0) or 0) / 100
        tp = float(row.get('tp_pips', 15) or 15)
        sl = max(float(row.get('sl_pips', 10) or 10), 0.1)
        rr = min(tp/sl/5, 1.0)
        extra = np.array([h4, st, cf, sc, rr], dtype=np.float32)
        r = float(row.get('profit_r', 0) or 0)
        X_gen.append(gen)
        X_ext.append(np.concatenate([gen, extra]))
        y2.append(r)
    except: continue

run_cv(X_gen, y2, "Generic only 63 features  ")
run_cv(X_ext, y2, "Generic+5 extras 68 feats ")

print("\nDONE")
