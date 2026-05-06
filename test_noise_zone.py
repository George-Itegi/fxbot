import numpy as np
import sys
sys.path.insert(0, 'D:/forexbot')
import warnings
warnings.filterwarnings('ignore')

from database.db_manager import get_connection
conn = get_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("""
    SELECT * FROM backtest_trades
    WHERE profit_r IS NOT NULL AND outcome IS NOT NULL AND win IS NOT NULL
    ORDER BY entry_time
""")
rows = cursor.fetchall()
conn.close()

from ai_engine.ml_gate import extract_features_from_db
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
import numpy as np

# Build full feature set (using best approach from previous experiment)
X_all, y_r_all = [], []
for row in rows:
    try:
        gen = extract_features_from_db(row)
        if gen is None: continue
        r = float(row.get('profit_r', 0) or 0)
        X_all.append(gen)
        y_r_all.append(r)
    except: continue

X_all = np.array(X_all, dtype=np.float32)
y_r_all = np.array(y_r_all, dtype=np.float32)
y_win_all = (y_r_all > 0).astype(int)

print(f"Total: {len(y_r_all)} trades")
print(f"  Wins (R>0):    {y_win_all.sum()} ({y_win_all.mean()*100:.1f}%)")
print(f"  Mean R:        {y_r_all.mean():.4f}")
print(f"  Median R:      {np.median(y_r_all):.4f}")

# Analyze noise zone
noise = (y_r_all > -0.2) & (y_r_all < 0.5)
clear_win  = y_r_all >= 0.5
clear_loss = y_r_all <= -0.2
print(f"\nNoise zone (-0.2 to 0.5): {noise.sum()} trades ({noise.mean()*100:.1f}%)")
print(f"Clear wins  (>=0.5R):    {clear_win.sum()} trades ({clear_win.mean()*100:.1f}%)")
print(f"Clear losses (<=-0.2R):  {clear_loss.sum()} trades ({clear_loss.mean()*100:.1f}%)")

def run_regression_cv(X, y_r, y_win, label, n_splits=5):
    """Test regression model: measure quintile quality (Q5 WR and avg R)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds, all_actuals_r, all_actuals_win = [], [], []
    
    for tr, val in kf.split(X):
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.65, min_child_weight=7,
            reg_alpha=0.5, reg_lambda=3.0,
            objective='reg:squarederror', random_state=42, n_jobs=-1)
        m.fit(X[tr], y_r[tr])
        preds = m.predict(X[val])
        all_preds.extend(preds)
        all_actuals_r.extend(y_r[val])
        all_actuals_win.extend(y_win[val])
    
    all_preds = np.array(all_preds)
    all_actuals_r = np.array(all_actuals_r)
    all_actuals_win = np.array(all_actuals_win)
    
    # Quintile analysis — the key metric for a regression model
    # Q5 = top 20% of predictions — these are trades you'd actually execute
    sorted_idx = np.argsort(all_preds)
    n = len(sorted_idx)
    q5_idx = sorted_idx[int(n*0.8):]  # top 20%
    q1_idx = sorted_idx[:int(n*0.2)]  # bottom 20%
    
    q5_wr  = all_actuals_win[q5_idx].mean() * 100
    q5_r   = all_actuals_r[q5_idx].mean()
    q1_wr  = all_actuals_win[q1_idx].mean() * 100
    q1_r   = all_actuals_r[q1_idx].mean()
    
    # TAKE threshold analysis (R >= 0.5)
    take_idx = all_preds >= 0.5
    take_wr  = all_actuals_win[take_idx].mean() * 100 if take_idx.sum() > 0 else 0
    take_r   = all_actuals_r[take_idx].mean() if take_idx.sum() > 0 else 0
    take_n   = take_idx.sum()
    
    mae = mean_absolute_error(all_actuals_r, all_preds)
    
    print(f"\n  [{label}] n_train={len(X)} features={X.shape[1]}")
    print(f"    MAE: {mae:.4f}")
    print(f"    Q5 (top 20%): WR={q5_wr:.1f}%  avg_R={q5_r:.3f}  n={len(q5_idx)}")
    print(f"    Q1 (bot 20%): WR={q1_wr:.1f}%  avg_R={q1_r:.3f}  n={len(q1_idx)}")
    print(f"    TAKE (R>=0.5): WR={take_wr:.1f}%  avg_R={take_r:.3f}  n={take_n}")
    
    return {'mae': mae, 'q5_wr': q5_wr, 'q5_r': q5_r, 'take_wr': take_wr, 'take_r': take_r, 'take_n': int(take_n)}

print("\n=== REGRESSION EXPERIMENT ===\n")

# A: All trades (current approach)
rA = run_regression_cv(X_all, y_r_all, y_win_all, "A: All trades (current)")

# B: Exclude noise zone (Fix 1 from other AI)
mask_B = clear_win | clear_loss
X_B = X_all[mask_B]
y_r_B = y_r_all[mask_B]
y_w_B = y_win_all[mask_B]
rB = run_regression_cv(X_B, y_r_B, y_w_B, "B: Exclude noise (-0.2 to 0.5R)")

# C: Sample weighting (Fix 2 from other AI)
weights_C = np.where(y_r_all >= 0.5, 2.0, np.where(y_r_all <= -0.5, 1.5, 0.3))
# For weighted CV we need custom loop
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_preds_C, all_actuals_r_C, all_actuals_win_C = [], [], []
for tr, val in kf.split(X_all):
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.65, min_child_weight=7,
        reg_alpha=0.5, reg_lambda=3.0, objective='reg:squarederror', random_state=42)
    m.fit(X_all[tr], y_r_all[tr], sample_weight=weights_C[tr])
    all_preds_C.extend(m.predict(X_all[val]))
    all_actuals_r_C.extend(y_r_all[val])
    all_actuals_win_C.extend(y_win_all[val])
all_preds_C = np.array(all_preds_C)
all_actuals_r_C = np.array(all_actuals_r_C)
all_actuals_win_C = np.array(all_actuals_win_C)
sorted_idx = np.argsort(all_preds_C)
n = len(sorted_idx)
q5 = sorted_idx[int(n*0.8):]
take_idx = all_preds_C >= 0.5
rC = {
    'mae': mean_absolute_error(all_actuals_r_C, all_preds_C),
    'q5_wr': all_actuals_win_C[q5].mean()*100,
    'q5_r': all_actuals_r_C[q5].mean(),
    'take_wr': all_actuals_win_C[take_idx].mean()*100 if take_idx.sum()>0 else 0,
    'take_r': all_actuals_r_C[take_idx].mean() if take_idx.sum()>0 else 0,
    'take_n': int(take_idx.sum())
}
print(f"\n  [C: Sample weighting (Fix 2)] n_train={len(X_all)}")
print(f"    MAE: {rC['mae']:.4f}")
print(f"    Q5 (top 20%): WR={rC['q5_wr']:.1f}%  avg_R={rC['q5_r']:.3f}")
print(f"    TAKE (R>=0.5): WR={rC['take_wr']:.1f}%  avg_R={rC['take_r']:.3f}  n={rC['take_n']}")

print(f"""
=== FINAL VERDICT ===
                            Q5 WR    Q5 avg_R    TAKE WR   TAKE avg_R  TAKE count  MAE
A (current, all trades):  {rA['q5_wr']:5.1f}%    {rA['q5_r']:+.3f}      {rA['take_wr']:5.1f}%    {rA['take_r']:+.3f}      {rA['take_n']:4d}       {rA['mae']:.4f}
B (exclude noise zone):   {rB['q5_wr']:5.1f}%    {rB['q5_r']:+.3f}      {rB['take_wr']:5.1f}%    {rB['take_r']:+.3f}      {rB['take_n']:4d}       {rB['mae']:.4f}
C (sample weighting):     {rC['q5_wr']:5.1f}%    {rC['q5_r']:+.3f}      {rC['take_wr']:5.1f}%    {rC['take_r']:+.3f}      {rC['take_n']:4d}       {rC['mae']:.4f}
""")
print("DONE")
