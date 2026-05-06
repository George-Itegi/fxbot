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
    WHERE profit_r IS NOT NULL AND outcome IS NOT NULL AND win IS NOT NULL
    ORDER BY entry_time
""")
rows = cursor.fetchall()
conn.close()
print(f"Total: {len(rows)} trades  WR={sum(1 for r in rows if r.get('win'))/len(rows)*100:.1f}%")

from ai_engine.ml_gate import extract_features_from_db, FEATURE_NAMES
from ai_engine.ema_cross_strategy_model import build_ema_cross_feature_vector
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# ── Strategy-specific builders (fixed size each) ──────────────
def build_trend_cont(row):
    return [
        float(row.get('htf_score',50) or 50)/100,
        float({'BULLISH':1.,'BEARISH':-1.,'RANGING':0.}.get(str(row.get('structure_trend','RANGING')),0.)),
        float(row.get('confluence_count',0) or 0)/10,
        float(row.get('score',0) or 0)/100,
        min(float(row.get('tp_pips',15) or 15)/max(float(row.get('sl_pips',10) or 10),0.1)/5,1.),
        abs(float(row.get('pip_to_poc',0) or 0))/50,
        1. if row.get('htf_approved') else -1.,
    ]

def build_smc_ob(row):
    pd_map = {'EXTREME_PREMIUM':-2,'PREMIUM':-1,'NEUTRAL':0,'DISCOUNT':1,'EXTREME_DISCOUNT':2}
    return [
        float(pd_map.get(str(row.get('pd_zone','NEUTRAL')),0))/2,
        float(row.get('pips_to_eq',0) or 0)/50,
        float(row.get('htf_score',50) or 50)/100,
        1. if row.get('htf_approved') else -1.,
        float({'BULLISH':1.,'BEARISH':-1.,'NEUTRAL':0.}.get(str(row.get('smc_bias','NEUTRAL')),0.)),
        float(row.get('score',0) or 0)/100,
        float(row.get('confluence_count',0) or 0)/10,
        float({'BULLISH':1.,'BEARISH':-1.,'RANGING':0.}.get(str(row.get('structure_trend','RANGING')),0.)),
    ]

def build_structure_align(row):
    return [
        float({'BULLISH':1.,'BEARISH':-1.,'RANGING':0.}.get(str(row.get('structure_trend','RANGING')),0.)),
        float(row.get('htf_score',50) or 50)/100,
        1. if row.get('htf_approved') else -1.,
        float(row.get('score',0) or 0)/100,
        float(row.get('confluence_count',0) or 0)/10,
        float({'BULLISH':1.,'BEARISH':-1.,'NEUTRAL':0.}.get(str(row.get('delta_bias','NEUTRAL')),0.)),
    ]

def build_breakout(row):
    return [
        float(row.get('atr',0) or 0)/30,
        float(row.get('va_width_pips',20) or 20)/50,
        float(row.get('score',0) or 0)/100,
        1. if row.get('vol_surge_detected') else 0.,
        min(float(row.get('vol_surge_ratio',1) or 1)/5,1.),
    ]

def build_liq_sweep(row):
    return [
        float({'BULLISH':1.,'BEARISH':-1.,'NEUTRAL':0.}.get(str(row.get('smc_bias','NEUTRAL')),0.)),
        float({'BULLISH':1.,'BEARISH':-1.,'RANGING':0.}.get(str(row.get('structure_trend','RANGING')),0.)),
        float({'BULLISH':1.,'BEARISH':-1.,'NEUTRAL':0.}.get(str(row.get('delta_bias','NEUTRAL')),0.)),
        float(row.get('of_imbalance',0) or 0),
        float(row.get('score',0) or 0)/100,
    ]

def build_generic(row):
    return [
        float(row.get('score',0) or 0)/100,
        float(row.get('confluence_count',0) or 0)/10,
        min(float(row.get('tp_pips',15) or 15)/max(float(row.get('sl_pips',10) or 10),0.1)/5,1.),
    ]

BUILDERS = {
    'EMA_CROSS_MOMENTUM':    (build_ema_cross_feature_vector, 23),
    'TREND_CONTINUATION':    (build_trend_cont,   7),
    'SMC_OB_REVERSAL':       (build_smc_ob,        8),
    'STRUCTURE_ALIGNMENT':   (build_structure_align, 6),
    'BREAKOUT_MOMENTUM':     (build_breakout,      5),
    'LIQUIDITY_SWEEP_ENTRY': (build_liq_sweep,     5),
}
MAX_SPEC = 23  # pad all to EMA_CROSS length

STRATEGY_LIST = list(BUILDERS.keys()) + ['VWAP_MEAN_REVERSION','FVG_REVERSION','DELTA_DIVERGENCE','RSI_DIVERGENCE_SMC']

def pad_spec(vec, n=MAX_SPEC):
    arr = np.array(vec, dtype=np.float32)
    if len(arr) >= n: return arr[:n]
    return np.concatenate([arr, np.zeros(n - len(arr), dtype=np.float32)])

def onehot_strategy(strat):
    return np.array([1. if strat == s else 0. for s in STRATEGY_LIST], dtype=np.float32)

# ── Build feature matrices ────────────────────────────────────
X_generic, X_full, y = [], [], []

for row in rows:
    try:
        gen = extract_features_from_db(row)
        if gen is None: continue
        strat = row.get('strategy','')
        builder_fn = BUILDERS.get(strat, (build_generic, 3))[0]
        spec_raw = builder_fn(row)
        spec = pad_spec(spec_raw)
        onehot = onehot_strategy(strat)
        label = 1 if float(row.get('profit_r',0) or 0) > 0 else 0
        X_generic.append(gen)
        X_full.append(np.concatenate([gen, spec, onehot]))
        y.append(label)
    except: continue

y = np.array(y)
print(f"Samples: {len(y)}  WR={y.mean()*100:.1f}%")
print(f"Generic features:  {len(X_generic[0])}")
print(f"Full features:     {len(X_full[0])} (66 generic + 23 padded-spec + {len(STRATEGY_LIST)} onehot)")

def cv_auc(X, y_arr, label, depth=4):
    X = np.array(X, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, val in kf.split(X):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=depth, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.65, min_child_weight=7,
            reg_alpha=0.5, reg_lambda=3.0, use_label_encoder=False,
            eval_metric='logloss', random_state=42, n_jobs=-1)
        m.fit(X[tr], y_arr[tr])
        aucs.append(roc_auc_score(y_arr[val], m.predict_proba(X[val])[:,1]))
    auc, std = np.mean(aucs), np.std(aucs)
    print(f"  {label:50s}  AUC={auc:.4f} +/-{std:.4f}  ({X.shape[1]} features)")
    return auc

print("\n=== RESULTS ===\n")
a_base = cv_auc(X_generic, y, "Current L2: 63 generic only")
a_full  = cv_auc(X_full,    y, "Your idea:  63 + 23-padded-spec + 10-onehot")

# Also test specific-only (no generic) to show why we need both
X_spec_only = [np.concatenate([pad_spec(BUILDERS.get(rows[i]['strategy'],(build_generic,3))[0](rows[i])), onehot_strategy(rows[i]['strategy'])]) for i in range(len(rows)) if extract_features_from_db(rows[i]) is not None]
# rebuild clean
X_spec_clean, y_spec = [], []
for row in rows:
    try:
        gen = extract_features_from_db(row)
        if gen is None: continue
        strat = row.get('strategy','')
        spec = pad_spec(BUILDERS.get(strat,(build_generic,3))[0](row))
        onehot = onehot_strategy(strat)
        X_spec_clean.append(np.concatenate([spec, onehot]))
        y_spec.append(1 if float(row.get('profit_r',0) or 0) > 0 else 0)
    except: continue

a_spec = cv_auc(X_spec_clean, np.array(y_spec), "Specific+onehot only (no generic 63)")

print(f"""
=== FINAL VERDICT ===
  Current L2 (63 generic):         AUC = {a_base:.4f}
  Your idea  (63+spec+onehot):     AUC = {a_full:.4f}  delta = {a_full-a_base:+.4f}
  Spec+onehot only (no generic):   AUC = {a_spec:.4f}  delta = {a_spec-a_base:+.4f}
""")

# Feature importance
print("=== TOP 20 FEATURES (Your idea model) ===\n")
X_full_arr = np.array(X_full, dtype=np.float32)
spec_names = [f'spec_{i:02d}' for i in range(MAX_SPEC)]
oh_names   = [f'is_{s[:15]}' for s in STRATEGY_LIST]
all_names  = FEATURE_NAMES[:66] + spec_names + oh_names

m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.65, min_child_weight=7,
    reg_alpha=0.5, reg_lambda=3.0, use_label_encoder=False,
    eval_metric='logloss', random_state=42)
m.fit(X_full_arr, y)
top = sorted(zip(all_names[:len(m.feature_importances_)], m.feature_importances_), key=lambda x: -x[1])[:20]
for name, score in top:
    print(f"  {name:35s}  {score:.4f}  {'|'*int(score*500)}")

print("\nDONE")
