"""
Definitive test: L2 with strategy-specific features added PROPERLY
No padding — each row gets the features relevant to ITS strategy,
all others are zero only for features that genuinely don't apply.

Key insight: instead of padding with zeros, we use a FIXED feature
set for L2 that represents ALL strategy-specific concepts, but each
strategy only fills in its relevant slots. The rest are genuine 0s
(meaning "this concept does not apply to this strategy").

This is the correct way — not arbitrary padding, but meaningful zeros.
"""
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
print(f"Total: {len(rows)} trades  WR={sum(1 for r in rows if r.get('win'))/len(rows)*100:.1f}%")

from ai_engine.ml_gate import extract_features_from_db, FEATURE_NAMES
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# ── Unified strategy feature block (30 slots, meaningful zeros) ──
# Every strategy fills in ONLY the slots relevant to it.
# Zeros mean "this concept does not apply here" — not padding.

STRAT_FEATURE_NAMES = [
    # Trend/momentum alignment (5)
    'sf_h4_alignment_score',    # EMA/TREND: H4 trend alignment 0-1
    'sf_h1_alignment_score',    # EMA/TREND/STRUCT: H1 alignment 0-1
    'sf_cross_bars_ago',        # EMA only: bars since cross (0=recent)
    'sf_cross_strength',        # EMA only: separation strength 0-1
    'sf_trend_maturity',        # TREND_CONT: early=1 late=0
    # Structure quality (5)
    'sf_structure_bull',        # +1 BULLISH, -1 BEARISH, 0 RANGING
    'sf_htf_score',             # All: HTF alignment score 0-1
    'sf_htf_approved',          # All: HTF approved +1/-1
    'sf_pd_zone',               # SMC/FVG: premium-discount -1 to +1
    'sf_pips_to_eq',            # SMC/FVG: pips to equilibrium 0-1
    # Order flow quality (5)
    'sf_delta_bias',            # All: +1 bull, -1 bear, 0 neutral
    'sf_of_imbalance',          # All: order flow imbalance raw
    'sf_of_strength',           # All: 0-3 strength level
    'sf_vol_surge',             # All: 1 if volume surge
    'sf_surge_ratio',           # All: surge multiplier 0-1
    # Entry quality (5)
    'sf_score_norm',            # All: strategy score 0-1
    'sf_confluence_norm',       # All: confluence factors 0-1
    'sf_rr_ratio_norm',         # All: R:R ratio 0-1
    'sf_sl_quality',            # All: sl_pips (lower = better) 0-1
    'sf_pullback_depth',        # TREND/EMA: pullback to key level 0-1
    # SMC-specific (5)
    'sf_smc_bias',              # SMC: +1 bull -1 bear 0 neutral
    'sf_ob_quality',            # SMC_OB: order block quality 0-1
    'sf_sweep_aligned',         # SWEEP/SMC: sweep alignment 1/0
    'sf_bos_present',           # SMC/STRUCT: BOS detected 1/0
    'sf_fvg_quality',           # FVG only: FVG quality 0-1
    # Momentum-specific (5)
    'sf_momentum_velocity',     # BREAK/EMA/M1: velocity 0-1
    'sf_atr_quality',           # All: ATR vs va_width ratio 0-1
    'sf_choppy',                # All: 1 if choppy
    'sf_breakout_quality',      # BREAKOUT: clean break 0-1
    'sf_va_width_quality',      # BREAK/VWAP: VA width 0-1
]

N_SF = len(STRAT_FEATURE_NAMES)  # 30

def build_strat_features(row):
    """
    Build unified 30-feature strategy-specific block.
    Each strategy fills in its meaningful slots only.
    Zeros = concept genuinely doesn't apply to this strategy.
    """
    strat = str(row.get('strategy', ''))
    sf = np.zeros(N_SF, dtype=np.float32)

    # Common to ALL strategies
    sf[5]  = float({'BULLISH':1.,'BEARISH':-1.,'RANGING':0.}.get(str(row.get('structure_trend','RANGING')),0.))
    sf[6]  = float(row.get('htf_score', 50) or 50) / 100
    sf[7]  = 1. if row.get('htf_approved') else -1.
    sf[10] = float({'BULLISH':1.,'BEARISH':-1.,'NEUTRAL':0.}.get(str(row.get('delta_bias','NEUTRAL')),0.))
    sf[11] = float(row.get('of_imbalance', 0) or 0)
    sf[12] = float({'EXTREME':3.,'STRONG':2.,'MODERATE':1.,'WEAK':0.,'NONE':0.}.get(str(row.get('of_strength','NONE')),0.))/3
    sf[13] = 1. if row.get('vol_surge_detected') else 0.
    sf[14] = min(float(row.get('vol_surge_ratio', 1) or 1) / 5, 1.)
    sf[15] = float(row.get('score', 0) or 0) / 100
    sf[16] = float(row.get('confluence_count', 0) or 0) / 10
    tp = float(row.get('tp_pips', 15) or 15)
    sl = max(float(row.get('sl_pips', 10) or 10), 0.1)
    sf[17] = min(tp / sl / 5, 1.)
    sf[18] = 1. - min(sl / 30, 1.)  # lower SL = higher quality
    sf[25] = float(row.get('momentum_velocity', 0) or 0) / 5
    atr = float(row.get('atr', 0) or 0)
    va  = float(row.get('va_width_pips', 20) or 20)
    sf[26] = min(atr / max(va, 0.1), 1.) if va > 0 else 0.
    sf[27] = 1. if row.get('is_choppy') else 0.

    # SMC bias for SMC-type strategies
    smc_bias = float({'BULLISH':1.,'BEARISH':-1.,'NEUTRAL':0.}.get(str(row.get('smc_bias','NEUTRAL')),0.))
    pd_zone  = float({'EXTREME_PREMIUM':-2,'PREMIUM':-1,'NEUTRAL':0,'DISCOUNT':1,'EXTREME_DISCOUNT':2}.get(str(row.get('pd_zone','NEUTRAL')),0.)) / 2

    if strat == 'EMA_CROSS_MOMENTUM':
        sf[0]  = float(row.get('htf_score', 50) or 50) / 100  # H4 alignment
        sf[1]  = float(row.get('htf_score', 50) or 50) / 100  # H1 proxy
        sf[2]  = 0.  # cross_bars_ago — not in DB, default fresh
        sf[3]  = min(abs(float(row.get('pip_to_poc', 0) or 0)) / 20, 1.)  # spread proxy
        sf[19] = min(abs(float(row.get('pip_from_vwap', 0) or 0)) / 20, 1.)

    elif strat == 'TREND_CONTINUATION':
        sf[0]  = float(row.get('htf_score', 50) or 50) / 100
        sf[1]  = float(row.get('htf_score', 50) or 50) / 100
        sf[4]  = 1. - min(abs(float(row.get('pip_to_poc', 0) or 0)) / 50, 1.)  # early trend = close to POC
        sf[19] = min(abs(float(row.get('pip_to_poc', 0) or 0)) / 30, 1.)

    elif strat == 'SMC_OB_REVERSAL':
        sf[8]  = pd_zone
        sf[9]  = min(abs(float(row.get('pips_to_eq', 0) or 0)) / 50, 1.)
        sf[20] = smc_bias
        sf[21] = float(row.get('smc_score', 0) or 0) / 100 if 'smc_score' in row else sf[15]
        sf[23] = 1.  # BOS assumed for OB reversal

    elif strat == 'STRUCTURE_ALIGNMENT':
        sf[0]  = float(row.get('htf_score', 50) or 50) / 100
        sf[1]  = float(row.get('htf_score', 50) or 50) / 100
        sf[23] = 1. if row.get('htf_approved') else 0.

    elif strat == 'LIQUIDITY_SWEEP_ENTRY':
        sf[20] = smc_bias
        sf[22] = 1.  # sweep aligned by definition
        sf[23] = 1. if row.get('htf_approved') else 0.

    elif strat == 'BREAKOUT_MOMENTUM':
        sf[14] = min(float(row.get('vol_surge_ratio', 1) or 1) / 5, 1.)
        sf[28] = float(row.get('score', 0) or 0) / 100
        sf[29] = 1. - min(va / 100, 1.)

    elif strat == 'VWAP_MEAN_REVERSION':
        sf[8]  = pd_zone
        sf[19] = min(abs(float(row.get('pip_from_vwap', 0) or 0)) / 30, 1.)
        sf[29] = 1. - min(va / 80, 1.)

    elif strat == 'FVG_REVERSION':
        sf[8]  = pd_zone
        sf[9]  = min(abs(float(row.get('pips_to_eq', 0) or 0)) / 30, 1.)
        sf[20] = smc_bias
        sf[24] = float(row.get('score', 0) or 0) / 100

    return sf

# Strategy one-hot (10 strategies)
STRATEGY_LIST = ['EMA_CROSS_MOMENTUM','TREND_CONTINUATION','SMC_OB_REVERSAL',
    'STRUCTURE_ALIGNMENT','BREAKOUT_MOMENTUM','LIQUIDITY_SWEEP_ENTRY',
    'VWAP_MEAN_REVERSION','FVG_REVERSION','DELTA_DIVERGENCE','RSI_DIVERGENCE_SMC']

def onehot(strat):
    return np.array([1. if strat == s else 0. for s in STRATEGY_LIST], dtype=np.float32)

# Build all feature matrices
X_base, X_sf, X_sf_onehot, y = [], [], [], []

for row in rows:
    try:
        gen = extract_features_from_db(row)
        if gen is None: continue
        sf  = build_strat_features(row)
        oh  = onehot(row.get('strategy',''))
        label = 1 if float(row.get('profit_r', 0) or 0) > 0 else 0
        X_base.append(gen)
        X_sf.append(np.concatenate([gen, sf]))
        X_sf_onehot.append(np.concatenate([gen, sf, oh]))
        y.append(label)
    except: continue

y = np.array(y)
print(f"Samples: {len(y)}  WR={y.mean()*100:.1f}%")
print(f"Base features:        {len(X_base[0])}")
print(f"+ Strategy features:  {len(X_sf[0])} (66 + 30)")
print(f"+ Onehot:             {len(X_sf_onehot[0])} (66 + 30 + 10)")

def cv_auc(X, y_arr, label):
    X = np.array(X, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, val in kf.split(X):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.65, min_child_weight=7,
            reg_alpha=0.5, reg_lambda=3.0, use_label_encoder=False,
            eval_metric='logloss', random_state=42, n_jobs=-1)
        m.fit(X[tr], y_arr[tr])
        aucs.append(roc_auc_score(y_arr[val], m.predict_proba(X[val])[:,1]))
    auc, std = np.mean(aucs), np.std(aucs)
    print(f"  {label:55s}  AUC={auc:.4f} ±{std:.4f}")
    return auc

print("\n=== RESULTS ===\n")
a1 = cv_auc(X_base,      y, "A: 66 generic only (current L2)")
a2 = cv_auc(X_sf,        y, "B: 66 + 30 strategy features (meaningful zeros)")
a3 = cv_auc(X_sf_onehot, y, "C: 66 + 30 strategy features + 10 onehot")

# Also test: train L2 with HIGH R as target (R >= 0.5 = win)
y_highR = np.array([1 if float(r.get('profit_r',0) or 0) >= 0.5 else 0 for r in rows if extract_features_from_db(r) is not None])
print(f"\n  High-R target (R>=0.5): {y_highR.sum()} positives / {len(y_highR)} total  ({y_highR.mean()*100:.1f}%)")
a4 = cv_auc(X_sf_onehot, y_highR, "D: Best features + High-R target (R>=0.5)")

print(f"""
=== FINAL VERDICT ===
  A: 66 generic only (current):      AUC = {a1:.4f}
  B: 66 + 30 strategy features:      AUC = {a2:.4f}  delta = {a2-a1:+.4f}
  C: 66 + 30 + 10 onehot:            AUC = {a3:.4f}  delta = {a3-a1:+.4f}
  D: Best + High-R target (R>=0.5):  AUC = {a4:.4f}

  Best approach: {"C" if a3 >= a2 and a3 >= a1 else "B" if a2 >= a1 else "A"}
  Include strategy-specific features: {"YES" if max(a2,a3) > a1 else "NO"}
  High-R target improvement: {(a4-a3)*100:+.2f}% AUC points
""")

# Feature importance of best model
best_X = X_sf_onehot
X_arr = np.array(best_X, dtype=np.float32)
m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.65, min_child_weight=7,
    reg_alpha=0.5, reg_lambda=3.0, use_label_encoder=False,
    eval_metric='logloss', random_state=42)
m.fit(X_arr, y)

all_names = FEATURE_NAMES[:66] + STRAT_FEATURE_NAMES + [f'is_{s[:12]}' for s in STRATEGY_LIST]
top = sorted(zip(all_names[:len(m.feature_importances_)], m.feature_importances_), key=lambda x: -x[1])[:20]
print("=== TOP 20 FEATURES ===\n")
for name, score in top:
    print(f"  {name:35s}  {score:.4f}  {'|'*int(score*400)}")
print("\nDONE")
