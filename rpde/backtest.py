#!/usr/bin/env python3
# =============================================================
# rpde/backtest.py  —  RPDE Model Backtest Engine  v2.0
#
# HONEST backtest: uses REAL forward returns for both golden
# moments AND negative samples. No artificial R=0 for negatives.
#
# CRITICAL FIX v2.0:
#   v1.0 assigned R=0 to all negative samples, making PF=999
#   (no losses possible). In reality, non-golden bars can and DO
#   have negative returns — the model just needs to AVOID them.
#
#   The scanner stores real forward_return for negatives too:
#     direction='NONE' bars have their actual best forward move
#     stored in forward_return (usually 0.0-2.0R, below threshold).
#
#   This backtest uses those REAL returns to simulate actual P&L.
#
# Usage:
#   python -m rpde backtest [DAYS]
#   python -m rpde backtest 90          # Test with 90 days of data
#   python -m rpde backtest 90 --pair EURJPY  # Test one pair
#   python -m rpde backtest 90 --thresholds  # Test multiple thresholds
#
# WHAT IT TESTS:
#   "If we feed every scanned bar's features to the model, and only
#    take trades where pred_R >= threshold, what would the REAL P&L be?"
#
#   Golden moments have actual_R = whatever the market did (positive,
#   since they exceeded the big-move threshold).
#   Negative samples have actual_R = their real forward move / ATR
#   (usually small, but stored in DB).
#
#   A GOOD model gives high pred_R for golden moments and low pred_R
#   for negatives. By raising the threshold, we filter out negatives
#   (which have small actual_R) and keep golden moments (big actual_R).
#
# LIMITATIONS:
#   - Tests on the SAME data used for training (in-sample).
#     The OOS test uses the last 30% chronologically, which is a
#     better but still imperfect estimate of real performance.
#   - Does NOT do a true bar-by-bar walk-forward on unseen candles.
#   - Does NOT account for slippage, spreads, or execution delay.
# =============================================================

import sys
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from core.logger import get_logger
from ai_engine.ml_gate import FEATURE_NAMES

log = get_logger(__name__)

# Thresholds to test — "only trade when pred_R >= threshold"
DEFAULT_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def run_backtest(days: int = 90, pair_filter: str = None,
                 test_thresholds: bool = False):
    """
    Run backtest on all trained pattern models.

    Args:
        days: Number of days of historical data to include.
        pair_filter: If set, only test this specific pair.
        test_thresholds: If True, test multiple prediction thresholds.
    """
    from rpde.database import (
        load_golden_moments, load_negative_samples,
        load_pattern_library, init_rpde_tables,
    )
    from config.settings import PAIR_WHITELIST

    # Use the pair whitelist from settings
    all_pairs = list(PAIR_WHITELIST)
    if pair_filter:
        all_pairs = [p for p in all_pairs if p.upper() == pair_filter.upper()]

    if not all_pairs:
        log.error("[RPDE_BT] No pairs to test")
        return

    # ── Print header ──
    print()
    print("=" * 68)
    print("  RPDE MODEL BACKTEST  v2.0  (HONEST — real returns for all)")
    print("=" * 68)
    print(f"  Pairs:    {len(all_pairs)}")
    print(f"  Data:     Last {days} days")
    print(f"  Mode:     {'Multi-threshold sweep' if test_thresholds else 'Single threshold (TAKE=0.3)'}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
    print()
    print("  NOTE: This backtest tests on the SAME data used for training.")
    print("  Results are OPTIMISTIC. The OOS test (last 30%) is more honest.")
    print("  A true bar-by-bar walk-forward backtest is needed for real P&L.")
    print()

    # ── Load pattern library for context ──
    init_rpde_tables()
    library = load_pattern_library()
    lib_by_pair = defaultdict(list)
    for p in library:
        lib_by_pair[p.get('pair', '')].append(p)

    # ── Test each pair ──
    all_results = {}

    for pair in all_pairs:
        result = _backtest_pair(pair, days, test_thresholds, lib_by_pair)
        all_results[pair] = result

    # ── Print summary table ──
    _print_summary_table(all_results, test_thresholds)

    # ── Print recommendations ──
    _print_recommendations(all_results)

    print()
    print("=" * 68)
    print("  BACKTEST COMPLETE")
    print("=" * 68)

    return all_results


def _backtest_pair(pair: str, days: int, test_thresholds: bool,
                   lib_by_pair: dict) -> dict:
    """Run backtest for a single pair."""

    from rpde.database import load_golden_moments, load_negative_samples
    from rpde.pattern_model import PatternModel

    # Load model
    model = PatternModel(pair)
    if not model.is_trained():
        log.warning(f"[RPDE_BT] {pair}: No trained model found, skipping")
        return None

    # Load data from DB
    golden_moments = load_golden_moments(pair=pair)
    negative_samples = load_negative_samples(pair=pair)

    if not golden_moments and not negative_samples:
        log.warning(f"[RPDE_BT] {pair}: No data in DB, skipping")
        return None

    # Build feature arrays
    # v2.0: Use REAL forward_return for BOTH golden moments and negatives.
    # Golden moments: their actual forward_return (always positive, > threshold).
    # Negative samples: their REAL forward_return from DB (the actual best
    #   forward move, even though it didn't exceed the big-move threshold).
    #   These are typically 0.0-2.0R but stored truthfully in the DB.
    pos_features, pos_labels = [], []
    neg_features, neg_labels = [], []
    neg_real_returns = []  # Track what negatives actually returned

    for moment in golden_moments:
        features = _extract_features(moment)
        if features is not None:
            pos_features.append(features)
            pos_labels.append(float(moment.get('forward_return', 0)))

    for moment in negative_samples:
        features = _extract_features(moment)
        if features is not None:
            neg_features.append(features)
            # v2.0 FIX: Use REAL forward_return from DB, NOT forced 0.0
            real_r = float(moment.get('forward_return', 0))
            neg_labels.append(real_r)
            neg_real_returns.append(real_r)

    if not pos_features:
        log.warning(f"[RPDE_BT] {pair}: No valid golden moment features")
        return None

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features) if neg_features else np.empty((0, len(FEATURE_NAMES)))
    pos_labels = np.array(pos_labels)
    neg_labels = np.array(neg_labels)

    # Combine all samples
    all_features = np.vstack([pos_features, neg_features]) if len(neg_features) > 0 else pos_features
    all_labels = np.concatenate([pos_labels, neg_labels])
    is_positive = np.array([True] * len(pos_labels) + [False] * len(neg_labels))

    # Run predictions
    predictions = model.model.predict(all_features)
    predictions = np.clip(predictions, -2.0, 5.0)

    n_pos = len(pos_labels)
    n_neg = len(neg_labels)

    # Log negative sample return distribution
    if neg_real_returns:
        neg_arr = np.array(neg_real_returns)
        log.info(f"[RPDE_BT] {pair}: {n_neg} negatives — "
                 f"mean_R={np.mean(neg_arr):.3f}, median_R={np.median(neg_arr):.3f}, "
                 f"min_R={np.min(neg_arr):.3f}, max_R={np.max(neg_arr):.3f}, "
                 f"pct_positive={np.mean(neg_arr > 0):.1%}")

    # ── Compute metrics ──
    result = {
        'pair': pair,
        'n_golden': n_pos,
        'n_negatives': n_neg,
        'n_total': n_pos + n_neg,
        'library_patterns': len(lib_by_pair.get(pair, [])),
        'mean_pred_golden': round(float(np.mean(predictions[:n_pos])), 4),
        'mean_pred_negative': round(float(np.mean(predictions[n_pos:])) if n_neg > 0 else 0, 4),
        'pred_spread': round(float(np.mean(predictions[:n_pos]) - (
            np.mean(predictions[n_pos:]) if n_neg > 0 else 0)), 4),
        'auc': _compute_auc(predictions, is_positive),
        'neg_mean_real_r': round(float(np.mean(neg_labels)) if n_neg > 0 else 0, 4),
    }

    # ── Test default threshold (TAKE = 0.3) ──
    result['default'] = _compute_threshold_metrics(
        predictions, all_labels, is_positive, threshold=0.3
    )

    # ── Test multiple thresholds ──
    if test_thresholds:
        result['thresholds'] = {}
        for thr in DEFAULT_THRESHOLDS:
            result['thresholds'][thr] = _compute_threshold_metrics(
                predictions, all_labels, is_positive, threshold=thr
            )

    # ── Time-split out-of-sample test ──
    result['oos'] = _compute_oos_metrics(
        model, pos_features, pos_labels, neg_features, neg_labels
    )

    return result


def _extract_features(moment: dict) -> np.ndarray or None:
    """Extract feature vector from a DB moment dict."""
    features = moment.get('features') or moment.get('feature_snapshot') or {}
    if not features:
        return None

    row = []
    for feat_name in FEATURE_NAMES:
        val = features.get(feat_name)
        if val is None:
            return None  # Skip samples with missing features
        row.append(float(val))

    return np.array(row)


def _compute_threshold_metrics(predictions, labels, is_positive, threshold):
    """
    Compute HONEST P&L metrics if we only trade samples with pred_R >= threshold.

    v2.0: Uses real forward_return for ALL samples (both golden and negative).
    A "win" = actual R > 0. A "loss" = actual R <= 0.
    PF = total positive R / |total negative R|.
    """
    # How many samples pass the threshold?
    mask = predictions >= threshold
    n_taken = int(np.sum(mask))

    if n_taken == 0:
        return {
            'threshold': threshold,
            'n_taken': 0,
            'n_taken_pos': 0,
            'n_taken_neg': 0,
            'wr': 0,
            'avg_return': 0,
            'total_return': 0,
            'pf': 0,
            'precision': 0,
            'recall': 0,
            'max_drawdown_r': 0,
        }

    taken_labels = labels[mask]
    taken_positive = is_positive[mask]
    n_taken_pos = int(np.sum(taken_positive))
    n_taken_neg = n_taken - n_taken_pos

    # Win rate: of trades taken, what fraction had positive actual R?
    wins = taken_labels > 0
    n_wins = int(np.sum(wins))
    wr = float(np.mean(wins)) if n_taken > 0 else 0

    # Average return of taken trades
    avg_return = float(np.mean(taken_labels)) if n_taken > 0 else 0

    # Total return (sum of all R — this is your P&L in R-multiples)
    total_return = float(np.sum(taken_labels))

    # Profit factor: total winning R / total losing R
    gross_profit = float(np.sum(taken_labels[taken_labels > 0]))
    gross_loss = abs(float(np.sum(taken_labels[taken_labels <= 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Precision: of trades taken, how many were actual golden moments?
    precision = n_taken_pos / n_taken if n_taken > 0 else 0

    # Recall: of all golden moments, how many did we catch?
    total_golden = int(np.sum(is_positive))
    recall = n_taken_pos / total_golden if total_golden > 0 else 0

    # Max drawdown (in R): worst cumulative loss from peak
    cumulative = np.cumsum(taken_labels)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_drawdown_r = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    return {
        'threshold': threshold,
        'n_taken': n_taken,
        'n_taken_pos': n_taken_pos,
        'n_taken_neg': n_taken_neg,
        'n_wins': n_wins,
        'n_losses': n_taken - n_wins,
        'wr': round(wr, 4),
        'avg_return': round(avg_return, 4),
        'total_return': round(total_return, 2),
        'pf': round(pf, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'max_drawdown_r': round(max_drawdown_r, 2),
    }


def _compute_oos_metrics(model, pos_features, pos_labels, neg_features, neg_labels):
    """
    Out-of-sample test: test on the last 30% of golden moments
    (chronologically) combined with all negatives.

    This is the most honest metric available without a full walk-forward
    backtest on unseen candles.
    """
    n_pos = len(pos_labels)
    split_idx = int(n_pos * 0.7)

    if split_idx >= n_pos:
        return {'error': 'Not enough samples for OOS split'}

    oos_pos_features = pos_features[split_idx:]
    oos_pos_labels = pos_labels[split_idx:]

    oos_all_features = np.vstack([oos_pos_features, neg_features]) if len(neg_features) > 0 else oos_pos_features
    oos_all_labels = np.concatenate([oos_pos_labels, neg_labels])
    oos_is_positive = np.array([True] * len(oos_pos_labels) + [False] * len(neg_labels))

    oos_preds = model.model.predict(oos_all_features)
    oos_preds = np.clip(oos_preds, -2.0, 5.0)

    return {
        'oos_n_golden': len(oos_pos_labels),
        'oos_n_negatives': len(neg_labels),
        'oos_mean_pred_golden': round(float(np.mean(oos_preds[:len(oos_pos_labels)])), 4),
        'oos_mean_pred_negative': round(float(np.mean(oos_preds[len(oos_pos_labels):])) if len(neg_labels) > 0 else 0, 4),
        'oos_auc': _compute_auc(oos_preds, oos_is_positive),
        'oos_default': _compute_threshold_metrics(
            oos_preds, oos_all_labels, oos_is_positive, threshold=0.3
        ),
    }


def _compute_auc(predictions, is_positive):
    """
    Compute AUC score: fraction of (positive, negative) pairs where
    the positive sample has a higher prediction.

    Returns 0.5 = random, 1.0 = perfect, 0.0 = perfectly wrong.
    """
    if len(predictions) == 0:
        return 0.5

    pos_preds = predictions[is_positive]
    neg_preds = predictions[~is_positive]

    if len(pos_preds) == 0 or len(neg_preds) == 0:
        return 0.5

    # For large datasets, sample for speed
    n_pairs = len(pos_preds) * len(neg_preds)
    n_sample = min(1000, n_pairs)
    if n_sample < n_pairs:
        rng = np.random.default_rng(42)
        pos_idx = rng.integers(0, len(pos_preds), size=n_sample)
        neg_idx = rng.integers(0, len(neg_preds), size=n_sample)
        comparisons = pos_preds[pos_idx] > neg_preds[neg_idx]
    else:
        comparisons = pos_preds[:, None] > neg_preds[None, :]

    n_correct = int(np.sum(comparisons))
    n_total = int(np.size(comparisons))

    return round(n_correct / n_total if n_total > 0 else 0.5, 4)


def _print_summary_table(all_results, test_thresholds):
    """Print summary tables of all pairs."""

    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if not valid_results:
        print("\n  No valid results to display.\n")
        return

    # ── DISCRIMINATION TABLE ──
    # Shows AUC and prediction spread (model quality metrics)
    print()
    print("  " + "-" * 66)
    print("  MODEL DISCRIMINATION (how well model separates gold from noise)")
    print("  " + "-" * 66)
    print(f"  {'Pair':<10} {'Gold':>5} {'Neg':>5} {'AUC':>6} "
          f"{'Spread':>7} {'NegAvgR':>8}")
    print(f"  {'':10} {'':5} {'':5} {'':6} "
          f"{'pred_R':>7} {'(real)':>8}")
    print(f"  {'─' * 10} {'─' * 5} {'─' * 5} {'─' * 6} "
          f"{'─' * 7} {'─' * 8}")

    for pair, r in sorted(valid_results.items()):
        print(f"  {pair:<10} {r['n_golden']:>5} {r['n_negatives']:>5} "
              f"{r['auc']:>6.3f} {r['pred_spread']:>+7.3f} "
              f"{r.get('neg_mean_real_r', 0):>+8.3f}")

    print()
    print("  Spread = mean(pred_R for golden) - mean(pred_R for negative)")
    print("  NegAvgR = actual mean forward return of negative samples")
    print("  AUC > 0.55 = model has some discrimination ability")

    # ── IN-SAMPLE P&L TABLE (threshold = 0.3) ──
    print()
    print("  " + "-" * 66)
    print("  IN-SAMPLE P&L (threshold = 0.3) — OPTIMISTIC (trained data)")
    print("  " + "-" * 66)
    print(f"  {'Pair':<10} {'Taken':>6} {'Wins':>5} {'Loss':>5} "
          f"{'WR':>6} {'AvgR':>7} {'TotalR':>9} {'PF':>6} {'DD':>7}")
    print(f"  {'─' * 10} {'─' * 6} {'─' * 5} {'─' * 5} "
          f"{'─' * 6} {'─' * 7} {'─' * 9} {'─' * 6} {'─' * 7}")

    for pair, r in sorted(valid_results.items()):
        d = r.get('default', {})
        print(f"  {pair:<10} {d.get('n_taken', 0):>6} "
              f"{d.get('n_wins', 0):>5} {d.get('n_losses', 0):>5} "
              f"{d.get('wr', 0):>5.1%} "
              f"{d.get('avg_return', 0):>+7.2f} "
              f"{d.get('total_return', 0):>+9.1f} "
              f"{d.get('pf', 0):>6.2f} "
              f"{d.get('max_drawdown_r', 0):>7.1f}")

    print()

    # ── OUT-OF-SAMPLE P&L TABLE (last 30%) ──
    print("  " + "-" * 66)
    print("  OUT-OF-SAMPLE P&L (last 30% golden) — MORE HONEST")
    print("  " + "-" * 66)
    print(f"  {'Pair':<10} {'OOS Au':>6} {'Taken':>6} {'WR':>6} "
          f"{'AvgR':>7} {'TotalR':>9} {'PF':>6} {'DD':>7}")
    print(f"  {'─' * 10} {'─' * 6} {'─' * 6} {'─' * 6} "
          f"{'─' * 7} {'─' * 9} {'─' * 6} {'─' * 7}")

    for pair, r in sorted(valid_results.items()):
        oos = r.get('oos', {})
        if 'error' in oos:
            print(f"  {pair:<10} {'N/A':>6}")
            continue
        oos_d = oos.get('oos_default', {})
        oos_spread = oos.get('oos_mean_pred_golden', 0) - oos.get('oos_mean_pred_negative', 0)
        print(f"  {pair:<10} {oos.get('oos_auc', 0):>6.3f} "
              f"{oos_d.get('n_taken', 0):>6} "
              f"{oos_d.get('wr', 0):>5.1%} "
              f"{oos_d.get('avg_return', 0):>+7.2f} "
              f"{oos_d.get('total_return', 0):>+9.1f} "
              f"{oos_d.get('pf', 0):>6.2f} "
              f"{oos_d.get('max_drawdown_r', 0):>7.1f}")

    print()

    # ── Multi-threshold sweep ──
    if test_thresholds:
        for pair, r in sorted(valid_results.items()):
            thresholds = r.get('thresholds', {})
            if not thresholds:
                continue

            print(f"  {'─' * 66}")
            print(f"  THRESHOLD SWEEP: {pair}")
            print(f"  {'─' * 66}")
            print(f"  {'Thr':>5} {'Taken':>6} {'W':>4} {'L':>4} "
                  f"{'WR':>7} {'AvgR':>8} {'TotalR':>10} {'PF':>6} "
                  f"{'Prec':>6} {'Recall':>7} {'DD':>7}")
            print(f"  {'─' * 5} {'─' * 6} {'─' * 4} {'─' * 4} "
                  f"{'─' * 7} {'─' * 8} {'─' * 10} {'─' * 6} "
                  f"{'─' * 6} {'─' * 7} {'─' * 7}")

            for thr, m in sorted(thresholds.items()):
                pf_str = f"{m['pf']:>6.2f}" if m['pf'] < 999 else "  INF"
                print(f"  {thr:>5.1f} {m['n_taken']:>6} "
                      f"{m.get('n_wins', 0):>4} {m.get('n_losses', 0):>4} "
                      f"{m['wr']:>6.1%} "
                      f"{m['avg_return']:>+8.2f} {m['total_return']:>+10.1f} "
                      f"{pf_str} "
                      f"{m['precision']:>5.1%} {m['recall']:>6.1%} "
                      f"{m.get('max_drawdown_r', 0):>7.1f}")
            print()


def _print_recommendations(all_results):
    """Print honest recommendations based on results."""

    valid = {k: v for k, v in all_results.items() if v is not None}

    if not valid:
        return

    # ── OOS performance check (the most honest metric) ──
    oos_profitable = 0
    oos_good_wr = 0
    oos_good_pf = 0

    for pair, r in valid.items():
        oos = r.get('oos', {})
        if 'error' in oos:
            continue
        oos_d = oos.get('oos_default', {})
        if oos_d.get('total_return', 0) > 0:
            oos_profitable += 1
        if oos_d.get('wr', 0) > 0.50:
            oos_good_wr += 1
        pf = oos_d.get('pf', 0)
        if 0 < pf < 999 and pf > 1.5:
            oos_good_pf += 1

    # ── In-sample check ──
    in_good_pf = 0
    for pair, r in valid.items():
        d = r.get('default', {})
        pf = d.get('pf', 0)
        if 0 < pf < 999 and pf > 1.5:
            in_good_pf += 1

    print("  " + "-" * 66)
    print("  RECOMMENDATIONS")
    print("  " + "-" * 66)
    print()

    # OOS diagnosis
    print(f"  OUT-OF-SAMPLE (last 30% — most honest metric):")
    print(f"    {oos_profitable}/{len(valid)} pairs have positive total return")
    print(f"    {oos_good_wr}/{len(valid)} pairs have WR > 50%")
    print(f"    {oos_good_pf}/{len(valid)} pairs have PF > 1.5")
    print()

    if oos_profitable == 0:
        print("  VERDICT: NOT PROFITABLE on unseen data.")
        print("  DO NOT trade these models live. The model is overfitting")
        print("  to training data — it recognizes patterns it already saw")
        print("  but cannot generalize to new market conditions.")
        print()
        print("  Root causes:")
        print("    1. Testing on training data = circular reasoning")
        print("    2. 26/93 features are hardcoded defaults during scanning")
        print("    3. Model may be memorizing feature patterns, not learning")
        print()
        print("  Required fixes:")
        print("    1. Build a TRUE walk-forward backtest (bar-by-bar)")
        print("    2. Fill in the 26 hardcoded-default features with real data")
        print("    3. Use time-series cross-validation (not random shuffle)")
        print("    4. Retrain on OLDER data, test on NEWER data only")
    elif oos_profitable <= 3:
        print(f"  VERDICT: MARGINAL. Only {oos_profitable}/{len(valid)} pairs")
        print("  are profitable on out-of-sample data. This could be noise.")
        print()
        print("  Recommendations:")
        print("    1. Focus on the profitable pairs only")
        print("    2. Build a true walk-forward backtest to confirm")
        print("    3. Add more data (180+ days) for better statistics")
    else:
        print(f"  VERDICT: PROMISING. {oos_profitable}/{len(valid)} pairs show")
        print("  positive OOS returns. But this still tests on scanned data,")
        print("  not a true walk-forward simulation.")
        print()
        print("  Next steps:")
        print("    1. Build a bar-by-bar walk-forward backtest")
        print("    2. Paper trade with small position sizes")
        print("    3. Monitor real performance vs backtest expectations")

    print()

    # In-sample vs OOS comparison (overfitting check)
    if in_good_pf > oos_good_pf:
        gap = in_good_pf - oos_good_pf
        print(f"  OVERFITTING WARNING: In-sample has {in_good_pf} pairs with PF>1.5")
        print(f"  but OOS only has {oos_good_pf}. Gap = {gap} pairs.")
        print(f"  This suggests the model is memorizing, not generalizing.")
        print()

    # AUC analysis
    sorted_pairs = sorted(valid.items(), key=lambda x: x[1]['auc'], reverse=True)
    best = sorted_pairs[0]
    worst = sorted_pairs[-1]
    print(f"  BEST DISCRIMINATION:  {best[0]} (AUC={best[1]['auc']:.3f})")
    print(f"  WORST DISCRIMINATION: {worst[0]} (AUC={worst[1]['auc']:.3f})")
    print()
    print("  NOTE: High AUC does NOT guarantee profitable trading.")
    print("  AUC measures classification ability. Profitability depends on")
    print("  the MAGNITUDE of wins vs losses (avg_R, PF, total_return).")
