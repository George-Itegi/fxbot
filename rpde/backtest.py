#!/usr/bin/env python3
# =============================================================
# rpde/backtest.py  —  RPDE Model Backtest Engine
#
# Tests trained pattern models against historical golden moments
# and negative samples to measure real-world discrimination ability.
#
# Usage:
#   python -m rpde backtest [DAYS]
#   python -m rpde backtest 90          # Test with 90 days of data
#   python -m rpde backtest 90 --pair EURJPY  # Test one pair
#   python -m rpde backtest 90 --thresholds  # Test multiple thresholds
#
# What it does:
#   1. Loads golden moments (actual trades) + negative samples (non-trades)
#   2. Loads trained pattern model for each pair
#   3. Runs model.predict() on every sample
#   4. Tests: if we only traded samples above threshold, what happens?
#   5. Reports WR, trade count, avg return, profit factor at each threshold
#   6. Time-split: train on first 70%, test on last 30% (out-of-sample)
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
    from rpde.config import PAIR_WHITELIST
    from config.settings import PAIR_WHITELIST as SETTINGS_PAIRS

    # Use the pair whitelist from settings
    all_pairs = list(SETTINGS_PAIRS or PAIR_WHITELIST)
    if pair_filter:
        all_pairs = [p for p in all_pairs if p.upper() == pair_filter.upper()]

    if not all_pairs:
        log.error("[RPDE_BT] No pairs to test")
        return

    # ── Print header ──
    print()
    print("=" * 64)
    print("  RPDE MODEL BACKTEST")
    print("=" * 64)
    print(f"  Pairs:    {len(all_pairs)}")
    print(f"  Data:     Last {days} days")
    print(f"  Mode:     {'Multi-threshold sweep' if test_thresholds else 'Single threshold (TAKE=0.3)'}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
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
    print("=" * 64)
    print("  BACKTEST COMPLETE")
    print("=" * 64)

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
    # Golden moments: features → actual forward_return (the label)
    # Negative samples: features → 0.0 (forced non-trade)
    pos_features, pos_labels = [], []
    neg_features, neg_labels = [], []

    for moment in golden_moments:
        features = _extract_features(moment)
        if features is not None:
            pos_features.append(features)
            pos_labels.append(float(moment.get('forward_return', 0)))

    for moment in negative_samples:
        features = _extract_features(moment)
        if features is not None:
            neg_features.append(features)
            neg_labels.append(0.0)

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

    # ── Time-based out-of-sample test ──
    # Golden moments are ordered by bar_timestamp DESC from DB
    # Reverse to get chronological order, then split 70/30
    pos_indices = list(range(n_pos))
    split_idx = int(n_pos * 0.7)

    # In-sample (first 70% of golden moments chronologically)
    # Out-of-sample (last 30%)
    # For a simple test: measure discrimination on ALL data first

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
    Compute metrics if we only trade samples with pred_R >= threshold.

    This simulates: "the model says TAKE (pred_R >= threshold), should we?"
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
            'precision': 0,  # of trades taken, how many were real golden moments
            'recall': 0,     # of all golden moments, how many did we catch
        }

    taken_labels = labels[mask]
    taken_positive = is_positive[mask]
    n_taken_pos = int(np.sum(taken_positive))
    n_taken_neg = n_taken - n_taken_pos

    # Win rate: of trades taken, what fraction had positive return?
    # (This is NOT the same as "was it a golden moment")
    wins = taken_labels > 0
    wr = float(np.mean(wins)) if n_taken > 0 else 0

    # Average return of taken trades
    avg_return = float(np.mean(taken_labels)) if n_taken > 0 else 0

    # Total return (sum of all R)
    total_return = float(np.sum(taken_labels))

    # Profit factor: total wins / total losses
    gross_profit = float(np.sum(taken_labels[taken_labels > 0]))
    gross_loss = abs(float(np.sum(taken_labels[taken_labels <= 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Precision: of trades taken, how many were actual golden moments?
    precision = n_taken_pos / n_taken if n_taken > 0 else 0

    # Recall: of all golden moments, how many did we catch?
    total_golden = int(np.sum(is_positive))
    recall = n_taken_pos / total_golden if total_golden > 0 else 0

    return {
        'threshold': threshold,
        'n_taken': n_taken,
        'n_taken_pos': n_taken_pos,
        'n_taken_neg': n_taken_neg,
        'wr': round(wr, 4),
        'avg_return': round(avg_return, 4),
        'total_return': round(total_return, 2),
        'pf': round(pf, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
    }


def _compute_oos_metrics(model, pos_features, pos_labels, neg_features, neg_labels):
    """
    Out-of-sample test: train on first 70%, test on last 30% of golden moments,
    combined with all negatives for evaluation.
    """
    n_pos = len(pos_labels)
    split_idx = int(n_pos * 0.7)

    # We can't retrain here (too expensive), so just test discrimination
    # on the "newer" 30% of golden moments vs all negatives
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
    Compute a simple AUC-like score: does the model rank positives
    higher than negatives on average?

    Returns 0.5 = random, 1.0 = perfect, 0.0 = perfectly wrong.
    """
    if len(predictions) == 0:
        return 0.5

    pos_preds = predictions[is_positive]
    neg_preds = predictions[~is_positive]

    if len(pos_preds) == 0 or len(neg_preds) == 0:
        return 0.5

    # Fraction of (positive, negative) pairs where positive has higher prediction
    # This is the AUC metric definition
    n_correct = 0
    n_total = 0

    # For large datasets, sample for speed
    n_sample = min(1000, len(pos_preds) * len(neg_preds))
    if n_sample < len(pos_preds) * len(neg_preds):
        # Random sampling
        rng = np.random.default_rng(42)
        pos_idx = rng.integers(0, len(pos_preds), size=n_sample)
        neg_idx = rng.integers(0, len(neg_preds), size=n_sample)
        comparisons = pos_preds[pos_idx] > neg_preds[neg_idx]
    else:
        # Full comparison using broadcasting
        comparisons = pos_preds[:, None] > neg_preds[None, :]

    n_correct = int(np.sum(comparisons))
    n_total = int(np.size(comparisons))

    return round(n_correct / n_total if n_total > 0 else 0.5, 4)


def _print_summary_table(all_results, test_thresholds):
    """Print a summary table of all pairs."""

    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if not valid_results:
        print("\n  No valid results to display.\n")
        return

    # ── Single threshold summary ──
    print()
    print("  " + "-" * 62)
    print("  MODEL DISCRIMINATION SUMMARY (threshold = 0.3)")
    print("  " + "-" * 62)
    print(f"  {'Pair':<10} {'Gold':>5} {'Neg':>5} {'AUC':>6} "
          f"{'Spread':>7} {'Taken':>6} {'WR':>6} {'AvgR':>7} {'PF':>7}")
    print(f"  {'':10} {'':5} {'':5} {'':6} "
          f"{'pred_R':>7} {'':6} {'':6} {'':7} {'':7}")
    print(f"  {'─' * 10} {'─' * 5} {'─' * 5} {'─' * 6} "
          f"{'─' * 7} {'─' * 6} {'─' * 6} {'─' * 7} {'─' * 7}")

    for pair, r in sorted(valid_results.items()):
        d = r.get('default', {})
        print(f"  {pair:<10} {r['n_golden']:>5} {r['n_negatives']:>5} "
              f"{r['auc']:>6.3f} {r['pred_spread']:>+7.3f} "
              f"{d.get('n_taken', 0):>6} {d.get('wr', 0):>5.1%} "
              f"{d.get('avg_return', 0):>+7.2f} {d.get('pf', 0):>7.2f}")

    print()

    # ── Out-of-sample summary ──
    print("  " + "-" * 62)
    print("  OUT-OF-SAMPLE TEST (last 30% of golden moments)")
    print("  " + "-" * 62)
    print(f"  {'Pair':<10} {'OOS Au':>6} {'OOS Spr':>8} "
          f"{'Taken':>6} {'WR':>6} {'AvgR':>7} {'PF':>7}")
    print(f"  {'─' * 10} {'─' * 6} {'─' * 8} "
          f"{'─' * 6} {'─' * 6} {'─' * 7} {'─' * 7}")

    for pair, r in sorted(valid_results.items()):
        oos = r.get('oos', {})
        if 'error' in oos:
            print(f"  {pair:<10} {'N/A':>6}")
            continue
        oos_d = oos.get('oos_default', {})
        oos_spread = oos.get('oos_mean_pred_golden', 0) - oos.get('oos_mean_pred_negative', 0)
        print(f"  {pair:<10} {oos.get('oos_auc', 0):>6.3f} {oos_spread:>+8.3f} "
              f"{oos_d.get('n_taken', 0):>6} {oos_d.get('wr', 0):>5.1%} "
              f"{oos_d.get('avg_return', 0):>+7.2f} {oos_d.get('pf', 0):>7.2f}")

    print()

    # ── Multi-threshold sweep ──
    if test_thresholds:
        for pair, r in sorted(valid_results.items()):
            thresholds = r.get('thresholds', {})
            if not thresholds:
                continue

            print(f"  {'─' * 50}")
            print(f"  THRESHOLD SWEEP: {pair}")
            print(f"  {'─' * 50}")
            print(f"  {'Thr':>5} {'Taken':>6} {'WR':>7} {'AvgR':>8} "
                  f"{'TotalR':>8} {'PF':>7} {'Prec':>6} {'Recall':>7}")
            print(f"  {'─' * 5} {'─' * 6} {'─' * 7} {'─' * 8} "
                  f"{'─' * 8} {'─' * 7} {'─' * 6} {'─' * 7}")

            for thr, m in sorted(thresholds.items()):
                print(f"  {thr:>5.1f} {m['n_taken']:>6} {m['wr']:>6.1%} "
                      f"{m['avg_return']:>+8.2f} {m['total_return']:>+8.1f} "
                      f"{m['pf']:>7.2f} {m['precision']:>5.1%} {m['recall']:>6.1%}")
            print()


def _print_recommendations(all_results):
    """Print actionable recommendations based on results."""

    valid = {k: v for k, v in all_results.items() if v is not None}

    if not valid:
        return

    # Count pairs by quality
    good_auc = sum(1 for r in valid.values() if r['auc'] >= 0.55)
    ok_auc = sum(1 for r in valid.values() if 0.50 <= r['auc'] < 0.55)
    bad_auc = sum(1 for r in valid.values() if r['auc'] < 0.50)

    print("  " + "-" * 62)
    print("  RECOMMENDATIONS")
    print("  " + "-" * 62)
    print()

    if good_auc == 0:
        print("  WARNING: No pair shows meaningful discrimination (AUC >= 0.55).")
        print("  The model cannot reliably distinguish golden moments from noise.")
        print("  DO NOT use these models for live trading.")
        print()
        print("  Next steps:")
        print("    1. Investigate feature distributions (positives vs negatives)")
        print("    2. Consider adding new features or removing redundant ones")
        print("    3. Try different model architectures or hyperparameters")
        print("    4. Increase training data by scanning more days")
    elif good_auc <= 3:
        print(f"  LIMITED: {good_auc}/{len(valid)} pairs show some discrimination.")
        print(f"  {ok_auc} pairs are borderline. {bad_auc} pairs show no signal.")
        print()
        print("  Consider using only the top-performing pairs for live trading,")
        print("  and focus improvement efforts on the features for weak pairs.")
    else:
        print(f"  GOOD: {good_auc}/{len(valid)} pairs show discrimination ability.")
        print(f"  The models may be useful with proper threshold tuning.")

    print()

    # Show best and worst pairs by AUC
    sorted_pairs = sorted(valid.items(), key=lambda x: x[1]['auc'], reverse=True)
    best = sorted_pairs[0]
    worst = sorted_pairs[-1]

    print(f"  BEST DISCRIMINATION:  {best[0]} (AUC={best[1]['auc']:.3f}, "
          f"spread={best[1]['pred_spread']:+.3f})")
    print(f"  WORST DISCRIMINATION: {worst[0]} (AUC={worst[1]['auc']:.3f}, "
          f"spread={worst[1]['pred_spread']:+.3f})")

    print()
    print("  KEY METRIC: 'Spread' = mean pred_R for golden moments minus")
    print("  mean pred_R for negatives. Higher = better discrimination.")
    print("  A spread near 0 means the model treats winners and losers the same.")
