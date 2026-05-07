# =============================================================
# rpde/pattern_miner.py  —  RPDE Pattern Mining Module
#
# Takes golden moments (feature snapshots before big moves) and
# clusters them into candidate patterns. Each cluster represents
# a recurring market condition that precedes profitable moves.
#
# Pipeline:
#   Golden Moments → Feature Extraction → Normalization →
#   DBSCAN Clustering → Cluster Analysis → Pattern Candidates
#
# Key concept:
#   Golden moments = feature snapshots at bars BEFORE big moves
#   Clustering     = grouping similar snapshots together
#   Each cluster   = a discovered "strategy" that the AI found
# =============================================================

import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from core.logger import get_logger
from database.db_manager import get_connection
from rpde.config import (
    CLUSTER_FEATURES,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    CLUSTER_ALGORITHM,
    MIN_PATTERN_OCCURRENCES,
    PAIR_MOVE_THRESHOLDS,
    DEFAULT_MIN_MOVE_PIPS,
)
from config.settings import PAIR_WHITELIST

logger = get_logger("rpde.pattern_miner")


# ── Golden Moment DB Access ──────────────────────────────────
# Uses rpde.database for consistent table access (rpde_pattern_scans).

from rpde.database import load_golden_moments as _db_load_golden_moments


def load_golden_moments(pair: str, scan_ids: list = None) -> list:
    """
    Load all golden moments for a pair from the database.

    Each row is a dict with keys including:
        id, pair, scan_id, bar_timestamp, direction, move_pips,
        move_duration_bars, features_json (JSON string of all 93 features)

    Parameters
    ----------
    pair : str
        Currency pair, e.g. 'EURUSD'.
    scan_ids : list or None
        Optional list of scan IDs to filter on.

    Returns
    -------
    list[dict]
        List of golden moment row dicts.
    """
    try:
        if scan_ids:
            # database.py only supports single scan_id, so load per scan
            # and deduplicate by id
            all_rows = {}
            for sid in scan_ids:
                rows = _db_load_golden_moments(pair=pair, scan_id=sid)
                for row in rows:
                    all_rows[row["id"]] = row
            moments = list(all_rows.values())
            moments.sort(key=lambda r: r.get("bar_timestamp", ""))
            logger.info(f"Loaded {len(moments)} golden moments for {pair}")
            return moments
        else:
            moments = _db_load_golden_moments(pair=pair)
            # database.py returns DESC order; reverse to ASC for miner
            moments.reverse()
            logger.info(f"Loaded {len(moments)} golden moments for {pair}")
            return moments

    except Exception as e:
        logger.error(f"Failed to load golden moments for {pair}: {e}")
        return []


# ── Feature Extraction ───────────────────────────────────────

def _extract_cluster_features(moments: list) -> tuple:
    """
    Extract the clustering features from golden moments.
    Uses only CLUSTER_FEATURES from rpde/config.py (about 25 features).

    Parameters
    ----------
    moments : list[dict]
        List of golden moment dicts (each must have a 'features' key
        which is a dict of feature_name → value).

    Returns
    -------
    tuple of (np.ndarray, list[int])
        (X, valid_indices)
        X is shape (n_valid_moments, n_features).
        valid_indices are the positions in `moments` that had valid data.
    """
    X_rows = []
    valid_indices = []

    for i, moment in enumerate(moments):
        features = moment.get("features", {})
        if not features:
            continue

        row = []
        missing = False
        for feat in CLUSTER_FEATURES:
            val = features.get(feat)
            if val is None:
                missing = True
                break
            row.append(float(val))

        if missing:
            continue

        X_rows.append(row)
        valid_indices.append(i)

    if not X_rows:
        logger.warning("No valid feature vectors could be extracted from moments")
        return np.empty((0, len(CLUSTER_FEATURES))), []

    X = np.array(X_rows, dtype=np.float64)
    logger.debug(
        f"Extracted feature matrix: {X.shape} "
        f"({len(valid_indices)}/{len(moments)} moments valid)"
    )
    return X, valid_indices


# ── Feature Normalization ────────────────────────────────────

def _normalize_features(X: np.ndarray) -> tuple:
    """
    Normalize features using StandardScaler.
    Handles edge cases where all values are identical in a column.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix of shape (n_samples, n_features).

    Returns
    -------
    tuple of (np.ndarray, StandardScaler)
        (X_normalized, scaler)
    """
    if X.shape[0] == 0:
        logger.warning("Cannot normalize empty feature matrix")
        return X, None

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Replace any NaN/Inf from zero-variance features with 0
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug(
        f"Normalized features: mean per feature "
        f"range [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]"
    )
    return X_norm, scaler


# ── DBSCAN Clustering ────────────────────────────────────────

def _cluster_dbscan(X: np.ndarray, eps: float = None,
                    min_samples: int = None) -> np.ndarray:
    """
    Cluster using DBSCAN algorithm.
    DBSCAN is preferred because:
    - Doesn't require specifying number of clusters
    - Can find arbitrarily shaped clusters
    - Labels noise points as -1 (outliers get discarded)

    Parameters
    ----------
    X : np.ndarray
        Normalized feature matrix of shape (n_samples, n_features).
    eps : float or None
        Maximum distance between samples. Defaults to DBSCAN_EPS from config.
    min_samples : int or None
        Min samples to form a cluster. Defaults to DBSCAN_MIN_SAMPLES from config.

    Returns
    -------
    np.ndarray
        Cluster labels array. -1 = noise.
    """
    if X.shape[0] == 0:
        logger.warning("Cannot cluster empty feature matrix")
        return np.array([], dtype=int)

    eps = eps if eps is not None else DBSCAN_EPS
    min_samples = min_samples if min_samples is not None else DBSCAN_MIN_SAMPLES

    # If we have very few samples, reduce min_samples to allow clustering
    if X.shape[0] < min_samples:
        logger.warning(
            f"Only {X.shape[0]} samples but min_samples={min_samples}. "
            f"Falling back to min_samples=3."
        )
        min_samples = max(3, X.shape[0] // 2)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(X)

    unique_labels = set(labels) - {-1}
    noise_count = int(np.sum(labels == -1))
    logger.info(
        f"DBSCAN clustering done: {len(unique_labels)} clusters, "
        f"{noise_count} noise points out of {len(labels)} total"
    )

    return labels


# ── Distinguishing Feature Identification ────────────────────

def _identify_distinguishing_features(X_cluster: np.ndarray,
                                       X_all: np.ndarray,
                                       feature_names: list,
                                       scaler) -> list:
    """
    Find features that distinguish this cluster from the overall population.

    Uses the difference in mean values (z-scores) to rank features.
    The features with the highest z-scores are the most distinctive
    characteristics of this pattern.

    Parameters
    ----------
    X_cluster : np.ndarray
        Raw (un-normalized) feature values for this cluster's members.
    X_all : np.ndarray
        Raw (un-normalized) feature values for the entire population.
    feature_names : list[str]
        Feature name for each column.
    scaler : StandardScaler
        Fitted scaler (used to get overall std per feature).

    Returns
    -------
    list[tuple]
        [(feature_name, z_score)] sorted by z_score descending, top 10.
    """
    if X_cluster.shape[0] == 0 or X_all.shape[0] == 0:
        return []

    cluster_mean = np.mean(X_cluster, axis=0)
    all_mean = np.mean(X_all, axis=0)

    # Use scaler std for z-score computation; guard against zero std
    if scaler is not None:
        all_std = np.array(scaler.scale_, dtype=np.float64)
        all_std[all_std == 0] = 1.0  # avoid division by zero
    else:
        all_std = np.std(X_all, axis=0)
        all_std[all_std == 0] = 1.0

    # Absolute z-score per feature
    z_scores = np.abs(cluster_mean - all_mean) / all_std

    # Rank descending
    ranked_indices = np.argsort(z_scores)[::-1]

    top_features = []
    for idx in ranked_indices[:10]:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        top_features.append((feat_name, round(float(z_scores[idx]), 3)))

    return top_features


# ── Single Cluster Analysis ──────────────────────────────────

def _analyze_cluster(moments: list, valid_indices: list,
                     labels: np.ndarray, cluster_id: int,
                     X: np.ndarray, X_raw: np.ndarray,
                     scaler) -> dict:
    """
    Analyze a single cluster to produce a pattern candidate.

    Computes:
    - Direction: majority of BUY vs SELL in cluster
    - Win rate: % of moments where move was successful
    - Feature ranges: min, max, mean, std per feature
    - Top distinguishing features: features where this cluster
      differs most from the overall distribution
    - Cluster center: mean of normalized feature values
    - Profit factor and expected R-multiple

    Parameters
    ----------
    moments : list[dict]
        All golden moment dicts for this pair.
    valid_indices : list[int]
        Indices into `moments` that have valid features.
    labels : np.ndarray
        Cluster labels from DBSCAN.
    cluster_id : int
        The cluster ID to analyze.
    X : np.ndarray
        Normalized feature matrix.
    X_raw : np.ndarray
        Raw (un-normalized) feature matrix.
    scaler : StandardScaler
        Fitted scaler.

    Returns
    -------
    dict
        Pattern candidate dict with all computed fields, or None if invalid.
    """
    # Get indices of moments belonging to this cluster
    cluster_mask = labels == cluster_id
    cluster_local_indices = np.where(cluster_mask)[0]

    if len(cluster_local_indices) == 0:
        return None

    # Map local indices back to original moments via valid_indices
    cluster_moment_indices = [valid_indices[i] for i in cluster_local_indices]
    cluster_moments = [moments[i] for i in cluster_moment_indices]
    cluster_X_norm = X[cluster_mask]
    cluster_X_raw = X_raw[cluster_mask]

    n_members = len(cluster_moments)

    # ── Minimum cluster size gate ──
    if n_members < MIN_PATTERN_OCCURRENCES:
        logger.debug(
            f"Cluster {cluster_id}: only {n_members} members, "
            f"need >= {MIN_PATTERN_OCCURRENCES} — skipping"
        )
        return None

    # ── Direction: majority vote ──
    buy_count = sum(1 for m in cluster_moments if m.get("direction") == "BUY")
    sell_count = n_members - buy_count
    direction = "BUY" if buy_count >= sell_count else "SELL"

    # ── Move pips ──
    move_pips_list = [
        m["move_pips"] for m in cluster_moments
        if m.get("move_pips") is not None
    ]
    if not move_pips_list:
        logger.debug(f"Cluster {cluster_id}: no move_pips data — skipping")
        return None

    avg_profit_pips = float(np.mean(move_pips_list))
    median_profit_pips = float(np.median(move_pips_list))

    # ── Win / Loss determination ──
    # A golden moment is a "win" if its move_pips >= threshold for its direction.
    pair = cluster_moments[0].get("pair", "")
    pip_threshold = PAIR_MOVE_THRESHOLDS.get(pair, DEFAULT_MIN_MOVE_PIPS)

    # For BUY moves, positive move_pips = win. For SELL moves, positive = win.
    # (move_pips is always stored as absolute positive value for the direction)
    wins = sum(1 for p in move_pips_list if p >= pip_threshold * 0.5)
    losses = n_members - wins
    win_rate = wins / n_members if n_members > 0 else 0.0

    # Separate profits and losses for profit factor
    profit_values = [p for p in move_pips_list if p > 0]
    loss_values = [p for p in move_pips_list if p <= 0]

    total_profit = sum(profit_values) if profit_values else 0.0
    total_loss = abs(sum(loss_values)) if loss_values else 1.0  # avoid /0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    avg_win_pips = float(np.mean(profit_values)) if profit_values else 0.0
    avg_loss_pips = float(np.mean(loss_values)) if loss_values else 0.0

    # ── Expected R-multiple ──
    # E[R] = win_rate * avg_win - (1 - win_rate) * avg_loss_abs
    expected_r = win_rate * avg_win_pips - (1 - win_rate) * abs(avg_loss_pips)

    # ── Cluster center (normalized) ──
    center = np.mean(cluster_X_norm, axis=0)

    # ── Feature ranges (raw values, with percentile bounds) ──
    feature_ranges = {}
    for j, feat_name in enumerate(CLUSTER_FEATURES):
        col = cluster_X_raw[:, j]
        feature_ranges[feat_name] = {
            "min": round(float(np.percentile(col, 5)), 6),
            "max": round(float(np.percentile(col, 95)), 6),
            "mean": round(float(np.mean(col)), 6),
            "std": round(float(np.std(col)), 6),
        }

    # ── Top distinguishing features ──
    top_features = _identify_distinguishing_features(
        cluster_X_raw, X_raw, CLUSTER_FEATURES, scaler
    )

    # ── Golden moment IDs ──
    moment_ids = [m.get("id") for m in cluster_moments]

    pattern = {
        "pair": pair,
        "cluster_id": int(cluster_id),
        "direction": direction,
        "occurrences": n_members,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "avg_profit_pips": round(avg_profit_pips, 2),
        "median_profit_pips": round(median_profit_pips, 2),
        "avg_win_pips": round(avg_win_pips, 2),
        "avg_loss_pips": round(avg_loss_pips, 2),
        "profit_factor": round(profit_factor, 2),
        "expected_r": round(expected_r, 2),
        "center": center,
        "feature_ranges": feature_ranges,
        "top_features": top_features,
        "golden_moments": moment_ids,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "direction_purity": round(
            max(buy_count, sell_count) / n_members, 3
        ),
    }

    logger.debug(
        f"Cluster {cluster_id} ({pair}): {n_members} occurrences, "
        f"WR={win_rate:.1%}, direction={direction}, "
        f"PF={profit_factor:.2f}, E[R]={expected_r:.2f}"
    )

    return pattern


# ── Main Mining Function ─────────────────────────────────────

def mine_patterns(pair: str, scan_ids: list = None) -> list:
    """
    Mine patterns from golden moments for a specific pair.

    Pipeline:
        1. Load all golden moments for this pair from DB
        2. Extract feature vectors (using CLUSTER_FEATURES from config)
        3. Normalize features (StandardScaler)
        4. Cluster using DBSCAN (or KMeans)
        5. For each cluster:
           - Compute cluster center (mean feature values)
           - Compute feature ranges (min/max per feature)
           - Calculate win rate, avg profit, occurrences
           - Identify top distinguishing features
        6. Return list of candidate patterns

    Parameters
    ----------
    pair : str
        Currency pair, e.g. 'EURUSD'.
    scan_ids : list or None
        Optional list of scan IDs to filter on.

    Returns
    -------
    list[dict]
        List of pattern candidate dicts, sorted by win_rate descending.
    """
    logger.info(f"{'='*50}")
    logger.info(f"Mining patterns for {pair}")
    logger.info(f"{'='*50}")

    # ── Step 1: Load golden moments ──
    moments = load_golden_moments(pair, scan_ids)
    if not moments:
        logger.warning(f"No golden moments found for {pair}")
        return []

    logger.info(f"Loaded {len(moments)} golden moments for {pair}")

    # ── Step 2: Extract feature vectors ──
    X_raw, valid_indices = _extract_cluster_features(moments)
    if X_raw.shape[0] == 0:
        logger.warning(f"No valid feature vectors for {pair}")
        return []

    if X_raw.shape[0] < MIN_PATTERN_OCCURRENCES:
        logger.warning(
            f"Only {X_raw.shape[0]} valid moments for {pair}, "
            f"need >= {MIN_PATTERN_OCCURRENCES} — skipping"
        )
        return []

    # ── Step 3: Normalize ──
    X_norm, scaler = _normalize_features(X_raw)

    # ── Step 4: Cluster ──
    if CLUSTER_ALGORITHM == "dbscan":
        labels = _cluster_dbscan(X_norm)
    else:
        # Fallback: use KMeans
        logger.info(f"Using KMeans fallback (config says '{CLUSTER_ALGORITHM}')")
        from sklearn.cluster import KMeans
        n_clusters = max(2, X_norm.shape[0] // MIN_PATTERN_OCCURRENCES)
        n_clusters = min(n_clusters, 20)  # cap at 20 clusters
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_norm)

    # ── Step 5: Analyze each cluster ──
    unique_labels = set(labels) - {-1}  # discard noise
    patterns = []

    for cid in sorted(unique_labels):
        pattern = _analyze_cluster(
            moments=moments,
            valid_indices=valid_indices,
            labels=labels,
            cluster_id=cid,
            X=X_norm,
            X_raw=X_raw,
            scaler=scaler,
        )
        if pattern is not None:
            patterns.append(pattern)

    # ── Step 6: Sort by expected_r descending ──
    patterns.sort(key=lambda p: p["expected_r"], reverse=True)

    noise_count = int(np.sum(labels == -1))
    logger.info(
        f"Mining complete for {pair}: "
        f"{len(patterns)} patterns from {len(unique_labels)} clusters "
        f"({noise_count} noise points discarded)"
    )
    for p in patterns:
        logger.info(
            f"  Pattern {p['cluster_id']}: {p['direction']} "
            f"{p['occurrences']}x, WR={p['win_rate']:.1%}, "
            f"PF={p['profit_factor']:.2f}, E[R]={p['expected_r']:.2f} "
            f"| top: {p['top_features'][0][0] if p['top_features'] else 'N/A'}"
        )

    return patterns


# ── Mine All Pairs ───────────────────────────────────────────

def mine_all_pairs(scan_ids: list = None) -> dict:
    """
    Mine patterns for all pairs in PAIR_WHITELIST.

    Parameters
    ----------
    scan_ids : list or None
        Optional list of scan IDs to filter on.

    Returns
    -------
    dict
        {
            'total_patterns': int,
            'per_pair': {
                'EURUSD': [pattern, pattern, ...],
                'GBPJPY': [pattern, pattern, ...],
            }
        }
    """
    logger.info(f"{'#'*60}")
    logger.info(f"MINING ALL PAIRS — {len(PAIR_WHITELIST)} pairs")
    logger.info(f"{'#'*60}")

    per_pair = {}
    total_patterns = 0

    for pair in PAIR_WHITELIST:
        try:
            patterns = mine_patterns(pair, scan_ids)
            per_pair[pair] = patterns
            total_patterns += len(patterns)
        except Exception as e:
            logger.error(f"Error mining {pair}: {e}")
            per_pair[pair] = []

    result = {
        "total_patterns": total_patterns,
        "per_pair": per_pair,
    }

    logger.info(f"{'#'*60}")
    logger.info(
        f"MINING COMPLETE: {total_patterns} total patterns across "
        f"{sum(1 for v in per_pair.values() if v)} pairs"
    )
    for pair, pats in per_pair.items():
        if pats:
            best = pats[0]
            logger.info(
                f"  {pair}: {len(pats)} patterns | "
                f"best: WR={best['win_rate']:.1%}, "
                f"PF={best['profit_factor']:.2f}, "
                f"E[R]={best['expected_r']:.2f}"
            )
    logger.info(f"{'#'*60}")

    return result


# ── Utility: Pattern Summary ─────────────────────────────────

def summarize_patterns(patterns: list) -> str:
    """
    Return a human-readable summary of mined patterns.

    Parameters
    ----------
    patterns : list[dict]
        List of pattern dicts from mine_patterns().

    Returns
    -------
    str
        Formatted summary string.
    """
    if not patterns:
        return "No patterns found."

    lines = [f"{'='*60}"]
    lines.append(f"  PATTERN SUMMARY — {patterns[0]['pair']}")
    lines.append(f"  {len(patterns)} patterns discovered")
    lines.append(f"{'='*60}")

    for i, p in enumerate(patterns, 1):
        lines.append(f"")
        lines.append(f"  Pattern #{p['cluster_id']} ({i})")
        lines.append(f"  {'─'*40}")
        lines.append(f"  Direction:      {p['direction']} "
                      f"({p['direction_purity']:.0%} purity)")
        lines.append(f"  Occurrences:    {p['occurrences']}")
        lines.append(f"  Win Rate:       {p['win_rate']:.1%} "
                      f"({p['wins']}W / {p['losses']}L)")
        lines.append(f"  Avg Profit:     {p['avg_profit_pips']:.1f} pips")
        lines.append(f"  Profit Factor:  {p['profit_factor']:.2f}")
        lines.append(f"  Expected R:     {p['expected_r']:.2f}")
        if p["top_features"]:
            lines.append(f"  Top Features:")
            for feat, zscore in p["top_features"][:5]:
                fr = p["feature_ranges"].get(feat, {})
                lines.append(
                    f"    {feat:>25s}: z={zscore:>5.2f}  "
                    f"mean={fr.get('mean', 0):>8.4f}  "
                    f"[{fr.get('min', 0):>8.4f}, {fr.get('max', 0):>8.4f}]"
                )

    lines.append(f"")
    lines.append(f"{'='*60}")
    return "\n".join(lines)
