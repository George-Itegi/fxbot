# =============================================================
# rpde/pattern_validator.py
# RPDE Pattern Validator — Strict statistical validation for
# discovered pattern candidates.
#
# Each pair gets its own validation. Patterns are NOT required
# to work across pairs — pair personality matters.
#
# NEGATIVE SAMPLING: Instead of only evaluating golden-moment
# members (which are all wins by definition, giving PF=999),
# we now search ALL historical bars for moments with similar
# features and compute REAL win rate and profit factor.
# =============================================================

import math
import numpy as np
from datetime import datetime

from rpde.config import (
    MIN_PATTERN_OCCURRENCES,
    MIN_PATTERN_WIN_RATE,
    MIN_PATTERN_PROFIT_FACTOR,
    MIN_BACKTEST_DAYS,
    MAX_AVG_MAE_MOVE_RATIO,
    PATTERN_TIERS,
    CURRENCY_CORRELATION_THRESHOLD,
    CLUSTER_FEATURES,
    NEGATIVE_SEARCH_COSINE_THRESHOLD,
)
from core.logger import get_logger

log = get_logger(__name__)

TIER_ORDER = {"GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1}
MAX_CONSECUTIVE_LOSSES_HARD = 10

# Spread cost threshold (pips) — a forward_return must exceed
# this to count as a "win" in the real-statistics calculation.
# Most major pairs have 0.3–1.0 pip spread at M5; 0.5 is a
# conservative middle ground.  Pairs with wider spreads can
# override via PAIR_SPREAD_PIPS below.
DEFAULT_SPREAD_COST_PIPS = 0.5

PAIR_SPREAD_PIPS = {
    "XAGUSD": 3.0,
    "XAUUSD": 3.0,
    "GBPJPY": 1.5,
    "CADJPY": 1.0,
    "AUDJPY": 1.2,
    "CHFJPY": 1.0,
    "AUDCAD": 1.2,
}


# ── Helper: cosine similarity ────────────────────────────

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors. Handles zero vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ── Helper: extract feature vector from a moment ──────────

def _extract_feature_vector(moment: dict) -> np.ndarray:
    """Extract a normalized feature vector from a golden moment for comparison."""
    features = moment.get('features') or moment.get('feature_snapshot') or {}
    if not features:
        return np.array([])

    # Use the CLUSTER_FEATURES list (26 features used for clustering)
    vec = []
    for fname in CLUSTER_FEATURES:
        val = features.get(fname)
        if val is None:
            return np.array([])  # Incomplete feature — can't compare
        vec.append(float(val))

    return np.array(vec, dtype=np.float64)


# ── Existing helper functions (unchanged) ────────────────

def compute_sharpe(returns: list, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from R-multiple returns."""
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std_r = np.std(arr, ddof=1)
    if std_r < 1e-9:
        return 0.0
    trades_per_year = 250 * 5
    rf_per_trade = risk_free_rate / trades_per_year
    return round(float((np.mean(arr) - rf_per_trade) / std_r * np.sqrt(trades_per_year)), 4)


def compute_max_consecutive_losses(outcomes: list) -> int:
    """Longest streak of consecutive losses."""
    if not outcomes:
        return 0
    max_streak = current = 0
    for o in outcomes:
        is_loss = (not o) if isinstance(o, bool) else (o <= 0)
        if is_loss:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_max_drawdown(returns: list) -> float:
    """Max drawdown in R-multiples from cumulative P&L curve."""
    if not returns:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return round(float(np.max(drawdown)) if len(drawdown) > 0 else 0.0, 4)


def assign_tier(stats: dict) -> str:
    """Assign confidence tier from strictest to most lenient."""
    for tier_name in ["GOD_TIER", "STRONG", "VALID", "PROBATIONARY"]:
        req = PATTERN_TIERS.get(tier_name)
        if not req:
            continue
        if (stats.get("occurrences", 0) >= req["min_occurrences"]
                and stats.get("win_rate", 0) >= req["min_win_rate"]
                and stats.get("profit_factor", 0) >= req["min_profit_factor"]
                and stats.get("backtest_days", 0) >= req["min_backtest_days"]):
            return tier_name
    return None


def _extract_base_currency(pair: str) -> str:
    return pair[:3] if pair and len(pair) >= 3 else ""


def _extract_quote_currency(pair: str) -> str:
    return pair[3:6] if pair and len(pair) >= 6 else ""


def _find_related_pairs(pair: str, all_pairs: list) -> list:
    """Find pairs sharing same base or quote currency."""
    base = _extract_base_currency(pair)
    quote = _extract_quote_currency(pair)
    return [p for p in all_pairs if p != pair and
            (_extract_base_currency(p) == base or _extract_quote_currency(p) == quote)]


def compute_currency_tag(pair: str, pattern_features: dict,
                          all_pair_patterns: dict) -> tuple:
    """
    Determine if this pattern is currency-specific.
    Returns (currency_tag, confirming_pairs).
    """
    if not pattern_features or not all_pair_patterns:
        return ("PAIR_ONLY", [])

    base = _extract_base_currency(pair)
    quote = _extract_quote_currency(pair)
    all_pairs = list(all_pair_patterns.keys())
    related = _find_related_pairs(pair, all_pairs)

    confirming = []
    for rp in related:
        for pat in all_pair_patterns.get(rp, []):
            rf = pat.get("cluster_center") or pat.get("cluster_center_json")
            if isinstance(rf, str):
                try:
                    import json
                    rf = json.loads(rf)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not rf or not isinstance(rf, dict):
                continue

            shared = set(pattern_features.keys()) & set(rf.keys())
            shared = {k for k in shared
                      if isinstance(pattern_features.get(k), (int, float))
                      and isinstance(rf.get(k), (int, float))
                      and not (math.isnan(pattern_features[k]) or math.isinf(pattern_features[k]))
                      and not (math.isnan(rf[k]) or math.isinf(rf[k]))}
            if len(shared) < 5:
                continue

            va = np.array([pattern_features[k] for k in shared], dtype=np.float64)
            vb = np.array([rf[k] for k in shared], dtype=np.float64)
            if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
                continue
            corr = float(np.corrcoef(va, vb)[0, 1])
            if np.isnan(corr) or np.isinf(corr):
                continue
            if corr >= CURRENCY_CORRELATION_THRESHOLD:
                confirming.append((rp, corr))

    if len(confirming) >= 2:
        tag = f"{base}_SPECIFIC" if base else "PAIR_ONLY"
        log.info(f"[RPDE_VAL] Currency tag '{tag}' for {pair}: {len(confirming)} confirming pairs")
        return (tag, confirming)

    return ("PAIR_ONLY", [])


def _empty_stats() -> dict:
    return {"occurrences": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "avg_profit_pips": 0.0, "avg_loss_pips": 0.0, "profit_factor": 0.0,
            "expected_r": 0.0, "max_drawdown_pips": 0.0,
            "max_consecutive_losses": 0, "sharpe_ratio": 0.0,
            "backtest_start": None, "backtest_end": None, "backtest_days": 0}


# ── MAIN: validate_pattern with NEGATIVE SAMPLING ────────

def validate_pattern(pattern: dict, pair: str, all_moments: list,
                     all_pair_patterns: dict = None) -> dict:
    """Validate a single pattern candidate against historical data.

    Uses NEGATIVE SAMPLING: searches all historical bars for moments
    with features similar to this pattern's centroid.  Win rate and
    profit factor are computed from the COMBINED set of members +
    similar non-members, giving a realistic out-of-sample estimate
    instead of the inflated PF=999 from member-only evaluation.
    """
    rejection_reasons = []

    # ── 1. Extract member moments (same logic as before) ────
    member_indices = pattern.get("member_indices", pattern.get("golden_moments", []))
    if not member_indices:
        rejection_reasons.append("No member moments in pattern")
        return {"valid": False, "tier": None, "statistics": _empty_stats(),
                "currency_tag": "PAIR_ONLY", "currency_boost_pairs": [],
                "rejection_reasons": rejection_reasons}

    if isinstance(member_indices, list) and len(member_indices) > 0:
        if isinstance(member_indices[0], dict):
            member_moments = member_indices
        else:
            is_db_ids = any(i >= len(all_moments) for i in member_indices if isinstance(i, int))
            if is_db_ids and all_moments:
                id_to_moment = {m.get("id"): m for m in all_moments if m.get("id") is not None}
                member_moments = [id_to_moment[mid] for mid in member_indices
                                 if mid in id_to_moment]
            else:
                valid_idx = [i for i in member_indices if isinstance(i, int) and 0 <= i < len(all_moments)]
                member_moments = [all_moments[i] for i in valid_idx] if valid_idx else []
    else:
        member_moments = []

    member_count = len(member_moments)
    if member_count < MIN_PATTERN_OCCURRENCES:
        rejection_reasons.append(f"Insufficient occurrences: {member_count} < {MIN_PATTERN_OCCURRENCES}")

    # ── 2. Compute pattern centroid from member feature vectors ──
    member_feature_vecs = []
    for m in member_moments:
        vec = _extract_feature_vector(m)
        if vec.size == len(CLUSTER_FEATURES):
            member_feature_vecs.append(vec)

    if not member_feature_vecs:
        # No complete feature vectors — can't do negative sampling
        log.warning(f"[RPDE_VAL] {pair} cluster={pattern.get('cluster_id', '?')}: "
                     f"no complete feature vectors in members, falling back to member-only stats")
        return _validate_member_only(pattern, pair, member_moments, member_count,
                                     all_pair_patterns, rejection_reasons)

    # Centroid = mean of member feature vectors
    centroid = np.mean(member_feature_vecs, axis=0)

    # ── 3. Build set of member IDs for O(1) lookup ─────────
    member_ids = set()
    for m in member_moments:
        mid = m.get("id")
        if mid is not None:
            member_ids.add(mid)
    # Also add bar timestamps as a secondary identifier
    member_timestamps = set()
    for m in member_moments:
        ts = m.get("bar_timestamp")
        if ts is not None:
            member_timestamps.add(ts)

    def _is_member(moment: dict) -> bool:
        """Check if a moment is already a member of this pattern."""
        mid = moment.get("id")
        if mid is not None and mid in member_ids:
            return True
        ts = moment.get("bar_timestamp")
        if ts is not None and ts in member_timestamps:
            return True
        return False

    # ── 4. Search for similar non-member moments ───────────
    # PRIORITY: Use TRUE negative samples from DB (bars that passed
    # regime filter but did NOT produce a big move). These have
    # direction='NONE' and near-zero forward_returns.
    # FALLBACK: Search all_moments for similar non-members (old approach,
    # but this is broken since all_moments are all winners).
    spread_cost = PAIR_SPREAD_PIPS.get(pair, DEFAULT_SPREAD_COST_PIPS)

    true_negatives = []
    try:
        from rpde.database import load_negative_samples
        true_negatives = load_negative_samples(pair=pair)
    except Exception:
        pass

    similar_non_members = []
    use_true_negatives = False

    if true_negatives:
        # Filter true negatives by cosine similarity to pattern centroid
        # Keep only those with similarity > threshold (similar features)
        for neg in true_negatives:
            vec = _extract_feature_vector(neg)
            if vec.size != len(CLUSTER_FEATURES):
                continue
            sim = _cosine_similarity(vec, centroid)
            if sim > NEGATIVE_SEARCH_COSINE_THRESHOLD:
                similar_non_members.append({
                    "moment": neg,
                    "similarity": sim,
                })
        use_true_negatives = True

    # Fallback: if no true negatives found with sufficient similarity,
    # try the old approach (searching all_moments for non-members)
    if not similar_non_members and all_moments:
        for moment in all_moments:
            if _is_member(moment):
                continue
            vec = _extract_feature_vector(moment)
            if vec.size != len(CLUSTER_FEATURES):
                continue
            sim = _cosine_similarity(vec, centroid)
            if sim > NEGATIVE_SEARCH_COSINE_THRESHOLD:
                similar_non_members.append({
                    "moment": moment,
                    "similarity": sim,
                })

    non_member_count = len(similar_non_members)

    # ── 5. Fall back to member-only if no negatives found ──
    if non_member_count == 0:
        log.info(f"[RPDE_VAL] {pair} cluster={pattern.get('cluster_id', '?')}: "
                 f"no similar non-members found (threshold={NEGATIVE_SEARCH_COSINE_THRESHOLD}), "
                 f"falling back to member-only stats")
        return _validate_member_only(pattern, pair, member_moments, member_count,
                                     all_pair_patterns, rejection_reasons)

    # ── 6. Compute REAL statistics from combined set ───────
    # Collect forward_returns for ALL moments (members + similar non-members)
    combined_returns = []      # forward_return in pips
    combined_outcomes = []     # True/False based on forward_return vs spread
    combined_timestamps = []
    combined_mae_values = []
    combined_mfe_after_mae_values = []
    combined_move_pips = []

    # Process member moments
    for m in member_moments:
        fr = float(m.get("forward_return", 0.0))
        combined_returns.append(fr)
        combined_outcomes.append(fr > spread_cost)
        ts = m.get("bar_timestamp")
        if ts is not None:
            combined_timestamps.append(ts)
        combined_mae_values.append(float(m.get("mae_pips", 0.0)))
        combined_mfe_after_mae_values.append(float(m.get("mfe_after_mae", 0.0)))
        combined_move_pips.append(float(m.get("move_pips", 0.0)))

    # Process similar non-member moments
    for entry in similar_non_members:
        m = entry["moment"]
        fr = float(m.get("forward_return", 0.0))
        combined_returns.append(fr)
        combined_outcomes.append(fr > spread_cost)
        ts = m.get("bar_timestamp")
        if ts is not None:
            combined_timestamps.append(ts)
        combined_mae_values.append(float(m.get("mae_pips", 0.0)))
        combined_mfe_after_mae_values.append(float(m.get("mfe_after_mae", 0.0)))
        combined_move_pips.append(float(m.get("move_pips", 0.0)))

    total = len(combined_returns)
    wins = sum(1 for o in combined_outcomes if o)
    losses = total - wins
    win_rate = wins / total if total > 0 else 0.0

    # Profit / loss sums in pips using forward_return
    positive_returns = [r for r in combined_returns if r > 0]
    negative_returns = [r for r in combined_returns if r < 0]
    total_profit = sum(positive_returns)
    total_loss = abs(sum(negative_returns))
    profit_factor = total_profit / total_loss if total_loss > 0 else (999.0 if total_profit > 0 else 0.0)

    avg_profit = float(np.mean(positive_returns)) if positive_returns else 0.0
    avg_loss = float(np.mean(negative_returns)) if negative_returns else 0.0
    expected_r = float(np.mean(combined_returns)) if combined_returns else 0.0

    # MAE statistics across the combined set
    avg_mae = float(np.mean(combined_mae_values)) if combined_mae_values else 0.0
    max_mae = float(np.max(combined_mae_values)) if combined_mae_values else 0.0
    p90_mae = float(np.percentile(combined_mae_values, 90)) if len(combined_mae_values) >= 5 else max_mae
    avg_mfe_after_mae = float(np.mean(combined_mfe_after_mae_values)) if combined_mfe_after_mae_values else 0.0
    mae_reward_ratio = avg_mfe_after_mae / avg_mae if avg_mae > 0 else 0.0

    # MAE quality check — across combined data
    avg_move_pips_val = float(np.mean(combined_move_pips)) if combined_move_pips else 0.0
    if avg_mae > 0 and avg_move_pips_val > 0:
        mae_move_ratio = avg_mae / avg_move_pips_val
        if mae_move_ratio > MAX_AVG_MAE_MOVE_RATIO:
            rejection_reasons.append(
                f"MAE too deep: {mae_move_ratio:.0%} of move "
                f"(avg MAE={avg_mae:.1f}pips, avg move={avg_move_pips_val:.1f}pips)"
            )

    # Backtest date range from combined timestamps
    backtest_start = backtest_end = None
    backtest_days = 0
    parsed_ts = []
    for ts in combined_timestamps:
        if isinstance(ts, datetime):
            parsed_ts.append(ts)
        elif isinstance(ts, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    parsed_ts.append(datetime.strptime(ts, fmt))
                    break
                except ValueError:
                    continue
    if len(parsed_ts) >= 2:
        parsed_ts.sort()
        backtest_start, backtest_end = parsed_ts[0], parsed_ts[-1]
        backtest_days = (backtest_end - backtest_start).days

    if backtest_days < MIN_BACKTEST_DAYS:
        rejection_reasons.append(f"Backtest too short: {backtest_days}d < {MIN_BACKTEST_DAYS}d")

    max_consec = compute_max_consecutive_losses(combined_outcomes)
    if max_consec >= MAX_CONSECUTIVE_LOSSES_HARD:
        rejection_reasons.append(f"Max consecutive losses: {max_consec}")

    if win_rate < MIN_PATTERN_WIN_RATE:
        rejection_reasons.append(f"Win rate too low: {win_rate:.1%} < {MIN_PATTERN_WIN_RATE:.0%}")
    if profit_factor < MIN_PATTERN_PROFIT_FACTOR:
        rejection_reasons.append(f"Profit factor too low: {profit_factor:.2f} < {MIN_PATTERN_PROFIT_FACTOR}")

    stats = {
        "occurrences": total, "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 6), "avg_profit_pips": round(avg_profit, 2),
        "avg_loss_pips": round(avg_loss, 2), "profit_factor": round(profit_factor, 4),
        "expected_r": round(expected_r, 4), "max_drawdown_pips": round(compute_max_drawdown(combined_returns), 2),
        "max_consecutive_losses": max_consec, "sharpe_ratio": compute_sharpe(combined_returns),
        "backtest_start": backtest_start, "backtest_end": backtest_end, "backtest_days": backtest_days,
        # MAE (Maximum Adverse Excursion) — stop loss intelligence
        "avg_mae_pips": round(avg_mae, 2),
        "max_mae_pips": round(max_mae, 2),
        "p90_mae_pips": round(p90_mae, 2),
        "avg_mfe_after_mae_pips": round(avg_mfe_after_mae, 2),
        "mae_reward_ratio": round(mae_reward_ratio, 4),
        # Negative sampling metadata
        "member_count": member_count,
        "similar_non_member_count": non_member_count,
        "spread_cost_pips": spread_cost,
    }

    tier = assign_tier(stats)
    valid = tier is not None
    if not valid and not rejection_reasons:
        rejection_reasons.append("Below all tier minimums")

    currency_tag = "PAIR_ONLY"
    currency_boost = []
    if all_pair_patterns and valid:
        pf = pattern.get("cluster_center") or {}
        if isinstance(pf, (list, np.ndarray)):
            if len(pf) == len(CLUSTER_FEATURES):
                pf = dict(zip(CLUSTER_FEATURES, pf))
            else:
                pf = {}
        currency_tag, currency_boost = compute_currency_tag(pair, pf, all_pair_patterns)

    log.info(f"[RPDE_VAL] {pair} cluster={pattern.get('cluster_id', '?')} "
             f"occ={total} wr={win_rate:.1%} pf={profit_factor:.2f} tier={tier or 'NONE'} "
             f"(members={member_count}, similar_non_members={non_member_count}, "
             f"negatives={'TRUE' if use_true_negatives else 'FALLBACK'})")

    return {"valid": valid, "tier": tier, "statistics": stats,
            "currency_tag": currency_tag, "currency_boost_pairs": currency_boost,
            "rejection_reasons": rejection_reasons}


def _validate_member_only(pattern: dict, pair: str, member_moments: list,
                          member_count: int, all_pair_patterns: dict,
                          rejection_reasons: list) -> dict:
    """Fall-back: validate using members only (original behaviour).

    Used when negative sampling can't be performed — e.g. no complete
    feature vectors on members, or no similar non-members found.
    """
    occurrences = member_count
    if occurrences < MIN_PATTERN_OCCURRENCES and f"Insufficient occurrences" not in str(rejection_reasons):
        rejection_reasons.append(f"Insufficient occurrences: {occurrences} < {MIN_PATTERN_OCCURRENCES}")

    outcomes, returns, profits, losses_list, timestamps = [], [], [], [], []
    mae_values, mfe_after_mae_values = [], []
    for m in member_moments:
        is_win = m.get("is_win", False)
        if isinstance(is_win, int):
            is_win = bool(is_win)
        outcomes.append(is_win)
        r = float(m.get("forward_return", 0.0))
        returns.append(r)
        pips = float(m.get("move_pips", 0.0))
        if is_win:
            profits.append(pips)
        else:
            losses_list.append(abs(pips))
        ts = m.get("bar_timestamp")
        if ts is not None:
            timestamps.append(ts)

        mae_pips = float(m.get("mae_pips", 0.0))
        mae_values.append(mae_pips)
        mfe_after_mae = float(m.get("mfe_after_mae", 0.0))
        mfe_after_mae_values.append(mfe_after_mae)

    wins = sum(1 for o in outcomes if o)
    win_rate = wins / occurrences if occurrences > 0 else 0.0
    avg_profit = float(np.mean(profits)) if profits else 0.0
    avg_loss = float(np.mean(losses_list)) if losses_list else 0.0
    total_profit_r = sum(r for r in returns if r > 0)
    total_loss_r = sum(abs(r) for r in returns if r < 0)
    profit_factor = total_profit_r / total_loss_r if total_loss_r > 0 else (999.0 if total_profit_r > 0 else 0.0)
    expected_r = float(np.mean(returns)) if returns else 0.0

    if win_rate < MIN_PATTERN_WIN_RATE:
        rejection_reasons.append(f"Win rate too low: {win_rate:.1%} < {MIN_PATTERN_WIN_RATE:.0%}")
    if profit_factor < MIN_PATTERN_PROFIT_FACTOR:
        rejection_reasons.append(f"Profit factor too low: {profit_factor:.2f} < {MIN_PATTERN_PROFIT_FACTOR}")

    avg_mae = float(np.mean(mae_values)) if mae_values else 0.0
    max_mae = float(np.max(mae_values)) if mae_values else 0.0
    p90_mae = float(np.percentile(mae_values, 90)) if len(mae_values) >= 5 else max_mae
    avg_mfe_after_mae = float(np.mean(mfe_after_mae_values)) if mfe_after_mae_values else 0.0
    mae_reward_ratio = avg_mfe_after_mae / avg_mae if avg_mae > 0 else 0.0

    avg_move_pips_val = float(np.mean([m.get("move_pips", 0) for m in member_moments])) if member_moments else 0.0
    if avg_mae > 0 and avg_move_pips_val > 0:
        mae_move_ratio = avg_mae / avg_move_pips_val
        if mae_move_ratio > MAX_AVG_MAE_MOVE_RATIO:
            rejection_reasons.append(
                f"MAE too deep: {mae_move_ratio:.0%} of move "
                f"(avg MAE={avg_mae:.1f}pips, avg move={avg_move_pips_val:.1f}pips)"
            )

    backtest_start = backtest_end = None
    backtest_days = 0
    parsed_ts = []
    for ts in timestamps:
        if isinstance(ts, datetime):
            parsed_ts.append(ts)
        elif isinstance(ts, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                try:
                    parsed_ts.append(datetime.strptime(ts, fmt))
                    break
                except ValueError:
                    continue
    if len(parsed_ts) >= 2:
        parsed_ts.sort()
        backtest_start, backtest_end = parsed_ts[0], parsed_ts[-1]
        backtest_days = (backtest_end - backtest_start).days

    if backtest_days < MIN_BACKTEST_DAYS:
        rejection_reasons.append(f"Backtest too short: {backtest_days}d < {MIN_BACKTEST_DAYS}d")

    max_consec = compute_max_consecutive_losses(outcomes)
    if max_consec >= MAX_CONSECUTIVE_LOSSES_HARD:
        rejection_reasons.append(f"Max consecutive losses: {max_consec}")

    stats = {
        "occurrences": occurrences, "wins": wins, "losses": occurrences - wins,
        "win_rate": round(win_rate, 6), "avg_profit_pips": round(avg_profit, 2),
        "avg_loss_pips": round(avg_loss, 2), "profit_factor": round(profit_factor, 4),
        "expected_r": round(expected_r, 4), "max_drawdown_pips": round(compute_max_drawdown(returns), 2),
        "max_consecutive_losses": max_consec, "sharpe_ratio": compute_sharpe(returns),
        "backtest_start": backtest_start, "backtest_end": backtest_end, "backtest_days": backtest_days,
        "avg_mae_pips": round(avg_mae, 2),
        "max_mae_pips": round(max_mae, 2),
        "p90_mae_pips": round(p90_mae, 2),
        "avg_mfe_after_mae_pips": round(avg_mfe_after_mae, 2),
        "mae_reward_ratio": round(mae_reward_ratio, 4),
        # Negative sampling metadata (member-only fallback)
        "member_count": member_count,
        "similar_non_member_count": 0,
        "spread_cost_pips": PAIR_SPREAD_PIPS.get(pair, DEFAULT_SPREAD_COST_PIPS),
    }

    tier = assign_tier(stats)
    valid = tier is not None
    if not valid and not rejection_reasons:
        rejection_reasons.append("Below all tier minimums")

    currency_tag = "PAIR_ONLY"
    currency_boost = []
    if all_pair_patterns and valid:
        pf = pattern.get("cluster_center") or {}
        if isinstance(pf, (list, np.ndarray)):
            if len(pf) == len(CLUSTER_FEATURES):
                pf = dict(zip(CLUSTER_FEATURES, pf))
            else:
                pf = {}
        currency_tag, currency_boost = compute_currency_tag(pair, pf, all_pair_patterns)

    log.info(f"[RPDE_VAL] {pair} cluster={pattern.get('cluster_id', '?')} "
             f"occ={occurrences} wr={win_rate:.1%} pf={profit_factor:.2f} tier={tier or 'NONE'} "
             f"(members={member_count}, similar_non_members=0, FALLBACK)")

    return {"valid": valid, "tier": tier, "statistics": stats,
            "currency_tag": currency_tag, "currency_boost_pairs": currency_boost,
            "rejection_reasons": rejection_reasons}


# ── Batch validator (unchanged structure) ────────────────

def validate_all_patterns(pair: str, candidates: list, all_moments: list,
                          all_pair_patterns: dict = None) -> list:
    """Validate all candidates for a pair. Returns valid patterns sorted by tier."""
    if not candidates:
        return []

    validated = []
    for candidate in candidates:
        try:
            result = validate_pattern(candidate, pair, all_moments, all_pair_patterns)
            validated.append({"pattern": candidate, "validation": result})
        except Exception as e:
            log.error(f"[RPDE_VAL] Failed to validate cluster={candidate.get('cluster_id', '?')}: {e}")

    valid = [v for v in validated if v["validation"]["valid"]]
    valid.sort(key=lambda x: (TIER_ORDER.get(x["validation"]["tier"], 0),
                             x["validation"]["statistics"].get("profit_factor", 0)), reverse=True)

    log.info(f"[RPDE_VAL] {pair}: {len(valid)}/{len(candidates)} patterns validated")
    return valid
