# =============================================================
# rpde/pattern_validator.py
# RPDE Pattern Validator — Strict statistical validation for
# discovered pattern candidates.
#
# Each pair gets its own validation. Patterns are NOT required
# to work across pairs — pair personality matters.
# =============================================================

import math
import numpy as np
from datetime import datetime

from rpde.config import (
    MIN_PATTERN_OCCURRENCES,
    MIN_PATTERN_WIN_RATE,
    MIN_PATTERN_PROFIT_FACTOR,
    MIN_BACKTEST_DAYS,
    PATTERN_TIERS,
    CURRENCY_CORRELATION_THRESHOLD,
    CLUSTER_FEATURES,
)
from core.logger import get_logger

log = get_logger(__name__)

TIER_ORDER = {"GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1}
MAX_CONSECUTIVE_LOSSES_HARD = 10


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


def validate_pattern(pattern: dict, pair: str, all_moments: list,
                     all_pair_patterns: dict = None) -> dict:
    """Validate a single pattern candidate against historical data."""
    rejection_reasons = []

    member_indices = pattern.get("member_indices", pattern.get("golden_moments", []))
    if not member_indices:
        rejection_reasons.append("No member moments in pattern")
        return {"valid": False, "tier": None, "statistics": _empty_stats(),
                "currency_tag": "PAIR_ONLY", "currency_boost_pairs": [],
                "rejection_reasons": rejection_reasons}

    # Support both index-based and direct list of moments
    if isinstance(member_indices, list) and len(member_indices) > 0:
        if isinstance(member_indices[0], dict):
            member_moments = member_indices  # Already a list of moment dicts
        else:
            valid_idx = [i for i in member_indices if 0 <= i < len(all_moments)]
            member_moments = [all_moments[i] for i in valid_idx] if valid_idx else []
    else:
        member_moments = []

    occurrences = len(member_moments)
    if occurrences < MIN_PATTERN_OCCURRENCES:
        rejection_reasons.append(f"Insufficient occurrences: {occurrences} < {MIN_PATTERN_OCCURRENCES}")

    outcomes, returns, profits, losses, timestamps = [], [], [], [], []
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
            losses.append(abs(pips))
        ts = m.get("bar_timestamp")
        if ts is not None:
            timestamps.append(ts)

    wins = sum(1 for o in outcomes if o)
    win_rate = wins / occurrences if occurrences > 0 else 0.0
    avg_profit = float(np.mean(profits)) if profits else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    total_profit_r = sum(r for r in returns if r > 0)
    total_loss_r = sum(abs(r) for r in returns if r < 0)
    profit_factor = total_profit_r / total_loss_r if total_loss_r > 0 else (999.0 if total_profit_r > 0 else 0.0)
    expected_r = float(np.mean(returns)) if returns else 0.0

    if win_rate < MIN_PATTERN_WIN_RATE:
        rejection_reasons.append(f"Win rate too low: {win_rate:.1%} < {MIN_PATTERN_WIN_RATE:.0%}")
    if profit_factor < MIN_PATTERN_PROFIT_FACTOR:
        rejection_reasons.append(f"Profit factor too low: {profit_factor:.2f} < {MIN_PATTERN_PROFIT_FACTOR}")

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

    log.info(f"[RPDE_VAL] {pair} cluster={pattern.get('cluster_id', '?')} occ={occurrences} "
             f"wr={win_rate:.1%} pf={profit_factor:.2f} tier={tier or 'NONE'} currency={currency_tag}")

    return {"valid": valid, "tier": tier, "statistics": stats,
            "currency_tag": currency_tag, "currency_boost_pairs": currency_boost,
            "rejection_reasons": rejection_reasons}


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
