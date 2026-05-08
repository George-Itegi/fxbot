# =============================================================
# rpde/pattern_library.py
# RPDE Pattern Library — Persistent store of all discovered,
# validated patterns per pair.
# =============================================================

import json
import os
import numpy as np
from datetime import datetime

from rpde.config import (
    MAX_PATTERNS_PER_PAIR, GATE_MIN_CONFIDENCE,
    CLUSTER_FEATURES, CURRENCY_CONFIRM_BOOST,
)
from rpde import database as rpde_db
from core.logger import get_logger

log = get_logger(__name__)

_cache = {}
_cache_loaded = False


def build_pattern_id(pair: str, cluster_id: int) -> str:
    return f"{pair}_P{cluster_id}"


def save_pattern(pattern: dict, stats: dict, validation: dict) -> str:
    """Save a validated pattern to the library (DB + cache)."""
    pair = pattern.get("pair", "")
    cid = pattern.get("cluster_id", 0)
    pattern_id = build_pattern_id(pair, cid)

    center = pattern.get("cluster_center")
    if isinstance(center, np.ndarray):
        center = center.tolist()
    center_json = json.dumps(center, default=str) if center else None

    ranges = pattern.get("feature_ranges")
    ranges_json = json.dumps(ranges, default=str) if ranges else None

    top_feat = pattern.get("top_features", [])
    top_json = json.dumps(top_feat, default=str) if top_feat else None

    cb = validation.get("currency_boost_pairs", [])
    if cb and isinstance(cb, list):
        cb = [{"pair": c, "correlation": round(float(r), 4)} for c, r in cb]

    pattern_dict = {
        "pattern_id": pattern_id, "pair": pair,
        "direction": pattern.get("direction", ""),
        "cluster_id": cid,
        "tier": validation.get("tier", "PROBATIONARY"),
        "occurrences": stats.get("occurrences", 0),
        "wins": stats.get("wins", 0),
        "losses": stats.get("losses", 0),
        "win_rate": stats.get("win_rate", 0.0),
        "avg_profit_pips": stats.get("avg_profit_pips", 0.0),
        "avg_loss_pips": stats.get("avg_loss_pips", 0.0),
        "profit_factor": stats.get("profit_factor", 0.0),
        "avg_expected_r": stats.get("expected_r", 0.0),
        "max_drawdown_pips": stats.get("max_drawdown_pips", 0.0),
        "max_consecutive_losses": stats.get("max_consecutive_losses", 0),
        "sharpe_ratio": stats.get("sharpe_ratio", 0.0),
        "backtest_start": stats.get("backtest_start"),
        "backtest_end": stats.get("backtest_end"),
        "backtest_days": stats.get("backtest_days", 0),
        "currency_tag": validation.get("currency_tag", "PAIR_ONLY"),
        "currency_boost_pairs": cb,
        "cluster_center_json": center_json,
        "feature_ranges_json": ranges_json,
        "top_features_json": top_json,
        "is_active": True,
        "last_validated": datetime.now(),
    }

    rpde_db.store_pattern(pattern_dict)
    _invalidate_cache(pair)
    log.info(f"[RPDE_LIB] Saved {pattern_id} tier={pattern_dict['tier']} "
             f"wr={stats.get('win_rate', 0):.1%} occ={stats.get('occurrences', 0)}")
    return pattern_id


def _invalidate_cache(pair: str = None):
    global _cache, _cache_loaded
    if pair:
        _cache.pop(pair, None)
    else:
        _cache = {}
        _cache_loaded = False


def load_patterns(pair: str = None, active_only: bool = True,
                  min_tier: str = None) -> dict:
    """Load patterns from DB with optional caching."""
    global _cache, _cache_loaded

    tier_levels = {"PROBATIONARY": 0, "VALID": 1, "STRONG": 2, "GOD_TIER": 3}

    if _cache_loaded and pair and pair in _cache:
        pats = _cache[pair]
        if active_only:
            pats = [p for p in pats if p.get("is_active", 1) == 1]
        if min_tier:
            ml = tier_levels.get(min_tier, 0)
            pats = [p for p in pats if tier_levels.get(p.get("tier"), 0) >= ml]
        return {pair: pats}

    rows = rpde_db.load_pattern_library(pair=pair, active_only=active_only, min_tier=min_tier)
    for row in rows:
        p = row.get("pair")
        if p:
            _cache.setdefault(p, []).append(row)
    _cache_loaded = True

    if pair:
        return {pair: _cache.get(pair, [])}
    return {k: v for k, v in _cache.items()}


def get_active_patterns_for_pair(pair: str) -> list:
    """Get active patterns for a pair, sorted by tier then win_rate."""
    patterns = load_patterns(pair=pair, active_only=True).get(pair, [])
    tier_order = {"GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1}
    patterns.sort(key=lambda p: (tier_order.get(p.get("tier"), 0), p.get("win_rate", 0)), reverse=True)
    return patterns


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(max(0.0, dot / (na * nb)))


def match_current_features(pair: str, features: np.ndarray) -> list:
    """Match current features against active patterns for a pair."""
    if not isinstance(features, np.ndarray):
        features = np.array(features, dtype=np.float64)

    active = get_active_patterns_for_pair(pair)
    if not active:
        return []

    matches = []
    for pat in active:
        center = pat.get("cluster_center")
        if center is None:
            cs = pat.get("cluster_center_json", "[]")
            try:
                center = json.loads(cs) if isinstance(cs, str) else cs
            except (json.JSONDecodeError, TypeError):
                continue
        if center is None:
            continue

        center_arr = np.array(center, dtype=np.float64)
        n_feat = len(CLUSTER_FEATURES)

        # Handle dimension mismatch
        if center_arr.shape[0] != features.shape[0]:
            if len(features) == n_feat and len(center_arr) >= n_feat:
                center_arr = center_arr[:n_feat]
            elif len(center_arr) == n_feat and len(features) >= n_feat:
                features = features[:n_feat]
            else:
                continue

        score = _cosine_similarity(features, center_arr)
        if score < 0.3:
            continue

        # Range compliance
        ranges = pat.get("feature_ranges")
        if isinstance(ranges, str):
            try:
                ranges = json.loads(ranges)
            except (json.JSONDecodeError, TypeError):
                ranges = {}

        compliance = 0.5  # neutral default
        if ranges and isinstance(ranges, dict):
            in_range = 0
            total = 0
            for idx, fname in enumerate(CLUSTER_FEATURES):
                if idx >= len(features):
                    break
                rdef = ranges.get(fname)
                if rdef and len(rdef) >= 2:
                    total += 1
                    val = float(features[idx])
                    rmin, rmax = float(rdef[0]), float(rdef[1])
                    if rmin <= val <= rmax:
                        in_range += 1
                    else:
                        dist = max(0, max(rmin - val, val - rmax))
                        width = rmax - rmin
                        if width > 1e-9:
                            in_range += max(0.0, 1.0 - dist / (width * 2)) * 0.5
            compliance = in_range / total if total > 0 else 0.5

        wr = pat.get("win_rate", 0.0)
        conf = wr * score * compliance
        tier = pat.get("tier", "PROBATIONARY")
        ctag = pat.get("currency_tag", "PAIR_ONLY")

        if ctag and ctag != "PAIR_ONLY":
            conf = min(1.0, conf + CURRENCY_CONFIRM_BOOST)

        if conf < GATE_MIN_CONFIDENCE:
            continue

        matches.append({
            "pattern_id": pat.get("pattern_id"),
            "match_score": round(float(score), 4),
            "range_compliance": round(float(compliance), 4),
            "direction": pat.get("direction"),
            "confidence": round(float(conf), 4),
            "expected_r": round(float(pat.get("avg_expected_r", 0.0)), 4),
            "tier": tier,
            "win_rate": wr,
            "currency_tag": ctag,
        })

    matches.sort(key=lambda m: m["confidence"], reverse=True)
    return matches


def hibernate_pattern(pattern_id: str, reason: str = ""):
    """Hibernate a decayed pattern."""
    try:
        conn = rpde_db.get_connection()
        c = conn.cursor(dictionary=True)
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute("UPDATE rpde_pattern_library SET is_active=0, hibernating_since=%s WHERE pattern_id=%s",
                      (now, pattern_id))
            _invalidate_cache()
            log.info(f"[RPDE_LIB] Hibernated {pattern_id}: {reason or 'decay'}")
        finally:
            c.close()
            conn.close()
    except Exception as e:
        log.error(f"[RPDE_LIB] Failed to hibernate {pattern_id}: {e}")


def reactivate_pattern(pattern_id: str):
    """Reactivate a hibernated pattern."""
    try:
        conn = rpde_db.get_connection()
        c = conn.cursor(dictionary=True)
        try:
            c.execute("UPDATE rpde_pattern_library SET is_active=1, hibernating_since=NULL WHERE pattern_id=%s",
                      (pattern_id,))
            _invalidate_cache()
            log.info(f"[RPDE_LIB] Reactivated {pattern_id}")
        finally:
            c.close()
            conn.close()
    except Exception as e:
        log.error(f"[RPDE_LIB] Failed to reactivate {pattern_id}: {e}")


def refresh_pattern_library(pair: str = None):
    """Monthly refresh: re-mine, re-validate, hibernate decayed."""
    from config.settings import PAIR_WHITELIST
    from rpde.pattern_miner import mine_patterns
    from rpde.pattern_validator import validate_all_patterns

    pairs = [pair] if pair else PAIR_WHITELIST
    total_new = 0
    total_hib = 0

    for p in pairs:
        try:
            moments = rpde_db.load_golden_moments(pair=p)
            if len(moments) < 30:
                continue
            candidates = mine_patterns(p)
            if not candidates:
                continue
            all_pp = load_patterns(active_only=True)
            validated = validate_all_patterns(p, candidates, moments, all_pp)
            for v in validated:
                save_pattern(v["pattern"], v["validation"]["statistics"], v["validation"])
                total_new += 1

            # Check decay
            existing = load_patterns(pair=p, active_only=False).get(p, [])
            for pat in existing:
                pid = pat.get("pattern_id")
                if pid:
                    rpde_db.update_pattern_stats(pid)
            log.info(f"[RPDE_LIB] Refreshed {p}: {len(validated)} patterns")
        except Exception as e:
            log.error(f"[RPDE_LIB] Refresh failed for {p}: {e}")

    _invalidate_cache()
    log.info(f"[RPDE_LIB] Library refresh done: {total_new} new patterns")


def print_pattern_report(pair: str = None):
    """Print human-readable pattern report."""
    patterns = load_patterns(pair=pair, active_only=False)
    total = sum(len(p) for p in patterns.values())
    active = sum(sum(1 for p2 in p if p2.get("is_active", 1)) for p in patterns.values())

    print()
    print("=" * 70)
    print("  RPDE PATTERN LIBRARY REPORT")
    print("=" * 70)
    print(f"  Total: {total} | Active: {active} | Pairs: {len(patterns)}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("-" * 70)

    for pair_name in sorted(patterns.keys()):
        pats = patterns[pair_name]
        act = [p for p in pats if p.get("is_active", 1) == 1]
        if not pats:
            continue
        best_tier = max(pats, key=lambda x: {"GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1}.get(x.get("tier"), 0)).get("tier", "NONE")
        avg_wr = np.mean([p.get("win_rate", 0) for p in act]) if act else 0
        avg_pf = np.mean([p.get("profit_factor", 0) for p in act]) if act else 0
        print(f"\n  {pair_name}: {len(act)}/{len(pats)} active | Best: {best_tier} | Avg WR: {avg_wr:.1%} | Avg PF: {avg_pf:.2f}")
        for pat in sorted(act, key=lambda x: x.get("win_rate", 0), reverse=True)[:5]:
            pid = pat.get("pattern_id", "?")
            print(f"    {pid}: dir={pat.get('direction', '?')} wr={pat.get('win_rate', 0):.1%} "
                  f"pf={pat.get('profit_factor', 0):.2f} occ={pat.get('occurrences', 0)} "
                  f"tier={pat.get('tier', '?')} cur={pat.get('currency_tag', 'PAIR_ONLY')}")

    print()
    print("=" * 70)
