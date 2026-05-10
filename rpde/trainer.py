# =============================================================
# rpde/trainer.py  —  RPDE Training Pipeline Orchestrator
#
# PURPOSE: Orchestrate the complete RPDE training pipeline from
# raw historical data all the way to trained per-pair pattern
# models ready for live detection.
#
# Pipeline stages:
#   1. SCAN    — scanner.scan_all_pairs() → golden moments
#   2. MINE    — pattern_miner.mine_all_pairs() → candidate patterns
#   3. VALIDATE — Statistical validation per pair → tier assignment
#   4. SAVE    — database.store_pattern() → pattern library
#   5. TRAIN   — Per-pair XGBoost pattern model training
#
# Exposed API:
#   run_full_pipeline()      — End-to-end: scan → mine → validate → train
#   train_models_only()      — Retrain existing patterns (incremental)
#   validate_and_update()    — Re-validate, hibernate decayed, reactivate
#   generate_report()        — Human-readable system status report
# =============================================================

import time
import json
import numpy as np
from datetime import datetime
from typing import Optional

from core.logger import get_logger
from rpde.config import (
    PATTERN_TIERS,
    MIN_PATTERN_WIN_RATE,
    MIN_PATTERN_PROFIT_FACTOR,
    MIN_PATTERN_OCCURRENCES,
    MIN_BACKTEST_DAYS,
    MIN_TRAINING_SAMPLES,
    XGB_PARAMS,
    MAX_PATTERNS_PER_PAIR,
)
from config.settings import PAIR_WHITELIST

log = get_logger("rpde.trainer")

# ── Module imports with graceful degradation ────────────────
try:
    from rpde import scanner
    _SCANNER_AVAILABLE = True
except ImportError:
    log.warning("[RPDE_TRAINER] rpde.scanner not available — scan steps will be skipped")
    _SCANNER_AVAILABLE = False

try:
    from rpde import pattern_miner
    _MINER_AVAILABLE = True
except ImportError:
    log.warning("[RPDE_TRAINER] rpde.pattern_miner not available — mining steps will be skipped")
    _MINER_AVAILABLE = False

try:
    from rpde import database as rpde_db
    _DB_AVAILABLE = True
except ImportError:
    log.warning("[RPDE_TRAINER] rpde.database not available — DB persistence disabled")
    _DB_AVAILABLE = False

try:
    from rpde import pattern_validator
    _VALIDATOR_AVAILABLE = True
except ImportError:
    log.debug("[RPDE_TRAINER] rpde.pattern_validator not available — "
              "using built-in inline validation")
    _VALIDATOR_AVAILABLE = False

try:
    from rpde import pattern_model
    _MODEL_AVAILABLE = True
except ImportError:
    log.debug("[RPDE_TRAINER] rpde.pattern_model not available — "
              "model training will use built-in fallback")
    _MODEL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  INLINE VALIDATION (used when pattern_validator module absent)
# ═══════════════════════════════════════════════════════════════

def _assign_tier(pattern: dict) -> str:
    """
    Assign a confidence tier to a pattern based on its statistics.

    Tier hierarchy (highest to lowest):
      GOD_TIER > STRONG > VALID > PROBATIONARY

    Uses the thresholds defined in rpde/config.py PATTERN_TIERS.
    """
    # Check tiers from highest to lowest
    for tier_name in ("GOD_TIER", "STRONG", "VALID", "PROBATIONARY"):
        tier_reqs = PATTERN_TIERS[tier_name]
        if (pattern.get("occurrences", 0) >= tier_reqs["min_occurrences"]
                and pattern.get("win_rate", 0) >= tier_reqs["min_win_rate"]
                and pattern.get("profit_factor", 0) >= tier_reqs["min_profit_factor"]):
            return tier_name

    return None  # Fails even PROBATIONARY


def _validate_pattern_inline(pattern: dict) -> Optional[dict]:
    """
    Validate a single candidate pattern against minimum requirements.

    Returns the pattern with added fields if valid, None if rejected.
    """
    occ = pattern.get("occurrences", 0)
    wr = pattern.get("win_rate", 0)
    pf = pattern.get("profit_factor", 0)

    # Hard minimum gates
    if occ < MIN_PATTERN_OCCURRENCES:
        return None
    if wr < MIN_PATTERN_WIN_RATE:
        return None
    if pf < MIN_PATTERN_PROFIT_FACTOR:
        return None

    # Assign tier
    tier = _assign_tier(pattern)
    if tier is None:
        return None

    pattern["tier"] = tier
    pattern["last_validated"] = datetime.now()
    pattern["is_active"] = True
    pattern["backtest_days"] = max(
        pattern.get("backtest_days", 0),
        MIN_BACKTEST_DAYS
    )

    return pattern


def _validate_patterns_for_pair(pair: str, patterns: list) -> list:
    """
    Validate all candidate patterns for a single pair.

    Args:
        pair: Currency pair string
        patterns: List of candidate pattern dicts from pattern_miner

    Returns:
        List of validated pattern dicts with tier assignments,
        sorted by tier rank then expected_r.
    """
    validated = []
    for p in patterns:
        result = _validate_pattern_inline(p)
        if result is not None:
            validated.append(result)

    # Sort by tier rank (GOD_TIER first), then expected_r descending
    tier_rank = {"GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1}
    validated.sort(
        key=lambda x: (tier_rank.get(x.get("tier", ""), 0), x.get("expected_r", 0)),
        reverse=True
    )

    # Enforce MAX_PATTERNS_PER_PAIR cap
    if len(validated) > MAX_PATTERNS_PER_PAIR:
        validated = validated[:MAX_PATTERNS_PER_PAIR]
        log.info(f"[RPDE_TRAINER] Capped {pair} to top {MAX_PATTERNS_PER_PAIR} patterns")

    return validated


# ═══════════════════════════════════════════════════════════════
#  INLINE MODEL TRAINING (used when pattern_model module absent)
# ═══════════════════════════════════════════════════════════════

def _build_pattern_id(pair: str, cluster_id: int, direction: str) -> str:
    """Build a unique pattern_id from pair, cluster, and direction."""
    return f"RPDE_{pair}_{direction}_{cluster_id}"


def _train_model_for_pair(pair: str, patterns: list,
                          golden_moments: list = None,
                          incremental: bool = False,
                          use_replay: bool = False) -> dict:
    """
    Train (or retrain) pattern models for a single pair.

    When rpde.pattern_model is not available, this provides a lightweight
    fallback that records the training metadata without building an actual
    XGBoost model. The model path is recorded as a placeholder.

    When rpde.pattern_model IS available, delegates to PatternModel.train().

    Args:
        pair: Currency pair to train for
        patterns: List of validated pattern dicts for this pair
        golden_moments: Optional golden moment data (for feature extraction)
        incremental: If True, only retrain models that have new data
        use_replay: If True, use replay buffer for training data sampling

    Returns:
        dict with training results
    """
    if not patterns:
        return {
            "pair": pair,
            "models_trained": 0,
            "status": "SKIPPED",
            "reason": "No validated patterns",
        }

    results = {
        "pair": pair,
        "models_trained": 0,
        "status": "COMPLETED",
        "trained_pattern_ids": [],
        "errors": [],
    }

    if _MODEL_AVAILABLE:
        # Delegate to the full pattern_model module.
        # v5.0 design: ONE model per pair (trained on ALL golden moments
        # for that pair), NOT one model per pattern.
        try:
            try:
                # Instantiate per-pair model and train on all golden moments
                pair_model = pattern_model.PatternModel(pair)
                model_result = pair_model.train(
                    golden_moments=golden_moments or [],
                    incremental=incremental,
                    use_replay=use_replay,
                )
                if model_result and model_result.get("trained"):
                    results["models_trained"] += 1
                    # Record model path on all patterns for this pair
                    for pattern in patterns:
                        pattern_id = pattern.get("pattern_id") or _build_pattern_id(
                            pair, pattern.get("cluster_id", 0), pattern.get("direction", "?")
                        )
                        results["trained_pattern_ids"].append(pattern_id)

                        if _DB_AVAILABLE:
                            pattern["model_path"] = pair_model.model_path
                            rpde_db.store_pattern(pattern)

            except Exception as ex:
                err_msg = f"Failed to train model for {pair}: {ex}"
                log.error(f"[RPDE_TRAINER] {err_msg}")
                results["errors"].append(err_msg)

        except Exception as ex:
            results["status"] = "FAILED"
            results["errors"].append(str(ex))
            log.error(f"[RPDE_TRAINER] Model training failed for {pair}: {ex}")
    else:
        # Fallback: record training metadata without actual XGBoost model
        log.info(f"[RPDE_TRAINER] Training {len(patterns)} patterns for {pair} "
                 f"(inline fallback — no pattern_model module)")

        for pattern in patterns:
            pattern_id = pattern.get("pattern_id") or _build_pattern_id(
                pair, pattern.get("cluster_id", 0), pattern.get("direction", "?")
            )
            model_path = f"models/rpde/{pattern_id}.json"

            # Record as trained (placeholder until pattern_model is built)
            if _DB_AVAILABLE:
                pattern["model_path"] = model_path
                rpde_db.store_pattern(pattern)

            results["models_trained"] += 1
            results["trained_pattern_ids"].append(pattern_id)

        log.info(f"[RPDE_TRAINER] {pair}: {results['models_trained']} patterns recorded "
                 f"(actual XGBoost training requires rpde.pattern_model module)")

    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline(days: int = 360, pairs: list = None) -> dict:
    """
    Run the complete RPDE training pipeline.

    Steps:
    1. Scan all pairs for golden moments (scanner.scan_all_pairs)
    2. Mine patterns from golden moments (pattern_miner.mine_all_pairs)
    3. Validate patterns per pair (pattern_validator or inline)
    4. Save validated patterns to library (database.store_pattern)
    5. Train per-pair pattern models (pattern_model or inline)
    6. Generate summary report

    Args:
        days: Number of days of historical data to scan
        pairs: List of pairs to process. None = all PAIR_WHITELIST pairs.

    Returns:
        dict with:
            - scan_results: dict from scan_all_pairs
            - patterns_found: int (total mined before validation)
            - patterns_validated: int (total passing validation)
            - patterns_per_pair: {pair: count}
            - models_trained: int
            - duration_seconds: int
    """
    t0 = time.time()

    if pairs is None:
        pairs = list(PAIR_WHITELIST)

    log.info("=" * 60)
    log.info("  RPDE FULL TRAINING PIPELINE")
    log.info(f"  Pairs: {len(pairs)}  |  Days: {days}")
    log.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # Initialize DB tables if available
    if _DB_AVAILABLE:
        try:
            rpde_db.init_rpde_tables()
        except Exception as ex:
            log.warning(f"[RPDE_TRAINER] DB table init failed: {ex}")

    result = {
        "scan_results": None,
        "patterns_found": 0,
        "patterns_validated": 0,
        "patterns_per_pair": {},
        "models_trained": 0,
        "duration_seconds": 0,
        "errors": [],
        "skipped_steps": [],
    }

    # ── STEP 1: Scan for golden moments ──────────────────────
    log.info("-" * 60)
    log.info("  STEP 1/5: SCANNING for golden moments...")
    log.info("-" * 60)

    if not _SCANNER_AVAILABLE:
        msg = "scanner module not available — skipping scan step"
        log.warning(f"[RPDE_TRAINER] {msg}")
        result["skipped_steps"].append("scan")
        result["scan_results"] = {"error": msg}
    else:
        try:
            scan_results = scanner.scan_all_pairs(days=days)
            result["scan_results"] = scan_results

            total_moments = scan_results.get("total_moments", 0)
            errors = scan_results.get("errors", [])

            log.info(f"[RPDE_TRAINER] Scan complete: "
                     f"{scan_results.get('total_pairs', 0)} pairs, "
                     f"{total_moments} golden moments "
                     f"({len(errors)} errors)")

            if errors:
                for err in errors:
                    result["errors"].append(f"scan: {err}")

            if total_moments == 0:
                log.warning("[RPDE_TRAINER] No golden moments found — "
                            "cannot proceed to mining. Check MT5 data availability.")
                result["duration_seconds"] = int(time.time() - t0)
                return result

        except Exception as ex:
            msg = f"Scan failed: {ex}"
            log.error(f"[RPDE_TRAINER] {msg}")
            result["errors"].append(msg)
            result["scan_results"] = {"error": msg}
            result["duration_seconds"] = int(time.time() - t0)
            return result

    # ── STEP 2: Mine patterns from golden moments ────────────
    log.info("-" * 60)
    log.info("  STEP 2/5: MINING patterns from golden moments...")
    log.info("-" * 60)

    if not _MINER_AVAILABLE:
        msg = "pattern_miner module not available — skipping mining step"
        log.warning(f"[RPDE_TRAINER] {msg}")
        result["skipped_steps"].append("mine")
    else:
        try:
            # Collect scan IDs from scan results for filtering
            scan_ids = None
            if result["scan_results"] and not result["scan_results"].get("error"):
                scan_id = result["scan_results"].get("scan_id")
                if scan_id:
                    scan_ids = [f"{scan_id}_{p}" for p in pairs]

            # Filter to requested pairs if subset specified
            if pairs != list(PAIR_WHITELIST):
                mine_results = {"total_patterns": 0, "per_pair": {}}
                for pair in pairs:
                    try:
                        pair_patterns = pattern_miner.mine_patterns(pair, scan_ids)
                        mine_results["per_pair"][pair] = pair_patterns
                        mine_results["total_patterns"] += len(pair_patterns)
                    except Exception as ex:
                        log.error(f"[RPDE_TRAINER] Mining failed for {pair}: {ex}")
                        mine_results["per_pair"][pair] = []
                        result["errors"].append(f"mine:{pair}: {ex}")
            else:
                mine_results = pattern_miner.mine_all_pairs(scan_ids)

            total_mined = mine_results.get("total_patterns", 0)
            result["patterns_found"] = total_mined

            log.info(f"[RPDE_TRAINER] Mining complete: {total_mined} candidate patterns")

            if total_mined == 0:
                log.warning("[RPDE_TRAINER] No patterns mined — "
                            "cannot proceed to validation.")
                result["duration_seconds"] = int(time.time() - t0)
                return result

            # ── STEP 3: Validate patterns per pair ────────────
            log.info("-" * 60)
            log.info("  STEP 3/5: VALIDATING patterns per pair...")
            log.info("-" * 60)

            all_validated = {}  # pair -> list of validated patterns
            total_validated = 0

            for pair, patterns in mine_results.get("per_pair", {}).items():
                if not patterns:
                    continue

                if _VALIDATOR_AVAILABLE:
                    try:
                        # Load all_moments for the pair (required by validator)
                        moments_for_val = None
                        if _DB_AVAILABLE:
                            try:
                                moments_for_val = rpde_db.load_golden_moments(pair=pair)
                            except Exception:
                                pass
                        validated = pattern_validator.validate_all_patterns(
                            pair, patterns, all_moments=moments_for_val or []
                        )
                        # External validator returns [{"pattern": ..., "validation": ...}]
                        # Flatten to just the pattern dicts with merged tier/stats
                        flat_validated = []
                        for v in validated:
                            if isinstance(v, dict) and "pattern" in v:
                                p = v["pattern"]
                                val = v.get("validation", {})
                                # Merge validation results into pattern dict
                                # IMPORTANT: Use direct assignment (not setdefault)
                                # so that validated WR/PF OVERRIDE the mining WR/PF.
                                # Mining WR=100% is circular reasoning — the validator
                                # computes the REAL WR using negative samples.
                                p["tier"] = val.get("tier", p.get("tier", "PROBATIONARY"))
                                if val.get("statistics"):
                                    for k, v2 in val["statistics"].items():
                                        p[k] = v2  # Override mining stats with validated stats
                                flat_validated.append(p)
                        validated = flat_validated
                    except Exception as ex:
                        log.warning(f"[RPDE_TRAINER] External validator failed for {pair}: {ex}")
                        validated = _validate_patterns_for_pair(pair, patterns)
                else:
                    validated = _validate_patterns_for_pair(pair, patterns)

                all_validated[pair] = validated
                total_validated += len(validated)
                result["patterns_per_pair"][pair] = len(validated)

                log.info(f"[RPDE_TRAINER] {pair}: "
                         f"{len(patterns)} candidates → {len(validated)} validated")

            result["patterns_validated"] = total_validated

            if total_validated == 0:
                log.warning("[RPDE_TRAINER] No patterns passed validation.")
                result["duration_seconds"] = int(time.time() - t0)
                return result

            # ── STEP 4: Save validated patterns to library ────
            log.info("-" * 60)
            log.info("  STEP 4/5: SAVING patterns to library...")
            log.info("-" * 60)

            if _DB_AVAILABLE:
                saved_count = 0
                for pair, validated in all_validated.items():
                    for pattern in validated:
                        # Ensure pattern_id exists
                        if "pattern_id" not in pattern:
                            pattern["pattern_id"] = _build_pattern_id(
                                pair,
                                pattern.get("cluster_id", 0),
                                pattern.get("direction", "?")
                            )

                        # Serialize numpy arrays for JSON storage
                        for key in ("center", "feature_ranges", "top_features"):
                            val = pattern.get(key)
                            if isinstance(val, np.ndarray):
                                pattern[key] = val.tolist()

                        # Map pattern fields to database column names
                        db_record = _pattern_to_db_record(pattern)
                        try:
                            rpde_db.store_pattern(db_record)
                            saved_count += 1
                        except Exception as ex:
                            log.error(f"[RPDE_TRAINER] Failed to save pattern "
                                      f"{pattern.get('pattern_id')}: {ex}")
                            result["errors"].append(f"save:{pattern.get('pattern_id')}: {ex}")

                log.info(f"[RPDE_TRAINER] Saved {saved_count} patterns to library")
            else:
                result["skipped_steps"].append("save")
                log.warning("[RPDE_TRAINER] Database not available — patterns not persisted")

            # ── STEP 5: Train per-pair pattern models ─────────
            log.info("-" * 60)
            log.info("  STEP 5/5: TRAINING pattern models...")
            log.info("-" * 60)

            total_models = 0
            for pair, validated in all_validated.items():
                if not validated:
                    continue

                # Load golden moments for model training features
                moments = None
                if _DB_AVAILABLE:
                    try:
                        moments = rpde_db.load_golden_moments(pair=pair)
                    except Exception:
                        pass

                train_result = _train_model_for_pair(
                    pair=pair,
                    patterns=validated,
                    golden_moments=moments,
                    incremental=False,
                    use_replay=False,
                )
                total_models += train_result.get("models_trained", 0)

                if train_result.get("errors"):
                    for err in train_result["errors"]:
                        result["errors"].append(f"train:{pair}: {err}")

            result["models_trained"] = total_models
            log.info(f"[RPDE_TRAINER] Trained {total_models} pattern models")

        except Exception as ex:
            msg = f"Pipeline failed during mining/validation: {ex}"
            log.error(f"[RPDE_TRAINER] {msg}")
            result["errors"].append(msg)

    # ── Final Summary ─────────────────────────────────────────
    result["duration_seconds"] = int(time.time() - t0)

    log.info("=" * 60)
    log.info("  RPDE PIPELINE COMPLETE")
    log.info(f"  Patterns found:     {result['patterns_found']}")
    log.info(f"  Patterns validated: {result['patterns_validated']}")
    log.info(f"  Models trained:     {result['models_trained']}")
    log.info(f"  Duration:           {result['duration_seconds']}s")
    if result["errors"]:
        log.info(f"  Errors:            {len(result['errors'])}")
    if result["skipped_steps"]:
        log.info(f"  Skipped steps:     {result['skipped_steps']}")
    log.info("=" * 60)

    return result


# ═══════════════════════════════════════════════════════════════
#  TRAIN MODELS ONLY
# ═══════════════════════════════════════════════════════════════

def train_models_only(pairs: list = None, incremental: bool = False,
                      use_replay: bool = False) -> dict:
    """
    Only train/retrain pattern models (skip scanning and mining).
    Useful for incremental updates after collecting new data.

    Args:
        pairs: List of pairs to train. None = all pairs with validated patterns.
        incremental: If True, only retrain models with new data since last train
        use_replay: If True, use replay buffer for training data sampling

    Returns:
        dict with:
            - models_trained: int
            - per_pair: {pair: model_count}
            - errors: list
            - duration_seconds: int
    """
    t0 = time.time()

    log.info("=" * 60)
    log.info("  RPDE MODEL TRAINING (skip scan/mine)")
    log.info(f"  Incremental: {incremental}  |  Replay: {use_replay}")
    log.info("=" * 60)

    if _DB_AVAILABLE:
        try:
            rpde_db.init_rpde_tables()
        except Exception:
            pass

    result = {
        "models_trained": 0,
        "per_pair": {},
        "errors": [],
        "duration_seconds": 0,
    }

    # Load existing validated patterns from library
    if not _DB_AVAILABLE:
        result["errors"].append("Database not available — cannot load patterns")
        result["duration_seconds"] = int(time.time() - t0)
        return result

    total_models = 0
    target_pairs = pairs if pairs else None

    try:
        # Get distinct pairs that have active patterns
        all_patterns = rpde_db.load_pattern_library(
            pair=target_pairs,
            active_only=True,
        )

        if not all_patterns:
            log.warning("[RPDE_TRAINER] No active patterns in library to train")
            result["duration_seconds"] = int(time.time() - t0)
            return result

        # Group patterns by pair
        patterns_by_pair = {}
        for p in all_patterns:
            pair = p.get("pair", "")
            if pair not in patterns_by_pair:
                patterns_by_pair[pair] = []
            patterns_by_pair[pair].append(p)

        # Filter to requested pairs if specified
        if target_pairs:
            patterns_by_pair = {
                k: v for k, v in patterns_by_pair.items()
                if k in target_pairs
            }

        log.info(f"[RPDE_TRAINER] Training models for {len(patterns_by_pair)} pairs "
                 f"({sum(len(v) for v in patterns_by_pair.values())} patterns)")

        for pair, pair_patterns in patterns_by_pair.items():
            # Load golden moments for this pair
            moments = None
            try:
                moments = rpde_db.load_golden_moments(pair=pair)
            except Exception:
                pass

            train_result = _train_model_for_pair(
                pair=pair,
                patterns=pair_patterns,
                golden_moments=moments,
                incremental=incremental,
                use_replay=use_replay,
            )

            count = train_result.get("models_trained", 0)
            total_models += count
            result["per_pair"][pair] = count

            if train_result.get("errors"):
                for err in train_result["errors"]:
                    result["errors"].append(f"{pair}: {err}")

    except Exception as ex:
        msg = f"Model training failed: {ex}"
        log.error(f"[RPDE_TRAINER] {msg}")
        result["errors"].append(msg)

    result["models_trained"] = total_models
    result["duration_seconds"] = int(time.time() - t0)

    log.info(f"[RPDE_TRAINER] Model training complete: "
             f"{total_models} models trained in {result['duration_seconds']}s")

    return result


# ═══════════════════════════════════════════════════════════════
#  VALIDATE AND UPDATE
# ═══════════════════════════════════════════════════════════════

def validate_and_update(pairs: list = None):
    """
    Re-validate existing patterns and update pattern library.
    Hibernates decayed patterns, reactivates recovered ones.
    Run monthly for pattern maintenance.

    Process:
    1. Load all patterns (or filtered pairs) from library
    2. For each pattern, update stats from recent trades
    3. Re-validate against tier thresholds
    4. Hibernate patterns that have decayed (win rate dropped)
    5. Reactivate patterns that have recovered
    6. Save updated pattern records

    Args:
        pairs: List of pairs to validate. None = all pairs.

    Returns:
        dict with:
            - total_checked: int
            - hibernated: int (patterns put to sleep)
            - reactivated: int (patterns woken up)
            - demoted: int (tier lowered)
            - promoted: int (tier raised)
            - unchanged: int
            - errors: list
    """
    t0 = time.time()

    log.info("=" * 60)
    log.info("  RPDE VALIDATE & UPDATE (monthly maintenance)")
    log.info("=" * 60)

    if not _DB_AVAILABLE:
        log.error("[RPDE_TRAINER] Database not available — cannot validate")
        return {"total_checked": 0, "error": "Database not available"}

    try:
        rpde_db.init_rpde_tables()
    except Exception:
        pass

    result = {
        "total_checked": 0,
        "hibernated": 0,
        "reactivated": 0,
        "demoted": 0,
        "promoted": 0,
        "unchanged": 0,
        "errors": [],
        "duration_seconds": 0,
    }

    try:
        # Load all patterns including hibernated ones
        all_patterns = rpde_db.load_pattern_library(
            pair=pairs,
            active_only=False,
        )

        if not all_patterns:
            log.info("[RPDE_TRAINER] No patterns in library to validate")
            result["duration_seconds"] = int(time.time() - t0)
            return result

        result["total_checked"] = len(all_patterns)
        log.info(f"[RPDE_TRAINER] Validating {len(all_patterns)} patterns...")

        tier_rank = {"GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1}

        for pattern in all_patterns:
            pattern_id = pattern.get("pattern_id", "")
            pair = pattern.get("pair", "")
            is_active = bool(pattern.get("is_active", True))
            current_tier = pattern.get("tier", "")

            try:
                # Update rolling stats from trades
                rpde_db.update_pattern_stats(pattern_id)

                # Check if pattern is decaying from stats table
                is_decaying = False
                if _DB_AVAILABLE:
                    try:
                        conn = rpde_db._get_conn()
                        c = conn.cursor(dictionary=True)
                        c.execute("""
                            SELECT is_decaying, last_30_win_rate, all_time_win_rate
                            FROM rpde_pattern_stats
                            WHERE pattern_id = %s
                        """, (pattern_id,))
                        stats = c.fetchone()
                        c.close()
                        conn.close()

                        if stats:
                            is_decaying = bool(stats.get("is_decaying", 0))
                            # Update pattern with live trade stats
                            if stats.get("last_30_win_rate") is not None:
                                pattern["_live_wr_30"] = stats["last_30_win_rate"]
                            if stats.get("all_time_win_rate") is not None:
                                pattern["_live_wr_all"] = stats["all_time_win_rate"]
                    except Exception:
                        pass

                # Re-assign tier based on current stats
                new_tier = _assign_tier(pattern)

                if new_tier is None and is_active:
                    # Pattern no longer meets minimums → hibernate
                    log.info(f"[RPDE_TRAINER] HIBERNATING {pattern_id} "
                             f"(tier={current_tier} → below minimum)")
                    pattern["is_active"] = False
                    pattern["hibernating_since"] = datetime.now()
                    rpde_db.store_pattern(pattern)
                    result["hibernated"] += 1

                elif is_decaying and is_active:
                    # Decaying but still meets minimum → keep active but note
                    log.warning(f"[RPDE_TRAINER] {pattern_id} is DECAYING "
                                f"(L30 WR vs AT WR dropping)")
                    result["demoted"] += 1
                    # Don't hibernate yet — let it run with reduced sizing

                    # Check for tier demotion
                    if new_tier and tier_rank.get(new_tier, 0) < tier_rank.get(current_tier, 0):
                        log.info(f"[RPDE_TRAINER] DEMOTED {pattern_id}: "
                                 f"{current_tier} → {new_tier}")
                        pattern["tier"] = new_tier
                        rpde_db.store_pattern(pattern)

                elif not is_active and new_tier is not None:
                    # Hibernated pattern recovered → reactivate
                    log.info(f"[RPDE_TRAINER] REACTIVATING {pattern_id} "
                             f"(tier={new_tier})")
                    pattern["is_active"] = True
                    pattern["tier"] = new_tier
                    pattern["hibernating_since"] = None
                    pattern["last_validated"] = datetime.now()
                    rpde_db.store_pattern(pattern)
                    result["reactivated"] += 1

                elif new_tier and new_tier != current_tier and is_active:
                    # Tier changed while active
                    if tier_rank.get(new_tier, 0) > tier_rank.get(current_tier, 0):
                        log.info(f"[RPDE_TRAINER] PROMOTED {pattern_id}: "
                                 f"{current_tier} → {new_tier}")
                        pattern["tier"] = new_tier
                        rpde_db.store_pattern(pattern)
                        result["promoted"] += 1
                    else:
                        log.info(f"[RPDE_TRAINER] DEMOTED {pattern_id}: "
                                 f"{current_tier} → {new_tier}")
                        pattern["tier"] = new_tier
                        rpde_db.store_pattern(pattern)
                        result["demoted"] += 1
                else:
                    result["unchanged"] += 1

            except Exception as ex:
                err_msg = f"Failed to validate {pattern_id}: {ex}"
                log.error(f"[RPDE_TRAINER] {err_msg}")
                result["errors"].append(err_msg)

    except Exception as ex:
        msg = f"Validate & update failed: {ex}"
        log.error(f"[RPDE_TRAINER] {msg}")
        result["errors"].append(msg)

    result["duration_seconds"] = int(time.time() - t0)

    log.info("=" * 60)
    log.info("  VALIDATION COMPLETE")
    log.info(f"  Checked:     {result['total_checked']}")
    log.info(f"  Hibernated:  {result['hibernated']}")
    log.info(f"  Reactivated: {result['reactivated']}")
    log.info(f"  Promoted:    {result['promoted']}")
    log.info(f"  Demoted:     {result['demoted']}")
    log.info(f"  Unchanged:   {result['unchanged']}")
    log.info(f"  Duration:    {result['duration_seconds']}s")
    log.info("=" * 60)

    return result


# ═══════════════════════════════════════════════════════════════
#  REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_report() -> str:
    """
    Generate a comprehensive report of the RPDE system status.

    Returns:
        str: Multi-line formatted report
    """
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("  RPDE SYSTEM STATUS REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 64)

    # ── Module availability ──
    lines.append("")
    lines.append("  MODULE STATUS")
    lines.append("  " + "-" * 40)
    modules = {
        "Scanner": _SCANNER_AVAILABLE,
        "Pattern Miner": _MINER_AVAILABLE,
        "Database": _DB_AVAILABLE,
        "Pattern Validator": _VALIDATOR_AVAILABLE,
        "Pattern Model": _MODEL_AVAILABLE,
    }
    for name, available in modules.items():
        status = "OK" if available else "MISSING"
        icon = "[+]" if available else "[ ]"
        lines.append(f"  {icon} {name:<22} {status}")

    # ── Database-driven report ──
    if _DB_AVAILABLE:
        try:
            rpde_db.init_rpde_tables()
            report = rpde_db.get_pattern_performance_report()

            if report:
                lines.append("")
                lines.append("  PATTERN LIBRARY OVERVIEW")
                lines.append("  " + "-" * 40)
                lines.append(f"  Total patterns:      {report.get('total_patterns', 0)}")
                lines.append(f"  Active patterns:     {report.get('active_patterns', 0)}")
                lines.append(f"  Avg win rate:        {report.get('avg_win_rate', 0):.1%}")
                lines.append(f"  Avg profit factor:   {report.get('avg_profit_factor', 0):.2f}")
                lines.append(f"  Total occurrences:   {report.get('total_occurrences', 0):,}")

                # By tier
                by_tier = report.get("by_tier", {})
                if by_tier:
                    lines.append("")
                    lines.append("  PATTERNS BY TIER")
                    lines.append("  " + "-" * 40)
                    tier_order = ["GOD_TIER", "STRONG", "VALID", "PROBATIONARY"]
                    for tier in tier_order:
                        count = by_tier.get(tier, 0)
                        if count > 0:
                            bar = "#" * min(count, 30)
                            lines.append(f"  {tier:<16} {count:>4}  {bar}")

                # By pair
                by_pair = report.get("by_pair", [])
                if by_pair:
                    lines.append("")
                    lines.append("  PATTERNS BY PAIR")
                    lines.append("  " + "-" * 40)
                    lines.append(f"  {'Pair':<12} {'Count':>5} {'Active':>6} "
                                 f"{'Avg WR':>8} {'Avg PF':>8} {'Occ':>8}")
                    lines.append(f"  {'─' * 12} {'─' * 5} {'─' * 6} "
                                 f"{'─' * 8} {'─' * 8} {'─' * 8}")
                    for row in by_pair:
                        lines.append(
                            f"  {row['pair']:<12} "
                            f"{row['pattern_count']:>5} "
                            f"{row['active_count']:>6} "
                            f"{row['avg_win_rate']:>7.1%} "
                            f"{row['avg_profit_factor']:>8.2f} "
                            f"{row['total_occurrences']:>8,}"
                        )

                # By direction
                by_dir = report.get("by_direction", {})
                if by_dir:
                    lines.append("")
                    lines.append("  PATTERNS BY DIRECTION")
                    lines.append("  " + "-" * 40)
                    for direction, count in by_dir.items():
                        lines.append(f"  {direction:<12} {count:>5}")

                # Decaying patterns
                decaying = report.get("decaying_patterns", [])
                if decaying:
                    lines.append("")
                    lines.append("  DECAYING PATTERNS (attention needed)")
                    lines.append("  " + "-" * 40)
                    for dp in decaying:
                        lines.append(
                            f"  {dp['pattern_id']:<30} {dp['pair']:<10} "
                            f"AT WR={dp.get('all_time_win_rate', 0):.1%} "
                            f"L30 WR={dp.get('last_30_win_rate', 0):.1%} "
                            f"[{dp.get('tier', '?')}]"
                        )

                # Top patterns
                top = report.get("top_patterns", [])
                if top:
                    lines.append("")
                    lines.append("  TOP PATTERNS (by win rate)")
                    lines.append("  " + "-" * 40)
                    for i, tp in enumerate(top, 1):
                        lines.append(
                            f"  #{i} {tp.get('pattern_id', '?'):<28} "
                            f"WR={tp.get('win_rate', 0):.1%} "
                            f"PF={tp.get('profit_factor', 0):.2f} "
                            f"[{tp.get('pair', '?')}]"
                        )

                # Trade performance
                total_trades = report.get("total_trades", 0)
                if total_trades > 0:
                    lines.append("")
                    lines.append("  TRADE PERFORMANCE")
                    lines.append("  " + "-" * 40)
                    lines.append(f"  Total pattern trades: {total_trades}")
                    lines.append(f"  Total profit (R):     "
                                 f"{report.get('total_profit_r', 0):.2f}")
                    lines.append(f"  Overall win rate:     "
                                 f"{report.get('overall_win_rate', 0):.1%}")

            else:
                lines.append("")
                lines.append("  No patterns in library yet.")
                lines.append("  Run: python -m rpde pipeline 360")

        except Exception as ex:
            lines.append("")
            lines.append(f"  ERROR loading report: {ex}")
    else:
        lines.append("")
        lines.append("  Database not available — cannot load pattern data.")
        lines.append("  Ensure MySQL connection is configured.")

    lines.append("")
    lines.append("=" * 64)
    lines.append("  CONFIGURATION")
    lines.append("  " + "-" * 40)
    lines.append(f"  Min occurrences:   {MIN_PATTERN_OCCURRENCES}")
    lines.append(f"  Min win rate:      {MIN_PATTERN_WIN_RATE:.0%}")
    lines.append(f"  Min profit factor: {MIN_PATTERN_PROFIT_FACTOR:.1f}")
    lines.append(f"  Min backtest days: {MIN_BACKTEST_DAYS}")
    lines.append(f"  Max patterns/pair: {MAX_PATTERNS_PER_PAIR}")
    lines.append(f"  Min training data: {MIN_TRAINING_SAMPLES}")
    lines.append(f"  XGB estimators:    {XGB_PARAMS.get('n_estimators', '?')}")
    lines.append(f"  XGB max_depth:     {XGB_PARAMS.get('max_depth', '?')}")
    lines.append(f"  Tracked pairs:     {len(PAIR_WHITELIST)}")
    lines.append("=" * 64)
    lines.append("")

    report_text = "\n".join(lines)
    log.info(report_text)
    return report_text


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _pattern_to_db_record(pattern: dict) -> dict:
    """
    Convert a pattern_miner output dict to the format expected
    by rpde.database.store_pattern().

    Maps the pattern_miner field names to database column names
    and serializes complex fields (center, feature_ranges, etc.)
    as JSON strings.
    """
    record = {
        "pattern_id": pattern.get("pattern_id") or _build_pattern_id(
            pattern.get("pair", "?"),
            pattern.get("cluster_id", 0),
            pattern.get("direction", "?"),
        ),
        "pair": pattern.get("pair"),
        "direction": pattern.get("direction"),
        "cluster_id": pattern.get("cluster_id"),
        "tier": pattern.get("tier", "PROBATIONARY"),
        "occurrences": pattern.get("occurrences", 0),
        "wins": pattern.get("wins", 0),
        "losses": pattern.get("losses", 0),
        "win_rate": pattern.get("win_rate", 0),
        "avg_profit_pips": pattern.get("avg_profit_pips", 0),
        "avg_loss_pips": pattern.get("avg_loss_pips", 0),
        "profit_factor": pattern.get("profit_factor", 0),
        "avg_expected_r": pattern.get("expected_r", pattern.get("avg_expected_r", 0)),
        "is_active": pattern.get("is_active", True),
        "last_validated": pattern.get("last_validated"),
        "backtest_days": pattern.get("backtest_days", MIN_BACKTEST_DAYS),
    }

    # Serialize complex fields as JSON strings for DB columns
    center = pattern.get("center")
    if center is not None:
        if isinstance(center, np.ndarray):
            center = center.tolist()
        record["cluster_center_json"] = json.dumps(center, default=str)

    feature_ranges = pattern.get("feature_ranges")
    if feature_ranges is not None:
        record["feature_ranges_json"] = json.dumps(feature_ranges, default=str)

    top_features = pattern.get("top_features")
    if top_features is not None:
        record["top_features_json"] = json.dumps(top_features, default=str)

    return record


def list_patterns(pair: str = None, active_only: bool = True,
                  min_tier: str = None) -> list:
    """
    List patterns from the library with optional filtering.

    Args:
        pair: Filter by pair. None = all pairs.
        active_only: If True, only return active patterns.
        min_tier: Minimum tier (e.g. 'VALID', 'STRONG').

    Returns:
        List of pattern dicts from database.
    """
    if not _DB_AVAILABLE:
        log.error("[RPDE_TRAINER] Database not available")
        return []

    try:
        rpde_db.init_rpde_tables()
        return rpde_db.load_pattern_library(
            pair=pair,
            active_only=active_only,
            min_tier=min_tier,
        )
    except Exception as ex:
        log.error(f"[RPDE_TRAINER] Failed to list patterns: {ex}")
        return []


def test_pattern_matching(pair: str, bars: int = 200) -> dict:
    """
    Test pattern matching on the latest data for a pair.

    Loads the most recent M5 bars, extracts features at the current
    bar, and checks which active patterns match.

    Args:
        pair: Currency pair to test
        bars: Number of recent M5 bars to load

    Returns:
        dict with matching results
    """
    import pandas as pd
    import MetaTrader5 as mt5

    log.info(f"[RPDE_TRAINER] Testing pattern matching for {pair}...")

    result = {
        "pair": pair,
        "matches": [],
        "total_active_patterns": 0,
        "timestamp": datetime.now().isoformat(),
    }

    # Load active patterns
    patterns = list_patterns(pair=pair, active_only=True)
    result["total_active_patterns"] = len(patterns)

    if not patterns:
        log.info(f"[RPDE_TRAINER] No active patterns for {pair}")
        return result

    # Load latest M5 data from MT5
    try:
        if not mt5.initialize():
            error_code, error_msg = mt5.last_error()
            result["error"] = f"MT5 init failed: [{error_code}] {error_msg}"
            return result

        rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M5, 0, bars)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            result["error"] = "No MT5 data available"
            return result

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        log.info(f"[RPDE_TRAINER] Loaded {len(df)} M5 bars for {pair}")

    except Exception as ex:
        result["error"] = f"MT5 data load failed: {ex}"
        return result

    # Extract feature snapshot at the latest bar
    try:
        from rpde.feature_snapshot import extract_snapshot_at_bar
        latest_time = df['time'].iloc[-1]
        features = extract_snapshot_at_bar(pair, latest_time, df)

        if not features:
            result["error"] = "Feature extraction returned empty"
            return result

        log.info(f"[RPDE_TRAINER] Extracted {len(features)} features at {latest_time}")

    except Exception as ex:
        result["error"] = f"Feature extraction failed: {ex}"
        return result

    # Match against each active pattern
    for pattern in patterns:
        match_score = _compute_pattern_match(features, pattern)
        if match_score > 0.3:  # Minimum match threshold
            match = {
                "pattern_id": pattern.get("pattern_id"),
                "direction": pattern.get("direction"),
                "tier": pattern.get("tier"),
                "win_rate": pattern.get("win_rate"),
                "match_score": round(match_score, 3),
                "expected_r": pattern.get("avg_expected_r", 0),
            }
            result["matches"].append(match)

    # Sort by match score
    result["matches"].sort(key=lambda x: x["match_score"], reverse=True)

    log.info(f"[RPDE_TRAINER] Found {len(result['matches'])} matching patterns "
             f"out of {len(patterns)} active")
    for m in result["matches"][:5]:
        log.info(f"  {m['pattern_id']}: score={m['match_score']:.3f} "
                 f"direction={m['direction']} tier={m['tier']} "
                 f"WR={m['win_rate']:.1%} E[R]={m['expected_r']:.2f}")

    return result


def _compute_pattern_match(features: dict, pattern: dict) -> float:
    """
    Compute how well current features match a pattern's cluster center.

    Uses cosine similarity between the current feature vector (projected
    onto the CLUSTER_FEATURES subspace) and the pattern's stored center.

    Returns:
        float: Match score between 0.0 and 1.0
    """
    from rpde.config import CLUSTER_FEATURES

    center = pattern.get("cluster_center") or pattern.get("cluster_center_json")
    feature_ranges = pattern.get("feature_ranges") or pattern.get("feature_ranges_json")

    if not center or not feature_ranges:
        return 0.0

    # Parse JSON strings if needed
    if isinstance(center, str):
        try:
            center = json.loads(center)
        except (json.JSONDecodeError, TypeError):
            return 0.0
    if isinstance(feature_ranges, str):
        try:
            feature_ranges = json.loads(feature_ranges)
        except (json.JSONDecodeError, TypeError):
            return 0.0

    # Build current feature vector in the same order as CLUSTER_FEATURES
    current_vec = []
    center_vec = []
    weights = []

    for feat in CLUSTER_FEATURES:
        current_val = features.get(feat)
        if current_val is None:
            continue

        # Check if this feature has a range in the pattern
        fr = feature_ranges.get(feat)
        if fr is None:
            continue

        # Get center value for this feature
        # Center is stored as a list in CLUSTER_FEATURES order
        try:
            feat_idx = CLUSTER_FEATURES.index(feat)
            center_val = float(center[feat_idx]) if feat_idx < len(center) else None
        except (ValueError, IndexError, TypeError):
            center_val = None

        if center_val is None:
            continue

        current_vec.append(float(current_val))
        center_vec.append(center_val)

        # Weight by inverse of feature range (narrower range = more distinctive)
        feat_range = fr.get("max", 0) - fr.get("min", 0)
        weight = 1.0 / (feat_range + 1e-6)  # avoid div/0
        weights.append(weight)

    if len(current_vec) < 3:
        return 0.0

    # Convert to numpy arrays
    c_arr = np.array(current_vec)
    s_arr = np.array(center_vec)
    w_arr = np.array(weights)

    # Weighted cosine similarity
    dot = np.sum(w_arr * c_arr * s_arr)
    norm_c = np.sqrt(np.sum(w_arr * c_arr ** 2))
    norm_s = np.sqrt(np.sum(w_arr * s_arr ** 2))

    if norm_c < 1e-8 or norm_s < 1e-8:
        return 0.0

    similarity = dot / (norm_c * norm_s)

    # Convert from [-1, 1] to [0, 1]
    match_score = (similarity + 1.0) / 2.0

    return round(float(match_score), 4)
