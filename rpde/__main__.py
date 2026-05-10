#!/usr/bin/env python3
# =============================================================
# rpde/__main__.py  —  CLI Entry Point for RPDE System
#
# Allows running the RPDE system from the command line:
#
#   python -m rpde scan [PAIR] [DAYS]        # Scan for golden moments
#   python -m rpde mine                       # Mine patterns from scans
#   python -m rpde validate                   # Validate candidate patterns
#   python -m rpde train [PAIR] [--incremental] [--replay]  # Train models
#   python -m rpde pipeline [DAYS]            # Full pipeline
#   python -m rpde report                     # Generate status report
#   python -m rpde list                       # List all patterns
#   python -m rpde test [PAIR]                # Test pattern matching
#
# Usage:
#   cd /home/z/my-project/fxbot
#   python -m rpde <command> [args]
# =============================================================

import sys
import argparse
import time
from datetime import datetime

from core.logger import get_logger

log = get_logger("rpde.cli")

# ── Banner ────────────────────────────────────────────────────

BANNER = r"""
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║   R P D E   —   Reverse Pattern Discovery Engine v5   ║
  ║                                                       ║
  ║   Self-evolving trading intelligence system            ║
  ║   Discover.  Validate.  Deploy.                       ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════
#  ARGUMENT PARSERS
# ═══════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="rpde",
        description=(
            "RPDE — Reverse Pattern Discovery Engine v5.0\n"
            "Self-evolving trading intelligence that discovers\n"
            "profitable patterns by reverse-engineering history."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m rpde scan EURJPY 360     Scan EURJPY for 360 days\n"
            "  python -m rpde mine                 Mine patterns from all scans\n"
            "  python -m rpde validate             Re-validate existing patterns\n"
            "  python -m rpde train --incremental  Incrementally retrain models\n"
            "  python -m rpde pipeline 180         Full pipeline with 180 days\n"
            "  python -m rpde report               Generate system status report\n"
            "  python -m rpde list                 List all active patterns\n"
            "  python -m rpde test EURUSD          Test pattern matching live\n"
            "  python -m rpde backtest 90         Backtest pattern models\n"
            "  python -m rpde backtest 90 --thresholds  Threshold sweep\n"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="RPDE command to run")

    # ── scan ──────────────────────────────────────────────
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan historical data for golden moments (big moves)",
        description="Scan M5 price data to find bars BEFORE significant moves happened.",
    )
    scan_parser.add_argument(
        "pair",
        nargs="?",
        default=None,
        help="Currency pair (e.g. EURJPY). Omit to scan all pairs.",
    )
    scan_parser.add_argument(
        "days",
        nargs="?",
        type=int,
        default=360,
        help="Number of days of history to scan (default: 360)",
    )

    # ── mine ──────────────────────────────────────────────
    mine_parser = subparsers.add_parser(
        "mine",
        help="Mine patterns from scanned golden moments",
        description="Cluster golden moments into candidate patterns using DBSCAN.",
    )
    mine_parser.add_argument(
        "--pair",
        default=None,
        help="Mine patterns for a specific pair only",
    )

    # ── validate ──────────────────────────────────────────
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate patterns and update library (monthly maintenance)",
        description="Re-validate existing patterns, hibernate decayed ones, "
                    "reactivate recovered ones.",
    )
    validate_parser.add_argument(
        "--pair",
        default=None,
        help="Validate patterns for a specific pair only",
    )

    # ── train ─────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train",
        help="Train or retrain per-pair pattern models",
        description="Train XGBoost models for each validated pattern. "
                    "Use --incremental for updates after new data collection.",
    )
    train_parser.add_argument(
        "pair",
        nargs="?",
        default=None,
        help="Currency pair to train. Omit to train all pairs.",
    )
    train_parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help="Only retrain models with new data",
    )
    train_parser.add_argument(
        "--replay",
        action="store_true",
        default=False,
        help="Use replay buffer for training data sampling",
    )

    # ── pipeline ──────────────────────────────────────────
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full RPDE pipeline (scan → mine → validate → train)",
        description="End-to-end pipeline: scan golden moments, mine patterns, "
                    "validate statistically, save to library, train models.",
    )
    pipeline_parser.add_argument(
        "days",
        nargs="?",
        type=int,
        default=360,
        help="Number of days of history to scan (default: 360)",
    )
    pipeline_parser.add_argument(
        "--pair",
        default=None,
        action="append",
        dest="pairs",
        help="Specific pair(s) to process. Can be repeated. "
             "Omit to process all whitelist pairs.",
    )

    # ── report ────────────────────────────────────────────
    subparsers.add_parser(
        "report",
        help="Generate a comprehensive RPDE system status report",
    )

    # ── list ──────────────────────────────────────────────
    list_parser = subparsers.add_parser(
        "list",
        help="List patterns from the pattern library",
    )
    list_parser.add_argument(
        "--pair",
        default=None,
        help="Filter by currency pair",
    )
    list_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        dest="show_all",
        help="Include inactive/hibernated patterns",
    )
    list_parser.add_argument(
        "--min-tier",
        default=None,
        help="Minimum tier to show (PROBATIONARY, VALID, STRONG, GOD_TIER)",
    )

    # ── test ──────────────────────────────────────────────
    test_parser = subparsers.add_parser(
        "test",
        help="Test pattern matching on the latest market data",
        description="Load latest M5 data, extract features, and check which "
                    "active patterns match the current market conditions.",
    )
    test_parser.add_argument(
        "pair",
        nargs="?",
        default=None,
        help="Currency pair to test (required)",
    )
    test_parser.add_argument(
        "--bars",
        type=int,
        default=200,
        help="Number of M5 bars to load (default: 200)",
    )

    # ── backtest ────────────────────────────────────────
    bt_parser = subparsers.add_parser(
        "backtest",
        help="Backtest trained pattern models on historical data",
        description="Test pattern model discrimination ability on golden moments "
                    "and negative samples. Reports WR, AUC, profit factor at "
                    "different prediction thresholds.",
    )
    bt_parser.add_argument(
        "days",
        nargs="?",
        type=int,
        default=90,
        help="Days of historical data to test (default: 90)",
    )
    bt_parser.add_argument(
        "--pair",
        default=None,
        help="Test a specific pair only",
    )
    bt_parser.add_argument(
        "--thresholds",
        action="store_true",
        default=False,
        help="Test multiple prediction thresholds (sweep 0.0 to 4.0)",
    )

    # ── tft-train (Phase 2) ────────────────────────────────
    tft_train_parser = subparsers.add_parser(
        "tft-train",
        help="Train TFT model for a pair (Phase 2: Temporal Fusion Transformer)",
        description="Train a multi-timeframe Temporal Fusion Transformer "
                    "model using golden moments from the RPDE database. "
                    "Requires PyTorch and sufficient golden moments.",
    )
    tft_train_parser.add_argument(
        "pair",
        nargs="?",
        default=None,
        help="Currency pair to train. Omit to train all pairs.",
    )
    tft_train_parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Days of golden moments to use (default: all available)",
    )
    tft_train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max training epochs (default: from config)",
    )
    tft_train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: from config)",
    )

    # ── tft-status (Phase 2) ──────────────────────────────
    subparsers.add_parser(
        "tft-status",
        help="Show TFT model status for all pairs (Phase 2)",
    )

    # ── rl-train (Phase 3) ─────────────────────────────────
    rl_train_parser = subparsers.add_parser(
        "rl-train",
        help="Train RL agent for a pair (Phase 3: PPO Decision Engine)",
        description="Train a PPO-based RL decision engine that learns optimal"
                    " entry timing, position sizing, and skip decisions from"
                    " historical trade experiences. Requires PyTorch.",
    )
    rl_train_parser.add_argument(
        "pair",
        nargs="?",
        default=None,
        help="Currency pair to train. Omit to train all pairs.",
    )
    rl_train_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (default: from config)",
    )
    rl_train_parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Rollout buffer steps per update (default: from config)",
    )

    # ── rl-status (Phase 3) ────────────────────────────────
    subparsers.add_parser(
        "rl-status",
        help="Show RL agent status for all pairs (Phase 3)",
    )

    # ── rl-decide (Phase 3) ────────────────────────────────
    rl_decide_parser = subparsers.add_parser(
        "rl-decide",
        help="Test RL decision on latest data (Phase 3)",
        description="Load latest market data, run fusion + RL agent, and "
                    "show what the RL agent would decide for each pair.",
    )
    rl_decide_parser.add_argument(
        "pair",
        nargs="?",
        default=None,
        help="Currency pair. Omit for all pairs.",
    )

    # ── safety-check (Phase 3) ─────────────────────────────
    subparsers.add_parser(
        "safety-check",
        help="Run safety guard diagnostics (Phase 3)",
    )

    # ── learning-status (Phase 3) ──────────────────────────
    subparsers.add_parser(
        "learning-status",
        help="Show continuous learning loop status (Phase 3)",
    )

    # ── live-start (Phase 3: Live Engine) ──────────────────
    live_start_parser = subparsers.add_parser(
        "live-start",
        help="Start the RPDE live trading engine (Phase 3)",
        description="Initialize and start the LiveEngine decision chain. "
                    "Connects M5 bars -> Features -> PatternGate -> RL -> "
                    "Safety -> Execute. Use --live to enable real execution "
                    "(default: paper mode).",
    )
    live_start_parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        dest="live_mode",
        help="Enable live execution (default: paper mode)",
    )
    live_start_parser.add_argument(
        "--pair",
        default=None,
        action="append",
        dest="pairs",
        help="Specific pair(s) to trade. Can be repeated.",
    )

    # ── live-stop (Phase 3: Live Engine) ────────────────────
    subparsers.add_parser(
        "live-stop",
        help="Stop the RPDE live trading engine gracefully",
    )

    # ── live-status (Phase 3: Live Engine) ──────────────────
    subparsers.add_parser(
        "live-status",
        help="Show RPDE live engine status (positions, P&L, signals)",
    )

    return parser


# ═══════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════

def _cmd_scan(args):
    """Handle the 'scan' command."""
    try:
        from rpde.scanner import scan_pair, scan_all_pairs
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.scanner: {e}")
        return 1

    pair = args.pair.upper() if args.pair else None
    days = args.days

    if pair:
        print(f"\n  Scanning {pair} for {days} days of history...")
        print(f"  {'─' * 50}")
        result = scan_pair(pair, days=days)

        print(f"\n  {'=' * 50}")
        print(f"  SCAN RESULTS: {result.get('pair', pair)}")
        print(f"  {'=' * 50}")
        print(f"  Bars scanned:    {result.get('bars_scanned', 0):,}")
        print(f"  Golden moments:  {result.get('golden_moments', 0)}")
        print(f"    BUY moments:   {result.get('buy_moments', 0)}")
        print(f"    SELL moments:  {result.get('sell_moments', 0)}")
        print(f"  Avg move:        {result.get('avg_move_pips', 0)} pips")
        print(f"  Duration:        {result.get('duration_seconds', 0)}s")
        scan_id = result.get('scan_id', '')
        if scan_id:
            print(f"  Scan ID:         {scan_id}")
        if result.get('error'):
            print(f"  ERROR: {result['error']}")
        print(f"  {'=' * 50}\n")
    else:
        print(f"\n  Scanning ALL pairs for {days} days of history...")
        print(f"  {'─' * 50}")
        result = scan_all_pairs(days=days)

        print(f"\n  {'=' * 50}")
        print(f"  BATCH SCAN RESULTS")
        print(f"  {'=' * 50}")
        print(f"  Total pairs:     {result.get('total_pairs', 0)}")
        print(f"  Total moments:   {result.get('total_moments', 0):,}")
        print(f"  Duration:        {result.get('duration_seconds', 0)}s")
        scan_id = result.get('scan_id', '')
        if scan_id:
            print(f"  Batch ID:        {scan_id}")

        # Per-pair summary
        per_pair = result.get("per_pair", {})
        if per_pair:
            print(f"\n  {'Pair':<12} {'Moments':>8} {'BUY':>5} {'SELL':>5} "
                  f"{'Avg Move':>10} {'Duration':>8}")
            print(f"  {'─' * 12} {'─' * 8} {'─' * 5} {'─' * 5} "
                  f"{'─' * 10} {'─' * 8}")
            sorted_pairs = sorted(
                per_pair.items(),
                key=lambda x: x[1].get("golden_moments", 0),
                reverse=True,
            )
            for pair_name, data in sorted_pairs:
                print(
                    f"  {pair_name:<12} "
                    f"{data.get('golden_moments', 0):>8} "
                    f"{data.get('buy_moments', 0):>5} "
                    f"{data.get('sell_moments', 0):>5} "
                    f"{data.get('avg_move_pips', 0):>9.1f}p "
                    f"{data.get('duration_seconds', 0):>7.0f}s"
                )

        errors = result.get("errors", [])
        if errors:
            print(f"\n  ERRORS ({len(errors)}):")
            for err in errors:
                if isinstance(err, dict):
                    print(f"    - {err.get('pair', '?')}: {err.get('error', '?')}")
                else:
                    print(f"    - {err}")

        print(f"  {'=' * 50}\n")

    return 0


def _cmd_mine(args):
    """Handle the 'mine' command."""
    try:
        from rpde.pattern_miner import mine_patterns, mine_all_pairs
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.pattern_miner: {e}")
        return 1

    pair = args.pair.upper() if args.pair else None

    if pair:
        print(f"\n  Mining patterns for {pair}...")
        print(f"  {'─' * 50}")
        patterns = mine_patterns(pair)

        if patterns:
            _print_pattern_table(patterns, pair)
        else:
            print(f"\n  No patterns found for {pair}.")
            print(f"  Ensure golden moments have been scanned first:")
            print(f"    python -m rpde scan {pair}\n")
    else:
        print(f"\n  Mining patterns for ALL pairs...")
        print(f"  {'─' * 50}")
        result = mine_all_pairs()

        total = result.get("total_patterns", 0)
        per_pair = result.get("per_pair", {})

        print(f"\n  {'=' * 50}")
        print(f"  MINING RESULTS")
        print(f"  {'=' * 50}")
        print(f"  Total patterns mined: {total}")

        if per_pair:
            print(f"\n  {'Pair':<12} {'Patterns':>9}")
            print(f"  {'─' * 12} {'─' * 9}")
            for pair_name, pair_patterns in sorted(per_pair.items()):
                if pair_patterns:
                    print(f"  {pair_name:<12} {len(pair_patterns):>9}")

        print(f"  {'=' * 50}\n")

    return 0


def _cmd_validate(args):
    """Handle the 'validate' command."""
    try:
        from rpde.trainer import validate_and_update
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.trainer: {e}")
        return 1

    pair = args.pair.upper() if args.pair else None
    pairs = [pair] if pair else None

    print(f"\n  Running pattern validation & update...")
    if pair:
        print(f"  Pair: {pair}")
    else:
        print(f"  Pair: ALL")
    print(f"  {'─' * 50}")

    result = validate_and_update(pairs=pairs)

    print(f"\n  {'=' * 50}")
    print(f"  VALIDATION RESULTS")
    print(f"  {'=' * 50}")
    print(f"  Patterns checked:   {result.get('total_checked', 0)}")
    print(f"  Hibernated:         {result.get('hibernated', 0)}")
    print(f"  Reactivated:        {result.get('reactivated', 0)}")
    print(f"  Promoted:           {result.get('promoted', 0)}")
    print(f"  Demoted:            {result.get('demoted', 0)}")
    print(f"  Unchanged:          {result.get('unchanged', 0)}")
    print(f"  Duration:           {result.get('duration_seconds', 0)}s")

    errors = result.get("errors", [])
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")

    print(f"  {'=' * 50}\n")
    return 0


def _cmd_train(args):
    """Handle the 'train' command."""
    try:
        from rpde.trainer import train_models_only
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.trainer: {e}")
        return 1

    pair = args.pair.upper() if args.pair else None
    pairs = [pair] if pair else None

    mode_parts = []
    if args.incremental:
        mode_parts.append("incremental")
    if args.replay:
        mode_parts.append("replay")
    mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""

    print(f"\n  Training pattern models{mode_str}...")
    if pair:
        print(f"  Pair: {pair}")
    else:
        print(f"  Pair: ALL (with active patterns)")
    print(f"  {'─' * 50}")

    result = train_models_only(
        pairs=pairs,
        incremental=args.incremental,
        use_replay=args.replay,
    )

    print(f"\n  {'=' * 50}")
    print(f"  TRAINING RESULTS")
    print(f"  {'=' * 50}")
    print(f"  Models trained:     {result.get('models_trained', 0)}")
    print(f"  Duration:           {result.get('duration_seconds', 0)}s")

    per_pair = result.get("per_pair", {})
    if per_pair:
        print(f"\n  {'Pair':<12} {'Models':>7}")
        print(f"  {'─' * 12} {'─' * 7}")
        for pair_name, count in sorted(per_pair.items()):
            print(f"  {pair_name:<12} {count:>7}")

    errors = result.get("errors", [])
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    - {err}")

    print(f"  {'=' * 50}\n")
    return 0


def _cmd_pipeline(args):
    """Handle the 'pipeline' command."""
    try:
        from rpde.trainer import run_full_pipeline
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.trainer: {e}")
        return 1

    days = args.days
    pairs = [p.upper() for p in args.pairs] if args.pairs else None

    print(f"\n  Starting full RPDE pipeline...")
    print(f"  Days: {days}")
    if pairs:
        print(f"  Pairs: {', '.join(pairs)}")
    else:
        print(f"  Pairs: ALL (whitelist)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'─' * 50}")

    result = run_full_pipeline(days=days, pairs=pairs)

    # Print summary
    print(f"\n  {'=' * 55}")
    print(f"  PIPELINE SUMMARY")
    print(f"  {'=' * 55}")
    print(f"  Scan results:       {'OK' if result.get('scan_results') and not result['scan_results'].get('error') else 'FAILED'}")
    print(f"  Patterns found:     {result.get('patterns_found', 0)}")
    print(f"  Patterns validated: {result.get('patterns_validated', 0)}")
    print(f"  Models trained:     {result.get('models_trained', 0)}")
    print(f"  Duration:           {result.get('duration_seconds', 0)}s")

    patterns_per_pair = result.get("patterns_per_pair", {})
    if patterns_per_pair:
        print(f"\n  {'Pair':<12} {'Validated':>10}")
        print(f"  {'─' * 12} {'─' * 10}")
        for pair_name, count in sorted(patterns_per_pair.items(),
                                       key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {pair_name:<12} {count:>10}")

    skipped = result.get("skipped_steps", [])
    if skipped:
        print(f"\n  Skipped steps:      {', '.join(skipped)}")

    errors = result.get("errors", [])
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for err in errors[:20]:  # Limit output
            print(f"    - {err}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")

    print(f"  {'=' * 55}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'=' * 55}\n")

    return 0


def _cmd_report(args):
    """Handle the 'report' command."""
    try:
        from rpde.trainer import generate_report
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.trainer: {e}")
        return 1

    report = generate_report()
    print(report)
    return 0


def _cmd_list(args):
    """Handle the 'list' command."""
    try:
        from rpde.trainer import list_patterns
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.trainer: {e}")
        return 1

    pair = args.pair.upper() if args.pair else None
    active_only = not args.show_all
    min_tier = args.min_tier.upper() if args.min_tier else None

    patterns = list_patterns(
        pair=pair,
        active_only=active_only,
        min_tier=min_tier,
    )

    if not patterns:
        print(f"\n  No patterns found.")
        if active_only:
            print(f"  (Showing active only. Use --all to include hibernated.)")
        if pair:
            print(f"  (Filtered to: {pair})")
        print(f"  Run the pipeline first: python -m rpde pipeline\n")
        return 0

    # Group by pair
    by_pair = {}
    for p in patterns:
        p_name = p.get("pair", "UNKNOWN")
        if p_name not in by_pair:
            by_pair[p_name] = []
        by_pair[p_name].append(p)

    print(f"\n  {'=' * 75}")
    print(f"  PATTERN LIBRARY  —  {len(patterns)} patterns "
          f"(active={active_only}, pair={pair or 'ALL'})")
    print(f"  {'=' * 75}")

    tier_icons = {
        "GOD_TIER": "GOD",
        "STRONG": "STR",
        "VALID": "VAL",
        "PROBATIONARY": "PROB",
    }

    for p_name in sorted(by_pair.keys()):
        pair_pats = by_pair[p_name]
        print(f"\n  ── {p_name} ({len(pair_pats)} patterns) ──")
        print(f"  {'ID':<30} {'Dir':<5} {'Tier':<5} {'Occ':>5} "
              f"{'WR':>7} {'PF':>6} {'E[R]':>6} {'Active':>7}")
        print(f"  {'─' * 30} {'─' * 5} {'─' * 5} {'─' * 5} "
              f"{'─' * 7} {'─' * 6} {'─' * 6} {'─' * 7}")

        # Sort by tier rank then win rate
        tier_rank = {"GOD_TIER": 4, "STRONG": 3, "VALID": 2, "PROBATIONARY": 1}
        pair_pats.sort(
            key=lambda x: (tier_rank.get(x.get("tier", ""), 0),
                           x.get("win_rate", 0)),
            reverse=True,
        )

        for p in pair_pats:
            pid = p.get("pattern_id", "?")
            if len(pid) > 28:
                pid = pid[:26] + ".."
            direction = p.get("direction", "?")
            tier = p.get("tier", "?")
            tier_short = tier_icons.get(tier, tier[:3]) if tier else "?"
            occ = p.get("occurrences", 0)
            wr = p.get("win_rate", 0)
            pf = p.get("profit_factor", 0)
            er = p.get("avg_expected_r", 0)
            active = bool(p.get("is_active", True))
            active_str = "YES" if active else "no"

            print(f"  {pid:<30} {direction:<5} {tier_short:<5} {occ:>5} "
                  f"{wr:>6.1%} {pf:>6.2f} {er:>6.2f} {active_str:>7}")

    print(f"\n  {'=' * 75}\n")
    return 0


def _cmd_test(args):
    """Handle the 'test' command."""
    pair = args.pair.upper() if args.pair else None

    if not pair:
        print(f"\n  ERROR: Pair is required for test command.")
        print(f"  Usage: python -m rpde test EURJPY\n")
        return 1

    try:
        from rpde.trainer import test_pattern_matching
    except ImportError as e:
        print(f"ERROR: Cannot import rpde.trainer: {e}")
        return 1

    print(f"\n  Testing pattern matching for {pair}...")
    print(f"  Loading {args.bars} M5 bars...")
    print(f"  {'─' * 50}")

    result = test_pattern_matching(pair=pair, bars=args.bars)

    if result.get("error"):
        print(f"\n  ERROR: {result['error']}\n")
        return 1

    total = result.get("total_active_patterns", 0)
    matches = result.get("matches", [])

    print(f"\n  {'=' * 60}")
    print(f"  PATTERN MATCH RESULTS: {pair}")
    print(f"  {'=' * 60}")
    print(f"  Active patterns:  {total}")
    print(f"  Matching:         {len(matches)}")

    if matches:
        print(f"\n  {'#':<3} {'Pattern ID':<30} {'Dir':<5} {'Tier':<5} "
              f"{'Score':>6} {'WR':>7} {'E[R]':>6}")
        print(f"  {'─' * 3} {'─' * 30} {'─' * 5} {'─' * 5} "
              f"{'─' * 6} {'─' * 7} {'─' * 6}")

        for i, m in enumerate(matches, 1):
            pid = m.get("pattern_id", "?")
            if len(pid) > 28:
                pid = pid[:26] + ".."
            print(f"  {i:<3} {pid:<30} "
                  f"{m.get('direction', '?'):<5} "
                  f"{m.get('tier', '?'):<5} "
                  f"{m.get('match_score', 0):>6.3f} "
                  f"{m.get('win_rate', 0):>6.1%} "
                  f"{m.get('expected_r', 0):>6.2f}")
    else:
        print(f"\n  No patterns currently match the market conditions.")

    print(f"\n  {'=' * 60}")
    print(f"  Timestamp: {result.get('timestamp', '?')}")
    print(f"  {'=' * 60}\n")
    return 0


# ═══════════════════════════════════════════════════════════════
#  BACKTEST COMMAND HANDLER
# ═══════════════════════════════════════════════════════════════

def _cmd_backtest(args):
    """Handle the 'backtest' command."""
    try:
        from rpde.backtest import run_backtest
    except ImportError as e:
        print(f"\n  ERROR: Cannot import rpde.backtest: {e}\n")
        return 1

    pair_filter = args.pair.upper() if args.pair else None
    run_backtest(
        days=args.days,
        pair_filter=pair_filter,
        test_thresholds=args.thresholds,
    )
    return 0


# ═══════════════════════════════════════════════════════════════
#  TFT COMMAND HANDLERS (Phase 2)
# ═══════════════════════════════════════════════════════════════

def _cmd_tft_train(args):
    """Handle the 'tft-train' command."""
    try:
        from rpde.tft_model import train_tft_model, get_device, get_gpu_info
        from rpde.tft_dataset import build_training_dataset, build_dataloaders
        from rpde.config import (
            TFT_BATCH_SIZE, TFT_EPOCHS, TFT_PATIENCE,
            TFT_MIN_TRAINING_SAMPLES, TFT_NUM_WORKERS,
        )
    except ImportError as e:
        print(f"ERROR: TFT dependencies not available: {e}")
        print(f"  Install PyTorch: pip install torch")
        return 1

    pair = args.pair.upper() if args.pair else None

    print(f"\n  Training TFT model{' for ' + pair if pair else ' for all pairs'}...")
    print(f"  {'─' * 50}")

    gpu_info = get_gpu_info()
    if gpu_info['available']:
        print(f"  GPU: {gpu_info['name']} ({gpu_info['memory_total_gb']}GB)")
    else:
        print(f"  GPU: Not available (using CPU)")
    print(f"  Device: {get_device()}")

    try:
        if pair:
            # Train single pair
            dataset = build_training_dataset(pair, days=args.days)
            if len(dataset) < TFT_MIN_TRAINING_SAMPLES:
                print(f"\n  ERROR: Insufficient data for {pair}: "
                      f"{len(dataset)} samples (need {TFT_MIN_TRAINING_SAMPLES})")
                return 1

            train_loader, val_loader, train_ds, val_ds = build_dataloaders(
                dataset,
                batch_size=args.batch_size or TFT_BATCH_SIZE,
                num_workers=TFT_NUM_WORKERS,
            )

            model = train_tft_model(
                dataset, train_loader, val_loader,
                pair=pair,
                n_timeframes=len(train_ds.timeframes),
                n_features=train_ds.n_features,
            )

            print(f"\n  {'=' * 50}")
            print(f"  TFT TRAINING COMPLETE: {pair}")
            print(f"  {'=' * 50}")
            print(f"  Train samples: {len(train_ds)}")
            print(f"  Val samples:   {len(val_ds)}")
            print(f"  Duration:      {model.get('duration_seconds', '?')}s")
            return 0

        else:
            # Train all pairs
            from config.settings import PAIR_WHITELIST
            results = {}

            for p in PAIR_WHITELIST:
                print(f"\n  ── Training {p}... ──")
                try:
                    dataset = build_training_dataset(p)
                    if len(dataset) < TFT_MIN_TRAINING_SAMPLES:
                        print(f"  SKIPPED: {len(dataset)} samples "
                              f"(need {TFT_MIN_TRAINING_SAMPLES})")
                        results[p] = {'status': 'SKIPPED', 'samples': len(dataset)}
                        continue

                    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
                        dataset,
                        batch_size=TFT_BATCH_SIZE,
                        num_workers=TFT_NUM_WORKERS,
                    )

                    result = train_tft_model(
                        dataset, train_loader, val_loader,
                        pair=p,
                        n_timeframes=len(train_ds.timeframes),
                        n_features=train_ds.n_features,
                    )
                    results[p] = result
                except Exception as ex:
                    results[p] = {'status': 'FAILED', 'error': str(ex)}
                    print(f"  FAILED: {ex}")

            # Summary
            print(f"\n  {'=' * 55}")
            print(f"  TFT BATCH TRAINING SUMMARY")
            print(f"  {'=' * 55}")
            trained = sum(1 for r in results.values()
                        if r.get('status') == 'COMPLETED')
            for p, r in results.items():
                status = r.get('status', '?')
                epochs = r.get('epochs_trained', '?')
                print(f"  {p:<12} {status:<12} epochs={epochs}")
            print(f"  Total trained: {trained}/{len(results)}")
            print(f"  {'=' * 55}\n")
            return 0

    except Exception as e:
        print(f"\n  FATAL ERROR: {e}\n")
        log.exception("[RPDE_CLI] TFT training failed")
        return 1


def _cmd_tft_status(args):
    """Handle the 'tft-status' command."""
    try:
        from rpde.tft_model import get_device, get_gpu_info, load_tft_model
    except ImportError as e:
        print(f"ERROR: TFT not available: {e}")
        return 1

    from config.settings import PAIR_WHITELIST

    gpu = get_gpu_info()
    print(f"\n  {'=' * 55}")
    print(f"  TFT MODEL STATUS (Phase 2)")
    print(f"  {'=' * 55}")
    print(f"  PyTorch:     AVAILABLE")
    print(f"  GPU:         {'YES - ' + gpu['name'] + ' (' + gpu['memory_total_gb'] + 'GB)' if gpu['available'] else 'NO (CPU only)'}")
    print(f"  Device:      {get_device()}")

    trained = 0
    print(f"\n  {'Pair':<12} {'Status':<10} {'Samples':>8} {'Epochs':>7} "
          f"{'Val Loss':>10} {'Trained':<20}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 7} "
          f"{'─' * 10} {'─' * 20}")

    for pair in PAIR_WHITELIST:
        try:
            meta = load_tft_model(pair)
            if meta is not None:
                trained += 1
                samples = meta.get('training_samples', '?')
                epochs = meta.get('epochs_trained', '?')
                val_loss = f"{meta.get('best_val_loss', 0):.4f}"
                trained_at = meta.get('trained_at', '?')[:19]
                print(f"  {pair:<12} {'TRAINED':<10} {samples:>8} {epochs:>7} "
                      f"{val_loss:>10} {trained_at:<20}")
            else:
                print(f"  {pair:<12} {'NOT TRAINED':<10}")
        except Exception:
            print(f"  {pair:<12} {'ERROR':<10}")

    print(f"\n  Models trained: {trained}/{len(PAIR_WHITELIST)}")
    print(f"  {'=' * 55}\n")
    return 0


# ═══════════════════════════════════════════════════════════════
#  RL COMMAND HANDLERS (Phase 3)
# ═══════════════════════════════════════════════════════════════

def _cmd_rl_train(args):
    """Handle the 'rl-train' command."""
    try:
        from rpde.rl_agent import (
            train_rl_agent, load_rl_agent, get_rl_device, get_gpu_info_rl,
        )
        from rpde.config import RL_MIN_TRAINING_EPISODES
    except ImportError as e:
        print(f"ERROR: RL dependencies not available: {e}")
        print(f"  Install PyTorch: pip install torch gymnasium")
        return 1

    pair = args.pair.upper() if args.pair else None
    episodes = args.episodes or RL_MIN_TRAINING_EPISODES

    print(f"\n  Training RL agent{' for ' + pair if pair else ' for all pairs'}...")
    print(f"  Episodes: {episodes}")
    print(f"  {'─' * 50}")

    gpu = get_gpu_info_rl()
    if gpu['available']:
        print(f"  GPU: {gpu['name']} ({gpu['memory_total_gb']}GB)")
    else:
        print(f"  GPU: Not available (using CPU)")
    print(f"  Device: {get_rl_device()}")

    try:
        if pair:
            result = train_rl_agent(pair, episodes=episodes)
            print(f"\n  {'=' * 50}")
            print(f"  RL TRAINING COMPLETE: {pair}")
            print(f"  {'=' * 50}")
            print(f"  Episodes trained: {result.get('episodes_trained', '?')}")
            print(f"  Avg reward:       {result.get('avg_reward', 0):.4f}")
            print(f"  Policy loss:      {result.get('policy_loss', 0):.4f}")
            print(f"  Value loss:       {result.get('value_loss', 0):.4f}")
            print(f"  Duration:         {result.get('duration_seconds', '?')}s")
            return 0
        else:
            from config.settings import PAIR_WHITELIST
            results = {}
            for p in PAIR_WHITELIST:
                print(f"\n  ── Training {p}... ──")
                try:
                    r = train_rl_agent(p, episodes=episodes)
                    results[p] = r
                except Exception as ex:
                    results[p] = {'status': 'FAILED', 'error': str(ex)}
                    print(f"  FAILED: {ex}")

            trained = sum(1 for r in results.values()
                         if r.get('status') == 'COMPLETED')
            print(f"\n  {'=' * 55}")
            print(f"  RL BATCH TRAINING SUMMARY")
            print(f"  {'=' * 55}")
            for p, r in results.items():
                status = r.get('status', '?')
                avg_r = r.get('avg_reward', 0)
                print(f"  {p:<12} {status:<12} avg_reward={avg_r:.4f}")
            print(f"  Total trained: {trained}/{len(results)}")
            print(f"  {'=' * 55}\n")
            return 0

    except Exception as e:
        print(f"\n  FATAL ERROR: {e}\n")
        log.exception("[RPDE_CLI] RL training failed")
        return 1


def _cmd_rl_status(args):
    """Handle the 'rl-status' command."""
    try:
        from rpde.rl_agent import load_rl_agent, get_rl_device, get_gpu_info_rl
    except ImportError as e:
        print(f"ERROR: RL not available: {e}")
        return 1

    from config.settings import PAIR_WHITELIST

    gpu = get_gpu_info_rl()
    print(f"\n  {'=' * 65}")
    print(f"  RL AGENT STATUS (Phase 3: PPO Decision Engine)")
    print(f"  {'=' * 65}")
    print(f"  PyTorch:     {'AVAILABLE' if gpu['pytorch_available'] else 'NOT INSTALLED'}")
    print(f"  GPU:         {'YES - ' + gpu['name'] + ' (' + gpu['memory_total_gb'] + 'GB)' if gpu['available'] else 'NO (CPU only)'}")
    print(f"  Device:      {get_rl_device()}")

    trained = 0
    print(f"\n  {'Pair':<12} {'Status':<12} {'Episodes':>9} {'Avg R':>8} "
          f"{'Policy L':>9} {'Value L':>9} {'Last Train':<20}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 9} {'─' * 8} "
          f"{'─' * 9} {'─' * 9} {'─' * 20}")

    for pair in PAIR_WHITELIST:
        try:
            agent = load_rl_agent(pair)
            if agent is not None:
                trained += 1
                status = agent.get("status", {})
                print(f"  {pair:<12} {'TRAINED':<12} "
                      f"{status.get('episodes_trained', '?'):>9} "
                      f"{status.get('avg_reward', 0):>8.4f} "
                      f"{status.get('policy_loss', 0):>9.4f} "
                      f"{status.get('value_loss', 0):>9.4f} "
                      f"{str(status.get('last_trained', '?'))[:19]:<20}")
            else:
                print(f"  {pair:<12} {'NOT TRAINED':<12}")
        except Exception:
            print(f"  {pair:<12} {'ERROR':<12}")

    print(f"\n  Agents trained: {trained}/{len(PAIR_WHITELIST)}")
    print(f"  {'=' * 65}\n")
    return 0


def _cmd_rl_decide(args):
    """Handle the 'rl-decide' command."""
    try:
        from rpde.rl_agent import load_rl_agent
        from rpde.fusion_layer import fuse_all_signals
        from rpde.safety_guards import check_trade_safety
    except ImportError as e:
        print(f"ERROR: Phase 3 modules not available: {e}")
        return 1

    pair = args.pair.upper() if args.pair else None

    if pair:
        pairs = [pair]
    else:
        try:
            from config.settings import PAIR_WHITELIST
            pairs = PAIR_WHITELIST
        except ImportError:
            print(f"  ERROR: No pair specified and PAIR_WHITELIST not available")
            return 1

    print(f"\n  RL Decision Test for {len(pairs)} pair(s)...")
    print(f"  {'─' * 60}")

    for p in pairs:
        print(f"\n  ── {p} ──")
        try:
            agent = load_rl_agent(p)
            if agent is None:
                print(f"    RL agent not trained for {p} — skipping")
                continue

            # Show agent status
            status = agent.get("status", {})
            print(f"    Episodes trained: {status.get('episodes_trained', '?')}")
            print(f"    Avg reward:       {status.get('avg_reward', 0):.4f}")

            # Test with a simulated fusion signal
            from rpde.config import GATE_MIN_CONFIDENCE
            test_fusion = {
                "combined_confidence": GATE_MIN_CONFIDENCE + 0.05,
                "combined_expected_r": 0.8,
                "direction": "BUY",
                "signal_agreement": "XGB_TFT_AGREE",
                "reversal_warning": False,
                "recommendation": "TAKE",
                "tft_contribution": 0.45,
            }

            decision = agent.decide(
                fusion_result=test_fusion,
                market_state={
                    "session": "London",
                    "hour_utc": 10,
                    "day_of_week": 2,
                    "spread_ratio": 1.2,
                    "atr_percentile": 0.6,
                    "volatility_regime": "normal",
                },
                portfolio_state={
                    "open_positions_ratio": 0.2,
                    "daily_pnl_ratio": 0.0,
                    "weekly_pnl_ratio": 0.0,
                    "equity_ratio": 1.0,
                    "unrealized_pnl_ratio": 0.0,
                    "current_drawdown": 0.0,
                    "consecutive_losses": 0,
                    "hours_since_last_trade": 2.0,
                },
            )

            print(f"    Decision:     {decision.get('action_name', '?')}")
            print(f"    Entry:        {decision.get('entry', False)}")
            print(f"    Direction:    {decision.get('direction', '?')}")
            print(f"    Size:         {decision.get('size_r', 0)}R")
            print(f"    Stop type:    {decision.get('stop_type', '?')}")
            print(f"    TP target:    {decision.get('tp_r', 0)}R")
            print(f"    Confidence:   {decision.get('confidence', 0):.3f}")
            print(f"    Value est:    {decision.get('value', 0):.3f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            log.exception(f"[RPDE_CLI] rl-decide failed for {p}")

    print(f"\n  {'=' * 60}\n")
    return 0


def _cmd_safety_check(args):
    """Handle the 'safety-check' command."""
    try:
        from rpde.safety_guards import SafetyGuardSystem, check_trade_safety
    except ImportError as e:
        print(f"ERROR: Safety guards not available: {e}")
        return 1

    print(f"\n  {'=' * 55}")
    print(f"  SAFETY GUARD DIAGNOSTICS (Phase 3)")
    print(f"  {'=' * 55}")

    # Run diagnostic checks with simulated data
    system = SafetyGuardSystem()
    summary = system.get_summary()

    print(f"\n  Guard Configuration:")
    print(f"    Daily loss limit:   {summary.get('config', {}).get('max_daily_loss_pct', '?')}%")
    print(f"    Weekly loss limit:  {summary.get('config', {}).get('max_weekly_loss_pct', '?')}%")
    print(f"    Max positions:      {summary.get('config', {}).get('max_positions', '?')}")
    print(f"    Max per pair:       {summary.get('config', {}).get('max_per_pair', '?')}")
    print(f"    News buffer:        {summary.get('config', {}).get('news_buffer_min', '?')} min")
    print(f"    Spread multiplier:  {summary.get('config', {}).get('spread_multiplier', '?')}x")

    print(f"\n  System State:")
    print(f"    Shutdown:       {'YES — ' + str(system.get_shutdown_reason()) if system.is_shutdown() else 'NO'}")
    print(f"    Guards active:  {len(summary.get('guards', []))}")
    print(f"    History entries: {summary.get('total_checks', 0)}")

    # Test with a sample trade request
    print(f"\n  Test Trade Check (simulated BUY EURJPY):")
    result = check_trade_safety(
        trade_request={
            "pair": "EURJPY", "direction": "BUY", "size_r": 1.0,
            "stop_type": "medium", "tp_r": 1.0, "estimated_margin": 100.0,
            "signal_confidence": 0.78, "source": "rpde_v5",
        },
        account_state={
            "balance": 10000.0, "equity": 9850.0, "margin": 500.0,
            "free_margin": 9350.0, "margin_level": 1970.0,
            "daily_pnl_pct": -0.5, "weekly_pnl_pct": -1.0,
            "open_positions": 1, "consecutive_losses": 0,
        },
        market_state={
            "current_spread": 1.5, "average_spread": 1.0,
            "atr": 0.0030, "average_atr": 0.0025,
            "session": "London", "hour_utc": 10,
            "day_of_week": 2, "next_high_impact_news_minutes": 45,
            "is_weekend": False,
        },
    )
    print(f"    Approved:  {'YES' if result['approved'] else 'NO'}")
    print(f"    Guard:     {result.get('guard', 'ALL PASSED')}")
    print(f"    Reason:    {result.get('reason', 'OK')}")
    print(f"    Severity:  {result.get('severity', 'NONE')}")

    print(f"\n  {'=' * 55}\n")
    return 0


def _cmd_learning_status(args):
    """Handle the 'learning-status' command."""
    try:
        from rpde.experience_buffer import (
            ExperienceReplayBuffer, ContinuousLearningLoop,
            load_experiences_from_db,
        )
    except ImportError as e:
        print(f"ERROR: Experience buffer not available: {e}")
        return 1

    print(f"\n  {'=' * 60}")
    print(f"  CONTINUOUS LEARNING LOOP STATUS (Phase 3)")
    print(f"  {'=' * 60}")

    try:
        loop = ContinuousLearningLoop()
        health = loop.get_system_health()

        print(f"\n  Buffer Status:")
        buf_info = health.get("buffers", {})
        for pair_name, info in buf_info.items():
            print(f"    {pair_name:<12} {info.get('total_experiences', 0):>6} trades "
                  f"(last: {str(info.get('last_trade_time', 'never'))[:19]})")

        print(f"\n  Retrain Schedule:")
        schedule = health.get("schedule", {})
        for component, info in schedule.items():
            status = info.get("status", "?")
            due = info.get("due", False)
            last = info.get("last_retrained", "never")
            marker = " << DUE" if due else ""
            print(f"    {component:<15} {status:<12} last={str(last)[:16]}{marker}")

        print(f"\n  System Health:")
        print(f"    Total experiences: {health.get('total_experiences', 0)}")
        print(f"    Active buffers:    {health.get('active_buffers', 0)}")
        print(f"    Learning active:   {health.get('learning_active', False)}")

    except Exception as e:
        print(f"  ERROR: {e}")
        log.exception("[RPDE_CLI] learning-status failed")

    print(f"\n  {'=' * 60}\n")
    return 0


# ═══════════════════════════════════════════════════════════════
#  LIVE ENGINE COMMAND HANDLERS (Phase 3)
# ═══════════════════════════════════════════════════════════════

# Module-level reference to the running engine (for live-stop)
_running_engine = None


def _cmd_live_start(args):
    """Handle the 'live-start' command."""
    global _running_engine

    try:
        from rpde.live_engine import LiveEngine
    except ImportError as e:
        print(f"ERROR: LiveEngine not available: {e}")
        return 1

    paper_mode = not args.live_mode
    pairs = [p.upper() for p in args.pairs] if args.pairs else None

    mode_str = "PAPER" if paper_mode else "LIVE (REAL MONEY)"
    print(f"\n  Starting RPDE Live Engine...")
    print(f"  Mode:   {mode_str}")
    if pairs:
        print(f"  Pairs:  {', '.join(pairs)}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  {'─' * 50}")

    if not paper_mode:
        print(f"\n  WARNING: LIVE MODE ENABLED — REAL ORDERS WILL BE PLACED")
        print(f"  Press Ctrl+C within 5 seconds to abort...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print(f"\n  Aborted by user.\n")
            return 130

    try:
        engine = LiveEngine(pairs=pairs, paper_mode=paper_mode)
        ok = engine.initialize()

        if not ok:
            print(f"\n  ERROR: LiveEngine initialization failed.")
            print(f"  Check logs for details.\n")
            return 1

        _running_engine = engine

        status = engine.get_status()
        rl_count = status.get('rl_agents_loaded', 0)
        total_pairs = status.get('pairs', 0)

        print(f"\n  {'=' * 55}")
        print(f"  LIVE ENGINE STARTED")
        print(f"  {'=' * 55}")
        print(f"  Mode:            {mode_str}")
        print(f"  Pairs:           {total_pairs}")
        print(f"  RL agents:       {rl_count}/{total_pairs}")
        print(f"  PatternGate:     {'READY' if status.get('gate_ready') else 'NOT READY'}")
        print(f"  SafetyGuards:    {'ACTIVE' if status.get('safety_ready') else 'NOT READY'}")
        print(f"  LearningLoop:    {'ACTIVE' if status.get('learning_ready') else 'NOT READY'}")
        print(f"  {'=' * 55}")

        print(f"\n  Engine is running. Waiting for M5 bar signals...")
        print(f"  Press Ctrl+C to stop gracefully.\n")

        # Block until interrupted
        try:
            while engine.is_running:
                time.sleep(5)
                # Periodically print a status line
                s = engine.get_status()
                stats = s.get('stats', {})
                if stats.get('total_signals', 0) > 0:
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                          f"signals={stats['total_signals']} "
                          f"gate_skips={stats['gate_skips']} "
                          f"rl_skips={stats['rl_skips']} "
                          f"safety={stats['safety_blocks']} "
                          f"exec={stats['executed']} "
                          f"paper={stats['paper_executed']} "
                          f"daily_pnl=${s.get('daily_pnl_usd', 0):+.2f}")
        except KeyboardInterrupt:
            print(f"\n\n  Stopping live engine...")

        engine.shutdown()
        _running_engine = None

        final = engine.get_status()
        stats = final.get('stats', {})
        print(f"\n  {'=' * 55}")
        print(f"  LIVE ENGINE STOPPED")
        print(f"  {'=' * 55}")
        print(f"  Total signals:    {stats.get('total_signals', 0)}")
        print(f"  Gate skips:       {stats.get('gate_skips', 0)}")
        print(f"  RL skips:         {stats.get('rl_skips', 0)}")
        print(f"  Safety blocks:    {stats.get('safety_blocks', 0)}")
        print(f"  Executed (live):  {stats.get('executed', 0)}")
        print(f"  Executed (paper): {stats.get('paper_executed', 0)}")
        print(f"  Outcomes:         {stats.get('outcomes_recorded', 0)}")
        print(f"  Daily P&L:        ${final.get('daily_pnl_usd', 0):+.2f}")
        print(f"  {'=' * 55}\n")

        return 0

    except Exception as e:
        print(f"\n  FATAL ERROR: {e}\n")
        log.exception("[RPDE_CLI] live-start failed")
        return 1


def _cmd_live_stop(args):
    """Handle the 'live-stop' command."""
    global _running_engine

    if _running_engine is None:
        print(f"\n  No live engine is running.\n")
        return 0

    print(f"\n  Stopping RPDE Live Engine...")
    _running_engine.shutdown()
    _running_engine = None
    print(f"  Done.\n")
    return 0


def _cmd_live_status(args):
    """Handle the 'live-status' command."""
    global _running_engine

    if _running_engine is None:
        # No running engine — try to show a static status report
        try:
            from rpde.live_engine import LiveEngine
            from rpde.config import LIVE_PAPER_MODE_DEFAULT
        except ImportError as e:
            print(f"ERROR: LiveEngine not available: {e}")
            return 1

        print(f"\n  {'=' * 65}")
        print(f"  LIVE ENGINE STATUS (Phase 3)")
        print(f"  {'=' * 65}")
        print(f"  Engine state:    NOT RUNNING")

        # Show subsystem readiness without starting
        print(f"\n  Subsystem Check:")
        try:
            from rpde.pattern_gate import PatternGate
            gate = PatternGate()
            print(f"    PatternGate:    IMPORTABLE (not initialized)")
        except Exception:
            print(f"    PatternGate:    NOT AVAILABLE")

        try:
            from rpde.rl_agent import RLDecisionEngine
            print(f"    RL Agent:       IMPORTABLE (not loaded)")
        except Exception:
            print(f"    RL Agent:       NOT AVAILABLE")

        try:
            from rpde.safety_guards import SafetyGuardSystem
            print(f"    SafetyGuards:   IMPORTABLE")
        except Exception:
            print(f"    SafetyGuards:   NOT AVAILABLE")

        try:
            from rpde.experience_buffer import ContinuousLearningLoop
            print(f"    LearningLoop:   IMPORTABLE")
        except Exception:
            print(f"    LearningLoop:   NOT AVAILABLE")

        try:
            import MetaTrader5 as mt5
            connected = mt5.connection_status() == mt5.ConnectionStatus.CONNECTED
            print(f"    MT5 Connection: {'CONNECTED' if connected else 'NOT CONNECTED'}")
        except Exception:
            print(f"    MT5 Connection: NOT AVAILABLE")

        print(f"\n  Default mode:     {'PAPER' if LIVE_PAPER_MODE_DEFAULT else 'LIVE'}")
        print(f"  {'=' * 65}\n")
        return 0

    # Engine is running — show live status
    status = _running_engine.get_status()
    stats = status.get('stats', {})
    account = status.get('account', {})

    print(f"\n  {'=' * 65}")
    print(f"  LIVE ENGINE STATUS")
    print(f"  {'=' * 65}")
    print(f"  Mode:            {'PAPER' if status.get('paper_mode') else 'LIVE'}")
    print(f"  Running:         {'YES' if status.get('running') else 'NO'}")
    print(f"  Pairs:           {status.get('pairs', 0)}")
    print(f"  RL agents:       {status.get('rl_agents_loaded', 0)}/{status.get('pairs', 0)}")
    print(f"  PatternGate:     {'READY' if status.get('gate_ready') else 'NOT READY'}")
    print(f"  SafetyGuards:    {'ACTIVE' if status.get('safety_ready') else 'NOT READY'}")
    print(f"  LearningLoop:    {'ACTIVE' if status.get('learning_ready') else 'NOT READY'}")

    print(f"\n  Signal Pipeline:")
    print(f"    Total signals:    {stats.get('total_signals', 0)}")
    print(f"    Gate skips:       {stats.get('gate_skips', 0)}")
    print(f"    RL skips:         {stats.get('rl_skips', 0)}")
    print(f"    Safety blocks:    {stats.get('safety_blocks', 0)}")
    print(f"    Executed (live):  {stats.get('executed', 0)}")
    print(f"    Executed (paper): {stats.get('paper_executed', 0)}")
    print(f"    Execution fails:  {stats.get('execution_failures', 0)}")
    print(f"    Outcomes:         {stats.get('outcomes_recorded', 0)}")

    print(f"\n  Account:")
    print(f"    Balance:      ${account.get('balance', 0):,.2f}")
    print(f"    Equity:       ${account.get('equity', 0):,.2f}")
    print(f"    Margin level: {account.get('margin_level', 0):,.1f}%")
    print(f"    Daily P&L:    ${status.get('daily_pnl_usd', 0):+.2f}")
    print(f"    Weekly P&L:   ${status.get('weekly_pnl_usd', 0):+.2f}")

    open_positions = status.get('active_positions', 0)
    print(f"\n  Positions: {open_positions} open")

    print(f"\n  Performance:")
    print(f"    Consecutive wins:  {status.get('consecutive_wins', 0)}")
    print(f"    Consecutive losses: {status.get('consecutive_losses', 0)}")

    print(f"\n  Uptime: {status.get('uptime_seconds', 0)}s")
    print(f"  {'=' * 65}\n")
    return 0


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _print_pattern_table(patterns: list, pair: str):
    """Print a formatted table of patterns for a single pair."""
    print(f"\n  {'=' * 60}")
    print(f"  PATTERNS: {pair} ({len(patterns)} found)")
    print(f"  {'=' * 60}")
    print(f"  {'#':<4} {'Cluster':>8} {'Dir':<5} {'Occ':>5} "
          f"{'WR':>7} {'PF':>6} {'Avg Pips':>9} {'E[R]':>6}")
    print(f"  {'─' * 4} {'─' * 8} {'─' * 5} {'─' * 5} "
          f"{'─' * 7} {'─' * 6} {'─' * 9} {'─' * 6}")

    for i, p in enumerate(patterns, 1):
        cid = p.get("cluster_id", "?")
        direction = p.get("direction", "?")
        occ = p.get("occurrences", 0)
        wr = p.get("win_rate", 0)
        pf = p.get("profit_factor", 0)
        avg_pips = p.get("avg_profit_pips", 0)
        er = p.get("expected_r", 0)

        print(f"  {i:<4} {cid:>8} {direction:<5} {occ:>5} "
              f"{wr:>6.1%} {pf:>6.2f} {avg_pips:>8.1f}p {er:>6.2f}")

    # Top features for the best pattern
    if patterns and patterns[0].get("top_features"):
        best = patterns[0]
        print(f"\n  Top features (best pattern, cluster {best.get('cluster_id')}):")
        for feat, zscore in best["top_features"][:5]:
            print(f"    {feat:>25s}  z-score={zscore:.3f}")

    print(f"  {'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Print banner for all commands except help
    if args.command and args.command != "help":
        print(BANNER)

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command handler
    handlers = {
        "scan": _cmd_scan,
        "mine": _cmd_mine,
        "validate": _cmd_validate,
        "train": _cmd_train,
        "pipeline": _cmd_pipeline,
        "report": _cmd_report,
        "list": _cmd_list,
        "test": _cmd_test,
        "backtest": _cmd_backtest,
        "tft-train": _cmd_tft_train,
        "tft-status": _cmd_tft_status,
        "rl-train": _cmd_rl_train,
        "rl-status": _cmd_rl_status,
        "rl-decide": _cmd_rl_decide,
        "safety-check": _cmd_safety_check,
        "learning-status": _cmd_learning_status,
        "live-start": _cmd_live_start,
        "live-stop": _cmd_live_stop,
        "live-status": _cmd_live_status,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return_code = handler(args)
    except KeyboardInterrupt:
        print(f"\n\n  Interrupted by user.\n")
        return 130
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}\n")
        log.exception("[RPDE_CLI] Unhandled exception")
        return 1

    return return_code


if __name__ == "__main__":
    sys.exit(main())
