# =============================================================
# RPDE — Reverse Pattern Discovery Engine (v5.0)
#
# A self-evolving trading intelligence system that discovers
# profitable patterns by reverse-engineering historical market
# moves. Instead of hardcoded strategies, the AI scans price
# data for significant moves, extracts feature snapshots at
# those moments, clusters them into patterns, validates each
# pattern per-pair, and builds XGBoost classifiers to recognize
# them in real-time.
#
# Architecture:
#   Phase 1: Pattern Discovery (Offline)
#     Historical Data → Big Move Scanner → Feature Snapshots →
#     Pattern Miner → Per-Pair Validator → Pattern Library
#
#   Phase 2: Live Detection (Online)
#     Current Features → Pattern Model → Pattern Gate → Trade
#
# Key Design Principles:
#   - Per-pair personality: Each pair gets its own model and patterns
#   - Currency-specific patterns: Shared base currencies boost confidence
#   - Statistical validation: Minimum occurrences, win rate, profit factor
#   - Zero interference with v4.2 system (separate tables, separate models)
# =============================================================

__version__ = "5.0.0"
