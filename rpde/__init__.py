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
#   Phase 1: Pattern Discovery (Offline) — 100% COMPLETE
#     Historical Data → Big Move Scanner → Feature Snapshots →
#     Pattern Miner → Per-Pair Validator → Pattern Library
#     → Per-Pair XGBoost Pattern Model
#
#   Phase 2: Temporal Fusion Transformer (Multi-TF Deep Learning)
#     Multi-TF Raw OHLCV → Separate Timeframe Encoders →
#     Cross-TF Attention → Variable Selection Network →
#     3 Output Heads → Pattern Fusion Layer → Enhanced Gate
#
#   Phase 3: RL Decision Engine (Planned)
#     State → Action → Reward → Policy Update (Future)
#
# Data Flow (Live):
#   Raw OHLCV ─┐
#   (4 timeframes) ├→ TFT Model ─┐
#                  │               ├→ Fusion Layer ─→ Pattern Gate ─→ Trade
#   93 features ─┘               │
#                              │
#   Pattern Library ─────────────┘
#
# Key Design Principles:
#   - Per-pair personality: Each pair gets its own model and patterns
#   - Currency-specific patterns: Shared base currencies boost confidence
#   - Statistical validation: Minimum occurrences, win rate, profit factor
#   - Zero interference with v4.2 system (separate tables, separate models)
#
# Modules:
#   scanner.py          — Big Move Scanner (finds golden moments)
#   feature_snapshot.py — Extracts 93 features at golden moments
#   pattern_miner.py    — DBSCAN clustering into candidate patterns
#   pattern_validator.py — Statistical validation with tier assignment
#   pattern_library.py  — Per-pair pattern storage (CRUD)
#   pattern_model.py    — Per-pair XGBoost regression model (Phase 1 L1)
#   pattern_gate.py     — Final decision layer with TFT fusion (Phase 1+2 L2)
#   tft_dataset.py      — Multi-TF PyTorch dataset builder (Phase 2)
#   tft_model.py        — Multi-TF TFT architecture (Phase 2)
#   fusion_layer.py     — XGB + TFT prediction fusion (Phase 2)
#   database.py         — Separate RPDE database tables
#   config.py           — All tunable parameters
#   trainer.py          — Full training pipeline orchestrator
#   __main__.py         — CLI entry point
# =============================================================

__version__ = "5.0.0"
