# dashboard/components/rpde.py — Page: RPDE Pattern Discovery Engine
#
# Real-time monitoring dashboard for the self-evolving pattern
# recognition system. Displays pattern library, fusion signals,
# RL decision engine, safety guards, and learning health.
#
# Usage:
#   In app.py sidebar radio, add "🧠 RPDE Engine" and route here.

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta


# ═════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ═════════════════════════════════════════════════════════════════

def _get_db():
    """Get a database cursor and connection, returning (cursor, conn).
    Returns (None, None) on failure so callers can degrade gracefully."""
    try:
        from rpde.database import _get_conn, init_rpde_tables
        init_rpde_tables()
        conn = _get_conn()
        c = conn.cursor(dictionary=True)
        return c, conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None, None


def _close_db(cursor, conn):
    """Safely close cursor and connection."""
    try:
        if cursor:
            cursor.close()
    except Exception:
        pass
    try:
        if conn:
            conn.close()
    except Exception:
        pass


def _safe_fmt(val, fmt="{:.2f}"):
    """Format a numeric value, returning 'N/A' on failure."""
    if val is None:
        return "N/A"
    try:
        return fmt.format(float(val))
    except (TypeError, ValueError):
        return str(val)


def _pct(val):
    """Format as percentage."""
    return _safe_fmt(val, "{:.1%}")


def _dt(val):
    """Format a datetime value for display."""
    if val is None:
        return "N/A"
    try:
        if isinstance(val, str):
            val = datetime.fromisoformat(val.replace("Z", "+00:00"))
        return val.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(val)


def _dt_ago(val):
    """Return human-readable 'X days ago' string."""
    if val is None:
        return "Never"
    try:
        if isinstance(val, str):
            val = datetime.fromisoformat(val.replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - val
        days = delta.total_seconds() / 86400
        if days < 0.01:
            return "Today"
        elif days < 1:
            return f"{int(days * 24)}h ago"
        elif days < 30:
            return f"{int(days)}d ago"
        else:
            return f"{int(days / 30)}mo ago"
    except Exception:
        return str(val)


# ═════════════════════════════════════════════════════════════════
#  COLOR HELPERS
# ═════════════════════════════════════════════════════════════════

def _wr_color(wr):
    """Return a colored win-rate badge."""
    if wr is None:
        return "<span style='color:gray'>N/A</span>"
    if wr >= 0.65:
        return f"<span style='color:#00ff88'>{_pct(wr)}</span>"
    elif wr >= 0.55:
        return f"<span style='color:#ffaa00'>{_pct(wr)}</span>"
    else:
        return f"<span style='color:#ff4444'>{_pct(wr)}</span>"


def _tier_badge(tier):
    """Return a styled tier badge."""
    styles = {
        "GOD_TIER":     "background:#ff4444;color:white;font-weight:bold",
        "STRONG":       "background:#00cc66;color:white;font-weight:bold",
        "VALID":        "background:#3388ff;color:white",
        "PROBATIONARY": "background:#666;color:white",
    }
    s = styles.get(tier, "background:#555;color:white")
    return f"<span style='{s};padding:2px 8px;border-radius:4px'>{tier}</span>"


def _rec_color(rec):
    """Color a recommendation."""
    colors = {"TAKE": "#00ff88", "CAUTION": "#ffaa00", "SKIP": "#ff4444"}
    c = colors.get(rec, "#888")
    return f"<span style='color:{c};font-weight:bold'>{rec}</span>"


def _bool_icon(val):
    """Return checkmark or X icon."""
    return "✅" if val else "❌"


# ═════════════════════════════════════════════════════════════════
#  TAB 1: PATTERN LIBRARY
# ═════════════════════════════════════════════════════════════════

def _tab_pattern_library():
    """Render the Pattern Library tab."""
    st.subheader("📖 Pattern Library")
    st.caption("Discovered and validated trading patterns across all pairs")

    c, conn = _get_db()
    if c is None:
        st.warning("Cannot connect to database — pattern data unavailable.")
        return

    try:
        c.execute(
            "SELECT * FROM rpde_pattern_library "
            "WHERE is_active = 1 ORDER BY win_rate DESC LIMIT 100"
        )
        patterns = c.fetchall()
    except Exception as e:
        st.error(f"Query failed: {e}")
        return
    finally:
        _close_db(c, conn)

    if not patterns:
        st.info("No active patterns in the library yet. "
                "Run pattern mining to discover patterns.")
        return

    # ── Summary metrics ────────────────────────────────────────
    total = len(patterns)
    tiers = {}
    best_patterns = {}
    for p in patterns:
        tier = p.get("tier", "UNKNOWN")
        tiers[tier] = tiers.get(tier, 0) + 1
        pair = p.get("pair", "?")
        wr = p.get("win_rate", 0)
        if pair not in best_patterns or wr > best_patterns[pair]["win_rate"]:
            best_patterns[pair] = p

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patterns", total)
    c2.metric("GOD_TIER", tiers.get("GOD_TIER", 0), delta_color="off")
    c3.metric("STRONG", tiers.get("STRONG", 0), delta_color="off")
    c4.metric("VALID", tiers.get("VALID", 0), delta_color="off")
    c5.metric("PROBATIONARY", tiers.get("PROBATIONARY", 0), delta_color="off")

    st.markdown("---")

    # ── Tier distribution bar ──────────────────────────────────
    if total > 0:
        st.markdown("**Tier Distribution**")
        tier_cols = st.columns(4)
        for i, (tier_name, count) in enumerate([
            ("GOD_TIER", tiers.get("GOD_TIER", 0)),
            ("STRONG", tiers.get("STRONG", 0)),
            ("VALID", tiers.get("VALID", 0)),
            ("PROBATIONARY", tiers.get("PROBATIONARY", 0)),
        ]):
            with tier_cols[i]:
                pct = count / total if total else 0
                st.progress(pct, text=f"{tier_name}: {count} ({pct:.0%})")

    st.markdown("---")

    # ── Best patterns per pair ─────────────────────────────────
    if best_patterns:
        with st.expander("🏆 Best Pattern Per Pair", expanded=False):
            best_rows = []
            for pair, bp in best_patterns.items():
                best_rows.append({
                    "Pair": bp.get("pair"),
                    "Pattern ID": bp.get("pattern_id"),
                    "Tier": bp.get("tier"),
                    "Win Rate": bp.get("win_rate"),
                    "Occurrences": bp.get("occurrences"),
                    "Avg Expected R": bp.get("avg_expected_r"),
                    "Profit Factor": bp.get("profit_factor"),
                    "Last Validated": _dt(bp.get("last_validated")),
                })
            st.dataframe(
                pd.DataFrame(best_rows),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ── Full pattern browser table ─────────────────────────────
    st.markdown("**Pattern Browser**")

    pair_filter = st.selectbox(
        "Filter by pair",
        options=["ALL"] + sorted(set(p.get("pair", "") for p in patterns)),
        format_func=lambda x: "All Pairs" if x == "ALL" else x,
        key="pattern_pair_filter",
    )

    tier_filter = st.selectbox(
        "Filter by tier",
        options=["ALL", "GOD_TIER", "STRONG", "VALID", "PROBATIONARY"],
        key="pattern_tier_filter",
    )

    filtered = patterns
    if pair_filter != "ALL":
        filtered = [p for p in filtered if p.get("pair") == pair_filter]
    if tier_filter != "ALL":
        filtered = [p for p in filtered if p.get("tier") == tier_filter]

    if not filtered:
        st.info("No patterns match the selected filters.")
        return

    # Build display dataframe
    rows = []
    for p in filtered:
        rows.append({
            "pattern_id": p.get("pattern_id", "?"),
            "pair": p.get("pair", "?"),
            "direction": p.get("direction", "?"),
            "tier": p.get("tier", "?"),
            "win_rate": p.get("win_rate"),
            "occurrences": p.get("occurrences", 0),
            "avg_expected_r": p.get("avg_expected_r", 0),
            "profit_factor": p.get("profit_factor", 0),
            "sharpe": p.get("sharpe_ratio", 0),
            "last_validated": p.get("last_validated"),
            "_full": p,  # hidden reference for expander
        })

    df = pd.DataFrame(rows)

    # Show main columns
    display_cols = [
        "pattern_id", "pair", "direction", "tier", "win_rate",
        "occurrences", "avg_expected_r", "profit_factor",
        "sharpe", "last_validated",
    ]
    st.dataframe(
        df[display_cols].style.format({
            "win_rate": lambda v: _pct(v),
            "avg_expected_r": lambda v: _safe_fmt(v, "{:+.2f}"),
            "profit_factor": lambda v: _safe_fmt(v, "{:.2f}"),
            "sharpe": lambda v: _safe_fmt(v, "{:.2f}"),
            "last_validated": lambda v: _dt(v),
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Pattern detail expanders ───────────────────────────────
    st.markdown("**Pattern Details**")
    search = st.text_input("Search pattern ID", key="pattern_search")
    shown = 0
    for p in filtered:
        pid = p.get("pattern_id", "")
        if search and search.lower() not in pid.lower():
            continue
        if shown >= 10:
            st.warning("Showing first 10 matches. Use search to narrow down.")
            break

        tier = p.get("tier", "PROBATIONARY")
        wr = p.get("win_rate", 0)
        direction = p.get("direction", "?")
        with st.expander(
            f"{pid} | {p.get('pair')} {direction} | "
            f"{tier} | WR: {_pct(wr)}",
            expanded=(tier == "GOD_TIER" and not search),
        ):
            detail_c1, detail_c2, detail_c3, detail_c4 = st.columns(4)
            detail_c1.metric("Occurrences", p.get("occurrences", 0))
            detail_c1.metric("Wins / Losses",
                             f"{p.get('wins', 0)} / {p.get('losses', 0)}")
            detail_c2.metric("Avg Expected R",
                             _safe_fmt(p.get("avg_expected_r"), "{:+.2f}"))
            detail_c2.metric("Profit Factor",
                             _safe_fmt(p.get("profit_factor"), "{:.2f}"))
            detail_c3.metric("Sharpe Ratio",
                             _safe_fmt(p.get("sharpe_ratio"), "{:.2f}"))
            detail_c3.metric("Max DD (pips)",
                             _safe_fmt(p.get("max_drawdown_pips"), "{:.1f}"))
            detail_c4.metric("Backtest Days", p.get("backtest_days", 0))
            detail_c4.metric("Max Consec Losses",
                             p.get("max_consecutive_losses", 0))

            # Top features
            top_features = p.get("top_features")
            if top_features and isinstance(top_features, dict):
                st.markdown("**Top Features:**")
                for feat_name, feat_val in list(top_features.items())[:8]:
                    st.write(f"- `{feat_name}`: {feat_val}")
        shown += 1


# ═════════════════════════════════════════════════════════════════
#  TAB 2: FUSION SIGNALS
# ═════════════════════════════════════════════════════════════════

def _tab_fusion_signals():
    """Render the Fusion Signals tab."""
    st.subheader("🔗 Fusion Signals")
    st.caption("XGBoost + TFT + Pattern Library combined signal overview")

    c, conn = _get_db()
    if c is None:
        st.warning("Cannot connect to database — signal data unavailable.")
        return

    try:
        # Load RL models as proxy for per-pair model status
        c.execute(
            "SELECT pair, trained_at, last_retrained_at, "
            "avg_reward, best_reward, device, is_active "
            "FROM rpde_rl_models ORDER BY pair"
        )
        rl_models = c.fetchall()

        # Load TFT models
        c.execute(
            "SELECT pair, trained_at, last_retrained_at, "
            "val_corr, val_r2, best_val_loss, device, is_active "
            "FROM rpde_tft_models ORDER BY pair"
        )
        tft_models = c.fetchall()

        # Load pattern trades (as proxy for recent signals)
        c.execute(
            "SELECT ticket, pair, direction, entry_time, exit_time, "
            "pattern_id, model_confidence, gate_confidence, "
            "pattern_win_rate_at_entry, pattern_tier_at_entry, "
            "profit_pips, profit_r, outcome, source "
            "FROM rpde_pattern_trades "
            "ORDER BY entry_time DESC LIMIT 50"
        )
        recent_trades = c.fetchall()

    except Exception as e:
        st.error(f"Query failed: {e}")
        return
    finally:
        _close_db(c, conn)

    # ── Per-pair model overview ────────────────────────────────
    st.markdown("**Per-Pair Model Status**")

    all_pairs = sorted(set(
        [m.get("pair") for m in rl_models]
        + [m.get("pair") for m in tft_models]
    ))

    if not all_pairs:
        st.info("No models found. Run training to populate this section.")
    else:
        overview_rows = []
        for pair in all_pairs:
            # Find matching models
            rl = next((m for m in rl_models if m["pair"] == pair), None)
            tft = next((m for m in tft_models if m["pair"] == pair), None)

            overview_rows.append({
                "Pair": pair,
                "XGB": "✅" if rl and rl.get("is_active") else "❌",
                "XGB Trained": _dt(rl.get("trained_at")) if rl else "N/A",
                "TFT": "✅" if tft and tft.get("is_active") else "❌",
                "TFT Trained": _dt(tft.get("trained_at")) if tft else "N/A",
                "TFT Val Loss": _safe_fmt(
                    tft.get("best_val_loss"), "{:.4f}") if tft else "N/A",
                "RL Active": "✅" if rl and rl.get("is_active") else "❌",
                "RL Avg Reward": _safe_fmt(
                    rl.get("avg_reward"), "{:.3f}") if rl else "N/A",
                "RL Device": rl.get("device", "N/A") if rl else "N/A",
            })

        st.dataframe(
            pd.DataFrame(overview_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Fusion signal status (from PatternGate if available) ───
    st.markdown("**Fusion Layer Status**")
    try:
        from rpde.pattern_gate import PatternGate
        gate = PatternGate()
        # Try to initialize if pairs config exists
        try:
            from config.settings import PAIR_WHITELIST
            gate.initialize(PAIR_WHITELIST)
        except Exception:
            pass

        status = gate.get_status()
        if status.get("is_initialized"):
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Models Loaded", status.get("models_loaded", 0))
            fc2.metric("Patterns Cached", status.get("patterns_loaded", 0))
            fc3.metric("TFT Managers", len(gate.tft_managers))

            if status.get("per_pair"):
                with st.expander("Per-Pair Fusion Details"):
                    for pair, ps in sorted(status["per_pair"].items()):
                        st.markdown(f"**{pair}**")
                        st.write(
                            f"  Model: {_bool_icon(ps.get('model_loaded'))} | "
                            f"Patterns: {ps.get('patterns_count', 0)} | "
                            f"Best WR: {_pct(ps.get('best_pattern_wr'))} | "
                            f"Best Tier: {ps.get('best_pattern_tier', 'N/A')}"
                        )
        else:
            st.warning("PatternGate not initialized. Start the bot first.")
    except Exception as e:
        st.info(f"PatternGate unavailable: {e}")

    st.markdown("---")

    # ── Recent pattern trades table ────────────────────────────
    st.markdown("**Recent Pattern Trades (Last 50)**")
    if not recent_trades:
        st.info("No pattern-based trades recorded yet.")
        return

    trade_rows = []
    for t in recent_trades:
        outcome = t.get("outcome", "?")
        rec = "TAKE" if outcome in ("WIN",) else (
            "CAUTION" if outcome == "BREAKEVEN" else "SKIP")
        trade_rows.append({
            "Time": _dt(t.get("entry_time")),
            "Pair": t.get("pair"),
            "Direction": t.get("direction"),
            "Model Conf": _safe_fmt(t.get("model_confidence"), "{:.2%}"),
            "Gate Conf": _safe_fmt(t.get("gate_confidence"), "{:.2%}"),
            "Pattern WR": _safe_fmt(t.get("pattern_win_rate_at_entry"), "{:.1%}"),
            "Tier": t.get("pattern_tier_at_entry", "?"),
            "R-Multiple": _safe_fmt(t.get("profit_r"), "{:+.2f}"),
            "Outcome": outcome,
        })

    st.dataframe(
        pd.DataFrame(trade_rows),
        use_container_width=True,
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════
#  TAB 3: RL DECISION ENGINE
# ═════════════════════════════════════════════════════════════════

def _tab_rl_decision_engine():
    """Render the RL Decision Engine tab."""
    st.subheader("🤖 RL Decision Engine")
    st.caption("Reinforcement learning agent status and recent decisions")

    c, conn = _get_db()
    if c is None:
        st.warning("Cannot connect to database — RL data unavailable.")
        return

    try:
        # RL model status
        c.execute(
            "SELECT pair, model_path, trained_at, last_retrained_at, "
            "episodes_trained, total_steps, avg_reward, best_reward, "
            "final_policy_loss, final_value_loss, device, "
            "n_parameters, is_active "
            "FROM rpde_rl_models ORDER BY pair"
        )
        rl_models = c.fetchall()

        # RL training history
        c.execute(
            "SELECT pair, training_id, started_at, completed_at, "
            "duration_seconds, status, episodes_trained, total_steps, "
            "avg_reward, best_reward, policy_loss, value_loss, "
            "entropy, device, notes "
            "FROM rpde_rl_training_log ORDER BY started_at DESC LIMIT 20"
        )
        rl_log = c.fetchall()

        # Trade experiences (for RL decisions)
        c.execute(
            "SELECT pair, entry_time, exit_time, direction, outcome, "
            "rl_action, rl_action_name, rl_predicted_value, "
            "fusion_confidence, profit_r "
            "FROM rpde_trade_experiences "
            "WHERE rl_action IS NOT NULL "
            "ORDER BY entry_time DESC LIMIT 50"
        )
        experiences = c.fetchall()

    except Exception as e:
        st.error(f"Query failed: {e}")
        return
    finally:
        _close_db(c, conn)

    # ── Per-pair agent status ──────────────────────────────────
    st.markdown("**Per-Pair Agent Status**")

    if not rl_models:
        st.info("No RL agents trained yet. Run training to populate this section.")
    else:
        rl1, rl2 = st.columns([1, 3])

        with rl1:
            trained_count = sum(
                1 for m in rl_models if m.get("is_active"))
            total_agents = len(rl_models)
            st.metric("Active Agents", f"{trained_count}/{total_agents}")

            if rl_models:
                best_agent = max(
                    rl_models, key=lambda m: m.get("avg_reward", 0) or 0)
                st.metric(
                    "Best Avg Reward",
                    _safe_fmt(best_agent.get("avg_reward"), "{:.3f}"),
                    delta=best_agent.get("pair", "?"),
                    delta_color="normal",
                )

        with rl2:
            agent_rows = []
            for m in rl_models:
                agent_rows.append({
                    "Pair": m.get("pair"),
                    "Trained": _bool_icon(m.get("is_active")),
                    "Episodes": m.get("episodes_trained", 0),
                    "Steps": m.get("total_steps", 0),
                    "Avg Reward": _safe_fmt(m.get("avg_reward"), "{:.3f}"),
                    "Best Reward": _safe_fmt(m.get("best_reward"), "{:.3f}"),
                    "Policy Loss": _safe_fmt(
                        m.get("final_policy_loss"), "{:.4f}"),
                    "Value Loss": _safe_fmt(
                        m.get("final_value_loss"), "{:.4f}"),
                    "Device": m.get("device", "N/A"),
                    "Parameters": f"{m.get('n_parameters', 0):,}",
                    "Last Trained": _dt(m.get("last_retrained_at")),
                })
            st.dataframe(
                pd.DataFrame(agent_rows),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ── Last N decisions table ─────────────────────────────────
    st.markdown("**Recent RL Decisions (Last 50)**")
    if not experiences:
        st.info("No RL decisions recorded yet.")
    else:
        dec_rows = []
        for exp in experiences:
            action_name = exp.get("rl_action_name", "N/A")
            predicted_val = exp.get("rl_predicted_value", 0)
            outcome = exp.get("outcome", "?")
            was_correct = (
                (action_name == "TAKE_LONG" and outcome == "WIN") or
                (action_name == "TAKE_SHORT" and outcome == "WIN") or
                (action_name == "SKIP" and outcome in ("LOSS",))
            )

            dec_rows.append({
                "Time": _dt(exp.get("entry_time")),
                "Pair": exp.get("pair"),
                "Direction": exp.get("direction"),
                "RL Action": action_name,
                "Predicted Value": _safe_fmt(predicted_val, "{:.3f}"),
                "Fusion Conf": _safe_fmt(
                    exp.get("fusion_confidence"), "{:.2%}"),
                "Actual R": _safe_fmt(exp.get("profit_r"), "{:+.2f}"),
                "Outcome": outcome,
                "Correct?": "✅" if was_correct else "❌",
            })
        st.dataframe(
            pd.DataFrame(dec_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── Performance metrics ────────────────────────────────────
    st.markdown("**RL Performance Metrics**")
    if experiences:
        take_exps = [e for e in experiences
                     if e.get("rl_action_name") in ("TAKE_LONG", "TAKE_SHORT")]
        if take_exps:
            take_wins = sum(
                1 for e in take_exps if e.get("outcome") == "WIN")
            take_total = len(take_exps)
            take_wr = take_wins / take_total if take_total else 0
            avg_r = (sum(float(e.get("profit_r", 0) or 0)
                         for e in take_exps) / take_total)

            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Trades with RL TAKE", take_total)
            pm2.metric("Win Rate (RL TAKE)",
                       _pct(take_wr),
                       delta_color="normal" if take_wr >= 0.5 else "inverse")
            pm3.metric("Avg R (RL TAKE)",
                       _safe_fmt(avg_r, "{:+.2f}"),
                       delta_color="normal" if avg_r > 0 else "inverse")
            pm4.metric("Skip Trades",
                       sum(1 for e in experiences
                           if e.get("rl_action_name") == "SKIP"))

    # ── Training history ───────────────────────────────────────
    with st.expander("Training History (Last 20)"):
        if not rl_log:
            st.info("No training history recorded.")
        else:
            log_rows = []
            for entry in rl_log:
                dur = entry.get("duration_seconds", 0)
                log_rows.append({
                    "Pair": entry.get("pair"),
                    "Started": _dt(entry.get("started_at")),
                    "Duration": f"{int(dur)}s" if dur else "N/A",
                    "Episodes": entry.get("episodes_trained", 0),
                    "Status": entry.get("status", "?"),
                    "Avg Reward": _safe_fmt(entry.get("avg_reward"), "{:.3f}"),
                    "Entropy": _safe_fmt(entry.get("entropy"), "{:.4f}"),
                    "Notes": entry.get("notes", "")[:80] if entry.get("notes") else "",
                })
            st.dataframe(
                pd.DataFrame(log_rows),
                use_container_width=True,
                hide_index=True,
            )


# ═════════════════════════════════════════════════════════════════
#  TAB 4: SAFETY GUARDS
# ═════════════════════════════════════════════════════════════════

def _tab_safety_guards():
    """Render the Safety Guards tab."""
    st.subheader("🛡️ Safety Guards")
    st.caption("Non-overridable human safety rails between AI and execution")

    # ── Try to get live safety system ──────────────────────────
    safety_system = None
    try:
        from rpde.safety_guards import SafetyGuardSystem
        safety_system = SafetyGuardSystem()
    except Exception:
        pass

    # ── Overall status ─────────────────────────────────────────
    is_shutdown = False
    shutdown_reason = None

    if safety_system:
        summary = safety_system.get_summary()
        is_shutdown = summary.get("is_shutdown", False)
        shutdown_reason = summary.get("shutdown_reason")
        total_checks = summary.get("total_checks", 0)
        guard_breakdown = summary.get("guard_breakdown", {})

        sc1, sc2, sc3 = st.columns(3)
        if is_shutdown:
            sc1.error("🔴 SYSTEM SHUTDOWN ACTIVE")
            if shutdown_reason:
                st.error(f"**Reason:** {shutdown_reason}")
        else:
            sc1.success("🟢 System Operating Normally")

        sc2.metric("Total Guard Checks", total_checks)

        failed_count = sum(
            v.get("failed_soft", 0) + v.get("failed_hard", 0)
            for v in guard_breakdown.values()
        )
        sc3.metric("Total Failures", failed_count,
                   delta_color="inverse" if failed_count > 0 else "off")

    else:
        st.warning("SafetyGuardSystem not instantiated. "
                   "Showing database events only.")

    st.markdown("---")

    # ── Per-guard statistics ───────────────────────────────────
    if safety_system and guard_breakdown:
        st.markdown("**Guard Breakdown**")
        guard_names = {
            "margin_call": "Margin Call (HARD)",
            "max_drawdown": "Max Drawdown (HARD)",
            "position_limit": "Position Limit (SOFT)",
            "spread_filter": "Spread Filter (SOFT)",
            "news_filter": "News Filter (SOFT)",
            "weekend_filter": "Weekend Filter (SOFT)",
            "consecutive_losses": "Consecutive Losses (SOFT)",
            "session_quality": "Session Quality (SOFT)",
            "volatility_extreme": "Volatility Extreme (SOFT)",
            "shutdown_cooldown": "Shutdown Cooldown (HARD)",
            "system_shutdown": "System Shutdown (HARD)",
            "all_guards": "All Guards Passed",
        }

        gb_rows = []
        for guard_name, stats in guard_breakdown.items():
            passed = stats.get("passed", 0)
            failed_soft = stats.get("failed_soft", 0)
            failed_hard = stats.get("failed_hard", 0)
            total = passed + failed_soft + failed_hard
            fail_rate = ((failed_soft + failed_hard) / total
                         if total > 0 else 0)

            gb_rows.append({
                "Guard": guard_names.get(guard_name, guard_name),
                "Passed": passed,
                "Soft Fail": failed_soft,
                "Hard Fail": failed_hard,
                "Total": total,
                "Failure Rate": _pct(fail_rate),
            })

        if gb_rows:
            st.dataframe(
                pd.DataFrame(gb_rows),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ── Recent guard events from DB ────────────────────────────
    st.markdown("**Recent Guard Events (Last 50)**")
    c, conn = _get_db()
    if c is None:
        st.warning("Database unavailable for event history.")
    else:
        try:
            c.execute(
                "SELECT pair, guard_name, severity, action_taken, "
                "trade_approved, reason, created_at "
                "FROM rpde_safety_events "
                "ORDER BY created_at DESC LIMIT 50"
            )
            events = c.fetchall()
        except Exception as e:
            st.error(f"Query failed: {e}")
            events = []
        finally:
            _close_db(c, conn)

        if not events:
            st.info("No safety events recorded.")
        else:
            ev_rows = []
            for ev in events:
                ev_rows.append({
                    "Time": _dt(ev.get("created_at")),
                    "Pair": ev.get("pair", "?"),
                    "Guard": ev.get("guard_name", "?"),
                    "Passed": _bool_icon(ev.get("trade_approved")),
                    "Severity": ev.get("severity", "?"),
                    "Action": ev.get("action_taken", "?"),
                    "Reason": (ev.get("reason", "") or "")[:80],
                })
            st.dataframe(
                pd.DataFrame(ev_rows),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ── Manual controls ────────────────────────────────────────
    st.markdown("**Manual Controls**")
    mc1, mc2 = st.columns(2)

    with mc1:
        reason_input = st.text_input(
            "Shutdown reason",
            placeholder="Enter reason for manual shutdown...",
            key="safety_shutdown_reason",
        )
        if st.button("🔴 Force Shutdown", type="primary",
                      use_container_width=True):
            if safety_system:
                reason = reason_input.strip() or "Manual dashboard shutdown"
                safety_system.force_shutdown(reason)
                st.success(f"Shutdown triggered: {reason}")
                st.rerun()
            else:
                st.error("SafetyGuardSystem not available.")

    with mc2:
        if st.button("🟢 Reset Shutdown", use_container_width=True):
            if safety_system:
                result = safety_system.reset_shutdown()
                if result:
                    st.success("Shutdown flag cleared. "
                               "Cooldown may still apply.")
                    st.rerun()
                else:
                    st.info("No active shutdown to reset.")
            else:
                st.error("SafetyGuardSystem not available.")


# ═════════════════════════════════════════════════════════════════
#  TAB 5: LEARNING HEALTH
# ═════════════════════════════════════════════════════════════════

def _tab_learning_health():
    """Render the Learning Health tab."""
    st.subheader("📊 Learning Health")
    st.caption("Continuous learning schedules, buffer stats, and retrain history")

    # ── Try to get live learning loop ──────────────────────────
    cll = None
    try:
        from rpde.experience_buffer import ContinuousLearningLoop
        cll = ContinuousLearningLoop()
    except Exception:
        pass

    # ── Schedule status ────────────────────────────────────────
    st.markdown("**Retrain Schedule**")

    if cll:
        schedule = cll.check_all_schedules()
        summary = schedule.get("summary", {})
        per_pair = schedule.get("per_pair", {})
        any_due = schedule.get("any_due", False)

        if any_due:
            st.warning("⚠️ Some retraining tasks are overdue!")
        else:
            st.success("✅ All retraining schedules are up to date.")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("XGB Due", summary.get("xgb", 0))
        sc2.metric("TFT Due", summary.get("tft", 0))
        sc3.metric("RL Due", summary.get("rl", 0))
        sc4.metric("Pattern Due", summary.get("pattern", 0))

        if per_pair:
            sched_rows = []
            schedule_intervals = {
                "xgb": 7, "tft": 14, "rl": 7, "pattern": 30,
            }
            for pair, ps in per_pair.items():
                row = {"Pair": pair}
                for comp in ("xgb", "tft", "rl", "pattern"):
                    due = ps.get(f"{comp}_retrain_due", False)
                    days = ps.get(f"{comp}_days_since")
                    interval = schedule_intervals.get(comp, 7)
                    if days is None:
                        row[comp.upper()] = "Never"
                    elif due:
                        row[comp.upper()] = f"🔴 {int(days)}d"
                    else:
                        progress = min(days / interval, 1.0)
                        row[comp.upper()] = f"🟢 {int(days)}/{interval}d"
                sched_rows.append(row)

            st.dataframe(
                pd.DataFrame(sched_rows),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("ContinuousLearningLoop not available. "
                "Showing database history only.")

    st.markdown("---")

    # ── Buffer statistics ──────────────────────────────────────
    st.markdown("**Experience Buffer Statistics**")

    if cll:
        buffers = cll.get_all_buffers()
        if not buffers:
            st.info("No experience buffers initialized.")
        else:
            buf_rows = []
            for pair, buf in buffers.items():
                stats = buf.stats()
                fill = stats.get("fill_ratio", 0)
                buf_rows.append({
                    "Pair": pair,
                    "Size": stats.get("size", 0),
                    "Max Size": stats.get("max_size", 0),
                    "Fill Ratio": _pct(fill),
                    "Dirty": "⚠️" if stats.get("dirty") else "✅",
                    "Total Added": stats.get("total_added", 0),
                    "Total Sampled": stats.get("total_sampled", 0),
                    "Last Save": _dt_ago(stats.get("last_save")),
                })
            st.dataframe(
                pd.DataFrame(buf_rows),
                use_container_width=True,
                hide_index=True,
            )

            # Fill ratio progress bars
            st.markdown("**Buffer Fill Levels**")
            for pair, buf in buffers.items():
                stats = buf.stats()
                fill = stats.get("fill_ratio", 0)
                label = f"{pair}: {stats.get('size', 0)}/{stats.get('max_size', 0)}"
                st.progress(fill, text=label)
    else:
        # Show from files
        try:
            from pathlib import Path
            exp_dir = Path(__file__).resolve().parent.parent.parent / \
                "rpde" / "models" / "experience"
            if exp_dir.exists():
                import json
                files = list(exp_dir.glob("*_experience.json"))
                if files:
                    buf_rows = []
                    for fp in files:
                        pair = fp.stem.replace("_experience", "").upper()
                        try:
                            data = json.loads(fp.read_text())
                            exp_count = len(data.get("experiences", []))
                            buf_rows.append({
                                "Pair": pair,
                                "File Experiences": exp_count,
                                "File Size (KB)": round(
                                    fp.stat().st_size / 1024, 1),
                            })
                        except Exception:
                            buf_rows.append({"Pair": pair,
                                             "File Experiences": "Error",
                                             "File Size (KB)": "Error"})
                    if buf_rows:
                        st.dataframe(
                            pd.DataFrame(buf_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.info("No experience buffer files found.")
            else:
                st.info("Experience directory does not exist yet.")
        except Exception as e:
            st.info(f"Cannot read buffer files: {e}")

    st.markdown("---")

    # ── Recent retrain history ─────────────────────────────────
    st.markdown("**Retrain History (Last 50)**")
    c, conn = _get_db()
    if c is None:
        st.warning("Database unavailable for retrain history.")
    else:
        try:
            c.execute(
                "SELECT pair, component, action, triggered_at, "
                "completed_at, duration_seconds, status, notes "
                "FROM rpde_learning_log "
                "ORDER BY triggered_at DESC LIMIT 50"
            )
            learning_log = c.fetchall()
        except Exception as e:
            st.error(f"Query failed: {e}")
            learning_log = []
        finally:
            _close_db(c, conn)

        if not learning_log:
            st.info("No retrain history recorded.")
        else:
            log_rows = []
            for entry in learning_log:
                status = entry.get("status", "?")
                status_icon = "✅" if status == "SUCCESS" else (
                    "⚠️" if status == "PARTIAL" else
                    "❌" if status == "FAILED" else "⏳")
                log_rows.append({
                    "Time": _dt(entry.get("triggered_at")),
                    "Pair": entry.get("pair", "?"),
                    "Component": entry.get("component", "?"),
                    "Action": entry.get("action", "?"),
                    "Duration": f"{int(entry.get('duration_seconds', 0))}s"
                    if entry.get("duration_seconds") else "N/A",
                    "Status": f"{status_icon} {status}",
                    "Notes": (entry.get("notes", "") or "")[:60],
                })
            st.dataframe(
                pd.DataFrame(log_rows),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ── Performance analysis ───────────────────────────────────
    st.markdown("**Performance Analysis**")
    c, conn = _get_db()
    if c is None:
        st.warning("Database unavailable for performance analysis.")
    else:
        try:
            # Get recent pattern trades for analysis
            c.execute(
                "SELECT pair, outcome, profit_r, profit_pips, entry_time "
                "FROM rpde_pattern_trades "
                "WHERE outcome IS NOT NULL "
                "ORDER BY entry_time DESC LIMIT 200"
            )
            recent = c.fetchall()

            # Get pattern stats for decay detection
            c.execute(
                "SELECT pattern_id, last_30_win_rate, all_time_win_rate, "
                "is_decaying, updated_at "
                "FROM rpde_pattern_stats "
                "ORDER BY updated_at DESC LIMIT 50"
            )
            pattern_stats = c.fetchall()

        except Exception as e:
            st.error(f"Query failed: {e}")
            recent = []
            pattern_stats = []
        finally:
            _close_db(c, conn)

        if not recent:
            st.info("No trade data for performance analysis.")
        else:
            wins = sum(1 for t in recent if t.get("outcome") == "WIN")
            total = len(recent)
            recent_wr = wins / total if total else 0
            avg_r = (sum(float(t.get("profit_r", 0) or 0)
                         for t in recent) / total)

            pa1, pa2, pa3 = st.columns(3)
            pa1.metric("Recent Win Rate", _pct(recent_wr))
            pa2.metric("Avg R-Multiple", _safe_fmt(avg_r, "{:+.2f}"))
            pa3.metric("Total Trades", total)

            # Decay detection
            decaying = [ps for ps in pattern_stats
                        if ps.get("is_decaying")]
            if decaying:
                st.warning(f"⚠️ **{len(decaying)} pattern(s) showing decay:**")
                decay_rows = []
                for ps in decaying:
                    l30 = ps.get("last_30_win_rate", 0)
                    all_t = ps.get("all_time_win_rate", 0)
                    decay_rows.append({
                        "Pattern ID": ps.get("pattern_id"),
                        "Last 30 WR": _pct(l30),
                        "All-Time WR": _pct(all_t),
                        "Decay": _pct(all_t - l30) if l30 < all_t else "None",
                        "Updated": _dt(ps.get("updated_at")),
                    })
                st.dataframe(
                    pd.DataFrame(decay_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("✅ No pattern decay detected.")

    # ── Regime change detection (via experience buffer) ────────
    if cll:
        with st.expander("Regime Change Detection"):
            try:
                from rpde.experience_buffer import detect_regime_change
                pairs = list(cll.get_all_buffers().keys())
                regime_rows = []
                for pair in pairs:
                    try:
                        regime = detect_regime_change(pair)
                        if regime:
                            regime_rows.append({
                                "Pair": pair,
                                "Changed": _bool_icon(regime.get("detected")),
                                "Type": regime.get("change_type", "N/A"),
                                "Score": _safe_fmt(
                                    regime.get("change_score"), "{:.3f}"),
                                "Details": str(
                                    regime.get("details", ""))[:60],
                            })
                    except Exception:
                        regime_rows.append({
                            "Pair": pair, "Changed": "N/A",
                            "Type": "Error", "Score": "N/A",
                            "Details": "Analysis failed",
                        })
                if regime_rows:
                    st.dataframe(
                        pd.DataFrame(regime_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No pairs available for regime analysis.")
            except ImportError:
                st.info("Regime change detection not available.")
            except Exception as e:
                st.info(f"Regime analysis error: {e}")


# ═════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═════════════════════════════════════════════════════════════════

def render():
    """Main render function for the RPDE dashboard page."""
    st.title("🧠 RPDE Pattern Discovery Engine")
    st.markdown("Real-time monitoring of the self-evolving pattern recognition system")

    # ── Header with timestamp and refresh ──────────────────────
    now = datetime.now(timezone.utc)
    rc1, rc2 = st.columns([4, 1])
    with rc1:
        st.caption(f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    with rc2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # ── Quick status overview ──────────────────────────────────
    try:
        from rpde.pattern_gate import PatternGate
        gate = PatternGate()
        try:
            from config.settings import PAIR_WHITELIST
            gate.initialize(PAIR_WHITELIST)
        except Exception:
            pass

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Pattern Models", gate.get_status().get("models_loaded", 0))
        q2.metric("Cached Patterns",
                  gate.get_status().get("patterns_loaded", 0))
        q3.metric("TFT Managers", len(gate.tft_managers))
        q4.metric("Initialized", _bool_icon(gate.is_initialized))
    except Exception:
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Pattern Models", "N/A")
        qc2.metric("Cached Patterns", "N/A")
        qc3.metric("TFT Managers", "N/A")
        qc4.metric("Initialized", "N/A")

    # ── Safety system status pill ──────────────────────────────
    try:
        from rpde.safety_guards import SafetyGuardSystem
        safety = SafetyGuardSystem()
        if safety.is_shutdown():
            st.error(
                f"🔴 **SAFETY SHUTDOWN ACTIVE** — "
                f"{safety.get_shutdown_reason() or 'Unknown reason'}"
            )
    except Exception:
        pass

    st.markdown("---")

    # ── Tab navigation ─────────────────────────────────────────
    tabs = st.tabs([
        "📖 Pattern Library",
        "🔗 Fusion Signals",
        "🤖 RL Decision Engine",
        "🛡️ Safety Guards",
        "📊 Learning Health",
    ])

    with tabs[0]:
        _tab_pattern_library()

    with tabs[1]:
        _tab_fusion_signals()

    with tabs[2]:
        _tab_rl_decision_engine()

    with tabs[3]:
        _tab_safety_guards()

    with tabs[4]:
        _tab_learning_health()
