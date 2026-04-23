# dashboard/components/strategies.py v2.0
# Updated to show: strategy groups, consensus logic, new scoring thresholds

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.components.db_helper import get_trades, get_strategy_stats
from strategies.strategy_registry import REGISTRY
from strategies.strategy_engine import STRATEGY_GROUPS, STRATEGY_MIN_SCORES


def render():
    st.title("🎯 Strategy Performance")

    # ── Group overview ─────────────────────────────────────────
    st.subheader("Strategy Groups (Consensus Engine)")
    st.caption("Trades require signals from 2+ different groups. Same group = correlated = not counted as consensus.")

    group_cols = st.columns(len(STRATEGY_GROUPS))
    group_colors = {
        "TREND_FOLLOWING": "🔵",
        "SMC_STRUCTURE":   "🟢",
        "ORDER_FLOW":      "🟠",
        "MOMENTUM":        "🔴",
        "MEAN_REVERSION":  "🟡",
    }
    for i, (group, members) in enumerate(STRATEGY_GROUPS.items()):
        with group_cols[i]:
            icon = group_colors.get(group, "⚪")
            st.markdown(f"**{icon} {group.replace('_',' ')}**")
            for m in members:
                min_s = STRATEGY_MIN_SCORES.get(m, 70)
                phase = REGISTRY.get(m, {}).get('phase', '?')
                st.caption(f"• {m} (min:{min_s})")

    st.markdown("---")

    # ── Registry table ─────────────────────────────────────────
    st.subheader("Strategy Registry")
    reg_data = []
    for name, info in REGISTRY.items():
        group = next((g for g, members in STRATEGY_GROUPS.items() if name in members), "OTHER")
        reg_data.append({
            'Strategy':   name,
            'Group':      group,
            'Phase':      info['phase'],
            'Status':     info['status'],
            'Min Score':  STRATEGY_MIN_SCORES.get(name, 70),
            'Win Rate':   f"{info['win_rate']}%",
            'Trades':     info['total_trades'],
            'P&L':        f"${info['total_pnl']:+.2f}",
            'Best State': ', '.join(info.get('best_state', [])[:2]),
        })
    reg_df = pd.DataFrame(reg_data)
    st.dataframe(reg_df, use_container_width=True)

    st.markdown("---")

    # ── Hard state gates display ───────────────────────────────
    st.subheader("Hard State Gates (blocks strategies in wrong conditions)")
    gate_data = {
        "VWAP_MEAN_REVERSION":    "Only: BALANCED, REVERSAL_RISK",
        "OPENING_RANGE_BREAKOUT": "Only: TRENDING_STRONG, BREAKOUT_ACCEPTED, BALANCED",
        "DELTA_DIVERGENCE":       "Only: REVERSAL_RISK, BREAKOUT_REJECTED, BALANCED",
        "ORDER_FLOW_EXHAUSTION":  "Only: REVERSAL_RISK, BREAKOUT_REJECTED",
        "All others":             "No state restriction — run in any condition",
    }
    for strategy, rule in gate_data.items():
        is_restricted = strategy != "All others"
        icon = "🚫" if is_restricted else "✅"
        st.markdown(f"{icon} **{strategy}** → {rule}")

    st.markdown("---")

    # ── Database performance stats ──────────────────────────────
    st.subheader("Live Performance from Database")
    stats = get_strategy_stats()

    if stats.empty:
        st.info("No completed trades yet — run the bot to see results.")
        return

    cols = st.columns(min(len(stats), 5))
    for i, (_, row) in enumerate(stats.iterrows()):
        if i >= len(cols):
            break
        with cols[i]:
            wr = row.get('win_rate', 0)
            color = "normal" if wr >= 55 else "inverse"
            st.metric(
                row['strategy'],
                f"{wr:.1f}% WR",
                f"${row.get('total_pnl',0):+.2f}",
                delta_color=color
            )
            st.caption(f"{row.get('total_trades',0)} trades")

    st.markdown("---")

    # ── Win rate chart ─────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win Rate by Strategy")
        fig = px.bar(stats, x='strategy', y='win_rate',
                     color='win_rate',
                     color_continuous_scale=['#ff4444','#ffaa00','#00ff88'],
                     range_color=[40, 80])
        fig.add_hline(y=60, line_dash='dash', line_color='white',
                      annotation_text='Target 60%')
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("P&L by Strategy")
        fig2 = px.bar(stats, x='strategy', y='total_pnl',
                      color='total_pnl',
                      color_continuous_scale=['#ff4444','#00ff88'])
        fig2.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='white'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Consensus stats from trade journal ─────────────────────
    trades = get_trades(1000)
    if not trades.empty:
        st.subheader("Performance by Session")
        if 'session' in trades.columns and 'outcome' in trades.columns:
            sess_perf = trades[trades['outcome'].notna()].groupby(
                ['strategy','session']).agg(
                wins=('outcome', lambda x: x.str.contains('WIN').sum()),
                total=('outcome','count')
            ).reset_index()
            sess_perf['win_rate'] = (
                sess_perf['wins'] / sess_perf['total'] * 100).round(1)
            st.dataframe(sess_perf, use_container_width=True)

        st.subheader("Group Performance (Consensus)")
        st.caption("Which groups appeared in winning signals most often")
        if 'strategy' in trades.columns and 'outcome' in trades.columns:
            def get_group(s):
                for g, members in STRATEGY_GROUPS.items():
                    if s in members:
                        return g
                return "OTHER"
            trades['group'] = trades['strategy'].apply(get_group)
            grp = trades[trades['outcome'].notna()].groupby('group').agg(
                wins=('outcome', lambda x: x.str.contains('WIN').sum()),
                total=('outcome','count'),
                pnl=('profit_loss','sum')
            ).reset_index()
            grp['win_rate'] = (grp['wins']/grp['total']*100).round(1)
            grp['pnl'] = grp['pnl'].round(2)
            st.dataframe(grp, use_container_width=True)
