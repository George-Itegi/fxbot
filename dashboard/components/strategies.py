# dashboard/components/strategies.py — Page 3: Strategy Performance

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.components.db_helper import get_trades, get_strategy_stats
from strategies.strategy_registry import REGISTRY


def render():
    st.title("🎯 Strategy Performance")

    # ── Registry status ───────────────────────────────────────
    st.subheader("Strategy Registry")
    reg_data = []
    for name, info in REGISTRY.items():
        reg_data.append({
            'Strategy':   name,
            'Phase':      info['phase'],
            'Status':     info['status'],
            'Win Rate':   f"{info['win_rate']}%",
            'Trades':     info['total_trades'],
            'P&L':        f"${info['total_pnl']:+.2f}",
            'Best State': ', '.join(info.get('best_state', [])),
        })
    reg_df = pd.DataFrame(reg_data)

    # Color-code phase
    def phase_color(val):
        colors = {
            'PAPER_TRADING': 'background-color: #1a3a5c',
            'LIVE_ACTIVE':   'background-color: #1a5c1a',
            'DEGRADING':     'background-color: #5c3a1a',
            'RETIRED':       'background-color: #3a1a1a',
        }
        return colors.get(val, '')

    st.dataframe(reg_df, use_container_width=True)

    st.markdown("---")

    # ── Database stats ────────────────────────────────────────
    st.subheader("Live Performance from Database")
    stats = get_strategy_stats()

    if stats.empty:
        st.info("No completed trades yet — run the bot to see stats.")
        return

    # Metric cards per strategy
    cols = st.columns(len(stats))
    for i, (_, row) in enumerate(stats.iterrows()):
        with cols[i]:
            st.metric(row['strategy'],
                      f"{row['win_rate']}% WR",
                      f"${row['total_pnl']:+.2f}")
            st.caption(f"{row['total_trades']} trades")

    st.markdown("---")

    # ── Win rate bar chart ────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win Rate by Strategy")
        fig = px.bar(stats, x='strategy', y='win_rate',
                     color='win_rate',
                     color_continuous_scale=['#ff4444','#ffaa00','#00ff88'],
                     range_color=[40, 80])
        fig.add_hline(y=60, line_dash='dash',
                      line_color='white',
                      annotation_text='Target 60%')
        fig.update_layout(height=350,
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Total P&L by Strategy")
        fig2 = px.bar(stats, x='strategy', y='total_pnl',
                      color='total_pnl',
                      color_continuous_scale=['#ff4444','#00ff88'])
        fig2.update_layout(height=350,
                           paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='white'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Best session per strategy ─────────────────────────────
    trades = get_trades(1000)
    if not trades.empty and 'session' in trades.columns:
        st.subheader("Best Session per Strategy")
        sess_perf = trades[trades['outcome'].notna()].groupby(
            ['strategy', 'session']).agg(
            win_rate=('outcome',
                      lambda x: round(
                          x.str.contains('WIN').sum()/len(x)*100, 1)),
            trades=('outcome', 'count')
        ).reset_index()
        st.dataframe(sess_perf, use_container_width=True)
