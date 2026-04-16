# dashboard/components/trade_journal.py — Page 2: Trade Journal

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.components.db_helper import get_trades


def render():
    st.title("📋 Trade Journal")

    trades = get_trades(1000)

    if trades.empty:
        st.info("No trades in database yet. Run the bot first.")
        return

    # ── Filters ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        symbols = ['All'] + sorted(trades['symbol'].dropna().unique().tolist())
        symbol  = st.selectbox("Symbol", symbols)
    with col2:
        strategies = ['All'] + sorted(
            trades['strategy'].dropna().unique().tolist())
        strategy = st.selectbox("Strategy", strategies)
    with col3:
        outcomes = ['All', 'WIN_TP1', 'WIN_TP2', 'LOSS', 'BREAKEVEN']
        outcome  = st.selectbox("Outcome", outcomes)

    # Apply filters
    df = trades.copy()
    if symbol   != 'All': df = df[df['symbol']   == symbol]
    if strategy != 'All': df = df[df['strategy'] == strategy]
    if outcome  != 'All': df = df[df['outcome']  == outcome]

    # ── Summary metrics ───────────────────────────────────────
    total  = len(df)
    wins   = len(df[df['outcome'].str.contains('WIN', na=False)])
    losses = len(df[df['outcome'] == 'LOSS'])
    wr     = round(wins / total * 100, 1) if total > 0 else 0
    pnl    = df['profit_loss'].sum() if 'profit_loss' in df else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Trades", total)
    c2.metric("Win Rate",     f"{wr}%")
    c3.metric("Total P&L",    f"${pnl:+.2f}")
    c4.metric("Wins / Losses",f"{wins} / {losses}")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Win/Loss Distribution")
        if not df.empty and 'outcome' in df.columns:
            counts = df['outcome'].value_counts().reset_index()
            counts.columns = ['Outcome', 'Count']
            fig = px.pie(counts, values='Count', names='Outcome',
                         color_discrete_map={
                             'WIN_TP1':'#00ff88','WIN_TP2':'#00cc66',
                             'LOSS':'#ff4444','BREAKEVEN':'#ffaa00'})
            fig.update_layout(height=300,
                              paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("P&L by Session")
        if 'session' in df.columns and 'profit_loss' in df.columns:
            sess_pnl = df.groupby('session')['profit_loss'].sum().reset_index()
            fig2 = px.bar(sess_pnl, x='session', y='profit_loss',
                          color='profit_loss',
                          color_continuous_scale=['#ff4444','#00ff88'])
            fig2.update_layout(height=300,
                               paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='white'))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Trades ({total})")
    display_cols = [c for c in [
        'timestamp_open','symbol','direction','strategy',
        'entry_price','exit_price','sl_price','tp_price',
        'lot_size','profit_loss','outcome','ai_score','session'
    ] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=400)
