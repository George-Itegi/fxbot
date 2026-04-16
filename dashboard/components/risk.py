# dashboard/components/risk.py — Page 5: Risk & Account

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dashboard.components.db_helper import get_trades


def render():
    st.title("🛡️ Risk & Account")

    # ── Live account info ─────────────────────────────────────
    st.subheader("Account Overview")
    try:
        import MetaTrader5 as mt5
        from dotenv import load_dotenv
        import os
        load_dotenv()
        mt5.initialize()
        mt5.login(int(os.getenv('MT5_LOGIN', 0)),
                  password=os.getenv('MT5_PASSWORD', ''),
                  server=os.getenv('MT5_SERVER', ''))
        info = mt5.account_info()

        if info:
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Balance",    f"${info.balance:,.2f}")
            c2.metric("Equity",     f"${info.equity:,.2f}",
                      f"{info.equity-info.balance:+.2f}")
            c3.metric("Used Margin",f"${info.margin:,.2f}")
            c4.metric("Free Margin",f"${info.margin_free:,.2f}")
            c5.metric("Leverage",   f"1:{info.leverage}")
            c6.metric("Open P&L",   f"${info.profit:+.2f}",
                      delta_color="normal")

            # Daily loss limit gauge
            from config.settings import MAX_DAILY_LOSS_PERCENT
            max_loss = info.balance * (MAX_DAILY_LOSS_PERCENT / 100)
            st.markdown("---")
            st.subheader("Daily Loss Limit")
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = max(0, max_loss + info.profit),
                title = {'text': f"Remaining (Limit: ${max_loss:.0f})"},
                gauge = {
                    'axis': {'range': [0, max_loss]},
                    'bar':  {'color': '#00ff88'},
                    'steps': [
                        {'range': [0, max_loss*0.3], 'color': '#ff4444'},
                        {'range': [max_loss*0.3, max_loss*0.7],
                         'color': '#ffaa00'},
                        {'range': [max_loss*0.7, max_loss],
                         'color': '#1a5c1a'},
                    ],
                },
                number={'suffix': '$', 'valueformat': '.2f'},
            ))
            fig.update_layout(height=250,
                              paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

        mt5.shutdown()
    except Exception as e:
        st.warning(f"MT5 not connected: {e}")

    st.markdown("---")
    _show_risk_charts()


def _show_risk_charts():
    """Show drawdown and daily P&L charts."""
    trades = get_trades(500)
    if trades.empty:
        st.info("No trade data yet.")
        return

    df = trades[trades['profit_loss'].notna()].copy()
    df = df.sort_values('timestamp_open')
    df['cumulative'] = df['profit_loss'].cumsum()
    df['peak']       = df['cumulative'].cummax()
    df['drawdown']   = df['cumulative'] - df['peak']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drawdown")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp_open'], y=df['drawdown'],
            fill='tozeroy', mode='lines',
            line=dict(color='#ff4444'),
            fillcolor='rgba(255,68,68,0.2)',
            name='Drawdown'))
        fig.update_layout(height=300,
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Daily P&L")
        if 'timestamp_open' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp_open']).dt.date
            daily = df.groupby('date')['profit_loss'].sum().reset_index()
            daily.columns = ['Date', 'P&L']
            fig2 = px.bar(daily, x='Date', y='P&L',
                          color='P&L',
                          color_continuous_scale=['#ff4444','#00ff88'])
            fig2.update_layout(height=300,
                               paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='white'))
            st.plotly_chart(fig2, use_container_width=True)
