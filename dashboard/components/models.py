# dashboard/components/models.py — Page 6: AI Model Status

import streamlit as st
import os
import pandas as pd
from dashboard.components.db_helper import get_trades
from strategies.strategy_registry import REGISTRY
from ai_engine.phase_manager import get_phase_summary


def render():
    st.title("🤖 AI Model Status")

    # ── Phase manager status ───────────────────────────────────
    st.subheader("Strategy Lifecycle")
    phase_data = []
    for name, info in REGISTRY.items():
        phase_data.append({
            'Strategy':    name,
            'Version':     info.get('version', '?'),
            'Phase':       info['phase'],
            'Status':      info['status'],
            'Win Rate':    f"{info['win_rate']}%",
            'Trades':      info['total_trades'],
            'Total P&L':   f"${info['total_pnl']:+.2f}",
            'Promoted Paper': info.get('promoted_to_paper', '—'),
            'Promoted Live':  info.get('promoted_to_live', '—'),
        })
    st.dataframe(pd.DataFrame(phase_data), use_container_width=True)

    # Phase promotion thresholds reminder
    col1, col2, col3 = st.columns(3)
    col1.info("**Virtual → Paper**\n50+ trades, 58%+ WR")
    col2.info("**Paper → Live**\n30+ trades, 62%+ WR")
    col3.info("**Live → Degrading**\n20+ trades, <45% WR")

    st.markdown("---")

    # ── XGBoost status ────────────────────────────────────────
    st.subheader("XGBoost Signal Classifier")
    xgb_path = os.path.join(os.path.dirname(__file__),
                             '..', '..', 'ai_engine',
                             'models', 'xgb_model.pkl')
    xgb_exists = os.path.exists(xgb_path)

    trades = get_trades(1000)
    trade_count = len(trades)

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", "✅ Trained" if xgb_exists else "⏳ Not trained")
    col2.metric("Trades in DB", trade_count)
    col3.metric("Needed to train", "50")

    if xgb_exists:
        mod_time = os.path.getmtime(xgb_path)
        import datetime
        trained_at = datetime.datetime.fromtimestamp(mod_time)
        st.success(f"Model trained at: {trained_at.strftime('%Y-%m-%d %H:%M')}")
    else:
        progress = min(trade_count / 50, 1.0)
        st.progress(progress,
                    text=f"Collecting trades: {trade_count}/50")
        st.warning("XGBoost not trained yet. "
                   "Run the bot to collect 50+ trades.")

    st.markdown("---")

    # ── LSTM status ───────────────────────────────────────────
    st.subheader("LSTM Price Predictor")
    lstm_path = os.path.join(os.path.dirname(__file__),
                              '..', '..', 'ai_engine',
                              'models', 'lstm_model.keras')
    lstm_exists = os.path.exists(lstm_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", "✅ Trained" if lstm_exists else "⏳ Not trained")
    col2.metric("Trades in DB", trade_count)
    col3.metric("Needed to train", "100")

    if lstm_exists:
        mod_time = os.path.getmtime(lstm_path)
        import datetime
        trained_at = datetime.datetime.fromtimestamp(mod_time)
        st.success(f"Model trained at: {trained_at.strftime('%Y-%m-%d %H:%M')}")
    else:
        progress = min(trade_count / 100, 1.0)
        st.progress(progress,
                    text=f"Collecting trades: {trade_count}/100")
        st.warning("LSTM not trained yet. "
                   "Run the bot to collect 100+ trades.")

    st.markdown("---")

    # ── Manual retrain button ──────────────────────────────────
    st.subheader("Manual Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Retrain XGBoost", use_container_width=True):
            with st.spinner("Training XGBoost..."):
                try:
                    from ai_engine.xgboost_classifier import train_model
                    ok = train_model()
                    if ok:
                        st.success("XGBoost retrained successfully!")
                    else:
                        st.warning("Not enough data to train yet.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with col2:
        if st.button("📊 Check Promotions", use_container_width=True):
            with st.spinner("Checking strategy promotions..."):
                from ai_engine.phase_manager import check_all_promotions
                check_all_promotions()
                st.success("Promotion check complete!")
                st.text(get_phase_summary())
