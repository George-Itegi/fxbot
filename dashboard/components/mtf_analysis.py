# dashboard/components/mtf_analysis.py
# Multi-Timeframe Analysis — D1, H4, H1, M30, M15, M5

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config.settings import WATCHLIST


TIMEFRAMES = ["D1", "H4", "H1", "M30", "M15", "M5"]

SESSION_COLORS = {
    "SYDNEY":            "#2a2a3a",
    "TOKYO":             "#1a3a5c",
    "LONDON_OPEN":       "#2a4a6c",
    "LONDON_SESSION":    "#1a5c1a",
    "NY_LONDON_OVERLAP":"#5c3a00",
    "NY_AFTERNOON":      "#5c1a00",
}


def render():
    st.title("📊 Multi-Timeframe Analysis")
    st.caption("D1 → H4 → H1 → M30 → M15 → M5 | Session-tagged")

    # Symbol selector
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox("Select Symbol", WATCHLIST)
    with col2:
        run = st.button("▶ Analyse", type="primary",
                        use_container_width=True)

    if not run:
        st.info("Select a symbol and click Analyse.")
        return

    import MetaTrader5 as mt5
    from dotenv import load_dotenv
    import os
    load_dotenv()
    mt5.initialize()
    mt5.login(int(os.getenv('MT5_LOGIN', 0)),
              password=os.getenv('MT5_PASSWORD', ''),
              server=os.getenv('MT5_SERVER', ''))

    from data_layer.price_feed import get_candles

    st.markdown("---")
    st.subheader(f"📈 {symbol} — All Timeframes")

    # Build summary table
    summary = []
    for tf in TIMEFRAMES:
        with st.spinner(f"Fetching {tf}..."):
            df = get_candles(symbol, tf, 100)
            if df is None or df.empty:
                summary.append({'TF': tf, 'Error': 'No data'})
                continue

            last    = df.iloc[-1]
            prev    = df.iloc[-2]
            trend   = _get_trend(df)
            session = last.get('session', '?')
            change  = round((float(last['close']) - float(prev['close']))
                            / float(prev['close']) * 100, 3)

            summary.append({
                'TF':         tf,
                'Session':    session,
                'Close':      round(float(last['close']), 5),
                'Change %':   f"{change:+.3f}%",
                'Trend':      trend['label'],
                'Supertrend': '📈 BULL' if int(last.get('supertrend_dir',0))==1
                              else '📉 BEAR',
                'RSI':        round(float(last.get('rsi', 0)), 1),
                'StochRSI K': round(float(last.get('stoch_rsi_k', 0)), 1),
                'ADX':        round(float(last.get('adx', 0)), 1),
                'ATR (pips)': round(float(last.get('atr', 0)) /
                                    (0.01 if float(last['close'])>50
                                     else 0.0001), 1),
            })

    # Display summary table
    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df, use_container_width=True)

    mt5.shutdown()
    _show_confluence_verdict(summary)

def _get_trend(df: pd.DataFrame) -> dict:
    """Determine trend from EMA alignment."""
    last = df.iloc[-1]
    e9   = float(last.get('ema_9',  0))
    e21  = float(last.get('ema_21', 0))
    e50  = float(last.get('ema_50', 0))
    close= float(last['close'])

    if e9 > e21 > e50 and close > e9:
        return {'label': '📈 STRONG BULL', 'score': 2}
    elif e9 > e21 and close > e21:
        return {'label': '📈 BULL',        'score': 1}
    elif e9 < e21 < e50 and close < e9:
        return {'label': '📉 STRONG BEAR', 'score': -2}
    elif e9 < e21 and close < e21:
        return {'label': '📉 BEAR',        'score': -1}
    return {'label': '↔️ NEUTRAL',          'score': 0}


def _show_confluence_verdict(summary: list):
    """Show overall confluence verdict from all timeframes."""
    st.markdown("---")
    st.subheader("🎯 Confluence Verdict")

    bull_count = sum(1 for s in summary
                     if 'BULL' in str(s.get('Trend', '')))
    bear_count = sum(1 for s in summary
                     if 'BEAR' in str(s.get('Trend', '')))
    total      = len([s for s in summary if 'Error' not in s])

    col1, col2, col3 = st.columns(3)
    col1.metric("Bullish TFs",  f"{bull_count}/{total}")
    col2.metric("Bearish TFs",  f"{bear_count}/{total}")
    col3.metric("Alignment",
                "STRONG" if abs(bull_count-bear_count) >= 4
                else "MODERATE" if abs(bull_count-bear_count) >= 2
                else "WEAK")

    if bull_count >= 4:
        st.success(f"✅ BULLISH CONFLUENCE — {bull_count}/{total} timeframes bullish. "
                   "Look for BUY setups on M15/M5.")
    elif bear_count >= 4:
        st.error(f"📉 BEARISH CONFLUENCE — {bear_count}/{total} timeframes bearish. "
                 "Look for SELL setups on M15/M5.")
    elif bull_count >= 3:
        st.warning(f"⚠️ MILD BULLISH — {bull_count}/{total} timeframes bullish. "
                   "Wait for stronger confluence.")
    elif bear_count >= 3:
        st.warning(f"⚠️ MILD BEARISH — {bear_count}/{total} timeframes bearish. "
                   "Wait for stronger confluence.")
    else:
        st.info("↔️ NO CLEAR DIRECTION — Timeframes conflicted. Skip this symbol.")

    # Session advice
    st.markdown("---")
    st.subheader("⏰ Session Context")
    from data_layer.market_regime import get_session
    current = get_session()
    session_advice = {
        "LONDON_OPEN":       "🟡 MANIPULATION — Watch for Judas Swing. Avoid first breakout.",
        "LONDON_SESSION":    "🟢 EXPANSION — Strong directional moves. Best for trend entries.",
        "NY_LONDON_OVERLAP": "🟢 DISTRIBUTION — Highest volume. Best for breakouts.",
        "NY_AFTERNOON":      "🟡 LATE DISTRIBUTION — Liquidation phase. Reversals possible.",
        "TOKYO":             "🔵 ACCUMULATION — Tight ranges. Smart money builds positions.",
        "SYDNEY":            "⚪ PRICE DISCOVERY — Thin liquidity. Early ranges forming.",
    }
    st.info(f"Current: **{current}**\n\n{session_advice.get(current, 'Unknown session')}")
