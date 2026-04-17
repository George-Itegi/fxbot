# =============================================================
# ai_engine/virtual_simulator.py
# PURPOSE: Replay historical candles and simulate strategy
# performance WITHOUT any real money or live connection.
# This is PHASE 1 — strategies must pass here before demo.
# =============================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from core.logger import get_logger
from database.db_manager import get_connection
import os

load_dotenv()
log = get_logger(__name__)

# Promotion threshold to pass virtual phase
VIRTUAL_MIN_TRADES  = 50
VIRTUAL_MIN_WINRATE = 58.0   # 58% win rate required


def fetch_historical_candles(symbol: str,
                             timeframe=None,
                             count: int = 2000) -> pd.DataFrame | None:
    """
    Fetch a large batch of historical candles for simulation.
    Default 2000 candles of H1 = ~83 days of data.
    """
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_H1
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators needed by strategies."""
    c = df['close']; h = df['high']; l = df['low']
    df['ema_9']   = c.ewm(span=9,   adjust=False).mean()
    df['ema_21']  = c.ewm(span=21,  adjust=False).mean()
    df['ema_50']  = c.ewm(span=50,  adjust=False).mean()
    df['ema_200'] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    hl  = h - l
    hc  = (h - c.shift()).abs()
    lc  = (l - c.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.ewm(com=13, min_periods=14).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_mid']   = sma20
    df['vol_ma20'] = df['tick_volume'].rolling(20).mean()
    # ADX
    dm_p = (h - h.shift()).clip(lower=0).where((h-h.shift())>(l.shift()-l), 0)
    dm_m = (l.shift()-l).clip(lower=0).where((l.shift()-l)>(h-h.shift()), 0)
    atr_adx = tr.ewm(com=13, min_periods=14).mean()
    di_p = 100 * dm_p.ewm(com=13, min_periods=14).mean() / atr_adx
    di_m = 100 * dm_m.ewm(com=13, min_periods=14).mean() / atr_adx
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    df['adx'] = dx.ewm(com=13, min_periods=14).mean()
    # StochRSI
    rsi_min = df['rsi'].rolling(14).min()
    rsi_max = df['rsi'].rolling(14).max()
    stoch_raw = 100 * (df['rsi'] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df['stoch_rsi_k'] = stoch_raw.rolling(3).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()
    # Supertrend
    hl2    = (h + l) / 2
    upper_b= hl2 + (3.0 * df['atr'])
    lower_b= hl2 - (3.0 * df['atr'])
    st     = pd.Series(np.nan, index=df.index)
    st_dir = pd.Series(0,      index=df.index)
    st.iloc[0] = upper_b.iloc[0]; st_dir.iloc[0] = -1
    for i in range(1, len(df)):
        prev = st.iloc[i-1]; prev_d = st_dir.iloc[i-1]
        ub = min(upper_b.iloc[i], upper_b.iloc[i-1]) if c.iloc[i] <= prev else upper_b.iloc[i]
        lb = max(lower_b.iloc[i], lower_b.iloc[i-1]) if c.iloc[i] >= prev else lower_b.iloc[i]
        if prev_d == -1:
            if c.iloc[i] > prev: st.iloc[i] = lb; st_dir.iloc[i] = 1
            else:                 st.iloc[i] = ub; st_dir.iloc[i] = -1
        else:
            if c.iloc[i] < prev: st.iloc[i] = ub; st_dir.iloc[i] = -1
            else:                 st.iloc[i] = lb; st_dir.iloc[i] = 1
    df['supertrend_val'] = st.round(5)
    df['supertrend_dir'] = st_dir
    df.dropna(inplace=True)
    return df

def simulate_trade(df: pd.DataFrame, signal: dict,
                   entry_idx: int) -> dict:
    """
    Simulate a trade outcome by replaying future candles.
    Checks if price hits TP1, TP2, or SL first.
    Returns trade result dict.
    """
    direction  = signal['direction']
    entry      = signal['entry_price']
    sl         = signal['sl_price']
    tp1        = signal['tp1_price']
    tp2        = signal['tp2_price']
    pip_size   = 0.01 if entry > 50 else 0.0001
    max_candles= 48   # Max hold time = 48 H1 candles (2 days)

    outcome    = 'TIMEOUT'
    exit_price = entry
    pnl_pips   = 0.0
    candles_held = 0

    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        candles_held = i - entry_idx

        if direction == 'BUY':
            if candle['low'] <= sl:
                outcome    = 'LOSS'
                exit_price = sl
                pnl_pips   = (sl - entry) / pip_size
                break
            if candle['high'] >= tp2:
                outcome    = 'WIN_TP2'
                exit_price = tp2
                pnl_pips   = (tp2 - entry) / pip_size
                break
            if candle['high'] >= tp1:
                outcome    = 'WIN_TP1'
                exit_price = tp1
                pnl_pips   = (tp1 - entry) / pip_size
                break

        elif direction == 'SELL':
            if candle['high'] >= sl:
                outcome    = 'LOSS'
                exit_price = sl
                pnl_pips   = (entry - sl) / pip_size
                break
            if candle['low'] <= tp2:
                outcome    = 'WIN_TP2'
                exit_price = tp2
                pnl_pips   = (entry - tp2) / pip_size
                break
            if candle['low'] <= tp1:
                outcome    = 'WIN_TP1'
                exit_price = tp1
                pnl_pips   = (entry - tp1) / pip_size
                break

    # Timeout = close at current price
    if outcome == 'TIMEOUT':
        exit_price = float(df.iloc[min(entry_idx + max_candles,
                                       len(df)-1)]['close'])
        if direction == 'BUY':
            pnl_pips = (exit_price - entry) / pip_size
        else:
            pnl_pips = (entry - exit_price) / pip_size
        outcome = 'WIN_TP1' if pnl_pips > 0 else 'LOSS'

    won = outcome in ('WIN_TP1', 'WIN_TP2')
    return {
        'outcome':      outcome,
        'won':          won,
        'exit_price':   round(exit_price, 5),
        'pnl_pips':     round(pnl_pips, 1),
        'candles_held': candles_held,
    }

def run_virtual_simulation(symbol: str,
                           candle_count: int = 2000) -> dict:
    """
    Run full virtual simulation for all strategies on one symbol.
    Replays historical data candle by candle.
    Returns performance summary per strategy.
    """
    from strategies.ema_trend import evaluate as ema_eval
    from strategies.vwap_mean_reversion import evaluate as vwap_eval

    log.info(f"[VIRTUAL] Starting simulation for {symbol}...")
    df_raw = fetch_historical_candles(symbol, mt5.TIMEFRAME_H1, candle_count)
    if df_raw is None:
        log.error(f"[VIRTUAL] No data for {symbol}")
        return {}

    df = add_indicators(df_raw)
    log.info(f"[VIRTUAL] {len(df)} candles ready for {symbol}")

    results = {
        'EMA_TREND_MTF':     {'trades': [], 'wins': 0, 'losses': 0},
        'VWAP_MEAN_REVERSION':{'trades': [], 'wins': 0, 'losses': 0},
    }

    pip_size = 0.01 if df['close'].iloc[0] > 50 else 0.0001
    warmup   = 100  # Skip first 100 candles — indicators need warmup

    for i in range(warmup, len(df) - 50):
        df_window = df.iloc[:i+1].copy()
        df_m15    = df_window  # Use H1 as M15 proxy in simulation
        df_h1     = df_window
        df_h4     = df_window

        # Run EMA Trend strategy
        sig = ema_eval(symbol, df_m15, df_h1, df_h4)
        if sig:
            result = simulate_trade(df, sig, i)
            results['EMA_TREND_MTF']['trades'].append({
                'time':      str(df_window.iloc[-1]['time']),
                'direction': sig['direction'],
                'score':     sig['score'],
                **result,
            })
            if result['won']:
                results['EMA_TREND_MTF']['wins'] += 1
            else:
                results['EMA_TREND_MTF']['losses'] += 1

        # Run VWAP Mean Reversion
        mock_market = _build_mock_market_report(df_window, pip_size)
        sig2 = vwap_eval(symbol, df_m15, df_h1,
                         market_report=mock_market)
        if sig2:
            result2 = simulate_trade(df, sig2, i)
            results['VWAP_MEAN_REVERSION']['trades'].append({
                'time':      str(df_window.iloc[-1]['time']),
                'direction': sig2['direction'],
                'score':     sig2['score'],
                **result2,
            })
            if result2['won']:
                results['VWAP_MEAN_REVERSION']['wins'] += 1
            else:
                results['VWAP_MEAN_REVERSION']['losses'] += 1

    # Build summary
    summary = {}
    for strat, data in results.items():
        total = data['wins'] + data['losses']
        wr    = round(data['wins'] / total * 100, 1) if total > 0 else 0
        summary[strat] = {
            'total_trades': total,
            'wins':         data['wins'],
            'losses':       data['losses'],
            'win_rate':     wr,
            'passed':       total >= VIRTUAL_MIN_TRADES and
                           wr >= VIRTUAL_MIN_WINRATE,
        }
        log.info(f"[VIRTUAL] {strat}: {total} trades | WR={wr}%"
                 f" | {'PASS ✅' if summary[strat]['passed'] else 'FAIL ❌'}")

    return summary

def _build_mock_market_report(df: pd.DataFrame,
                               pip_size: float) -> dict:
    """Build a minimal market report from historical candle data."""
    last  = df.iloc[-1]
    close = float(last['close'])
    vwap  = float(df['close'].mean())
    poc   = float(df.groupby(
        df['close'].round(4))['tick_volume'].sum().idxmax())
    pip_from_vwap = round((close - vwap) / pip_size, 1)

    if close > vwap:
        position = 'ABOVE_VWAP'
    else:
        position = 'BELOW_VWAP'

    return {
        'vwap': {
            'vwap':         round(vwap, 5),
            'position':     position,
            'pip_from_vwap':pip_from_vwap,
        },
        'profile': {
            'poc':            round(poc, 5),
            'vah':            round(df['high'].quantile(0.75), 5),
            'val':            round(df['low'].quantile(0.25), 5),
            'price_position': 'ABOVE_VAH' if close > df['high'].quantile(0.75)
                              else 'BELOW_VAL' if close < df['low'].quantile(0.25)
                              else 'INSIDE_VA',
        },
        'rolling_delta': {'bias': 'NEUTRAL', 'strength': 'WEAK'},
        'delta':         {'bias': 'NEUTRAL', 'strength': 'WEAK'},
    }


# =============================================================
# STANDALONE TEST
# =============================================================
if __name__ == "__main__":
    import MetaTrader5 as mt5
    from dotenv import load_dotenv
    import os
    load_dotenv()
    mt5.initialize()
    mt5.login(int(os.getenv('MT5_LOGIN')),
              password=os.getenv('MT5_PASSWORD'),
              server=os.getenv('MT5_SERVER'))
    print("Running Virtual Simulation on EURUSD...\n")
    summary = run_virtual_simulation("EURUSD", candle_count=1000)
    print("\n=== VIRTUAL SIMULATION RESULTS ===")
    for strat, data in summary.items():
        status = "✅ PASS" if data['passed'] else "❌ FAIL"
        print(f"  {strat}")
        print(f"    Trades  : {data['total_trades']}")
        print(f"    Win Rate: {data['win_rate']}%")
        print(f"    Result  : {status}")
    mt5.shutdown()
