import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import time

load_dotenv()

def connect_to_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    login_id = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not mt5.login(login_id, password=password, server=server):
        print(f"Failed to connect, error code = {mt5.last_error()}")
        return False
    
    return True

def get_ai_ready_data(symbol, timeframe, num_candles=100):
    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1}
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_candles)
    if rates is None: return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=14 - 1, min_periods=14).mean()
    loss = -delta.clip(upper=0).ewm(com=14 - 1, min_periods=14).mean()
    df['rsi_14'] = 100 - (100 / (1 + (gain / loss)))
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.ewm(com=14 - 1, min_periods=14).mean()

    df.dropna(inplace=True)
    return df

def send_trade(symbol, order_type, tp_pips, sl_pips):
    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    
    if tick is None or symbol_info is None: return False

    if order_type == "BUY":
        price = tick.ask
        type_mt5 = mt5.ORDER_TYPE_BUY
        sl_price = price - (sl_pips * symbol_info.point * 10) 
        tp_price = price + (tp_pips * symbol_info.point * 10)
    elif order_type == "SELL":
        price = tick.bid
        type_mt5 = mt5.ORDER_TYPE_SELL
        sl_price = price + (sl_pips * symbol_info.point * 10)
        tp_price = price - (tp_pips * symbol_info.point * 10)

    lot_size = 0.01 

    filling_type = mt5.ORDER_FILLING_FOK
    if symbol_info.filling_mode & 2:
        filling_type = mt5.ORDER_FILLING_IOC

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": type_mt5,
        "price": price,
        "sl": round(sl_price, symbol_info.digits),
        "tp": round(tp_price, symbol_info.digits),
        "deviation": 20,  
        "magic": 100001,  
        "comment": "ScalpV1",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type, 
    }

    result = mt5.order_send(request)
    
    if result is None:
        print(f"❌ TRADE FAILED for {symbol}!")
        return False
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ TRADE FAILED for {symbol}! Error: {result.comment}")
        return False
    else:
        print(f"\n{'='*50}")
        print(f"✅ TRADE EXECUTED! Ticket: {result.order}")
        print(f"   {order_type} {symbol} | Lot: {lot_size} | TP: +{tp_pips} pips | SL: -{sl_pips} pips")
        print(f"{'='*50}\n")
        return True

def check_for_scalp_signal(df, symbol, timeframe):
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return None
    
    spread_pips = (tick.ask - tick.bid) * 10000
    
    # -------------------------------------------------------------
    # REAL SCALPING LOGIC (No more forced trades!)
    # -------------------------------------------------------------
    
    # SELL LOGIC
    sell_trend = previous['ema_9'] < previous['ema_21']  # Fast EMA below Slow EMA
    sell_momentum = previous['rsi_14'] < 45               # RSI showing downward momentum
    spread_ok = spread_pips < 1.5                          # Reject if spread is too wide
    
    if sell_trend and sell_momentum and spread_ok:
        atr_pips = latest['atr_14'] * 10000
        tp_pips = round(atr_pips * 1.5, 1)
        sl_pips = round(atr_pips * 2.0, 1)
        
        if tp_pips > 2.0: # Minimum profit target check
            return {"direction": "SELL", "tp": tp_pips, "sl": sl_pips}

    # BUY LOGIC
    buy_trend = previous['ema_9'] > previous['ema_21']
    buy_momentum = previous['rsi_14'] > 55
    spread_ok = spread_pips < 1.5
    
    if buy_trend and buy_momentum and spread_ok:
        atr_pips = latest['atr_14'] * 10000
        tp_pips = round(atr_pips * 1.5, 1)
        sl_pips = round(atr_pips * 2.0, 1)
        
        if tp_pips > 2.0:
            return {"direction": "BUY", "tp": tp_pips, "sl": sl_pips}
            
    return None

if __name__ == "__main__":
    if connect_to_mt5():
        print("CONNECTED! 🚀")
        
        terminal_info = mt5.terminal_info()
        if not terminal_info.trade_allowed:
            print("🛑 STOP: AlgoTrading is DISABLED in MT5!")
            mt5.shutdown()
            exit()
        
        # Multi-Market Watchlist
        watchlist = {
            "EURUSD": ["M1", "M5"],
            "GBPUSD": ["M1", "M5"],
            "AUDUSD": ["M1", "M5"],
        }
        
        print("Scalping Bot V1 is live! Watching multiple markets...\n")
        
        try:
            while True:
                total_scans = 0
                
                for symbol, timeframes in watchlist.items():
                    # CRITICAL: Check if we already have a trade open for this pair
                    positions = mt5.positions_get(symbol=symbol)
                    if positions is not None and len(positions) > 0:
                        continue # Skip this pair entirely if we are already in a trade
                    
                    for tf in timeframes:
                        df = get_ai_ready_data(symbol, tf, 100)
                        
                        if df is not None:
                            signal = check_for_scalp_signal(df, symbol, tf)
                            total_scans += 1
                            
                            if signal:
                                print(f"🎯 Setup found on {symbol} {tf}! Executing...")
                                send_trade(symbol, signal["direction"], signal["tp"], signal["sl"])
                
                # Print status update
                current_time = time.strftime("%H:%M:%S")
                print(f"[{current_time}] Scanned {total_scans} combos. Waiting 15 seconds...  ", end="\r")
                
                time.sleep(15)
                
        except KeyboardInterrupt:
            print("\n\nBot stopped by user.")
            
    mt5.shutdown()