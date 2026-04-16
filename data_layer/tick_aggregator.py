# =============================================================
# data_layer/tick_aggregator.py
# PURPOSE: Continuous tick-streaming and volume-bar aggregator.
# Normalizes market volatility for world-class scalping.
# =============================================================

import time
import threading
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from typing import Dict, List, Optional
from core.logger import get_logger

log = get_logger("TICK_AGGREGATOR")

class TickAggregator:
    """
    Background worker that streams ticks and builds Volume/Tick bars.
    Replaces static time-based windows with volatility-normalized data.
    """
    def __init__(self, symbols: List[str], volume_threshold: int = 1000):
        self.symbols = symbols
        self.volume_threshold = volume_threshold
        self.tick_data: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in symbols}
        self.volume_bars: Dict[str, List[Dict]] = {s: [] for s in symbols}
        self.current_bar_vol: Dict[str, float] = {s: 0.0 for s in symbols}
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.running: return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info(f"Tick Aggregator started for: {', '.join(self.symbols)}")

    def stop(self):
        self.running = False
        if self._thread: self._thread.join()

    def _run(self):
        """Continuous background loop for tick streaming."""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Fetch only the latest ticks since last poll
                    ticks = mt5.copy_ticks_from(symbol, int(time.time() - 1), 100, mt5.COPY_TICKS_ALL)
                    if ticks is None or len(ticks) == 0: continue
                    
                    df = pd.DataFrame(ticks)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # 1. Update rolling tick buffer (for delta)
                    self.tick_data[symbol] = pd.concat([self.tick_data[symbol], df]).tail(2000)
                    
                    # 2. Build Volume Bars
                    for _, tick in df.iterrows():
                        self._process_tick_for_bar(symbol, tick)
                        
                except Exception as e:
                    log.error(f"Error in aggregator for {symbol}: {e}")
            
            time.sleep(0.5) # Fast poll for scalping

    def _process_tick_for_bar(self, symbol: str, tick: pd.Series):
        """Logic to aggregate ticks into a volume-based bar."""
        vol = tick.get('volume', 1.0) # Fallback to 1 if no volume
        self.current_bar_vol[symbol] += vol
        
        # If threshold reached, close bar and start new one
        if self.current_bar_vol[symbol] >= self.volume_threshold:
            # Simplistic bar creation - in production, this would track OHLC
            new_bar = {
                'time': tick['time'],
                'close': tick['bid'],
                'volume': self.current_bar_vol[symbol]
            }
            self.volume_bars[symbol].append(new_bar)
            self.volume_bars[symbol] = self.volume_bars[symbol][-100:] # Keep last 100 bars
            self.current_bar_vol[symbol] = 0.0

    def get_latest_delta(self, symbol: str, window: int = 100) -> float:
        """Calculate delta from the continuous tick buffer."""
        df = self.tick_data.get(symbol)
        if df is None or df.empty: return 0.0
        
        # Identify aggressive side (MT5 specific tick flags or simple bid/ask logic)
        # For simplicity, assuming 'last' price vs bid/ask
        recent = df.tail(window)
        buys = len(recent[recent['bid'] >= recent['ask']]) # Aggressive buy approximation
        sells = len(recent) - buys
        return buys - sells

# Global singleton
aggregator = None

def init_aggregator(symbols: List[str]):
    global aggregator
    aggregator = TickAggregator(symbols)
    aggregator.start()
