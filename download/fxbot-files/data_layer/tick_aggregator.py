# =============================================================
# data_layer/tick_aggregator.py
# PURPOSE: Continuous tick-streaming and volume-bar aggregator.
# Normalizes market volatility for world-class scalping.
#
# v4.1 FIX: Delta calculation was completely broken — used
#   `bid >= ask` which is ALWAYS False (ask > bid in all markets).
#   Now uses proper tick classification from tick_fetcher.
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
                    ticks = mt5.copy_ticks_from(symbol, int(time.time() - 1), 100, mt5.COPY_TICKS_ALL)
                    if ticks is None or len(ticks) == 0: continue
                    
                    df = pd.DataFrame(ticks)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Classify each tick for direction (FIXED v4.1)
                    from data_layer.tick_fetcher import classify_tick, get_tick_threshold
                    df['prev_bid'] = df['bid'].shift(1)
                    df['prev_ask'] = df['ask'].shift(1)
                    df['side'] = df.apply(
                        lambda row: classify_tick(
                            row['flags'], row['bid'], row['ask'], row.get('last', 0),
                            row['prev_bid'], row['prev_ask'], symbol
                        ),
                        axis=1
                    )
                    
                    # 1. Update rolling tick buffer (with proper classification)
                    self.tick_data[symbol] = pd.concat(
                        [self.tick_data[symbol], df]
                    ).tail(2000)
                    
                    # 2. Build Volume Bars
                    for _, tick in df.iterrows():
                        self._process_tick_for_bar(symbol, tick)
                        
                except Exception as e:
                    log.error(f"Error in aggregator for {symbol}: {e}")
            
            time.sleep(0.5)

    def _process_tick_for_bar(self, symbol: str, tick: pd.Series):
        """Logic to aggregate ticks into a volume-based bar."""
        vol = tick.get('volume', 1.0)
        self.current_bar_vol[symbol] += vol
        
        if self.current_bar_vol[symbol] >= self.volume_threshold:
            new_bar = {
                'time': tick['time'],
                'close': tick['bid'],
                'volume': self.current_bar_vol[symbol]
            }
            self.volume_bars[symbol].append(new_bar)
            self.volume_bars[symbol] = self.volume_bars[symbol][-100:]
            self.current_bar_vol[symbol] = 0.0

    def get_latest_delta(self, symbol: str, window: int = 100) -> float:
        """
        Calculate delta from the continuous tick buffer.
        v4.1 FIX: Uses 'side' column from tick_fetcher classification
        instead of the broken `bid >= ask` comparison.
        """
        df = self.tick_data.get(symbol)
        if df is None or df.empty:
            return 0.0
        
        recent = df.tail(window)
        
        # Check if 'side' column exists (it should after v4.1 fix)
        if 'side' in recent.columns:
            buys = len(recent[recent['side'] == 'BUY'])
            sells = len(recent[recent['side'] == 'SELL'])
        else:
            # Fallback: use consecutive bid comparison
            buys = 0
            sells = 0
            for i in range(1, len(recent)):
                if recent.iloc[i]['bid'] > recent.iloc[i-1]['bid']:
                    buys += 1
                elif recent.iloc[i]['bid'] < recent.iloc[i-1]['bid']:
                    sells += 1
        
        return buys - sells

# Global singleton
aggregator = None

def init_aggregator(symbols: List[str]):
    global aggregator
    aggregator = TickAggregator(symbols)
    aggregator.start()
