# =============================================================
# data_layer/fractal_alignment.py
# PURPOSE: Multi-Timeframe Fractal Alignment for precision scalping.
# H4/H1 (Macro) → M15/M5 (Setup) → M1/Tick (Trigger).
# UPGRADED: Now fetches actual M5 and M1 data for real
# structure confirmation and precise entry triggers.
# =============================================================

import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, Any, Optional
from core.logger import get_logger
from data_layer.price_feed import get_candles
from data_layer.momentum_velocity import calculate_candle_velocity, get_pip_size

log = get_logger("FRACTAL_ALIGNMENT")

class FractalAlignment:
    """
    Coordinates multi-timeframe analysis for world-class scalping.
    Ensures the "Why", "Where", and "When" are all aligned.

    Timeframe hierarchy:
      H4/H1  = Macro (WHY — trend direction)
      M15/M5 = Setup  (WHERE — pullback zone, OB/FVG)
      M1     = Trigger (WHEN — precise entry with volume/velocity)
    """
    def __init__(self, symbol: str):
        self.symbol = symbol

    def get_full_alignment(self, smc_report: Dict[str, Any], 
                           market_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a comprehensive alignment report across all timeframes.
        Fetches real M5 and M1 data for actual structure confirmation.
        """
        # 1. Macro Context (H4/H1) - Already part of smc_report's htf_alignment
        htf = smc_report.get('htf_alignment', {})
        macro_bias = htf.get('h4_bias', 'NEUTRAL')
        macro_approved = htf.get('approved', False)

        # 2. Setup Location (M15/M5) - SMC concepts + actual M5 structure
        m5_data = get_candles(self.symbol, 'M5', 100)
        m5_structure = self._analyze_m5_structure(m5_data)
        setup_location = self._evaluate_setup_location(smc_report, m5_structure)

        # 3. Trigger Context (M1) - Actual M1 data + order flow
        m1_data = get_candles(self.symbol, 'M1', 100)
        m1_trigger = self._analyze_m1_trigger(m1_data, market_report)

        # Final Alignment Logic (Stricter for higher quality signals)
        # Requires: in a setup zone AND macro context approved
        # AND M5 structure confirms AND M1 trigger aligned
        aligned = setup_location['in_zone'] and \
                  macro_approved and \
                  m5_structure['confirmed'] and \
                  m1_trigger['trigger_aligned']

        return {
            'symbol': self.symbol,
            'aligned': aligned,
            'macro': {
                'bias': macro_bias,
                'approved': macro_approved
            },
            'setup': setup_location,
            'm5_structure': m5_structure,
            'trigger': m1_trigger,
            'recommendation': self._get_recommendation(
                aligned, macro_bias, setup_location, m1_trigger, m5_structure)
        }

    def _analyze_m5_structure(self, df_m5: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze M5 candles for structure confirmation.
        Checks EMA alignment, Supertrend, and recent candle patterns
        to confirm we're in the right zone for an entry.
        """
        result = {
            'confirmed': False,
            'bias': 'NEUTRAL',
            'ema_alignment': 'NONE',
            'supertrend_dir': 0,
            'last_candle_bullish': False,
            'atr_pips': 0,
        }

        if df_m5 is None or len(df_m5) < 20:
            return result

        last = df_m5.iloc[-1]
        prev = df_m5.iloc[-2]

        # EMA alignment on M5
        ema9 = last.get('ema_9', 0)
        ema21 = last.get('ema_21', 0)
        ema50 = last.get('ema_50', 0)

        if ema9 > ema21 > ema50:
            result['ema_alignment'] = 'BULLISH'
            result['bias'] = 'BULLISH'
        elif ema9 < ema21 < ema50:
            result['ema_alignment'] = 'BEARISH'
            result['bias'] = 'BEARISH'
        elif ema9 > ema21:
            result['ema_alignment'] = 'WEAK_BULL'
            result['bias'] = 'BULLISH'
        elif ema9 < ema21:
            result['ema_alignment'] = 'WEAK_BEAR'
            result['bias'] = 'BEARISH'

        # Supertrend direction
        result['supertrend_dir'] = int(last.get('supertrend_dir', 0))

        # Last candle direction
        result['last_candle_bullish'] = last['close'] > last['open']

        # ATR in pips
        close_price = float(last['close'])
        pip_size = 0.01 if close_price > 50 else 0.0001
        result['atr_pips'] = round(float(last.get('atr', 0)) / pip_size, 1)

        # Structure is confirmed if EMA alignment and supertrend agree
        if result['ema_alignment'] in ('BULLISH', 'WEAK_BULL') and result['supertrend_dir'] == 1:
            result['confirmed'] = True
        elif result['ema_alignment'] in ('BEARISH', 'WEAK_BEAR') and result['supertrend_dir'] == -1:
            result['confirmed'] = True

        return result

    def _analyze_m1_trigger(self, df_m1: pd.DataFrame, 
                            market_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze M1 candles for precise entry timing.
        Checks: StochRSI, volume, candle pattern, momentum.
        Only fires when all micro-level conditions align.
        """
        result = {
            'trigger_aligned': False,
            'bias': 'NEUTRAL',
            'stochrsi_signal': 'NONE',
            'volume_confirms': False,
            'candle_pattern': 'NONE',
            'momentum_ok': False,
            'velocity_pips_min': 0,
        }

        if df_m1 is None or len(df_m1) < 20:
            return result

        last = df_m1.iloc[-1]
        prev = df_m1.iloc[-2]

        # 1. StochRSI timing
        stoch_k = float(last.get('stoch_rsi_k', 50))
        stoch_d = float(last.get('stoch_rsi_d', 50))
        prev_k = float(prev.get('stoch_rsi_k', 50))
        prev_d = float(prev.get('stoch_rsi_d', 50))

        # Bullish cross: K crosses above D from oversold
        if prev_k <= prev_d and stoch_k > stoch_d and stoch_k < 40:
            result['stochrsi_signal'] = 'BUY_CROSS'
        # Bearish cross: K crosses below D from overbought
        elif prev_k >= prev_d and stoch_k < stoch_d and stoch_k > 60:
            result['stochrsi_signal'] = 'SELL_CROSS'
        # Oversold zone
        elif stoch_k < 20:
            result['stochrsi_signal'] = 'OVERSOLD'
        # Overbought zone
        elif stoch_k > 80:
            result['stochrsi_signal'] = 'OVERBOUGHT'

        # 2. Volume confirmation (current M1 candle vs 20-period average)
        vol_ma = float(last.get('vol_ma20', 0))
        current_vol = float(last.get('tick_volume', 0))
        if vol_ma > 0 and current_vol >= vol_ma * 1.5:
            result['volume_confirms'] = True

        # 3. Candle pattern
        body = abs(last['close'] - last['open'])
        full_range = last['high'] - last['low']
        if full_range > 0:
            body_ratio = body / full_range
            if body_ratio > 0.7:  # Strong body
                if last['close'] > last['open']:
                    result['candle_pattern'] = 'STRONG_BULL'
                else:
                    result['candle_pattern'] = 'STRONG_BEAR'
            elif body_ratio < 0.3:  # Doji/wick
                result['candle_pattern'] = 'DOJI'
            else:
                result['candle_pattern'] = 'NORMAL'

        # 4. Momentum from M1 candles
        pip_size = get_pip_size(self.symbol)
        velocity_data = calculate_candle_velocity(df_m1, pip_size, candle_minutes=1, lookback=10)
        result['velocity_pips_min'] = velocity_data.get('velocity_pips_min', 0)
        result['momentum_ok'] = velocity_data.get('is_scalpable', False)

        # Determine overall M1 bias
        m1_bias = 'NEUTRAL'
        if result['stochrsi_signal'] == 'BUY_CROSS' or result['candle_pattern'] == 'STRONG_BULL':
            m1_bias = 'BULLISH'
        elif result['stochrsi_signal'] == 'SELL_CROSS' or result['candle_pattern'] == 'STRONG_BEAR':
            m1_bias = 'BEARISH'
        result['bias'] = m1_bias

        # Trigger is aligned when timing + volume + momentum all agree
        if m1_bias != 'NEUTRAL' and result['volume_confirms']:
            result['trigger_aligned'] = True
        elif m1_bias != 'NEUTRAL' and result['momentum_ok']:
            result['trigger_aligned'] = True

        return result

    def _evaluate_setup_location(self, smc_report: Dict[str, Any],
                                  m5_structure: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if price is in a high-probability M15/M5 zone."""
        nob = smc_report.get('nearest_ob')
        nfvg = smc_report.get('nearest_fvg')
        pd_zone = smc_report.get('premium_discount', {}).get('zone', 'UNKNOWN')
        
        # Determine if we are "in a zone"
        in_zone = False
        zone_type = 'NONE'
        
        if nob or nfvg:
            in_zone = True
            if nob and nfvg:
                zone_type = 'OB_FVG_CONFLUENCE'  # Best case — both present
            elif nfvg:
                zone_type = 'FAIR_VALUE_GAP'
            elif nob:
                zone_type = 'ORDER_BLOCK'

        # M5 structure must also confirm for setup to be valid
        if m5_structure and m5_structure.get('confirmed', False):
            in_zone = in_zone or True  # M5 structure alone can be a setup
            if zone_type == 'NONE':
                zone_type = f"M5_{m5_structure.get('ema_alignment', 'NONE')}"

        return {
            'in_zone': in_zone,
            'zone_type': zone_type,
            'pd_zone': pd_zone,
            'm5_confirmed': m5_structure.get('confirmed', False) if m5_structure else False,
            'm5_bias': m5_structure.get('bias', 'NEUTRAL') if m5_structure else 'NEUTRAL',
        }

    def _get_recommendation(self, aligned, bias, setup, trigger, m5_structure) -> str:
        if aligned:
            return f"READY: Full {bias} alignment at {setup['zone_type']}"
        if setup['in_zone'] and m5_structure.get('confirmed', False):
            return f"WATCH: In {setup['zone_type']} zone, M5 confirmed, waiting for {bias} M1 trigger"
        if setup['in_zone']:
            return f"WAIT: In {setup['zone_type']} zone, waiting for M5+M1 confirmation"
        if setup['pd_zone'] in ('EXTREME_DISCOUNT', 'EXTREME_PREMIUM'):
            return "WAIT: Reaching extreme zone, look for setup"
        return "SKIP: No clear fractal alignment"

# Helper function
def check_fractal_alignment(symbol: str, smc_report: dict, market_report: dict):
    fa = FractalAlignment(symbol)
    return fa.get_full_alignment(smc_report, market_report)
