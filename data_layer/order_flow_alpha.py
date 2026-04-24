# =============================================================
# data_layer/order_flow_alpha.py
# PURPOSE: Advanced order flow signals used by institutional traders
# These are the "hidden secrets" that retail traders don't see:
#   1. Cumulative Delta Divergence (CDD)
#   2. Absorption Detection (Iceberg Orders)
#   3. Stop Hunt Identification
#   4. Volume Node Rejection
#   5. Order Flow Imbalance Velocity
#   6. Smart Money Footprint Score
# =============================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.logger import get_logger

log = get_logger("ORDER_FLOW_ALPHA")


class OrderFlowAlpha:
    """
    Advanced order flow analytics for institutional-grade entries.
    Uses tick-level data to detect what smart money is doing.
    """
    
    def __init__(self):
        self.cumulative_delta_history: Dict[str, List[Dict]] = {}
        self.absorption_levels: Dict[str, List[Dict]] = {}
        
    # ════════════════════════════════════════════════════════════
    # SECRET #1: CUMULATIVE DELTA DIVERGENCE
    # ════════════════════════════════════════════════════════════
    # When price makes a new high but cumulative delta does NOT,
    # it means the move is exhausted and reversal is imminent.
    # This is ONE OF THE MOST POWERFUL reversal signals.
    # ════════════════════════════════════════════════════════════
    
    def calculate_cumulative_delta_divergence(
        self, 
        df_ticks: pd.DataFrame, 
        symbol: str,
        lookback_bars: int = 50
    ) -> Dict:
        """
        Detect bullish/bearish divergences between price and cumulative delta.
        
        Bullish Divergence: Price makes lower low, delta makes higher low = BUY
        Bearish Divergence: Price makes higher high, delta makes lower high = SELL
        
        Returns:
            {
                'divergence': 'BULLISH' | 'BEARISH' | 'NONE',
                'strength': 0-100,
                'price_extreme': float,
                'delta_extreme': float,
                'confidence': 'HIGH' | 'MEDIUM' | 'LOW',
                'reversal_imminent': bool
            }
        """
        if df_ticks is None or len(df_ticks) < lookback_bars:
            return {'divergence': 'NONE', 'strength': 0, 'confidence': 'LOW'}
        
        recent = df_ticks.tail(lookback_bars).copy()
        
        # Calculate cumulative delta
        recent['delta_tick'] = recent['side'].map({'BUY': 1, 'SELL': -1, 'NEUTRAL': 0})
        recent['cum_delta'] = recent['delta_tick'].cumsum()
        
        # Find price extremes (highs/lows)
        recent['price_mid'] = (recent['bid'] + recent['ask']) / 2
        
        # Split into two halves for comparison
        mid_point = len(recent) // 2
        first_half = recent.iloc[:mid_point]
        second_half = recent.iloc[mid_point:]
        
        # First half extremes
        first_price_high = first_half['price_mid'].max()
        first_price_low = first_half['price_mid'].min()
        first_delta_high = first_half['cum_delta'].max()
        first_delta_low = first_half['cum_delta'].min()
        
        # Second half extremes
        second_price_high = second_half['price_mid'].max()
        second_price_low = second_half['price_mid'].min()
        second_delta_high = second_half['cum_delta'].max()
        second_delta_low = second_half['cum_delta'].min()
        
        divergence = 'NONE'
        strength = 0
        confidence = 'LOW'
        
        # BEARISH DIVERGENCE: Price higher high, delta lower high
        if second_price_high > first_price_high and second_delta_high < first_delta_high:
            divergence = 'BEARISH'
            # Strength based on magnitude of divergence
            price_diff_pct = (second_price_high - first_price_high) / first_price_high * 100
            delta_diff_pct = (first_delta_high - second_delta_high) / abs(first_delta_high) * 100 if first_delta_high != 0 else 0
            strength = min(100, (price_diff_pct + delta_diff_pct) * 5)
            
            if strength > 70:
                confidence = 'HIGH'
            elif strength > 40:
                confidence = 'MEDIUM'
                
            log.info(f"[{symbol}] BEARISH DIVERGENCE detected! Strength: {strength:.1f}%")
            
        # BULLISH DIVERGENCE: Price lower low, delta higher low
        elif second_price_low < first_price_low and second_delta_low > first_delta_low:
            divergence = 'BULLISH'
            price_diff_pct = (first_price_low - second_price_low) / first_price_low * 100
            delta_diff_pct = (second_delta_low - first_delta_low) / abs(first_delta_low) * 100 if first_delta_low != 0 else 0
            strength = min(100, (price_diff_pct + delta_diff_pct) * 5)
            
            if strength > 70:
                confidence = 'HIGH'
            elif strength > 40:
                confidence = 'MEDIUM'
                
            log.info(f"[{symbol}] BULLISH DIVERGENCE detected! Strength: {strength:.1f}%")
        
        return {
            'divergence': divergence,
            'strength': round(strength, 1),
            'price_extreme': second_price_high if divergence == 'BEARISH' else second_price_low,
            'delta_extreme': second_delta_high if divergence == 'BEARISH' else second_delta_low,
            'confidence': confidence,
            'reversal_imminent': strength > 60 and confidence in ('HIGH', 'MEDIUM')
        }
    
    # ════════════════════════════════════════════════════════════
    # SECRET #2: ABSORPTION DETECTION (ICEBERG ORDERS)
    # ════════════════════════════════════════════════════════════
    # When large aggressive orders hit a level but price doesn't
    # move, it means a passive iceberg order is absorbing them.
    # This shows where institutions are defending levels.
    # ════════════════════════════════════════════════════════════
    
    def detect_absorption(
        self,
        df_ticks: pd.DataFrame,
        symbol: str,
        price_window: float = 0.0005,  # 0.5 pips for forex
        min_aggressive_volume: int = 50
    ) -> Dict:
        """
        Detect absorption zones where aggressive orders are being absorbed.
        
        Logic:
        - Count aggressive buy/sell ticks at each price level
        - If many aggressive orders but price doesn't move = absorption
        - These levels become strong support/resistance
        
        Returns:
            {
                'absorption_detected': bool,
                'levels': [
                    {
                        'price': float,
                        'type': 'SUPPORT' | 'RESISTANCE',
                        'absorbed_buys': int,
                        'absorbed_sells': int,
                        'total_volume': int,
                        'strength': 0-100
                    }
                ]
            }
        """
        if df_ticks is None or len(df_ticks) < 100:
            return {'absorption_detected': False, 'levels': []}
        
        recent = df_ticks.tail(200).copy()
        recent['price_mid'] = (recent['bid'] + recent['ask']) / 2
        recent['price_level'] = (recent['price_mid'] / price_window).round() * price_window
        
        absorption_levels = []
        
        # Group by price level
        grouped = recent.groupby('price_level')
        
        for price_level, group in grouped:
            if len(group) < min_aggressive_volume:
                continue
            
            # Count aggressive buys/sells at this level
            buys = len(group[group['side'] == 'BUY'])
            sells = len(group[group['side'] == 'SELL'])
            total = buys + sells
            
            # Calculate price movement while these orders occurred
            price_start = group.iloc[0]['price_mid']
            price_end = group.iloc[-1]['price_mid']
            price_change = abs(price_end - price_start)
            
            # ABSORPTION: High volume but minimal price movement
            if total >= min_aggressive_volume and price_change <= price_window * 2:
                # Determine type
                if buys > sells * 1.5:
                    level_type = 'RESISTANCE'  # Buyers absorbed = can't push up
                elif sells > buys * 1.5:
                    level_type = 'SUPPORT'  # Sellers absorbed = can't push down
                else:
                    continue  # Mixed, unclear
                
                # Strength based on volume and lack of movement
                volume_score = min(100, total / min_aggressive_volume * 50)
                stagnation_score = max(0, 100 - (price_change / price_window * 50))
                strength = (volume_score + stagnation_score) / 2
                
                absorption_levels.append({
                    'price': price_level,
                    'type': level_type,
                    'absorbed_buys': buys,
                    'absorbed_sells': sells,
                    'total_volume': total,
                    'strength': round(strength, 1),
                    'price_range': f"{price_start:.5f} - {price_end:.5f}"
                })
        
        # Sort by strength
        absorption_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        result = {
            'absorption_detected': len(absorption_levels) > 0,
            'levels': absorption_levels[:5]  # Top 5 levels
        }
        
        if result['absorption_detected']:
            top = absorption_levels[0]
            log.info(f"[{symbol}] ABSORPTION at {top['price']} ({top['type']}) "
                    f"Strength: {top['strength']}% | Vol: {top['total_volume']}")
        
        return result
    
    # ════════════════════════════════════════════════════════════
    # SECRET #3: STOP HUNT DETECTION
    # ════════════════════════════════════════════════════════════
    # Identify when price spikes to take out obvious stops then
    # immediately reverses. This is institutional manipulation.
    # Entry: Fade the stop hunt (trade the reversal)
    # ════════════════════════════════════════════════════════════
    
    def detect_stop_hunt(
        self,
        df_candles: pd.DataFrame,
        symbol: str,
        lookback: int = 20
    ) -> Dict:
        """
        Detect stop hunt patterns (liquidity grabs).
        
        Pattern:
        1. Price makes a sharp spike (3+x average candle range)
        2. Creates a new high/low
        3. Immediately reverses within 1-2 candles
        4. Close is back inside the previous range
        
        Returns:
            {
                'stop_hunt_detected': bool,
                'direction': 'BULLISH' | 'BEARISH' | 'NONE',
                # (bullish = hunted stops below, now reversing up)
                'hunt_pips': float,
                'reversal_strength': 0-100,
                'entry_zone': float,
                'invalidation': float
            }
        """
        if df_candles is None or len(df_candles) < lookback + 5:
            return {'stop_hunt_detected': False, 'direction': 'NONE'}
        
        recent = df_candles.tail(lookback + 5).copy()
        
        # Calculate average candle range
        recent['range'] = recent['high'] - recent['low']
        avg_range = recent['range'].iloc[:-5].mean()
        
        stop_hunts = []
        
        # Check each candle in the last 5 candles for stop hunt pattern
        for i in range(len(recent) - 5, len(recent) - 1):
            candle = recent.iloc[i]
            prev_candle = recent.iloc[i - 1] if i > 0 else candle
            next_candle = recent.iloc[i + 1]
            
            candle_range = candle['range']
            
            # Condition 1: Spike candle (3x+ average range)
            if candle_range < avg_range * 3:
                continue
            
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            body = abs(candle['close'] - candle['open'])
            
            # BULLISH STOP HUNT: Long lower wick, close in upper 30%
            if lower_wick > candle_range * 0.6 and body < candle_range * 0.3:
                if candle['close'] > candle['open']:
                    # Check if next candle confirms reversal
                    if next_candle['close'] > candle['high']:
                        hunt_pips = lower_wick / self._get_pip_size(symbol, candle['low'])
                        stop_hunts.append({
                            'type': 'BULLISH',
                            'candle_idx': i,
                            'hunt_low': candle['low'],
                            'hunt_pips': round(hunt_pips, 1),
                            'reversal_confirm': True
                        })
            
            # BEARISH STOP HUNT: Long upper wick, close in lower 30%
            elif upper_wick > candle_range * 0.6 and body < candle_range * 0.3:
                if candle['close'] < candle['open']:
                    if next_candle['close'] < candle['low']:
                        hunt_pips = upper_wick / self._get_pip_size(symbol, candle['high'])
                        stop_hunts.append({
                            'type': 'BEARISH',
                            'candle_idx': i,
                            'hunt_high': candle['high'],
                            'hunt_pips': round(hunt_pips, 1),
                            'reversal_confirm': True
                        })
        
        if not stop_hunts:
            return {'stop_hunt_detected': False, 'direction': 'NONE'}
        
        # Take the most recent confirmed stop hunt
        latest = stop_hunts[-1]
        
        result = {
            'stop_hunt_detected': True,
            'direction': latest['type'],
            'hunt_pips': latest['hunt_pips'],
            'reversal_strength': min(100, latest['hunt_pips'] * 10),
            'entry_zone': latest.get('hunt_low' if latest['type'] == 'BULLISH' else 'hunt_high'),
            'invalidation': latest.get('hunt_high' if latest['type'] == 'BULLISH' else 'hunt_low')
        }
        
        log.info(f"[{symbol}] STOP HUNT {result['direction']} detected! "
                f"Hunt: {result['hunt_pips']} pips | Strength: {result['reversal_strength']}%")
        
        return result
    
    # ════════════════════════════════════════════════════════════
    # SECRET #4: VOLUME NODE REJECTION
    # ════════════════════════════════════════════════════════════
    # When price enters a high-volume node (POC/VAH/VAL) and gets
    # rejected with strong opposing order flow, it's a high-prob
    # mean reversion trade.
    # ════════════════════════════════════════════════════════════
    
    def detect_volume_node_rejection(
        self,
        df_candles: pd.DataFrame,
        volume_profile: Dict,
        symbol: str
    ) -> Dict:
        """
        Detect when price rejects from key volume profile levels.
        
        Levels: POC (Point of Control), VAH (Value Area High), VAL (Value Area Low)
        
        Returns:
            {
                'rejection_detected': bool,
                'level_type': 'POC' | 'VAH' | 'VAL',
                'level_price': float,
                'direction': 'BULLISH' | 'BEARISH',
                'rejection_strength': 0-100
            }
        """
        if df_candles is None or not volume_profile:
            return {'rejection_detected': False}
        
        recent = df_candles.tail(10).copy()
        current_price = recent.iloc[-1]['close']
        
        poc = volume_profile.get('poc')
        vah = volume_profile.get('vah')
        val = volume_profile.get('val')
        
        if not all([poc, vah, val]):
            return {'rejection_detected': False}
        
        # Check proximity to each level (within 5 pips)
        pip_size = self._get_pip_size(symbol, current_price)
        proximity_pips = 5
        
        levels_to_check = [
            ('POC', poc),
            ('VAH', vah),
            ('VAL', val)
        ]
        
        for level_name, level_price in levels_to_check:
            distance = abs(current_price - level_price) / pip_size
            
            if distance <= proximity_pips:
                # Price is at the level - check for rejection
                # Look for strong rejection candle
                last_candle = recent.iloc[-1]
                prev_candle = recent.iloc[-2]
                
                if level_name == 'VAH' and current_price <= level_price:
                    # Potential bearish rejection at VAH
                    if last_candle['close'] < last_candle['open']:
                        rejection_strength = self._calculate_rejection_strength(
                            last_candle, prev_candle, pip_size
                        )
                        if rejection_strength > 50:
                            log.info(f"[{symbol}] BEARISH rejection at {level_name} ({level_price:.5f})")
                            return {
                                'rejection_detected': True,
                                'level_type': level_name,
                                'level_price': level_price,
                                'direction': 'BEARISH',
                                'rejection_strength': rejection_strength
                            }
                
                elif level_name == 'VAL' and current_price >= level_price:
                    # Potential bullish rejection at VAL
                    if last_candle['close'] > last_candle['open']:
                        rejection_strength = self._calculate_rejection_strength(
                            last_candle, prev_candle, pip_size
                        )
                        if rejection_strength > 50:
                            log.info(f"[{symbol}] BULLISH rejection at {level_name} ({level_price:.5f})")
                            return {
                                'rejection_detected': True,
                                'level_type': level_name,
                                'level_price': level_price,
                                'direction': 'BULLISH',
                                'rejection_strength': rejection_strength
                            }
                
                elif level_name == 'POC':
                    # POC rejection can go either way
                    if distance < 2:  # Very close to POC
                        # Check candle direction
                        if last_candle['close'] > last_candle['open']:
                            return {
                                'rejection_detected': True,
                                'level_type': level_name,
                                'level_price': level_price,
                                'direction': 'BULLISH',
                                'rejection_strength': 60
                            }
                        else:
                            return {
                                'rejection_detected': True,
                                'level_type': level_name,
                                'level_price': level_price,
                                'direction': 'BEARISH',
                                'rejection_strength': 60
                            }
        
        return {'rejection_detected': False}
    
    def _calculate_rejection_strength(
        self, 
        candle: pd.Series, 
        prev_candle: pd.Series,
        pip_size: float
    ) -> float:
        """Calculate how strong the rejection candle is."""
        candle_range = candle['high'] - candle['low']
        body = abs(candle['close'] - candle['open'])
        
        # Wick ratio (longer wick = stronger rejection)
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        wick_ratio = max(upper_wick, lower_wick) / candle_range if candle_range > 0 else 0
        
        # Body ratio (smaller body = stronger rejection)
        body_ratio = 1 - (body / candle_range) if candle_range > 0 else 0
        
        # Volume confirmation (if available)
        vol_factor = 1.0
        if 'tick_volume' in candle and 'tick_volume' in prev_candle:
            if candle['tick_volume'] > prev_candle['tick_volume'] * 1.5:
                vol_factor = 1.2
        
        strength = (wick_ratio * 50 + body_ratio * 50) * vol_factor
        return min(100, round(strength, 1))
    
    # ════════════════════════════════════════════════════════════
    # SECRET #5: ORDER FLOW IMBALANCE VELOCITY
    # ════════════════════════════════════════════════════════════
    # Not just the imbalance itself, but how FAST it's changing.
    # Rapid shift in order flow = institutions entering aggressively.
    # This is an early entry signal before price moves.
    # ════════════════════════════════════════════════════════════
    
    def calculate_order_flow_velocity(
        self,
        df_ticks: pd.DataFrame,
        symbol: str,
        window: int = 20
    ) -> Dict:
        """
        Calculate the velocity (rate of change) of order flow imbalance.
        
        Returns:
            {
                'velocity': float,  # Rate of change per tick
                'acceleration': float,  # Is velocity increasing?
                'signal': 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG_SELL',
                'institutional_activity': bool
            }
        """
        if df_ticks is None or len(df_ticks) < window * 2:
            return {'velocity': 0, 'acceleration': 0, 'signal': 'NEUTRAL'}
        
        recent = df_ticks.tail(window * 2).copy()
        
        # Calculate rolling imbalance
        recent['buy_tick'] = (recent['side'] == 'BUY').astype(int)
        recent['sell_tick'] = (recent['side'] == 'SELL').astype(int)
        
        # Rolling imbalance over 'window' ticks
        recent['rolling_imb'] = (
            recent['buy_tick'].rolling(window).sum() - 
            recent['sell_tick'].rolling(window).sum()
        ) / window
        
        recent = recent.dropna()
        
        if len(recent) < 10:
            return {'velocity': 0, 'acceleration': 0, 'signal': 'NEUTRAL'}
        
        # Velocity = rate of change of imbalance
        recent['velocity'] = recent['rolling_imb'].diff()
        
        # Acceleration = rate of change of velocity
        recent['acceleration'] = recent['velocity'].diff()
        
        latest = recent.iloc[-1]
        velocity = latest['velocity']
        acceleration = latest['acceleration']
        
        # Determine signal
        if velocity > 0.3 and acceleration > 0:
            signal = 'STRONG_BUY'
            institutional = True
        elif velocity > 0.15:
            signal = 'BUY'
            institutional = velocity > 0.2
        elif velocity < -0.3 and acceleration < 0:
            signal = 'STRONG_SELL'
            institutional = True
        elif velocity < -0.15:
            signal = 'SELL'
            institutional = velocity < -0.2
        else:
            signal = 'NEUTRAL'
            institutional = False
        
        return {
            'velocity': round(velocity, 4),
            'acceleration': round(acceleration, 4),
            'signal': signal,
            'institutional_activity': institutional,
            'current_imbalance': round(latest['rolling_imb'], 4)
        }
    
    # ════════════════════════════════════════════════════════════
    # SECRET #6: SMART MONEY FOOTPRINT SCORE
    # ════════════════════════════════════════════════════════════
    # Combine ALL order flow signals into one score that shows
    # whether smart money is accumulating (buying) or distributing
    # (selling). This is the ultimate confluence indicator.
    # ════════════════════════════════════════════════════════════
    
    def calculate_smart_money_score(
        self,
        symbol: str,
        df_ticks: pd.DataFrame,
        df_candles: pd.DataFrame,
        volume_profile: Dict,
        divergence_result: Dict,
        absorption_result: Dict,
        stop_hunt_result: Dict,
        velocity_result: Dict
    ) -> Dict:
        """
        Combine all order flow alpha signals into one master score.
        
        Score components:
        - Delta divergence: ±25 points
        - Absorption levels: ±20 points
        - Stop hunt reversal: ±25 points
        - Order flow velocity: ±20 points
        - Volume profile rejection: ±10 points
        
        Returns:
            {
                'score': -100 to +100,
                'bias': 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG_SELL',
                'components': {...},
                'confidence': 0-100,
                'recommended_action': 'ENTER_LONG' | 'ENTER_SHORT' | 'WAIT'
            }
        """
        score = 0
        components = {}
        
        # Component 1: Delta Divergence (±25)
        div = divergence_result.get('divergence', 'NONE')
        div_strength = divergence_result.get('strength', 0) / 100
        if div == 'BULLISH':
            score += 25 * div_strength
        elif div == 'BEARISH':
            score -= 25 * div_strength
        components['divergence'] = {'signal': div, 'contribution': round(25 * div_strength, 1)}
        
        # Component 2: Absorption (±20)
        if absorption_result.get('absorption_detected'):
            levels = absorption_result.get('levels', [])
            if levels:
                top_level = levels[0]
                if top_level['type'] == 'SUPPORT':
                    score += 20 * (top_level['strength'] / 100)
                else:  # RESISTANCE
                    score -= 20 * (top_level['strength'] / 100)
        components['absorption'] = {'detected': absorption_result.get('absorption_detected', False)}
        
        # Component 3: Stop Hunt (±25)
        if stop_hunt_result.get('stop_hunt_detected'):
            sh_direction = stop_hunt_result.get('direction', 'NONE')
            sh_strength = stop_hunt_result.get('reversal_strength', 0) / 100
            if sh_direction == 'BULLISH':
                score += 25 * sh_strength
            elif sh_direction == 'BEARISH':
                score -= 25 * sh_strength
        components['stop_hunt'] = {'detected': stop_hunt_result.get('stop_hunt_detected', False)}
        
        # Component 4: Order Flow Velocity (±20)
        vel_signal = velocity_result.get('signal', 'NEUTRAL')
        if vel_signal == 'STRONG_BUY':
            score += 20
        elif vel_signal == 'BUY':
            score += 10
        elif vel_signal == 'STRONG_SELL':
            score -= 20
        elif vel_signal == 'SELL':
            score -= 10
        components['velocity'] = {'signal': vel_signal}
        
        # Component 5: Volume Profile Rejection (±10)
        vp_rejection = self.detect_volume_node_rejection(df_candles, volume_profile, symbol)
        if vp_rejection.get('rejection_detected'):
            vp_direction = vp_rejection.get('direction', 'NEUTRAL')
            vp_strength = vp_rejection.get('rejection_strength', 0) / 100
            if vp_direction == 'BULLISH':
                score += 10 * vp_strength
            elif vp_direction == 'BEARISH':
                score -= 10 * vp_strength
        components['vp_rejection'] = {'detected': vp_rejection.get('rejection_detected', False)}
        
        # Normalize score to -100 to +100
        score = max(-100, min(100, score))
        
        # Determine bias
        if score >= 60:
            bias = 'STRONG_BUY'
            action = 'ENTER_LONG'
        elif score >= 30:
            bias = 'BUY'
            action = 'ENTER_LONG'
        elif score <= -60:
            bias = 'STRONG_SELL'
            action = 'ENTER_SHORT'
        elif score <= -30:
            bias = 'SELL'
            action = 'ENTER_SHORT'
        else:
            bias = 'NEUTRAL'
            action = 'WAIT'
        
        # Confidence based on number of confirming signals
        confirming_signals = sum([
            1 if div in ['BULLISH', 'BEARISH'] and div_strength > 0.5 else 0,
            1 if absorption_result.get('absorption_detected') else 0,
            1 if stop_hunt_result.get('stop_hunt_detected') else 0,
            1 if vel_signal in ['STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL'] else 0,
            1 if vp_rejection.get('rejection_detected') else 0
        ])
        confidence = min(100, confirming_signals * 20)
        
        result = {
            'score': round(score, 1),
            'bias': bias,
            'components': components,
            'confidence': confidence,
            'recommended_action': action,
            'confirming_signals': confirming_signals
        }
        
        if abs(score) > 50:
            log.info(f"[{symbol}] SMART MONEY SCORE: {score:+.1f} | {bias} | Action: {action}")
        
        return result
    
    def _get_pip_size(self, symbol: str, price: float) -> float:
        """Get correct pip size for any symbol."""
        sym = str(symbol).upper()
        # Indices — trade in full points
        if any(x in sym for x in ["US30", "US500", "USTEC", "JP225", "DE30", "UK100"]):
            return 1.0
        # Gold
        if "XAU" in sym:
            return 0.1
        # Silver
        if "XAG" in sym:
            return 0.01
        # Oil
        if any(x in sym for x in ["WTI", "BRN"]):
            return 0.01
        # JPY pairs
        if "JPY" in sym:
            return 0.01
        # Standard forex
        return 0.0001


# Global instance
order_flow_alpha = OrderFlowAlpha()


def get_order_flow_alpha():
    """Get the global OrderFlowAlpha instance."""
    return order_flow_alpha
