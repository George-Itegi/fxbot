# =============================================================
# data_layer/feature_store.py
# PURPOSE: Unified Feature Store for fxbot.
# Ensures every strategy and model uses the same consistent schema.
# =============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class FeatureStore:
    """
    A centralized repository for all market features.
    Standardizes data for strategies, AI models, and the Strategy Lab.
    """
    def __init__(self):
        self.features: Dict[str, Dict[str, Any]] = {}

    def update_symbol_features(self, symbol: str, 
                               market_data: Dict[str, Any], 
                               smc_data: Dict[str, Any],
                               external_data: Optional[Dict[str, Any]] = None):
        """
        Store and normalize features for a specific symbol.
        """
        # Flattening and standardizing the schema
        standardized = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            
            # --- Price & Volatility ---
            'current_price': float(market_data.get('current_price', 0) or 0),
            'atr_m15': float(market_data.get('atr', 0) or 0),
            
            # --- Order Flow (Microstructure) ---
            'delta_full': int(market_data.get('delta', {}).get('delta', 0) or 0),
            'delta_rolling': int(market_data.get('rolling_delta', {}).get('delta', 0) or 0),
            'delta_bias': str(market_data.get('delta', {}).get('bias', 'NEUTRAL')),
            'delta_strength': str(market_data.get('delta', {}).get('strength', 'WEAK')),

            # --- Order Flow Imbalance (NEW) ---
            'of_imbalance': float(market_data.get('order_flow_imbalance', {}).get('imbalance', 0) or 0),
            'of_direction': str(market_data.get('order_flow_imbalance', {}).get('direction', 'NEUTRAL')),
            'of_strength': str(market_data.get('order_flow_imbalance', {}).get('strength', 'NONE')),
            'of_can_buy': bool(market_data.get('order_flow_imbalance', {}).get('can_buy', False)),
            'of_can_sell': bool(market_data.get('order_flow_imbalance', {}).get('can_sell', False)),

            # --- Volume Surge Detection (NEW) ---
            'volume_surge_detected': bool(market_data.get('volume_surge', {}).get('surge_detected', False)),
            'volume_surge_ratio': float(market_data.get('volume_surge', {}).get('surge_ratio', 0) or 0),
            'volume_surge_direction': str(market_data.get('volume_surge', {}).get('surge_direction', 'NEUTRAL')),

            # --- Momentum Velocity (NEW) ---
            'momentum_velocity': float(market_data.get('momentum', {}).get('velocity_pips_min', 0) or 0),
            'momentum_direction': str(market_data.get('momentum', {}).get('velocity_direction', 'FLAT')),
            'momentum_is_scalpable': bool(market_data.get('momentum', {}).get('is_scalpable', False)),
            'momentum_is_choppy': bool(market_data.get('momentum', {}).get('is_choppy', True)),
            
            # --- Context (VWAP & Volume) ---
            'dist_from_vwap': float(market_data.get('vwap', {}).get('pip_from_vwap', 0) or 0),
            'vwap_position': str(market_data.get('vwap', {}).get('position', 'UNKNOWN')),
            'dist_from_poc': float(market_data.get('profile', {}).get('pip_to_poc', 0) or 0),
            'va_zone': str(market_data.get('profile', {}).get('price_position', 'UNKNOWN')),
            
            # --- SMC (Institutional Structure) ---
            'smc_trend': str(smc_data.get('structure', {}).get('trend', 'NEUTRAL')),
            'smc_bias': str(smc_data.get('smc_bias', 'NEUTRAL')),
            'smc_score': int(smc_data.get('smc_score', 0) or 0),
            'htf_approved': bool(smc_data.get('htf_alignment', {}).get('approved', False)),
            'pd_zone': str(smc_data.get('premium_discount', {}).get('zone', 'UNKNOWN')),
            'pd_bias': str(smc_data.get('premium_discount', {}).get('bias', 'NEUTRAL')),
            
            # --- Liquidity & Blocks ---
            'dist_to_nearest_ob': self._calc_dist(market_data.get('current_price'), smc_data.get('nearest_ob')),
            'dist_to_nearest_pool': self._calc_dist(market_data.get('current_price'), smc_data.get('nearest_pool')),
            'last_sweep_bias': str(smc_data.get('last_sweep', {}).get('bias', 'NONE')),
            'last_sweep_reversal': float(smc_data.get('last_sweep', {}).get('reversal_pips', 0) or 0),
        }
        
        if external_data:
            standardized.update({
                'fear_greed': external_data.get('fear_greed'),
                'news_impact': external_data.get('news_impact', 'LOW'),
            })

        self.features[symbol] = standardized

    def get_features(self, symbol: str) -> Dict[str, Any]:
        return self.features.get(symbol, {})

    def _calc_dist(self, current_price, level_data):
        if not current_price or not level_data:
            return 9999.0
        # level_data could be OB (mid) or Pool (level)
        target = level_data.get('mid') or level_data.get('level')
        if not target: return 9999.0
        
        pip_size = 0.01 if current_price > 50 else 0.0001
        return abs(current_price - target) / pip_size

# Global instance
store = FeatureStore()
