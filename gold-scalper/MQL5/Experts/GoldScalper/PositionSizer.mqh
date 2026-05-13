//+------------------------------------------------------------------+
//| PositionSizer.mqh — Dynamic lot sizing based on risk %         |
//| Calculates lot size so that SL hit = X% of account balance     |
//+------------------------------------------------------------------+
#ifndef POSITION_SIZER_MQH
#define POSITION_SIZER_MQH

class CPositionSizer {
private:
    double   m_risk_pct;           // Risk % per trade
    double   m_max_risk_pct;       // Max risk % per session
    double   m_session_risk_used;  // Risk already used this session
    double   m_min_lot;            // Minimum lot size
    double   m_max_lot;            // Maximum lot size
    double   m_lot_step;           // Lot step size
    
public:
    CPositionSizer(double risk_pct = 1.0, double max_risk = 3.0) 
        : m_risk_pct(risk_pct), m_max_risk_pct(max_risk) {
        m_session_risk_used = 0;
        m_min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        m_max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
        m_lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        
        if(m_min_lot <= 0) m_min_lot = 0.01;
        if(m_max_lot <= 0) m_max_lot = 100.0;
        if(m_lot_step <= 0) m_lot_step = 0.01;
    }
    
    double Calculate(double sl_distance_price, double confidence = 0.7) {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        if(balance <= 0) return m_min_lot;
        
        // Check session risk limit
        double remaining_risk_pct = m_max_risk_pct - m_session_risk_used;
        if(remaining_risk_pct <= 0) {
            Print("⚠ Session risk limit reached: ", m_session_risk_used, "%");
            return 0; // No more trades this session
        }
        
        // Use the smaller of per-trade risk and remaining session risk
        double effective_risk_pct = MathMin(m_risk_pct, remaining_risk_pct);
        
        // Scale by confidence (0.5-1.0 multiplier)
        // Higher confidence = larger position
        double conf_mult = 0.5 + 0.5 * confidence; // 0.5 at 0 confidence, 1.0 at 1.0
        effective_risk_pct *= conf_mult;
        
        // Calculate dollar risk amount
        double risk_amount = balance * effective_risk_pct / 100.0;
        
        // Calculate lot size from SL distance
        double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        
        if(tick_value <= 0 || tick_size <= 0 || sl_distance_price <= 0)
            return m_min_lot;
        
        // Number of ticks in SL distance
        double sl_ticks = sl_distance_price / tick_size;
        
        // Dollar loss per lot if SL hit
        double loss_per_lot = sl_ticks * tick_value;
        
        if(loss_per_lot <= 0) return m_min_lot;
        
        // Lot size = risk_amount / loss_per_lot
        double lots = risk_amount / loss_per_lot;
        
        // Normalize to lot step
        lots = MathFloor(lots / m_lot_step) * m_lot_step;
        
        // Apply limits
        lots = MathMax(lots, m_min_lot);
        lots = MathMin(lots, m_max_lot);
        
        // Cap at 1% of balance in notional value (safety)
        double notional = lots * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE) * 
                          SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        if(notional > balance * 0.1) { // Max 10% of balance as notional
            lots = (balance * 0.1) / (SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE) * 
                   SymbolInfoDouble(_Symbol, SYMBOL_ASK));
            lots = MathFloor(lots / m_lot_step) * m_lot_step;
        }
        
        return NormalizeDouble(lots, 2);
    }
    
    void RecordTradeRisk(double risk_pct) {
        m_session_risk_used += risk_pct;
    }
    
    void ResetSession() {
        m_session_risk_used = 0;
    }
    
    double GetSessionRiskUsed() const { return m_session_risk_used; }
};

#endif
//+------------------------------------------------------------------+
