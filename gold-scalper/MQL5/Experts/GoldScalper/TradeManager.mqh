//+------------------------------------------------------------------+
//| TradeManager.mqh — Virtual SL/TP + Time Stops + Partial Close   |
//|                                                                  |
//| MODES:                                                            |
//| - Physical SL only (safest, no VPS needed)                       |
//| - Virtual SL + Physical backup (balanced)                        |
//| - Full virtual SL/TP (most stealth, needs VPS for safety)        |
//|                                                                  |
//| Time Stop: Close position if no profit after N seconds           |
//| Partial Close: Close 70% at first TP, trail remaining 30%        |
//+------------------------------------------------------------------+
#ifndef TRADE_MANAGER_MQH
#define TRADE_MANAGER_MQH

struct VirtualPosition {
    ulong    ticket;
    double   entry_price;
    double   virtual_sl;         // Hidden from broker
    double   virtual_tp;         // Hidden from broker
    double   first_tp_price;     // Partial close target
    double   partial_close_pct;  // % to close at first TP
    datetime entry_time;
    int      time_stop_sec;      // Max seconds before time stop
    int      direction;          // ORDER_TYPE_BUY or ORDER_TYPE_SELL
    bool     partial_closed;     // Already did partial close?
    double   original_lots;      // Original position size
    double   trailing_sl;        // Trailing stop level (moves with price)
    bool     use_virtual_sl;
    bool     use_virtual_tp;
};

class CTradeManager {
private:
    VirtualPosition  m_positions[];
    int              m_max_positions;
    int              m_magic;
    bool             m_physical_sl;       // Also set physical SL as backup?
    int              m_physical_sl_pips;  // Physical SL distance in pips
    
    // Statistics
    int              m_total_trades;
    int              m_wins;
    int              m_losses;
    double           m_total_profit;
    double           m_total_loss;

public:
    CTradeManager(int max = 5, int magic = 20240513, 
                  bool physical_sl = true, int physical_sl_pips = 100) 
        : m_max_positions(max), m_magic(magic), 
          m_physical_sl(physical_sl), m_physical_sl_pips(physical_sl_pips) {
        m_total_trades = 0;
        m_wins = 0;
        m_losses = 0;
        m_total_profit = 0;
        m_total_loss = 0;
    }
    
    // ─── Open position with virtual and/or physical stops ───
    bool OpenPosition(int direction, double lots, 
                      double sl_price, double tp_price,
                      double first_tp = 0, double partial_pct = 0.7,
                      int time_stop = 30,
                      bool use_vsl = false, bool use_vtp = true) {
        
        MqlTradeRequest request = {};
        MqlTradeResult  result  = {};
        
        request.action    = TRADE_ACTION_DEAL;
        request.symbol    = _Symbol;
        request.volume    = lots;
        request.type      = (ENUM_ORDER_TYPE)direction;
        request.price     = (direction == ORDER_TYPE_BUY) ? 
                            SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                            SymbolInfoDouble(_Symbol, SYMBOL_BID);
        request.deviation = 50;
        request.magic     = m_magic;
        request.comment   = "GS_V1";
        
        // ─── Physical SL (safety net — always set if enabled) ───
        if(m_physical_sl && !use_vsl) {
            // Physical SL only (no virtual SL)
            request.sl = sl_price;
        } else if(m_physical_sl && use_vsl) {
            // Virtual SL + physical backup (wider)
            double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
            double backup_distance = m_physical_sl_pips * point;
            request.sl = (direction == ORDER_TYPE_BUY) ? 
                         request.price - backup_distance :
                         request.price + backup_distance;
        }
        // If use_vsl && !m_physical_sl → no physical SL at all (risky without VPS!)
        
        // Physical TP — only if NOT using virtual TP
        if(!use_vtp) {
            request.tp = tp_price;
        }
        // Virtual TP → no request.tp (broker can't see our target)
        
        if(!OrderSend(request, result)) {
            Print("Order FAILED: retcode=", result.retcode, " ", result.comment);
            return false;
        }
        
        if(result.retcode != TRADE_RETCODE_DONE && 
           result.retcode != TRADE_RETCODE_PLACED) {
            Print("Order rejected: retcode=", result.retcode);
            return false;
        }
        
        // Track virtual position
        VirtualPosition pos;
        pos.ticket            = result.order;
        pos.entry_price       = result.price;
        pos.virtual_sl        = sl_price;
        pos.virtual_tp        = tp_price;
        pos.first_tp_price    = first_tp;
        pos.partial_close_pct = partial_pct;
        pos.entry_time        = TimeCurrent();
        pos.time_stop_sec     = time_stop;
        pos.direction         = direction;
        pos.partial_closed    = false;
        pos.original_lots     = lots;
        pos.trailing_sl       = sl_price;  // Start at SL, will trail
        pos.use_virtual_sl    = use_vsl;
        pos.use_virtual_tp    = use_vtp;
        
        _AddPosition(pos);
        m_total_trades++;
        
        Print("Position OPENED: ", EnumToString((ENUM_ORDER_TYPE)direction),
              " @ ", DoubleToString(result.price, _Digits), 
              " lots=", DoubleToString(lots, 2),
              " vSL=", DoubleToString(sl_price, _Digits), 
              " vTP=", DoubleToString(tp_price, _Digits));
        return true;
    }
    
    // ─── Check virtual stops on every tick ───
    void OnTick() {
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        datetime now = TimeCurrent();
        
        for(int i = ArraySize(m_positions) - 1; i >= 0; i--) {
            // Check if position still exists
            if(!PositionSelectByTicket(m_positions[i].ticket)) {
                // Position was closed externally (physical SL hit, manual close, etc.)
                _RecordResult(m_positions[i], 0, "EXTERNAL_CLOSE");
                _RemovePosition(i);
                continue;
            }
            
            double current_price = (m_positions[i].direction == ORDER_TYPE_BUY) ? bid : ask;
            double profit_pips = _GetProfitPips(m_positions[i], current_price);
            
            // ─── Virtual SL hit ───
            if(m_positions[i].use_virtual_sl) {
                if(m_positions[i].direction == ORDER_TYPE_BUY && bid <= m_positions[i].trailing_sl) {
                    _ClosePosition(m_positions[i], "VSL_HIT");
                    _RemovePosition(i);
                    continue;
                }
                if(m_positions[i].direction == ORDER_TYPE_SELL && ask >= m_positions[i].trailing_sl) {
                    _ClosePosition(m_positions[i], "VSL_HIT");
                    _RemovePosition(i);
                    continue;
                }
                
                // ─── Trailing Stop: move SL up as price moves in our favor ───
                _UpdateTrailingStop(m_positions[i], current_price);
            }
            
            // ─── Virtual TP hit ───
            if(m_positions[i].use_virtual_tp) {
                if(m_positions[i].direction == ORDER_TYPE_BUY && bid >= m_positions[i].virtual_tp) {
                    _ClosePosition(m_positions[i], "VTP_HIT");
                    _RemovePosition(i);
                    continue;
                }
                if(m_positions[i].direction == ORDER_TYPE_SELL && ask <= m_positions[i].virtual_tp) {
                    _ClosePosition(m_positions[i], "VTP_HIT");
                    _RemovePosition(i);
                    continue;
                }
            }
            
            // ─── Partial close at first TP ───
            if(!m_positions[i].partial_closed && m_positions[i].first_tp_price > 0) {
                bool hit = (m_positions[i].direction == ORDER_TYPE_BUY) ? 
                           bid >= m_positions[i].first_tp_price :
                           ask <= m_positions[i].first_tp_price;
                if(hit) {
                    _PartialClose(m_positions[i]);
                    m_positions[i].partial_closed = true;
                }
            }
            
            // ─── Time stop: no profit after N seconds ───
            int elapsed = (int)(now - m_positions[i].entry_time);
            if(elapsed >= m_positions[i].time_stop_sec && profit_pips < 10) {
                _ClosePosition(m_positions[i], "TIME_STOP");
                _RemovePosition(i);
                continue;
            }
        }
    }
    
    // ─── Statistics ───
    int    GetTotalTrades() const { return m_total_trades; }
    int    GetWins()         const { return m_wins; }
    int    GetLosses()       const { return m_losses; }
    double GetWinRate()      const { return (m_total_trades > 0) ? (double)m_wins / m_total_trades : 0; }
    double GetTotalProfit()  const { return m_total_profit; }
    double GetTotalLoss()    const { return m_total_loss; }
    int    GetOpenPositions() const { return ArraySize(m_positions); }

private:
    double _GetProfitPips(const VirtualPosition &pos, double current_price) {
        double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        if(point <= 0) return 0;
        if(pos.direction == ORDER_TYPE_BUY)
            return (current_price - pos.entry_price) / point;
        else
            return (pos.entry_price - current_price) / point;
    }
    
    void _UpdateTrailingStop(VirtualPosition &pos, double current_price) {
        double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        double atr = iATRValue(14);
        if(atr <= 0) return;
        
        double trail_distance = atr * 0.5; // Trail at 0.5x ATR
        
        if(pos.direction == ORDER_TYPE_BUY) {
            double new_sl = current_price - trail_distance;
            if(new_sl > pos.trailing_sl) // Only move SL up, never down
                pos.trailing_sl = new_sl;
        } else {
            double new_sl = current_price + trail_distance;
            if(new_sl < pos.trailing_sl || pos.trailing_sl == pos.virtual_sl) // Only move SL down
                pos.trailing_sl = new_sl;
        }
    }
    
    void _ClosePosition(VirtualPosition &pos, string reason) {
        if(!PositionSelectByTicket(pos.ticket)) return;
        
        MqlTradeRequest request = {};
        MqlTradeResult  result  = {};
        
        request.action   = TRADE_ACTION_DEAL;
        request.symbol   = _Symbol;
        request.position = pos.ticket;
        request.volume   = PositionGetDouble(POSITION_VOLUME);
        request.type     = (pos.direction == ORDER_TYPE_BUY) ? 
                           ORDER_TYPE_SELL : ORDER_TYPE_BUY;
        request.price    = (pos.direction == ORDER_TYPE_BUY) ? 
                           SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                           SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        request.deviation = 50;
        
        if(OrderSend(request, result)) {
            double profit = PositionGetDouble(POSITION_PROFIT);
            _RecordResult(pos, profit, reason);
            Print("CLOSED [", reason, "] ticket=", pos.ticket, 
                  " profit=", DoubleToString(profit, 2));
        } else {
            Print("CLOSE FAILED [", reason, "] ticket=", pos.ticket,
                  " retcode=", result.retcode);
        }
    }
    
    void _PartialClose(VirtualPosition &pos) {
        if(!PositionSelectByTicket(pos.ticket)) return;
        
        double current_lots = PositionGetDouble(POSITION_VOLUME);
        double close_lots = NormalizeDouble(current_lots * pos.partial_close_pct, 2);
        double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        
        if(close_lots < min_lot) return;
        
        MqlTradeRequest request = {};
        MqlTradeResult  result  = {};
        
        request.action   = TRADE_ACTION_DEAL;
        request.symbol   = _Symbol;
        request.position = pos.ticket;
        request.volume   = close_lots;
        request.type     = (pos.direction == ORDER_TYPE_BUY) ? 
                           ORDER_TYPE_SELL : ORDER_TYPE_BUY;
        request.price    = (pos.direction == ORDER_TYPE_BUY) ? 
                           SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                           SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        request.deviation = 50;
        
        if(OrderSend(request, result)) {
            Print("PARTIAL CLOSE ", DoubleToString(pos.partial_close_pct * 100, 0),
                  "% @ ", DoubleToString(request.price, _Digits),
                  " remaining=", DoubleToString(current_lots - close_lots, 2), " lots");
        }
    }
    
    void _RecordResult(VirtualPosition &pos, double profit, string reason) {
        // Check if it was a win or loss based on close reason
        if(reason == "VTP_HIT" || reason == "EXTERNAL_CLOSE") {
            // Check actual profit from position
            if(PositionSelectByTicket(pos.ticket)) {
                profit = PositionGetDouble(POSITION_PROFIT);
            }
            if(profit >= 0) {
                m_wins++;
                m_total_profit += profit;
            } else {
                m_losses++;
                m_total_loss += MathAbs(profit);
            }
        } else {
            // VSL_HIT or TIME_STOP — likely a loss
            m_losses++;
            m_total_loss += MathAbs(profit);
        }
    }
    
    double iATRValue(int period) {
        int handle = iATR(_Symbol, PERIOD_M1, period);
        if(handle == INVALID_HANDLE) return 0;
        double atr[];
        ArraySetAsSeries(atr, true);
        int copied = CopyBuffer(handle, 0, 0, 1, atr);
        IndicatorRelease(handle);
        return (copied > 0) ? atr[0] : 0;
    }
    
    void _AddPosition(VirtualPosition &pos) {
        int size = ArraySize(m_positions);
        ArrayResize(m_positions, size + 1);
        m_positions[size] = pos;
    }
    
    void _RemovePosition(int index) {
        int last = ArraySize(m_positions) - 1;
        if(index < last)
            m_positions[index] = m_positions[last];
        ArrayResize(m_positions, last);
    }
};

#endif
//+------------------------------------------------------------------+
