# dashboard/components/db_helper.py
# Shared database query helpers for all dashboard pages

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__),
                       '..', '..', 'database', 'apex_trader.db')


def get_trades(limit: int = 500) -> pd.DataFrame:
    """Fetch all completed trades."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query("""
            SELECT * FROM trades
            ORDER BY timestamp_open DESC
            LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_open_positions() -> pd.DataFrame:
    """Fetch trades with no close timestamp."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query("""
            SELECT * FROM trades
            WHERE timestamp_close IS NULL
            ORDER BY timestamp_open DESC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_signals(limit: int = 200) -> pd.DataFrame:
    """Fetch recent signals — traded and skipped."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query("""
            SELECT * FROM signals
            ORDER BY timestamp DESC
            LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_strategy_stats() -> pd.DataFrame:
    """Aggregate win rate and P&L per strategy."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query("""
            SELECT
                strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome LIKE 'WIN%' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                ROUND(AVG(profit_loss), 2) as avg_pnl,
                ROUND(SUM(profit_loss), 2) as total_pnl,
                ROUND(100.0 * SUM(CASE WHEN outcome LIKE 'WIN%' THEN 1 ELSE 0 END)
                      / COUNT(*), 1) as win_rate
            FROM trades
            WHERE outcome IS NOT NULL
            GROUP BY strategy
            ORDER BY win_rate DESC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def get_live_positions_mt5() -> list:
    """Fetch live open positions directly from MT5 with current P&L."""
    try:
        import MetaTrader5 as mt5
        from dotenv import load_dotenv
        import os
        load_dotenv()
        mt5.initialize()
        mt5.login(int(os.getenv('MT5_LOGIN', 0)),
                  password=os.getenv('MT5_PASSWORD', ''),
                  server=os.getenv('MT5_SERVER', ''))
        positions = mt5.positions_get()
        result = []
        if positions:
            for p in positions:
                result.append({
                    'ticket':      p.ticket,
                    'symbol':      p.symbol,
                    'direction':   'BUY' if p.type == 0 else 'SELL',
                    'volume':      p.volume,
                    'entry_price': p.price_open,
                    'current_price': p.price_current,
                    'sl':          p.sl,
                    'tp':          p.tp,
                    'profit':      round(p.profit, 2),
                    'swap':        round(p.swap, 2),
                    'comment':     p.comment,
                    'open_time':   p.time,
                    'magic':       p.magic,
                })
        mt5.shutdown()
        return result
    except Exception as e:
        return []


def close_position_mt5(ticket: int) -> dict:
    """Close a specific position by ticket number."""
    try:
        import MetaTrader5 as mt5
        from dotenv import load_dotenv
        import os
        load_dotenv()
        mt5.initialize()
        mt5.login(int(os.getenv('MT5_LOGIN', 0)),
                  password=os.getenv('MT5_PASSWORD', ''),
                  server=os.getenv('MT5_SERVER', ''))

        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return {'success': False, 'error': 'Position not found'}

        p    = pos[0]
        tick = mt5.symbol_info_tick(p.symbol)
        price = tick.bid if p.type == 0 else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY

        request = {
            'action':      mt5.TRADE_ACTION_DEAL,
            'symbol':      p.symbol,
            'volume':      p.volume,
            'type':        order_type,
            'position':    ticket,
            'price':       price,
            'deviation':   20,
            'magic':       p.magic,
            'comment':     'Manual close from dashboard',
            'type_time':   mt5.ORDER_TIME_GTC,
            'type_filling':mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        mt5.shutdown()

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {'success': True, 'profit': p.profit}
        return {'success': False,
                'error': f'retcode={result.retcode if result else "None"}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def clear_all_database_data() -> dict:
    """Delete ALL data from all database tables."""
    try:
        conn = sqlite3.connect(DB_PATH)
        tables = ['trades', 'signals', 'market_snapshots',
                  'strategy_performance']
        deleted = {}
        for table in tables:
            try:
                cursor = conn.cursor()
                cursor.execute(f"DELETE FROM {table}")
                deleted[table] = cursor.rowcount
            except Exception:
                deleted[table] = 0
        conn.commit()
        conn.close()
        return {'success': True, 'deleted': deleted}
    except Exception as e:
        return {'success': False, 'error': str(e)}
