# =============================================================
# backtest/trade_tracker.py
# Tracks open positions and records completed trades.
# Manages the P&L for each backtest trade.
# =============================================================

import datetime
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Represents a single backtest trade."""
    ticket: int
    symbol: str
    direction: str          # BUY or SELL
    strategy: str
    entry_time: datetime.datetime
    entry_price: float
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    score: int
    confluence: list = field(default_factory=list)
    session: str = 'UNKNOWN'
    market_state: str = 'UNKNOWN'

    # Filled when trade closes
    exit_time: datetime.datetime = None
    exit_price: float = 0.0
    profit_pips: float = 0.0
    profit_usd: float = 0.0
    outcome: str = ''        # WIN_TP, LOSS_SL, etc.
    duration_bars: int = 0


class TradeTracker:
    """Manages open positions and trade history for backtesting."""

    def __init__(self, pip_value_per_lot: float = 10.0,
                 starting_balance: float = 20000.0):
        self.open_trades: list[BacktestTrade] = []
        self.closed_trades: list[BacktestTrade] = []
        self.ticket_counter = 100000
        self.pip_value_per_lot = pip_value_per_lot
        self.balance = starting_balance
        self.equity_curve = [(datetime.datetime.now(datetime.timezone.utc), starting_balance)]
        self.max_open = 5
        self.max_per_symbol = 1

    def can_open(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        if len(self.open_trades) >= self.max_open:
            return False
        if any(t.symbol == symbol for t in self.open_trades):
            return False
        return True

    def open_trade(self, symbol: str, direction: str, strategy: str,
                   entry_time: datetime.datetime, entry_price: float,
                   sl_price: float, tp_price: float, sl_pips: float,
                   tp_pips: float, score: int, confluence: list,
                   session: str, market_state: str) -> BacktestTrade:
        """Record a new trade opening."""
        self.ticket_counter += 1
        trade = BacktestTrade(
            ticket=self.ticket_counter,
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            entry_time=entry_time,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            score=score,
            confluence=confluence,
            session=session,
            market_state=market_state,
        )
        self.open_trades.append(trade)
        return trade

    def check_exits(self, bar_time: datetime.datetime,
                    bar_high: float, bar_low: float,
                    bar_close: float, pip_size: float,
                    pip_value: float) -> list[BacktestTrade]:
        """
        Check all open trades for TP/SL hits on this bar.
        Returns list of trades that were closed this bar.
        """
        closed = []

        for trade in self.open_trades[:]:
            hit_tp = False
            hit_sl = False

            if trade.direction == 'BUY':
                if bar_high >= trade.tp_price:
                    hit_tp = True
                    trade.exit_price = trade.tp_price
                elif bar_low <= trade.sl_price:
                    hit_sl = True
                    trade.exit_price = trade.sl_price
            else:  # SELL
                if bar_low <= trade.tp_price:
                    hit_tp = True
                    trade.exit_price = trade.tp_price
                elif bar_high >= trade.sl_price:
                    hit_sl = True
                    trade.exit_price = trade.sl_price

            if hit_tp or hit_sl:
                trade.exit_time = bar_time
                trade.duration_bars = 1  # simplified

                # Calculate P&L
                if trade.direction == 'BUY':
                    trade.profit_pips = (trade.exit_price - trade.entry_price) / pip_size
                else:
                    trade.profit_pips = (trade.entry_price - trade.exit_price) / pip_size

                # Estimate USD P&L (0.01 lot, 1% risk)
                trade.profit_usd = trade.profit_pips * pip_value * 0.01

                trade.outcome = 'WIN_TP' if hit_tp else 'LOSS_SL'

                # Update balance
                self.balance += trade.profit_usd

                closed.append(trade)
                self.open_trades.remove(trade)
                self.closed_trades.append(trade)

                # Update equity curve
                self.equity_curve.append((bar_time, self.balance))

        return closed

    def get_summary(self) -> dict:
        """Get performance summary of all closed trades."""
        if not self.closed_trades:
            return {'total_trades': 0, 'wins': 0, 'losses': 0,
                    'win_rate': 0, 'total_pnl': 0, 'avg_win': 0,
                    'avg_loss': 0, 'profit_factor': 0, 'max_drawdown': 0}

        wins = [t for t in self.closed_trades if 'WIN' in t.outcome]
        losses = [t for t in self.closed_trades if 'LOSS' in t.outcome]
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = len(self.closed_trades)

        win_rate = (n_wins / n_total * 100) if n_total > 0 else 0

        total_pnl = sum(t.profit_usd for t in self.closed_trades)
        avg_win = sum(t.profit_pips for t in wins) / n_wins if n_wins > 0 else 0
        avg_loss = sum(t.profit_pips for t in losses) / n_losses if n_losses > 0 else 0

        gross_profit = sum(t.profit_usd for t in wins)
        gross_loss = abs(sum(t.profit_usd for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown from equity curve
        max_dd = self._calc_max_drawdown()

        # Per-strategy stats
        strategy_stats = {}
        for t in self.closed_trades:
            if t.strategy not in strategy_stats:
                strategy_stats[t.strategy] = {
                    'trades': 0, 'wins': 0, 'pnl': 0.0,
                    'pips': 0.0, 'scores': []
                }
            s = strategy_stats[t.strategy]
            s['trades'] += 1
            if 'WIN' in t.outcome:
                s['wins'] += 1
            s['pnl'] += t.profit_usd
            s['pips'] += t.profit_pips
            s['scores'].append(t.score)

        return {
            'total_trades': n_total,
            'wins': n_wins,
            'losses': n_losses,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'avg_win_pips': round(avg_win, 1),
            'avg_loss_pips': round(avg_loss, 1),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.0,
            'max_drawdown': round(max_dd, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'strategy_stats': strategy_stats,
            'final_balance': round(self.balance, 2),
        }

    def _calc_max_drawdown(self) -> float:
        """Calculate max drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        peak = self.equity_curve[0][1]
        max_dd = 0.0

        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        return max_dd
