# =============================================================
# backtest/trade_tracker.py  v2.0
# Tracks open positions and records completed trades.
# Manages the P&L, partial TP, ATR trailing, dynamic TP extension.
# Matches live order_manager.py position management logic.
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
    direction: str              # BUY or SELL
    strategy: str
    entry_time: datetime.datetime
    entry_price: float
    sl_price: float             # Current SL (gets updated by BE/trail)
    tp_price: float             # Current TP (gets updated by extension)
    original_sl_price: float    # Original SL at entry
    original_tp_price: float    # Original TP at entry
    sl_pips: float
    tp_pips: float
    score: int
    confluence: list = field(default_factory=list)
    session: str = 'UNKNOWN'
    market_state: str = 'UNKNOWN'
    agreement_groups: int = 2   # Number of consensus groups

    # Lot sizing
    lot_size: float = 0.01
    risk_percent: float = 1.0
    conviction: str = 'MEDIUM'  # LOW, MEDIUM, HIGH

    # Filled when trade closes
    exit_time: datetime.datetime = None
    exit_price: float = 0.0
    profit_pips: float = 0.0
    profit_r: float = 0.0       # R-multiple (relative to original SL)
    profit_usd: float = 0.0
    outcome: str = ''           # WIN_TP, WIN_PARTIAL_TP_TRAIL, LOSS_SL, etc.
    duration_bars: int = 0
    exit_reason: str = ''       # TP, PARTIAL_TP, SL, TRAIL_SL, EXTENDED_TP, END_OF_DATA

    # Partial TP tracking
    partial_tp_triggered: bool = False
    partial_tp_time: datetime.datetime = None
    partial_tp_price: float = 0.0
    partial_tp_pips: float = 0.0
    partial_tp_usd: float = 0.0   # Profit from the closed 50%

    # ATR trailing
    trail_activated: bool = False
    trail_distance: float = 0.0   # In price units (ATR × multiplier)
    highest_profit_pips: float = 0.0  # Peak profit (for max favorable excursion)

    # Dynamic TP extension
    tp_extended: bool = False
    extended_tp_price: float = 0.0


class TradeTracker:
    """Manages open positions and trade history for backtesting."""

    def __init__(self, starting_balance: float = 20000.0,
                 pip_value_per_lot: float = 10.0,
                 max_open: int = 5, max_per_symbol: int = 1,
                 partial_tp_enabled: bool = True,
                 atr_trail_enabled: bool = True,
                 dynamic_tp_enabled: bool = True,
                 dynamic_sizing_enabled: bool = True,
                 base_risk_percent: float = 1.0):
        self.open_trades: list[BacktestTrade] = []
        self.closed_trades: list[BacktestTrade] = []
        self.ticket_counter = 100000
        self.pip_value_per_lot = pip_value_per_lot
        self.balance = starting_balance
        self.equity_curve = [(datetime.datetime.now(datetime.timezone.utc), starting_balance)]
        self.max_open = max_open
        self.max_per_symbol = max_per_symbol
        self.starting_balance = starting_balance

        # Position management settings
        self.partial_tp_enabled = partial_tp_enabled
        self.atr_trail_enabled = atr_trail_enabled
        self.dynamic_tp_enabled = dynamic_tp_enabled
        self.dynamic_sizing_enabled = dynamic_sizing_enabled
        self.base_risk_percent = base_risk_percent

        # Consecutive loss tracking
        self._consecutive_losses = 0

    def get_consecutive_losses(self) -> int:
        return self._consecutive_losses

    def get_risk_percent(self, conviction: str = 'MEDIUM') -> float:
        """Get risk percent based on conviction and consecutive losses."""
        from backtest.config import (CONVICTION_LOW_SCORE_MAX, CONVICTION_MED_SCORE_MAX,
                                     CONSECUTIVE_LOSS_HALVE_THRESHOLD)

        if not self.dynamic_sizing_enabled:
            return self.base_risk_percent

        # Base conviction sizing
        if conviction == 'LOW':
            risk = 0.50
        elif conviction == 'HIGH':
            risk = 1.50
        else:
            risk = self.base_risk_percent

        # Halve after consecutive losses
        if self._consecutive_losses >= CONSECUTIVE_LOSS_HALVE_THRESHOLD:
            risk *= 0.5

        return risk

    def determine_conviction(self, score: int, agreement_groups: int) -> str:
        """Determine conviction level from score and agreement groups."""
        from backtest.config import (CONVICTION_LOW_SCORE_MAX, CONVICTION_MED_SCORE_MAX,
                                     CONVICTION_HIGH_MIN_GROUPS)
        if score >= CONVICTION_MED_SCORE_MAX and agreement_groups >= CONVICTION_HIGH_MIN_GROUPS:
            return 'HIGH'
        elif score >= CONVICTION_LOW_SCORE_MAX:
            return 'MEDIUM'
        else:
            return 'LOW'

    def can_open(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        if len(self.open_trades) >= self.max_open:
            return False
        open_for_symbol = sum(1 for t in self.open_trades if t.symbol == symbol)
        if open_for_symbol >= self.max_per_symbol:
            return False
        return True

    def open_trade(self, symbol: str, direction: str, strategy: str,
                   entry_time: datetime.datetime, entry_price: float,
                   sl_price: float, tp_price: float, sl_pips: float,
                   tp_pips: float, score: int, confluence: list,
                   session: str, market_state: str,
                   agreement_groups: int = 2,
                   atr_value: float = 0.0) -> BacktestTrade:
        """Record a new trade opening."""
        self.ticket_counter += 1

        # Determine conviction
        conviction = self.determine_conviction(score, agreement_groups)
        risk_percent = self.get_risk_percent(conviction)

        # Calculate lot size based on risk
        # Risk USD = balance × risk_percent / 100
        # Lot size = Risk USD / (SL_pips × pip_value_per_lot)
        risk_usd = self.balance * risk_percent / 100.0
        lot_size = risk_usd / (sl_pips * self.pip_value_per_lot) if sl_pips > 0 else 0.01
        lot_size = max(0.01, round(lot_size, 2))  # Floor at 0.01 lot

        # Calculate ATR trail distance
        trail_distance = 0.0
        if self.atr_trail_enabled and atr_value > 0:
            from backtest.config import ATR_TRAIL_MULTIPLIER
            trail_distance = atr_value * ATR_TRAIL_MULTIPLIER

        trade = BacktestTrade(
            ticket=self.ticket_counter,
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            entry_time=entry_time,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            original_sl_price=sl_price,
            original_tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            score=score,
            confluence=confluence,
            session=session,
            market_state=market_state,
            agreement_groups=agreement_groups,
            lot_size=lot_size,
            risk_percent=risk_percent,
            conviction=conviction,
            trail_distance=trail_distance,
        )
        self.open_trades.append(trade)
        return trade

    def check_exits(self, bar_time: datetime.datetime,
                    bar_high: float, bar_low: float,
                    bar_close: float, pip_size: float,
                    pip_value: float) -> list[BacktestTrade]:
        """
        Check all open trades for TP/SL/partial TP/trailing on this bar.
        Returns list of trades that were closed this bar.

        Order of operations (matches live order_manager.py):
        1. Check TP hit
        2. Check SL hit (after TP — TP takes priority if both hit)
        3. Check partial TP trigger
        4. Apply BE move after partial TP
        5. Apply ATR trailing
        6. Apply dynamic TP extension
        """
        from backtest.config import (PARTIAL_TP_ENABLED, PARTIAL_TP_RATIO,
                                     PARTIAL_TP_AT_R_MULTIPLE,
                                     DYNAMIC_TP_TRIGGER_PCT,
                                     DYNAMIC_TP_MULTIPLIER_ATR)

        closed = []

        for trade in self.open_trades[:]:
            current_profit_pips = self._calc_profit_pips(trade, bar_close, pip_size)
            trade.highest_profit_pips = max(trade.highest_profit_pips, current_profit_pips)

            hit_tp = False
            hit_sl = False

            # ── Check if TP hit (considering extended TP) ──
            effective_tp = trade.extended_tp_price if trade.tp_extended else trade.tp_price

            if trade.direction == 'BUY':
                if bar_high >= effective_tp:
                    hit_tp = True
                    trade.exit_price = effective_tp
                elif bar_low <= trade.sl_price:
                    hit_sl = True
                    trade.exit_price = trade.sl_price
            else:  # SELL
                if bar_low <= effective_tp:
                    hit_tp = True
                    trade.exit_price = effective_tp
                elif bar_high >= trade.sl_price:
                    hit_sl = True
                    trade.exit_price = trade.sl_price

            # ── Process TP hit ──
            if hit_tp:
                trade.exit_time = bar_time
                trade.profit_pips = self._calc_profit_pips(trade, trade.exit_price, pip_size)
                trade.profit_r = trade.profit_pips / trade.sl_pips if trade.sl_pips > 0 else 0

                # P&L for full position (if no partial TP was triggered)
                # Or for remaining 50% (if partial TP was triggered)
                if trade.partial_tp_triggered:
                    # Remaining 50% hit TP/SL
                    remaining_lot = trade.lot_size * (1.0 - PARTIAL_TP_RATIO)
                    trade.profit_usd = trade.profit_pips * pip_value * remaining_lot
                    trade.outcome = 'WIN_TRAIL_TP' if trade.trail_activated else 'WIN_TP_REMAINING'
                    trade.exit_reason = 'EXTENDED_TP' if trade.tp_extended else 'TP'
                else:
                    # Full position hit TP
                    trade.profit_usd = trade.profit_pips * pip_value * trade.lot_size
                    trade.outcome = 'WIN_TP'
                    trade.exit_reason = 'TP'

                self.balance += trade.profit_usd
                closed.append(trade)
                self.open_trades.remove(trade)
                self.closed_trades.append(trade)
                self.equity_curve.append((bar_time, self.balance))
                self._consecutive_losses = 0
                continue

            # ── Process SL hit ──
            if hit_sl:
                trade.exit_time = bar_time
                trade.profit_pips = self._calc_profit_pips(trade, trade.exit_price, pip_size)
                trade.profit_r = trade.profit_pips / trade.sl_pips if trade.sl_pips > 0 else 0

                if trade.partial_tp_triggered:
                    # Only 50% remaining — reduced loss
                    remaining_lot = trade.lot_size * (1.0 - PARTIAL_TP_RATIO)
                    trade.profit_usd = trade.profit_pips * pip_value * remaining_lot
                    trade.outcome = 'LOSS_SL_PARTIAL'
                else:
                    trade.profit_usd = trade.profit_pips * pip_value * trade.lot_size
                    trade.outcome = 'LOSS_SL'

                trade.exit_reason = 'SL'
                self.balance += trade.profit_usd
                self._consecutive_losses += 1
                closed.append(trade)
                self.open_trades.remove(trade)
                self.closed_trades.append(trade)
                self.equity_curve.append((bar_time, self.balance))
                continue

            # ── Partial TP Check ──
            if (self.partial_tp_enabled and PARTIAL_TP_ENABLED
                    and not trade.partial_tp_triggered):

                one_r_pips = trade.sl_pips * PARTIAL_TP_AT_R_MULTIPLE
                if current_profit_pips >= one_r_pips:
                    # Trigger partial TP: close 50% at current price
                    trade.partial_tp_triggered = True
                    trade.partial_tp_time = bar_time
                    trade.partial_tp_pips = current_profit_pips

                    # Close 50% at favorable exit (use midpoint between entry + 1R for realism)
                    if trade.direction == 'BUY':
                        trade.partial_tp_price = trade.entry_price + one_r_pips * pip_size
                    else:
                        trade.partial_tp_price = trade.entry_price - one_r_pips * pip_size

                    partial_lot = trade.lot_size * PARTIAL_TP_RATIO
                    trade.partial_tp_usd = current_profit_pips * pip_value * partial_lot

                    # Move SL to breakeven (entry price + 0.5 pip buffer)
                    be_buffer = 0.5 * pip_size
                    if trade.direction == 'BUY':
                        trade.sl_price = trade.entry_price + be_buffer
                    else:
                        trade.sl_price = trade.entry_price - be_buffer

                    # Book partial profit to balance
                    self.balance += trade.partial_tp_usd
                    self.equity_curve.append((bar_time, self.balance))

                    log.info(f"  [Partial TP] {trade.symbol} {trade.direction} "
                             f"closed 50% at {trade.partial_tp_pips:.1f} pips "
                             f"(${trade.partial_tp_usd:+.2f}), SL → BE")

            # ── ATR Trailing Stop (only after partial TP triggered) ──
            if (self.atr_trail_enabled and trade.partial_tp_triggered
                    and trade.trail_distance > 0):

                trade.trail_activated = True

                if trade.direction == 'BUY':
                    # Trail SL up: new SL = max(current SL, bar_high - trail_distance)
                    new_sl = bar_high - trade.trail_distance
                    if new_sl > trade.sl_price:
                        trade.sl_price = new_sl
                else:
                    # Trail SL down: new SL = min(current SL, bar_low + trail_distance)
                    new_sl = bar_low + trade.trail_distance
                    if new_sl < trade.sl_price:
                        trade.sl_price = new_sl

            # ── Dynamic TP Extension ──
            if (self.dynamic_tp_enabled and not trade.tp_extended
                    and trade.partial_tp_triggered):

                # Trigger when 60% of the way to original TP
                if trade.direction == 'BUY':
                    progress = (bar_close - trade.entry_price) / (trade.original_tp_price - trade.entry_price)
                else:
                    progress = (trade.entry_price - bar_close) / (trade.entry_price - trade.original_tp_price)

                if progress >= DYNAMIC_TP_TRIGGER_PCT:
                    # Extend TP by max(2×ATR, 1.5×trail_distance)
                    atr_extend = 2.0 * trade.trail_distance if trade.trail_distance > 0 else 0
                    trail_extend = 1.5 * trade.trail_distance if trade.trail_distance > 0 else 0
                    extension = max(atr_extend, trail_extend)

                    if extension > 0:
                        if trade.direction == 'BUY':
                            trade.extended_tp_price = trade.original_tp_price + extension
                        else:
                            trade.extended_tp_price = trade.original_tp_price - extension

                        trade.tp_extended = True
                        trade.tp_price = trade.extended_tp_price

        return closed

    def _calc_profit_pips(self, trade: BacktestTrade,
                          current_price: float, pip_size: float) -> float:
        """Calculate profit in pips for a trade at current price."""
        if trade.direction == 'BUY':
            return (current_price - trade.entry_price) / pip_size
        else:
            return (trade.entry_price - current_price) / pip_size

    def close_remaining_at_end(self, last_bar_time: datetime.datetime,
                                last_bar_close: float,
                                pip_size: float,
                                pip_value: float):
        """Force-close all remaining open trades at end of backtest."""
        for trade in self.open_trades[:]:
            trade.exit_time = last_bar_time
            trade.exit_price = last_bar_close
            trade.profit_pips = self._calc_profit_pips(trade, trade.exit_price, pip_size)
            trade.profit_r = trade.profit_pips / trade.sl_pips if trade.sl_pips > 0 else 0

            remaining_lot = trade.lot_size * (1.0 - 0.5) if trade.partial_tp_triggered else trade.lot_size
            trade.profit_usd = trade.profit_pips * pip_value * remaining_lot
            trade.outcome = 'WIN' if trade.profit_pips > 0 else 'LOSS'
            trade.exit_reason = 'END_OF_DATA'

            self.balance += trade.profit_usd
            self.closed_trades.append(trade)
            self.equity_curve.append((last_bar_time, self.balance))

        self.open_trades = []

    def get_summary(self) -> dict:
        """Get performance summary of all closed trades."""
        if not self.closed_trades:
            return {'total_trades': 0, 'wins': 0, 'losses': 0,
                    'win_rate': 0, 'total_pnl': 0, 'avg_win': 0,
                    'avg_loss': 0, 'profit_factor': 0, 'max_drawdown': 0}

        wins = [t for t in self.closed_trades if t.profit_pips > 0]
        losses = [t for t in self.closed_trades if t.profit_pips <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = len(self.closed_trades)

        win_rate = (n_wins / n_total * 100) if n_total > 0 else 0

        # Total P&L includes partial TP profits
        total_pnl = sum(t.profit_usd for t in self.closed_trades)
        # Add partial TP profits (already added to balance during trade)
        partial_profits = sum(t.partial_tp_usd for t in self.closed_trades
                              if t.partial_tp_triggered)
        total_pnl += partial_profits

        avg_win = sum(t.profit_pips for t in wins) / n_wins if n_wins > 0 else 0
        avg_loss = sum(t.profit_pips for t in losses) / n_losses if n_losses > 0 else 0

        # R-multiple stats
        avg_win_r = sum(t.profit_r for t in wins) / n_wins if n_wins > 0 else 0
        avg_loss_r = sum(t.profit_r for t in losses) / n_losses if n_losses > 0 else 0

        # Expected value per trade
        ev_per_trade = 0.0
        if n_total > 0:
            win_rate_dec = n_wins / n_total
            loss_rate_dec = n_losses / n_total
            ev_per_trade = (win_rate_dec * avg_win_r) - (loss_rate_dec * abs(avg_loss_r))

        gross_profit = sum(t.profit_usd for t in wins)
        gross_loss = abs(sum(t.profit_usd for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown from equity curve
        max_dd, max_dd_pct = self._calc_max_drawdown()

        # Max drawdown duration (in bars/periods)
        dd_duration = self._calc_max_dd_duration()

        # Sharpe ratio (annualized, assuming ~252 trading days × 6 hours active = ~1500 bars scanned)
        sharpe = self._calc_sharpe_ratio()

        # Per-strategy stats
        strategy_stats = {}
        for t in self.closed_trades:
            if t.strategy not in strategy_stats:
                strategy_stats[t.strategy] = {
                    'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0,
                    'pips': 0.0, 'scores': [], 'avg_r': 0.0, 'r_multiples': [],
                    'partial_tp_count': 0, 'partial_tp_profit': 0.0,
                    'trail_count': 0,
                }
            s = strategy_stats[t.strategy]
            s['trades'] += 1
            if t.profit_pips > 0:
                s['wins'] += 1
            else:
                s['losses'] += 1
            s['pnl'] += t.profit_usd
            s['pips'] += t.profit_pips
            s['scores'].append(t.score)
            s['r_multiples'].append(t.profit_r)
            if t.partial_tp_triggered:
                s['partial_tp_count'] += 1
                s['partial_tp_profit'] += t.partial_tp_usd
            if t.trail_activated:
                s['trail_count'] += 1

        # Calculate avg R per strategy
        for s in strategy_stats.values():
            if s['r_multiples']:
                s['avg_r'] = round(sum(s['r_multiples']) / len(s['r_multiples']), 2)
            del s['r_multiples']

        # Per-session stats
        session_stats = {}
        for t in self.closed_trades:
            if t.session not in session_stats:
                session_stats[t.session] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pips': 0.0}
            session_stats[t.session]['trades'] += 1
            if t.profit_pips > 0:
                session_stats[t.session]['wins'] += 1
            session_stats[t.session]['pnl'] += t.profit_usd
            session_stats[t.session]['pips'] += t.profit_pips

        # Per-market-state stats
        state_stats = {}
        for t in self.closed_trades:
            if t.market_state not in state_stats:
                state_stats[t.market_state] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pips': 0.0}
            state_stats[t.market_state]['trades'] += 1
            if t.profit_pips > 0:
                state_stats[t.market_state]['wins'] += 1
            state_stats[t.market_state]['pnl'] += t.profit_usd
            state_stats[t.market_state]['pips'] += t.profit_pips

        # Conviction stats
        conviction_stats = {}
        for t in self.closed_trades:
            if t.conviction not in conviction_stats:
                conviction_stats[t.conviction] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            conviction_stats[t.conviction]['trades'] += 1
            if t.profit_pips > 0:
                conviction_stats[t.conviction]['wins'] += 1
            conviction_stats[t.conviction]['pnl'] += t.profit_usd

        # Outcome breakdown
        outcome_counts = {}
        for t in self.closed_trades:
            outcome_counts[t.outcome] = outcome_counts.get(t.outcome, 0) + 1

        # Partial TP stats
        partial_trades = [t for t in self.closed_trades if t.partial_tp_triggered]
        partial_count = len(partial_trades)
        partial_wins = len([t for t in partial_trades if t.profit_pips > 0])
        partial_total_profit = sum(t.partial_tp_usd for t in partial_trades)

        # Max favorable excursion (best trade)
        mfe = max((t.highest_profit_pips for t in self.closed_trades), default=0)

        return {
            'total_trades': n_total,
            'wins': n_wins,
            'losses': n_losses,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'avg_win_pips': round(avg_win, 1),
            'avg_loss_pips': round(avg_loss, 1),
            'avg_win_r': round(avg_win_r, 2),
            'avg_loss_r': round(avg_loss_r, 2),
            'ev_per_trade': round(ev_per_trade, 3),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.0,
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct, 1),
            'max_dd_duration': dd_duration,
            'sharpe_ratio': round(sharpe, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'final_balance': round(self.balance, 2),
            'return_pct': round((self.balance - self.starting_balance) / self.starting_balance * 100, 2),
            'strategy_stats': strategy_stats,
            'session_stats': session_stats,
            'state_stats': state_stats,
            'conviction_stats': conviction_stats,
            'outcome_counts': outcome_counts,
            'partial_tp_stats': {
                'trades_triggered': partial_count,
                'wins_after_partial': partial_wins,
                'total_partial_profit': round(partial_total_profit, 2),
                'win_rate_after_partial': round(partial_wins / partial_count * 100, 1) if partial_count > 0 else 0,
            },
            'mfe_pips': round(mfe, 1),
        }

    def _calc_max_drawdown(self) -> tuple:
        """Calculate max drawdown ($ and %) from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0

        peak = self.equity_curve[0][1]
        max_dd = 0.0
        max_dd_pct = 0.0

        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return max_dd, max_dd_pct

    def _calc_max_dd_duration(self) -> int:
        """Calculate max drawdown duration in equity curve points."""
        if len(self.equity_curve) < 2:
            return 0

        peak = self.equity_curve[0][1]
        max_duration = 0
        current_duration = 0

        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
                current_duration = 0
            elif equity < peak:
                current_duration += 1
                max_duration = max(max_duration, current_duration)

        return max_duration

    def _calc_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade returns."""
        if len(self.closed_trades) < 2:
            return 0.0

        returns = [t.profit_r for t in self.closed_trades]
        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_return = variance ** 0.5

        if std_return == 0:
            return 0.0

        # Annualize: ~250 trading days, ~4 trades per day in active market
        trades_per_year = len(returns) / 0.5  # 6 months backtest → annualize
        sharpe = (avg_return / std_return) * (trades_per_year ** 0.5)

        return sharpe
