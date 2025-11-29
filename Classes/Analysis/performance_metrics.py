"""
Performance metrics calculation.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..Models.trade import Trade
from ..Engine.backtest_result import BacktestResult


class PerformanceMetrics:
    """
    Calculates performance metrics for backtest results.
    """

    @staticmethod
    def calculate_metrics(result: BacktestResult) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            result: Backtest result

        Returns:
            Dictionary of metrics
        """
        trades = result.trades
        equity_curve = result.equity_curve

        metrics = {
            'symbol': result.symbol,
            'strategy': result.strategy_name,
            'total_return': result.total_return,
            'total_return_pct': result.total_return_pct,
            'final_equity': result.final_equity,
            'num_trades': len(trades),
        }

        if len(trades) == 0:
            # No trades - add default values
            metrics.update({
                'win_rate': 0.0,
                'num_wins': 0,
                'num_losses': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'avg_trade_duration': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0
            })
            return metrics

        # Trade statistics
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = num_wins / len(trades) if len(trades) > 0 else 0.0

        # Win/loss amounts
        wins = [t.pl for t in winning_trades]
        losses = [t.pl for t in losing_trades]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Duration
        durations = [t.duration_days for t in trades]
        avg_duration = np.mean(durations) if durations else 0.0

        # FX P&L statistics
        total_security_pl = sum(t.security_pl for t in trades)
        total_fx_pl = sum(t.fx_pl for t in trades)
        fx_contribution_pct = (total_fx_pl / result.total_return * 100) if result.total_return != 0 else 0.0

        # Count trades with positive/negative FX impact
        fx_positive_trades = [t for t in trades if t.fx_pl > 0]
        fx_negative_trades = [t for t in trades if t.fx_pl < 0]
        fx_neutral_trades = [t for t in trades if t.fx_pl == 0]

        avg_fx_pl = np.mean([t.fx_pl for t in trades]) if trades else 0.0
        largest_fx_gain = max([t.fx_pl for t in trades]) if trades else 0.0
        largest_fx_loss = min([t.fx_pl for t in trades]) if trades else 0.0

        # Sharpe ratio (annualized)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(equity_curve)

        # Max drawdown
        max_dd, max_dd_pct = PerformanceMetrics.calculate_max_drawdown(equity_curve)

        metrics.update({
            'win_rate': win_rate,
            'num_wins': num_wins,
            'num_losses': num_losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_duration,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            # FX P&L metrics
            'total_security_pl': total_security_pl,
            'total_fx_pl': total_fx_pl,
            'fx_contribution_pct': fx_contribution_pct,
            'num_fx_positive': len(fx_positive_trades),
            'num_fx_negative': len(fx_negative_trades),
            'num_fx_neutral': len(fx_neutral_trades),
            'avg_fx_pl': avg_fx_pl,
            'largest_fx_gain': largest_fx_gain,
            'largest_fx_loss': largest_fx_loss
        })

        return metrics

    @staticmethod
    def calculate_sharpe_ratio(equity_curve: pd.DataFrame,
                              risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate daily returns
        returns = equity_curve['equity'].pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize (assuming daily data, 252 trading days)
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf

        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

        return sharpe

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.DataFrame) -> tuple:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column

        Returns:
            Tuple of (max_drawdown_dollars, max_drawdown_percent)
        """
        if len(equity_curve) == 0:
            return 0.0, 0.0

        equity = equity_curve['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        drawdown_pct = (drawdown / running_max) * 100

        max_dd = np.max(drawdown)
        max_dd_pct = np.max(drawdown_pct)

        return max_dd, max_dd_pct

    @staticmethod
    def print_metrics(metrics: Dict[str, Any]) -> None:
        """
        Print metrics in a formatted way.

        Args:
            metrics: Metrics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Performance Metrics: {metrics['symbol']} - {metrics['strategy']}")
        print(f"{'='*60}")
        print(f"Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        print(f"Final Equity:        ${metrics['final_equity']:,.2f}")
        print(f"Number of Trades:    {metrics['num_trades']}")
        print(f"Win Rate:            {metrics['win_rate']*100:.2f}%")
        print(f"Winning Trades:      {metrics['num_wins']}")
        print(f"Losing Trades:       {metrics['num_losses']}")
        print(f"Average Win:         ${metrics['avg_win']:,.2f}")
        print(f"Average Loss:        ${metrics['avg_loss']:,.2f}")
        print(f"Largest Win:         ${metrics['largest_win']:,.2f}")
        print(f"Largest Loss:        ${metrics['largest_loss']:,.2f}")
        print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"Avg Trade Duration:  {metrics['avg_trade_duration']:.1f} days")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:        ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")

        # FX P&L breakdown (if available)
        if 'total_fx_pl' in metrics and metrics.get('num_trades', 0) > 0:
            print(f"\n{'-'*60}")
            print(f"FX P&L Breakdown:")
            print(f"{'-'*60}")
            print(f"Security P&L:        ${metrics.get('total_security_pl', 0):,.2f}")
            print(f"FX P&L:              ${metrics.get('total_fx_pl', 0):,.2f}")
            print(f"FX Contribution:     {metrics.get('fx_contribution_pct', 0):.2f}% of total return")
            print(f"Avg FX P&L per trade: ${metrics.get('avg_fx_pl', 0):,.2f}")
            print(f"Largest FX Gain:     ${metrics.get('largest_fx_gain', 0):,.2f}")
            print(f"Largest FX Loss:     ${metrics.get('largest_fx_loss', 0):,.2f}")
            print(f"Trades with FX gain: {metrics.get('num_fx_positive', 0)}")
            print(f"Trades with FX loss: {metrics.get('num_fx_negative', 0)}")
            print(f"Trades with no FX:   {metrics.get('num_fx_neutral', 0)}")

        print(f"{'='*60}\n")
