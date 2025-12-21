"""
Performance metrics calculation.

IMPORTANT: Risk-free rate is standardized at 3.5% (UK base rate approximation)
across all risk-adjusted metrics for consistency.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..Models.trade import Trade
from ..Engine.backtest_result import BacktestResult

# Standardized risk-free rate for all calculations (UK base rate approximation)
DEFAULT_RISK_FREE_RATE = 0.035  # 3.5% annual


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
            # No trades - add default trade-specific values
            # BUT still calculate Sharpe/Sortino from equity curve
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(equity_curve)
            sortino = PerformanceMetrics.calculate_sortino_ratio(equity_curve)
            max_dd, max_dd_pct = PerformanceMetrics.calculate_max_drawdown(equity_curve)

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
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'max_drawdown_pct': max_dd_pct
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
        # When all trades are winners, use a large value (not infinity) to indicate excellent performance
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        elif total_wins > 0:
            profit_factor = 999.99  # All winning trades - exceptional performance
        else:
            profit_factor = 0.0  # No trades or no profit

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

        # Volatility and downside deviation
        volatility = PerformanceMetrics.calculate_volatility(equity_curve)
        downside_deviation = PerformanceMetrics.calculate_downside_deviation(equity_curve)

        # Sortino ratio
        sortino = PerformanceMetrics.calculate_sortino_ratio(equity_curve)

        # Calculate CAGR
        if len(equity_curve) >= 2:
            start_date = pd.Timestamp(equity_curve['date'].iloc[0])
            end_date = pd.Timestamp(equity_curve['date'].iloc[-1])
            years = (end_date - start_date).days / 365.25
            initial_equity = equity_curve['equity'].iloc[0]
            final_equity = equity_curve['equity'].iloc[-1]

            if years > 0 and initial_equity > 0:
                cagr = (pow(final_equity / initial_equity, 1 / years) - 1) * 100
            else:
                cagr = 0.0
        else:
            cagr = 0.0

        # Calmar ratio (CAGR / Max DD)
        calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

        # Best and worst day
        best_day = PerformanceMetrics.calculate_best_day(equity_curve)
        worst_day = PerformanceMetrics.calculate_worst_day(equity_curve)

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
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'best_day': best_day,
            'worst_day': worst_day,
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
    def calculate_from_trades(trades: List['Trade'], initial_capital: float) -> Dict[str, Any]:
        """
        Calculate performance metrics from a list of trades without requiring a BacktestResult.

        This is useful for calculating metrics for subsets of trades (e.g., per-security
        in a portfolio backtest).

        Args:
            trades: List of Trade objects
            initial_capital: Initial capital for percentage calculations

        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'num_trades': len(trades),
        }

        if len(trades) == 0:
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
                'max_drawdown_pct': 0.0,
                'total_return': 0.0,
                'total_return_pct': 0.0
            })
            return metrics

        # Trade statistics
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = (num_wins / len(trades) * 100) if len(trades) > 0 else 0.0

        # Win/loss amounts
        wins = [t.pl for t in winning_trades]
        losses = [t.pl for t in losing_trades]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Total return
        total_return = sum(t.pl for t in trades)
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        elif total_wins > 0:
            profit_factor = 999.99
        else:
            profit_factor = 0.0

        # Duration
        durations = [t.duration_days for t in trades]
        avg_duration = np.mean(durations) if durations else 0.0

        # Simple drawdown estimation from trades (not as accurate as equity curve)
        # This is an approximation based on cumulative P/L
        cumulative_pl = []
        running_pl = 0
        for t in sorted(trades, key=lambda x: x.exit_date):
            running_pl += t.pl
            cumulative_pl.append(running_pl)

        if cumulative_pl:
            running_max = 0
            max_dd = 0
            for pl in cumulative_pl:
                running_max = max(running_max, pl)
                drawdown = running_max - pl
                max_dd = max(max_dd, drawdown)
            max_dd_pct = (max_dd / initial_capital * 100) if initial_capital > 0 else 0.0
        else:
            max_dd = 0.0
            max_dd_pct = 0.0

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
            'sharpe_ratio': 0.0,  # Cannot calculate without equity curve
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'total_return': total_return,
            'total_return_pct': total_return_pct
        })

        return metrics

    @staticmethod
    def calculate_sharpe_ratio(equity_curve: pd.DataFrame,
                              risk_free_rate: float = None) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column
            risk_free_rate: Annual risk-free rate (default: DEFAULT_RISK_FREE_RATE = 3.5%)

        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = DEFAULT_RISK_FREE_RATE

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
    def detect_rolling_anomalies(
        equity_curve: pd.DataFrame,
        window: int = 90,
        absolute_threshold: float = 10.0,
        zscore_threshold: float = 3.0,
        risk_free_rate: float = None
    ) -> tuple:
        """
        Detect anomalous spikes in rolling Sharpe/Sortino ratios.

        Uses a hybrid approach:
        - Absolute threshold: |value| > absolute_threshold
        - Z-score threshold: values > zscore_threshold standard deviations from mean

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column
            window: Rolling window in days (default 90)
            absolute_threshold: Values with |value| > threshold are anomalies (default 10)
            zscore_threshold: Z-score threshold for anomaly detection (default 3.0)
            risk_free_rate: Annual risk-free rate (default: DEFAULT_RISK_FREE_RATE)

        Returns:
            Tuple of (list of anomaly dicts, filtered_sharpe_ratio)
            Each anomaly dict has: date, metric, value, reason
        """
        if risk_free_rate is None:
            risk_free_rate = DEFAULT_RISK_FREE_RATE

        detected_anomalies = []

        if len(equity_curve) < window + 10:
            return detected_anomalies, PerformanceMetrics.calculate_sharpe_ratio(equity_curve, risk_free_rate)

        df = equity_curve.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # Calculate returns
        df['returns'] = df['equity'].pct_change()
        returns = df['returns'].dropna()

        if len(returns) == 0:
            return detected_anomalies, 0.0

        daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # Rolling Sharpe - handle division by zero
        rolling_mean = df['returns'].rolling(window).mean()
        rolling_std = df['returns'].rolling(window).std()

        # Avoid division by zero: set to 0 where std is 0 or NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            rolling_sharpe = ((rolling_mean - daily_rf) / rolling_std) * np.sqrt(252)
        # Replace inf, -inf, and NaN with 0
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Detect anomalies in rolling Sharpe
        sharpe_series = rolling_sharpe.dropna()
        if len(sharpe_series) >= 10:
            sharpe_mean = sharpe_series.mean()
            sharpe_std = sharpe_series.std()

            anomaly_dates = set()

            for date, value in sharpe_series.items():
                is_anomaly = False
                reason = []

                # Check absolute threshold
                if abs(value) > absolute_threshold:
                    is_anomaly = True
                    reason.append(f"|value| > {absolute_threshold}")

                # Check z-score threshold
                if sharpe_std > 0:
                    z_score = abs(value - sharpe_mean) / sharpe_std
                    if z_score > zscore_threshold:
                        is_anomaly = True
                        reason.append(f"z-score {z_score:.1f} > {zscore_threshold}")

                if is_anomaly:
                    detected_anomalies.append({
                        'date': date,
                        'metric': 'Sharpe Ratio',
                        'value': value,
                        'reason': ' AND '.join(reason)
                    })
                    anomaly_dates.add(date)

        # Calculate filtered Sharpe ratio
        # Exclude returns from anomalous periods (the window leading up to each anomaly)
        if detected_anomalies:
            # Get all unique dates with anomalies
            anomaly_date_set = {a['date'] for a in detected_anomalies if a['metric'] == 'Sharpe Ratio'}

            # Create a mask for non-anomalous periods
            # We exclude the rolling window period that led to each anomalous reading
            mask = pd.Series(True, index=returns.index)
            for anomaly_date in anomaly_date_set:
                # Find the window start
                try:
                    idx = returns.index.get_loc(anomaly_date)
                    start_idx = max(0, idx - window)
                    mask.iloc[start_idx:idx+1] = False
                except (KeyError, TypeError):
                    continue

            filtered_returns = returns[mask]

            if len(filtered_returns) > 10 and filtered_returns.std() > 0:
                excess_returns = filtered_returns - daily_rf
                filtered_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
            else:
                # Fall back to regular calculation if too few filtered returns
                filtered_sharpe = PerformanceMetrics.calculate_sharpe_ratio(equity_curve, risk_free_rate)
        else:
            filtered_sharpe = PerformanceMetrics.calculate_sharpe_ratio(equity_curve, risk_free_rate)

        return detected_anomalies, filtered_sharpe

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

        # Filter out NaN and invalid values
        if np.any(np.isnan(equity)) or np.any(np.isinf(equity)):
            equity = np.nan_to_num(equity, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure we have valid data
        if len(equity) == 0 or np.all(equity <= 0):
            return 0.0, 0.0

        running_max = np.maximum.accumulate(equity)

        # Prevent division by zero - use safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = running_max - equity
            drawdown_pct = np.where(running_max > 0, (drawdown / running_max) * 100, 0.0)

        # Remove any NaN or inf values that might have slipped through
        drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
        drawdown_pct = np.nan_to_num(drawdown_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Cap drawdown percentage at 100% (can't lose more than 100% in real terms)
        drawdown_pct = np.clip(drawdown_pct, 0, 100)

        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        max_dd_pct = np.max(drawdown_pct) if len(drawdown_pct) > 0 else 0.0

        return max_dd, max_dd_pct

    @staticmethod
    def calculate_volatility(equity_curve: pd.DataFrame) -> float:
        """
        Calculate annualized volatility.

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column

        Returns:
            Annualized volatility as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        return returns.std() * np.sqrt(252) * 100  # Annualized, as percentage

    @staticmethod
    def calculate_downside_deviation(equity_curve: pd.DataFrame) -> float:
        """
        Calculate downside deviation (volatility of negative returns).

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column

        Returns:
            Annualized downside deviation as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return 0.0

        return negative_returns.std() * np.sqrt(252) * 100  # Annualized

    @staticmethod
    def calculate_sortino_ratio(equity_curve: pd.DataFrame,
                               risk_free_rate: float = None) -> float:
        """
        Calculate Sortino ratio using downside deviation of EXCESS returns.

        The Sortino ratio measures risk-adjusted return using only downside volatility.
        Unlike Sharpe which penalizes all volatility, Sortino only penalizes
        returns below the risk-free rate.

        Formula: (Mean Excess Return) / (Downside Deviation of Excess Returns) * sqrt(252)

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column
            risk_free_rate: Annual risk-free rate (default: DEFAULT_RISK_FREE_RATE = 3.5%)

        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = DEFAULT_RISK_FREE_RATE

        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        daily_rf = pow(1 + risk_free_rate, 1/252) - 1
        excess_returns = returns - daily_rf

        # CORRECTED: Downside deviation uses NEGATIVE EXCESS RETURNS, not raw negative returns
        # This measures volatility of returns that fall below the risk-free rate
        downside_excess_returns = excess_returns[excess_returns < 0]

        if len(downside_excess_returns) == 0 or downside_excess_returns.std() == 0:
            # No negative excess returns = excellent performance, return large positive value
            if excess_returns.mean() > 0:
                return 99.99  # Capped to avoid infinity
            return 0.0

        sortino = (excess_returns.mean() / downside_excess_returns.std()) * np.sqrt(252)
        return sortino

    @staticmethod
    def calculate_best_day(equity_curve: pd.DataFrame) -> float:
        """
        Calculate best single day return.

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column

        Returns:
            Best day return as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        return returns.max() * 100 if len(returns) > 0 else 0.0

    @staticmethod
    def calculate_worst_day(equity_curve: pd.DataFrame) -> float:
        """
        Calculate worst single day return.

        Args:
            equity_curve: Equity curve DataFrame with 'equity' column

        Returns:
            Worst day return as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        return returns.min() * 100 if len(returns) > 0 else 0.0

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
