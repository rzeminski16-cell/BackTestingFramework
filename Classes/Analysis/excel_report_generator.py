"""
Excel report generation for backtest results.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import LineChart, BarChart, ScatterChart, Reference, AreaChart
from openpyxl.chart.marker import Marker
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule

from ..Engine.backtest_result import BacktestResult
from ..Models.trade import Trade


class ExcelReportGenerator:
    """
    Generates comprehensive Excel reports for backtests.

    Creates a multi-sheet Excel workbook with:
    - Sheet 1: Summary Dashboard
    - Sheet 2: Trade Log
    - Sheet 3: Performance Metrics & Analysis
    - Sheet 4: Visualizations & Charts
    - Sheet 5: Market Condition Breakdown (optional)
    """

    def __init__(self, output_directory: Path, initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.02, benchmark_name: str = "S&P 500"):
        """
        Initialize Excel report generator.

        Args:
            output_directory: Directory to save reports
            initial_capital: Starting capital for calculations
            risk_free_rate: Annual risk-free rate (default 2%)
            benchmark_name: Name of benchmark for comparison
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.benchmark_name = benchmark_name

        # Styling
        self.header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        self.header_font = Font(bold=True, color="FFFFFF", size=11)
        self.section_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        self.section_font = Font(bold=True, size=12)
        self.metric_font = Font(size=10)
        self.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

    def generate_report(self, result: BacktestResult, filename: Optional[str] = None) -> Path:
        """
        Generate comprehensive Excel report for backtest result.

        Args:
            result: Backtest result
            filename: Optional custom filename

        Returns:
            Path to generated Excel file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.strategy_name}_{result.symbol}_{timestamp}.xlsx"

        filepath = self.output_directory / filename

        # Calculate comprehensive metrics
        metrics = self._calculate_all_metrics(result)

        # Create workbook
        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet

        # Create all sheets
        self._create_summary_dashboard(wb, result, metrics)
        self._create_trade_log(wb, result, metrics)
        self._create_performance_analysis(wb, result, metrics)
        self._create_visualizations(wb, result, metrics)
        self._create_market_conditions(wb, result, metrics)

        # Save workbook
        wb.save(filepath)

        return filepath

    def _calculate_all_metrics(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Calculate all metrics needed for the report.

        Args:
            result: Backtest result

        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        trades = result.trades
        equity_curve = result.equity_curve.copy()

        # Basic information
        metrics['symbol'] = result.symbol
        metrics['strategy_name'] = result.strategy_name
        metrics['initial_capital'] = self.initial_capital
        metrics['final_capital'] = result.final_equity
        metrics['num_trades'] = len(trades)

        # Date range
        if len(equity_curve) > 0:
            metrics['start_date'] = equity_curve['date'].iloc[0]
            metrics['end_date'] = equity_curve['date'].iloc[-1]
            metrics['total_days'] = (metrics['end_date'] - metrics['start_date']).days
            metrics['years'] = metrics['total_days'] / 365.25
        else:
            metrics['start_date'] = None
            metrics['end_date'] = None
            metrics['total_days'] = 0
            metrics['years'] = 0

        if len(trades) == 0:
            return self._get_empty_metrics(metrics)

        # A. Overall Performance Metrics
        metrics['total_return'] = result.total_return
        metrics['total_return_pct'] = result.total_return_pct
        metrics['cagr'] = self._calculate_cagr(self.initial_capital, result.final_equity, metrics['years'])
        metrics['annualized_return'] = metrics['cagr']  # Same as CAGR

        # Win/Loss statistics
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]

        metrics['num_wins'] = len(winning_trades)
        metrics['num_losses'] = len(losing_trades)
        metrics['win_rate'] = metrics['num_wins'] / metrics['num_trades'] if metrics['num_trades'] > 0 else 0
        metrics['loss_rate'] = 1 - metrics['win_rate']

        # B. Risk Metrics
        drawdown_metrics = self._calculate_drawdown_metrics(equity_curve)
        metrics.update(drawdown_metrics)

        metrics['volatility'] = self._calculate_volatility(equity_curve)
        metrics['downside_deviation'] = self._calculate_downside_deviation(equity_curve)
        metrics['best_day'] = self._calculate_best_day(equity_curve)
        metrics['worst_day'] = self._calculate_worst_day(equity_curve)

        # C. Risk-Adjusted Performance Ratios
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(equity_curve)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(equity_curve)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(metrics['cagr'], metrics['max_drawdown_pct'])
        metrics['recovery_factor'] = self._calculate_recovery_factor(metrics['total_return'], metrics['max_drawdown'])
        metrics['profit_factor'] = self._calculate_profit_factor(winning_trades, losing_trades)

        # D. Trade Quality Metrics
        wins = [t.pl for t in winning_trades]
        losses = [t.pl for t in losing_trades]

        metrics['avg_win'] = np.mean(wins) if wins else 0
        metrics['avg_loss'] = np.mean(losses) if losses else 0
        metrics['largest_win'] = max(wins) if wins else 0
        metrics['largest_loss'] = min(losses) if losses else 0
        metrics['risk_reward_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0

        # Win/loss streaks
        streaks = self._calculate_streaks(trades)
        metrics['max_win_streak'] = streaks['max_win_streak']
        metrics['max_loss_streak'] = streaks['max_loss_streak']

        # Trade durations
        durations = [t.duration_days for t in trades]
        metrics['avg_trade_duration'] = np.mean(durations)
        metrics['median_trade_duration'] = np.median(durations)

        # E. Consistency Metrics
        consistency = self._calculate_consistency_metrics(equity_curve, trades)
        metrics.update(consistency)

        # F. Additional metrics
        metrics['strategy_exposure'] = self._calculate_exposure(equity_curve, trades)

        # Monthly/Quarterly returns
        metrics['monthly_returns'] = self._calculate_period_returns(equity_curve, 'M')
        metrics['quarterly_returns'] = self._calculate_period_returns(equity_curve, 'Q')

        # Trade distribution
        metrics['return_distribution'] = self._calculate_return_distribution(trades)

        return metrics

    def _get_empty_metrics(self, base_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Return metrics dictionary with zero values for no-trade scenario."""
        base_metrics.update({
            'total_return': 0, 'total_return_pct': 0, 'cagr': 0, 'annualized_return': 0,
            'num_wins': 0, 'num_losses': 0, 'win_rate': 0, 'loss_rate': 0,
            'max_drawdown': 0, 'max_drawdown_pct': 0, 'avg_drawdown': 0,
            'max_drawdown_duration': 0, 'volatility': 0, 'downside_deviation': 0,
            'best_day': 0, 'worst_day': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
            'calmar_ratio': 0, 'recovery_factor': 0, 'profit_factor': 0,
            'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0,
            'risk_reward_ratio': 0, 'max_win_streak': 0, 'max_loss_streak': 0,
            'avg_trade_duration': 0, 'median_trade_duration': 0,
            'strategy_exposure': 0, 'monthly_returns': pd.DataFrame(),
            'quarterly_returns': pd.DataFrame(), 'return_distribution': {}
        })
        return base_metrics

    # ==================== METRIC CALCULATIONS ====================

    def _calculate_cagr(self, initial: float, final: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate."""
        if years <= 0 or initial <= 0:
            return 0.0
        return (pow(final / initial, 1 / years) - 1) * 100

    def _calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        if len(equity_curve) == 0:
            return {
                'max_drawdown': 0, 'max_drawdown_pct': 0, 'avg_drawdown': 0,
                'max_drawdown_duration': 0, 'num_drawdowns_over_5pct': 0
            }

        equity = equity_curve['equity'].values
        dates = equity_curve['date'].values

        # Running maximum
        running_max = np.maximum.accumulate(equity)

        # Drawdown in dollars and percent
        drawdown = running_max - equity
        drawdown_pct = (drawdown / running_max) * 100

        max_dd = np.max(drawdown)
        max_dd_pct = np.max(drawdown_pct)

        # Average drawdown (only count periods in drawdown)
        in_drawdown = drawdown > 0
        avg_dd = np.mean(drawdown[in_drawdown]) if np.any(in_drawdown) else 0

        # Max drawdown duration
        max_dd_duration = self._calculate_max_drawdown_duration(equity, dates)

        # Count drawdowns over 5%
        num_dd_over_5 = np.sum(drawdown_pct > 5)

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'avg_drawdown': avg_dd,
            'max_drawdown_duration': max_dd_duration,
            'num_drawdowns_over_5pct': num_dd_over_5
        }

    def _calculate_max_drawdown_duration(self, equity: np.ndarray, dates: np.ndarray) -> int:
        """Calculate maximum drawdown duration in days."""
        running_max = np.maximum.accumulate(equity)
        in_drawdown = equity < running_max

        if not np.any(in_drawdown):
            return 0

        # Find drawdown periods
        dd_start = None
        max_duration = 0

        for i in range(len(equity)):
            if in_drawdown[i] and dd_start is None:
                dd_start = dates[i]
            elif not in_drawdown[i] and dd_start is not None:
                duration = (dates[i] - dd_start).days if hasattr(dates[i], 'days') else 0
                max_duration = max(max_duration, duration)
                dd_start = None

        # Handle if still in drawdown at end
        if dd_start is not None:
            duration = (dates[-1] - dd_start).days if hasattr(dates[-1], 'days') else 0
            max_duration = max(max_duration, duration)

        return max_duration

    def _calculate_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        return returns.std() * np.sqrt(252) * 100  # Annualized, as percentage

    def _calculate_downside_deviation(self, equity_curve: pd.DataFrame) -> float:
        """Calculate downside deviation (volatility of negative returns)."""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return 0.0

        return negative_returns.std() * np.sqrt(252) * 100  # Annualized

    def _calculate_best_day(self, equity_curve: pd.DataFrame) -> float:
        """Calculate best single day return."""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        return returns.max() * 100 if len(returns) > 0 else 0.0

    def _calculate_worst_day(self, equity_curve: pd.DataFrame) -> float:
        """Calculate worst single day return."""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()
        return returns.min() * 100 if len(returns) > 0 else 0.0

    def _calculate_sharpe_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        daily_rf = pow(1 + self.risk_free_rate, 1/252) - 1
        excess_returns = returns - daily_rf

        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe

    def _calculate_sortino_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['equity'].pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        daily_rf = pow(1 + self.risk_free_rate, 1/252) - 1
        excess_returns = returns - daily_rf

        # Downside returns
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        return sortino

    def _calculate_calmar_ratio(self, cagr: float, max_dd_pct: float) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        if max_dd_pct == 0:
            return 0.0
        return cagr / max_dd_pct

    def _calculate_recovery_factor(self, total_return: float, max_dd: float) -> float:
        """Calculate recovery factor (Total Profit / Max Drawdown)."""
        if max_dd == 0:
            return 0.0
        return total_return / max_dd

    def _calculate_profit_factor(self, winning_trades: List[Trade], losing_trades: List[Trade]) -> float:
        """Calculate profit factor (Gross Profit / Gross Loss)."""
        gross_profit = sum(t.pl for t in winning_trades)
        gross_loss = abs(sum(t.pl for t in losing_trades))

        if gross_loss == 0:
            return 0.0

        return gross_profit / gross_loss

    def _calculate_streaks(self, trades: List[Trade]) -> Dict[str, int]:
        """Calculate win and loss streaks."""
        if len(trades) == 0:
            return {'max_win_streak': 0, 'max_loss_streak': 0}

        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for trade in trades:
            if trade.is_winner:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }

    def _calculate_consistency_metrics(self, equity_curve: pd.DataFrame, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate consistency metrics."""
        metrics = {}

        if len(equity_curve) < 2:
            return {
                'profitable_months_pct': 0, 'profitable_weeks_pct': 0,
                'monthly_consistency': 0, 'consecutive_losing_months': 0
            }

        # Calculate monthly returns
        equity_df = equity_curve.copy()
        equity_df.set_index('date', inplace=True)

        # Resample to month-end
        monthly = equity_df['equity'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()

        # Profitable months
        profitable_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        metrics['profitable_months_pct'] = (profitable_months / total_months * 100) if total_months > 0 else 0

        # Weekly returns
        weekly = equity_df['equity'].resample('W').last()
        weekly_returns = weekly.pct_change().dropna()
        profitable_weeks = (weekly_returns > 0).sum()
        total_weeks = len(weekly_returns)
        metrics['profitable_weeks_pct'] = (profitable_weeks / total_weeks * 100) if total_weeks > 0 else 0

        # Monthly consistency (std dev of monthly returns)
        metrics['monthly_consistency'] = monthly_returns.std() * 100 if len(monthly_returns) > 0 else 0

        # Consecutive losing months
        max_consecutive_losing = 0
        current_consecutive_losing = 0

        for ret in monthly_returns:
            if ret < 0:
                current_consecutive_losing += 1
                max_consecutive_losing = max(max_consecutive_losing, current_consecutive_losing)
            else:
                current_consecutive_losing = 0

        metrics['consecutive_losing_months'] = max_consecutive_losing

        return metrics

    def _calculate_exposure(self, equity_curve: pd.DataFrame, trades: List[Trade]) -> float:
        """Calculate percentage of time in market."""
        if len(equity_curve) == 0:
            return 0.0

        # Count days with open positions
        equity_df = equity_curve.copy()
        days_in_market = (equity_df['position_value'] > 0).sum()
        total_days = len(equity_df)

        return (days_in_market / total_days * 100) if total_days > 0 else 0.0

    def _calculate_period_returns(self, equity_curve: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        Calculate returns by period (monthly 'M' or quarterly 'Q').

        Returns DataFrame with columns: period, return_pct, cumulative_return
        """
        if len(equity_curve) < 2:
            return pd.DataFrame(columns=['period', 'return_pct', 'cumulative_return'])

        equity_df = equity_curve.copy()
        equity_df.set_index('date', inplace=True)

        # Resample to period end
        period_equity = equity_df['equity'].resample(period).last()
        period_returns = period_equity.pct_change().dropna()

        # Create result DataFrame
        result = pd.DataFrame({
            'period': period_returns.index,
            'return_pct': period_returns.values * 100,
            'cumulative_return': (1 + period_returns).cumprod().values * 100 - 100
        })

        return result

    def _calculate_return_distribution(self, trades: List[Trade]) -> Dict[str, int]:
        """
        Calculate distribution of returns in bins.

        Returns dictionary with bins as keys and counts as values.
        """
        if len(trades) == 0:
            return {}

        returns = [t.pl_pct for t in trades]

        # Define bins
        bins = [-100, -20, -10, -5, 0, 5, 10, 20, 100]
        labels = ['< -20%', '-20% to -10%', '-10% to -5%', '-5% to 0%',
                  '0% to 5%', '5% to 10%', '10% to 20%', '> 20%']

        # Count returns in each bin
        hist, _ = np.histogram(returns, bins=bins)

        distribution = dict(zip(labels, hist))

        return distribution

    # ==================== SHEET CREATION ====================

    def _create_summary_dashboard(self, wb: Workbook, result: BacktestResult, metrics: Dict[str, Any]):
        """Create Sheet 1: Summary Dashboard."""
        ws = wb.create_sheet("Summary Dashboard")

        row = 1

        # Title
        ws.merge_cells(f'A{row}:D{row}')
        title_cell = ws[f'A{row}']
        title_cell.value = f"BACKTEST SUMMARY DASHBOARD"
        title_cell.font = Font(bold=True, size=16)
        title_cell.alignment = Alignment(horizontal='center')
        row += 1

        # Subtitle
        ws.merge_cells(f'A{row}:D{row}')
        subtitle = ws[f'A{row}']
        subtitle.value = f"{metrics['strategy_name']} - {metrics['symbol']}"
        subtitle.font = Font(size=12)
        subtitle.alignment = Alignment(horizontal='center')
        row += 2

        # Date range and basic info
        ws[f'A{row}'] = "Backtest Period:"
        ws[f'B{row}'] = f"{metrics['start_date'].strftime('%Y-%m-%d') if metrics['start_date'] else 'N/A'} to {metrics['end_date'].strftime('%Y-%m-%d') if metrics['end_date'] else 'N/A'}"
        row += 1
        ws[f'A{row}'] = "Total Days:"
        ws[f'B{row}'] = metrics['total_days']
        row += 2

        # Strategy Parameters Section
        if result.strategy_params:
            row = self._add_section_header(ws, row, "STRATEGY PARAMETERS")

            params_data = []
            for param_name, param_value in result.strategy_params.items():
                # Format parameter name (convert snake_case to Title Case)
                display_name = param_name.replace('_', ' ').title()
                # Keep the value as-is (will be formatted by _add_metrics_table)
                params_data.append((display_name, param_value, "auto"))

            row = self._add_metrics_table(ws, row, params_data)
            row += 2

        # A. Overall Performance Metrics
        row = self._add_section_header(ws, row, "A. OVERALL PERFORMANCE METRICS")

        performance_data = [
            ("Total Return ($)", metrics['total_return'], "currency"),
            ("Total Return (%)", metrics['total_return_pct'], "percentage"),
            ("CAGR (%)", metrics['cagr'], "percentage"),
            ("Annualized Return (%)", metrics['annualized_return'], "percentage"),
            ("Initial Capital", metrics['initial_capital'], "currency"),
            ("Final Capital", metrics['final_capital'], "currency"),
            ("Number of Trades", metrics['num_trades'], "number"),
            ("Win Rate", metrics['win_rate'] * 100, "percentage"),
            ("Loss Rate", metrics['loss_rate'] * 100, "percentage"),
        ]

        row = self._add_metrics_table(ws, row, performance_data)
        row += 1

        # B. Risk Metrics
        row = self._add_section_header(ws, row, "B. RISK METRICS")

        risk_data = [
            ("Maximum Drawdown ($)", metrics['max_drawdown'], "currency"),
            ("Maximum Drawdown (%)", metrics['max_drawdown_pct'], "percentage"),
            ("Average Drawdown ($)", metrics['avg_drawdown'], "currency"),
            ("Drawdown Duration (days)", metrics['max_drawdown_duration'], "number"),
            ("Drawdowns > 5%", metrics['num_drawdowns_over_5pct'], "number"),
            ("Volatility (annualized %)", metrics['volatility'], "percentage"),
            ("Downside Deviation (%)", metrics['downside_deviation'], "percentage"),
            ("Best Day (%)", metrics['best_day'], "percentage"),
            ("Worst Day (%)", metrics['worst_day'], "percentage"),
        ]

        row = self._add_metrics_table(ws, row, risk_data)
        row += 1

        # C. Risk-Adjusted Performance Ratios
        row = self._add_section_header(ws, row, "C. RISK-ADJUSTED PERFORMANCE RATIOS")

        ratios_data = [
            ("Sharpe Ratio", metrics['sharpe_ratio'], "decimal", 1.0),
            ("Sortino Ratio", metrics['sortino_ratio'], "decimal", 2.0),
            ("Calmar Ratio", metrics['calmar_ratio'], "decimal", 2.0),
            ("Recovery Factor", metrics['recovery_factor'], "decimal", None),
            ("Profit Factor", metrics['profit_factor'], "decimal", 1.5),
        ]

        row = self._add_metrics_with_targets(ws, row, ratios_data)
        row += 1

        # D. Trade Quality Metrics
        row = self._add_section_header(ws, row, "D. TRADE QUALITY METRICS")

        trade_quality_data = [
            ("Average Win ($)", metrics['avg_win'], "currency"),
            ("Average Loss ($)", metrics['avg_loss'], "currency"),
            ("Risk/Reward Ratio", metrics['risk_reward_ratio'], "decimal"),
            ("Largest Win ($)", metrics['largest_win'], "currency"),
            ("Largest Loss ($)", metrics['largest_loss'], "currency"),
            ("Max Win Streak", metrics['max_win_streak'], "number"),
            ("Max Loss Streak", metrics['max_loss_streak'], "number"),
            ("Average Trade Duration (days)", metrics['avg_trade_duration'], "decimal"),
            ("Median Trade Duration (days)", metrics['median_trade_duration'], "number"),
        ]

        row = self._add_metrics_table(ws, row, trade_quality_data)
        row += 1

        # E. Consistency Metrics
        row = self._add_section_header(ws, row, "E. CONSISTENCY METRICS")

        consistency_data = [
            ("Profitable Months (%)", metrics['profitable_months_pct'], "percentage"),
            ("Profitable Weeks (%)", metrics['profitable_weeks_pct'], "percentage"),
            ("Monthly Consistency (std dev %)", metrics['monthly_consistency'], "percentage"),
            ("Consecutive Losing Months (max)", metrics['consecutive_losing_months'], "number"),
            ("Strategy Exposure (%)", metrics['strategy_exposure'], "percentage"),
        ]

        row = self._add_metrics_table(ws, row, consistency_data)

        # Format columns
        ws.column_dimensions['A'].width = 35
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 15
        ws.column_dimensions['D'].width = 15

    def _create_trade_log(self, wb: Workbook, result: BacktestResult, metrics: Dict[str, Any]):
        """Create Sheet 2: Trade Log."""
        ws = wb.create_sheet("Trade Log")

        # Create DataFrame from trades
        if len(result.trades) == 0:
            ws['A1'] = "No trades executed"
            return

        trade_data = []
        cumulative_pl = 0

        for idx, trade in enumerate(result.trades, 1):
            cumulative_pl += trade.pl

            trade_data.append({
                'Trade #': idx,
                'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                'Entry Time': trade.entry_date.strftime('%H:%M:%S') if hasattr(trade.entry_date, 'hour') else 'N/A',
                'Entry Price': trade.entry_price,
                'Entry Reason': trade.entry_reason,
                'Exit Date': trade.exit_date.strftime('%Y-%m-%d'),
                'Exit Price': trade.exit_price,
                'Exit Reason': trade.exit_reason,
                'Trade Type': trade.side,
                'Position Size': trade.quantity,
                'Gross P/L': trade.pl + trade.commission_paid,  # P/L before fees
                'Fees/Commission': trade.commission_paid,
                'Net P/L': trade.pl,
                'Return %': trade.pl_pct,
                'Trade Duration': trade.duration_days,
                'Cumulative P/L': cumulative_pl,
                'Initial Stop Loss': trade.initial_stop_loss if trade.initial_stop_loss else 'N/A',
                'Final Stop Loss': trade.final_stop_loss if trade.final_stop_loss else 'N/A',
                'Take Profit': trade.take_profit if trade.take_profit else 'N/A',
                'Partial Exits': trade.partial_exits,
            })

        df = pd.DataFrame(trade_data)

        # Write headers
        headers = list(df.columns)
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            cell.font = self.header_font
            cell.fill = self.header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = self.border

        # Write data
        for row_idx, row_data in enumerate(df.values, 2):
            for col_idx, value in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.border = self.border

                # Format currency columns
                if headers[col_idx-1] in ['Entry Price', 'Exit Price', 'Gross P/L', 'Fees/Commission', 'Net P/L', 'Cumulative P/L']:
                    cell.number_format = '$#,##0.00'

                    # Color code P/L
                    if headers[col_idx-1] in ['Net P/L', 'Gross P/L', 'Cumulative P/L']:
                        if isinstance(value, (int, float)) and value < 0:
                            cell.font = Font(color="FF0000")  # Red
                        elif isinstance(value, (int, float)) and value > 0:
                            cell.font = Font(color="00B050")  # Green

                # Format percentage
                elif headers[col_idx-1] == 'Return %':
                    cell.number_format = '0.00"%"'
                    if isinstance(value, (int, float)) and value < 0:
                        cell.font = Font(color="FF0000")
                    elif isinstance(value, (int, float)) and value > 0:
                        cell.font = Font(color="00B050")

        # Auto-fit columns
        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column].width = adjusted_width

        # Freeze header row
        ws.freeze_panes = 'A2'

    def _create_performance_analysis(self, wb: Workbook, result: BacktestResult, metrics: Dict[str, Any]):
        """Create Sheet 3: Performance Metrics & Analysis."""
        ws = wb.create_sheet("Performance Analysis")

        row = 1

        # Title
        ws.merge_cells(f'A{row}:D{row}')
        ws[f'A{row}'] = "PERFORMANCE METRICS & ANALYSIS"
        ws[f'A{row}'].font = Font(bold=True, size=14)
        ws[f'A{row}'].alignment = Alignment(horizontal='center')
        row += 2

        # A. Drawdown Analysis Table
        row = self._add_section_header(ws, row, "A. DRAWDOWN ANALYSIS")

        drawdown_data = [
            ("Max Drawdown ($)", metrics['max_drawdown'], "currency"),
            ("Max Drawdown (%)", metrics['max_drawdown_pct'], "percentage"),
            ("Average Drawdown ($)", metrics['avg_drawdown'], "currency"),
            ("Max Drawdown Duration (days)", metrics['max_drawdown_duration'], "number"),
            ("Number of Drawdowns > 5%", metrics['num_drawdowns_over_5pct'], "number"),
        ]

        row = self._add_metrics_table(ws, row, drawdown_data)
        row += 2

        # B. Monthly Returns Table
        row = self._add_section_header(ws, row, "B. MONTHLY RETURNS")

        monthly_df = metrics['monthly_returns']

        if len(monthly_df) > 0:
            # Add month/year columns
            monthly_df_copy = monthly_df.copy()
            monthly_df_copy['Year'] = monthly_df_copy['period'].dt.year
            monthly_df_copy['Month'] = monthly_df_copy['period'].dt.strftime('%b')

            # Pivot table: rows=months, columns=years
            pivot_data = monthly_df_copy.pivot_table(
                index='Month',
                columns='Year',
                values='return_pct',
                aggfunc='first'
            )

            # Write pivot table
            pivot_row = row
            ws.cell(row=pivot_row, column=1, value="Month")

            # Year headers
            for col_idx, year in enumerate(pivot_data.columns, 2):
                cell = ws.cell(row=pivot_row, column=col_idx, value=year)
                cell.font = self.header_font
                cell.fill = self.header_fill

            # Month rows
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            for month in month_order:
                if month in pivot_data.index:
                    pivot_row += 1
                    ws.cell(row=pivot_row, column=1, value=month).font = Font(bold=True)

                    for col_idx, year in enumerate(pivot_data.columns, 2):
                        value = pivot_data.loc[month, year] if not pd.isna(pivot_data.loc[month, year]) else None
                        if value is not None:
                            cell = ws.cell(row=pivot_row, column=col_idx, value=value)
                            cell.number_format = '0.00"%"'

                            # Color coding
                            if value > 0:
                                cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                                cell.font = Font(color="006100")
                            elif value < 0:
                                cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                                cell.font = Font(color="9C0006")

            row = pivot_row + 2
        else:
            ws[f'A{row}'] = "Insufficient data for monthly analysis"
            row += 2

        # C. Distribution Analysis
        row = self._add_section_header(ws, row, "C. TRADE RETURN DISTRIBUTION")

        distribution = metrics['return_distribution']

        if distribution:
            ws.cell(row=row, column=1, value="Return Range").font = self.header_font
            ws.cell(row=row, column=2, value="Number of Trades").font = self.header_font
            ws.cell(row=row, column=3, value="Percentage").font = self.header_font
            row += 1

            total_trades = sum(distribution.values())

            for bin_label, count in distribution.items():
                ws.cell(row=row, column=1, value=bin_label)
                ws.cell(row=row, column=2, value=count)
                pct = (count / total_trades * 100) if total_trades > 0 else 0
                cell = ws.cell(row=row, column=3, value=pct)
                cell.number_format = '0.00"%"'
                row += 1

        row += 1

        # Format columns
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 20
        ws.column_dimensions['D'].width = 20

    def _create_visualizations(self, wb: Workbook, result: BacktestResult, metrics: Dict[str, Any]):
        """Create Sheet 4: Visualizations & Charts."""
        ws = wb.create_sheet("Visualizations")

        # Title
        ws['A1'] = "CHARTS & VISUALIZATIONS"
        ws['A1'].font = Font(bold=True, size=14)
        ws['A1'].alignment = Alignment(horizontal='center')

        # Prepare equity curve data
        equity_df = result.equity_curve.copy()

        if len(equity_df) == 0:
            ws['A3'] = "No data available for charts"
            return

        # Write equity curve data starting at row 4
        data_start_row = 4
        ws.cell(row=data_start_row, column=1, value="Date").font = self.header_font
        ws.cell(row=data_start_row, column=2, value="Equity").font = self.header_font
        ws.cell(row=data_start_row, column=3, value="Drawdown %").font = self.header_font

        # Calculate drawdown for chart
        equity_values = equity_df['equity'].values
        running_max = np.maximum.accumulate(equity_values)
        drawdown_pct = ((equity_values - running_max) / running_max * 100)

        for idx, (date, equity, dd) in enumerate(zip(equity_df['date'], equity_values, drawdown_pct)):
            row_num = data_start_row + 1 + idx
            ws.cell(row=row_num, column=1, value=date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date))
            ws.cell(row=row_num, column=2, value=equity)
            ws.cell(row=row_num, column=3, value=dd)

        # Create Equity Curve Chart
        chart1 = LineChart()
        chart1.title = "Equity Curve"
        chart1.style = 13
        chart1.y_axis.title = 'Portfolio Value ($)'
        chart1.x_axis.title = 'Date'

        data_ref = Reference(ws, min_col=2, min_row=data_start_row, max_row=data_start_row + len(equity_df))
        cats_ref = Reference(ws, min_col=1, min_row=data_start_row+1, max_row=data_start_row + len(equity_df))

        chart1.add_data(data_ref, titles_from_data=True)
        chart1.set_categories(cats_ref)
        chart1.height = 10
        chart1.width = 20

        ws.add_chart(chart1, "E4")

        # Create Drawdown Chart
        chart2 = AreaChart()
        chart2.title = "Drawdown %"
        chart2.style = 13
        chart2.y_axis.title = 'Drawdown %'
        chart2.x_axis.title = 'Date'

        dd_ref = Reference(ws, min_col=3, min_row=data_start_row, max_row=data_start_row + len(equity_df))

        chart2.add_data(dd_ref, titles_from_data=True)
        chart2.set_categories(cats_ref)
        chart2.height = 10
        chart2.width = 20

        ws.add_chart(chart2, "E25")

        # Create return distribution chart
        distribution = metrics['return_distribution']

        if distribution:
            # Write distribution data
            dist_start_row = data_start_row + len(equity_df) + 5
            ws.cell(row=dist_start_row, column=1, value="Return Range").font = self.header_font
            ws.cell(row=dist_start_row, column=2, value="Count").font = self.header_font

            for idx, (bin_label, count) in enumerate(distribution.items(), 1):
                ws.cell(row=dist_start_row + idx, column=1, value=bin_label)
                ws.cell(row=dist_start_row + idx, column=2, value=count)

            # Create bar chart
            chart3 = BarChart()
            chart3.title = "Trade Return Distribution"
            chart3.y_axis.title = 'Number of Trades'
            chart3.x_axis.title = 'Return Range'

            data_ref = Reference(ws, min_col=2, min_row=dist_start_row, max_row=dist_start_row + len(distribution))
            cats_ref = Reference(ws, min_col=1, min_row=dist_start_row+1, max_row=dist_start_row + len(distribution))

            chart3.add_data(data_ref, titles_from_data=True)
            chart3.set_categories(cats_ref)
            chart3.height = 10
            chart3.width = 15

            ws.add_chart(chart3, "E46")

    def _create_market_conditions(self, wb: Workbook, result: BacktestResult, metrics: Dict[str, Any]):
        """Create Sheet 5: Market Condition Breakdown (Optional)."""
        ws = wb.create_sheet("Market Conditions")

        # Title
        ws['A1'] = "MARKET CONDITION BREAKDOWN"
        ws['A1'].font = Font(bold=True, size=14)

        # Note: This would require market regime classification
        # For now, provide a template/placeholder

        ws['A3'] = "Market Condition"
        ws['B3'] = "# Trades"
        ws['C3'] = "Win Rate %"
        ws['D3'] = "Avg Return %"
        ws['E3'] = "Sharpe Ratio"
        ws['F3'] = "Notes"

        # Apply header formatting
        for col in range(1, 7):
            cell = ws.cell(row=3, column=col)
            cell.font = self.header_font
            cell.fill = self.header_fill

        # Placeholder rows
        conditions = ["Bull Market", "Bear Market", "High Volatility", "Low Volatility", "Ranging Market"]

        for idx, condition in enumerate(conditions, 4):
            ws.cell(row=idx, column=1, value=condition)
            ws.cell(row=idx, column=2, value="N/A")
            ws.cell(row=idx, column=3, value="N/A")
            ws.cell(row=idx, column=4, value="N/A")
            ws.cell(row=idx, column=5, value="N/A")
            ws.cell(row=idx, column=6, value="Requires market regime data")

        ws['A10'] = "Note: Market condition analysis requires additional market regime classification."
        ws['A10'].font = Font(italic=True)

        # Format columns
        for col in ['A', 'B', 'C', 'D', 'E', 'F']:
            ws.column_dimensions[col].width = 20

    # ==================== HELPER METHODS ====================

    def _add_section_header(self, ws, row: int, title: str) -> int:
        """Add a section header and return next row."""
        ws.merge_cells(f'A{row}:D{row}')
        cell = ws[f'A{row}']
        cell.value = title
        cell.font = self.section_font
        cell.fill = self.section_fill
        cell.alignment = Alignment(horizontal='left')
        return row + 1

    def _add_metrics_table(self, ws, row: int, data: List[tuple]) -> int:
        """
        Add a metrics table.

        Args:
            ws: Worksheet
            row: Starting row
            data: List of tuples (label, value, format_type)
                  format_type: 'currency', 'percentage', 'number', 'decimal', 'auto'

        Returns:
            Next available row
        """
        for label, value, format_type in data:
            ws[f'A{row}'] = label
            ws[f'A{row}'].font = self.metric_font

            value_cell = ws[f'B{row}']
            value_cell.value = value

            if format_type == 'currency':
                value_cell.number_format = '$#,##0.00'
            elif format_type == 'percentage':
                value_cell.number_format = '0.00"%"'
            elif format_type == 'decimal':
                value_cell.number_format = '0.00'
            elif format_type == 'number':
                value_cell.number_format = '#,##0'
            elif format_type == 'auto':
                # Auto-detect format based on value type
                if isinstance(value, float):
                    value_cell.number_format = '0.00'
                elif isinstance(value, int):
                    value_cell.number_format = '#,##0'
                # else: keep default (General) format for strings

            row += 1

        return row

    def _add_metrics_with_targets(self, ws, row: int, data: List[tuple]) -> int:
        """
        Add metrics with target values for comparison.

        Args:
            data: List of tuples (label, value, format_type, target)
        """
        # Headers
        ws[f'A{row}'] = "Metric"
        ws[f'B{row}'] = "Value"
        ws[f'C{row}'] = "Target"
        ws[f'D{row}'] = "Status"

        for col in ['A', 'B', 'C', 'D']:
            ws[f'{col}{row}'].font = Font(bold=True)
            ws[f'{col}{row}'].fill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")

        row += 1

        for label, value, format_type, target in data:
            ws[f'A{row}'] = label
            ws[f'A{row}'].font = self.metric_font

            # Value
            value_cell = ws[f'B{row}']
            value_cell.value = value

            if format_type == 'decimal':
                value_cell.number_format = '0.00'

            # Target
            if target is not None:
                ws[f'C{row}'] = f"> {target}"

                # Status
                status_cell = ws[f'D{row}']
                if value >= target:
                    status_cell.value = "✓ Pass"
                    status_cell.font = Font(color="00B050", bold=True)
                else:
                    status_cell.value = "✗ Below"
                    status_cell.font = Font(color="FF0000", bold=True)
            else:
                ws[f'C{row}'] = "N/A"
                ws[f'D{row}'] = "-"

            row += 1

        return row
