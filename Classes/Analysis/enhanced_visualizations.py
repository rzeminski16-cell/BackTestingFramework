"""
Enhanced Visualization Module for Portfolio Reports.

Provides sophisticated matplotlib-based charts that can be embedded in Excel reports:
- Equity curve with drawdown overlay (dual-axis)
- Monthly/yearly returns heatmaps
- Trade distribution histograms
- Rolling performance metrics
- Underwater charts
- MAE/MFE analysis
- Contribution analysis
- 3D parameter surface plots
- Correlation heatmaps
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from io import BytesIO

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.figure import Figure
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Check for seaborn availability (optional, for enhanced styling)
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class EnhancedVisualizations:
    """
    Creates sophisticated matplotlib visualizations for financial reports.

    All methods return BytesIO objects containing PNG images that can be
    embedded directly into Excel reports using openpyxl.
    """

    # Color scheme for consistent styling
    COLORS = {
        'primary': '#1f4e79',      # Dark blue
        'secondary': '#2e75b6',    # Medium blue
        'positive': '#28a745',     # Green
        'negative': '#dc3545',     # Red
        'neutral': '#6c757d',      # Gray
        'warning': '#ffc107',      # Yellow/Amber
        'equity': '#2e75b6',       # Blue for equity curves
        'drawdown': '#dc3545',     # Red for drawdowns
        'benchmark': '#6c757d',    # Gray for benchmarks
        'grid': '#e0e0e0',         # Light gray for grid
    }

    # Custom colormap for heatmaps (red-white-green)
    HEATMAP_COLORS = ['#dc3545', '#ffffff', '#28a745']

    def __init__(self, style: str = 'default', dpi: int = 150):
        """
        Initialize visualization generator.

        Args:
            style: Plot style ('default', 'dark', 'minimal')
            dpi: Resolution for generated images
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for enhanced visualizations. Install with: pip install matplotlib")

        self.dpi = dpi
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style settings."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': '#fafafa',
            'figure.facecolor': 'white',
            'axes.edgecolor': '#cccccc',
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

    def _get_figure_bytes(self, fig: Figure) -> BytesIO:
        """Convert matplotlib figure to BytesIO for embedding in Excel."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf

    def create_equity_curve_with_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Portfolio Equity Curve with Drawdown",
        figsize: Tuple[int, int] = (14, 8)
    ) -> BytesIO:
        """
        Create equity curve with drawdown overlay (dual-axis chart).

        Args:
            equity_curve: DataFrame with 'date' and 'equity' columns
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        dates = pd.to_datetime(equity_curve['date'])
        equity = equity_curve['equity'].values

        # Calculate drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown_pct = ((equity - running_max) / running_max) * 100

        # Plot equity curve
        ax1.plot(dates, equity, color=self.COLORS['equity'], linewidth=1.5, label='Portfolio Equity')
        ax1.fill_between(dates, equity.min() * 0.95, equity, alpha=0.1, color=self.COLORS['equity'])

        # Add running max line
        ax1.plot(dates, running_max, color=self.COLORS['neutral'], linewidth=0.8,
                 linestyle='--', alpha=0.7, label='High Water Mark')

        ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot drawdown (underwater chart)
        ax2.fill_between(dates, drawdown_pct, 0,
                         where=drawdown_pct < 0,
                         color=self.COLORS['drawdown'], alpha=0.6)
        ax2.plot(dates, drawdown_pct, color=self.COLORS['drawdown'], linewidth=0.8)

        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylim(min(drawdown_pct) * 1.1, 5)
        ax2.grid(True, alpha=0.3)

        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 200)))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_monthly_returns_heatmap(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Monthly Returns Heatmap (%)",
        figsize: Tuple[int, int] = (12, 8)
    ) -> BytesIO:
        """
        Create monthly returns heatmap (year vs month grid).

        Args:
            equity_curve: DataFrame with 'date' and 'equity' columns
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        # Calculate monthly returns
        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Resample to monthly and calculate returns
        monthly = df['equity'].resample('ME').last()
        monthly_returns = monthly.pct_change() * 100
        monthly_returns = monthly_returns.dropna()

        if len(monthly_returns) < 2:
            # Not enough data
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Insufficient data for monthly heatmap',
                    ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        # Create pivot table
        monthly_returns_df = monthly_returns.reset_index()
        monthly_returns_df.columns = ['date', 'return']
        monthly_returns_df['year'] = monthly_returns_df['date'].dt.year
        monthly_returns_df['month'] = monthly_returns_df['date'].dt.month

        pivot = monthly_returns_df.pivot_table(
            values='return',
            index='year',
            columns='month',
            aggfunc='first'
        )

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[m-1] for m in pivot.columns]

        # Add yearly total column
        pivot['Year Total'] = pivot.sum(axis=1)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('rg', self.HEATMAP_COLORS, N=256)

        # Determine color scale limits (symmetric around 0)
        max_abs = max(abs(pivot.values.min()), abs(pivot.values.max()))
        max_abs = min(max_abs, 20)  # Cap at 20% for better visualization

        # Create heatmap
        if SEABORN_AVAILABLE:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap,
                        center=0, vmin=-max_abs, vmax=max_abs,
                        ax=ax, cbar_kws={'label': 'Return (%)', 'shrink': 0.8},
                        linewidths=0.5, linecolor='white')
        else:
            # Fallback without seaborn
            im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                           vmin=-max_abs, vmax=max_abs)

            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.iloc[i, j]
                    if not pd.isna(val):
                        text_color = 'white' if abs(val) > max_abs * 0.5 else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                color=text_color, fontsize=8)

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)

            plt.colorbar(im, ax=ax, label='Return (%)', shrink=0.8)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Year', fontsize=11)

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_trade_distribution_histogram(
        self,
        trades: List[Any],
        title: str = "Trade Return Distribution",
        figsize: Tuple[int, int] = (10, 6)
    ) -> BytesIO:
        """
        Create histogram of trade returns with statistics.

        Args:
            trades: List of Trade objects with pl_pct attribute
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        if not trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        returns = [t.pl_pct for t in trades]

        fig, ax = plt.subplots(figsize=figsize)

        # Create bins
        n_bins = min(50, max(10, len(returns) // 5))

        # Create histogram with colors based on positive/negative
        n, bins, patches = ax.hist(returns, bins=n_bins, edgecolor='white', linewidth=0.5)

        # Color bars based on value
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center >= 0:
                patch.set_facecolor(self.COLORS['positive'])
            else:
                patch.set_facecolor(self.COLORS['negative'])

        # Add statistics
        mean_ret = np.mean(returns)
        median_ret = np.median(returns)
        std_ret = np.std(returns)

        # Add vertical lines for mean and median
        ax.axvline(mean_ret, color=self.COLORS['primary'], linestyle='--',
                   linewidth=2, label=f'Mean: {mean_ret:.2f}%')
        ax.axvline(median_ret, color=self.COLORS['secondary'], linestyle='-',
                   linewidth=2, label=f'Median: {median_ret:.2f}%')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Add statistics box
        stats_text = (f'n = {len(returns)}\n'
                      f'Mean: {mean_ret:.2f}%\n'
                      f'Median: {median_ret:.2f}%\n'
                      f'Std Dev: {std_ret:.2f}%\n'
                      f'Win Rate: {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%')

        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Return (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_rolling_metrics_chart(
        self,
        equity_curve: pd.DataFrame,
        window: int = 90,  # ~3.5 months (increased for stability)
        title: str = "Rolling Performance Metrics",
        figsize: Tuple[int, int] = (14, 10),
        filter_anomalies: bool = True,
        anomaly_absolute_threshold: float = 10.0,
        anomaly_zscore_threshold: float = 3.0
    ) -> Tuple[BytesIO, List[Dict]]:
        """
        Create rolling Sharpe, Sortino, and volatility chart with anomaly filtering.

        Anomalies are detected using a hybrid approach:
        - Absolute threshold: |value| > anomaly_absolute_threshold (default 10)
        - Z-score threshold: values > anomaly_zscore_threshold standard deviations from rolling mean

        Args:
            equity_curve: DataFrame with 'date' and 'equity' columns
            window: Rolling window in days (default 90 for stability)
            title: Chart title
            figsize: Figure size
            filter_anomalies: Whether to filter out anomalous spikes
            anomaly_absolute_threshold: Values with |value| > threshold are anomalies (default 10)
            anomaly_zscore_threshold: Z-score threshold for anomaly detection (default 3.0)

        Returns:
            Tuple of (BytesIO containing PNG image, List of detected anomalies)
            Each anomaly is a dict with: date, metric, value, reason
        """
        detected_anomalies = []

        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Calculate returns
        df['returns'] = df['equity'].pct_change()

        if len(df) < window + 10:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'Insufficient data for {window}-day rolling metrics',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig), detected_anomalies

        # Calculate rolling metrics
        risk_free_daily = (1 + 0.035) ** (1/252) - 1  # 3.5% annual

        # Rolling Sharpe
        rolling_mean = df['returns'].rolling(window).mean()
        rolling_std = df['returns'].rolling(window).std()
        df['rolling_sharpe'] = ((rolling_mean - risk_free_daily) / rolling_std) * np.sqrt(252)

        # Rolling Sortino
        def rolling_sortino(returns, window):
            result = []
            for i in range(len(returns)):
                if i < window:
                    result.append(np.nan)
                else:
                    window_returns = returns.iloc[i-window:i]
                    excess = window_returns - risk_free_daily
                    downside = window_returns[window_returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        sortino = (excess.mean() / downside.std()) * np.sqrt(252)
                    else:
                        sortino = np.nan
                    result.append(sortino)
            return result

        df['rolling_sortino'] = rolling_sortino(df['returns'], window)

        # Rolling Volatility (annualized)
        df['rolling_volatility'] = rolling_std * np.sqrt(252) * 100

        # Detect and filter anomalies if enabled
        if filter_anomalies:
            for metric in ['rolling_sharpe', 'rolling_sortino']:
                series = df[metric].dropna()
                if len(series) < 10:
                    continue

                # Calculate rolling statistics for z-score
                metric_mean = series.mean()
                metric_std = series.std()

                for date, value in series.items():
                    is_anomaly = False
                    reason = []

                    # Check absolute threshold
                    if abs(value) > anomaly_absolute_threshold:
                        is_anomaly = True
                        reason.append(f"|value| > {anomaly_absolute_threshold}")

                    # Check z-score threshold
                    if metric_std > 0:
                        z_score = abs(value - metric_mean) / metric_std
                        if z_score > anomaly_zscore_threshold:
                            is_anomaly = True
                            reason.append(f"z-score {z_score:.1f} > {anomaly_zscore_threshold}")

                    if is_anomaly:
                        detected_anomalies.append({
                            'date': date,
                            'metric': 'Sharpe Ratio' if metric == 'rolling_sharpe' else 'Sortino Ratio',
                            'value': value,
                            'reason': ' AND '.join(reason)
                        })
                        # Replace with NaN in the dataframe for plotting
                        df.loc[date, metric] = np.nan

        # Create copies for plotting (filtered values)
        sharpe_plot = df['rolling_sharpe'].copy()
        sortino_plot = df['rolling_sortino'].copy()

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Update title to indicate filtering
        if filter_anomalies and detected_anomalies:
            title_suffix = f"\n({len(detected_anomalies)} anomalous values filtered - see report for details)"
            fig.suptitle(title + title_suffix, fontsize=14, fontweight='bold', y=0.99)
        else:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        dates = df.index

        # Plot Rolling Sharpe (filtered)
        ax1 = axes[0]
        ax1.plot(dates, sharpe_plot, color=self.COLORS['primary'], linewidth=1.2)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.axhline(1, color=self.COLORS['positive'], linestyle='--', linewidth=0.8, alpha=0.5, label='Good (1.0)')
        ax1.axhline(2, color=self.COLORS['positive'], linestyle='--', linewidth=0.8, alpha=0.3, label='Excellent (2.0)')
        ax1.fill_between(dates, 0, sharpe_plot,
                         where=sharpe_plot >= 0, color=self.COLORS['positive'], alpha=0.2)
        ax1.fill_between(dates, 0, sharpe_plot,
                         where=sharpe_plot < 0, color=self.COLORS['negative'], alpha=0.2)
        ax1.set_ylabel(f'{window}-Day Rolling\nSharpe Ratio', fontsize=10)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot Rolling Sortino (filtered)
        ax2 = axes[1]
        ax2.plot(dates, sortino_plot, color=self.COLORS['secondary'], linewidth=1.2)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.axhline(1.5, color=self.COLORS['positive'], linestyle='--', linewidth=0.8, alpha=0.5, label='Good (1.5)')
        ax2.fill_between(dates, 0, sortino_plot,
                         where=sortino_plot >= 0, color=self.COLORS['positive'], alpha=0.2)
        ax2.fill_between(dates, 0, sortino_plot,
                         where=sortino_plot < 0, color=self.COLORS['negative'], alpha=0.2)
        ax2.set_ylabel(f'{window}-Day Rolling\nSortino Ratio', fontsize=10)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot Rolling Volatility
        ax3 = axes[2]
        ax3.fill_between(dates, 0, df['rolling_volatility'],
                         color=self.COLORS['warning'], alpha=0.4)
        ax3.plot(dates, df['rolling_volatility'], color=self.COLORS['warning'], linewidth=1.2)
        ax3.set_ylabel(f'{window}-Day Rolling\nVolatility (%)', fontsize=10)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)

        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 200)))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return self._get_figure_bytes(fig), detected_anomalies

    def create_win_rate_over_time(
        self,
        trades: List[Any],
        window: int = 20,
        title: str = "Win Rate and Profit Factor Over Time",
        figsize: Tuple[int, int] = (14, 8)
    ) -> BytesIO:
        """
        Create rolling win rate and profit factor chart.

        Args:
            trades: List of Trade objects
            window: Number of trades for rolling calculation
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        if len(trades) < window + 5:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'Insufficient trades for {window}-trade rolling analysis',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        # Sort trades by exit date
        sorted_trades = sorted(trades, key=lambda t: t.exit_date)

        dates = []
        win_rates = []
        profit_factors = []

        for i in range(window, len(sorted_trades) + 1):
            window_trades = sorted_trades[i-window:i]
            dates.append(window_trades[-1].exit_date)

            # Win rate
            wins = sum(1 for t in window_trades if t.pl > 0)
            win_rates.append(wins / window * 100)

            # Profit factor
            gross_profit = sum(t.pl for t in window_trades if t.pl > 0)
            gross_loss = abs(sum(t.pl for t in window_trades if t.pl < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 10
            profit_factors.append(min(pf, 10))  # Cap at 10 for visualization

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Plot Win Rate
        ax1.plot(dates, win_rates, color=self.COLORS['primary'], linewidth=1.5)
        ax1.fill_between(dates, 50, win_rates,
                         where=np.array(win_rates) >= 50, color=self.COLORS['positive'], alpha=0.3)
        ax1.fill_between(dates, 50, win_rates,
                         where=np.array(win_rates) < 50, color=self.COLORS['negative'], alpha=0.3)
        ax1.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
        ax1.set_ylabel(f'{window}-Trade Rolling\nWin Rate (%)', fontsize=11)
        ax1.set_ylim(0, 100)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot Profit Factor
        ax2.plot(dates, profit_factors, color=self.COLORS['secondary'], linewidth=1.5)
        ax2.fill_between(dates, 1, profit_factors,
                         where=np.array(profit_factors) >= 1, color=self.COLORS['positive'], alpha=0.3)
        ax2.fill_between(dates, 1, profit_factors,
                         where=np.array(profit_factors) < 1, color=self.COLORS['negative'], alpha=0.3)
        ax2.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
        ax2.axhline(1.5, color=self.COLORS['positive'], linestyle=':', linewidth=1, alpha=0.5, label='Good (1.5)')
        ax2.set_ylabel(f'{window}-Trade Rolling\nProfit Factor', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylim(0, min(10, max(profit_factors) * 1.1))
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_mae_mfe_analysis(
        self,
        trades: List[Any],
        title: str = "MAE/MFE Analysis",
        figsize: Tuple[int, int] = (14, 6)
    ) -> BytesIO:
        """
        Create Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE) analysis.

        Note: This requires trades to have mae and mfe attributes. If not available,
        it will show estimated values based on entry_price, exit_price, and stop_loss.

        Args:
            trades: List of Trade objects
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        if not trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Calculate returns
        returns = [t.pl_pct for t in trades]
        winners = [t for t in trades if t.pl > 0]
        losers = [t for t in trades if t.pl <= 0]

        # MAE Analysis - estimate based on stop loss distance
        # (Lower stop loss = larger potential MAE)
        mae_estimates = []
        for t in trades:
            if t.initial_stop_loss and t.entry_price > 0:
                mae_pct = abs((t.initial_stop_loss - t.entry_price) / t.entry_price * 100)
                mae_estimates.append(mae_pct)
            else:
                mae_estimates.append(abs(min(0, t.pl_pct)))  # Use actual loss as floor

        # Scatter plot: MAE vs Trade Result
        colors = [self.COLORS['positive'] if r > 0 else self.COLORS['negative'] for r in returns]
        ax1.scatter(mae_estimates, returns, c=colors, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)

        # Add trend line for losers
        if losers:
            loser_maes = [mae_estimates[i] for i, t in enumerate(trades) if t.pl <= 0]
            loser_returns = [t.pl_pct for t in losers]
            if len(loser_maes) > 2:
                z = np.polyfit(loser_maes, loser_returns, 1)
                p = np.poly1d(z)
                x_line = np.linspace(0, max(loser_maes), 100)
                ax1.plot(x_line, p(x_line), '--', color=self.COLORS['negative'], alpha=0.7, label='Loser trend')

        ax1.set_xlabel('Stop Loss Distance (%)', fontsize=11)
        ax1.set_ylabel('Trade Return (%)', fontsize=11)
        ax1.set_title('Risk vs. Reward Distribution', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Trade Duration Analysis
        durations = [t.duration_days for t in trades]

        ax2.scatter(durations, returns, c=colors, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

        ax2.set_xlabel('Trade Duration (Days)', fontsize=11)
        ax2.set_ylabel('Trade Return (%)', fontsize=11)
        ax2.set_title('Duration vs. Return Analysis', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add statistics
        avg_winner_duration = np.mean([t.duration_days for t in winners]) if winners else 0
        avg_loser_duration = np.mean([t.duration_days for t in losers]) if losers else 0

        stats_text = (f'Avg Winner Duration: {avg_winner_duration:.1f} days\n'
                      f'Avg Loser Duration: {avg_loser_duration:.1f} days')
        ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_contribution_analysis(
        self,
        symbol_pnl: Dict[str, float],
        title: str = "P/L Contribution by Security",
        figsize: Tuple[int, int] = (12, 8)
    ) -> BytesIO:
        """
        Create contribution analysis chart showing P/L by security.

        Args:
            symbol_pnl: Dictionary mapping symbol to P/L
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        if not symbol_pnl:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        # Sort by absolute contribution
        sorted_symbols = sorted(symbol_pnl.keys(), key=lambda x: abs(symbol_pnl[x]), reverse=True)
        pnls = [symbol_pnl[s] for s in sorted_symbols]
        total_pnl = sum(pnls)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Bar chart of contributions
        colors = [self.COLORS['positive'] if p > 0 else self.COLORS['negative'] for p in pnls]
        bars = ax1.barh(sorted_symbols, pnls, color=colors, edgecolor='white', linewidth=0.5)

        ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('P/L ($)', fontsize=11)
        ax1.set_ylabel('Security', fontsize=11)
        ax1.set_title('Absolute Contribution', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Add value labels
        for bar, pnl in zip(bars, pnls):
            width = bar.get_width()
            label_x = width + (total_pnl * 0.02 if width >= 0 else -total_pnl * 0.02)
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'${pnl:,.0f}',
                     ha='left' if width >= 0 else 'right', va='center', fontsize=8)

        # Pie chart of contribution percentages (absolute values)
        abs_pnls = [abs(p) for p in pnls if p != 0]
        pie_symbols = [s for s, p in zip(sorted_symbols, pnls) if p != 0]

        if abs_pnls:
            pie_colors = [self.COLORS['positive'] if symbol_pnl[s] > 0 else self.COLORS['negative']
                          for s in pie_symbols]

            wedges, texts, autotexts = ax2.pie(abs_pnls, labels=pie_symbols, autopct='%1.1f%%',
                                                colors=pie_colors, startangle=90)

            # Style the labels
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')

            ax2.set_title('Contribution Share', fontsize=12)

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_correlation_heatmap(
        self,
        returns_dict: Dict[str, pd.Series],
        title: str = "Security Returns Correlation Matrix",
        figsize: Tuple[int, int] = (10, 8)
    ) -> BytesIO:
        """
        Create correlation heatmap for security returns.

        Args:
            returns_dict: Dictionary mapping symbol to returns series
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        if len(returns_dict) < 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Need at least 2 securities for correlation analysis',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        # Create correlation matrix
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        cmap = LinearSegmentedColormap.from_list('correlation',
                                                  ['#dc3545', '#ffffff', '#28a745'], N=256)

        if SEABORN_AVAILABLE:
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
                        center=0, vmin=-1, vmax=1, ax=ax,
                        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
                        linewidths=0.5, linecolor='white', square=True)
        else:
            im = ax.imshow(corr_matrix.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=text_color, fontsize=9)

            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_yticklabels(corr_matrix.index)

            plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_3d_parameter_surface(
        self,
        param1_values: List[float],
        param2_values: List[float],
        performance_matrix: np.ndarray,
        param1_name: str = "Parameter 1",
        param2_name: str = "Parameter 2",
        metric_name: str = "Sortino Ratio",
        title: str = "Parameter Sensitivity Surface",
        figsize: Tuple[int, int] = (12, 9)
    ) -> BytesIO:
        """
        Create 3D surface plot for parameter sensitivity analysis.

        Args:
            param1_values: Values for first parameter
            param2_values: Values for second parameter
            performance_matrix: 2D array of performance metric values
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            metric_name: Name of the performance metric
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(param1_values, param2_values)
        Z = performance_matrix

        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', linewidth=0,
                               antialiased=True, alpha=0.8)

        # Add contour lines on the bottom
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='RdYlGn', alpha=0.5)

        ax.set_xlabel(param1_name, fontsize=10, labelpad=10)
        ax.set_ylabel(param2_name, fontsize=10, labelpad=10)
        ax.set_zlabel(metric_name, fontsize=10, labelpad=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=metric_name)

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_trade_clustering_analysis(
        self,
        trades: List[Any],
        title: str = "Trade Performance by Time Period",
        figsize: Tuple[int, int] = (14, 10)
    ) -> BytesIO:
        """
        Create trade clustering analysis showing performance by day of week, month, etc.

        Args:
            trades: List of Trade objects
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        if not trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        # Extract trade data
        df = pd.DataFrame([{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'pnl': t.pl,
            'pnl_pct': t.pl_pct,
            'duration': t.duration_days
        } for t in trades])

        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['entry_dow'] = df['entry_date'].dt.day_name()
        df['entry_month'] = df['entry_date'].dt.month_name()
        df['entry_hour'] = df['entry_date'].dt.hour

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Day of Week Analysis
        ax1 = axes[0, 0]
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dow_stats = df.groupby('entry_dow').agg({
            'pnl': ['mean', 'sum', 'count'],
            'pnl_pct': 'mean'
        }).reindex(dow_order)

        dow_stats.columns = ['avg_pnl', 'total_pnl', 'count', 'avg_pnl_pct']
        colors = [self.COLORS['positive'] if x > 0 else self.COLORS['negative']
                  for x in dow_stats['avg_pnl_pct'].values]

        bars = ax1.bar(dow_order, dow_stats['avg_pnl_pct'], color=colors, edgecolor='white')
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Avg Return (%)', fontsize=10)
        ax1.set_title('Performance by Day of Week', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add count labels
        for bar, count in zip(bars, dow_stats['count'].values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'n={int(count)}', ha='center', va='bottom', fontsize=8)

        # Month Analysis
        ax2 = axes[0, 1]
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        month_stats = df.groupby('entry_month').agg({
            'pnl_pct': 'mean',
            'pnl': 'count'
        }).reindex([m for m in month_order if m in df['entry_month'].values])

        if len(month_stats) > 0:
            colors = [self.COLORS['positive'] if x > 0 else self.COLORS['negative']
                      for x in month_stats['pnl_pct'].values]
            month_labels = [m[:3] for m in month_stats.index]
            bars = ax2.bar(month_labels, month_stats['pnl_pct'], color=colors, edgecolor='white')
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

        ax2.set_ylabel('Avg Return (%)', fontsize=10)
        ax2.set_title('Performance by Month', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # Trade Duration Distribution
        ax3 = axes[1, 0]
        duration_bins = [0, 5, 10, 20, 30, 60, 90, float('inf')]
        duration_labels = ['0-5d', '5-10d', '10-20d', '20-30d', '30-60d', '60-90d', '90d+']
        df['duration_bucket'] = pd.cut(df['duration'], bins=duration_bins, labels=duration_labels)

        duration_stats = df.groupby('duration_bucket', observed=True).agg({
            'pnl_pct': 'mean',
            'pnl': 'count'
        })

        colors = [self.COLORS['positive'] if x > 0 else self.COLORS['negative']
                  for x in duration_stats['pnl_pct'].values]
        bars = ax3.bar(duration_stats.index.astype(str), duration_stats['pnl_pct'],
                       color=colors, edgecolor='white')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Avg Return (%)', fontsize=10)
        ax3.set_xlabel('Trade Duration', fontsize=10)
        ax3.set_title('Performance by Duration', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Consecutive Trades Analysis
        ax4 = axes[1, 1]
        df_sorted = df.sort_values('exit_date')
        df_sorted['win'] = df_sorted['pnl'] > 0

        # Calculate streak lengths
        streaks = []
        current_streak = 0
        current_type = None

        for win in df_sorted['win']:
            if current_type is None or win == current_type:
                current_streak += 1
                current_type = win
            else:
                streaks.append((current_streak, 'Win' if current_type else 'Loss'))
                current_streak = 1
                current_type = win
        streaks.append((current_streak, 'Win' if current_type else 'Loss'))

        # Plot streak distribution
        win_streaks = [s[0] for s in streaks if s[1] == 'Win']
        loss_streaks = [s[0] for s in streaks if s[1] == 'Loss']

        max_streak = max(max(win_streaks, default=0), max(loss_streaks, default=0))
        bins = range(1, max_streak + 2)

        ax4.hist(win_streaks, bins=bins, alpha=0.6, color=self.COLORS['positive'],
                 label='Win Streaks', edgecolor='white')
        ax4.hist(loss_streaks, bins=bins, alpha=0.6, color=self.COLORS['negative'],
                 label='Loss Streaks', edgecolor='white')

        ax4.set_xlabel('Streak Length', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Win/Loss Streak Distribution', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_streak_visualization(
        self,
        trades: List[Any],
        title: str = "Win/Loss Streak Analysis",
        figsize: Tuple[int, int] = (14, 8)
    ) -> BytesIO:
        """
        Create comprehensive streak visualization with multiple views.

        Shows:
        - Waterfall chart of consecutive wins/losses
        - Streak length distribution (grouped bar chart)
        - Streak length statistics table

        Args:
            trades: List of Trade objects
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        if not trades:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return self._get_figure_bytes(fig)

        # Sort trades by exit date
        sorted_trades = sorted(trades, key=lambda t: t.exit_date)

        # Calculate streaks
        streaks = []  # (length, type, start_idx, end_idx, total_pnl)
        current_streak_length = 0
        current_type = None
        streak_pnl = 0
        start_idx = 0

        for i, trade in enumerate(sorted_trades):
            is_win = trade.pl > 0
            if current_type is None:
                current_type = is_win
                current_streak_length = 1
                streak_pnl = trade.pl
                start_idx = i
            elif is_win == current_type:
                current_streak_length += 1
                streak_pnl += trade.pl
            else:
                streaks.append((current_streak_length, 'Win' if current_type else 'Loss',
                               start_idx, i - 1, streak_pnl))
                current_type = is_win
                current_streak_length = 1
                streak_pnl = trade.pl
                start_idx = i

        streaks.append((current_streak_length, 'Win' if current_type else 'Loss',
                       start_idx, len(sorted_trades) - 1, streak_pnl))

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # 1. Waterfall chart showing sequence of streaks
        ax1 = axes[0, 0]
        positions = range(len(streaks))
        colors = [self.COLORS['positive'] if s[1] == 'Win' else self.COLORS['negative'] for s in streaks]
        lengths = [s[0] if s[1] == 'Win' else -s[0] for s in streaks]

        bars = ax1.bar(positions, lengths, color=colors, edgecolor='white', width=0.8)
        ax1.axhline(0, color='black', linewidth=1)

        # Add labels
        for i, (bar, streak) in enumerate(zip(bars, streaks)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + (0.2 if height > 0 else -0.4),
                    f'{abs(streak[0])}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')

        ax1.set_xlabel('Streak Sequence', fontsize=10)
        ax1.set_ylabel('Streak Length (+ Win / - Loss)', fontsize=10)
        ax1.set_title('Sequential Streak Waterfall', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Grouped bar chart - streak length distribution
        ax2 = axes[0, 1]
        win_streaks = [s[0] for s in streaks if s[1] == 'Win']
        loss_streaks = [s[0] for s in streaks if s[1] == 'Loss']

        max_streak = max(max(win_streaks, default=1), max(loss_streaks, default=1))
        streak_lengths = list(range(1, max_streak + 1))

        win_counts = [win_streaks.count(x) for x in streak_lengths]
        loss_counts = [loss_streaks.count(x) for x in streak_lengths]

        x = np.arange(len(streak_lengths))
        width = 0.35

        bars1 = ax2.bar(x - width/2, win_counts, width, label='Win Streaks',
                        color=self.COLORS['positive'], edgecolor='white')
        bars2 = ax2.bar(x + width/2, loss_counts, width, label='Loss Streaks',
                        color=self.COLORS['negative'], edgecolor='white')

        ax2.set_xlabel('Streak Length', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Streak Length Distribution', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(streak_lengths)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            if bar.get_height() > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            if bar.get_height() > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)

        # 3. Streak P&L distribution
        ax3 = axes[1, 0]
        streak_pnls = [s[4] for s in streaks]
        streak_colors = [self.COLORS['positive'] if s[1] == 'Win' else self.COLORS['negative'] for s in streaks]

        bars = ax3.bar(range(len(streaks)), streak_pnls, color=streak_colors, edgecolor='white')
        ax3.axhline(0, color='black', linewidth=1)
        ax3.set_xlabel('Streak Sequence', fontsize=10)
        ax3.set_ylabel('P/L ($)', fontsize=10)
        ax3.set_title('P/L by Streak', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Statistics summary as text
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Calculate statistics
        win_streak_stats = {
            'count': len(win_streaks),
            'max': max(win_streaks) if win_streaks else 0,
            'avg': np.mean(win_streaks) if win_streaks else 0,
            'total_trades': sum(win_streaks)
        }
        loss_streak_stats = {
            'count': len(loss_streaks),
            'max': max(loss_streaks) if loss_streaks else 0,
            'avg': np.mean(loss_streaks) if loss_streaks else 0,
            'total_trades': sum(loss_streaks)
        }

        stats_text = f"""
        STREAK STATISTICS
        ═══════════════════════════════════════

        WIN STREAKS
        ───────────────────────────────────────
        Number of Win Streaks:     {win_streak_stats['count']}
        Maximum Win Streak:        {win_streak_stats['max']} trades
        Average Win Streak:        {win_streak_stats['avg']:.1f} trades
        Total Winning Trades:      {win_streak_stats['total_trades']}

        LOSS STREAKS
        ───────────────────────────────────────
        Number of Loss Streaks:    {loss_streak_stats['count']}
        Maximum Loss Streak:       {loss_streak_stats['max']} trades
        Average Loss Streak:       {loss_streak_stats['avg']:.1f} trades
        Total Losing Trades:       {loss_streak_stats['total_trades']}

        OVERALL
        ───────────────────────────────────────
        Total Streaks:             {len(streaks)}
        Win Streak Ratio:          {win_streak_stats['count'] / len(streaks) * 100:.1f}%
        """

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_capital_utilization_chart(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Capital Utilization Over Time",
        figsize: Tuple[int, int] = (14, 6)
    ) -> BytesIO:
        """
        Create capital utilization chart showing cash vs invested capital.

        Args:
            equity_curve: DataFrame with 'date', 'equity', 'capital', 'position_value' columns
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])

        if 'capital' not in df.columns or 'position_value' not in df.columns:
            # Fall back to simple equity chart
            return self.create_equity_curve_with_drawdown(equity_curve, title=title, figsize=figsize)

        fig, ax = plt.subplots(figsize=figsize)

        dates = df['date']

        # Stack plot showing cash vs positions
        ax.fill_between(dates, 0, df['capital'], alpha=0.6,
                        color=self.COLORS['secondary'], label='Cash')
        ax.fill_between(dates, df['capital'], df['equity'], alpha=0.6,
                        color=self.COLORS['positive'], label='Invested')

        # Add equity line
        ax.plot(dates, df['equity'], color=self.COLORS['primary'], linewidth=1.5, label='Total Equity')

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Value ($)', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 150)))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return self._get_figure_bytes(fig)

    def create_executive_dashboard(
        self,
        metrics: Dict[str, Any],
        title: str = "Executive Summary Dashboard",
        figsize: Tuple[int, int] = (16, 12)
    ) -> BytesIO:
        """
        Create executive summary dashboard with key KPIs.

        Args:
            metrics: Dictionary containing all portfolio metrics
            title: Chart title
            figsize: Figure size

        Returns:
            BytesIO containing PNG image
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # KPI Cards (Top Row)
        kpi_data = [
            ('Total Return', f"${metrics.get('total_return', 0):,.0f}",
             f"{metrics.get('total_return_pct', 0):.1f}%", metrics.get('total_return', 0) >= 0),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}",
             'Good' if metrics.get('sharpe_ratio', 0) > 1 else 'Fair', metrics.get('sharpe_ratio', 0) > 0),
            ('Max Drawdown', f"{metrics.get('max_drawdown_pct', 0):.1f}%",
             'Acceptable' if metrics.get('max_drawdown_pct', 0) < 20 else 'High', metrics.get('max_drawdown_pct', 0) < 30),
            ('Win Rate', f"{metrics.get('win_rate', 0)*100:.1f}%",
             f"{metrics.get('num_wins', 0)}/{metrics.get('num_trades', 0)} trades", metrics.get('win_rate', 0) > 0.5),
        ]

        for i, (label, value, subtext, is_good) in enumerate(kpi_data):
            ax = fig.add_subplot(gs[0, i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Background color based on status
            bg_color = '#e8f5e9' if is_good else '#ffebee'
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=bg_color, edgecolor='#cccccc'))

            # Traffic light indicator
            indicator_color = self.COLORS['positive'] if is_good else self.COLORS['negative']
            ax.add_patch(plt.Circle((0.9, 0.85), 0.06, color=indicator_color))

            # Text
            ax.text(0.5, 0.7, value, ha='center', va='center', fontsize=18, fontweight='bold')
            ax.text(0.5, 0.4, label, ha='center', va='center', fontsize=11, color='#666666')
            ax.text(0.5, 0.2, subtext, ha='center', va='center', fontsize=9, color='#888888')

            ax.axis('off')

        # Performance Summary Table (Middle Left)
        ax_table = fig.add_subplot(gs[1, :2])
        ax_table.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['CAGR', f"{metrics.get('cagr', 0):.2f}%"],
            ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"],
            ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"],
            ['Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"],
            ['Recovery Factor', f"{metrics.get('recovery_factor', 0):.2f}"],
            ['Avg Trade Duration', f"{metrics.get('avg_trade_duration', 0):.1f} days"],
            ['Profitable Months', f"{metrics.get('profitable_months_pct', 0):.1f}%"],
        ]

        table = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                                colWidths=[0.4, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header row
        for j in range(2):
            table[(0, j)].set_facecolor(self.COLORS['primary'])
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax_table.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)

        # Risk Summary Table (Middle Right)
        ax_risk = fig.add_subplot(gs[1, 2:])
        ax_risk.axis('off')

        risk_data = [
            ['Risk Metric', 'Value'],
            ['Max Drawdown ($)', f"${metrics.get('max_drawdown', 0):,.0f}"],
            ['Avg Drawdown', f"${metrics.get('avg_drawdown', 0):,.0f}"],
            ['Volatility (Ann.)', f"{metrics.get('volatility', 0):.2f}%"],
            ['Downside Deviation', f"{metrics.get('downside_deviation', 0):.2f}%"],
            ['Best Day', f"{metrics.get('best_day', 0):.2f}%"],
            ['Worst Day', f"{metrics.get('worst_day', 0):.2f}%"],
            ['Max Loss Streak', f"{metrics.get('max_loss_streak', 0)} trades"],
        ]

        table2 = ax_risk.table(cellText=risk_data, loc='center', cellLoc='center',
                                colWidths=[0.4, 0.3])
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.5)

        for j in range(2):
            table2[(0, j)].set_facecolor(self.COLORS['negative'])
            table2[(0, j)].set_text_props(color='white', fontweight='bold')

        ax_risk.set_title('Risk Metrics', fontsize=12, fontweight='bold', pad=10)

        # Trade Statistics (Bottom Row)
        ax_trade = fig.add_subplot(gs[2, :2])
        ax_trade.axis('off')

        trade_data = [
            ['Trade Metric', 'Value'],
            ['Total Trades', f"{metrics.get('num_trades', 0)}"],
            ['Winning Trades', f"{metrics.get('num_wins', 0)}"],
            ['Losing Trades', f"{metrics.get('num_losses', 0)}"],
            ['Average Win', f"${metrics.get('avg_win', 0):,.2f}"],
            ['Average Loss', f"${metrics.get('avg_loss', 0):,.2f}"],
            ['Largest Win', f"${metrics.get('largest_win', 0):,.2f}"],
            ['Largest Loss', f"${metrics.get('largest_loss', 0):,.2f}"],
        ]

        table3 = ax_trade.table(cellText=trade_data, loc='center', cellLoc='center',
                                 colWidths=[0.4, 0.3])
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1.2, 1.5)

        for j in range(2):
            table3[(0, j)].set_facecolor(self.COLORS['secondary'])
            table3[(0, j)].set_text_props(color='white', fontweight='bold')

        ax_trade.set_title('Trade Statistics', fontsize=12, fontweight='bold', pad=10)

        # Recommendation Box (Bottom Right)
        ax_rec = fig.add_subplot(gs[2, 2:])
        ax_rec.axis('off')

        # Generate recommendations based on metrics
        recommendations = []

        if metrics.get('sharpe_ratio', 0) > 1.5:
            recommendations.append("+ Strong risk-adjusted returns")
        elif metrics.get('sharpe_ratio', 0) > 0.5:
            recommendations.append("~ Acceptable risk-adjusted returns")
        else:
            recommendations.append("- Weak risk-adjusted returns")

        if metrics.get('max_drawdown_pct', 0) > 30:
            recommendations.append("- High drawdown risk detected")
        elif metrics.get('max_drawdown_pct', 0) > 20:
            recommendations.append("~ Moderate drawdown risk")
        else:
            recommendations.append("+ Drawdown well controlled")

        if metrics.get('profit_factor', 0) > 1.5:
            recommendations.append("+ Good profit factor")
        elif metrics.get('profit_factor', 0) > 1.0:
            recommendations.append("~ Acceptable profit factor")
        else:
            recommendations.append("- Profit factor needs improvement")

        rec_text = "Strategy Assessment:\n\n" + "\n".join(recommendations)

        ax_rec.text(0.1, 0.9, rec_text, transform=ax_rec.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc', pad=1),
                    family='monospace')

        ax_rec.set_title('Assessment', fontsize=12, fontweight='bold', pad=10)

        return self._get_figure_bytes(fig)
