"""
Stable Performance Metrics Module.

This module provides regression-based "stable" performance metrics that aim to
measure underlying strategy performance in a way that is less sensitive to
lucky/unlucky start and end points.

Metrics included:
    - Regressed Annual Return (RAR%): Annualized return derived from log-equity regression
    - R² of log-equity regression: Measures how "straight" the equity curve is
    - RAR% Adjusted: RAR% × R² to penalize noisy equity curves
    - R-Cubed: RAR% / (Avg Max Drawdown × Length Adjustment) - risk/reward with duration
    - Robust Sharpe Ratio: RAR% / Annualized Std Dev of Monthly Returns

CONSTANTS:
    - BARS_PER_YEAR: 365 (calendar days, as specified)
    - ROLLING_WINDOW_DAYS: 21 (approximate trading month for rolling monthly returns)
    - MIN_DRAWDOWNS_FOR_R_CUBED: 5 (preferred, will use available if fewer)

Usage:
    from Classes.Core.stable_metrics import StableMetricsCalculator

    # From equity curve
    stable = StableMetricsCalculator.calculate_all(equity_curve)

    # Individual calculations
    rar = StableMetricsCalculator.calculate_rar(equity_curve)
    r_cubed = StableMetricsCalculator.calculate_r_cubed(equity_curve)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================

# Bars per year for annualization (calendar days as specified)
BARS_PER_YEAR: int = 365

# Rolling window for "monthly" returns calculation (trading days approximating a month)
ROLLING_WINDOW_DAYS: int = 21

# Preferred number of drawdowns for R-Cubed calculation
PREFERRED_DRAWDOWNS_FOR_R_CUBED: int = 5

# Minimum R² threshold for reliability warning
MIN_R_SQUARED_THRESHOLD: float = 0.5

# Minimum periods needed for robust calculations
MIN_PERIODS_FOR_MONTHLY_RETURNS: int = 42  # About 2 months of trading days

# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

@dataclass
class StableMetricDefinition:
    """Definition of a stable performance metric."""
    name: str
    description: str
    higher_is_better: bool = True
    format_str: str = "{:.2f}"
    category: str = "stable"

    def format(self, value: float) -> str:
        """Format the metric value for display."""
        if pd.isna(value) or np.isinf(value):
            return "N/A"
        try:
            return self.format_str.format(value)
        except (ValueError, TypeError):
            return str(value)


STABLE_METRIC_DEFINITIONS: Dict[str, StableMetricDefinition] = {
    # Regression Analysis
    "rar_pct": StableMetricDefinition(
        name="Regressed Annual Return (RAR%)",
        description="Annualized return from log-equity regression slope",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="regression"
    ),
    "r_squared": StableMetricDefinition(
        name="R² (Log-Equity Regression)",
        description="Goodness of fit for log-equity vs time regression",
        higher_is_better=True,
        format_str="{:.4f}",
        category="regression"
    ),
    "rar_adjusted": StableMetricDefinition(
        name="RAR% Adjusted",
        description="RAR% × R² - penalizes noisy equity curves",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="regression"
    ),
    "cagr": StableMetricDefinition(
        name="CAGR",
        description="Compound Annual Growth Rate (classical)",
        higher_is_better=True,
        format_str="{:.2f}%",
        category="regression"
    ),
    "bars_per_year": StableMetricDefinition(
        name="Bars Per Year",
        description="Number of bars used for annualization",
        higher_is_better=True,
        format_str="{:.0f}",
        category="regression"
    ),

    # R-Cubed Components
    "r_cubed": StableMetricDefinition(
        name="R-Cubed",
        description="RAR% / (Avg Max Drawdown × Length Adjustment)",
        higher_is_better=True,
        format_str="{:.2f}",
        category="r_cubed"
    ),
    "avg_max_drawdown_pct": StableMetricDefinition(
        name="Avg Max Drawdown (%)",
        description="Average of largest drawdowns (up to 5)",
        higher_is_better=False,
        format_str="{:.2f}%",
        category="r_cubed"
    ),
    "avg_max_drawdown_length_days": StableMetricDefinition(
        name="Avg Max Drawdown Length (days)",
        description="Average duration of largest drawdowns",
        higher_is_better=False,
        format_str="{:.1f}",
        category="r_cubed"
    ),
    "length_adjustment_factor": StableMetricDefinition(
        name="Length Adjustment Factor",
        description="Avg DD Length / 365",
        higher_is_better=False,
        format_str="{:.4f}",
        category="r_cubed"
    ),
    "num_drawdowns_used": StableMetricDefinition(
        name="Drawdowns Used",
        description="Number of drawdowns used in R-Cubed calculation",
        higher_is_better=True,
        format_str="{:.0f}",
        category="r_cubed"
    ),

    # Robust Sharpe
    "robust_sharpe_ratio": StableMetricDefinition(
        name="Robust Sharpe Ratio",
        description="RAR% / Annualized Std Dev of Monthly Returns",
        higher_is_better=True,
        format_str="{:.3f}",
        category="robust_sharpe"
    ),
    "monthly_return_std_annualized": StableMetricDefinition(
        name="Monthly Return Std (Annualized)",
        description="Annualized standard deviation of rolling monthly returns",
        higher_is_better=False,
        format_str="{:.2f}%",
        category="robust_sharpe"
    ),

    # Warnings
    "r_squared_warning": StableMetricDefinition(
        name="R² Warning",
        description="Warning if R² is too low for reliable RAR%",
        higher_is_better=True,
        format_str="{}",
        category="warnings"
    ),
    "drawdown_warning": StableMetricDefinition(
        name="Drawdown Warning",
        description="Warning if fewer than 5 drawdowns available",
        higher_is_better=True,
        format_str="{}",
        category="warnings"
    ),
}


# ============================================================================
# DRAWDOWN DETAIL CLASS
# ============================================================================

@dataclass
class DrawdownDetail:
    """Details of a single drawdown period."""
    start_date: Any
    trough_date: Any
    recovery_date: Optional[Any]
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovered: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'start_date': str(self.start_date)[:10] if self.start_date else "N/A",
            'trough_date': str(self.trough_date)[:10] if self.trough_date else "N/A",
            'recovery_date': str(self.recovery_date)[:10] if self.recovery_date else "Not recovered",
            'drawdown_pct': f"{self.drawdown_pct:.2f}%",
            'duration_days': self.duration_days
        }


# ============================================================================
# STABLE METRICS RESULT CLASS
# ============================================================================

@dataclass
class StableMetricsResult:
    """Complete results of stable metrics calculations."""

    # Regression Analysis
    rar_pct: float = 0.0
    r_squared: float = 0.0
    rar_adjusted: float = 0.0
    cagr: float = 0.0
    bars_per_year: int = BARS_PER_YEAR
    regression_slope: float = 0.0
    regression_intercept: float = 0.0

    # R-Cubed Components
    r_cubed: float = 0.0
    avg_max_drawdown_pct: float = 0.0
    avg_max_drawdown_length_days: float = 0.0
    length_adjustment_factor: float = 0.0
    num_drawdowns_used: int = 0
    largest_drawdowns: List[DrawdownDetail] = field(default_factory=list)

    # Robust Sharpe
    robust_sharpe_ratio: float = 0.0
    monthly_return_std_annualized: float = 0.0

    # Warnings
    r_squared_warning: str = ""
    drawdown_warning: str = ""
    monthly_returns_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for report generation."""
        return {
            # Regression
            'rar_pct': self.rar_pct,
            'r_squared': self.r_squared,
            'rar_adjusted': self.rar_adjusted,
            'cagr': self.cagr,
            'bars_per_year': self.bars_per_year,

            # R-Cubed
            'r_cubed': self.r_cubed,
            'avg_max_drawdown_pct': self.avg_max_drawdown_pct,
            'avg_max_drawdown_length_days': self.avg_max_drawdown_length_days,
            'length_adjustment_factor': self.length_adjustment_factor,
            'num_drawdowns_used': self.num_drawdowns_used,

            # Robust Sharpe
            'robust_sharpe_ratio': self.robust_sharpe_ratio,
            'monthly_return_std_annualized': self.monthly_return_std_annualized,

            # Warnings
            'r_squared_warning': self.r_squared_warning,
            'drawdown_warning': self.drawdown_warning,
            'monthly_returns_warning': self.monthly_returns_warning,
        }

    def get_formatted_metrics(self) -> Dict[str, str]:
        """Get all metrics formatted for display."""
        result = {}
        data = self.to_dict()

        for key, definition in STABLE_METRIC_DEFINITIONS.items():
            if key in data:
                result[key] = definition.format(data[key])

        return result


# ============================================================================
# STABLE METRICS CALCULATOR CLASS
# ============================================================================

class StableMetricsCalculator:
    """
    Calculator for stable performance metrics.

    Provides regression-based metrics that are more robust to lucky/unlucky
    start and end points than traditional metrics.
    """

    BARS_PER_YEAR = BARS_PER_YEAR

    @classmethod
    def calculate_all(
        cls,
        equity_curve: pd.DataFrame,
        equity_column: str = 'equity',
        date_column: str = 'date'
    ) -> StableMetricsResult:
        """
        Calculate all stable metrics from an equity curve.

        Args:
            equity_curve: DataFrame with equity values and dates
            equity_column: Name of equity column
            date_column: Name of date column

        Returns:
            StableMetricsResult with all calculated metrics
        """
        result = StableMetricsResult()

        if equity_curve is None or len(equity_curve) < 2:
            return result

        # Get equity series
        if equity_column in equity_curve.columns:
            equity = equity_curve[equity_column].values
        elif len(equity_curve.columns) > 0:
            equity = equity_curve.iloc[:, 0].values
        else:
            return result

        # Clean data - remove NaN and non-positive values
        equity = np.array(equity, dtype=float)
        valid_mask = ~np.isnan(equity) & (equity > 0)
        if not np.any(valid_mask):
            return result

        equity_clean = equity[valid_mask]

        # Get dates if available
        dates = None
        if date_column in equity_curve.columns:
            dates = pd.to_datetime(equity_curve[date_column].values[valid_mask])

        # Calculate RAR% and R²
        rar_result = cls._calculate_rar_and_r_squared(equity_clean)
        result.rar_pct = rar_result['rar_pct']
        result.r_squared = rar_result['r_squared']
        result.rar_adjusted = rar_result['rar_adjusted']
        result.regression_slope = rar_result['slope']
        result.regression_intercept = rar_result['intercept']
        result.bars_per_year = BARS_PER_YEAR

        # R² warning
        if result.r_squared < MIN_R_SQUARED_THRESHOLD:
            result.r_squared_warning = (
                f"R² = {result.r_squared:.3f} is below {MIN_R_SQUARED_THRESHOLD}. "
                "RAR% may be unreliable due to non-linear equity curve."
            )

        # Calculate CAGR
        result.cagr = cls._calculate_cagr(equity_clean, dates)

        # Calculate drawdowns for R-Cubed
        drawdowns = cls._calculate_all_drawdowns(equity_clean, dates)
        result.largest_drawdowns = drawdowns[:PREFERRED_DRAWDOWNS_FOR_R_CUBED]
        result.num_drawdowns_used = len(result.largest_drawdowns)

        # Drawdown warning
        if result.num_drawdowns_used < PREFERRED_DRAWDOWNS_FOR_R_CUBED:
            result.drawdown_warning = (
                f"Only {result.num_drawdowns_used} drawdowns available "
                f"(preferred: {PREFERRED_DRAWDOWNS_FOR_R_CUBED}). "
                "R-Cubed calculation uses all available drawdowns."
            )

        # Calculate R-Cubed
        if result.largest_drawdowns:
            r_cubed_result = cls._calculate_r_cubed(
                result.rar_pct,
                result.largest_drawdowns
            )
            result.r_cubed = r_cubed_result['r_cubed']
            result.avg_max_drawdown_pct = r_cubed_result['avg_max_drawdown_pct']
            result.avg_max_drawdown_length_days = r_cubed_result['avg_max_drawdown_length_days']
            result.length_adjustment_factor = r_cubed_result['length_adjustment_factor']

        # Calculate Robust Sharpe Ratio
        if len(equity_clean) >= MIN_PERIODS_FOR_MONTHLY_RETURNS:
            robust_sharpe_result = cls._calculate_robust_sharpe(
                equity_clean,
                result.rar_pct
            )
            result.robust_sharpe_ratio = robust_sharpe_result['robust_sharpe_ratio']
            result.monthly_return_std_annualized = robust_sharpe_result['monthly_std_annualized']
        else:
            result.monthly_returns_warning = (
                f"Insufficient data for robust Sharpe calculation "
                f"(need {MIN_PERIODS_FOR_MONTHLY_RETURNS} periods, have {len(equity_clean)})"
            )

        return result

    @classmethod
    def _calculate_rar_and_r_squared(
        cls,
        equity: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Regressed Annual Return and R² from equity curve.

        Algorithm:
        1. Create time index x = 1, 2, ..., N
        2. Take natural log of equity: y = ln(E)
        3. Fit OLS: y = a + b*x
        4. RAR% = (exp(b * B) - 1) * 100 where B = bars per year
        5. R² from regression

        Args:
            equity: Array of equity values (must be positive)

        Returns:
            Dict with rar_pct, r_squared, rar_adjusted, slope, intercept
        """
        n = len(equity)

        if n < 2:
            return {
                'rar_pct': 0.0,
                'r_squared': 0.0,
                'rar_adjusted': 0.0,
                'slope': 0.0,
                'intercept': 0.0
            }

        # Time index
        x = np.arange(1, n + 1, dtype=float)

        # Log equity
        y = np.log(equity)

        # OLS regression: y = a + b*x
        # Using numpy for efficiency
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Calculate slope (b) and intercept (a)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-10:
            return {
                'rar_pct': 0.0,
                'r_squared': 0.0,
                'rar_adjusted': 0.0,
                'slope': 0.0,
                'intercept': 0.0
            }

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R²
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]

        # Calculate RAR%
        # Annual growth factor = exp(slope * bars_per_year)
        # RAR% = (growth_factor - 1) * 100
        annual_growth_factor = np.exp(slope * BARS_PER_YEAR)
        rar_pct = (annual_growth_factor - 1) * 100

        # RAR% Adjusted = RAR% × R²
        rar_adjusted = rar_pct * r_squared

        return {
            'rar_pct': float(rar_pct),
            'r_squared': float(r_squared),
            'rar_adjusted': float(rar_adjusted),
            'slope': float(slope),
            'intercept': float(intercept)
        }

    @classmethod
    def _calculate_cagr(
        cls,
        equity: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Args:
            equity: Array of equity values
            dates: Optional dates for accurate year calculation

        Returns:
            CAGR as percentage
        """
        if len(equity) < 2:
            return 0.0

        initial = equity[0]
        final = equity[-1]

        if initial <= 0:
            return 0.0

        # Calculate time period
        if dates is not None and len(dates) >= 2:
            days = (dates[-1] - dates[0]).days
        else:
            days = len(equity)

        if days <= 0:
            return 0.0

        years = days / 365.25

        if years <= 0:
            return 0.0

        cagr = (pow(final / initial, 1 / years) - 1) * 100

        return float(cagr) if not np.isnan(cagr) else 0.0

    @classmethod
    def _calculate_all_drawdowns(
        cls,
        equity: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> List[DrawdownDetail]:
        """
        Calculate all drawdown periods, sorted by severity (largest first).

        Args:
            equity: Array of equity values
            dates: Optional dates for duration calculation

        Returns:
            List of DrawdownDetail sorted by drawdown_pct descending
        """
        n = len(equity)
        if n < 2:
            return []

        # Running maximum
        running_max = np.maximum.accumulate(equity)

        # Drawdown percentage
        drawdown_pct = (running_max - equity) / running_max * 100

        # Find drawdown periods
        in_drawdown = drawdown_pct > 0

        drawdowns = []
        dd_start_idx = None
        peak_idx = None

        for i in range(n):
            if in_drawdown[i] and dd_start_idx is None:
                # Start of drawdown
                dd_start_idx = i - 1 if i > 0 else 0
                peak_idx = dd_start_idx
            elif not in_drawdown[i] and dd_start_idx is not None:
                # End of drawdown (recovery)
                trough_idx = dd_start_idx + np.argmax(drawdown_pct[dd_start_idx:i])
                recovery_idx = i

                # Calculate duration
                if dates is not None:
                    start_date = dates[dd_start_idx]
                    trough_date = dates[trough_idx]
                    recovery_date = dates[recovery_idx]
                    duration = (recovery_date - start_date).days
                else:
                    start_date = dd_start_idx
                    trough_date = trough_idx
                    recovery_date = recovery_idx
                    duration = recovery_idx - dd_start_idx

                drawdowns.append(DrawdownDetail(
                    start_date=start_date,
                    trough_date=trough_date,
                    recovery_date=recovery_date,
                    peak_value=equity[peak_idx],
                    trough_value=equity[trough_idx],
                    drawdown_pct=float(drawdown_pct[trough_idx]),
                    duration_days=duration,
                    recovered=True
                ))

                dd_start_idx = None
                peak_idx = None

        # Handle if still in drawdown at end
        if dd_start_idx is not None:
            trough_idx = dd_start_idx + np.argmax(drawdown_pct[dd_start_idx:])

            if dates is not None:
                start_date = dates[dd_start_idx]
                trough_date = dates[trough_idx]
                duration = (dates[-1] - start_date).days
            else:
                start_date = dd_start_idx
                trough_date = trough_idx
                duration = n - 1 - dd_start_idx

            drawdowns.append(DrawdownDetail(
                start_date=start_date,
                trough_date=trough_date,
                recovery_date=None,
                peak_value=equity[peak_idx],
                trough_value=equity[trough_idx],
                drawdown_pct=float(drawdown_pct[trough_idx]),
                duration_days=duration,
                recovered=False
            ))

        # Sort by drawdown percentage (largest first)
        drawdowns.sort(key=lambda x: x.drawdown_pct, reverse=True)

        return drawdowns

    @classmethod
    def _calculate_r_cubed(
        cls,
        rar_pct: float,
        drawdowns: List[DrawdownDetail]
    ) -> Dict[str, float]:
        """
        Calculate R-Cubed metric.

        R-Cubed = RAR% / (Avg Max Drawdown × Length Adjustment)

        Where:
        - Avg Max Drawdown = average of the N largest drawdowns
        - Length Adjustment = Avg Drawdown Length (days) / 365

        Args:
            rar_pct: Regressed Annual Return percentage
            drawdowns: List of drawdown details (sorted by severity)

        Returns:
            Dict with r_cubed, avg_max_drawdown_pct, avg_max_drawdown_length_days,
                 length_adjustment_factor
        """
        if not drawdowns:
            return {
                'r_cubed': 0.0,
                'avg_max_drawdown_pct': 0.0,
                'avg_max_drawdown_length_days': 0.0,
                'length_adjustment_factor': 0.0
            }

        # Use up to PREFERRED_DRAWDOWNS_FOR_R_CUBED drawdowns
        n_drawdowns = min(len(drawdowns), PREFERRED_DRAWDOWNS_FOR_R_CUBED)
        selected = drawdowns[:n_drawdowns]

        # Average max drawdown (percentage)
        avg_max_dd_pct = np.mean([dd.drawdown_pct for dd in selected])

        # Average max drawdown length (days)
        avg_max_dd_length = np.mean([dd.duration_days for dd in selected])

        # Length adjustment = avg length / 365
        length_adjustment = avg_max_dd_length / 365.0

        # R-Cubed = RAR% / (Avg Max DD × Length Adjustment)
        denominator = avg_max_dd_pct * length_adjustment

        if denominator > 1e-10:
            r_cubed = rar_pct / denominator
        else:
            r_cubed = 0.0 if rar_pct <= 0 else 999.99  # Cap for edge case

        return {
            'r_cubed': float(r_cubed),
            'avg_max_drawdown_pct': float(avg_max_dd_pct),
            'avg_max_drawdown_length_days': float(avg_max_dd_length),
            'length_adjustment_factor': float(length_adjustment)
        }

    @classmethod
    def _calculate_robust_sharpe(
        cls,
        equity: np.ndarray,
        rar_pct: float
    ) -> Dict[str, float]:
        """
        Calculate Robust Sharpe Ratio.

        Robust Sharpe = RAR% / Annualized Std Dev of Monthly Returns

        Uses rolling 21-day window for "monthly" returns to avoid
        calendar month dependencies.

        Args:
            equity: Array of equity values
            rar_pct: Regressed Annual Return percentage

        Returns:
            Dict with robust_sharpe_ratio and monthly_std_annualized
        """
        n = len(equity)

        if n < ROLLING_WINDOW_DAYS:
            return {
                'robust_sharpe_ratio': 0.0,
                'monthly_std_annualized': 0.0
            }

        # Calculate rolling 21-day returns
        equity_series = pd.Series(equity)
        rolling_returns = equity_series.pct_change(periods=ROLLING_WINDOW_DAYS).dropna()

        if len(rolling_returns) < 2:
            return {
                'robust_sharpe_ratio': 0.0,
                'monthly_std_annualized': 0.0
            }

        # Standard deviation of monthly returns
        monthly_std = rolling_returns.std()

        # Annualize: there are approximately 12 "monthly" periods per year
        # (365 / 21 ≈ 17.4, but we use 12 for traditional monthly interpretation)
        periods_per_year = 12
        monthly_std_annualized = monthly_std * np.sqrt(periods_per_year) * 100

        # Robust Sharpe = RAR% / annualized monthly std
        if monthly_std_annualized > 1e-10:
            # Both are in percentage terms, so divide directly
            robust_sharpe = rar_pct / monthly_std_annualized
        else:
            robust_sharpe = 0.0 if rar_pct <= 0 else 99.99

        return {
            'robust_sharpe_ratio': float(robust_sharpe),
            'monthly_std_annualized': float(monthly_std_annualized)
        }

    @classmethod
    def get_regression_line_data(
        cls,
        equity: np.ndarray,
        slope: float,
        intercept: float
    ) -> np.ndarray:
        """
        Get the fitted regression line values for plotting.

        Args:
            equity: Original equity array
            slope: Regression slope from log-equity fit
            intercept: Regression intercept from log-equity fit

        Returns:
            Array of fitted equity values (exp of log regression line)
        """
        n = len(equity)
        x = np.arange(1, n + 1, dtype=float)
        log_fitted = intercept + slope * x
        return np.exp(log_fitted)

    @staticmethod
    def get_metric_definition(metric_name: str) -> Optional[StableMetricDefinition]:
        """Get the definition for a stable metric."""
        return STABLE_METRIC_DEFINITIONS.get(metric_name)

    @staticmethod
    def get_available_metrics() -> List[str]:
        """Get list of all available stable metric names."""
        return list(STABLE_METRIC_DEFINITIONS.keys())
