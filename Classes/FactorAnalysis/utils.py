"""
Utility functions for Factor Analysis Module.

Provides common helper functions used across the module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path


def parse_date(
    date_value: Union[str, datetime, pd.Timestamp],
    dayfirst: bool = True
) -> pd.Timestamp:
    """
    Parse a date value to pandas Timestamp.

    Args:
        date_value: Date as string, datetime, or Timestamp
        dayfirst: Whether to interpret ambiguous dates as day-first

    Returns:
        pandas Timestamp

    Raises:
        ValueError: If date cannot be parsed
    """
    if pd.isna(date_value):
        raise ValueError("Cannot parse null date value")

    if isinstance(date_value, pd.Timestamp):
        return date_value

    if isinstance(date_value, datetime):
        return pd.Timestamp(date_value)

    if isinstance(date_value, str):
        try:
            return pd.to_datetime(date_value, dayfirst=dayfirst)
        except Exception as e:
            raise ValueError(f"Cannot parse date '{date_value}': {e}")

    raise ValueError(f"Unsupported date type: {type(date_value)}")


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide, returning default value for division by zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return for division by zero

    Returns:
        Division result with default for zero denominators
    """
    if isinstance(denominator, (np.ndarray, pd.Series)):
        result = np.where(denominator != 0, numerator / denominator, default)
        if isinstance(denominator, pd.Series):
            return pd.Series(result, index=denominator.index)
        return result
    else:
        return numerator / denominator if denominator != 0 else default


def calculate_zscore(
    values: pd.Series,
    min_std: float = 1e-10
) -> pd.Series:
    """
    Calculate z-scores for a series.

    Args:
        values: Series of values
        min_std: Minimum standard deviation to avoid division by zero

    Returns:
        Series of z-scores
    """
    mean = values.mean()
    std = max(values.std(), min_std)
    return (values - mean) / std


def calculate_percentile_rank(values: pd.Series) -> pd.Series:
    """
    Calculate percentile ranks for a series.

    Args:
        values: Series of values

    Returns:
        Series of percentile ranks (0-100)
    """
    return values.rank(pct=True) * 100


def winsorize(
    values: pd.Series,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0
) -> pd.Series:
    """
    Winsorize a series by capping extreme values.

    Args:
        values: Series of values
        lower_percentile: Lower percentile threshold
        upper_percentile: Upper percentile threshold

    Returns:
        Winsorized series
    """
    lower = np.percentile(values.dropna(), lower_percentile)
    upper = np.percentile(values.dropna(), upper_percentile)
    return values.clip(lower=lower, upper=upper)


def detect_outliers_zscore(
    values: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers using z-score method.

    Args:
        values: Series of values
        threshold: Z-score threshold for outlier detection

    Returns:
        Boolean series (True = outlier)
    """
    zscores = calculate_zscore(values)
    return zscores.abs() > threshold


def get_business_days_offset(
    date: pd.Timestamp,
    days: int
) -> pd.Timestamp:
    """
    Get a date offset by a number of business days.

    Args:
        date: Starting date
        days: Number of business days (negative for past)

    Returns:
        Offset date
    """
    return date + pd.offsets.BDay(days)


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate annualized return
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdowns.min())

    if max_drawdown == 0:
        return float('inf') if annualized_return > 0 else 0.0

    return annualized_return / max_drawdown


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.035,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def calculate_win_rate(pl_series: pd.Series) -> float:
    """
    Calculate win rate from P&L series.

    Args:
        pl_series: Series of P&L values

    Returns:
        Win rate as decimal (0-1)
    """
    if len(pl_series) == 0:
        return 0.0
    return (pl_series > 0).sum() / len(pl_series)


def bootstrap_ci(
    values: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        values: Array of values
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(values)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = values[sample_idx]
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper


def encode_categorical(
    values: pd.Series,
    categories: Optional[List[str]] = None
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Encode categorical values as integers.

    Args:
        values: Series of categorical values
        categories: Optional predefined category order

    Returns:
        Tuple of (encoded series, mapping dict)
    """
    if categories is None:
        categories = values.dropna().unique().tolist()

    mapping = {cat: i for i, cat in enumerate(categories)}
    encoded = values.map(mapping)

    return encoded, mapping


def merge_asof_with_delay(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    by: Optional[str] = None,
    delay_days: int = 0,
    direction: str = "backward"
) -> pd.DataFrame:
    """
    Merge with as-of logic and optional delay.

    Args:
        left_df: Left DataFrame (e.g., trades)
        right_df: Right DataFrame (e.g., fundamentals)
        left_on: Date column in left DataFrame
        right_on: Date column in right DataFrame
        by: Optional column to match on (e.g., symbol)
        delay_days: Number of days to subtract from left date before matching
        direction: 'backward' to get most recent, 'forward' for next

    Returns:
        Merged DataFrame
    """
    # Ensure date columns are datetime
    left_df = left_df.copy()
    right_df = right_df.copy()

    left_df[left_on] = pd.to_datetime(left_df[left_on])
    right_df[right_on] = pd.to_datetime(right_df[right_on])

    # Apply delay to left dates
    if delay_days > 0:
        left_df['_merge_date'] = left_df[left_on] - pd.Timedelta(days=delay_days)
    else:
        left_df['_merge_date'] = left_df[left_on]

    # Sort for merge_asof
    left_df = left_df.sort_values('_merge_date')
    right_df = right_df.sort_values(right_on)

    # Perform merge
    if by:
        result = pd.merge_asof(
            left_df,
            right_df,
            left_on='_merge_date',
            right_on=right_on,
            by=by,
            direction=direction
        )
    else:
        result = pd.merge_asof(
            left_df,
            right_df,
            left_on='_merge_date',
            right_on=right_on,
            direction=direction
        )

    # Clean up
    result = result.drop(columns=['_merge_date'])

    return result


def calculate_rolling_window_stats(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    window_days: int,
    symbol_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate rolling window statistics.

    Args:
        df: DataFrame with time series data
        date_column: Name of date column
        value_column: Name of value column
        window_days: Window size in days
        symbol_column: Optional symbol column for grouping

    Returns:
        DataFrame with rolling stats columns added
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    if symbol_column:
        grouped = df.groupby(symbol_column)[value_column]
    else:
        grouped = df[value_column]

    # Calculate rolling stats
    df[f'{value_column}_rolling_mean'] = grouped.transform(
        lambda x: x.rolling(window=window_days, min_periods=1).mean()
    )
    df[f'{value_column}_rolling_std'] = grouped.transform(
        lambda x: x.rolling(window=window_days, min_periods=1).std()
    )
    df[f'{value_column}_rolling_min'] = grouped.transform(
        lambda x: x.rolling(window=window_days, min_periods=1).min()
    )
    df[f'{value_column}_rolling_max'] = grouped.transform(
        lambda x: x.rolling(window=window_days, min_periods=1).max()
    )

    return df


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    return f"{value:,.{decimals}f}"


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    source_name: str = "DataFrame"
) -> List[str]:
    """
    Validate that required columns exist in DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        source_name: Name of data source for error messages

    Returns:
        List of missing columns (empty if all present)

    Raises:
        ValueError: If columns are missing
    """
    # Normalize column names to lowercase
    df_columns = [c.lower() for c in df.columns]
    required_lower = [c.lower() for c in required_columns]

    missing = [c for c in required_lower if c not in df_columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {source_name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return missing


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase and replace spaces with underscores.

    Args:
        df: DataFrame to normalize

    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    df.columns = [
        str(c).lower().strip().replace(' ', '_').replace('-', '_')
        for c in df.columns
    ]
    return df


def get_date_range(df: pd.DataFrame, date_column: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the date range of a DataFrame.

    Args:
        df: DataFrame with date column
        date_column: Name of date column

    Returns:
        Tuple of (min_date, max_date)
    """
    dates = pd.to_datetime(df[date_column])
    return dates.min(), dates.max()


def create_output_directory(path: Union[str, Path]) -> Path:
    """
    Create output directory if it doesn't exist.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with timestamp.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if prefix:
        return f"{prefix}_{timestamp}"
    return timestamp
