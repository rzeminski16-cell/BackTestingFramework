"""
Alpha Trend Indicator Explorer - Streamlit GUI

An interactive educational tool for understanding the AlphaTrend indicator.
This GUI enables exploration of:
- How the AlphaTrend line is calculated
- The role of each parameter
- How different market conditions affect the indicator
- Component breakdown (ATR bands, MFI, signals)
- Parameter sensitivity analysis

Usage:
    streamlit run apps/alphatrend_explorer.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AlphaTrend Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .explanation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .formula-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-family: monospace;
    }
    .insight-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .warning-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class DataSource(Enum):
    """Data source options."""
    SYNTHETIC = "Synthetic Data"
    FILE_UPLOAD = "Upload File"
    LOCAL_FILE = "Local File Path"


class MarketCondition(Enum):
    """Types of synthetic market conditions."""
    STRONG_UPTREND = "Strong Uptrend"
    STRONG_DOWNTREND = "Strong Downtrend"
    SIDEWAYS = "Sideways/Ranging"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"
    CHOPPY = "Choppy (Frequent Reversals)"
    BREAKOUT_UP = "Breakout (Upward)"
    BREAKOUT_DOWN = "Breakout (Downward)"
    TREND_REVERSAL = "Trend Reversal"
    CUSTOM = "Custom Parameters"


@dataclass
class AlphaTrendParams:
    """AlphaTrend indicator parameters."""
    atr_period: int = 14
    atr_multiplier: float = 1.0
    mfi_period: int = 14
    smoothing_length: int = 3
    percentile_period: int = 100


@dataclass
class SyntheticDataParams:
    """Parameters for synthetic data generation."""
    n_bars: int = 500
    initial_price: float = 100.0
    trend_strength: float = 0.0  # -1 to 1
    volatility: float = 0.02  # Daily volatility
    noise_level: float = 0.5  # 0 to 1
    volume_base: int = 1000000
    volume_variation: float = 0.3
    # Advanced realism parameters
    leverage_effect: float = 0.5  # How much negative returns increase future vol
    jump_probability: float = 0.02  # Probability of a jump (news event)
    jump_mean: float = 0.0  # Mean jump size
    jump_std: float = 0.03  # Jump size standard deviation
    return_autocorr: float = 0.05  # Short-term return autocorrelation
    volume_price_corr: float = 0.3  # Volume-price change correlation


# =============================================================================
# REAL DATA LOADING
# =============================================================================

def load_data_from_file(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load OHLCV data from a CSV file.

    Supports various column naming conventions and attempts to auto-detect format.

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of (DataFrame or None, error message or None)
    """
    try:
        # Read the file
        df = pd.read_csv(file_path)

        # Standardize column names
        df, error = standardize_columns(df)
        if error:
            return None, error

        # Validate data
        df, error = validate_ohlcv_data(df)
        if error:
            return None, error

        return df, None

    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except pd.errors.EmptyDataError:
        return None, "File is empty"
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def load_data_from_upload(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load OHLCV data from an uploaded file."""
    try:
        df = pd.read_csv(uploaded_file)
        df, error = standardize_columns(df)
        if error:
            return None, error
        df, error = validate_ohlcv_data(df)
        if error:
            return None, error
        return df, None
    except Exception as e:
        return None, f"Error reading uploaded file: {str(e)}"


def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Standardize column names to lowercase open, high, low, close, volume.

    Supports various naming conventions:
    - Standard: open, high, low, close, volume
    - Capitalized: Open, High, Low, Close, Volume
    - Abbreviated: o, h, l, c, v
    - With prefixes: adj_close, adjusted_close
    """
    df = df.copy()

    # Convert all column names to lowercase for matching
    col_map = {col: col.lower().strip() for col in df.columns}
    df.columns = [col_map[col] for col in df.columns]

    # Define possible mappings for each required column
    column_mappings = {
        'open': ['open', 'o', 'open_price', 'opening', 'first'],
        'high': ['high', 'h', 'high_price', 'highest', 'max'],
        'low': ['low', 'l', 'low_price', 'lowest', 'min'],
        'close': ['close', 'c', 'close_price', 'closing', 'last', 'adj_close',
                  'adjusted_close', 'adj close', 'adjclose'],
        'volume': ['volume', 'v', 'vol', 'turnover', 'qty', 'quantity']
    }

    # Try to find each required column
    final_mapping = {}
    for target, possibilities in column_mappings.items():
        found = False
        for col in df.columns:
            if col in possibilities:
                final_mapping[col] = target
                found = True
                break
        if not found:
            return df, f"Could not find '{target}' column. Available columns: {list(df.columns)}"

    # Rename columns
    df = df.rename(columns=final_mapping)

    # Handle date column if present
    date_possibilities = ['date', 'datetime', 'time', 'timestamp', 'dt', 'index']
    date_col = None
    for col in df.columns:
        if col in date_possibilities:
            date_col = col
            break

    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.index.name = 'date'
        except Exception:
            pass  # If date parsing fails, just continue with numeric index

    # If no date column and index is not datetime, create one
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        df = df.set_index('date')

    return df, None


def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Validate OHLCV data for consistency and quality."""
    required_cols = ['open', 'high', 'low', 'close', 'volume']

    # Check required columns exist
    for col in required_cols:
        if col not in df.columns:
            return df, f"Missing required column: {col}"

    # Convert to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for NaN values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        # Drop rows with NaN but warn
        original_len = len(df)
        df = df.dropna(subset=required_cols)
        dropped = original_len - len(df)
        if dropped > 0:
            st.warning(f"Dropped {dropped} rows with missing values")

    # Check minimum data requirement
    if len(df) < 100:
        return df, f"Need at least 100 bars of data, got {len(df)}"

    # Validate OHLC relationships
    invalid_rows = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    if invalid_rows.any():
        invalid_count = invalid_rows.sum()
        st.warning(f"Found {invalid_count} rows with invalid OHLC relationships (H<L, etc.). These may affect indicator accuracy.")

    # Check for zero or negative prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        return df, "Data contains zero or negative prices"

    # Check for zero volume (just warn)
    if (df['volume'] == 0).any():
        zero_vol = (df['volume'] == 0).sum()
        st.warning(f"Found {zero_vol} bars with zero volume")

    return df, None


def get_local_csv_files(directory: str = "raw_data") -> List[str]:
    """Get list of CSV files in the specified directory."""
    try:
        path = Path(directory)
        if not path.exists():
            return []
        return sorted([f.name for f in path.glob("*.csv")])
    except Exception:
        return []


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(
    condition: MarketCondition,
    params: SyntheticDataParams
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for different market conditions.

    Uses a more realistic model that includes:
    - GARCH-like volatility clustering (high volatility tends to follow high volatility)
    - Fat tails via Student-t distribution
    - Realistic OHLC relationships based on daily range dynamics
    - Volume correlation with price movement and volatility

    Args:
        condition: Type of market condition to simulate
        params: Data generation parameters

    Returns:
        DataFrame with OHLCV data
    """
    # Use a different seed based on condition for variety, but still reproducible
    seed = hash(condition.value) % 2**31
    np.random.seed(seed)
    n = params.n_bars

    # Base parameters based on market condition
    condition_params = _get_condition_params(condition, params)

    # Generate price series with GARCH-like volatility
    prices, volatilities = _generate_price_series_garch(n, condition_params, params)

    # Generate OHLCV from price series with realistic intraday dynamics
    df = _prices_to_ohlcv_realistic(prices, volatilities, params)

    # Add date index
    df['date'] = pd.date_range(start='2023-01-01', periods=n, freq='D')
    df = df.set_index('date')

    return df


def _get_condition_params(
    condition: MarketCondition,
    params: SyntheticDataParams
) -> Dict:
    """Get condition-specific parameters for market simulation."""
    base_params = {
        'trend': params.trend_strength,
        'volatility': params.volatility,
        'noise': params.noise_level,
        'regime_changes': 0,
        'breakout_bar': None,
        'reversal_bar': None,
        # GARCH parameters for volatility clustering
        'garch_alpha': 0.1,  # Impact of recent shocks
        'garch_beta': 0.85,  # Persistence of volatility
        # Fat tails - degrees of freedom for Student-t (lower = fatter tails)
        'df': 5,
        # Mean reversion strength (0 = none, higher = stronger)
        'mean_reversion': 0.0,
        'mean_price': params.initial_price
    }

    if condition == MarketCondition.STRONG_UPTREND:
        base_params['trend'] = 0.0015  # ~0.15% daily drift up
        base_params['volatility'] = 0.018
        base_params['noise'] = 0.2
        base_params['garch_alpha'] = 0.08
        base_params['garch_beta'] = 0.88
        base_params['df'] = 6

    elif condition == MarketCondition.STRONG_DOWNTREND:
        base_params['trend'] = -0.0015
        base_params['volatility'] = 0.022  # Downtrends typically more volatile
        base_params['noise'] = 0.2
        base_params['garch_alpha'] = 0.12  # More reactive in downtrends
        base_params['garch_beta'] = 0.85
        base_params['df'] = 4  # Fatter tails in downtrends

    elif condition == MarketCondition.SIDEWAYS:
        base_params['trend'] = 0.0
        base_params['volatility'] = 0.012
        base_params['noise'] = 0.4
        base_params['mean_reversion'] = 0.03  # Price tends to revert in ranging markets
        base_params['garch_alpha'] = 0.05
        base_params['garch_beta'] = 0.90
        base_params['df'] = 8

    elif condition == MarketCondition.HIGH_VOLATILITY:
        base_params['trend'] = 0.0003
        base_params['volatility'] = 0.04
        base_params['noise'] = 0.3
        base_params['garch_alpha'] = 0.15
        base_params['garch_beta'] = 0.80
        base_params['df'] = 3  # Very fat tails

    elif condition == MarketCondition.LOW_VOLATILITY:
        base_params['trend'] = 0.0005
        base_params['volatility'] = 0.006
        base_params['noise'] = 0.3
        base_params['garch_alpha'] = 0.03
        base_params['garch_beta'] = 0.95  # Very persistent low vol
        base_params['df'] = 10  # Near-normal distribution

    elif condition == MarketCondition.CHOPPY:
        base_params['trend'] = 0.0
        base_params['volatility'] = 0.025
        base_params['noise'] = 0.6
        base_params['regime_changes'] = 8
        base_params['garch_alpha'] = 0.12
        base_params['garch_beta'] = 0.75  # Less persistent - changes faster
        base_params['df'] = 5
        base_params['mean_reversion'] = 0.02

    elif condition == MarketCondition.BREAKOUT_UP:
        base_params['trend'] = 0.0
        base_params['volatility'] = 0.01
        base_params['noise'] = 0.25
        base_params['breakout_bar'] = params.n_bars // 2
        base_params['breakout_direction'] = 1
        base_params['garch_alpha'] = 0.08
        base_params['garch_beta'] = 0.88
        base_params['df'] = 6
        base_params['mean_reversion'] = 0.02  # Before breakout

    elif condition == MarketCondition.BREAKOUT_DOWN:
        base_params['trend'] = 0.0
        base_params['volatility'] = 0.01
        base_params['noise'] = 0.25
        base_params['breakout_bar'] = params.n_bars // 2
        base_params['breakout_direction'] = -1
        base_params['garch_alpha'] = 0.08
        base_params['garch_beta'] = 0.88
        base_params['df'] = 6
        base_params['mean_reversion'] = 0.02

    elif condition == MarketCondition.TREND_REVERSAL:
        base_params['trend'] = 0.0012
        base_params['volatility'] = 0.016
        base_params['noise'] = 0.25
        base_params['reversal_bar'] = params.n_bars // 2
        base_params['garch_alpha'] = 0.10
        base_params['garch_beta'] = 0.85
        base_params['df'] = 5

    return base_params


def _generate_price_series_garch(
    n: int,
    condition_params: Dict,
    data_params: SyntheticDataParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a price series with advanced realistic features:

    1. GARCH-like volatility clustering
    2. Leverage effect (negative returns increase future volatility more)
    3. Jump diffusion (occasional large moves from news/events)
    4. Return autocorrelation (short-term momentum)
    5. Fat tails via Student-t distribution

    Returns both prices and volatilities for use in OHLC generation.
    """
    prices = np.zeros(n)
    volatilities = np.zeros(n)
    returns = np.zeros(n)
    prices[0] = data_params.initial_price

    base_trend = condition_params['trend']
    base_vol = condition_params['volatility']
    noise = condition_params['noise']
    alpha = condition_params['garch_alpha']
    beta = condition_params['garch_beta']
    df = condition_params['df']
    mean_reversion = condition_params.get('mean_reversion', 0.0)
    mean_price = condition_params.get('mean_price', data_params.initial_price)

    # Advanced parameters from data_params
    leverage_effect = data_params.leverage_effect
    jump_prob = data_params.jump_probability
    jump_mean = data_params.jump_mean
    jump_std = data_params.jump_std
    return_autocorr = data_params.return_autocorr

    # Initialize volatility
    current_vol = base_vol
    volatilities[0] = current_vol
    last_shock = 0
    last_return = 0

    trend = base_trend

    for i in range(1, n):
        # Handle regime changes for choppy markets
        if condition_params.get('regime_changes', 0) > 0:
            regime_length = n // (condition_params['regime_changes'] + 1)
            if i % regime_length == 0:
                # Flip trend with some randomness
                trend = -trend * np.random.uniform(0.8, 1.2)

        # Handle breakout
        if condition_params.get('breakout_bar') and i >= condition_params['breakout_bar']:
            if i == condition_params['breakout_bar']:
                # Sharp move on breakout - this IS a jump
                direction = condition_params.get('breakout_direction', 1)
                gap = direction * base_vol * np.random.uniform(2.5, 4.0)
                prices[i] = prices[i-1] * (1 + gap)
                volatilities[i] = base_vol * 2.0  # Vol spike on breakout
                returns[i] = gap
                last_shock = gap
                last_return = gap
                continue
            elif i < condition_params['breakout_bar'] + 20:
                # Elevated trend and vol after breakout
                direction = condition_params.get('breakout_direction', 1)
                trend = direction * 0.003  # Strong trend
                base_vol_adjusted = base_vol * 1.8
            else:
                # Settle into new trend
                direction = condition_params.get('breakout_direction', 1)
                trend = direction * 0.0015
                base_vol_adjusted = base_vol * 1.3
        else:
            base_vol_adjusted = base_vol

        # Handle trend reversal
        if condition_params.get('reversal_bar') and i >= condition_params['reversal_bar']:
            bars_since_reversal = i - condition_params['reversal_bar']
            if bars_since_reversal < 10:
                # Transition period - higher vol, gradual trend change
                trend = base_trend * (1 - bars_since_reversal / 5)
                base_vol_adjusted = base_vol * 1.5
            else:
                trend = -abs(base_trend) * 1.2

        # === LEVERAGE EFFECT ===
        # Negative returns increase volatility more than positive returns
        # This captures the asymmetric volatility response observed in real markets
        leverage_adjustment = 1.0
        if last_return < 0:
            # Negative return: increase vol by leverage_effect * magnitude
            leverage_adjustment = 1.0 + leverage_effect * abs(last_return) / base_vol_adjusted
        leverage_adjustment = np.clip(leverage_adjustment, 1.0, 2.0)

        # === GARCH-like volatility update with leverage ===
        # sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        omega = base_vol_adjusted * (1 - alpha - beta)  # Long-run variance
        current_vol = np.sqrt(omega + alpha * (last_shock ** 2) + beta * (current_vol ** 2))
        current_vol = current_vol * leverage_adjustment
        current_vol = np.clip(current_vol, base_vol_adjusted * 0.3, base_vol_adjusted * 3.0)
        volatilities[i] = current_vol

        # === JUMP DIFFUSION ===
        # Model occasional large moves (earnings, news, etc.)
        jump = 0.0
        if np.random.random() < jump_prob:
            # Jump occurs - draw from normal with jump parameters
            jump = np.random.normal(jump_mean, jump_std)

        # === BASE SHOCK from Student-t for fat tails ===
        t_shock = np.random.standard_t(df)
        # Scale to have unit variance
        t_shock = t_shock / np.sqrt(df / (df - 2)) if df > 2 else t_shock

        # Add noise component
        noise_component = np.random.uniform(-1, 1) * noise

        # === RETURN AUTOCORRELATION ===
        # Add short-term momentum component
        momentum_component = return_autocorr * last_return

        # === MEAN REVERSION ===
        if mean_reversion > 0 and i > 0:
            log_deviation = np.log(prices[i-1] / mean_price)
            mean_rev_force = -mean_reversion * log_deviation
        else:
            mean_rev_force = 0

        # === COMBINE ALL COMPONENTS ===
        base_shock = current_vol * (t_shock * (1 - noise) + noise_component)
        daily_return = trend + mean_rev_force + momentum_component + base_shock + jump

        # Store for next iteration
        last_shock = base_shock
        last_return = daily_return
        returns[i] = daily_return

        prices[i] = prices[i-1] * (1 + daily_return)
        prices[i] = max(prices[i], 0.01)  # Prevent negative prices

    return prices, volatilities


def _prices_to_ohlcv_realistic(
    prices: np.ndarray,
    volatilities: np.ndarray,
    params: SyntheticDataParams
) -> pd.DataFrame:
    """
    Convert price series to OHLCV data with realistic intraday dynamics.

    Key improvements:
    - High/Low range proportional to volatility
    - Open gaps based on overnight news (occasional)
    - Close position within range reflects trend
    - Volume correlates with volatility and price moves
    - Intraday volatility pattern (U-shape: higher at open/close)
    - Volume-price correlation (configurable)
    """
    n = len(prices)

    # Calculate daily returns for close position bias
    returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices, 0.01)

    # Generate opens with occasional gaps
    gap_probability = 0.15  # 15% chance of noticeable gap
    gaps = np.where(
        np.random.random(n) < gap_probability,
        np.random.normal(0, volatilities * 0.5),
        np.random.normal(0, volatilities * 0.1)
    )
    opens = np.roll(prices, 1) * (1 + gaps)
    opens[0] = prices[0] * (1 + np.random.uniform(-0.002, 0.002))

    # High/Low range based on volatility
    # Average true range is typically 1-2x daily volatility
    range_multiplier = np.random.uniform(1.0, 2.0, n)
    daily_range = volatilities * range_multiplier

    # Determine close position within range (0 = at low, 1 = at high)
    # Bias towards high in uptrends, towards low in downtrends
    close_position = 0.5 + returns * 10  # Returns affect position
    close_position = np.clip(close_position + np.random.normal(0, 0.2, n), 0.1, 0.9)

    closes = prices

    # Calculate high and low
    # The range spans from some point below to some point above
    highs = np.maximum(opens, closes) + daily_range * close_position
    lows = np.minimum(opens, closes) - daily_range * (1 - close_position)

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    lows = np.maximum(lows, 0.001)  # Prevent negative

    # === ADVANCED VOLUME GENERATION ===
    abs_returns = np.abs(returns)
    vol_factor = volatilities / np.mean(volatilities)

    # Base volume from volatility
    vol_component = (vol_factor - 1) * 0.5

    # Price move component with configurable correlation
    # Higher correlation = volume more predictive of price moves
    price_move_component = abs_returns * 20 * params.volume_price_corr

    # Add volume clustering (volume tends to cluster like volatility)
    volume_persistence = np.zeros(n)
    volume_persistence[0] = 0
    for i in range(1, n):
        volume_persistence[i] = 0.7 * volume_persistence[i-1] + 0.3 * np.random.normal(0, 0.3)

    # Combine components
    volume_multiplier = (
        1.0 +
        vol_component +
        price_move_component +
        volume_persistence +
        np.random.uniform(-0.2, 0.2, n)  # Random variation
    )
    volume_multiplier = np.clip(volume_multiplier, 0.2, 4.0)

    volumes = params.volume_base * volume_multiplier
    volumes = volumes * (1 + np.random.uniform(-params.volume_variation, params.volume_variation, n))

    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes.astype(int)
    })


# =============================================================================
# ALPHATREND CALCULATION
# =============================================================================

def calculate_alphatrend_components(
    df: pd.DataFrame,
    params: AlphaTrendParams
) -> pd.DataFrame:
    """
    Calculate all AlphaTrend components with full breakdown.

    Args:
        df: OHLCV DataFrame
        params: AlphaTrend parameters

    Returns:
        DataFrame with all calculated components
    """
    result = df.copy()

    # === STEP 1: Calculate ATR ===
    result['tr'] = _calculate_true_range(result)
    result['atr'] = result['tr'].rolling(window=params.atr_period).mean()

    # === STEP 2: Calculate Adaptive Coefficient ===
    result['atr_ema_long'] = result['atr'].ewm(span=params.atr_period * 3, adjust=False).mean()
    result['volatility_ratio'] = result['atr'] / result['atr_ema_long']
    result['adaptive_coeff'] = params.atr_multiplier * result['volatility_ratio']

    # === STEP 3: Calculate AlphaTrend Bands ===
    result['up_band'] = result['low'] - result['atr'] * result['adaptive_coeff']
    result['down_band'] = result['high'] + result['atr'] * result['adaptive_coeff']

    # === STEP 4: Calculate Money Flow Index (MFI) ===
    result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
    result['raw_money_flow'] = result['typical_price'] * result['volume']

    result['price_change'] = result['typical_price'].diff()
    result['positive_flow'] = np.where(result['price_change'] > 0, result['raw_money_flow'], 0)
    result['negative_flow'] = np.where(result['price_change'] < 0, result['raw_money_flow'], 0)

    positive_mf = result['positive_flow'].rolling(window=params.mfi_period).sum()
    negative_mf = result['negative_flow'].rolling(window=params.mfi_period).sum()

    mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
    result['mfi'] = 100 - (100 / (1 + mfi_ratio))

    # === STEP 5: Calculate Dynamic MFI Thresholds ===
    result['mfi_upper'] = result['mfi'].rolling(window=params.percentile_period).quantile(0.70)
    result['mfi_lower'] = result['mfi'].rolling(window=params.percentile_period).quantile(0.30)
    result['mfi_threshold'] = (result['mfi_upper'] + result['mfi_lower']) / 2

    # === STEP 6: Determine Momentum Direction ===
    # Fill NaN with False for momentum_bullish to handle early bars
    result['momentum_bullish'] = (result['mfi'] >= result['mfi_threshold']).fillna(False)

    # === STEP 7: Calculate AlphaTrend Line ===
    alphatrend_values = _calculate_alphatrend_line(
        result['up_band'].values,
        result['down_band'].values,
        result['momentum_bullish'].values
    )
    result['alphatrend'] = alphatrend_values  # Direct numpy array assignment

    # === STEP 8: Smooth AlphaTrend ===
    result['smooth_at'] = pd.Series(result['alphatrend']).ewm(
        span=params.smoothing_length, adjust=False
    ).mean().values

    # === STEP 9: Generate Signals ===
    # Use numpy arrays for crossover detection to avoid index issues
    at_values = result['alphatrend'].values if hasattr(result['alphatrend'], 'values') else result['alphatrend']
    smooth_values = result['smooth_at'].values if hasattr(result['smooth_at'], 'values') else result['smooth_at']

    # Crossover: current AT > smooth AND previous AT <= previous smooth
    cross_up = np.zeros(len(at_values), dtype=bool)
    cross_down = np.zeros(len(at_values), dtype=bool)

    for i in range(1, len(at_values)):
        # Cross up: AT crosses above smooth_at
        cross_up[i] = (at_values[i] > smooth_values[i]) and (at_values[i-1] <= smooth_values[i-1])
        # Cross down: AT crosses below smooth_at
        cross_down[i] = (at_values[i] < smooth_values[i]) and (at_values[i-1] >= smooth_values[i-1])

    result['cross_up'] = cross_up
    result['cross_down'] = cross_down

    # Filter for alternating signals
    buy_signals, sell_signals = _filter_alternating_signals(cross_up, cross_down)
    result['buy_signal'] = buy_signals
    result['sell_signal'] = sell_signals

    # === STEP 10: Trend State ===
    result['trend_state'] = np.where(
        at_values > smooth_values,
        'Bullish',
        np.where(at_values < smooth_values, 'Bearish', 'Neutral')
    )

    return result


def _calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr


def _calculate_alphatrend_line(
    up_band: np.ndarray,
    down_band: np.ndarray,
    momentum_bullish: np.ndarray
) -> np.ndarray:
    """
    Calculate the AlphaTrend line using state-dependent logic.

    The key insight: AlphaTrend acts as a trailing stop that:
    - In uptrend: Can only rise or stay flat (never falls)
    - In downtrend: Can only fall or stay flat (never rises)

    This creates the characteristic "staircase" pattern.

    Returns numpy array (not Series) to avoid index alignment issues.
    """
    n = len(up_band)
    alphatrend = np.zeros(n)

    # Find first valid value for initialization
    first_valid = 0
    for i in range(n):
        if not (np.isnan(up_band[i]) or np.isnan(down_band[i])):
            first_valid = i
            break

    # Initialize with first valid up_band or a reasonable default
    if first_valid < n and not np.isnan(up_band[first_valid]):
        alphatrend[0:first_valid+1] = up_band[first_valid]
    else:
        alphatrend[0] = 100.0  # Fallback

    for i in range(first_valid + 1, n):
        # Handle NaN in bands
        if np.isnan(up_band[i]) or np.isnan(down_band[i]):
            alphatrend[i] = alphatrend[i-1]
            continue

        # Handle NaN in momentum - default to previous state behavior
        # np.isnan on bool array returns False, but momentum_bullish might have NaN
        # if it was computed from NaN threshold values
        is_bullish = momentum_bullish[i]

        # Check if is_bullish is actually a valid boolean
        # (NaN comparisons return False, so NaN >= threshold = False)
        if isinstance(is_bullish, (bool, np.bool_)):
            if is_bullish:
                # Uptrend: AlphaTrend follows up_band but can only rise
                alphatrend[i] = max(alphatrend[i-1], up_band[i])
            else:
                # Downtrend: AlphaTrend follows down_band but can only fall
                alphatrend[i] = min(alphatrend[i-1], down_band[i])
        else:
            # If momentum is somehow not boolean (shouldn't happen), keep previous
            alphatrend[i] = alphatrend[i-1]

    return alphatrend


def _filter_alternating_signals(
    cross_up: np.ndarray,
    cross_down: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter signals to ensure alternating buy/sell pattern.

    Returns numpy arrays (not Series) to avoid index alignment issues.
    """
    n = len(cross_up)
    filtered_buy = np.zeros(n, dtype=bool)
    filtered_sell = np.zeros(n, dtype=bool)

    last_signal = 0  # 0=none, 1=buy, 2=sell

    for i in range(n):
        # Handle potential NaN or non-boolean values
        is_cross_up = bool(cross_up[i]) if not pd.isna(cross_up[i]) else False
        is_cross_down = bool(cross_down[i]) if not pd.isna(cross_down[i]) else False

        if is_cross_up and last_signal != 1:
            filtered_buy[i] = True
            last_signal = 1
        elif is_cross_down and last_signal != 2:
            filtered_sell[i] = True
            last_signal = 2

    return filtered_buy, filtered_sell


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_main_chart(
    df: pd.DataFrame,
    show_bands: bool = True,
    show_signals: bool = True,
    show_smooth: bool = True
) -> go.Figure:
    """Create the main AlphaTrend visualization chart."""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & AlphaTrend', 'Money Flow Index (MFI)', 'Volume')
    )

    # === Row 1: Price and AlphaTrend ===

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # AlphaTrend bands
    if show_bands and 'up_band' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['up_band'],
                mode='lines',
                name='Up Band',
                line=dict(color='rgba(76, 175, 80, 0.3)', width=1, dash='dot'),
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['down_band'],
                mode='lines',
                name='Down Band',
                line=dict(color='rgba(244, 67, 54, 0.3)', width=1, dash='dot'),
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # AlphaTrend line
    if 'alphatrend' in df.columns:
        # Color based on trend state
        colors = ['#26a69a' if state == 'Bullish' else '#ef5350'
                  for state in df['trend_state']]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['alphatrend'],
                mode='lines',
                name='AlphaTrend',
                line=dict(color='#1976d2', width=2),
            ),
            row=1, col=1
        )

    # Smoothed AlphaTrend
    if show_smooth and 'smooth_at' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['smooth_at'],
                mode='lines',
                name='Smoothed AT',
                line=dict(color='#ff9800', width=1.5, dash='dash'),
            ),
            row=1, col=1
        )

    # Buy/Sell signals
    if show_signals and 'buy_signal' in df.columns:
        buy_signals = df[df['buy_signal'] == True]
        sell_signals = df[df['sell_signal'] == True]

        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['low'] * 0.99,
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#4caf50',
                    line=dict(width=2, color='white')
                )
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['high'] * 1.01,
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#f44336',
                    line=dict(width=2, color='white')
                )
            ),
            row=1, col=1
        )

    # === Row 2: MFI ===
    if 'mfi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['mfi'],
                mode='lines',
                name='MFI',
                line=dict(color='#9c27b0', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(156, 39, 176, 0.1)'
            ),
            row=2, col=1
        )

        # MFI threshold
        if 'mfi_threshold' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['mfi_threshold'],
                    mode='lines',
                    name='MFI Threshold',
                    line=dict(color='#ff5722', width=1, dash='dash')
                ),
                row=2, col=1
            )

        # Reference lines
        fig.add_hline(y=70, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    # === Row 3: Volume ===
    colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350'
              for i in range(len(df))]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=3, col=1
    )

    # Layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MFI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)

    return fig


def create_component_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Create a detailed breakdown of AlphaTrend components."""

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.3, 0.25, 0.25, 0.2],
        subplot_titles=(
            'ATR & Adaptive Coefficient',
            'AlphaTrend Bands vs Price',
            'MFI with Dynamic Thresholds',
            'Momentum State'
        )
    )

    # === Row 1: ATR Analysis ===
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['atr'],
            mode='lines',
            name='ATR',
            line=dict(color='#2196f3', width=1.5)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['atr_ema_long'],
            mode='lines',
            name='ATR EMA (Long)',
            line=dict(color='#ff9800', width=1, dash='dash')
        ),
        row=1, col=1
    )

    # Adaptive coefficient on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['adaptive_coeff'],
            mode='lines',
            name='Adaptive Coeff',
            line=dict(color='#4caf50', width=1.5),
            yaxis='y2'
        ),
        row=1, col=1
    )

    # === Row 2: Bands vs Price ===
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Close',
            line=dict(color='#607d8b', width=1)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['up_band'],
            mode='lines',
            name='Up Band',
            line=dict(color='#4caf50', width=1.5),
            fill=None
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['down_band'],
            mode='lines',
            name='Down Band',
            line=dict(color='#f44336', width=1.5),
            fill='tonexty',
            fillcolor='rgba(156, 39, 176, 0.1)'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['alphatrend'],
            mode='lines',
            name='AlphaTrend',
            line=dict(color='#1976d2', width=2)
        ),
        row=2, col=1
    )

    # === Row 3: MFI Detail ===
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['mfi'],
            mode='lines',
            name='MFI',
            line=dict(color='#9c27b0', width=1.5)
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['mfi_upper'],
            mode='lines',
            name='MFI Upper (70th %ile)',
            line=dict(color='#4caf50', width=1, dash='dot')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['mfi_lower'],
            mode='lines',
            name='MFI Lower (30th %ile)',
            line=dict(color='#f44336', width=1, dash='dot')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['mfi_threshold'],
            mode='lines',
            name='MFI Threshold',
            line=dict(color='#ff9800', width=1.5, dash='dash')
        ),
        row=3, col=1
    )

    # === Row 4: Momentum State ===
    momentum_colors = ['#4caf50' if bull else '#f44336' for bull in df['momentum_bullish']]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=[1 if bull else -1 for bull in df['momentum_bullish']],
            name='Momentum',
            marker_color=momentum_colors,
            opacity=0.7
        ),
        row=4, col=1
    )

    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )

    return fig


def create_parameter_sensitivity_chart(
    base_df: pd.DataFrame,
    param_name: str,
    param_values: list,
    base_params: AlphaTrendParams
) -> go.Figure:
    """Create a chart showing how different parameter values affect AlphaTrend."""

    fig = go.Figure()

    # Add price for reference
    fig.add_trace(
        go.Scatter(
            x=base_df.index,
            y=base_df['close'],
            mode='lines',
            name='Price',
            line=dict(color='gray', width=1),
            opacity=0.5
        )
    )

    # Calculate AlphaTrend for each parameter value
    colors = ['#1976d2', '#4caf50', '#ff9800', '#f44336', '#9c27b0']

    for i, value in enumerate(param_values):
        # Create modified params
        test_params = AlphaTrendParams(
            atr_period=base_params.atr_period,
            atr_multiplier=base_params.atr_multiplier,
            mfi_period=base_params.mfi_period,
            smoothing_length=base_params.smoothing_length,
            percentile_period=base_params.percentile_period
        )
        setattr(test_params, param_name, value)

        # Calculate
        result = calculate_alphatrend_components(base_df[['open', 'high', 'low', 'close', 'volume']], test_params)

        fig.add_trace(
            go.Scatter(
                x=result.index,
                y=result['alphatrend'],
                mode='lines',
                name=f'{param_name}={value}',
                line=dict(color=colors[i % len(colors)], width=2)
            )
        )

    fig.update_layout(
        title=f'AlphaTrend Sensitivity to {param_name}',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_signal_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart analyzing signal quality."""

    # Calculate signal statistics
    buy_signals = df[df['buy_signal'] == True]
    sell_signals = df[df['sell_signal'] == True]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Signal Distribution Over Time',
            'Price at Signal Points',
            'MFI at Signal Points',
            'Trend State Transitions'
        )
    )

    # Signal timeline
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=[1] * len(buy_signals),
            mode='markers',
            name='Buy',
            marker=dict(color='#4caf50', size=10, symbol='triangle-up')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=[-1] * len(sell_signals),
            mode='markers',
            name='Sell',
            marker=dict(color='#f44336', size=10, symbol='triangle-down')
        ),
        row=1, col=1
    )

    # Price distribution at signals
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Histogram(
                x=buy_signals['close'],
                name='Buy Price',
                marker_color='#4caf50',
                opacity=0.7
            ),
            row=1, col=2
        )

    if len(sell_signals) > 0:
        fig.add_trace(
            go.Histogram(
                x=sell_signals['close'],
                name='Sell Price',
                marker_color='#f44336',
                opacity=0.7
            ),
            row=1, col=2
        )

    # MFI at signals
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Box(
                y=buy_signals['mfi'],
                name='MFI at Buy',
                marker_color='#4caf50'
            ),
            row=2, col=1
        )

    if len(sell_signals) > 0:
        fig.add_trace(
            go.Box(
                y=sell_signals['mfi'],
                name='MFI at Sell',
                marker_color='#f44336'
            ),
            row=2, col=1
        )

    # Trend state over time
    state_numeric = [1 if s == 'Bullish' else -1 for s in df['trend_state']]
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=state_numeric,
            mode='lines',
            name='Trend State',
            fill='tozeroy',
            line=dict(color='#1976d2', width=1)
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='closest'
    )

    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<p class="main-header">AlphaTrend Indicator Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An interactive tool to understand how the AlphaTrend indicator works</p>', unsafe_allow_html=True)

    # Sidebar - Parameters
    st.sidebar.title("Data & Parameters")

    # === DATA SOURCE SELECTION ===
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        options=[ds.value for ds in DataSource],
        index=0,
        help="Choose between synthetic data or real market data"
    )
    data_source_enum = DataSource(data_source)

    # Initialize variables
    df = None
    data_loaded = False
    data_info = ""

    if data_source_enum == DataSource.SYNTHETIC:
        # === SYNTHETIC DATA OPTIONS ===
        st.sidebar.subheader("Synthetic Data")
        market_condition = st.sidebar.selectbox(
            "Market Condition",
            options=[c.value for c in MarketCondition],
            index=0,
            help="Select the type of market condition to simulate"
        )
        condition = MarketCondition(market_condition)

        # Basic parameters
        st.sidebar.subheader("Data Parameters")
        n_bars = st.sidebar.slider("Number of Bars", 100, 1000, 500, 50)
        initial_price = st.sidebar.number_input("Initial Price", 10.0, 1000.0, 100.0, 10.0)

        if condition == MarketCondition.CUSTOM:
            trend_strength = st.sidebar.slider("Trend Strength", -0.003, 0.003, 0.0, 0.0001, format="%.4f")
            volatility = st.sidebar.slider("Volatility", 0.005, 0.05, 0.02, 0.001, format="%.3f")
            noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.5, 0.1)
        else:
            trend_strength = 0.0
            volatility = 0.02
            noise_level = 0.5

        # Advanced parameters in expander
        with st.sidebar.expander("Advanced Realism Settings", expanded=False):
            st.caption("These parameters make synthetic data more realistic")

            leverage_effect = st.slider(
                "Leverage Effect",
                0.0, 1.0, 0.5, 0.1,
                help="How much negative returns increase future volatility (asymmetric vol)"
            )

            jump_probability = st.slider(
                "Jump Probability",
                0.0, 0.1, 0.02, 0.005,
                format="%.3f",
                help="Probability of a large jump (news/earnings event) per bar"
            )

            jump_std = st.slider(
                "Jump Size (Std Dev)",
                0.01, 0.10, 0.03, 0.01,
                help="Standard deviation of jump sizes when they occur"
            )

            return_autocorr = st.slider(
                "Return Autocorrelation",
                -0.2, 0.2, 0.05, 0.01,
                help="Short-term momentum in returns (positive = trending, negative = mean-reverting)"
            )

            volume_price_corr = st.slider(
                "Volume-Price Correlation",
                0.0, 1.0, 0.3, 0.1,
                help="How much volume correlates with price movement magnitude"
            )

        data_params = SyntheticDataParams(
            n_bars=n_bars,
            initial_price=initial_price,
            trend_strength=trend_strength,
            volatility=volatility,
            noise_level=noise_level,
            leverage_effect=leverage_effect,
            jump_probability=jump_probability,
            jump_mean=0.0,
            jump_std=jump_std,
            return_autocorr=return_autocorr,
            volume_price_corr=volume_price_corr
        )

        data_info = f"**{market_condition}** - {n_bars} bars of synthetic data"
        data_loaded = True

    elif data_source_enum == DataSource.FILE_UPLOAD:
        # === FILE UPLOAD ===
        st.sidebar.subheader("Upload Data File")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with OHLCV columns (open, high, low, close, volume)"
        )

        if uploaded_file is not None:
            df, error = load_data_from_upload(uploaded_file)
            if error:
                st.sidebar.error(error)
            else:
                data_loaded = True
                data_info = f"**Uploaded File:** {uploaded_file.name} - {len(df)} bars"
                st.sidebar.success(f"Loaded {len(df)} bars")

        st.sidebar.markdown("""
        **Expected format:**
        - CSV with columns: date, open, high, low, close, volume
        - Various column name formats are supported
        """)

    else:  # LOCAL_FILE
        # === LOCAL FILE PATH ===
        st.sidebar.subheader("Local File")

        # Show available files in raw_data directory
        raw_data_dir = "raw_data"
        available_files = get_local_csv_files(raw_data_dir)

        if available_files:
            st.sidebar.caption(f"Files in `{raw_data_dir}/`:")
            selected_file = st.sidebar.selectbox(
                "Select a file",
                options=[""] + available_files,
                index=0
            )

            if selected_file:
                file_path = os.path.join(raw_data_dir, selected_file)
                df, error = load_data_from_file(file_path)
                if error:
                    st.sidebar.error(error)
                else:
                    data_loaded = True
                    data_info = f"**Local File:** {selected_file} - {len(df)} bars"
                    st.sidebar.success(f"Loaded {len(df)} bars")

        # Also allow custom path input
        custom_path = st.sidebar.text_input(
            "Or enter custom file path",
            placeholder="/path/to/your/data.csv"
        )

        if custom_path:
            df, error = load_data_from_file(custom_path)
            if error:
                st.sidebar.error(error)
            else:
                data_loaded = True
                data_info = f"**Custom Path:** {os.path.basename(custom_path)} - {len(df)} bars"
                st.sidebar.success(f"Loaded {len(df)} bars")

    # Generate synthetic data if that's the source
    if data_source_enum == DataSource.SYNTHETIC and data_loaded:
        with st.spinner("Generating synthetic data..."):
            df = generate_synthetic_data(condition, data_params)

    # Check if we have data to work with
    if not data_loaded or df is None:
        st.warning("Please select a data source and load data to continue.")
        st.info("""
        **Options:**
        1. **Synthetic Data**: Generate realistic market data with various conditions
        2. **Upload File**: Upload your own CSV file with OHLCV data
        3. **Local File**: Select a CSV file from the raw_data directory or enter a custom path
        """)
        return

    # AlphaTrend Parameters
    st.sidebar.subheader("AlphaTrend Parameters")

    atr_period = st.sidebar.slider(
        "ATR Period",
        5, 30, 14, 1,
        help="Period for Average True Range calculation. Longer periods = smoother ATR."
    )

    atr_multiplier = st.sidebar.slider(
        "ATR Multiplier",
        0.5, 3.0, 1.0, 0.1,
        help="Multiplier for ATR bands. Higher = wider bands, fewer signals."
    )

    mfi_period = st.sidebar.slider(
        "MFI Period",
        7, 21, 14, 1,
        help="Period for Money Flow Index calculation."
    )

    smoothing_length = st.sidebar.slider(
        "Smoothing Length",
        1, 10, 3, 1,
        help="EMA period for smoothing the AlphaTrend line."
    )

    percentile_period = st.sidebar.slider(
        "Percentile Period",
        50, 200, 100, 10,
        help="Lookback period for dynamic MFI thresholds."
    )

    at_params = AlphaTrendParams(
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        mfi_period=mfi_period,
        smoothing_length=smoothing_length,
        percentile_period=percentile_period
    )

    # Display options
    st.sidebar.subheader("Display Options")
    show_bands = st.sidebar.checkbox("Show ATR Bands", True)
    show_signals = st.sidebar.checkbox("Show Buy/Sell Signals", True)
    show_smooth = st.sidebar.checkbox("Show Smoothed AlphaTrend", True)

    # Calculate AlphaTrend components on the loaded data
    with st.spinner("Calculating AlphaTrend indicators..."):
        df = calculate_alphatrend_components(df, at_params)

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Main Chart",
        "Component Breakdown",
        "Parameter Sensitivity",
        "Signal Analysis"
    ])

    # === TAB 1: Overview ===
    with tab1:
        st.header("What is AlphaTrend?")

        st.markdown("""
        <div class="explanation-box">
        <b>AlphaTrend</b> is a trend-following indicator that combines volatility (ATR) with
        momentum (MFI) to create an adaptive support/resistance line that:
        <ul>
        <li>Acts as a <b>trailing stop</b> that only moves in the trend direction</li>
        <li>Uses <b>adaptive bands</b> that widen in high volatility and narrow in low volatility</li>
        <li>Employs <b>dynamic thresholds</b> based on recent market conditions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_signals = df['buy_signal'].sum() + df['sell_signal'].sum()
            st.metric("Total Signals", int(total_signals))

        with col2:
            buy_signals = df['buy_signal'].sum()
            st.metric("Buy Signals", int(buy_signals))

        with col3:
            sell_signals = df['sell_signal'].sum()
            st.metric("Sell Signals", int(sell_signals))

        with col4:
            bullish_pct = (df['trend_state'] == 'Bullish').mean() * 100
            st.metric("Time in Uptrend", f"{bullish_pct:.1f}%")

        st.subheader("How AlphaTrend Works")

        st.markdown("""
        <div class="formula-box">
        <b>Step 1: Calculate ATR Bands</b><br>
        <code>up_band = low - (ATR × adaptive_coefficient)</code><br>
        <code>down_band = high + (ATR × adaptive_coefficient)</code><br><br>

        <b>Step 2: Determine Momentum (via MFI)</b><br>
        <code>momentum_bullish = MFI >= dynamic_threshold</code><br><br>

        <b>Step 3: Calculate AlphaTrend Line</b><br>
        <code>IF momentum_bullish:</code><br>
        <code>    alphatrend = MAX(previous_alphatrend, up_band)</code><br>
        <code>ELSE:</code><br>
        <code>    alphatrend = MIN(previous_alphatrend, down_band)</code><br><br>

        <b>Step 4: Generate Signals</b><br>
        <code>BUY = AlphaTrend crosses above Smoothed AlphaTrend</code><br>
        <code>SELL = AlphaTrend crosses below Smoothed AlphaTrend</code>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Key Insight")

        st.markdown("""
        <div class="insight-box">
        <b>The "Ratchet" Effect:</b> The AlphaTrend line can only move in the direction of the
        current momentum. In an uptrend, it acts like a rising floor that cannot fall. In a
        downtrend, it acts like a falling ceiling that cannot rise. This creates the characteristic
        "staircase" pattern you'll see in the chart.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Current Data")
        st.info(data_info)

        # Show quick stats
        st.subheader("Data Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Price Statistics**")
            st.dataframe(df[['open', 'high', 'low', 'close']].describe().round(2))

        with col2:
            st.write("**Indicator Statistics**")
            indicator_cols = ['atr', 'mfi', 'alphatrend', 'adaptive_coeff']
            st.dataframe(df[indicator_cols].describe().round(2))

    # === TAB 2: Main Chart ===
    with tab2:
        st.header("AlphaTrend Visualization")

        fig = create_main_chart(df, show_bands, show_signals, show_smooth)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explanation-box">
        <b>Reading the Chart:</b>
        <ul>
        <li><b>Blue Line (AlphaTrend):</b> The main indicator line - notice how it only moves in one direction at a time</li>
        <li><b>Orange Dashed Line (Smoothed AT):</b> Smoothed version used for signal generation</li>
        <li><b>Green/Red Triangles:</b> Buy and Sell signals when AlphaTrend crosses its smoothed version</li>
        <li><b>Dotted Lines (Bands):</b> The ATR-based bands that define potential support/resistance levels</li>
        <li><b>Purple Area (MFI):</b> Money Flow Index - determines momentum direction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # === TAB 3: Component Breakdown ===
    with tab3:
        st.header("Component Breakdown")

        st.markdown("""
        This view shows each component of the AlphaTrend calculation separately,
        helping you understand how they combine to form the final indicator.
        """)

        fig = create_component_breakdown_chart(df)
        st.plotly_chart(fig, use_container_width=True)

        # Component explanations
        st.subheader("Component Explanations")

        with st.expander("ATR & Adaptive Coefficient", expanded=True):
            st.markdown("""
            **Average True Range (ATR)** measures volatility by looking at the range of price movement.

            The **Adaptive Coefficient** adjusts based on current volatility vs. average volatility:
            - When ATR > Average ATR: Coefficient > 1.0 (wider bands)
            - When ATR < Average ATR: Coefficient < 1.0 (tighter bands)

            This makes the indicator more responsive in volatile markets and more stable in calm markets.
            """)

        with st.expander("AlphaTrend Bands"):
            st.markdown("""
            The bands define the boundaries for the AlphaTrend line:

            - **Up Band** = Low - (ATR × Coefficient): Support level in uptrend
            - **Down Band** = High + (ATR × Coefficient): Resistance level in downtrend

            The AlphaTrend line follows these bands but with a crucial constraint:
            it can only move toward the current trend direction.
            """)

        with st.expander("Money Flow Index (MFI)"):
            st.markdown("""
            MFI is a volume-weighted momentum indicator (like RSI but with volume):

            - **MFI > Threshold**: Momentum is bullish, AlphaTrend follows Up Band
            - **MFI < Threshold**: Momentum is bearish, AlphaTrend follows Down Band

            The threshold is **dynamic**, calculated as the average of the 30th and 70th
            percentiles over the lookback period. This adapts to the current market regime.
            """)

        with st.expander("Momentum State"):
            st.markdown("""
            The bottom panel shows the momentum state over time:

            - **Green (+1)**: Bullish momentum (MFI above threshold)
            - **Red (-1)**: Bearish momentum (MFI below threshold)

            Notice how the AlphaTrend line behavior changes with momentum state.
            """)

    # === TAB 4: Parameter Sensitivity ===
    with tab4:
        st.header("Parameter Sensitivity Analysis")

        st.markdown("""
        Explore how different parameter values affect the AlphaTrend indicator.
        Select a parameter to see how varying its value changes the indicator behavior.
        """)

        param_to_test = st.selectbox(
            "Parameter to Analyze",
            ["atr_period", "atr_multiplier", "mfi_period", "smoothing_length", "percentile_period"]
        )

        # Define test values for each parameter
        test_values = {
            "atr_period": [7, 14, 21, 28],
            "atr_multiplier": [0.5, 1.0, 1.5, 2.0],
            "mfi_period": [7, 10, 14, 21],
            "smoothing_length": [1, 3, 5, 7],
            "percentile_period": [50, 100, 150, 200]
        }

        fig = create_parameter_sensitivity_chart(
            df[['open', 'high', 'low', 'close', 'volume']],
            param_to_test,
            test_values[param_to_test],
            at_params
        )
        st.plotly_chart(fig, use_container_width=True)

        # Parameter impact explanations
        param_explanations = {
            "atr_period": """
            **ATR Period** controls how responsive the volatility measurement is:
            - **Shorter periods (7-10)**: More reactive to recent volatility, more signals
            - **Longer periods (20-30)**: Smoother, fewer false signals in choppy markets
            """,
            "atr_multiplier": """
            **ATR Multiplier** controls the width of the bands:
            - **Lower values (0.5-0.8)**: Tighter bands, more signals, earlier entries/exits
            - **Higher values (1.5-2.0)**: Wider bands, fewer signals, avoids noise
            """,
            "mfi_period": """
            **MFI Period** affects momentum calculation sensitivity:
            - **Shorter periods (7-10)**: Faster momentum shifts, more whipsaws
            - **Longer periods (14-21)**: Smoother momentum, misses quick reversals
            """,
            "smoothing_length": """
            **Smoothing Length** affects signal generation:
            - **Shorter (1-2)**: Faster signals, more trades, potential whipsaws
            - **Longer (5-7)**: Slower signals, fewer trades, might miss moves
            """,
            "percentile_period": """
            **Percentile Period** controls dynamic threshold adaptation:
            - **Shorter (50-75)**: Thresholds adapt quickly to recent conditions
            - **Longer (150-200)**: Thresholds based on longer-term market behavior
            """
        }

        st.markdown(f"""
        <div class="explanation-box">
        {param_explanations[param_to_test]}
        </div>
        """, unsafe_allow_html=True)

        # Signal count comparison
        st.subheader("Signal Count Comparison")

        signal_counts = []
        for value in test_values[param_to_test]:
            test_params = AlphaTrendParams(
                atr_period=at_params.atr_period,
                atr_multiplier=at_params.atr_multiplier,
                mfi_period=at_params.mfi_period,
                smoothing_length=at_params.smoothing_length,
                percentile_period=at_params.percentile_period
            )
            setattr(test_params, param_to_test, value)

            result = calculate_alphatrend_components(df[['open', 'high', 'low', 'close', 'volume']], test_params)
            total = result['buy_signal'].sum() + result['sell_signal'].sum()
            signal_counts.append({
                'Parameter Value': value,
                'Buy Signals': int(result['buy_signal'].sum()),
                'Sell Signals': int(result['sell_signal'].sum()),
                'Total Signals': int(total)
            })

        st.dataframe(pd.DataFrame(signal_counts), use_container_width=True)

    # === TAB 5: Signal Analysis ===
    with tab5:
        st.header("Signal Analysis")

        fig = create_signal_analysis_chart(df)
        st.plotly_chart(fig, use_container_width=True)

        # Signal statistics
        st.subheader("Signal Statistics")

        buy_signals_df = df[df['buy_signal'] == True]
        sell_signals_df = df[df['sell_signal'] == True]

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Buy Signal Statistics**")
            if len(buy_signals_df) > 0:
                st.write(f"- Count: {len(buy_signals_df)}")
                st.write(f"- Avg Price: ${buy_signals_df['close'].mean():.2f}")
                st.write(f"- Avg MFI: {buy_signals_df['mfi'].mean():.1f}")
                st.write(f"- Avg ATR: {buy_signals_df['atr'].mean():.4f}")
            else:
                st.write("No buy signals generated")

        with col2:
            st.write("**Sell Signal Statistics**")
            if len(sell_signals_df) > 0:
                st.write(f"- Count: {len(sell_signals_df)}")
                st.write(f"- Avg Price: ${sell_signals_df['close'].mean():.2f}")
                st.write(f"- Avg MFI: {sell_signals_df['mfi'].mean():.1f}")
                st.write(f"- Avg ATR: {sell_signals_df['atr'].mean():.4f}")
            else:
                st.write("No sell signals generated")

        # Signal timing analysis
        st.subheader("Signal Timing")

        if len(buy_signals_df) > 1:
            # Calculate bars between signals
            buy_indices = buy_signals_df.index.tolist()
            bars_between_buys = []
            for i in range(1, len(buy_indices)):
                days = (buy_indices[i] - buy_indices[i-1]).days
                bars_between_buys.append(days)

            if bars_between_buys:
                st.write(f"**Average bars between buy signals:** {np.mean(bars_between_buys):.1f}")
                st.write(f"**Min/Max bars between buys:** {min(bars_between_buys)} / {max(bars_between_buys)}")

        # Export data option
        st.subheader("Export Data")

        if st.button("Download Full Dataset (CSV)"):
            csv = df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="alphatrend_data.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    AlphaTrend Explorer | Part of the BackTesting Framework<br>
    Adjust parameters in the sidebar to explore different scenarios
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
