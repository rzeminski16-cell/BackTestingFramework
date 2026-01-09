"""
Data Aggregation Module for Per-Trade Analysis

Handles loading, validation, and aggregation of trade logs and supporting data
including price action, fundamentals, insider activity, and options data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Required columns for trade log validation
REQUIRED_TRADE_COLUMNS = [
    'trade_id', 'symbol', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
    'quantity', 'side', 'pl', 'pl_pct', 'duration_days', 'entry_reason', 'exit_reason'
]

# Optional but useful columns
OPTIONAL_TRADE_COLUMNS = [
    'initial_stop_loss', 'final_stop_loss', 'take_profit', 'security_pl', 'fx_pl',
    'entry_fx_rate', 'exit_fx_rate', 'security_currency', 'entry_equity',
    'entry_capital_available', 'entry_capital_required', 'concurrent_positions',
    'competing_signals', 'commission_paid', 'partial_exits'
]

# Alpha Vantage indicator column mappings (actual -> simplified)
INDICATOR_MAPPINGS = {
    'atr_14_atr': 'atr',
    'rsi_14_rsi': 'rsi',
    'sma_20_sma': 'sma_20',
    'sma_50_sma': 'sma_50',
    'sma_200_sma': 'sma_200',
    'bbands_20_real lower band': 'bb_lower',
    'bbands_20_real middle band': 'bb_middle',
    'bbands_20_real upper band': 'bb_upper',
    'macd_14_macd': 'macd',
    'macd_14_macd_signal': 'macd_signal',
    'macd_14_macd_hist': 'macd_histogram',
    'mfi_14_mfi': 'mfi',
    'obv_14_obv': 'obv',
    'ema_12_ema': 'ema_12',
    'ema_26_ema': 'ema_26',
    'natr_14_natr': 'natr',
    'adx_14_adx': 'adx',
    'cci_20_cci': 'cci',
    'willr_14_willr': 'willr',
    'mom_10_mom': 'momentum'
}

# Default thresholds for analysis
DEFAULT_THRESHOLDS = {
    'insider_activity_min_value_usd': 50000,
    'insider_buying_lookback_days': 30,
    'insider_selling_lookback_days': 30,
    'correlation_threshold_significant': 0.70,
    'market_regime_sma_period_short': 20,
    'market_regime_sma_period_long': 200,
    'iv_percentile_threshold': 75,
    'extreme_trade_pl_percentile': 95,
    'volume_spike_threshold': 2.0,
    'atr_spike_threshold': 1.5
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataQuality:
    """Track data quality for a trade analysis."""
    has_daily_prices: bool = False
    has_weekly_prices: bool = False
    has_fundamentals: bool = False
    has_insider_data: bool = False
    has_options_data: bool = False
    has_sector_data: bool = False
    has_index_data: bool = False
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    @property
    def is_complete(self) -> bool:
        return self.has_daily_prices

    @property
    def completeness_score(self) -> float:
        """Return a 0-100 score for data completeness."""
        checks = [
            self.has_daily_prices,
            self.has_weekly_prices,
            self.has_fundamentals,
            self.has_insider_data,
            self.has_options_data,
            self.has_sector_data
        ]
        return (sum(checks) / len(checks)) * 100


@dataclass
class MAEMFEResult:
    """Maximum Adverse/Favorable Excursion results."""
    mae_pct: float
    mae_price: float
    mae_date: datetime
    mae_days_into_trade: int
    mfe_pct: float
    mfe_price: float
    mfe_date: datetime
    mfe_days_into_trade: int
    actual_pl_pct: float
    mfe_capture_pct: float  # What % of MFE was captured


@dataclass
class MarketRegime:
    """Market regime at a point in time."""
    trend: str  # 'uptrend', 'downtrend', 'ranging'
    volatility: str  # 'high', 'normal', 'low'
    sma_alignment: str  # 'bullish', 'bearish', 'mixed'
    description: str


@dataclass
class TradeAnalysisData:
    """Complete aggregated data for a single trade analysis."""
    trade_id: str
    symbol: str
    trade_info: Dict[str, Any]
    price_data: Optional[pd.DataFrame] = None
    weekly_price_data: Optional[pd.DataFrame] = None
    fundamentals_entry: Optional[Dict] = None
    fundamentals_exit: Optional[Dict] = None
    fundamentals_delta: Optional[Dict] = None
    fundamentals_history: Optional[pd.DataFrame] = None  # Full history up to entry
    insider_activity: Optional[pd.DataFrame] = None
    insider_flags: List[str] = field(default_factory=list)
    options_data: Optional[Dict] = None
    sector_correlation: Optional[float] = None
    index_correlation: Optional[float] = None
    correlation_analysis: Optional[str] = None
    mae_mfe: Optional[MAEMFEResult] = None
    market_regime: Optional[MarketRegime] = None
    data_quality: DataQuality = field(default_factory=DataQuality)


# =============================================================================
# TRADE LOG LOADING AND VALIDATION
# =============================================================================

class TradeLogLoader:
    """Load and validate trade log CSV files."""

    def __init__(self, logs_base_path: Optional[Path] = None):
        """
        Initialize trade log loader.

        Args:
            logs_base_path: Base path for logs directory. Defaults to standard location.
        """
        if logs_base_path is None:
            self.logs_base_path = Path(__file__).parent.parent.parent / 'logs'
        else:
            self.logs_base_path = Path(logs_base_path)

    def load_trade_log(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single trade log CSV file.

        Args:
            file_path: Path to the trade log CSV

        Returns:
            DataFrame with validated trade data

        Raises:
            ValueError: If required columns are missing
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Trade log not found: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        # Validate required columns
        missing = set(REQUIRED_TRADE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse dates
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])

        # Validate date logic
        invalid_dates = df[df['exit_date'] < df['entry_date']]
        if len(invalid_dates) > 0:
            warnings.warn(f"Found {len(invalid_dates)} trades with exit_date < entry_date")

        # Add source file tracking
        df['source_file'] = str(file_path)

        return df

    def load_multiple_trade_logs(self, file_paths: List[Path]) -> pd.DataFrame:
        """
        Load and concatenate multiple trade log files.

        Args:
            file_paths: List of paths to trade log CSVs

        Returns:
            Concatenated DataFrame with all trades
        """
        all_trades = []
        errors = []

        for path in file_paths:
            try:
                df = self.load_trade_log(path)
                all_trades.append(df)
            except Exception as e:
                errors.append(f"{path}: {str(e)}")

        if errors:
            warnings.warn(f"Failed to load {len(errors)} files:\n" + "\n".join(errors))

        if not all_trades:
            raise ValueError("No trade logs could be loaded")

        # Concatenate and remove duplicates
        combined = pd.concat(all_trades, ignore_index=True)
        combined = combined.drop_duplicates(subset=['trade_id'], keep='first')

        # Sort by exit date descending
        combined = combined.sort_values('exit_date', ascending=False)

        return combined.reset_index(drop=True)

    def discover_backtest_trades(self, backtest_name: str) -> List[Path]:
        """
        Discover all trade files for a given backtest.

        Args:
            backtest_name: Name of the backtest folder

        Returns:
            List of paths to trade CSV files
        """
        trades_dir = self.logs_base_path / 'backtests' / 'portfolio' / backtest_name / 'trades'

        if not trades_dir.exists():
            raise FileNotFoundError(f"Backtest trades directory not found: {trades_dir}")

        return list(trades_dir.glob('*_trades.csv'))

    def list_available_backtests(self) -> List[str]:
        """List all available backtest names."""
        portfolio_dir = self.logs_base_path / 'backtests' / 'portfolio'
        if not portfolio_dir.exists():
            return []

        return [d.name for d in portfolio_dir.iterdir()
                if d.is_dir() and (d / 'trades').exists()]


# =============================================================================
# RAW DATA LOADING
# =============================================================================

class RawDataLoader:
    """Load raw price, fundamental, insider, and options data."""

    def __init__(self, data_base_path: Optional[Path] = None):
        """
        Initialize raw data loader.

        Args:
            data_base_path: Base path for raw_data directory
        """
        if data_base_path is None:
            self.data_base_path = Path(__file__).parent.parent.parent / 'raw_data'
        else:
            self.data_base_path = Path(data_base_path)

    def load_daily_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load daily price data for a symbol."""
        file_path = self.data_base_path / 'daily' / f'{symbol}_daily.csv'

        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def load_weekly_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load weekly price data for a symbol."""
        file_path = self.data_base_path / 'weekly' / f'{symbol}_weekly.csv'

        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def load_fundamentals(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load fundamental data for a symbol."""
        file_path = self.data_base_path / 'fundamentals' / f'{symbol}_fundamental.csv'

        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def load_insider_transactions(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load insider transaction data for a symbol."""
        file_path = self.data_base_path / 'insider_transactions' / f'{symbol}_insider.csv'

        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def load_options_data(self, symbol: str,
                          start_date: datetime,
                          end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load options data for a symbol within date range.
        Options data is stored in yearly files under options/{SYMBOL}/ directory.
        """
        options_dir = self.data_base_path / 'options' / symbol

        if not options_dir.exists():
            return None

        # Find relevant year files
        start_year = start_date.year
        end_year = end_date.year

        all_data = []
        for year in range(start_year, end_year + 1):
            file_path = options_dir / f'{symbol}_options_{year}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.lower().str.strip()
                all_data.append(df)

        if not all_data:
            return None

        combined = pd.concat(all_data, ignore_index=True)

        # Parse date columns
        date_col = 'snapshot_date' if 'snapshot_date' in combined.columns else 'date'
        if date_col in combined.columns:
            combined[date_col] = pd.to_datetime(combined[date_col])
            combined = combined[
                (combined[date_col] >= start_date) &
                (combined[date_col] <= end_date)
            ]

        return combined if len(combined) > 0 else None

    def get_available_symbols(self) -> Dict[str, Dict[str, bool]]:
        """
        Get all available symbols and what data types exist for each.

        Returns:
            Dict mapping symbol -> dict of available data types
        """
        symbols = {}

        # Check daily prices (required base)
        daily_dir = self.data_base_path / 'daily'
        if daily_dir.exists():
            for f in daily_dir.glob('*_daily.csv'):
                symbol = f.stem.replace('_daily', '')
                symbols[symbol] = {
                    'has_daily_prices': True,
                    'has_weekly_prices': False,
                    'has_fundamentals': False,
                    'has_insider_data': False,
                    'has_options_data': False
                }

        # Check weekly prices
        weekly_dir = self.data_base_path / 'weekly'
        if weekly_dir.exists():
            for f in weekly_dir.glob('*_weekly.csv'):
                symbol = f.stem.replace('_weekly', '')
                if symbol in symbols:
                    symbols[symbol]['has_weekly_prices'] = True

        # Check fundamentals
        fund_dir = self.data_base_path / 'fundamentals'
        if fund_dir.exists():
            for f in fund_dir.glob('*_fundamental.csv'):
                symbol = f.stem.replace('_fundamental', '')
                if symbol in symbols:
                    symbols[symbol]['has_fundamentals'] = True

        # Check insider data
        insider_dir = self.data_base_path / 'insider_transactions'
        if insider_dir.exists():
            for f in insider_dir.glob('*_insider.csv'):
                symbol = f.stem.replace('_insider', '')
                if symbol in symbols:
                    symbols[symbol]['has_insider_data'] = True

        # Check options data
        options_dir = self.data_base_path / 'options'
        if options_dir.exists():
            for d in options_dir.iterdir():
                if d.is_dir() and d.name in symbols:
                    symbols[d.name]['has_options_data'] = True

        return symbols


# =============================================================================
# REPRESENTATIVE TRADE SAMPLING
# =============================================================================

class TradeSampler:
    """Select representative trades from a trade log for analysis."""

    def __init__(self, target_sample_size: int = 12):
        """
        Initialize sampler.

        Args:
            target_sample_size: Target number of trades to select (8-15 typical)
        """
        self.target_sample_size = target_sample_size

    def sample_representative_trades(self,
                                     trades_df: pd.DataFrame,
                                     random_seed: int = 42) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Select statistically representative trades across outcome distributions.

        Args:
            trades_df: DataFrame with all trades
            random_seed: Seed for reproducible sampling

        Returns:
            Tuple of (selected trades DataFrame, list of selection rationale dicts)
        """
        np.random.seed(random_seed)

        if len(trades_df) == 0:
            return pd.DataFrame(), []

        # Calculate percentiles for outcome buckets
        percentiles = trades_df['pl_pct'].quantile([0.25, 0.5, 0.75])
        p25, p50, p75 = percentiles[0.25], percentiles[0.5], percentiles[0.75]

        # Define outcome buckets
        trades_df = trades_df.copy()
        trades_df['outcome_bucket'] = pd.cut(
            trades_df['pl_pct'],
            bins=[-np.inf, p25, p75, np.inf],
            labels=['loser', 'breakeven', 'winner']
        )

        # Normalize exit reasons
        trades_df['exit_type'] = trades_df['exit_reason'].apply(self._normalize_exit_reason)

        selected_trades = []
        rationales = []

        # Sample from each bucket + exit type combination
        for bucket in ['loser', 'breakeven', 'winner']:
            bucket_trades = trades_df[trades_df['outcome_bucket'] == bucket]

            for exit_type in bucket_trades['exit_type'].unique():
                subset = bucket_trades[bucket_trades['exit_type'] == exit_type]

                if len(subset) == 0:
                    continue

                # Select 1 trade, favoring median performers within subset
                if len(subset) >= 3:
                    # Sort by pl_pct and pick from middle
                    sorted_subset = subset.sort_values('pl_pct')
                    mid_idx = len(sorted_subset) // 2
                    selected = sorted_subset.iloc[mid_idx:mid_idx+1]
                else:
                    selected = subset.sample(n=1, random_state=random_seed)

                selected_trades.append(selected)
                rationales.append({
                    'trade_id': selected.iloc[0]['trade_id'],
                    'bucket': bucket,
                    'exit_type': exit_type,
                    'reason': f"Representative {bucket} with {exit_type} exit (from {len(subset)} similar trades)"
                })

        if not selected_trades:
            return pd.DataFrame(), []

        result = pd.concat(selected_trades, ignore_index=True)

        # Enforce diversity: ensure multiple symbols and time periods if possible
        result = self._enforce_diversity(result, trades_df, rationales)

        # Trim to target size if needed
        if len(result) > self.target_sample_size:
            result = result.head(self.target_sample_size)
            rationales = rationales[:self.target_sample_size]

        return result, rationales

    def _normalize_exit_reason(self, reason: str) -> str:
        """Normalize exit reason to standard categories."""
        if pd.isna(reason):
            return 'other'

        reason_lower = str(reason).lower()

        if 'stop' in reason_lower:
            return 'stop_loss'
        elif 'target' in reason_lower or 'take profit' in reason_lower:
            return 'target_hit'
        elif 'vulnerability' in reason_lower or 'swap' in reason_lower:
            return 'vulnerability_swap'
        elif 'manual' in reason_lower:
            return 'manual_exit'
        elif 'time' in reason_lower or 'duration' in reason_lower:
            return 'time_based'
        else:
            return 'other'

    def _enforce_diversity(self,
                          selected: pd.DataFrame,
                          all_trades: pd.DataFrame,
                          rationales: List[Dict]) -> pd.DataFrame:
        """Ensure sample includes diverse symbols and time periods."""
        # Check symbol diversity
        if len(selected['symbol'].unique()) < 2 and len(all_trades['symbol'].unique()) >= 2:
            # Try to add a trade from a different symbol
            other_symbols = all_trades[~all_trades['symbol'].isin(selected['symbol'])]
            if len(other_symbols) > 0:
                additional = other_symbols.sample(n=1)
                selected = pd.concat([selected, additional], ignore_index=True)
                rationales.append({
                    'trade_id': additional.iloc[0]['trade_id'],
                    'bucket': 'diversity',
                    'exit_type': additional.iloc[0].get('exit_type', 'other'),
                    'reason': f"Added for symbol diversity ({additional.iloc[0]['symbol']})"
                })

        # Check for both LONG and SHORT if available
        if 'side' in selected.columns:
            sides = selected['side'].unique()
            all_sides = all_trades['side'].unique()

            for side in all_sides:
                if side not in sides:
                    side_trades = all_trades[all_trades['side'] == side]
                    if len(side_trades) > 0:
                        additional = side_trades.sample(n=1)
                        selected = pd.concat([selected, additional], ignore_index=True)
                        rationales.append({
                            'trade_id': additional.iloc[0]['trade_id'],
                            'bucket': 'diversity',
                            'exit_type': additional.iloc[0].get('exit_type', 'other'),
                            'reason': f"Added for side diversity ({side})"
                        })
                        break

        return selected


# =============================================================================
# DATA AGGREGATION PIPELINE
# =============================================================================

class TradeDataAggregator:
    """Aggregate all data sources for trade analysis."""

    def __init__(self,
                 raw_data_loader: RawDataLoader,
                 thresholds: Optional[Dict] = None):
        """
        Initialize aggregator.

        Args:
            raw_data_loader: RawDataLoader instance
            thresholds: Custom thresholds dict (uses defaults if not provided)
        """
        self.raw_data = raw_data_loader
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def aggregate_trade_data(self,
                            trade: pd.Series,
                            pre_entry_days: int = 365,
                            post_exit_days: int = 30) -> TradeAnalysisData:
        """
        Aggregate all data for a single trade.

        Args:
            trade: Series containing trade information
            pre_entry_days: Days of data to load before entry
            post_exit_days: Days of data to load after exit

        Returns:
            TradeAnalysisData with all aggregated information
        """
        symbol = trade['symbol']
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        side = trade.get('side', 'LONG')

        # Initialize result
        result = TradeAnalysisData(
            trade_id=trade['trade_id'],
            symbol=symbol,
            trade_info=trade.to_dict()
        )

        # Calculate date ranges
        data_start = entry_date - timedelta(days=pre_entry_days)
        data_end = exit_date + timedelta(days=post_exit_days)

        # Load price data
        daily_prices = self.raw_data.load_daily_prices(symbol)
        if daily_prices is not None:
            result.data_quality.has_daily_prices = True
            result.price_data = self._filter_date_range(daily_prices, data_start, data_end)

            # Calculate MAE/MFE
            trade_prices = self._filter_date_range(daily_prices, entry_date, exit_date)
            if len(trade_prices) > 0:
                result.mae_mfe = self._calculate_mae_mfe(
                    trade_prices,
                    float(trade['entry_price']),
                    float(trade['exit_price']),
                    float(trade['pl_pct']),
                    side,
                    entry_date
                )

            # Determine market regime at entry
            entry_prices = self._filter_date_range(daily_prices, data_start, entry_date)
            if len(entry_prices) > 0:
                result.market_regime = self._detect_market_regime(entry_prices)
        else:
            result.data_quality.add_warning(f"No daily price data for {symbol}")

        # Load weekly prices
        weekly_prices = self.raw_data.load_weekly_prices(symbol)
        if weekly_prices is not None:
            result.data_quality.has_weekly_prices = True
            result.weekly_price_data = self._filter_date_range(weekly_prices, data_start, data_end)

        # Load and process fundamentals
        fundamentals = self.raw_data.load_fundamentals(symbol)
        if fundamentals is not None:
            result.data_quality.has_fundamentals = True
            result.fundamentals_entry = self._get_fundamentals_at_date(fundamentals, entry_date)
            result.fundamentals_exit = self._get_fundamentals_at_date(fundamentals, exit_date)
            result.fundamentals_delta = self._calculate_fundamental_deltas(
                result.fundamentals_entry,
                result.fundamentals_exit
            )
            # Store fundamentals history up to entry date
            result.fundamentals_history = self._get_fundamentals_history(fundamentals, entry_date)

        # Load and process insider activity
        insider_data = self.raw_data.load_insider_transactions(symbol)
        if insider_data is not None:
            result.data_quality.has_insider_data = True
            insider_window_start = entry_date - timedelta(days=365)
            insider_window_end = exit_date + timedelta(days=30)
            result.insider_activity = self._filter_date_range(
                insider_data, insider_window_start, insider_window_end
            )
            result.insider_flags = self._analyze_insider_activity(
                result.insider_activity, entry_date, exit_date
            )

        # Load options data if available
        options_data = self.raw_data.load_options_data(symbol, entry_date, exit_date)
        if options_data is not None:
            result.data_quality.has_options_data = True
            result.options_data = self._analyze_options_data(options_data, entry_date, exit_date)

        return result

    def _filter_date_range(self,
                           df: pd.DataFrame,
                           start: datetime,
                           end: datetime) -> pd.DataFrame:
        """Filter DataFrame to date range."""
        if 'date' not in df.columns:
            return df

        mask = (df['date'] >= start) & (df['date'] <= end)
        return df[mask].copy()

    def _calculate_mae_mfe(self,
                           trade_prices: pd.DataFrame,
                           entry_price: float,
                           exit_price: float,
                           actual_pl_pct: float,
                           side: str,
                           entry_date: datetime) -> MAEMFEResult:
        """Calculate Maximum Adverse and Favorable Excursion."""
        if side.upper() == 'LONG':
            # For LONG: MAE = lowest low, MFE = highest high
            mae_price = trade_prices['low'].min()
            mfe_price = trade_prices['high'].max()
            mae_idx = trade_prices['low'].idxmin()
            mfe_idx = trade_prices['high'].idxmax()
        else:
            # For SHORT: MAE = highest high, MFE = lowest low
            mae_price = trade_prices['high'].max()
            mfe_price = trade_prices['low'].min()
            mae_idx = trade_prices['high'].idxmax()
            mfe_idx = trade_prices['low'].idxmin()

        mae_date = trade_prices.loc[mae_idx, 'date']
        mfe_date = trade_prices.loc[mfe_idx, 'date']

        # Calculate percentages
        if side.upper() == 'LONG':
            mae_pct = ((mae_price - entry_price) / entry_price) * 100
            mfe_pct = ((mfe_price - entry_price) / entry_price) * 100
        else:
            mae_pct = ((entry_price - mae_price) / entry_price) * 100
            mfe_pct = ((entry_price - mfe_price) / entry_price) * 100

        # Calculate days into trade
        mae_days = (mae_date - entry_date).days
        mfe_days = (mfe_date - entry_date).days

        # Calculate capture percentage
        mfe_capture = (actual_pl_pct / mfe_pct * 100) if mfe_pct != 0 else 0

        return MAEMFEResult(
            mae_pct=abs(mae_pct),
            mae_price=mae_price,
            mae_date=mae_date,
            mae_days_into_trade=mae_days,
            mfe_pct=mfe_pct,
            mfe_price=mfe_price,
            mfe_date=mfe_date,
            mfe_days_into_trade=mfe_days,
            actual_pl_pct=actual_pl_pct,
            mfe_capture_pct=mfe_capture
        )

    def _detect_market_regime(self, prices: pd.DataFrame) -> MarketRegime:
        """Detect market regime from price data."""
        if len(prices) < 50:
            return MarketRegime(
                trend='unknown',
                volatility='unknown',
                sma_alignment='unknown',
                description='Insufficient data for regime detection'
            )

        latest = prices.iloc[-1]

        # Determine trend from SMA alignment
        sma_20 = latest.get('sma_20_sma', latest.get('sma_20'))
        sma_50 = latest.get('sma_50_sma', latest.get('sma_50'))
        sma_200 = latest.get('sma_200_sma', latest.get('sma_200'))
        close = latest['close']

        if pd.notna(sma_20) and pd.notna(sma_50) and pd.notna(sma_200):
            if sma_20 > sma_50 > sma_200:
                trend = 'uptrend'
                sma_alignment = 'bullish'
            elif sma_20 < sma_50 < sma_200:
                trend = 'downtrend'
                sma_alignment = 'bearish'
            else:
                trend = 'ranging'
                sma_alignment = 'mixed'
        else:
            trend = 'unknown'
            sma_alignment = 'unknown'

        # Determine volatility regime
        atr_col = 'atr_14_atr' if 'atr_14_atr' in prices.columns else 'atr'
        if atr_col in prices.columns and len(prices) >= 60:
            current_atr = latest.get(atr_col, 0)
            avg_atr = prices[atr_col].tail(60).mean()

            if avg_atr > 0:
                atr_ratio = current_atr / avg_atr
                if atr_ratio > 1.5:
                    volatility = 'high'
                elif atr_ratio < 0.66:
                    volatility = 'low'
                else:
                    volatility = 'normal'
            else:
                volatility = 'unknown'
        else:
            volatility = 'unknown'

        description = f"{trend.title()} market with {volatility} volatility. SMAs: {sma_alignment}"

        return MarketRegime(
            trend=trend,
            volatility=volatility,
            sma_alignment=sma_alignment,
            description=description
        )

    def _get_fundamentals_at_date(self,
                                   fundamentals: pd.DataFrame,
                                   target_date: datetime) -> Optional[Dict]:
        """Get most recent fundamental data before/on target date."""
        if 'date' not in fundamentals.columns:
            return None

        before_date = fundamentals[fundamentals['date'] <= target_date]

        if len(before_date) == 0:
            return None

        # Get most recent row
        latest = before_date.iloc[-1]

        # Extract key metrics
        return {
            'date': latest['date'],
            'pe_ratio': latest.get('pe_ratio'),
            'eps': latest.get('eps'),
            'earnings_growth_yoy': latest.get('earnings_growth_yoy'),
            'revenue_growth_yoy': latest.get('revenue_growth_yoy'),
            'profit_margin': latest.get('profit_margin'),
            'return_on_assets': latest.get('return_on_assets_ttm'),
            'return_on_equity': latest.get('return_on_equity_ttm'),
            'revenue_ttm': latest.get('revenue_ttm'),
            'gross_profit_ttm': latest.get('gross_profit_ttm'),
        }

    def _get_fundamentals_history(self,
                                   fundamentals: pd.DataFrame,
                                   entry_date: datetime,
                                   lookback_quarters: int = 12) -> Optional[pd.DataFrame]:
        """Get historical fundamental data up to entry date."""
        if 'date' not in fundamentals.columns:
            return None

        before_date = fundamentals[fundamentals['date'] <= entry_date].copy()

        if len(before_date) == 0:
            return None

        # Return most recent quarters (usually quarterly data)
        history = before_date.tail(lookback_quarters).copy()
        history = history.sort_values('date', ascending=True)

        return history

    def _calculate_fundamental_deltas(self,
                                       entry_fundamentals: Optional[Dict],
                                       exit_fundamentals: Optional[Dict]) -> Optional[Dict]:
        """Calculate changes in fundamentals between entry and exit."""
        if not entry_fundamentals or not exit_fundamentals:
            return None

        deltas = {}
        numeric_keys = ['pe_ratio', 'eps', 'earnings_growth_yoy', 'revenue_growth_yoy',
                       'profit_margin', 'return_on_assets', 'return_on_equity']

        for key in numeric_keys:
            entry_val = entry_fundamentals.get(key)
            exit_val = exit_fundamentals.get(key)

            if pd.notna(entry_val) and pd.notna(exit_val):
                deltas[key] = exit_val - entry_val
            else:
                deltas[key] = None

        return deltas

    def _analyze_insider_activity(self,
                                   insider_df: pd.DataFrame,
                                   entry_date: datetime,
                                   exit_date: datetime) -> List[str]:
        """Analyze insider activity and generate flags."""
        flags = []

        if insider_df is None or len(insider_df) == 0:
            return flags

        min_value = self.thresholds['insider_activity_min_value_usd']
        lookback = self.thresholds['insider_buying_lookback_days']

        # Pre-entry window
        pre_entry_start = entry_date - timedelta(days=lookback)
        pre_entry = insider_df[
            (insider_df['date'] >= pre_entry_start) &
            (insider_df['date'] < entry_date)
        ]

        # Check for significant buying before entry
        if 'transaction_type' in pre_entry.columns and 'value' in pre_entry.columns:
            buys = pre_entry[pre_entry['transaction_type'].str.upper() == 'BUY']
            total_buy_value = buys['value'].sum() if len(buys) > 0 else 0

            if total_buy_value >= min_value:
                flags.append(f"FLAG_INSIDER_BUYING_PREENTRY: ${total_buy_value:,.0f} in buying {lookback}d pre-entry")

        # Post-exit window
        post_exit_end = exit_date + timedelta(days=lookback)
        post_exit = insider_df[
            (insider_df['date'] > exit_date) &
            (insider_df['date'] <= post_exit_end)
        ]

        # Check for buying after exit (timing validation)
        if 'transaction_type' in post_exit.columns and 'value' in post_exit.columns:
            buys = post_exit[post_exit['transaction_type'].str.upper() == 'BUY']
            total_buy_value = buys['value'].sum() if len(buys) > 0 else 0

            if total_buy_value >= min_value:
                flags.append(f"FLAG_INSIDER_BUYING_POSTEXIT: ${total_buy_value:,.0f} in buying within 7d post-exit")

        # Check for selling during trade
        during_trade = insider_df[
            (insider_df['date'] >= entry_date) &
            (insider_df['date'] <= exit_date)
        ]

        if 'transaction_type' in during_trade.columns and 'value' in during_trade.columns:
            sells = during_trade[during_trade['transaction_type'].str.upper() == 'SELL']
            total_sell_value = sells['value'].sum() if len(sells) > 0 else 0

            if total_sell_value >= min_value:
                flags.append(f"FLAG_INSIDER_SELLING_DURING_TRADE: ${total_sell_value:,.0f} in selling during trade")

        # Check for coordinated activity (multiple insiders in same week)
        if 'executive' in insider_df.columns:
            insider_df_copy = insider_df.copy()
            insider_df_copy['week'] = insider_df_copy['date'].dt.isocalendar().week
            weekly_insiders = insider_df_copy.groupby('week')['executive'].nunique()

            if (weekly_insiders >= 2).any():
                flags.append("FLAG_COORDINATED_INSIDER_ACTIVITY: Multiple insiders active in same week")

        return flags

    def _analyze_options_data(self,
                               options_df: pd.DataFrame,
                               entry_date: datetime,
                               exit_date: datetime) -> Dict:
        """Analyze options data for the trade period."""
        result = {
            'iv_at_entry': None,
            'iv_at_exit': None,
            'iv_percentile_entry': None,
            'put_call_ratio_entry': None,
            'earnings_during_trade': False
        }

        if options_df is None or len(options_df) == 0:
            return result

        date_col = 'snapshot_date' if 'snapshot_date' in options_df.columns else 'date'

        # Get entry day data
        entry_data = options_df[options_df[date_col].dt.date == entry_date.date()]
        if len(entry_data) > 0:
            if 'implied_volatility' in entry_data.columns:
                result['iv_at_entry'] = entry_data['implied_volatility'].mean()

            # Calculate put/call ratio if possible
            if 'option_type' in entry_data.columns and 'open_interest' in entry_data.columns:
                puts = entry_data[entry_data['option_type'].str.upper() == 'PUT']['open_interest'].sum()
                calls = entry_data[entry_data['option_type'].str.upper() == 'CALL']['open_interest'].sum()
                if calls > 0:
                    result['put_call_ratio_entry'] = puts / calls

        # Get exit day data
        exit_data = options_df[options_df[date_col].dt.date == exit_date.date()]
        if len(exit_data) > 0:
            if 'implied_volatility' in exit_data.columns:
                result['iv_at_exit'] = exit_data['implied_volatility'].mean()

        # Calculate IV percentile
        if 'implied_volatility' in options_df.columns and result['iv_at_entry']:
            all_iv = options_df['implied_volatility'].dropna()
            if len(all_iv) > 0:
                result['iv_percentile_entry'] = (all_iv < result['iv_at_entry']).mean() * 100

        return result


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_correlation(symbol_prices: pd.DataFrame,
                         benchmark_prices: pd.DataFrame,
                         start_date: datetime,
                         end_date: datetime) -> Optional[float]:
    """
    Calculate correlation between symbol and benchmark returns.

    Args:
        symbol_prices: DataFrame with symbol daily prices
        benchmark_prices: DataFrame with benchmark daily prices
        start_date: Start of period
        end_date: End of period

    Returns:
        Pearson correlation coefficient or None if insufficient data
    """
    if symbol_prices is None or benchmark_prices is None:
        return None

    # Filter to date range
    symbol_filtered = symbol_prices[
        (symbol_prices['date'] >= start_date) &
        (symbol_prices['date'] <= end_date)
    ].copy()

    benchmark_filtered = benchmark_prices[
        (benchmark_prices['date'] >= start_date) &
        (benchmark_prices['date'] <= end_date)
    ].copy()

    if len(symbol_filtered) < 5 or len(benchmark_filtered) < 5:
        return None

    # Calculate daily returns
    symbol_filtered['return'] = symbol_filtered['close'].pct_change()
    benchmark_filtered['return'] = benchmark_filtered['close'].pct_change()

    # Merge on date
    merged = pd.merge(
        symbol_filtered[['date', 'return']],
        benchmark_filtered[['date', 'return']],
        on='date',
        suffixes=('_symbol', '_benchmark')
    ).dropna()

    if len(merged) < 5:
        return None

    correlation = merged['return_symbol'].corr(merged['return_benchmark'])

    return correlation


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_security_metadata(config_path: Optional[Path] = None) -> Dict:
    """Load security metadata from config file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'security_metadata.json'

    if not config_path.exists():
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)


def get_sector_ticker(symbol: str, metadata: Dict) -> Optional[str]:
    """Get sector ETF ticker for a symbol."""
    if symbol in metadata:
        return metadata[symbol].get('sector_ticker')
    return None


def get_index_ticker() -> str:
    """Get main market index ticker."""
    return 'SPX'


# =============================================================================
# MAIN AGGREGATION FUNCTION
# =============================================================================

def aggregate_trades_for_analysis(
    trades_df: pd.DataFrame,
    raw_data_loader: Optional[RawDataLoader] = None,
    max_trades: int = 15,
    pre_entry_days: int = 365,
    post_exit_days: int = 30
) -> List[TradeAnalysisData]:
    """
    Main function to aggregate data for multiple trades.

    Args:
        trades_df: DataFrame with trades to analyze
        raw_data_loader: RawDataLoader instance (created if not provided)
        max_trades: Maximum number of trades to process
        pre_entry_days: Days of pre-entry data to load
        post_exit_days: Days of post-exit data to load

    Returns:
        List of TradeAnalysisData objects
    """
    if raw_data_loader is None:
        raw_data_loader = RawDataLoader()

    aggregator = TradeDataAggregator(raw_data_loader)

    results = []
    trades_to_process = trades_df.head(max_trades)

    for idx, trade in trades_to_process.iterrows():
        try:
            analysis = aggregator.aggregate_trade_data(
                trade,
                pre_entry_days=pre_entry_days,
                post_exit_days=post_exit_days
            )
            results.append(analysis)
        except Exception as e:
            warnings.warn(f"Failed to aggregate data for trade {trade['trade_id']}: {e}")

    return results
