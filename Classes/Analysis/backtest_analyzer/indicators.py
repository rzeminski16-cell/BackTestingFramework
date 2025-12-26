"""
Weekly technical indicator calculations for backtest analysis.

This module handles:
- Loading daily price data
- Resampling to weekly OHLCV data
- Calculating technical indicators on weekly timeframe
- Providing lookup methods for indicator values at specific dates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .config import AnalysisConfig


class WeeklyIndicatorCalculator:
    """
    Calculates weekly technical indicators from daily price data.

    Indicators calculated:
    - 14-week RSI (SMA smoothed)
    - 52-week High/Low Distance
    - Bollinger Bands Position (20 periods, 2 std dev)
    - BB Band Width
    - Weekly Volume Trend (vs 50-week MA)
    - ATR-based Volatility (14-week)
    - Weekly ATR as % of price

    Additional indicators:
    - Price vs SMA200 (on daily data)
    - Volume Ratio at entry (entry day vs 20-day avg)
    - Days Since 52-Week High/Low
    """

    def __init__(self, config: AnalysisConfig, raw_data_dir: Path):
        """
        Initialize the indicator calculator.

        Args:
            config: Analysis configuration
            raw_data_dir: Directory containing raw price CSV files
        """
        self.config = config
        self.raw_data_dir = Path(raw_data_dir)
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._weekly_cache: Dict[str, pd.DataFrame] = {}
        self._indicators_cache: Dict[str, pd.DataFrame] = {}

    def load_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load daily OHLCV data for a symbol.

        Args:
            symbol: Security symbol

        Returns:
            DataFrame with daily OHLCV data, or None if not found
        """
        if symbol in self._daily_cache:
            return self._daily_cache[symbol]

        file_path = self.raw_data_dir / f"{symbol}.csv"
        if not file_path.exists():
            print(f"Warning: Raw data file not found for {symbol}: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()

            # Handle 'time' column
            if 'time' in df.columns and 'date' not in df.columns:
                df.rename(columns={'time': 'date'}, inplace=True)

            # Parse dates
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Ensure required columns exist
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"Warning: Missing columns for {symbol}: {missing}")
                return None

            self._daily_cache[symbol] = df
            return df

        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None

    def resample_to_weekly(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Resample daily data to weekly OHLCV.

        Uses Friday as the week end day to align with trading weeks.

        Args:
            symbol: Security symbol

        Returns:
            DataFrame with weekly OHLCV data
        """
        if symbol in self._weekly_cache:
            return self._weekly_cache[symbol]

        daily = self.load_daily_data(symbol)
        if daily is None:
            return None

        # Set date as index for resampling
        df = daily.set_index('date')

        # Resample to weekly (W-FRI = week ending Friday)
        weekly = df.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        weekly.reset_index(inplace=True)
        weekly.rename(columns={'date': 'week_end'}, inplace=True)

        self._weekly_cache[symbol] = weekly
        return weekly

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI using SMA smoothing (Cutler's RSI).

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI values
        """
        delta = prices.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Use SMA for smoothing (Cutler's RSI)
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_sma_smoothed_rsi(self, prices: pd.Series, rsi_period: int,
                                    sma_period: int) -> pd.Series:
        """
        Calculate RSI with additional SMA smoothing.

        Args:
            prices: Price series
            rsi_period: RSI calculation period
            sma_period: SMA smoothing period for RSI

        Returns:
            SMA-smoothed RSI values
        """
        rsi = self.calculate_rsi(prices, rsi_period)
        smoothed_rsi = rsi.rolling(window=sma_period, min_periods=1).mean()
        return smoothed_rsi

    def calculate_atr(self, high: pd.Series, low: pd.Series,
                      close: pd.Series, period: int) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    def calculate_bollinger_bands(self, prices: pd.Series, period: int,
                                   std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Price series
            period: SMA period
            std_dev: Number of standard deviations

        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        middle = prices.rolling(window=period, min_periods=period).mean()
        std = prices.rolling(window=period, min_periods=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return middle, upper, lower

    def calculate_all_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Calculate all weekly indicators for a symbol.

        Args:
            symbol: Security symbol

        Returns:
            DataFrame with all weekly indicators
        """
        if symbol in self._indicators_cache:
            return self._indicators_cache[symbol]

        weekly = self.resample_to_weekly(symbol)
        if weekly is None:
            return None

        df = weekly.copy()
        cfg = self.config

        # 14-week RSI (SMA smoothed)
        df['rsi_14w'] = self.calculate_sma_smoothed_rsi(
            df['close'],
            rsi_period=cfg.rsi_period_weeks,
            sma_period=cfg.rsi_sma_period_weeks
        )

        # 52-week High/Low
        df['high_52w'] = df['high'].rolling(
            window=cfg.high_low_period_weeks, min_periods=1
        ).max()
        df['low_52w'] = df['low'].rolling(
            window=cfg.high_low_period_weeks, min_periods=1
        ).min()

        # 52-week High/Low Distance: (Current - Low) / (High - Low)
        high_low_range = df['high_52w'] - df['low_52w']
        df['high_low_distance_52w'] = np.where(
            high_low_range > 0,
            (df['close'] - df['low_52w']) / high_low_range,
            0.5  # Neutral if no range
        )

        # Days since 52-week high/low (in weeks for weekly data)
        df['weeks_since_52w_high'] = self._calculate_periods_since_extreme(
            df['high'], df['high_52w'], cfg.high_low_period_weeks
        )
        df['weeks_since_52w_low'] = self._calculate_periods_since_extreme(
            df['low'], df['low_52w'], cfg.high_low_period_weeks, is_low=True
        )

        # Bollinger Bands (20 periods, 2 std dev)
        middle_bb, upper_bb, lower_bb = self.calculate_bollinger_bands(
            df['close'], cfg.bb_period_weeks, cfg.bb_std_dev
        )
        df['bb_middle'] = middle_bb
        df['bb_upper'] = upper_bb
        df['bb_lower'] = lower_bb

        # Bollinger Bands Position: (Price - Middle) / (Upper - Middle)
        # Result ranges from -1 (at lower band) to +1 (at upper band), 0 at middle
        band_half_width = upper_bb - middle_bb  # This equals std_dev * rolling_std
        df['bb_position'] = np.where(
            band_half_width > 0,
            (df['close'] - middle_bb) / band_half_width,
            0
        )

        # BB Band Width: (Upper - Lower) / Middle * 100
        df['bb_width'] = np.where(
            middle_bb > 0,
            (upper_bb - lower_bb) / middle_bb * 100,
            0
        )

        # Weekly Volume Trend: Current volume vs 50-week MA
        df['volume_ma_50w'] = df['volume'].rolling(
            window=cfg.volume_ma_period_weeks, min_periods=1
        ).mean()
        df['volume_trend'] = np.where(
            df['volume_ma_50w'] > 0,
            df['volume'] / df['volume_ma_50w'],
            1.0
        )

        # ATR-based Volatility (14-week)
        df['atr_14w'] = self.calculate_atr(
            df['high'], df['low'], df['close'], cfg.atr_period_weeks
        )

        # Weekly ATR as % of price
        df['atr_pct'] = np.where(
            df['close'] > 0,
            (df['atr_14w'] / df['close']) * 100,
            0
        )

        self._indicators_cache[symbol] = df
        return df

    def _calculate_periods_since_extreme(self, current_series: pd.Series,
                                          extreme_series: pd.Series,
                                          lookback: int,
                                          is_low: bool = False) -> pd.Series:
        """
        Calculate periods since the extreme (high or low) was hit.

        Args:
            current_series: Current high/low series
            extreme_series: Rolling extreme (max high or min low)
            lookback: Lookback period
            is_low: If True, looking for low; if False, looking for high

        Returns:
            Series with periods since extreme
        """
        result = pd.Series(index=current_series.index, dtype=float)

        for i in range(len(current_series)):
            if i < lookback:
                window = current_series.iloc[:i + 1]
            else:
                window = current_series.iloc[i - lookback + 1:i + 1]

            if len(window) == 0:
                result.iloc[i] = 0
                continue

            if is_low:
                extreme_idx = window.idxmin()
            else:
                extreme_idx = window.idxmax()

            periods_since = i - extreme_idx
            result.iloc[i] = periods_since

        return result

    def get_indicators_at_date(self, symbol: str, date: datetime,
                                daily_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get all indicator values for a specific date.

        Uses the most recent weekly data point at or before the given date,
        plus some daily indicators.

        Args:
            symbol: Security symbol
            date: Date to get indicators for
            daily_data: Optional pre-loaded daily data

        Returns:
            Dictionary of indicator values
        """
        indicators = {}

        # Get weekly indicators
        weekly_df = self.calculate_all_indicators(symbol)
        if weekly_df is not None and len(weekly_df) > 0:
            # Find the most recent week at or before the date
            weekly_df = weekly_df[weekly_df['week_end'] <= pd.Timestamp(date)]
            if len(weekly_df) > 0:
                latest = weekly_df.iloc[-1]
                indicators.update({
                    'rsi_14w': latest.get('rsi_14w', np.nan),
                    'high_low_distance_52w': latest.get('high_low_distance_52w', np.nan),
                    'bb_position': latest.get('bb_position', np.nan),
                    'bb_width': latest.get('bb_width', np.nan),
                    'volume_trend': latest.get('volume_trend', np.nan),
                    'atr_14w': latest.get('atr_14w', np.nan),
                    'atr_pct': latest.get('atr_pct', np.nan),
                    'weeks_since_52w_high': latest.get('weeks_since_52w_high', np.nan),
                    'weeks_since_52w_low': latest.get('weeks_since_52w_low', np.nan),
                })

        # Get daily indicators
        if daily_data is None:
            daily_data = self.load_daily_data(symbol)

        if daily_data is not None and len(daily_data) > 0:
            daily_indicators = self._calculate_daily_indicators_at_date(
                daily_data, date
            )
            indicators.update(daily_indicators)

        return indicators

    def _calculate_daily_indicators_at_date(self, daily_data: pd.DataFrame,
                                             date: datetime) -> Dict:
        """
        Calculate daily-based indicators for a specific date.

        Args:
            daily_data: Daily OHLCV data
            date: Target date

        Returns:
            Dictionary of daily indicator values
        """
        indicators = {}
        cfg = self.config

        # Filter data up to the date
        df = daily_data[daily_data['date'] <= pd.Timestamp(date)].copy()
        if len(df) == 0:
            return indicators

        # Price vs SMA200
        if len(df) >= cfg.sma_200_period_days:
            sma_200 = df['close'].rolling(window=cfg.sma_200_period_days).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            indicators['price_vs_sma200'] = (current_price - sma_200) / sma_200 * 100
            indicators['above_sma200'] = 1 if current_price > sma_200 else 0
        else:
            indicators['price_vs_sma200'] = np.nan
            indicators['above_sma200'] = np.nan

        # Volume Ratio at entry (current day vs 20-day avg)
        if len(df) >= cfg.volume_ratio_period_days:
            avg_volume = df['volume'].rolling(
                window=cfg.volume_ratio_period_days
            ).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            indicators['volume_ratio'] = np.nan

        # Days since 52-week high/low (in trading days)
        lookback_days = cfg.high_low_period_days
        if len(df) >= lookback_days:
            recent = df.iloc[-lookback_days:]
            high_idx = recent['high'].idxmax()
            low_idx = recent['low'].idxmin()
            indicators['days_since_52w_high'] = len(df) - 1 - high_idx
            indicators['days_since_52w_low'] = len(df) - 1 - low_idx
        else:
            # Use available data
            high_idx = df['high'].idxmax()
            low_idx = df['low'].idxmin()
            indicators['days_since_52w_high'] = len(df) - 1 - high_idx
            indicators['days_since_52w_low'] = len(df) - 1 - low_idx

        return indicators

    def get_price_after_days(self, symbol: str, date: datetime,
                              days_after: int) -> Optional[float]:
        """
        Get the closing price N trading days after a given date.

        Args:
            symbol: Security symbol
            date: Start date
            days_after: Number of trading days to look ahead

        Returns:
            Closing price or None if not available
        """
        daily = self.load_daily_data(symbol)
        if daily is None:
            return None

        # Find dates after the given date
        future_data = daily[daily['date'] > pd.Timestamp(date)]

        if len(future_data) >= days_after:
            return future_data.iloc[days_after - 1]['close']
        elif len(future_data) > 0:
            # Return the last available price
            return future_data.iloc[-1]['close']
        else:
            return None

    def get_max_price_in_period(self, symbol: str, start_date: datetime,
                                 days: int) -> Optional[float]:
        """
        Get the maximum high price in the N trading days after a date.

        Args:
            symbol: Security symbol
            start_date: Start date (exclusive)
            days: Number of trading days to look ahead

        Returns:
            Maximum high price or None if not available
        """
        daily = self.load_daily_data(symbol)
        if daily is None:
            return None

        # Find dates after the given date
        future_data = daily[daily['date'] > pd.Timestamp(start_date)]

        if len(future_data) > 0:
            # Take at most 'days' number of rows
            period_data = future_data.head(days)
            return period_data['high'].max()
        else:
            return None

    def save_weekly_data(self, symbol: str, output_dir: Path) -> Optional[Path]:
        """
        Save weekly data with indicators to CSV for validation.

        Args:
            symbol: Security symbol
            output_dir: Directory to save the file

        Returns:
            Path to saved file or None if failed
        """
        weekly_indicators = self.calculate_all_indicators(symbol)
        if weekly_indicators is None:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{symbol}_weekly_indicators.csv"
        weekly_indicators.to_csv(file_path, index=False)

        return file_path

    def clear_cache(self):
        """Clear all cached data."""
        self._daily_cache.clear()
        self._weekly_cache.clear()
        self._indicators_cache.clear()
