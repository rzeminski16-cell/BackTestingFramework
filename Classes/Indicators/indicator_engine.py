"""
Indicator Engine for validation of pre-calculated indicators from raw data.

This module validates that required technical indicators exist in the raw data.
All indicators are expected to be pre-calculated in the CSV data files with
standardized column names from Alpha Vantage.

IMPORTANT: All indicators MUST be read from raw data - NO calculations are
performed in this module or anywhere else in the backtesting system.

Column Naming Convention (from Alpha Vantage):
    {indicator}_{period}_{output}

Examples:
    - atr_14_atr: Average True Range (14-period)
    - ema_50_ema: Exponential Moving Average (50-period)
    - sma_200_sma: Simple Moving Average (200-period)
    - rsi_14_rsi: Relative Strength Index (14-period)
    - mfi_14_mfi: Money Flow Index (14-period)
    - bbands_20_real middle band: Bollinger Bands middle (20-period)

If a required indicator is missing from the raw data, a MissingColumnError
will be raised with clear instructions on how to collect the missing data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


class IndicatorEngine:
    """
    Indicator validation engine for pre-calculated indicators.

    This class validates that required indicators exist in the raw data
    and provides utility methods for accessing them. All indicators are
    read from the CSV data - NO calculations are performed.

    CRITICAL: If you need an indicator that doesn't exist in your raw data,
    you must re-run data collection with that indicator enabled. Do NOT
    add calculation code here.
    """

    # Standard indicator column names from Alpha Vantage raw data
    # Format: column_name -> description
    STANDARD_INDICATORS = {
        # Price & Volume (always available)
        'date': 'Date',
        'open': 'Open Price',
        'high': 'High Price',
        'low': 'Low Price',
        'close': 'Close Price',
        'volume': 'Volume',

        # Trend Indicators
        'sma_20_sma': 'Simple Moving Average (20-period)',
        'sma_50_sma': 'Simple Moving Average (50-period)',
        'sma_200_sma': 'Simple Moving Average (200-period)',
        'ema_12_ema': 'Exponential Moving Average (12-period)',
        'ema_26_ema': 'Exponential Moving Average (26-period)',
        'dema_14_dema': 'Double EMA (14-period)',
        'tema_14_tema': 'Triple EMA (14-period)',
        'kama_14_kama': 'Kaufman Adaptive MA (14-period)',
        't3_14_t3': 'T3 Moving Average (14-period)',
        'trima_14_trima': 'Triangular MA (14-period)',
        'wma_14_wma': 'Weighted MA (14-period)',
        'ht_trendline_14_ht_trendline': 'Hilbert Transform Trendline',

        # Volatility Indicators
        'atr_14_atr': 'Average True Range (14-period)',
        'natr_14_natr': 'Normalized ATR (14-period)',
        'trange_14_trange': 'True Range',
        'bbands_20_real lower band': 'Bollinger Bands Lower (20-period)',
        'bbands_20_real middle band': 'Bollinger Bands Middle (20-period)',
        'bbands_20_real upper band': 'Bollinger Bands Upper (20-period)',

        # Momentum Indicators
        'rsi_14_rsi': 'Relative Strength Index (14-period)',
        'mfi_14_mfi': 'Money Flow Index (14-period)',
        'cci_20_cci': 'Commodity Channel Index (20-period)',
        'cmo_14_cmo': 'Chande Momentum Oscillator (14-period)',
        'mom_10_mom': 'Momentum (10-period)',
        'roc_10_roc': 'Rate of Change (10-period)',
        'rocr_14_rocr': 'Rate of Change Ratio (14-period)',
        'willr_14_willr': 'Williams %R (14-period)',
        'ultosc_14_ultosc': 'Ultimate Oscillator (14-period)',
        'bop_14_bop': 'Balance of Power',

        # Trend Strength Indicators
        'adx_14_adx': 'Average Directional Index (14-period)',
        'adxr_14_adxr': 'ADX Rating (14-period)',
        'dx_14_dx': 'Directional Movement Index (14-period)',
        'plus_di_14_plus_di': 'Plus Directional Indicator (14-period)',
        'minus_di_14_minus_di': 'Minus Directional Indicator (14-period)',
        'aroon_14_aroon up': 'Aroon Up (14-period)',
        'aroon_14_aroon down': 'Aroon Down (14-period)',
        'aroonosc_14_aroonosc': 'Aroon Oscillator (14-period)',

        # MACD
        'macd_14_macd': 'MACD Line',
        'macd_14_macd_signal': 'MACD Signal',
        'macd_14_macd_hist': 'MACD Histogram',
        'macdext_14_macd': 'MACD Extended Line',
        'macdext_14_macd_signal': 'MACD Extended Signal',
        'macdext_14_macd_hist': 'MACD Extended Histogram',
        'apo_14_apo': 'Absolute Price Oscillator (14-period)',
        'ppo_14_ppo': 'Percentage Price Oscillator (14-period)',
        'trix_14_trix': 'TRIX (14-period)',

        # Stochastic
        'stoch_14_slowk': 'Stochastic Slow %K',
        'stoch_14_slowd': 'Stochastic Slow %D',
        'stochf_14_fastk': 'Stochastic Fast %K',
        'stochf_14_fastd': 'Stochastic Fast %D',
        'stochrsi_14_fastk': 'Stochastic RSI Fast %K',
        'stochrsi_14_fastd': 'Stochastic RSI Fast %D',

        # Volume Indicators
        'obv_14_obv': 'On Balance Volume',
        'ad_14_chaikin a/d': 'Chaikin A/D Line',
        'adosc_14_adosc': 'Chaikin A/D Oscillator',

        # Hilbert Transform
        'ht_dcperiod_14_dcperiod': 'HT Dominant Cycle Period',
        'ht_dcphase_14_ht_dcphase': 'HT Dominant Cycle Phase',
        'ht_phasor_14_phase': 'HT Phasor Phase',
        'ht_phasor_14_quadrature': 'HT Phasor Quadrature',
        'ht_sine_14_sine': 'HT Sine',
        'ht_sine_14_lead sine': 'HT Lead Sine',
        'ht_trendmode_14_trendmode': 'HT Trend Mode',

        # Other
        'sar_14_sar': 'Parabolic SAR',
        'midpoint_14_midpoint': 'Midpoint',
        'midprice_14_midprice': 'Midprice',

        # MAMA indicators
        'mama_14_mama': 'MESA Adaptive MA',
        'mama_14_fama': 'Following Adaptive MA',
    }

    # Mapping from legacy column names to new Alpha Vantage names
    # This helps with backward compatibility
    LEGACY_COLUMN_MAPPING = {
        'atr_14': 'atr_14_atr',
        'ema_50': 'sma_50_sma',  # Fallback to SMA if EMA not available
        'sma_200': 'sma_200_sma',
        'rsi_14': 'rsi_14_rsi',
        'mfi': 'mfi_14_mfi',
        'cmf': 'ad_14_chaikin a/d',  # CMF approximated by A/D
        'par_sar': 'sar_14_sar',
        'base_bb': 'bbands_20_real middle band',
        'upper_bb': 'bbands_20_real upper band',
        'lower_bb': 'bbands_20_real lower band',
    }

    @classmethod
    def resolve_column_name(cls, column: str, available_columns: List[str]) -> Optional[str]:
        """
        Resolve a column name, checking for both new and legacy names.

        Args:
            column: The column name to resolve
            available_columns: List of columns available in the DataFrame

        Returns:
            The resolved column name if found, None otherwise
        """
        # Direct match
        if column in available_columns:
            return column

        # Check if this is a legacy name that needs mapping
        if column in cls.LEGACY_COLUMN_MAPPING:
            new_name = cls.LEGACY_COLUMN_MAPPING[column]
            if new_name in available_columns:
                return new_name

        # Check if the new name exists for a legacy request
        for legacy, new in cls.LEGACY_COLUMN_MAPPING.items():
            if new == column and legacy in available_columns:
                return legacy

        return None

    @classmethod
    def validate_indicators(
        cls,
        data: pd.DataFrame,
        required_indicators: List[str]
    ) -> tuple[bool, List[str]]:
        """
        Validate that required indicators exist in the data.

        Checks both new Alpha Vantage column names and legacy names.

        Args:
            data: DataFrame to validate
            required_indicators: List of required indicator column names

        Returns:
            Tuple of (all_present, missing_indicators)
        """
        available = data.columns.tolist()
        missing = []

        for col in required_indicators:
            resolved = cls.resolve_column_name(col, available)
            if resolved is None:
                missing.append(col)

        return (len(missing) == 0, missing)

    @classmethod
    def get_available_indicators(cls, data: pd.DataFrame) -> List[str]:
        """
        Get list of available standard indicators in the data.

        Args:
            data: DataFrame to check

        Returns:
            List of available standard indicator column names
        """
        return [col for col in cls.STANDARD_INDICATORS.keys()
                if col in data.columns]

    @classmethod
    def get_indicator_value(cls, data: pd.DataFrame, column: str, index: int) -> Optional[float]:
        """
        Get an indicator value from the data, handling column name resolution.

        Args:
            data: DataFrame with indicator data
            column: Column name (can be legacy or new format)
            index: Row index to retrieve

        Returns:
            The indicator value, or None if column not found

        Raises:
            ValueError: If column is not found in data
        """
        available = data.columns.tolist()
        resolved = cls.resolve_column_name(column, available)

        if resolved is None:
            raise ValueError(
                f"Indicator '{column}' not found in data. "
                f"Available indicators: {[c for c in available if c in cls.STANDARD_INDICATORS]}"
            )

        return data[resolved].iloc[index]

    @staticmethod
    def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting (no-op, returns data unchanged).

        This method exists for backward compatibility. Since all indicators
        are pre-calculated in the raw data, no processing is needed.

        IMPORTANT: Do NOT add calculation code here. All indicators must
        come from the raw data CSV files.

        Args:
            data: Raw OHLCV data with pre-calculated indicators

        Returns:
            Data unchanged
        """
        return data

    @classmethod
    def list_missing_indicators(cls, data: pd.DataFrame, required: List[str]) -> Dict[str, str]:
        """
        Get a detailed report of missing indicators with suggestions.

        Args:
            data: DataFrame to check
            required: List of required column names

        Returns:
            Dict mapping missing column to suggestion message
        """
        available = data.columns.tolist()
        missing_report = {}

        for col in required:
            resolved = cls.resolve_column_name(col, available)
            if resolved is None:
                # Check if it's a known indicator
                if col in cls.STANDARD_INDICATORS:
                    missing_report[col] = (
                        f"Re-run data collection with '{col.split('_')[0].upper()}' indicator enabled"
                    )
                elif col in cls.LEGACY_COLUMN_MAPPING:
                    new_name = cls.LEGACY_COLUMN_MAPPING[col]
                    missing_report[col] = (
                        f"Column '{col}' is a legacy name. "
                        f"The new name '{new_name}' is also not available. "
                        f"Re-run data collection to add this indicator."
                    )
                else:
                    missing_report[col] = (
                        f"Unknown indicator. Check if the column name is correct."
                    )

        return missing_report
