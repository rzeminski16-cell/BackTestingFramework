"""
Indicator Engine for validation of pre-calculated indicators from raw data.

This module validates that required technical indicators exist in the raw data.
All indicators are expected to be pre-calculated in the CSV data files with
standardized column names.

Standard Indicator Columns (read from raw data):
- atr_14: Average True Range with 14-period (static, no parameters)
- ema_50: Exponential Moving Average with 50-period (static, no parameters)
- sma_200: Simple Moving Average with 200-period (static, no parameters)
- rsi_14: Relative Strength Index with 14-period (static, no parameters)
- cmf: Chaikin Money Flow (static, no parameters)
- par_sar: Parabolic SAR (static, no parameters)
- base_bb: Base Bollinger Bands (static, no parameters)
- upper_bb: Upper Bollinger Bands (static, no parameters)
- lower_bb: Lower Bollinger Bands (static, no parameters)

All indicators have fixed settings and cannot be altered via parameters.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


class IndicatorEngine:
    """
    Indicator validation engine for pre-calculated indicators.

    This class validates that required indicators exist in the raw data
    and provides utility methods for accessing them. All indicators are
    read from the CSV data - no calculations are performed.
    """

    # Standard indicator column names (all static, no parameters)
    STANDARD_INDICATORS = {
        'atr_14': 'Average True Range (14-period)',
        'ema_50': 'Exponential Moving Average (50-period)',
        'sma_200': 'Simple Moving Average (200-period)',
        'rsi_14': 'Relative Strength Index (14-period)',
        'cmf': 'Chaikin Money Flow',
        'par_sar': 'Parabolic SAR',
        'base_bb': 'Bollinger Bands Base',
        'upper_bb': 'Bollinger Bands Upper',
        'lower_bb': 'Bollinger Bands Lower'
    }

    @staticmethod
    def validate_indicators(
        data: pd.DataFrame,
        required_indicators: List[str]
    ) -> tuple[bool, List[str]]:
        """
        Validate that required indicators exist in the data.

        Args:
            data: DataFrame to validate
            required_indicators: List of required indicator column names

        Returns:
            Tuple of (all_present, missing_indicators)
        """
        missing = [col for col in required_indicators if col not in data.columns]
        return (len(missing) == 0, missing)

    @staticmethod
    def get_available_indicators(data: pd.DataFrame) -> List[str]:
        """
        Get list of available standard indicators in the data.

        Args:
            data: DataFrame to check

        Returns:
            List of available standard indicator column names
        """
        return [col for col in IndicatorEngine.STANDARD_INDICATORS.keys()
                if col in data.columns]

    @staticmethod
    def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting (no-op, returns data unchanged).

        This method exists for backward compatibility. Since all indicators
        are pre-calculated in the raw data, no processing is needed.

        Args:
            data: Raw OHLCV data with pre-calculated indicators

        Returns:
            Data unchanged
        """
        return data
