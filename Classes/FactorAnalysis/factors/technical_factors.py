"""
Technical Factors for Factor Analysis.

Extracts and processes technical indicator factors from price data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..logging.audit_logger import AuditLogger


@dataclass
class TechnicalFactorResult:
    """Result of technical factor extraction."""
    factors_extracted: int
    trades_with_data: int
    total_trades: int
    factor_names: List[str]


class TechnicalFactors:
    """
    Extracts technical indicator factors from price data.

    Technical indicators are expected to be pre-calculated in the price data
    (as per the backtesting framework pattern). This class:
    - Identifies available indicator columns
    - Extracts values at trade entry dates
    - Computes derived technical factors
    """

    # Common indicator column patterns
    INDICATOR_PATTERNS = {
        'momentum': ['rsi', 'roc', 'momentum', 'cci', 'willr', 'stoch', 'mfi'],
        'trend': ['sma', 'ema', 'macd', 'adx', 'aroon', 'dema', 'tema', 'kama'],
        'volatility': ['atr', 'natr', 'bollinger', 'keltner'],
        'volume': ['obv', 'ad', 'adosc', 'vwap'],
        'other': ['sar', 'trix', 'ultosc', 'bop', 'ppo']
    }

    def __init__(self, logger: Optional[AuditLogger] = None):
        """
        Initialize TechnicalFactors.

        Args:
            logger: Optional audit logger
        """
        self.logger = logger

    def identify_indicator_columns(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Identify technical indicator columns in DataFrame.

        Args:
            df: Price data DataFrame

        Returns:
            Dictionary mapping category to list of column names
        """
        identified = {category: [] for category in self.INDICATOR_PATTERNS}

        for col in df.columns:
            col_lower = col.lower()

            # Skip non-indicator columns
            if col_lower in ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']:
                continue

            # Check each category
            for category, patterns in self.INDICATOR_PATTERNS.items():
                if any(pattern in col_lower for pattern in patterns):
                    identified[category].append(col)
                    break
            else:
                # Check if it looks like an indicator (has numbers or underscores)
                if any(c.isdigit() for c in col) or '_' in col:
                    identified['other'].append(col)

        return identified

    def extract_factors(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        indicator_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, TechnicalFactorResult]:
        """
        Extract technical factors for each trade.

        Args:
            trades_df: Trade log DataFrame with 'symbol' and 'entry_date'
            price_df: Price data with indicator columns
            indicator_columns: Specific columns to extract (None = auto-detect)

        Returns:
            Tuple of (DataFrame with factors, TechnicalFactorResult)
        """
        if self.logger:
            self.logger.start_section("TECHNICAL_FACTORS")

        # Identify indicator columns if not specified
        if indicator_columns is None:
            categorized = self.identify_indicator_columns(price_df)
            indicator_columns = []
            for cols in categorized.values():
                indicator_columns.extend(cols)

        if self.logger:
            self.logger.info(f"Extracting technical factors", {
                'indicator_count': len(indicator_columns),
                'samples': indicator_columns[:10]
            })

        # Ensure dates are datetime
        trades_df = trades_df.copy()
        price_df = price_df.copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        price_df['date'] = pd.to_datetime(price_df['date'])

        # Check symbol overlap before processing
        use_symbol_filtering = False
        if 'symbol' in price_df.columns and 'symbol' in trades_df.columns:
            price_symbols = set(price_df['symbol'].str.upper().dropna().unique())
            trade_symbols = set(trades_df['symbol'].str.upper().dropna().unique())
            symbol_overlap = price_symbols & trade_symbols
            use_symbol_filtering = len(symbol_overlap) > 0
            if self.logger:
                self.logger.info("Symbol overlap check", {
                    'price_symbols': len(price_symbols),
                    'trade_symbols': len(trade_symbols),
                    'overlap': len(symbol_overlap),
                    'using_symbol_filter': use_symbol_filtering
                })

        results = []
        trades_with_data = 0

        for idx, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            symbol = trade.get('symbol', '')

            # Filter price data for symbol only if there's overlap
            if use_symbol_filtering and symbol:
                symbol_prices = price_df[price_df['symbol'].str.upper() == symbol.upper()]
                # Fall back to all prices if no match for this specific symbol
                if len(symbol_prices) == 0:
                    symbol_prices = price_df
            else:
                symbol_prices = price_df

            # Get data at entry date (or most recent before)
            exact = symbol_prices[symbol_prices['date'] == entry_date]

            if len(exact) > 0:
                row_data = exact.iloc[0][indicator_columns].to_dict()
                trades_with_data += 1
            else:
                # Forward fill from last available
                before = symbol_prices[symbol_prices['date'] < entry_date]
                if len(before) > 0:
                    row_data = before.iloc[-1][indicator_columns].to_dict()
                    trades_with_data += 1
                else:
                    row_data = {col: np.nan for col in indicator_columns}

            # Add prefix for clarity
            row_data = {f'tech_{k}': v for k, v in row_data.items()}
            row_data['_trade_idx'] = idx
            results.append(row_data)

        factors_df = pd.DataFrame(results)

        result = TechnicalFactorResult(
            factors_extracted=len(indicator_columns),
            trades_with_data=trades_with_data,
            total_trades=len(trades_df),
            factor_names=[f'tech_{c}' for c in indicator_columns]
        )

        if self.logger:
            self.logger.log_factor_engineering(
                category='Technical',
                factor_count=result.factors_extracted,
                trades_with_data=result.trades_with_data,
                total_trades=result.total_trades
            )
            self.logger.end_section()

        return factors_df, result

    def compute_derived_factors(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute derived technical factors.

        Args:
            df: DataFrame with technical factor columns

        Returns:
            DataFrame with additional derived factors
        """
        df = df.copy()

        # RSI regime (overbought/oversold)
        rsi_cols = [c for c in df.columns if 'rsi' in c.lower()]
        for col in rsi_cols:
            if col in df.columns:
                df[f'{col}_regime'] = pd.cut(
                    df[col],
                    bins=[0, 30, 70, 100],
                    labels=['oversold', 'neutral', 'overbought']
                )

        # MACD signal (bullish/bearish crossover state)
        macd_cols = [c for c in df.columns if 'macd' in c.lower() and 'signal' not in c.lower()]
        signal_cols = [c for c in df.columns if 'signal' in c.lower() or 'macdsignal' in c.lower()]

        if macd_cols and signal_cols:
            macd_col = macd_cols[0]
            signal_col = signal_cols[0]
            df['tech_macd_position'] = np.where(
                df[macd_col] > df[signal_col], 'bullish', 'bearish'
            )

        # Bollinger Band position
        bb_upper = [c for c in df.columns if 'upper' in c.lower() and 'bollinger' in c.lower()]
        bb_lower = [c for c in df.columns if 'lower' in c.lower() and 'bollinger' in c.lower()]

        # ADX trend strength
        adx_cols = [c for c in df.columns if 'adx' in c.lower()]
        for col in adx_cols:
            if col in df.columns:
                df[f'{col}_strength'] = pd.cut(
                    df[col],
                    bins=[0, 20, 40, 60, 100],
                    labels=['weak', 'moderate', 'strong', 'extreme']
                )

        return df

    def get_factor_categories(
        self,
        factor_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Categorize factor names by indicator type.

        Args:
            factor_names: List of factor column names

        Returns:
            Dictionary mapping category to factor names
        """
        categorized = {category: [] for category in self.INDICATOR_PATTERNS}

        for name in factor_names:
            name_lower = name.lower()
            for category, patterns in self.INDICATOR_PATTERNS.items():
                if any(pattern in name_lower for pattern in patterns):
                    categorized[category].append(name)
                    break
            else:
                categorized['other'].append(name)

        return categorized

    def compute_all(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute all technical factors for trades.

        This is the main entry point for the analyzer. It:
        1. Extracts technical factors from price data
        2. Computes derived factors
        3. Merges results with trades DataFrame

        Args:
            trades_df: Trade log DataFrame with 'symbol' and 'entry_date'
            price_df: Price data with indicator columns

        Returns:
            trades_df with technical factor columns added
        """
        # Extract raw technical factors
        factors_df, result = self.extract_factors(trades_df, price_df)

        # Store factor names for later retrieval
        self._factor_names = result.factor_names.copy()

        # Merge factors with trades
        if '_trade_idx' in factors_df.columns:
            trades_df = trades_df.copy()
            for col in factors_df.columns:
                if col != '_trade_idx':
                    trades_df[col] = factors_df.set_index('_trade_idx')[col]

        # Compute derived factors
        trades_df = self.compute_derived_factors(trades_df)

        # Add any new derived factor names
        derived_names = [
            col for col in trades_df.columns
            if col.startswith('tech_') and col not in self._factor_names
        ]
        self._factor_names.extend(derived_names)

        return trades_df

    def get_factor_names(self) -> List[str]:
        """
        Get list of factor column names produced by compute_all.

        Returns:
            List of factor column names
        """
        if hasattr(self, '_factor_names'):
            return self._factor_names
        return []

