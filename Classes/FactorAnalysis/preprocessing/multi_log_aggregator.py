"""
Multi-Log Aggregator for Factor Analysis.

Handles aggregation of trades from multiple trade logs with proper
metadata tagging and stratification options.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..config.factor_config import MultiLogAggregationConfig, AggregationMode
from ..logging.audit_logger import AuditLogger


@dataclass
class AggregationResult:
    """Result of multi-log aggregation."""
    total_trades: int
    total_logs: int
    unique_symbols: int
    unique_periods: int
    aggregation_mode: str
    strata_counts: Dict[str, int] = field(default_factory=dict)
    symbol_distribution: Dict[str, int] = field(default_factory=dict)
    period_distribution: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class MultiLogAggregator:
    """
    Aggregates trades from multiple logs with metadata tagging.

    Supports:
    - Pooled aggregation (combine all trades)
    - Stratified aggregation (analyze by symbol, period, strategy)
    - Confounder control variables for regression
    """

    def __init__(
        self,
        config: Optional[MultiLogAggregationConfig] = None,
        logger: Optional[AuditLogger] = None
    ):
        """
        Initialize MultiLogAggregator.

        Args:
            config: Aggregation configuration
            logger: Optional audit logger
        """
        self.config = config or MultiLogAggregationConfig()
        self.logger = logger

    def aggregate(
        self,
        trade_dfs: List[pd.DataFrame],
        log_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[pd.DataFrame, AggregationResult]:
        """
        Aggregate multiple trade DataFrames.

        Args:
            trade_dfs: List of trade log DataFrames
            log_metadata: Optional metadata for each log (strategy, period, etc.)

        Returns:
            Tuple of (aggregated DataFrame, AggregationResult)
        """
        if self.logger:
            self.logger.start_section("MULTI_LOG_AGGREGATION")

        if not trade_dfs:
            return pd.DataFrame(), AggregationResult(
                total_trades=0, total_logs=0, unique_symbols=0,
                unique_periods=0, aggregation_mode=self.config.aggregation_mode.value
            )

        # Add metadata tags to each DataFrame
        tagged_dfs = []
        for i, df in enumerate(trade_dfs):
            tagged = df.copy()

            # Add log ID if not present
            if '_log_id' not in tagged.columns:
                tagged['_log_id'] = f"log_{i}"

            # Add metadata from provided list
            if log_metadata and i < len(log_metadata):
                meta = log_metadata[i]
                for key, value in meta.items():
                    tagged[f'_meta_{key}'] = value

            # Extract period from dates if not provided
            if '_meta_period' not in tagged.columns and 'entry_date' in tagged.columns:
                tagged['_meta_period'] = pd.to_datetime(tagged['entry_date']).dt.year

            # Ensure strategy tag
            if '_meta_strategy' not in tagged.columns:
                tagged['_meta_strategy'] = 'unknown'

            tagged_dfs.append(tagged)

        # Combine all DataFrames
        combined = pd.concat(tagged_dfs, ignore_index=True)

        # Ensure unique trade IDs across logs
        if 'trade_id' in combined.columns:
            combined['_original_trade_id'] = combined['trade_id']
            combined['trade_id'] = combined['_log_id'].astype(str) + '_' + combined['trade_id'].astype(str)

        # Add confounder control variables
        combined = self._add_confounder_controls(combined)

        # Compute result statistics
        result = self._compute_aggregation_result(combined, len(trade_dfs))

        if self.logger:
            self.logger.info("Aggregation complete", {
                'total_trades': result.total_trades,
                'logs': result.total_logs,
                'symbols': result.unique_symbols,
                'mode': result.aggregation_mode
            })

            # Check for potential confounding
            if result.warnings:
                for warning in result.warnings:
                    self.logger.warning("Aggregation warning", {'message': warning})

            self.logger.end_section()

        return combined, result

    def _add_confounder_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add encoded control variables for regression."""
        df = df.copy()

        controls = self.config.confounder_controls

        # Symbol fixed effects
        if 'control_symbol_effects' in controls and 'symbol' in df.columns:
            # Create dummy encoding
            symbol_dummies = pd.get_dummies(df['symbol'], prefix='_ctrl_symbol')
            # Drop one category to avoid multicollinearity
            if len(symbol_dummies.columns) > 1:
                symbol_dummies = symbol_dummies.iloc[:, 1:]
            df = pd.concat([df, symbol_dummies], axis=1)

        # Period fixed effects
        if 'control_period_effects' in controls and '_meta_period' in df.columns:
            period_dummies = pd.get_dummies(df['_meta_period'], prefix='_ctrl_period')
            if len(period_dummies.columns) > 1:
                period_dummies = period_dummies.iloc[:, 1:]
            df = pd.concat([df, period_dummies], axis=1)

        # Strategy fixed effects
        if 'control_strategy_effects' in controls and '_meta_strategy' in df.columns:
            strategy_dummies = pd.get_dummies(df['_meta_strategy'], prefix='_ctrl_strategy')
            if len(strategy_dummies.columns) > 1:
                strategy_dummies = strategy_dummies.iloc[:, 1:]
            df = pd.concat([df, strategy_dummies], axis=1)

        return df

    def _compute_aggregation_result(
        self,
        df: pd.DataFrame,
        n_logs: int
    ) -> AggregationResult:
        """Compute aggregation statistics and warnings."""
        warnings = []

        # Symbol distribution
        symbol_dist = {}
        if 'symbol' in df.columns:
            symbol_dist = df['symbol'].value_counts().to_dict()

            # Check for symbol dominance
            total = len(df)
            for symbol, count in symbol_dist.items():
                if count / total > 0.5:
                    warnings.append(
                        f"Symbol '{symbol}' dominates dataset ({count}/{total} = {count/total:.1%}). "
                        "Consider symbol controls in analysis."
                    )

        # Period distribution
        period_dist = {}
        if '_meta_period' in df.columns:
            period_dist = df['_meta_period'].value_counts().to_dict()

        # Strata counts
        strata_counts = {}
        for tag in self.config.metadata_tags:
            col = f'_meta_{tag}' if tag not in df.columns else tag
            if col in df.columns:
                strata_counts[tag] = df[col].nunique()

        return AggregationResult(
            total_trades=len(df),
            total_logs=n_logs,
            unique_symbols=df['symbol'].nunique() if 'symbol' in df.columns else 0,
            unique_periods=df['_meta_period'].nunique() if '_meta_period' in df.columns else 0,
            aggregation_mode=self.config.aggregation_mode.value,
            strata_counts=strata_counts,
            symbol_distribution=symbol_dist,
            period_distribution=period_dist,
            warnings=warnings
        )

    def stratify(
        self,
        df: pd.DataFrame,
        by: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Stratify trades by metadata dimensions.

        Args:
            df: Aggregated trade DataFrame
            by: Dimensions to stratify by (default from config)

        Returns:
            Dictionary mapping stratum name to DataFrame
        """
        stratify_by = by or list(self.config.metadata_tags)

        # Convert to actual column names
        cols = []
        for tag in stratify_by:
            if tag in df.columns:
                cols.append(tag)
            elif f'_meta_{tag}' in df.columns:
                cols.append(f'_meta_{tag}')

        if not cols:
            return {'all': df}

        strata = {}

        if len(cols) == 1:
            # Single dimension stratification
            for value in df[cols[0]].unique():
                stratum_df = df[df[cols[0]] == value].copy()
                strata[f"{cols[0]}={value}"] = stratum_df
        else:
            # Multi-dimension stratification
            for group_keys, group_df in df.groupby(cols):
                if isinstance(group_keys, tuple):
                    stratum_name = '_'.join(f"{c}={v}" for c, v in zip(cols, group_keys))
                else:
                    stratum_name = f"{cols[0]}={group_keys}"
                strata[stratum_name] = group_df.copy()

        if self.logger:
            self.logger.info("Stratification complete", {
                'dimensions': cols,
                'strata_count': len(strata),
                'strata': list(strata.keys())[:10]
            })

        return strata

    def get_stratum_summary(
        self,
        strata: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get summary statistics for each stratum.

        Args:
            strata: Dictionary of stratified DataFrames

        Returns:
            Summary DataFrame
        """
        summaries = []

        for name, df in strata.items():
            summary = {
                'stratum': name,
                'n_trades': len(df),
                'n_good': (df['trade_class'] == 'good').sum() if 'trade_class' in df.columns else 0,
                'n_bad': (df['trade_class'] == 'bad').sum() if 'trade_class' in df.columns else 0,
                'avg_pl_pct': df['pl_pct'].mean() if 'pl_pct' in df.columns else 0,
                'win_rate': (df['pl'] > 0).mean() if 'pl' in df.columns else 0
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)

    def check_class_balance(
        self,
        df: pd.DataFrame,
        min_ratio: float = 0.2
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if trade classes are reasonably balanced.

        Args:
            df: Trade DataFrame with 'trade_class' column
            min_ratio: Minimum acceptable ratio for minority class

        Returns:
            Tuple of (is_balanced, balance_info)
        """
        if 'trade_class' not in df.columns:
            return True, {'message': 'No trade_class column'}

        class_counts = df['trade_class'].value_counts()
        total = len(df)

        balance_info = {
            'class_counts': class_counts.to_dict(),
            'class_ratios': (class_counts / total).to_dict()
        }

        # Check if any class is below minimum ratio
        min_class_ratio = (class_counts / total).min()
        is_balanced = min_class_ratio >= min_ratio

        if not is_balanced:
            balance_info['warning'] = (
                f"Class imbalance detected. Smallest class has only {min_class_ratio:.1%} of trades. "
                "Consider using class weights in analysis."
            )

            if self.logger:
                self.logger.warning("Class imbalance detected", balance_info)

        return is_balanced, balance_info

    def get_control_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of control variable columns.

        Args:
            df: DataFrame with control columns

        Returns:
            List of control column names
        """
        return [c for c in df.columns if c.startswith('_ctrl_')]

    def remove_controls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove control variable columns from DataFrame.

        Args:
            df: DataFrame with control columns

        Returns:
            DataFrame without control columns
        """
        control_cols = self.get_control_columns(df)
        return df.drop(columns=control_cols, errors='ignore')
