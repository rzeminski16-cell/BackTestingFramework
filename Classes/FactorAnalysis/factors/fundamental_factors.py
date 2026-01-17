"""
Fundamental Factors for Factor Analysis.

Computes value, quality, and growth factors from fundamental data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..logging.audit_logger import AuditLogger


@dataclass
class FundamentalFactorResult:
    """Result of fundamental factor computation."""
    value_factors: int
    quality_factors: int
    growth_factors: int
    trades_with_data: int
    total_trades: int
    factor_names: List[str]


class FundamentalFactors:
    """
    Computes fundamental factors from financial data.

    Factor Categories:
    - Value: P/E, P/B, P/S, dividend yield, PEG
    - Quality: ROE, ROA, current ratio, debt/equity, profit margins
    - Growth: Revenue growth, earnings growth, earnings surprise
    - EPS-Only: EPS-related factors only (for use when fundamental data is sparse)
    """

    VALUE_FACTORS = {
        'pe_ratio': {'source': 'pe_ratio', 'lower_better': True},
        'price_to_book': {'source': 'price_to_book', 'lower_better': True},
        'price_to_sales': {'source': 'price_to_sales_ttm', 'lower_better': True},
        'peg_ratio': {'source': 'peg_ratio', 'lower_better': True},
        'dividend_yield': {'source': 'dividend_yield', 'lower_better': False},
        'ev_to_ebitda': {'source': 'ev_to_ebitda', 'lower_better': True},
    }

    QUALITY_FACTORS = {
        'return_on_equity': {'source': 'return_on_equity_ttm', 'lower_better': False},
        'return_on_assets': {'source': 'return_on_assets_ttm', 'lower_better': False},
        'profit_margin': {'source': 'profit_margin', 'lower_better': False},
        'operating_margin': {'source': 'operating_margin_ttm', 'lower_better': False},
        'current_ratio': {'source': 'currentratio', 'lower_better': False},
        'debt_to_equity': {'source': 'debt_to_equity', 'lower_better': True},
    }

    GROWTH_FACTORS = {
        'revenue_growth': {'source': 'revenue_growth_yoy', 'lower_better': False},
        'earnings_growth': {'source': 'earnings_growth_yoy', 'lower_better': False},
        'earnings_surprise': {'source': 'earnings_surprise', 'lower_better': False},
        'earnings_surprise_pct': {'source': 'surprise_pct', 'lower_better': False},
    }

    # EPS-related factors only - use when fundamental data is mostly missing
    # These factors focus on earnings per share and earnings surprise data
    EPS_FACTORS = {
        'eps': {'source': 'eps', 'lower_better': False},
        'estimated_eps': {'source': 'estimated_eps', 'lower_better': False},
        'earnings_growth': {'source': 'earnings_growth_yoy', 'lower_better': False},
        'earnings_surprise': {'source': 'earnings_surprise', 'lower_better': False},
        'earnings_surprise_pct': {'source': 'surprise_pct', 'lower_better': False},
    }

    def __init__(self, logger: Optional[AuditLogger] = None):
        """
        Initialize FundamentalFactors.

        Args:
            logger: Optional audit logger
        """
        self.logger = logger

    def compute_factors(
        self,
        trades_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        include_value: bool = True,
        include_quality: bool = True,
        include_growth: bool = True,
        eps_only: bool = True
    ) -> Tuple[pd.DataFrame, FundamentalFactorResult]:
        """
        Compute fundamental factors for each trade.

        Args:
            trades_df: Trade log DataFrame
            fundamental_df: Aligned fundamental data DataFrame
            include_value: Whether to include value factors (ignored if eps_only=True)
            include_quality: Whether to include quality factors (ignored if eps_only=True)
            include_growth: Whether to include growth factors (ignored if eps_only=True)
            eps_only: If True, only include EPS-related factors (eps, estimated_eps,
                     earnings_growth, earnings_surprise). This is the default because
                     fundamental data is often missing for most periods.

        Returns:
            Tuple of (DataFrame with factors, FundamentalFactorResult)
        """
        if self.logger:
            self.logger.start_section("FUNDAMENTAL_FACTORS")

        # Determine which factors to compute
        factors_to_compute = {}

        if eps_only:
            # Only include EPS-related factors
            factors_to_compute.update({f'eps_{k}': v for k, v in self.EPS_FACTORS.items()})
            if self.logger:
                self.logger.info("Using EPS-only mode for fundamental factors", {
                    "reason": "Fundamental data is mostly missing for whole periods",
                    "factors": list(self.EPS_FACTORS.keys())
                })
        else:
            # Include traditional factor categories
            if include_value:
                factors_to_compute.update({f'value_{k}': v for k, v in self.VALUE_FACTORS.items()})
            if include_quality:
                factors_to_compute.update({f'quality_{k}': v for k, v in self.QUALITY_FACTORS.items()})
            if include_growth:
                factors_to_compute.update({f'growth_{k}': v for k, v in self.GROWTH_FACTORS.items()})

        # Normalize column names in fundamental data
        fundamental_df = fundamental_df.copy()
        fundamental_df.columns = [c.lower().strip() for c in fundamental_df.columns]

        results = []
        trades_with_data = 0

        for idx in range(len(trades_df)):
            row_data = {'_trade_idx': idx}
            has_any_data = False

            # Get corresponding fundamental row
            if idx < len(fundamental_df):
                fund_row = fundamental_df.iloc[idx]

                for factor_name, config in factors_to_compute.items():
                    source_col = config['source'].lower()

                    # Try to find the source column
                    if source_col in fund_row.index:
                        value = fund_row[source_col]
                        if pd.notna(value):
                            row_data[factor_name] = float(value)
                            has_any_data = True
                        else:
                            row_data[factor_name] = np.nan
                    else:
                        row_data[factor_name] = np.nan

                # Compute derived factors
                derived = self._compute_derived_factors(fund_row)
                row_data.update(derived)
                if any(pd.notna(v) for v in derived.values()):
                    has_any_data = True

            else:
                # No fundamental data for this trade
                for factor_name in factors_to_compute:
                    row_data[factor_name] = np.nan

            if has_any_data:
                trades_with_data += 1

            results.append(row_data)

        factors_df = pd.DataFrame(results)

        # Count factors by category
        value_count = len([c for c in factors_df.columns if c.startswith('value_')])
        quality_count = len([c for c in factors_df.columns if c.startswith('quality_')])
        growth_count = len([c for c in factors_df.columns if c.startswith('growth_')])
        eps_count = len([c for c in factors_df.columns if c.startswith('eps_')])

        result = FundamentalFactorResult(
            value_factors=value_count,
            quality_factors=quality_count,
            growth_factors=growth_count + eps_count,  # EPS factors are growth-related
            trades_with_data=trades_with_data,
            total_trades=len(trades_df),
            factor_names=list(factors_to_compute.keys())
        )

        if self.logger:
            self.logger.log_factor_engineering(
                category='Fundamental',
                factor_count=value_count + quality_count + growth_count + eps_count,
                trades_with_data=trades_with_data,
                total_trades=len(trades_df)
            )
            self.logger.info("Factor breakdown", {
                'value': value_count,
                'quality': quality_count,
                'growth': growth_count,
                'eps_only': eps_count
            })
            self.logger.end_section()

        return factors_df, result

    def _compute_derived_factors(self, fund_row: pd.Series) -> Dict[str, float]:
        """
        Compute derived fundamental factors.

        Args:
            fund_row: Series with fundamental data

        Returns:
            Dictionary of derived factor values
        """
        derived = {}

        # Debt-to-equity (if not present)
        if 'debt_to_equity' not in fund_row.index or pd.isna(fund_row.get('debt_to_equity')):
            liabilities = fund_row.get('totalliabilities', fund_row.get('total_liabilities'))
            equity = fund_row.get('totalshareholderequity', fund_row.get('total_shareholder_equity'))
            if pd.notna(liabilities) and pd.notna(equity) and equity != 0:
                derived['quality_debt_to_equity_derived'] = float(liabilities) / float(equity)

        # Gross margin
        gross_profit = fund_row.get('gross_profit_ttm', fund_row.get('grossprofit'))
        revenue = fund_row.get('revenue_ttm', fund_row.get('totalrevenue'))
        if pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0:
            derived['quality_gross_margin'] = float(gross_profit) / float(revenue)

        # FCF yield (simplified - without market cap)
        operating_cf = fund_row.get('operatingcashflow')
        capex = fund_row.get('capitalexpenditures')
        if pd.notna(operating_cf) and pd.notna(capex):
            derived['quality_free_cash_flow'] = float(operating_cf) - abs(float(capex))

        # Earnings quality (accruals ratio proxy)
        net_income = fund_row.get('netincome')
        if pd.notna(operating_cf) and pd.notna(net_income) and net_income != 0:
            derived['quality_earnings_quality'] = float(operating_cf) / float(net_income)

        return derived

    def create_composite_scores(
        self,
        factors_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Create composite factor scores.

        Args:
            factors_df: DataFrame with fundamental factors
            weights: Optional custom weights for factors

        Returns:
            DataFrame with added composite scores
        """
        df = factors_df.copy()

        # Default equal weights within categories
        if weights is None:
            weights = {}

        # Value composite
        value_cols = [c for c in df.columns if c.startswith('value_') and pd.api.types.is_numeric_dtype(df[c])]
        if value_cols:
            # Normalize each factor (z-score)
            value_normalized = df[value_cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            # Some factors are "lower is better" - invert them
            for col in value_cols:
                factor_name = col.replace('value_', '')
                if factor_name in self.VALUE_FACTORS and self.VALUE_FACTORS[factor_name].get('lower_better'):
                    value_normalized[col] = -value_normalized[col]
            df['composite_value'] = value_normalized.mean(axis=1)

        # Quality composite
        quality_cols = [c for c in df.columns if c.startswith('quality_') and pd.api.types.is_numeric_dtype(df[c])]
        if quality_cols:
            quality_normalized = df[quality_cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            for col in quality_cols:
                factor_name = col.replace('quality_', '')
                if factor_name in self.QUALITY_FACTORS and self.QUALITY_FACTORS[factor_name].get('lower_better'):
                    quality_normalized[col] = -quality_normalized[col]
            df['composite_quality'] = quality_normalized.mean(axis=1)

        # Growth composite
        growth_cols = [c for c in df.columns if c.startswith('growth_') and pd.api.types.is_numeric_dtype(df[c])]
        if growth_cols:
            growth_normalized = df[growth_cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            df['composite_growth'] = growth_normalized.mean(axis=1)

        # EPS composite (for eps_only mode)
        eps_cols = [c for c in df.columns if c.startswith('eps_') and pd.api.types.is_numeric_dtype(df[c])]
        if eps_cols:
            eps_normalized = df[eps_cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            # Apply lower_better inversions for EPS factors
            for col in eps_cols:
                factor_name = col.replace('eps_', '')
                if factor_name in self.EPS_FACTORS and self.EPS_FACTORS[factor_name].get('lower_better'):
                    eps_normalized[col] = -eps_normalized[col]
            df['composite_eps'] = eps_normalized.mean(axis=1)

        # Overall fundamental score
        composite_cols = ['composite_value', 'composite_quality', 'composite_growth', 'composite_eps']
        available_composites = [c for c in composite_cols if c in df.columns]
        if available_composites:
            df['composite_fundamental'] = df[available_composites].mean(axis=1)

        return df

    def compute_all(
        self,
        trades_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        eps_only: bool = True
    ) -> pd.DataFrame:
        """
        Compute all fundamental factors for trades.

        This is the main entry point for the analyzer.

        Args:
            trades_df: Trade log DataFrame with 'symbol' and 'entry_date'
            fundamental_df: Fundamental data (already aligned to trade dates)
            eps_only: If True, only include EPS-related factors (default True).
                     Set to False to include all fundamental factors (value, quality, growth).

        Returns:
            trades_df with fundamental factor columns added
        """
        # Compute factors
        result_df, result = self.compute_factors(trades_df, fundamental_df, eps_only=eps_only)

        # Store factor names
        self._factor_names = result.factor_names.copy()

        # Merge with trades
        trades_df = trades_df.copy()
        for col in result_df.columns:
            if col != '_trade_idx' and col not in trades_df.columns:
                if '_trade_idx' in result_df.columns:
                    trades_df[col] = result_df.set_index('_trade_idx')[col]
                else:
                    trades_df[col] = result_df[col].values

        # Create composite scores
        trades_df = self.create_composite_scores(trades_df)

        # Add composite names to factor list
        composite_cols = [c for c in trades_df.columns if c.startswith('composite_')]
        self._factor_names.extend(composite_cols)

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
