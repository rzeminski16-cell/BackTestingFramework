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
        'gross_margin': {'source': 'gross_margin', 'lower_better': False},
        'ebitda_margin': {'source': 'ebitda_margin', 'lower_better': False},
        'fcf_margin': {'source': 'fcf_margin', 'lower_better': False},
        'asset_turnover': {'source': 'asset_turnover', 'lower_better': False},
        'current_ratio': {'source': 'currentratio', 'lower_better': False},
        'debt_to_equity': {'source': 'debt_to_equity', 'lower_better': True},
        'accruals_ratio': {'source': 'accruals_ratio', 'lower_better': True},
    }

    GROWTH_FACTORS = {
        'revenue_growth': {'source': 'revenue_growth_yoy', 'lower_better': False},
        'earnings_growth': {'source': 'earnings_growth_yoy', 'lower_better': False},
        'eps_growth': {'source': 'eps_growth_yoy', 'lower_better': False},
        'fcf_growth': {'source': 'fcf_growth_yoy', 'lower_better': False},
        'earnings_surprise': {'source': 'earnings_surprise', 'lower_better': False},
        'earnings_surprise_pct': {'source': 'surprise_pct', 'lower_better': False},
        'share_count_change': {'source': 'shares_growth_yoy', 'lower_better': True},
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

        # Log available columns and required source columns
        available_cols = list(fundamental_df.columns)
        required_sources = {v['source'].lower() for v in factors_to_compute.values()}
        missing_sources = required_sources - set(available_cols)

        if missing_sources:
            print(f"[WARNING] Missing source columns in fundamental_data: {missing_sources}")
            print(f"[INFO] Available columns: {available_cols}")
            print(f"[INFO] Required sources: {required_sources}")

        if self.logger:
            self.logger.info("Fundamental data columns", {
                "available": available_cols,
                "required_sources": list(required_sources),
                "missing_sources": list(missing_sources)
            })

        results = []
        trades_with_data = 0

        # Debug: Show first few rows of fundamental data
        print(f"[DEBUG] fundamental_df shape: {fundamental_df.shape}")
        print(f"[DEBUG] trades_df shape: {trades_df.shape}")
        if len(fundamental_df) > 0:
            print(f"[DEBUG] First fundamental row columns: {list(fundamental_df.iloc[0].index)}")
            print(f"[DEBUG] factors_to_compute: {list(factors_to_compute.keys())}")
            print(f"[DEBUG] Source columns needed: {[v['source'] for v in factors_to_compute.values()]}")

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
                        if idx == 0:  # Only log once
                            print(f"[DEBUG] Source column '{source_col}' NOT FOUND in fund_row.index")

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

        # Get all factor column names (exclude _trade_idx)
        all_factor_cols = [c for c in factors_df.columns if c != '_trade_idx']

        # Log factor computation results
        print(f"[INFO] Computed {len(all_factor_cols)} fundamental factors: {all_factor_cols}")
        print(f"[INFO] Trades with data: {trades_with_data}/{len(trades_df)}")

        # Count factors by category
        value_count = len([c for c in all_factor_cols if c.startswith('value_')])
        quality_count = len([c for c in all_factor_cols if c.startswith('quality_')])
        growth_count = len([c for c in all_factor_cols if c.startswith('growth_')])
        eps_count = len([c for c in all_factor_cols if c.startswith('eps_')])

        result = FundamentalFactorResult(
            value_factors=value_count,
            quality_factors=quality_count,
            growth_factors=growth_count + eps_count,  # EPS factors are growth-related
            trades_with_data=trades_with_data,
            total_trades=len(trades_df),
            factor_names=all_factor_cols
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
        Compute derived quality factors, preferring the point-in-time TTM ratios
        produced by Classes/FactorAnalysis/data/fundamental_panel.py and falling
        back to single-quarter raw values only when TTM columns are absent.

        (debt_to_equity / margins / ROE / ROA / growth are mapped directly as
        factor sources elsewhere; here we add gross margin, free cash flow and
        an earnings-quality accruals proxy.)
        """
        derived = {}

        def val(*names):
            for n in names:
                if n in fund_row.index and pd.notna(fund_row.get(n)):
                    return float(fund_row[n])
            return None

        # Gross margin (prefer TTM from the panel).
        gross_margin = val('gross_margin')
        if gross_margin is None:
            gp = val('gross_profit_ttm', 'grossprofit')
            rev = val('revenue_ttm', 'totalrevenue')
            if gp is not None and rev:
                gross_margin = gp / rev * 100
        if gross_margin is not None:
            derived['quality_gross_margin'] = gross_margin

        # Free cash flow (prefer TTM from the panel).
        fcf = val('freecashflow', 'free_cash_flow')
        if fcf is None:
            ocf = val('operating_cash_flow_ttm', 'operatingcashflow')
            capex = val('capex_ttm', 'capitalexpenditures')
            if ocf is not None and capex is not None:
                fcf = ocf - abs(capex)
        if fcf is not None:
            derived['quality_free_cash_flow'] = fcf

        # Earnings quality: TTM operating cash flow / TTM net income (accruals proxy).
        ocf_ttm = val('operating_cash_flow_ttm', 'operatingcashflow')
        ni_ttm = val('net_income_ttm', 'netincome')
        if ocf_ttm is not None and ni_ttm not in (None, 0):
            derived['quality_earnings_quality'] = ocf_ttm / ni_ttm

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

        def _composite(prefix: str, factor_map: Dict) -> Optional[pd.Series]:
            """Z-score-normalize a factor category and average into a composite.

            Robust to all-NaN / zero-variance columns (those contribute NaN rather
            than crashing), and skips categories with no usable data entirely.
            """
            cols = [
                c for c in df.columns
                if c.startswith(prefix)
                and pd.api.types.is_numeric_dtype(df[c])
                and df[c].notna().any()
            ]
            if not cols:
                return None

            def zscore(col):
                std = col.std()
                if std and std > 0:
                    return (col - col.mean()) / std
                return col * 0.0  # preserves the Series shape (NaN stays NaN)

            normalized = df[cols].apply(zscore)
            if isinstance(normalized, pd.Series):
                normalized = normalized.to_frame()

            # Invert "lower is better" factors so higher composite = better.
            for col in cols:
                factor_name = col[len(prefix):]
                if factor_name in factor_map and factor_map[factor_name].get('lower_better'):
                    normalized[col] = -normalized[col]

            return normalized[cols].mean(axis=1) if len(cols) > 1 else normalized[cols[0]]

        for prefix, factor_map, name in (
            ('value_', self.VALUE_FACTORS, 'composite_value'),
            ('quality_', self.QUALITY_FACTORS, 'composite_quality'),
            ('growth_', self.GROWTH_FACTORS, 'composite_growth'),
            ('eps_', self.EPS_FACTORS, 'composite_eps'),
        ):
            composite = _composite(prefix, factor_map)
            if composite is not None:
                df[name] = composite

        # Overall fundamental score = mean of available category composites.
        composite_cols = [c for c in ('composite_value', 'composite_quality',
                                      'composite_growth', 'composite_eps') if c in df.columns]
        if composite_cols:
            df['composite_fundamental'] = (
                df[composite_cols].mean(axis=1) if len(composite_cols) > 1 else df[composite_cols[0]]
            )

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

        # Merge with trades - use positional assignment since _trade_idx is positional
        trades_df = trades_df.copy()
        trades_df = trades_df.reset_index(drop=True)  # Reset to 0-based index for alignment

        cols_added = []
        for col in result_df.columns:
            if col != '_trade_idx' and col not in trades_df.columns:
                # Use positional assignment - result_df rows align with trades_df rows by position
                trades_df[col] = result_df[col].values
                cols_added.append(col)

        print(f"[INFO] Added {len(cols_added)} factor columns to trades_df: {cols_added}")
        print(f"[DEBUG] trades_df columns with eps_: {[c for c in trades_df.columns if c.startswith('eps_')]}")
        print(f"[DEBUG] trades_df shape before composite: {trades_df.shape}")

        # Create composite scores
        try:
            trades_df = self.create_composite_scores(trades_df)
            print(f"[DEBUG] trades_df shape after composite: {trades_df.shape}")
        except Exception as e:
            print(f"[ERROR] create_composite_scores failed: {e}")
            import traceback
            traceback.print_exc()
            raise

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
