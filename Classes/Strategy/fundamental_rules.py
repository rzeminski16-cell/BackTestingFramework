"""
Fundamental rules framework for strategy entry filtering.

Fundamental rules are the first filter applied before any trade entry.
They use fundamental data (EPS, growth, institutional ownership, etc.)
to determine if a security meets the requirements for trading.

Rules can have different parameters/thresholds based on:
- Sector of the security
- Market capitalization
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd

from .strategy_context import StrategyContext


class MarketCapCategory(Enum):
    """Market capitalization categories."""
    MICRO = "MICRO"        # < $300M
    SMALL = "SMALL"        # $300M - $2B
    MID = "MID"            # $2B - $10B
    LARGE = "LARGE"        # $10B - $200B
    MEGA = "MEGA"          # > $200B
    UNKNOWN = "UNKNOWN"    # Not classified


@dataclass
class FundamentalData:
    """
    Container for fundamental data of a security.

    All values should come from raw data CSV files.
    """
    # Earnings metrics
    eps: Optional[float] = None                    # Earnings per share
    eps_growth_yoy: Optional[float] = None         # YoY EPS growth %
    eps_growth_qoq: Optional[float] = None         # QoQ EPS growth %

    # Revenue metrics
    revenue: Optional[float] = None                # Revenue
    revenue_growth_yoy: Optional[float] = None     # YoY revenue growth %
    revenue_growth_qoq: Optional[float] = None     # QoQ revenue growth %

    # Profitability metrics
    gross_margin: Optional[float] = None           # Gross margin %
    operating_margin: Optional[float] = None       # Operating margin %
    net_margin: Optional[float] = None             # Net profit margin %
    roe: Optional[float] = None                    # Return on equity %
    roa: Optional[float] = None                    # Return on assets %

    # Valuation metrics
    pe_ratio: Optional[float] = None               # Price to earnings
    pb_ratio: Optional[float] = None               # Price to book
    ps_ratio: Optional[float] = None               # Price to sales
    peg_ratio: Optional[float] = None              # PEG ratio

    # Ownership metrics
    institutional_ownership: Optional[float] = None  # Institutional ownership %
    insider_ownership: Optional[float] = None        # Insider ownership %
    short_interest: Optional[float] = None           # Short interest %

    # Balance sheet metrics
    debt_to_equity: Optional[float] = None         # Debt to equity ratio
    current_ratio: Optional[float] = None          # Current ratio
    quick_ratio: Optional[float] = None            # Quick ratio

    # Other
    market_cap: Optional[float] = None             # Market capitalization
    sector: Optional[str] = None                   # Sector classification
    industry: Optional[str] = None                 # Industry classification

    @classmethod
    def from_bar(cls, bar: pd.Series) -> 'FundamentalData':
        """
        Create FundamentalData from a data bar.

        Looks for fundamental data columns in the bar and populates
        the corresponding fields.

        Args:
            bar: Data bar (row from DataFrame)

        Returns:
            FundamentalData instance
        """
        return cls(
            eps=bar.get('eps'),
            eps_growth_yoy=bar.get('eps_growth_yoy'),
            eps_growth_qoq=bar.get('eps_growth_qoq'),
            revenue=bar.get('revenue'),
            revenue_growth_yoy=bar.get('revenue_growth_yoy'),
            revenue_growth_qoq=bar.get('revenue_growth_qoq'),
            gross_margin=bar.get('gross_margin'),
            operating_margin=bar.get('operating_margin'),
            net_margin=bar.get('net_margin'),
            roe=bar.get('roe'),
            roa=bar.get('roa'),
            pe_ratio=bar.get('pe_ratio'),
            pb_ratio=bar.get('pb_ratio'),
            ps_ratio=bar.get('ps_ratio'),
            peg_ratio=bar.get('peg_ratio'),
            institutional_ownership=bar.get('institutional_ownership'),
            insider_ownership=bar.get('insider_ownership'),
            short_interest=bar.get('short_interest'),
            debt_to_equity=bar.get('debt_to_equity'),
            current_ratio=bar.get('current_ratio'),
            quick_ratio=bar.get('quick_ratio'),
            market_cap=bar.get('market_cap'),
            sector=bar.get('sector'),
            industry=bar.get('industry'),
        )

    def get_market_cap_category(self) -> MarketCapCategory:
        """
        Determine market cap category.

        Returns:
            MarketCapCategory based on market_cap value
        """
        if self.market_cap is None:
            return MarketCapCategory.UNKNOWN

        if self.market_cap < 300_000_000:
            return MarketCapCategory.MICRO
        elif self.market_cap < 2_000_000_000:
            return MarketCapCategory.SMALL
        elif self.market_cap < 10_000_000_000:
            return MarketCapCategory.MID
        elif self.market_cap < 200_000_000_000:
            return MarketCapCategory.LARGE
        else:
            return MarketCapCategory.MEGA


@dataclass
class FundamentalCheckResult:
    """Result of a fundamental rules check."""
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BaseFundamentalRules(ABC):
    """
    Abstract base class for fundamental rules.

    Implement this class to define fundamental filters for your strategy.
    The check_fundamentals() method is called before every potential entry
    and must return True for the trade to proceed.

    Rules can be differentiated by sector and market cap by implementing
    get_rules_for_category().
    """

    def __init__(self, **params):
        """
        Initialize fundamental rules with parameters.

        Args:
            **params: Rule-specific parameters
        """
        self.params = params

    @abstractmethod
    def check_fundamentals(
        self,
        context: StrategyContext,
        fundamentals: FundamentalData
    ) -> FundamentalCheckResult:
        """
        Check if fundamentals meet requirements for trade entry.

        This method is called every time there is a potential entry signal.
        If it returns False, the trade will not be executed.

        Args:
            context: Current strategy context
            fundamentals: Fundamental data for the security

        Returns:
            FundamentalCheckResult with passed=True/False and reason
        """
        pass

    def get_rules_for_category(
        self,
        sector: Optional[str],
        market_cap_category: MarketCapCategory
    ) -> Dict[str, Any]:
        """
        Get rule parameters specific to sector and market cap.

        Override this method to return different thresholds based on
        the security's sector and market cap category.

        Default: Returns the base params unchanged.

        Args:
            sector: Security sector (e.g., "Technology", "Healthcare")
            market_cap_category: Market cap category

        Returns:
            Dictionary of rule parameters for this category
        """
        return self.params.copy()

    def required_fundamental_columns(self) -> List[str]:
        """
        Return list of required fundamental data columns.

        Override to specify which fundamental data columns must be
        present in the raw data for this rule set to work.

        Default: Returns empty list (no requirements).

        Returns:
            List of required column names
        """
        return []


class AlwaysPassFundamentalRules(BaseFundamentalRules):
    """
    Fundamental rules that always pass.

    Use this when a strategy does not require fundamental filtering.
    """

    def check_fundamentals(
        self,
        context: StrategyContext,
        fundamentals: FundamentalData
    ) -> FundamentalCheckResult:
        """Always returns passed=True."""
        return FundamentalCheckResult(
            passed=True,
            reason="No fundamental filters applied"
        )


class SimpleFundamentalRules(BaseFundamentalRules):
    """
    Simple fundamental rules with configurable thresholds.

    Example implementation that checks common fundamental metrics.
    """

    def __init__(
        self,
        min_eps: Optional[float] = None,
        min_eps_growth: Optional[float] = None,
        min_revenue_growth: Optional[float] = None,
        max_pe_ratio: Optional[float] = None,
        min_institutional_ownership: Optional[float] = None,
        max_debt_to_equity: Optional[float] = None,
        **params
    ):
        """
        Initialize with threshold parameters.

        Args:
            min_eps: Minimum EPS required
            min_eps_growth: Minimum EPS growth % required
            min_revenue_growth: Minimum revenue growth % required
            max_pe_ratio: Maximum P/E ratio allowed
            min_institutional_ownership: Minimum institutional ownership %
            max_debt_to_equity: Maximum debt to equity ratio
            **params: Additional parameters
        """
        super().__init__(**params)
        self.min_eps = min_eps
        self.min_eps_growth = min_eps_growth
        self.min_revenue_growth = min_revenue_growth
        self.max_pe_ratio = max_pe_ratio
        self.min_institutional_ownership = min_institutional_ownership
        self.max_debt_to_equity = max_debt_to_equity

    def check_fundamentals(
        self,
        context: StrategyContext,
        fundamentals: FundamentalData
    ) -> FundamentalCheckResult:
        """
        Check if fundamentals meet threshold requirements.
        """
        failed_checks = []

        # Check EPS
        if self.min_eps is not None and fundamentals.eps is not None:
            if fundamentals.eps < self.min_eps:
                failed_checks.append(
                    f"EPS {fundamentals.eps:.2f} < min {self.min_eps:.2f}"
                )

        # Check EPS growth
        if self.min_eps_growth is not None and fundamentals.eps_growth_yoy is not None:
            if fundamentals.eps_growth_yoy < self.min_eps_growth:
                failed_checks.append(
                    f"EPS growth {fundamentals.eps_growth_yoy:.1f}% < min {self.min_eps_growth:.1f}%"
                )

        # Check revenue growth
        if self.min_revenue_growth is not None and fundamentals.revenue_growth_yoy is not None:
            if fundamentals.revenue_growth_yoy < self.min_revenue_growth:
                failed_checks.append(
                    f"Revenue growth {fundamentals.revenue_growth_yoy:.1f}% < min {self.min_revenue_growth:.1f}%"
                )

        # Check P/E ratio
        if self.max_pe_ratio is not None and fundamentals.pe_ratio is not None:
            if fundamentals.pe_ratio > self.max_pe_ratio:
                failed_checks.append(
                    f"P/E {fundamentals.pe_ratio:.1f} > max {self.max_pe_ratio:.1f}"
                )

        # Check institutional ownership
        if self.min_institutional_ownership is not None and fundamentals.institutional_ownership is not None:
            if fundamentals.institutional_ownership < self.min_institutional_ownership:
                failed_checks.append(
                    f"Inst. ownership {fundamentals.institutional_ownership:.1f}% < min {self.min_institutional_ownership:.1f}%"
                )

        # Check debt to equity
        if self.max_debt_to_equity is not None and fundamentals.debt_to_equity is not None:
            if fundamentals.debt_to_equity > self.max_debt_to_equity:
                failed_checks.append(
                    f"D/E {fundamentals.debt_to_equity:.2f} > max {self.max_debt_to_equity:.2f}"
                )

        if failed_checks:
            return FundamentalCheckResult(
                passed=False,
                reason="; ".join(failed_checks),
                details={"failed_checks": failed_checks}
            )

        return FundamentalCheckResult(
            passed=True,
            reason="All fundamental checks passed"
        )

    def required_fundamental_columns(self) -> List[str]:
        """Return columns needed based on configured thresholds."""
        columns = []
        if self.min_eps is not None:
            columns.append('eps')
        if self.min_eps_growth is not None:
            columns.append('eps_growth_yoy')
        if self.min_revenue_growth is not None:
            columns.append('revenue_growth_yoy')
        if self.max_pe_ratio is not None:
            columns.append('pe_ratio')
        if self.min_institutional_ownership is not None:
            columns.append('institutional_ownership')
        if self.max_debt_to_equity is not None:
            columns.append('debt_to_equity')
        return columns
