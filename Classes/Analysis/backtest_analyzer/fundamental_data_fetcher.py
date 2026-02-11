"""
Fundamental data fetcher using Alpha Vantage API - Unified Quarterly Data.

This module fetches and unifies fundamental metrics from multiple sources into
a single quarterly timeline with one unique row per report publication date.

Key Features:
- Uses earnings reportedDate as the master timeline (actual publication date)
- Maps all financial statements (income, balance, cash flow) to nearest earnings date
- One unique row per quarterly report date
- Forward-fills missing data to ensure completeness
- Quarterly data only (no annual reports)

Core Metrics Included:
- EPS (TTM) Diluted, Growth, Surprise Trend
- Revenue (TTM) and Growth
- Operating Margin, Gross Margin, Profit Margin
- P/E Ratio (Forward & Trailing), PEG Ratio
- Price/Book, Price/Cash Flow
- Free Cash Flow (TTM), FCF Trend, FCF Yield
- Debt-to-Equity Ratio, Current Ratio
- EBITDA, ROE, ROA, Beta
- Analyst Target Price
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .alpha_vantage_client import AlphaVantageClient
from .alpha_vantage_config import AlphaVantageConfig
from .interactive_handler import InteractiveHandler


logger = logging.getLogger(__name__)


# Number of quarters to look back for "recent" Forward P/E
RECENT_QUARTERS_FOR_FORWARD_PE = 8  # ~2 years


@dataclass
class QuarterData:
    """Data for a single quarter."""
    report_date: datetime  # Actual publication date (from earnings)
    fiscal_quarter_end: datetime  # Fiscal period ending date
    year: int  # Calendar year
    quarter: int  # Calendar quarter (1-4)
    metrics: Dict[str, Any] = field(default_factory=dict)


def parse_float(value: Any) -> Optional[float]:
    """Safely parse a value to float."""
    if value is None or value == 'None' or value == '-':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def get_quarter_end_date(year: int, quarter: int) -> datetime:
    """Get the end date for a calendar quarter."""
    quarter_ends = {
        1: (3, 31),   # Q1: March 31
        2: (6, 30),   # Q2: June 30
        3: (9, 30),   # Q3: September 30
        4: (12, 31),  # Q4: December 31
    }
    month, day = quarter_ends[quarter]
    return datetime(year, month, day)


def get_quarter_from_date(date: datetime) -> Tuple[int, int]:
    """Get (year, quarter) from a date."""
    quarter = (date.month - 1) // 3 + 1
    return date.year, quarter


class FundamentalDataFetcher:
    """
    Fetches and unifies fundamental metrics for securities.

    Creates a single quarterly timeline based on earnings publication dates,
    with all financial data from multiple sources unified into one row per quarter.

    Usage:
        config = AlphaVantageConfig.load()
        client = AlphaVantageClient(config)
        handler = InteractiveHandler(log_dir=Path("logs"))

        fetcher = FundamentalDataFetcher(client, handler)
        data = fetcher.fetch_fundamental_data("AAPL", start_year=2020, end_year=2024)
    """

    def __init__(self,
                 client: AlphaVantageClient,
                 interactive_handler: InteractiveHandler,
                 price_data_dir: Optional[Path] = None):
        """
        Initialize the fetcher.

        Args:
            client: Alpha Vantage API client
            interactive_handler: Handler for user prompts
            price_data_dir: Directory containing historical price CSVs (for P/E calculation)
        """
        self.client = client
        self.handler = interactive_handler
        self.price_data_dir = Path(price_data_dir) if price_data_dir else None

        # Cache for raw API data
        self._raw_data_cache: Dict[str, Dict] = {}

        # Cache for historical prices
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _get_raw_data(self, symbol: str) -> Dict[str, Any]:
        """Get all raw API data for a symbol (with caching)."""
        if symbol not in self._raw_data_cache:
            self._raw_data_cache[symbol] = self.client.get_all_fundamental_data(symbol)
        return self._raw_data_cache[symbol]

    def _get_historical_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get historical daily prices for a symbol.

        First tries local price data directory, then Alpha Vantage API.
        """
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        # Try local file first
        if self.price_data_dir:
            local_file = self.price_data_dir / f"{symbol}.csv"
            if local_file.exists():
                try:
                    df = pd.read_csv(local_file)
                    df.columns = df.columns.str.lower().str.strip()

                    if 'time' in df.columns and 'date' not in df.columns:
                        df.rename(columns={'time': 'date'}, inplace=True)

                    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
                    df.sort_values('date', inplace=True)
                    df.set_index('date', inplace=True)

                    self._price_cache[symbol] = df
                    return df

                except Exception as e:
                    logger.warning(f"Error loading local price data for {symbol}: {e}")

        # Fall back to Alpha Vantage
        try:
            data = self.client.get_daily_prices(symbol, outputsize='full')
            time_series = data.get('Time Series (Daily)', {})

            if not time_series:
                logger.warning(f"No price data from Alpha Vantage for {symbol}")
                return None

            # Convert to DataFrame
            records = []
            for date_str, values in time_series.items():
                records.append({
                    'date': pd.to_datetime(date_str),
                    'open': parse_float(values.get('1. open')),
                    'high': parse_float(values.get('2. high')),
                    'low': parse_float(values.get('3. low')),
                    'close': parse_float(values.get('4. close')),
                    'adjusted_close': parse_float(values.get('5. adjusted close')),
                    'volume': parse_float(values.get('6. volume')),
                })

            df = pd.DataFrame(records)
            df.sort_values('date', inplace=True)
            df.set_index('date', inplace=True)

            self._price_cache[symbol] = df
            return df

        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None

    def _get_price_at_date(self, symbol: str, target_date: datetime) -> Optional[float]:
        """Get the closing price at or near a specific date."""
        prices = self._get_historical_prices(symbol)
        if prices is None or prices.empty:
            return None

        # Find closest date at or before target
        target_ts = pd.Timestamp(target_date)
        valid_prices = prices[prices.index <= target_ts]

        if valid_prices.empty:
            return None

        # Get the closest date
        closest_date = valid_prices.index[-1]
        return valid_prices.loc[closest_date, 'close']

    def _find_quarterly_report(self,
                               reports: List[Dict],
                               target_date: datetime,
                               tolerance_days: int = 45) -> Optional[Dict]:
        """
        Find a quarterly report closest to a target date.

        Args:
            reports: List of quarterly reports from API
            target_date: Target quarter end date
            tolerance_days: Maximum days difference to accept

        Returns:
            Matching report or None
        """
        if not reports:
            return None

        best_match = None
        best_diff = timedelta(days=tolerance_days + 1)

        for report in reports:
            fiscal_date_str = report.get('fiscalDateEnding')
            if not fiscal_date_str:
                continue

            try:
                fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d')
                diff = abs(fiscal_date - target_date)

                if diff < best_diff:
                    best_diff = diff
                    best_match = report

            except ValueError:
                continue

        if best_diff <= timedelta(days=tolerance_days):
            return best_match

        return None

    def _calculate_ttm_sum(self,
                           reports: List[Dict],
                           field_name: str,
                           as_of_date: datetime) -> Optional[float]:
        """
        Calculate trailing twelve months sum for a field.

        Args:
            reports: List of quarterly reports (should be sorted newest first)
            field_name: Field to sum
            as_of_date: Calculate TTM as of this date

        Returns:
            TTM sum or None if insufficient data
        """
        # Find 4 quarters ending at or before as_of_date
        relevant_reports = []

        for report in reports:
            fiscal_date_str = report.get('fiscalDateEnding')
            if not fiscal_date_str:
                continue

            try:
                fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d')
                if fiscal_date <= as_of_date:
                    relevant_reports.append(report)

                if len(relevant_reports) >= 4:
                    break

            except ValueError:
                continue

        if len(relevant_reports) < 4:
            return None

        # Sum the field
        total = 0.0
        for report in relevant_reports[:4]:
            value = parse_float(report.get(field_name))
            if value is None:
                return None
            total += value

        return total

    def _calculate_yoy_growth(self,
                              current_value: Optional[float],
                              prior_value: Optional[float]) -> Optional[float]:
        """Calculate year-over-year growth percentage."""
        if current_value is None or prior_value is None:
            return None

        if prior_value == 0:
            return None

        # Use absolute value for denominator to handle negative prior values
        return ((current_value - prior_value) / abs(prior_value)) * 100

    def _get_eps_surprise_trend(self,
                                earnings_data: Dict,
                                as_of_date: datetime,
                                num_quarters: int = 4) -> Optional[str]:
        """
        Get EPS surprise trend for last N quarters.

        Returns string like "3/4 Beat" or "2/4 Beat"
        """
        quarterly_earnings = earnings_data.get('quarterlyEarnings', [])
        if not quarterly_earnings:
            return None

        surprises = []
        for earning in quarterly_earnings:
            fiscal_date_str = earning.get('fiscalDateEnding')
            if not fiscal_date_str:
                continue

            try:
                fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d')
                if fiscal_date > as_of_date:
                    continue

                surprise = parse_float(earning.get('surprise'))
                if surprise is not None:
                    surprises.append('Beat' if surprise >= 0 else 'Miss')

                if len(surprises) >= num_quarters:
                    break

            except ValueError:
                continue

        if not surprises:
            return None

        beat_count = sum(1 for s in surprises if s == 'Beat')
        return f"{beat_count}/{len(surprises)} Beat"

    def _create_quarterly_timeline(self,
                                   symbol: str,
                                   raw_data: Dict[str, Any],
                                   start_year: int,
                                   end_year: int) -> List[QuarterData]:
        """
        Create the master quarterly timeline based on earnings publication dates.

        This is the key method that establishes the unified timeline.
        """
        earnings = raw_data.get('earnings', {})
        quarterly_earnings = earnings.get('quarterlyEarnings', [])

        if not quarterly_earnings:
            logger.warning(f"No quarterly earnings data for {symbol}")
            return []

        timeline = []

        for earning in quarterly_earnings:
            fiscal_date_str = earning.get('fiscalDateEnding')
            reported_date_str = earning.get('reportedDate')

            if not fiscal_date_str or not reported_date_str:
                continue

            try:
                fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d')
                reported_date = datetime.strptime(reported_date_str, '%Y-%m-%d')

                year, quarter = get_quarter_from_date(fiscal_date)

                # Filter by year range
                if year < start_year or year > end_year:
                    continue

                # Skip future reports
                if reported_date > datetime.now():
                    continue

                quarter_data = QuarterData(
                    report_date=reported_date,
                    fiscal_quarter_end=fiscal_date,
                    year=year,
                    quarter=quarter
                )

                timeline.append(quarter_data)

            except ValueError as e:
                logger.warning(f"Error parsing dates for {symbol}: {e}")
                continue

        # Sort by report date (chronological)
        timeline.sort(key=lambda x: x.report_date)

        # Remove duplicates (keep first occurrence)
        seen_dates = set()
        unique_timeline = []
        for qtr in timeline:
            date_key = qtr.report_date.date()
            if date_key not in seen_dates:
                seen_dates.add(date_key)
                unique_timeline.append(qtr)

        logger.info(f"Created timeline with {len(unique_timeline)} quarters for {symbol}")
        return unique_timeline

    def _calculate_metrics_for_quarter(self,
                                       symbol: str,
                                       quarter_data: QuarterData,
                                       raw_data: Dict[str, Any],
                                       is_recent: bool = False) -> Dict[str, Any]:
        """
        Calculate all metrics for a specific quarter.

        Args:
            symbol: Stock symbol
            quarter_data: Quarter timeline data
            raw_data: Raw API data
            is_recent: Whether this is a recent quarter (for Forward P/E)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        report_date = quarter_data.report_date
        fiscal_end = quarter_data.fiscal_quarter_end

        # Extract data from API responses
        overview = raw_data.get('overview', {})
        income_stmt = raw_data.get('income_statement', {})
        balance_sheet = raw_data.get('balance_sheet', {})
        cash_flow = raw_data.get('cash_flow', {})
        earnings = raw_data.get('earnings', {})

        quarterly_income = income_stmt.get('quarterlyReports', [])
        quarterly_balance = balance_sheet.get('quarterlyReports', [])
        quarterly_cash = cash_flow.get('quarterlyReports', [])
        quarterly_earnings = earnings.get('quarterlyEarnings', [])

        # Get price at quarter end for ratio calculations
        price = self._get_price_at_date(symbol, fiscal_end)
        metrics['price_at_quarter_end'] = price

        # =====================================================================
        # EPS Metrics
        # =====================================================================

        # EPS (TTM) - Sum of last 4 quarters reported EPS
        eps_ttm = None
        quarterly_eps_values = []

        for earning in quarterly_earnings:
            fiscal_date_str = earning.get('fiscalDateEnding')
            if not fiscal_date_str:
                continue

            try:
                fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d')
                if fiscal_date <= fiscal_end:
                    reported_eps = parse_float(earning.get('reportedEPS'))
                    if reported_eps is not None:
                        quarterly_eps_values.append({
                            'date': fiscal_date,
                            'eps': reported_eps,
                            'estimated': parse_float(earning.get('estimatedEPS')),
                            'surprise': parse_float(earning.get('surprise')),
                            'surprise_pct': parse_float(earning.get('surprisePercentage')),
                        })

                if len(quarterly_eps_values) >= 8:  # Get enough for YoY
                    break

            except ValueError:
                continue

        if len(quarterly_eps_values) >= 4:
            eps_ttm = sum(q['eps'] for q in quarterly_eps_values[:4])
        metrics['eps_ttm'] = eps_ttm

        # Current quarter EPS metrics
        if quarterly_eps_values:
            current_eps = quarterly_eps_values[0]
            metrics['eps_reported'] = current_eps['eps']
            metrics['eps_estimated'] = current_eps['estimated']
            metrics['eps_surprise'] = current_eps['surprise']
            metrics['eps_surprise_pct'] = current_eps['surprise_pct']
        else:
            metrics['eps_reported'] = None
            metrics['eps_estimated'] = None
            metrics['eps_surprise'] = None
            metrics['eps_surprise_pct'] = None

        # EPS Growth Rate (YoY)
        if len(quarterly_eps_values) >= 8:
            current_eps_ttm = sum(q['eps'] for q in quarterly_eps_values[:4])
            prior_eps_ttm = sum(q['eps'] for q in quarterly_eps_values[4:8])
            metrics['eps_growth_yoy_pct'] = self._calculate_yoy_growth(current_eps_ttm, prior_eps_ttm)
        else:
            metrics['eps_growth_yoy_pct'] = None

        # EPS Surprise Trend
        metrics['eps_surprise_trend'] = self._get_eps_surprise_trend(earnings, fiscal_end, 4)

        # =====================================================================
        # Revenue Metrics
        # =====================================================================

        # Revenue (TTM)
        revenue_ttm = self._calculate_ttm_sum(quarterly_income, 'totalRevenue', fiscal_end)
        metrics['revenue_ttm'] = revenue_ttm

        # Revenue Growth (YoY)
        prior_year_end = fiscal_end.replace(year=fiscal_end.year - 1)
        prior_revenue_ttm = self._calculate_ttm_sum(quarterly_income, 'totalRevenue', prior_year_end)
        metrics['revenue_growth_yoy_pct'] = self._calculate_yoy_growth(revenue_ttm, prior_revenue_ttm)

        # =====================================================================
        # Profitability Metrics
        # =====================================================================

        # Operating Income (TTM)
        operating_income_ttm = self._calculate_ttm_sum(quarterly_income, 'operatingIncome', fiscal_end)
        metrics['operating_income_ttm'] = operating_income_ttm

        # Operating Margin (TTM)
        if operating_income_ttm is not None and revenue_ttm and revenue_ttm != 0:
            metrics['operating_margin_ttm_pct'] = (operating_income_ttm / revenue_ttm) * 100
        else:
            metrics['operating_margin_ttm_pct'] = None

        # Gross Profit (TTM)
        gross_profit_ttm = self._calculate_ttm_sum(quarterly_income, 'grossProfit', fiscal_end)
        metrics['gross_profit_ttm'] = gross_profit_ttm

        # Gross Margin (TTM)
        if gross_profit_ttm is not None and revenue_ttm and revenue_ttm != 0:
            metrics['gross_margin_ttm_pct'] = (gross_profit_ttm / revenue_ttm) * 100
        else:
            metrics['gross_margin_ttm_pct'] = None

        # Net Income (TTM)
        net_income_ttm = self._calculate_ttm_sum(quarterly_income, 'netIncome', fiscal_end)
        metrics['net_income_ttm'] = net_income_ttm

        # Profit Margin (TTM)
        if net_income_ttm is not None and revenue_ttm and revenue_ttm != 0:
            metrics['profit_margin_ttm_pct'] = (net_income_ttm / revenue_ttm) * 100
        else:
            metrics['profit_margin_ttm_pct'] = None

        # EBITDA (TTM)
        metrics['ebitda_ttm'] = self._calculate_ttm_sum(quarterly_income, 'ebitda', fiscal_end)

        # =====================================================================
        # Valuation Ratios
        # =====================================================================

        # P/E Ratio (Trailing)
        if price and eps_ttm and eps_ttm != 0:
            metrics['pe_ratio_trailing'] = price / eps_ttm
        else:
            metrics['pe_ratio_trailing'] = None

        # P/E Ratio (Forward) - Only for recent quarters, use current from overview
        if is_recent:
            forward_pe = parse_float(overview.get('ForwardPE'))
            metrics['pe_ratio_forward'] = forward_pe
        else:
            metrics['pe_ratio_forward'] = None  # Not available for historical

        # PEG Ratio - Only for recent quarters
        if is_recent:
            peg = parse_float(overview.get('PEGRatio'))
            metrics['peg_ratio'] = peg
        else:
            # Try to calculate from trailing P/E and EPS growth
            if metrics.get('pe_ratio_trailing') and metrics.get('eps_growth_yoy_pct'):
                eps_growth = metrics['eps_growth_yoy_pct']
                if eps_growth > 0:
                    metrics['peg_ratio'] = metrics['pe_ratio_trailing'] / eps_growth
                else:
                    metrics['peg_ratio'] = None
            else:
                metrics['peg_ratio'] = None

        # Price/Book
        book_value_report = self._find_quarterly_report(quarterly_balance, fiscal_end)
        if book_value_report:
            total_equity = parse_float(book_value_report.get('totalShareholderEquity'))
            shares_outstanding = parse_float(overview.get('SharesOutstanding'))

            if total_equity and shares_outstanding and shares_outstanding > 0:
                book_value_per_share = total_equity / shares_outstanding
                metrics['book_value_per_share'] = book_value_per_share
                if price and book_value_per_share > 0:
                    metrics['price_to_book'] = price / book_value_per_share
                else:
                    metrics['price_to_book'] = None
            else:
                metrics['book_value_per_share'] = None
                metrics['price_to_book'] = None
        else:
            metrics['book_value_per_share'] = None
            metrics['price_to_book'] = None

        # Price/Sales (TTM)
        shares_outstanding = parse_float(overview.get('SharesOutstanding'))
        if revenue_ttm and shares_outstanding and shares_outstanding > 0:
            revenue_per_share = revenue_ttm / shares_outstanding
            if price and revenue_per_share > 0:
                metrics['price_to_sales_ttm'] = price / revenue_per_share
            else:
                metrics['price_to_sales_ttm'] = None
        else:
            metrics['price_to_sales_ttm'] = None

        # Price/Cash Flow
        ocf_ttm = self._calculate_ttm_sum(quarterly_cash, 'operatingCashflow', fiscal_end)

        if ocf_ttm and shares_outstanding and shares_outstanding > 0:
            ocf_per_share = ocf_ttm / shares_outstanding
            if price and ocf_per_share > 0:
                metrics['price_to_cash_flow'] = price / ocf_per_share
            else:
                metrics['price_to_cash_flow'] = None
        else:
            metrics['price_to_cash_flow'] = None

        # =====================================================================
        # Cash Flow Metrics
        # =====================================================================

        # Operating Cash Flow (TTM)
        metrics['operating_cash_flow_ttm'] = ocf_ttm

        # Capital Expenditures (TTM)
        capex_ttm = self._calculate_ttm_sum(quarterly_cash, 'capitalExpenditures', fiscal_end)
        metrics['capex_ttm'] = capex_ttm

        # Free Cash Flow (TTM) = OCF - CapEx
        if ocf_ttm is not None and capex_ttm is not None:
            # CapEx is usually reported as negative, so we add it
            if capex_ttm < 0:
                fcf = ocf_ttm + capex_ttm  # Adding negative = subtracting
            else:
                fcf = ocf_ttm - capex_ttm
            metrics['fcf_ttm'] = fcf
        else:
            metrics['fcf_ttm'] = None

        # FCF Trend (YoY)
        prior_ocf_ttm = self._calculate_ttm_sum(quarterly_cash, 'operatingCashflow', prior_year_end)
        prior_capex_ttm = self._calculate_ttm_sum(quarterly_cash, 'capitalExpenditures', prior_year_end)

        if prior_ocf_ttm is not None and prior_capex_ttm is not None:
            if prior_capex_ttm < 0:
                prior_fcf = prior_ocf_ttm + prior_capex_ttm
            else:
                prior_fcf = prior_ocf_ttm - prior_capex_ttm

            metrics['fcf_growth_yoy_pct'] = self._calculate_yoy_growth(metrics.get('fcf_ttm'), prior_fcf)
        else:
            metrics['fcf_growth_yoy_pct'] = None

        # FCF Yield
        market_cap = parse_float(overview.get('MarketCapitalization'))
        if metrics.get('fcf_ttm') and market_cap and market_cap > 0:
            metrics['fcf_yield_pct'] = (metrics['fcf_ttm'] / market_cap) * 100
        else:
            metrics['fcf_yield_pct'] = None

        # =====================================================================
        # Balance Sheet Metrics
        # =====================================================================

        balance_report = self._find_quarterly_report(quarterly_balance, fiscal_end)
        if balance_report:
            # Total Assets, Liabilities, Equity
            metrics['total_assets'] = parse_float(balance_report.get('totalAssets'))
            metrics['total_liabilities'] = parse_float(balance_report.get('totalLiabilities'))
            metrics['total_equity'] = parse_float(balance_report.get('totalShareholderEquity'))

            # Cash and Equivalents
            metrics['cash_and_equivalents'] = parse_float(balance_report.get('cashAndCashEquivalentsAtCarryingValue'))

            # Debt-to-Equity
            short_term_debt = parse_float(balance_report.get('shortTermDebt')) or 0
            long_term_debt = parse_float(balance_report.get('longTermDebt')) or 0
            current_long_term_debt = parse_float(balance_report.get('currentLongTermDebt')) or 0
            total_equity = parse_float(balance_report.get('totalShareholderEquity'))

            total_debt = short_term_debt + long_term_debt
            # Avoid double counting current portion
            if current_long_term_debt > 0 and short_term_debt == 0:
                total_debt = current_long_term_debt + long_term_debt

            if total_equity and total_equity != 0:
                metrics['debt_to_equity'] = total_debt / total_equity
            else:
                metrics['debt_to_equity'] = None

            # Current Ratio
            current_assets = parse_float(balance_report.get('totalCurrentAssets'))
            current_liabilities = parse_float(balance_report.get('totalCurrentLiabilities'))

            if current_assets and current_liabilities and current_liabilities != 0:
                metrics['current_ratio'] = current_assets / current_liabilities
            else:
                metrics['current_ratio'] = None

        else:
            metrics['total_assets'] = None
            metrics['total_liabilities'] = None
            metrics['total_equity'] = None
            metrics['cash_and_equivalents'] = None
            metrics['debt_to_equity'] = None
            metrics['current_ratio'] = None

        # =====================================================================
        # Additional Metrics from Overview (current values only for recent)
        # =====================================================================

        if is_recent:
            metrics['roe_ttm_pct'] = parse_float(overview.get('ReturnOnEquityTTM'))
            if metrics['roe_ttm_pct'] is not None:
                metrics['roe_ttm_pct'] = metrics['roe_ttm_pct'] * 100  # Convert to percentage

            metrics['roa_ttm_pct'] = parse_float(overview.get('ReturnOnAssetsTTM'))
            if metrics['roa_ttm_pct'] is not None:
                metrics['roa_ttm_pct'] = metrics['roa_ttm_pct'] * 100  # Convert to percentage

            metrics['dividend_yield_pct'] = parse_float(overview.get('DividendYield'))
            if metrics['dividend_yield_pct'] is not None:
                metrics['dividend_yield_pct'] = metrics['dividend_yield_pct'] * 100  # Convert to percentage

            metrics['beta'] = parse_float(overview.get('Beta'))
            metrics['analyst_target_price'] = parse_float(overview.get('AnalystTargetPrice'))
            metrics['market_cap'] = parse_float(overview.get('MarketCapitalization'))
            metrics['shares_outstanding'] = parse_float(overview.get('SharesOutstanding'))
        else:
            # These are current values, not historical - set to None for old quarters
            metrics['roe_ttm_pct'] = None
            metrics['roa_ttm_pct'] = None
            metrics['dividend_yield_pct'] = None
            metrics['beta'] = None
            metrics['analyst_target_price'] = None
            metrics['market_cap'] = None
            metrics['shares_outstanding'] = None

        # Interest Coverage (calculate from income statement)
        interest_expense_ttm = self._calculate_ttm_sum(quarterly_income, 'interestExpense', fiscal_end)
        if operating_income_ttm is not None and interest_expense_ttm and interest_expense_ttm != 0:
            metrics['interest_coverage'] = operating_income_ttm / abs(interest_expense_ttm)
        else:
            metrics['interest_coverage'] = None

        return metrics

    def _forward_fill_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill missing fundamental data.

        Rules:
        1. Only fill forward, never backward (prevents lookahead bias)
        2. Fill within same symbol only
        3. Preserve None for first occurrence (no data available)
        4. Fill across all metric columns except identifiers
        """
        if df.empty:
            return df

        df = df.copy()
        df = df.sort_values(['symbol', 'report_date'])

        # Identify metric columns (exclude identifiers)
        exclude_cols = ['report_date', 'symbol', 'fiscal_quarter_end', 'year', 'quarter']
        metric_cols = [col for col in df.columns if col not in exclude_cols]

        # Forward-fill within each symbol group
        df[metric_cols] = df.groupby('symbol')[metric_cols].fillna(method='ffill')

        return df

    def fetch_fundamental_data(self,
                               symbol: str,
                               start_year: int,
                               end_year: int) -> pd.DataFrame:
        """
        Fetch fundamental data for a symbol across all quarters.

        Creates a unified quarterly timeline with one row per earnings report date,
        all financial data unified, and missing values forward-filled.

        Args:
            symbol: Stock symbol
            start_year: First year to include
            end_year: Last year to include

        Returns:
            DataFrame with quarterly fundamental data (one row per quarter)
        """
        logger.info(f"Fetching fundamental data for {symbol} ({start_year}-{end_year})")

        try:
            raw_data = self._get_raw_data(symbol)
        except Exception as e:
            self.handler.log_issue(
                'api_error',
                f"Failed to fetch data for {symbol}: {e}",
                {'symbol': symbol},
                'error'
            )
            return pd.DataFrame()

        # Check if we got valid data
        if not raw_data.get('overview') or 'Symbol' not in raw_data.get('overview', {}):
            self.handler.log_issue(
                'no_data',
                f"No fundamental data available for {symbol}",
                {'symbol': symbol},
                'warning'
            )
            return pd.DataFrame()

        # Create the unified quarterly timeline
        timeline = self._create_quarterly_timeline(symbol, raw_data, start_year, end_year)

        if not timeline:
            logger.warning(f"No quarterly timeline created for {symbol}")
            return pd.DataFrame()

        # Determine recent quarters cutoff
        current_date = datetime.now()
        recent_cutoff = current_date - timedelta(days=RECENT_QUARTERS_FOR_FORWARD_PE * 91)

        # Calculate metrics for each quarter
        records = []
        for quarter_data in timeline:
            is_recent = quarter_data.report_date >= recent_cutoff

            try:
                metrics = self._calculate_metrics_for_quarter(
                    symbol, quarter_data, raw_data, is_recent
                )

                record = {
                    'report_date': quarter_data.report_date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'fiscal_quarter_end': quarter_data.fiscal_quarter_end.strftime('%Y-%m-%d'),
                    'year': quarter_data.year,
                    'quarter': quarter_data.quarter,
                    **metrics
                }
                records.append(record)

            except Exception as e:
                self.handler.log_issue(
                    'calculation_error',
                    f"Error calculating metrics for {symbol} {quarter_data.year}Q{quarter_data.quarter}: {e}",
                    {'symbol': symbol, 'year': quarter_data.year, 'quarter': quarter_data.quarter},
                    'warning'
                )
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Forward-fill missing values
        df = self._forward_fill_metrics(df)

        # Log summary
        total_quarters = len(df)
        logger.info(f"Generated {total_quarters} quarters of unified data for {symbol}")

        return df

    def fetch_for_multiple_symbols(self,
                                   symbols: List[str],
                                   start_year: int,
                                   end_year: int,
                                   output_dir: Path) -> Dict[str, Path]:
        """
        Fetch fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_year: First year to include
            end_year: Last year to include
            output_dir: Directory to save output files

        Returns:
            Dictionary mapping symbol to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        total = len(symbols)

        print(f"\nFetching fundamental data for {total} symbols...")
        print(f"Date range: {start_year} - {end_year}")
        print(f"Output directory: {output_dir}")
        print(f"Mode: Unified quarterly data only\n")

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{total}] Processing {symbol}...", end=" ", flush=True)

            try:
                df = self.fetch_fundamental_data(symbol, start_year, end_year)

                if df.empty:
                    print("No data")
                    continue

                output_file = output_dir / f"{symbol}_fundamental_quarterly.csv"
                df.to_csv(output_file, index=False)
                results[symbol] = output_file
                print(f"OK ({len(df)} quarters)")

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break

            except Exception as e:
                print(f"ERROR: {e}")
                self.handler.log_issue(
                    'processing_error',
                    f"Failed to process {symbol}: {e}",
                    {'symbol': symbol},
                    'error'
                )

        print(f"\nCompleted: {len(results)}/{total} symbols")
        return results

    def clear_cache(self):
        """Clear all cached data."""
        self._raw_data_cache.clear()
        self._price_cache.clear()
