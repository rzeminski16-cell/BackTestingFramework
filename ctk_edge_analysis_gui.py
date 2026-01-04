"""
Edge Analysis GUI (CustomTkinter)

A specialized GUI for analyzing entry edge and R-multiples from existing trade logs.

Key Features:
- Load multiple trade log CSV files at once
- Configurable price data path for loading historical data (TICKER_daily.csv format)
- Three visualization tabs:
  1. E-ratio: Aggregate entry edge across all trades per horizon
  2. R-multiple: Distribution of all trade outcomes
  3. Trade Details: Detailed view of selected trade

E-ratio Formula (aggregate across all trades):
    For each horizon n (days after entry):
    1. Calculate MFE_n and MAE_n for each trade
    2. Normalize by ATR at entry: NormMFE = MFE/ATR, NormMAE = MAE/ATR
    3. Average across all trades: AvgNormMFE_n, AvgNormMAE_n
    4. E-ratio(n) = AvgNormMFE_n / AvgNormMAE_n

    Where:
    - MFE_n = Maximum Favorable Excursion over n days from entry
    - MAE_n = Maximum Adverse Excursion over n days from entry
    - ATR = Average True Range at entry (volatility normalization)

    E-ratio > 1 indicates positive edge at that horizon

R-multiple Formula:
    R-multiple = (Exit Price - Entry Price) / (Entry Price - Stop Loss)
    Expresses profit/loss as a multiple of initial risk (R)
"""

import customtkinter as ctk
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info


# =============================================================================
# VALIDATION FRAMEWORK
# =============================================================================

class ValidationResult:
    """Represents a single validation check result."""

    def __init__(self, check_name: str, passed: bool, message: str = "",
                 severity: str = "info", value: Any = None, expected: Any = None):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.severity = severity  # 'info', 'warning', 'error'
        self.value = value
        self.expected = expected

    def __repr__(self):
        status = "✓" if self.passed else "✗"
        return f"[{status}] {self.check_name}: {self.message}"


class ValidationReport:
    """Aggregates validation results across all checks."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.data_checks: List[ValidationResult] = []
        self.trade_checks: List[ValidationResult] = []
        self.aggregation_checks: List[ValidationResult] = []
        self.bias_checks: List[ValidationResult] = []
        self.outliers: List[Dict[str, Any]] = []

    def add_result(self, result: ValidationResult, category: str = "data"):
        self.results.append(result)
        if category == "data":
            self.data_checks.append(result)
        elif category == "trade":
            self.trade_checks.append(result)
        elif category == "aggregation":
            self.aggregation_checks.append(result)
        elif category == "bias":
            self.bias_checks.append(result)

    def add_outlier(self, trade_id: str, symbol: str, field: str, value: float, threshold: float):
        self.outliers.append({
            'trade_id': trade_id,
            'symbol': symbol,
            'field': field,
            'value': value,
            'threshold': threshold
        })

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "warning")

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "error")

    def get_summary(self) -> str:
        lines = [
            f"Validation Summary: {self.passed_count} passed, {self.failed_count} failed",
            f"  - Errors: {self.error_count}, Warnings: {self.warning_count}",
            ""
        ]

        for category_name, checks in [
            ("Data Input & Preprocessing", self.data_checks),
            ("Per-Trade MFE/MAE", self.trade_checks),
            ("Aggregation & E-ratio", self.aggregation_checks),
            ("Bias & Statistical", self.bias_checks)
        ]:
            if checks:
                failed = [r for r in checks if not r.passed]
                lines.append(f"{category_name}: {len(checks) - len(failed)}/{len(checks)} passed")
                for r in failed[:5]:  # Show first 5 failures
                    lines.append(f"  [{r.severity.upper()}] {r.check_name}: {r.message}")

        if self.outliers:
            lines.append(f"\nOutliers logged: {len(self.outliers)}")

        return "\n".join(lines)


class ERatioValidator:
    """Validates E-ratio calculation data and results."""

    def __init__(self):
        self.report = ValidationReport()

    def reset(self):
        self.report = ValidationReport()

    # -------------------------------------------------------------------------
    # 1. Data Input & Preprocessing Checks
    # -------------------------------------------------------------------------

    def validate_price_data_integrity(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check for NaNs/gaps and sorted dates."""
        checks_passed = True

        # Check for NaN in close prices
        nan_count = df['close'].isna().sum()
        if nan_count > 0:
            self.report.add_result(ValidationResult(
                f"price_data_nan_{symbol}",
                False,
                f"{symbol}: {nan_count} NaN values in close prices",
                severity="error",
                value=nan_count,
                expected=0
            ), "data")
            checks_passed = False
        else:
            self.report.add_result(ValidationResult(
                f"price_data_nan_{symbol}",
                True,
                f"{symbol}: No NaN values in close prices"
            ), "data")

        # Check date sorting (monotonic increasing)
        dates = pd.to_datetime(df['date'])
        is_sorted = dates.is_monotonic_increasing
        if not is_sorted:
            self.report.add_result(ValidationResult(
                f"price_data_sorted_{symbol}",
                False,
                f"{symbol}: Dates are not sorted in ascending order",
                severity="error"
            ), "data")
            checks_passed = False
        else:
            self.report.add_result(ValidationResult(
                f"price_data_sorted_{symbol}",
                True,
                f"{symbol}: Dates properly sorted"
            ), "data")

        return checks_passed

    def validate_atr(self, df: pd.DataFrame, symbol: str, atr_col: str) -> bool:
        """Validate ATR is positive everywhere."""
        if atr_col not in df.columns:
            self.report.add_result(ValidationResult(
                f"atr_exists_{symbol}",
                False,
                f"{symbol}: ATR column '{atr_col}' not found",
                severity="error"
            ), "data")
            return False

        atr_values = df[atr_col].dropna()
        non_positive = (atr_values <= 0).sum()

        if non_positive > 0:
            self.report.add_result(ValidationResult(
                f"atr_positive_{symbol}",
                False,
                f"{symbol}: {non_positive} non-positive ATR values",
                severity="warning",
                value=non_positive
            ), "data")
            return False

        self.report.add_result(ValidationResult(
            f"atr_positive_{symbol}",
            True,
            f"{symbol}: All {len(atr_values)} ATR values positive"
        ), "data")
        return True

    def validate_date_alignment(self, trade_entry_date: datetime, price_date: datetime,
                                 trade_id: str) -> Tuple[bool, int]:
        """Check if entry date aligns with price data (within 1 day)."""
        diff_days = abs((price_date - trade_entry_date).days)

        passed = diff_days <= 1
        if not passed:
            self.report.add_result(ValidationResult(
                f"date_alignment_{trade_id}",
                False,
                f"Trade {trade_id}: Entry date differs by {diff_days} days",
                severity="warning",
                value=diff_days,
                expected="≤1"
            ), "data")
        else:
            self.report.add_result(ValidationResult(
                f"date_alignment_{trade_id}",
                True,
                f"Trade {trade_id}: Date aligned within {diff_days} days"
            ), "data")

        return passed, diff_days

    def validate_entry_price_consistency(self, trade_entry_price: float, data_close: float,
                                          trade_id: str, symbol: str) -> Tuple[bool, float]:
        """Check if trade entry price matches price data close (<5% mismatch)."""
        if data_close == 0:
            self.report.add_result(ValidationResult(
                f"entry_price_{trade_id}",
                False,
                f"Trade {trade_id}: Zero close price in data",
                severity="error"
            ), "data")
            return False, 0.0

        price_diff_pct = abs(trade_entry_price - data_close) / data_close * 100

        passed = price_diff_pct < 5.0
        if not passed:
            self.report.add_result(ValidationResult(
                f"entry_price_{trade_id}",
                False,
                f"Trade {trade_id} ({symbol}): Entry price differs by {price_diff_pct:.2f}%",
                severity="warning",
                value=price_diff_pct,
                expected="<5%"
            ), "data")
            self.report.add_outlier(trade_id, symbol, "entry_price_diff_pct", price_diff_pct, 5.0)
        else:
            self.report.add_result(ValidationResult(
                f"entry_price_{trade_id}",
                True,
                f"Trade {trade_id}: Entry price within {price_diff_pct:.2f}%"
            ), "data")

        return passed, price_diff_pct

    # -------------------------------------------------------------------------
    # 2. Per-Trade MFE/MAE Loop Checks
    # -------------------------------------------------------------------------

    def validate_forward_slice(self, entry_pos: int, n: int, df_len: int,
                                actual_len: int, trade_id: str) -> bool:
        """Validate forward slice bounds."""
        expected_len = min(n, df_len - entry_pos - 1)

        passed = actual_len == expected_len or actual_len < n
        if not passed:
            self.report.add_result(ValidationResult(
                f"slice_bounds_{trade_id}_n{n}",
                False,
                f"Trade {trade_id}, n={n}: Expected {expected_len} bars, got {actual_len}",
                severity="error",
                value=actual_len,
                expected=expected_len
            ), "trade")

        return passed

    def validate_excursions(self, mfe: float, mae: float, trade_id: str, n: int,
                             max_possible_mfe: float = None) -> bool:
        """Validate MFE/MAE are non-negative and within bounds."""
        checks_passed = True

        if mfe < 0:
            self.report.add_result(ValidationResult(
                f"mfe_positive_{trade_id}_n{n}",
                False,
                f"Trade {trade_id}, n={n}: MFE is negative ({mfe:.4f})",
                severity="error",
                value=mfe,
                expected="≥0"
            ), "trade")
            checks_passed = False

        if mae < 0:
            self.report.add_result(ValidationResult(
                f"mae_positive_{trade_id}_n{n}",
                False,
                f"Trade {trade_id}, n={n}: MAE is negative ({mae:.4f})",
                severity="error",
                value=mae,
                expected="≥0"
            ), "trade")
            checks_passed = False

        if max_possible_mfe is not None and mfe > max_possible_mfe * 1.001:  # Allow 0.1% tolerance
            self.report.add_result(ValidationResult(
                f"mfe_bounds_{trade_id}_n{n}",
                False,
                f"Trade {trade_id}, n={n}: MFE ({mfe:.4f}) exceeds max possible ({max_possible_mfe:.4f})",
                severity="error",
                value=mfe,
                expected=f"≤{max_possible_mfe:.4f}"
            ), "trade")
            checks_passed = False

        return checks_passed

    def validate_normalization(self, norm_mfe: float, norm_mae: float,
                                trade_id: str, n: int) -> bool:
        """Validate normalized values are not inf/nan."""
        checks_passed = True

        if np.isinf(norm_mfe) or np.isnan(norm_mfe):
            self.report.add_result(ValidationResult(
                f"norm_mfe_valid_{trade_id}_n{n}",
                False,
                f"Trade {trade_id}, n={n}: Invalid normalized MFE ({norm_mfe})",
                severity="error"
            ), "trade")
            checks_passed = False

        if np.isinf(norm_mae) or np.isnan(norm_mae):
            self.report.add_result(ValidationResult(
                f"norm_mae_valid_{trade_id}_n{n}",
                False,
                f"Trade {trade_id}, n={n}: Invalid normalized MAE ({norm_mae})",
                severity="error"
            ), "trade")
            checks_passed = False

        return checks_passed

    # -------------------------------------------------------------------------
    # 3. Aggregation & E-ratio Checks
    # -------------------------------------------------------------------------

    def validate_trades_per_horizon(self, trades_per_horizon: Dict[int, int],
                                     min_required: int = 20, check_up_to_day: int = 60) -> bool:
        """Validate minimum trades per horizon."""
        checks_passed = True

        for n in range(1, min(check_up_to_day + 1, max(trades_per_horizon.keys()) + 1)):
            count = trades_per_horizon.get(n, 0)
            if count < min_required:
                self.report.add_result(ValidationResult(
                    f"trades_per_horizon_n{n}",
                    False,
                    f"Day {n}: Only {count} trades (min {min_required})",
                    severity="warning",
                    value=count,
                    expected=f"≥{min_required}"
                ), "aggregation")
                checks_passed = False

        # Add summary check
        min_count = min(trades_per_horizon.get(n, 0) for n in range(1, min(31, max(trades_per_horizon.keys()) + 1)))
        self.report.add_result(ValidationResult(
            "trades_per_horizon_summary",
            min_count >= min_required,
            f"Min trades in first 30 days: {min_count}",
            severity="warning" if min_count < min_required else "info",
            value=min_count,
            expected=f"≥{min_required}"
        ), "aggregation")

        return checks_passed

    def validate_eratio_bounds(self, e_ratios: Dict[int, float],
                                min_bound: float = 0.1, max_bound: float = 5.0) -> bool:
        """Validate E-ratio is within reasonable bounds."""
        checks_passed = True
        violations = []

        for n, e_ratio in e_ratios.items():
            if np.isinf(e_ratio) or np.isnan(e_ratio):
                violations.append((n, e_ratio, "inf/nan"))
                checks_passed = False
            elif e_ratio < min_bound:
                violations.append((n, e_ratio, f"<{min_bound}"))
            elif e_ratio > max_bound:
                violations.append((n, e_ratio, f">{max_bound}"))

        if violations:
            self.report.add_result(ValidationResult(
                "eratio_bounds",
                False,
                f"{len(violations)} E-ratio values outside bounds [{min_bound}, {max_bound}]",
                severity="warning",
                value=len(violations)
            ), "aggregation")

            # Log first few violations
            for n, e_ratio, reason in violations[:5]:
                self.report.add_result(ValidationResult(
                    f"eratio_bounds_n{n}",
                    False,
                    f"Day {n}: E-ratio={e_ratio:.4f} ({reason})",
                    severity="info"
                ), "aggregation")
        else:
            self.report.add_result(ValidationResult(
                "eratio_bounds",
                True,
                f"All E-ratios within bounds [{min_bound}, {max_bound}]"
            ), "aggregation")

        return checks_passed

    # -------------------------------------------------------------------------
    # 4. Bias & Statistical Checks
    # -------------------------------------------------------------------------

    def validate_sample_size_consistency(self, trades_per_horizon: Dict[int, int]) -> bool:
        """
        Validate trade count is CONSISTENT across all horizons.

        With the survivorship bias fix, we pre-filter trades to ensure
        all included trades have data for all horizons. This check verifies
        that the sample size remains constant.
        """
        if not trades_per_horizon:
            return True

        horizons = sorted(trades_per_horizon.keys())
        counts = [trades_per_horizon[h] for h in horizons]

        min_count = min(counts)
        max_count = max(counts)
        is_consistent = min_count == max_count

        if is_consistent:
            self.report.add_result(ValidationResult(
                "sample_consistency",
                True,
                f"Sample size consistent: {min_count} trades across all {len(horizons)} horizons",
                severity="info",
                value=min_count
            ), "bias")
        else:
            # If not consistent, check how much variation
            variation_pct = (max_count - min_count) / max_count * 100 if max_count > 0 else 0
            self.report.add_result(ValidationResult(
                "sample_consistency",
                False,
                f"Sample size varies: {min_count}-{max_count} trades ({variation_pct:.1f}% variation)",
                severity="warning",
                value=variation_pct
            ), "bias")

        return is_consistent

    def validate_survivorship(self, trades_per_horizon: Dict[int, int],
                               total_trades: int, day: int = 200) -> bool:
        """Check survivorship rate at given day."""
        count_at_day = trades_per_horizon.get(day, 0)
        surv_rate = count_at_day / total_trades if total_trades > 0 else 0

        passed = surv_rate < 0.1  # Less than 10% should remain at day 200

        self.report.add_result(ValidationResult(
            f"survivorship_{day}",
            passed,
            f"Survivorship at day {day}: {surv_rate*100:.1f}% ({count_at_day}/{total_trades})",
            severity="warning" if not passed else "info",
            value=surv_rate,
            expected="<10%"
        ), "bias")

        return passed

    def calculate_winsorized_eratio(self, all_mfe_norm: Dict[int, List[float]],
                                     all_mae_norm: Dict[int, List[float]],
                                     percentile: float = 99) -> Dict[int, float]:
        """Calculate E-ratio with winsorized (capped) outliers."""
        winsorized_eratios = {}

        for n in all_mfe_norm:
            if not all_mfe_norm[n] or not all_mae_norm[n]:
                continue

            mfe_vals = np.array(all_mfe_norm[n])
            mae_vals = np.array(all_mae_norm[n])

            # Cap at 99th percentile
            mfe_cap = np.percentile(mfe_vals, percentile)
            mae_cap = np.percentile(mae_vals, percentile)

            mfe_capped = np.clip(mfe_vals, None, mfe_cap)
            mae_capped = np.clip(mae_vals, None, mae_cap)

            avg_mfe = np.mean(mfe_capped)
            avg_mae = np.mean(mae_capped)

            if avg_mae > 0:
                winsorized_eratios[n] = avg_mfe / avg_mae
            else:
                winsorized_eratios[n] = float('inf') if avg_mfe > 0 else 1.0

        return winsorized_eratios

    def compare_with_random_baseline(self, e_ratios: Dict[int, float],
                                      check_days: int = 20) -> Tuple[bool, float]:
        """Compare E-ratios against random baseline (should be near 1.0)."""
        early_eratios = [e_ratios.get(n, 1.0) for n in range(1, check_days + 1)]
        avg_early = np.mean(early_eratios) if early_eratios else 1.0

        # For random entries, E-ratio should be ~1.0
        # A good strategy should have E-ratio > 1.0 in early days
        has_edge = avg_early > 1.0

        self.report.add_result(ValidationResult(
            "random_baseline",
            has_edge,
            f"Avg E-ratio (days 1-{check_days}): {avg_early:.3f} vs random baseline ~1.0",
            severity="info",
            value=avg_early,
            expected=">1.0 for edge"
        ), "bias")

        return has_edge, avg_early


# =============================================================================
# TRADE LOG LOADER
# =============================================================================

class TradeLogEntry:
    """Represents a single trade from a trade log."""

    def __init__(self, row: pd.Series):
        self.trade_id = row.get('trade_id', '')
        self.symbol = row.get('symbol', '')
        self.entry_date = self._parse_date(row.get('entry_date'))
        self.entry_price = float(row.get('entry_price', 0))
        self.exit_date = self._parse_date(row.get('exit_date'))
        self.exit_price = float(row.get('exit_price', 0))
        self.quantity = float(row.get('quantity', 0))
        self.side = str(row.get('side', 'LONG')).upper()  # 'LONG' or 'SHORT'
        self.initial_stop_loss = self._parse_float(row.get('initial_stop_loss'))
        self.final_stop_loss = self._parse_float(row.get('final_stop_loss'))
        self.pl = float(row.get('pl', 0))
        self.pl_pct = float(row.get('pl_pct', 0))
        self.entry_reason = row.get('entry_reason', '')
        self.exit_reason = row.get('exit_reason', '')

        # Calculate R-multiple if we have stop loss
        self.r_multiple = self._calculate_r_multiple()

    @property
    def is_long(self) -> bool:
        """Returns True if this is a long trade."""
        return self.side == 'LONG'

    def _parse_date(self, val) -> Optional[datetime]:
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, datetime):
            return val
        try:
            return pd.to_datetime(val)
        except:
            return None

    def _parse_float(self, val) -> Optional[float]:
        if pd.isna(val) or val is None:
            return None
        try:
            return float(val)
        except:
            return None

    def _calculate_r_multiple(self) -> Optional[float]:
        """Calculate R-multiple based on entry, exit, and stop loss."""
        if self.initial_stop_loss is None or self.initial_stop_loss <= 0:
            return None

        # Initial risk (R) = entry price - stop loss
        initial_risk = self.entry_price - self.initial_stop_loss
        if initial_risk <= 0:
            return None

        # P/L per unit
        pl_per_unit = self.exit_price - self.entry_price

        # R-multiple = P/L per unit / R
        return pl_per_unit / initial_risk

    @property
    def display_name(self) -> str:
        """Display name for trade selection list."""
        date_str = self.entry_date.strftime('%Y-%m-%d') if self.entry_date else 'N/A'
        r_str = f"{self.r_multiple:.2f}R" if self.r_multiple is not None else "N/A"
        return f"{self.trade_id} | {self.symbol} | {date_str} | {r_str}"

    def __str__(self):
        return self.display_name


class TradeLogLoader:
    """Loads and parses trade log CSV files."""

    def __init__(self):
        self.trades: List[TradeLogEntry] = []
        self.source_files: List[Path] = []

    def load(self, filepath: Path) -> List[TradeLogEntry]:
        """Load trades from a single CSV file (appends to existing trades)."""
        df = pd.read_csv(filepath)

        new_trades = []
        for _, row in df.iterrows():
            try:
                trade = TradeLogEntry(row)
                if trade.symbol and trade.entry_date:
                    new_trades.append(trade)
            except Exception as e:
                print(f"Error parsing trade row: {e}")
                continue

        self.trades.extend(new_trades)
        self.source_files.append(filepath)
        return new_trades

    def load_multiple(self, filepaths: List[Path]) -> List[TradeLogEntry]:
        """Load trades from multiple CSV files."""
        self.clear()
        all_trades = []
        for filepath in filepaths:
            trades = self.load(filepath)
            all_trades.extend(trades)
        return all_trades

    def clear(self):
        """Clear all loaded trades."""
        self.trades = []
        self.source_files = []

    def get_symbols(self) -> List[str]:
        """Get unique symbols in loaded trades."""
        return list(set(t.symbol for t in self.trades))


# =============================================================================
# E-RATIO CALCULATOR
# =============================================================================

class ERatioCalculator:
    """Calculates aggregate E-ratio across all trades using historical price data."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self.validator = ERatioValidator()
        self._validated_symbols: set = set()

    def set_data_path(self, data_path: Path):
        """Update data path and clear cache."""
        self.data_path = data_path
        self._price_cache.clear()
        self._validated_symbols.clear()

    def debug_single_trade(self, trade: TradeLogEntry, max_days: int = 30) -> None:
        """Print debug info for a single trade's MFE/MAE calculation."""
        print(f"\n{'='*60}")
        print(f"DEBUG: Trade {trade.trade_id} ({trade.symbol})")
        print(f"  Entry: {trade.entry_date}, Price: {trade.entry_price}")
        print(f"  Side: {trade.side} (is_long={trade.is_long})")
        print(f"{'='*60}")

        price_data = self._get_price_data(trade.symbol, validate=False)
        if price_data is None:
            print("  ERROR: Could not load price data")
            return

        price_data = price_data.copy()
        price_data['date'] = pd.to_datetime(price_data['date'])

        # Find entry
        entry_date_only = pd.to_datetime(trade.entry_date).normalize()
        exact_match = price_data['date'] == entry_date_only
        if exact_match.any():
            entry_idx = price_data[exact_match].index[0]
        else:
            mask = price_data['date'] >= trade.entry_date
            if not mask.any():
                print("  ERROR: No price data after entry")
                return
            entry_idx = price_data[mask].index[0]

        entry_bar = price_data.loc[entry_idx]
        entry_price = entry_bar['close']
        entry_pos = price_data.index.get_loc(entry_idx)
        df_len = len(price_data)

        # Get ATR
        atr_col = None
        for col in ['atr_14_atr', 'atr_14', 'atr']:
            if col in price_data.columns:
                atr_col = col
                break
        atr_at_entry = entry_bar[atr_col] if atr_col else 1.0

        print(f"  Entry Position: {entry_pos}, Data Length: {df_len}")
        print(f"  Entry Price (from data): {entry_price:.4f}")
        print(f"  ATR at Entry: {atr_at_entry:.4f}")
        print(f"  Max Horizon Available: {df_len - entry_pos - 1}")
        print()
        print(f"  {'n':>3} | {'MFE':>10} | {'MAE':>10} | {'NormMFE':>10} | {'NormMAE':>10} | {'E-ratio':>10} | {'#bars':>5}")
        print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}")

        prev_mfe, prev_mae = 0, 0
        for n in range(1, min(max_days + 1, df_len - entry_pos)):
            forward_data = price_data.iloc[entry_pos + 1:entry_pos + n + 1]
            closes = forward_data['close'].values
            max_close = closes.max()
            min_close = closes.min()

            if trade.is_long:
                mfe_n = max(0, max_close - entry_price)
                mae_n = max(0, entry_price - min_close)
            else:
                mfe_n = max(0, entry_price - min_close)
                mae_n = max(0, max_close - entry_price)

            norm_mfe = mfe_n / atr_at_entry
            norm_mae = mae_n / atr_at_entry
            e_ratio = norm_mfe / norm_mae if norm_mae > 0 else float('inf')

            # Check monotonicity
            mfe_flag = "!" if mfe_n < prev_mfe else " "
            mae_flag = "!" if mae_n < prev_mae else " "

            print(f"  {n:3d} | {mfe_n:10.4f}{mfe_flag}| {mae_n:10.4f}{mae_flag}| {norm_mfe:10.4f} | {norm_mae:10.4f} | {e_ratio:10.4f} | {len(closes):5d}")

            prev_mfe, prev_mae = mfe_n, mae_n

    def debug_aggregate_diagnostics(self, trades: List[TradeLogEntry], max_days: int = 30,
                                      use_consistent_sample: bool = True) -> Dict[str, Any]:
        """
        Compute aggregate E-ratio with detailed diagnostics.

        Args:
            trades: List of trades to analyze
            max_days: Maximum horizon to analyze
            use_consistent_sample: If True (default), only include trades with >= max_days
                                   forward data to avoid survivorship bias
        """
        print(f"\n{'='*80}")
        print("AGGREGATE E-RATIO DIAGNOSTICS")
        print(f"{'='*80}")
        print(f"Total trades: {len(trades)}")
        print(f"Consistent sample mode: {use_consistent_sample}")

        # Determine max horizon for each trade
        trade_max_horizons = {}
        for trade in trades:
            max_horizon = self._get_trade_max_horizon(trade)
            trade_max_horizons[trade] = max_horizon

        # Filter trades if using consistent sample
        if use_consistent_sample:
            eligible_trades = [t for t in trades if trade_max_horizons[t] >= max_days]
            filtered_count = len(trades) - len(eligible_trades)
            print(f"Filtered {filtered_count} trades with <{max_days} days forward data")
            print(f"Eligible trades: {len(eligible_trades)}")
        else:
            eligible_trades = trades
            print("WARNING: Not using consistent sample - survivorship bias may occur!")

        # Print horizon distribution
        print(f"\nTrade Horizon Distribution (all trades):")
        horizon_counts = {}
        for h in trade_max_horizons.values():
            horizon_counts[h] = horizon_counts.get(h, 0) + 1
        for h in sorted(horizon_counts.keys())[-10:]:  # Show last 10
            print(f"  Max horizon {h}: {horizon_counts[h]} trades")

        # Collect data from eligible trades
        all_mfe_norm: Dict[int, List[float]] = {n: [] for n in range(1, max_days + 1)}
        all_mae_norm: Dict[int, List[float]] = {n: [] for n in range(1, max_days + 1)}

        for trade in eligible_trades:
            excursions = self.calculate_trade_excursions(trade, max_days, validate=False)
            if excursions['error']:
                continue
            if excursions['mfe_norm']:
                for n in excursions['mfe_norm']:
                    all_mfe_norm[n].append(excursions['mfe_norm'][n])
                    all_mae_norm[n].append(excursions['mae_norm'][n])

        # Print aggregate stats per horizon
        print(f"\n{'n':>3} | {'#trades':>7} | {'AvgMFE':>10} | {'AvgMAE':>10} | {'E-ratio':>10} | {'MFE_med':>10} | {'MAE_med':>10}")
        print(f"{'-'*3}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        results = {'e_ratios': {}, 'trades_per_horizon': {}}

        for n in range(1, max_days + 1):
            if not all_mfe_norm[n]:
                break
            mfe_arr = np.array(all_mfe_norm[n])
            mae_arr = np.array(all_mae_norm[n])

            avg_mfe = mfe_arr.mean()
            avg_mae = mae_arr.mean()
            med_mfe = np.median(mfe_arr)
            med_mae = np.median(mae_arr)
            e_ratio = avg_mfe / avg_mae if avg_mae > 0 else float('inf')

            print(f"{n:3d} | {len(mfe_arr):7d} | {avg_mfe:10.4f} | {avg_mae:10.4f} | {e_ratio:10.4f} | {med_mfe:10.4f} | {med_mae:10.4f}")

            results['e_ratios'][n] = e_ratio
            results['trades_per_horizon'][n] = len(mfe_arr)

        # Sample consistency check
        if results['trades_per_horizon']:
            counts = list(results['trades_per_horizon'].values())
            print(f"\nSAMPLE CONSISTENCY CHECK:")
            print(f"  Min trades across horizons: {min(counts)}")
            print(f"  Max trades across horizons: {max(counts)}")
            if min(counts) == max(counts):
                print(f"  Status: CONSISTENT - sample size is constant across all horizons")
            else:
                print(f"  Status: INCONSISTENT - sample varies (potential survivorship bias)")

        return results

    def _get_price_data(self, symbol: str, validate: bool = True) -> Optional[pd.DataFrame]:
        """Get price data for a symbol, using cache if available."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        # Try loading with _daily suffix (TICKER_daily.csv format)
        file_path = self.data_path / f"{symbol}_daily.csv"

        if not file_path.exists():
            # Fallback to plain symbol.csv
            file_path = self.data_path / f"{symbol}.csv"

        if not file_path.exists():
            print(f"Price data file not found: {symbol}_daily.csv or {symbol}.csv")
            return None

        try:
            data = pd.read_csv(file_path)
            self._price_cache[symbol] = data

            # Validate price data integrity on first load
            if validate and symbol not in self._validated_symbols:
                self._validated_symbols.add(symbol)
                self.validator.validate_price_data_integrity(data, symbol)

                # Validate ATR
                atr_col = None
                for col in ['atr_14_atr', 'atr_14', 'atr']:
                    if col in data.columns:
                        atr_col = col
                        break
                if atr_col:
                    self.validator.validate_atr(data, symbol, atr_col)

            return data
        except Exception as e:
            print(f"Error loading price data for {symbol}: {e}")
            return None

    def calculate_trade_excursions(self, trade: TradeLogEntry, max_days: int = 30,
                                     validate: bool = True) -> Dict[str, Any]:
        """
        Calculate MFE and MAE for each horizon for a single trade.

        Returns:
            Dict containing:
            - mfe_norm: Dict[int, float] mapping day -> normalized MFE (MFE/ATR)
            - mae_norm: Dict[int, float] mapping day -> normalized MAE (MAE/ATR)
            - mfe_raw: Dict[int, float] mapping day -> raw MFE values
            - mae_raw: Dict[int, float] mapping day -> raw MAE values
            - atr_at_entry: ATR value at entry
            - entry_price: Entry price used (from price data)
            - error: Optional error message
        """
        result = {
            'mfe_norm': {},
            'mae_norm': {},
            'mfe_raw': {},
            'mae_raw': {},
            'atr_at_entry': None,
            'entry_price': None,
            'error': None
        }

        trade_id = str(trade.trade_id)

        # Get price data
        price_data = self._get_price_data(trade.symbol, validate=validate)
        if price_data is None:
            result['error'] = f"Could not load price data for {trade.symbol}"
            return result

        # Find entry date in data
        entry_date = trade.entry_date
        if entry_date is None:
            result['error'] = "Trade has no entry date"
            return result

        # Convert entry_date to same type as data
        if 'date' not in price_data.columns:
            result['error'] = "Price data missing 'date' column"
            return result

        # Find the entry bar index
        price_data = price_data.copy()
        price_data['date'] = pd.to_datetime(price_data['date'])

        # Find exact date match first, then closest date on or after entry
        entry_date_only = pd.to_datetime(entry_date).normalize()  # Remove time component
        exact_match = price_data['date'] == entry_date_only

        if exact_match.any():
            entry_idx = price_data[exact_match].index[0]
        else:
            # Fallback to first date >= entry_date
            mask = price_data['date'] >= entry_date
            if not mask.any():
                result['error'] = f"No price data on or after entry date {entry_date}"
                return result
            entry_idx = price_data[mask].index[0]

        entry_bar = price_data.loc[entry_idx]
        price_date = pd.to_datetime(entry_bar['date'])

        # VALIDATION: Date alignment check
        if validate:
            self.validator.validate_date_alignment(entry_date, price_date, trade_id)

        # Use close from price data as entry price (ensures consistency with future closes)
        entry_price = entry_bar['close']
        result['entry_price'] = float(entry_price)

        # VALIDATION: Entry price consistency check
        if validate:
            self.validator.validate_entry_price_consistency(
                trade.entry_price, entry_price, trade_id, trade.symbol
            )

        # Get ATR at entry (ATR is already calculated from adjusted close prices)
        atr_col = None
        for col in ['atr_14_atr', 'atr_14', 'atr']:
            if col in price_data.columns:
                atr_col = col
                break

        if atr_col is None:
            result['error'] = "No ATR column found in price data"
            return result

        atr_at_entry = entry_bar[atr_col]
        if pd.isna(atr_at_entry) or atr_at_entry <= 0:
            result['error'] = f"Invalid ATR at entry: {atr_at_entry}"
            return result

        result['atr_at_entry'] = float(atr_at_entry)

        # Get position in the dataframe for slicing
        entry_pos = price_data.index.get_loc(entry_idx)
        df_len = len(price_data)

        # Calculate MFE and MAE for each horizon n (1 to max_days)
        for n in range(1, max_days + 1):
            end_pos = entry_pos + n
            if end_pos >= df_len:
                break

            # Get forward data (bars 1 to n after entry)
            forward_data = price_data.iloc[entry_pos + 1:end_pos + 1]

            if len(forward_data) == 0:
                continue

            # VALIDATION: Forward slice bounds check (sample check, not every n)
            if validate and n == 1:
                self.validator.validate_forward_slice(
                    entry_pos, n, df_len, len(forward_data), trade_id
                )

            # Use 'close' prices for MFE/MAE (adjusted prices, consistent with entry_price)
            closes = forward_data['close'].values
            max_close = closes.max()
            min_close = closes.min()

            # Calculate MFE and MAE based on trade direction
            if trade.is_long:
                # Long trade: profit when price rises, loss when price falls
                mfe_n = max(0, max_close - entry_price)
                mae_n = max(0, entry_price - min_close)
                max_possible_mfe = max_close - entry_price if max_close > entry_price else 0
            else:
                # Short trade: profit when price falls, loss when price rises
                mfe_n = max(0, entry_price - min_close)
                mae_n = max(0, max_close - entry_price)
                max_possible_mfe = entry_price - min_close if min_close < entry_price else 0

            # Store raw values
            result['mfe_raw'][n] = mfe_n
            result['mae_raw'][n] = mae_n

            # VALIDATION: Excursions check (sample check at key horizons)
            if validate and n in [1, 5, 10, 20, 30]:
                self.validator.validate_excursions(
                    mfe_n, mae_n, trade_id, n, max_possible_mfe
                )

            # Normalize by ATR at entry
            norm_mfe = mfe_n / atr_at_entry
            norm_mae = mae_n / atr_at_entry

            result['mfe_norm'][n] = norm_mfe
            result['mae_norm'][n] = norm_mae

            # VALIDATION: Normalization check (sample check at key horizons)
            if validate and n in [1, 5, 10, 20, 30]:
                self.validator.validate_normalization(norm_mfe, norm_mae, trade_id, n)

        return result

    def _get_trade_max_horizon(self, trade: TradeLogEntry) -> int:
        """
        Determine the maximum available horizon for a trade based on price data availability.

        Returns:
            Maximum number of forward days available, or 0 if trade cannot be analyzed.
        """
        price_data = self._get_price_data(trade.symbol, validate=False)
        if price_data is None:
            return 0

        entry_date = trade.entry_date
        if entry_date is None:
            return 0

        price_data = price_data.copy()
        price_data['date'] = pd.to_datetime(price_data['date'])

        # Find entry position
        entry_date_only = pd.to_datetime(entry_date).normalize()
        exact_match = price_data['date'] == entry_date_only

        if exact_match.any():
            entry_idx = price_data[exact_match].index[0]
        else:
            mask = price_data['date'] >= entry_date
            if not mask.any():
                return 0
            entry_idx = price_data[mask].index[0]

        entry_pos = price_data.index.get_loc(entry_idx)
        df_len = len(price_data)

        # Maximum horizon = days of data after entry
        return df_len - entry_pos - 1

    def calculate_aggregate_eratio(self, trades: List[TradeLogEntry], max_days: int = 30,
                                    progress_callback=None, validate: bool = True) -> Dict[str, Any]:
        """
        Calculate aggregate E-ratio across all trades for each horizon.

        IMPORTANT: Uses a CONSISTENT SAMPLE across all horizons to avoid survivorship bias.
        Only trades with sufficient forward data (>= max_days) are included in the calculation.

        For each horizon n:
        1. Collect normalized MFE and MAE from all trades
        2. Compute average normalized MFE and MAE
        3. E-ratio_n = AvgNormMFE_n / AvgNormMAE_n

        Returns:
            Dict containing:
            - e_ratios: Dict[int, float] mapping horizon -> E-ratio
            - e_ratios_winsorized: Dict[int, float] mapping horizon -> winsorized E-ratio
            - avg_mfe_norm: Dict[int, float] mapping horizon -> avg normalized MFE
            - avg_mae_norm: Dict[int, float] mapping horizon -> avg normalized MAE
            - trades_per_horizon: Dict[int, int] mapping horizon -> number of trades with data
            - trades_analyzed: int - total trades successfully analyzed
            - trades_with_errors: int - trades that had errors
            - trades_filtered_insufficient_data: int - trades filtered due to insufficient forward data
            - errors: List[str] - error messages
            - validation_report: ValidationReport - detailed validation results
        """
        # Reset validator for fresh calculation
        if validate:
            self.validator.reset()
            self._validated_symbols.clear()

        result = {
            'e_ratios': {},
            'e_ratios_winsorized': {},
            'avg_mfe_norm': {},
            'avg_mae_norm': {},
            'trades_per_horizon': {},
            'trades_analyzed': 0,
            'trades_with_errors': 0,
            'trades_filtered_insufficient_data': 0,
            'errors': [],
            'validation_report': None
        }

        # SURVIVORSHIP BIAS FIX: Pre-filter trades to ensure consistent sample
        # Only include trades that have AT LEAST max_days of forward data
        # This prevents the sample from shrinking at larger horizons
        eligible_trades = []
        for trade in trades:
            max_horizon = self._get_trade_max_horizon(trade)
            if max_horizon >= max_days:
                eligible_trades.append(trade)
            else:
                result['trades_filtered_insufficient_data'] += 1

        if validate and len(eligible_trades) < len(trades):
            filtered_count = len(trades) - len(eligible_trades)
            self.validator.report.add_result(ValidationResult(
                "consistent_sample_filter",
                True,
                f"Filtered {filtered_count} trades with <{max_days} days forward data for consistent sampling",
                severity="info",
                value=filtered_count
            ), "bias")

        # Collect all normalized MFE/MAE values per horizon
        all_mfe_norm: Dict[int, List[float]] = {n: [] for n in range(1, max_days + 1)}
        all_mae_norm: Dict[int, List[float]] = {n: [] for n in range(1, max_days + 1)}

        for i, trade in enumerate(eligible_trades):
            if progress_callback:
                progress_callback(i + 1, len(eligible_trades))

            excursions = self.calculate_trade_excursions(trade, max_days, validate=validate)

            if excursions['error']:
                result['trades_with_errors'] += 1
                if len(result['errors']) < 10:  # Limit error messages
                    result['errors'].append(f"{trade.symbol}: {excursions['error']}")
                continue

            if excursions['mfe_norm']:
                result['trades_analyzed'] += 1
                for n in excursions['mfe_norm']:
                    all_mfe_norm[n].append(excursions['mfe_norm'][n])
                    all_mae_norm[n].append(excursions['mae_norm'][n])

        # Calculate averages and E-ratio for each horizon
        for n in range(1, max_days + 1):
            if all_mfe_norm[n] and all_mae_norm[n]:
                avg_mfe = sum(all_mfe_norm[n]) / len(all_mfe_norm[n])
                avg_mae = sum(all_mae_norm[n]) / len(all_mae_norm[n])

                result['avg_mfe_norm'][n] = avg_mfe
                result['avg_mae_norm'][n] = avg_mae
                result['trades_per_horizon'][n] = len(all_mfe_norm[n])

                # E-ratio = AvgNormMFE / AvgNormMAE
                if avg_mae > 0:
                    result['e_ratios'][n] = avg_mfe / avg_mae
                else:
                    result['e_ratios'][n] = float('inf') if avg_mfe > 0 else 1.0

        # =========================================================================
        # VALIDATION CHECKS
        # =========================================================================
        if validate and result['trades_per_horizon']:
            # 1. Aggregation checks
            # Validate trades per horizon (min 20 trades recommended)
            min_trades = min(20, len(trades) // 2)  # Adjust based on dataset size
            self.validator.validate_trades_per_horizon(
                result['trades_per_horizon'],
                min_required=min_trades,
                check_up_to_day=min(60, max_days)
            )

            # Validate E-ratio bounds
            self.validator.validate_eratio_bounds(result['e_ratios'])

            # 2. Bias & Statistical checks
            # Sample size consistency (should be constant due to survivorship bias fix)
            self.validator.validate_sample_size_consistency(result['trades_per_horizon'])

            # Survivorship check (at 200 days if we have that much data)
            if max_days >= 200:
                self.validator.validate_survivorship(
                    result['trades_per_horizon'],
                    result['trades_analyzed'],
                    day=200
                )
            elif max_days >= 60:
                # Check at max available day
                self.validator.validate_survivorship(
                    result['trades_per_horizon'],
                    result['trades_analyzed'],
                    day=max_days
                )

            # Random baseline comparison
            self.validator.compare_with_random_baseline(result['e_ratios'])

            # 3. Calculate winsorized E-ratio (outlier impact check)
            result['e_ratios_winsorized'] = self.validator.calculate_winsorized_eratio(
                all_mfe_norm, all_mae_norm, percentile=99
            )

            # Compare original vs winsorized
            if result['e_ratios'] and result['e_ratios_winsorized']:
                # Calculate average difference
                diffs = []
                for n in result['e_ratios']:
                    if n in result['e_ratios_winsorized']:
                        orig = result['e_ratios'][n]
                        wins = result['e_ratios_winsorized'][n]
                        if not np.isinf(orig) and not np.isinf(wins):
                            diffs.append(abs(orig - wins) / max(orig, 0.001))

                if diffs:
                    avg_diff_pct = np.mean(diffs) * 100
                    self.validator.report.add_result(ValidationResult(
                        "outlier_impact",
                        avg_diff_pct < 20,  # Less than 20% difference is acceptable
                        f"Avg difference original vs winsorized: {avg_diff_pct:.1f}%",
                        severity="warning" if avg_diff_pct >= 20 else "info",
                        value=avg_diff_pct
                    ), "bias")

            # Store validation report in result
            result['validation_report'] = self.validator.report

        return result


# =============================================================================
# MAIN GUI
# =============================================================================

class CTkEdgeAnalysisGUI(ctk.CTk):
    """Main Edge Analysis GUI application."""

    # Default price data path - can be configured by user
    DEFAULT_DATA_PATH = Path('raw_data/daily')

    def __init__(self):
        super().__init__()

        # Initialize theme
        Theme.setup()

        self.title("Edge Analysis - Trade Log Analyzer")
        self.geometry("1400x900")
        self.configure(fg_color=Colors.BG_DARK)

        # Data path (configurable)
        self.data_path = self.DEFAULT_DATA_PATH

        # Data
        self.trade_loader = TradeLogLoader()
        self.eratio_calculator = ERatioCalculator(self.data_path)

        # State
        self.current_trades: List[TradeLogEntry] = []
        self.selected_trade: Optional[TradeLogEntry] = None
        self.max_eratio_days = 30

        self._create_layout()

    def _create_layout(self):
        """Create the main layout."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top bar - Load and Settings
        self._create_top_bar()

        # Main content area
        self._create_main_content()

    def _create_top_bar(self):
        """Create top bar with load button and settings."""
        top_frame = Theme.create_card(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Top row - main controls
        content = Theme.create_frame(top_frame)
        content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        # Title
        Theme.create_header(content, "Edge Analysis", size="l").pack(side="left")

        # Load button
        Theme.create_button(
            content, "Load Trade Logs...",
            command=self._load_trade_log,
            style="primary",
            width=150
        ).pack(side="left", padx=(Sizes.PAD_XL, Sizes.PAD_M))

        # File path display
        self.file_label = Theme.create_label(
            content, "No files loaded",
            text_color=Colors.TEXT_MUTED
        )
        self.file_label.pack(side="left", padx=Sizes.PAD_M)

        # Max E-ratio days setting
        settings_frame = Theme.create_frame(content)
        settings_frame.pack(side="right")

        Theme.create_label(settings_frame, "Max Days:").pack(side="left", padx=(0, Sizes.PAD_S))

        self.max_days_var = ctk.StringVar(value="30")
        days_entry = Theme.create_entry(settings_frame, width=50)
        days_entry.configure(textvariable=self.max_days_var)
        days_entry.pack(side="left")

        # Bottom row - data path configuration
        path_row = Theme.create_frame(top_frame)
        path_row.pack(fill="x", padx=Sizes.PAD_L, pady=(0, Sizes.PAD_M))

        Theme.create_label(path_row, "Price Data Path:", font=Fonts.BODY_S).pack(side="left")

        self.data_path_var = ctk.StringVar(value=str(self.data_path))
        data_path_entry = Theme.create_entry(path_row, width=400)
        data_path_entry.configure(textvariable=self.data_path_var)
        data_path_entry.pack(side="left", padx=(Sizes.PAD_S, Sizes.PAD_M))

        Theme.create_button(
            path_row, "Browse...",
            command=self._browse_data_path,
            style="secondary",
            width=80
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        Theme.create_button(
            path_row, "Apply",
            command=self._apply_data_path,
            style="secondary",
            width=60
        ).pack(side="left")

    def _create_main_content(self):
        """Create main content area with trade list and visualization tabs."""
        main_frame = Theme.create_frame(self)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Left panel - Trade list
        self._create_trade_list_panel(main_frame)

        # Right panel - Visualization tabs
        self._create_visualization_panel(main_frame)

    def _create_trade_list_panel(self, parent):
        """Create trade list panel on the left."""
        list_card = Theme.create_card(parent)
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_M))
        list_card.configure(width=400)
        list_card.grid_propagate(False)

        list_content = Theme.create_frame(list_card)
        list_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Header
        header_frame = Theme.create_frame(list_content)
        header_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_header(header_frame, "Trades", size="s").pack(side="left")

        self.trade_count_label = Theme.create_label(
            header_frame, "(0 trades)",
            text_color=Colors.TEXT_MUTED,
            font=Fonts.BODY_S
        )
        self.trade_count_label.pack(side="right")

        # Search/filter
        search_frame = Theme.create_frame(list_content)
        search_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self._on_search_change)

        search_entry = Theme.create_entry(search_frame, placeholder="Filter trades...")
        search_entry.configure(textvariable=self.search_var)
        search_entry.pack(fill="x")

        # Trade listbox
        self.trade_listbox = ctk.CTkScrollableFrame(
            list_content,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_M
        )
        self.trade_listbox.pack(fill="both", expand=True)

        # Instructions
        self.instructions_label = Theme.create_label(
            list_content,
            "Load trade log CSV files to begin.\nSelect a trade to view its details.",
            text_color=Colors.TEXT_MUTED,
            justify="left"
        )
        self.instructions_label.pack(pady=Sizes.PAD_M)

    def _create_visualization_panel(self, parent):
        """Create visualization tabs panel on the right."""
        viz_card = Theme.create_card(parent)
        viz_card.grid(row=0, column=1, sticky="nsew")

        # Create tabview
        self.tabview = Theme.create_tabview(viz_card)
        self.tabview.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Create tabs
        self.tabview.add("E-Ratio")
        self.tabview.add("R-Multiple")
        self.tabview.add("Validation")
        self.tabview.add("Trade Details")

        # Initialize tab contents
        self._create_eratio_tab(self.tabview.tab("E-Ratio"))
        self._create_rmultiple_tab(self.tabview.tab("R-Multiple"))
        self._create_validation_tab(self.tabview.tab("Validation"))
        self._create_details_tab(self.tabview.tab("Trade Details"))

    def _create_eratio_tab(self, parent):
        """Create E-ratio tab content."""
        self.eratio_frame = Theme.create_frame(parent)
        self.eratio_frame.pack(fill="both", expand=True)

        # Placeholder message
        self.eratio_placeholder = Theme.create_label(
            self.eratio_frame,
            "Load trade logs to view aggregate E-ratio curve.\n\n"
            "E-ratio measures entry edge across ALL trades by comparing:\n"
            "• AvgNormMFE (Average normalized Maximum Favorable Excursion)\n"
            "• AvgNormMAE (Average normalized Maximum Adverse Excursion)\n\n"
            "For each horizon n days after entry:\n"
            "  E-ratio(n) = AvgNormMFE(n) / AvgNormMAE(n)\n\n"
            "E-ratio > 1.0 indicates the strategy has positive edge at that horizon.",
            text_color=Colors.TEXT_MUTED,
            wraplength=500,
            justify="left"
        )
        self.eratio_placeholder.pack(expand=True)

    def _create_rmultiple_tab(self, parent):
        """Create R-multiple tab content."""
        self.rmultiple_frame = Theme.create_frame(parent)
        self.rmultiple_frame.pack(fill="both", expand=True)

        # Placeholder message
        self.rmultiple_placeholder = Theme.create_label(
            self.rmultiple_frame,
            "Load a trade log to view R-multiple distribution.\n\n"
            "R-multiple expresses profit/loss as a multiple of initial risk:\n"
            "• R = Entry Price - Stop Loss (your initial risk)\n"
            "• R-multiple = Profit/Loss ÷ R\n\n"
            "Example: +2R means you made twice your initial risk\n"
            "Example: -0.5R means you lost half your initial risk",
            text_color=Colors.TEXT_MUTED,
            wraplength=500,
            justify="left"
        )
        self.rmultiple_placeholder.pack(expand=True)

    def _create_validation_tab(self, parent):
        """Create validation report tab content."""
        self.validation_frame = Theme.create_frame(parent)
        self.validation_frame.pack(fill="both", expand=True)

        # Scrollable frame for validation results
        self.validation_scroll = ctk.CTkScrollableFrame(
            self.validation_frame,
            fg_color="transparent"
        )
        self.validation_scroll.pack(fill="both", expand=True)

        # Placeholder
        self.validation_placeholder = Theme.create_label(
            self.validation_scroll,
            "Load trade logs to view validation report.\n\n"
            "This tab shows data quality checks including:\n"
            "• Price data integrity (NaNs, date sorting)\n"
            "• ATR computation validation\n"
            "• Date alignment checks\n"
            "• Entry price consistency\n"
            "• E-ratio bounds validation\n"
            "• Sample size decay analysis\n"
            "• Outlier impact assessment\n"
            "• Random baseline comparison",
            text_color=Colors.TEXT_MUTED,
            wraplength=500,
            justify="left"
        )
        self.validation_placeholder.pack(expand=True, pady=Sizes.PAD_XL)

    def _create_details_tab(self, parent):
        """Create trade details tab content."""
        self.details_frame = Theme.create_frame(parent)
        self.details_frame.pack(fill="both", expand=True)

        # Scrollable frame for details
        self.details_scroll = ctk.CTkScrollableFrame(
            self.details_frame,
            fg_color="transparent"
        )
        self.details_scroll.pack(fill="both", expand=True)

        # Placeholder
        self.details_placeholder = Theme.create_label(
            self.details_scroll,
            "Select a trade to view its details.",
            text_color=Colors.TEXT_MUTED
        )
        self.details_placeholder.pack(expand=True, pady=Sizes.PAD_XL)

    def _browse_data_path(self):
        """Open folder dialog to select price data directory."""
        from tkinter import filedialog

        folder = filedialog.askdirectory(
            title="Select Price Data Directory",
            initialdir=self.data_path if self.data_path.exists() else Path.home()
        )

        if folder:
            self.data_path_var.set(folder)

    def _apply_data_path(self):
        """Apply the configured data path."""
        new_path = Path(self.data_path_var.get())

        if not new_path.exists():
            show_error(self, "Invalid Path", f"Directory does not exist:\n{new_path}")
            return

        self.data_path = new_path
        self.eratio_calculator.set_data_path(self.data_path)

        show_info(self, "Path Updated", f"Price data path set to:\n{self.data_path}")

    def _load_trade_log(self):
        """Open file dialog to load multiple trade logs."""
        from tkinter import filedialog

        filepaths = filedialog.askopenfilenames(
            title="Select Trade Log CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=Path("logs")
        )

        if not filepaths:
            return

        try:
            # Load all selected files
            trades = self.trade_loader.load_multiple([Path(f) for f in filepaths])
            self.current_trades = trades

            # Update UI
            if len(filepaths) == 1:
                self.file_label.configure(text=Path(filepaths[0]).name)
            else:
                self.file_label.configure(text=f"{len(filepaths)} files loaded")

            self.trade_count_label.configure(text=f"({len(trades)} trades)")
            self.instructions_label.pack_forget()

            # Populate trade list
            self._populate_trade_list()

            # Update charts
            self._update_eratio_chart()
            self._update_rmultiple_chart()

        except Exception as e:
            show_error(self, "Error Loading Files", str(e))

    def _bind_click_recursive(self, widget, trade, row):
        """Recursively bind click events to widget and all children."""
        # Bind click to select trade
        widget.bind("<Button-1>", lambda e, t=trade: self._on_trade_selected(t))

        # Bind hover effects to the row
        widget.bind("<Enter>", lambda e, r=row: r.configure(fg_color=Colors.SURFACE_HOVER))
        widget.bind("<Leave>", lambda e, r=row: r.configure(fg_color="transparent"))

        # Recursively bind to all children
        for child in widget.winfo_children():
            self._bind_click_recursive(child, trade, row)

    def _populate_trade_list(self, filter_text: str = ""):
        """Populate the trade list with loaded trades."""
        # Clear existing
        for widget in self.trade_listbox.winfo_children():
            widget.destroy()

        filter_text = filter_text.lower()

        for trade in self.current_trades:
            # Apply filter
            if filter_text:
                if filter_text not in trade.display_name.lower():
                    continue

            # Create trade row - use CTkButton-like styling for better clickability
            row = ctk.CTkFrame(
                self.trade_listbox,
                fg_color="transparent",
                corner_radius=Sizes.RADIUS_S,
                height=60
            )
            row.pack(fill="x", pady=2, padx=2)
            row.pack_propagate(False)  # Maintain consistent height

            # Trade info container
            info_frame = Theme.create_frame(row)
            info_frame.pack(fill="both", expand=True, padx=Sizes.PAD_S, pady=Sizes.PAD_XS)

            # Symbol and date
            date_str = trade.entry_date.strftime('%Y-%m-%d') if trade.entry_date else 'N/A'
            header_text = f"{trade.symbol} - {date_str}"

            header_label = Theme.create_label(
                info_frame, header_text,
                font=Fonts.LABEL_BOLD
            )
            header_label.pack(anchor="w")

            # R-multiple and P/L
            r_str = f"{trade.r_multiple:.2f}R" if trade.r_multiple is not None else "N/A"
            pl_color = Colors.SUCCESS if trade.pl > 0 else Colors.ERROR if trade.pl < 0 else Colors.TEXT_MUTED

            detail_frame = Theme.create_frame(info_frame)
            detail_frame.pack(anchor="w")

            id_label = Theme.create_label(
                detail_frame, f"ID: {trade.trade_id}",
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            )
            id_label.pack(side="left", padx=(0, Sizes.PAD_M))

            r_label = Theme.create_label(
                detail_frame, r_str,
                font=Fonts.BODY_S,
                text_color=pl_color
            )
            r_label.pack(side="left", padx=(0, Sizes.PAD_M))

            pl_label = Theme.create_label(
                detail_frame, f"${trade.pl:,.2f}",
                font=Fonts.BODY_S,
                text_color=pl_color
            )
            pl_label.pack(side="left")

            # Bind click events recursively to all widgets in the row
            self._bind_click_recursive(row, trade, row)

    def _on_search_change(self, *args):
        """Handle search text change."""
        self._populate_trade_list(self.search_var.get())

    def _on_trade_selected(self, trade: TradeLogEntry):
        """Handle trade selection - shows trade details."""
        self.selected_trade = trade

        # Update details tab
        self._update_details_tab(trade)

        # Switch to Trade Details tab
        self.tabview.set("Trade Details")

    def _update_eratio_chart(self):
        """Update aggregate E-ratio chart across all trades."""
        # Clear existing content
        for widget in self.eratio_frame.winfo_children():
            widget.destroy()

        if not MATPLOTLIB_AVAILABLE:
            Theme.create_label(
                self.eratio_frame,
                "Matplotlib not available. Install it with: pip install matplotlib",
                text_color=Colors.ERROR
            ).pack(expand=True)
            return

        if not self.current_trades:
            Theme.create_label(
                self.eratio_frame,
                "No trades loaded.",
                text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            return

        # Show loading message
        loading_label = Theme.create_label(
            self.eratio_frame,
            "Calculating aggregate E-ratio across all trades...\nThis may take a moment.",
            text_color=Colors.TEXT_MUTED
        )
        loading_label.pack(expand=True)
        self.update_idletasks()

        # Get max days setting
        try:
            max_days = int(self.max_days_var.get())
        except ValueError:
            max_days = 30

        # Calculate aggregate E-ratio
        result = self.eratio_calculator.calculate_aggregate_eratio(self.current_trades, max_days)

        # Remove loading message
        loading_label.destroy()

        if result['trades_analyzed'] == 0:
            error_msg = "Could not calculate E-ratio for any trades."
            if result['errors']:
                error_msg += f"\n\nSample errors:\n" + "\n".join(result['errors'][:5])
            error_msg += "\n\nMake sure the Price Data Path is configured correctly."
            Theme.create_label(
                self.eratio_frame,
                error_msg,
                text_color=Colors.ERROR,
                wraplength=500
            ).pack(expand=True)
            return

        if not result['e_ratios']:
            Theme.create_label(
                self.eratio_frame,
                "No E-ratio data available.",
                text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            return

        # Create figure
        fig = Figure(figsize=(10, 7), facecolor=Colors.BG_DARK)

        # Create two subplots - E-ratio on top, AvgMFE/AvgMAE on bottom
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for ax in [ax1, ax2]:
            ax.set_facecolor(Colors.SURFACE)
            ax.tick_params(colors=Colors.TEXT_PRIMARY)
            for spine in ax.spines.values():
                spine.set_color(Colors.BORDER)

        # Plot E-ratio
        days = sorted(result['e_ratios'].keys())
        ratios = [result['e_ratios'][d] for d in days]
        trade_counts = [result['trades_per_horizon'].get(d, 0) for d in days]

        # Cap infinite values for display
        display_ratios = [min(r, 10) for r in ratios]

        ax1.plot(days, display_ratios, color=Colors.PRIMARY_LIGHT, linewidth=2, marker='o', markersize=4, label='E-ratio')

        # Plot winsorized E-ratio for comparison (if available)
        if result.get('e_ratios_winsorized'):
            winsorized_ratios = [min(result['e_ratios_winsorized'].get(d, r), 10) for d, r in zip(days, ratios)]
            ax1.plot(days, winsorized_ratios, color='#FFD700', linewidth=1.5, linestyle='--',
                     marker='s', markersize=3, alpha=0.7, label='Winsorized (99th pct)')

        ax1.axhline(y=1.0, color=Colors.TEXT_MUTED, linestyle='--', alpha=0.7, label='E-ratio = 1.0 (no edge)')
        ax1.fill_between(days, 1.0, display_ratios, where=[r > 1 for r in display_ratios],
                         color=Colors.SUCCESS, alpha=0.3, label='Positive edge')
        ax1.fill_between(days, 1.0, display_ratios, where=[r < 1 for r in display_ratios],
                         color=Colors.ERROR, alpha=0.3, label='Negative edge')

        ax1.set_ylabel('E-Ratio', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_title(f'Aggregate E-Ratio Across {result["trades_analyzed"]} Trades',
                      color=Colors.TEXT_PRIMARY, fontsize=12, pad=10)
        ax1.legend(facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                   labelcolor=Colors.TEXT_PRIMARY, fontsize=8, loc='upper left')
        ax1.grid(True, alpha=0.3, color=Colors.BORDER)
        ax1.set_xlim(0, max(days) + 1)

        # Add secondary y-axis for trade count
        ax1_twin = ax1.twinx()
        ax1_twin.bar(days, trade_counts, alpha=0.3, color='#888888', width=0.8, label='Trade Count')
        ax1_twin.set_ylabel('Trade Count', color='#888888', fontsize=10)
        ax1_twin.tick_params(axis='y', colors='#888888')
        ax1_twin.spines['right'].set_color('#888888')

        # Set trade count y-axis to start at 0 and have some headroom
        if trade_counts:
            ax1_twin.set_ylim(0, max(trade_counts) * 1.5)

        # Add trade count to legend
        ax1_twin.legend(facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                        labelcolor='#888888', fontsize=8, loc='upper right')

        # Plot Average Normalized MFE and MAE
        mfe_values = [result['avg_mfe_norm'].get(d, 0) for d in days]
        mae_values = [result['avg_mae_norm'].get(d, 0) for d in days]

        ax2.plot(days, mfe_values, color=Colors.SUCCESS, linewidth=2, marker='o', markersize=4, label='Avg Norm MFE')
        ax2.plot(days, mae_values, color=Colors.ERROR, linewidth=2, marker='o', markersize=4, label='Avg Norm MAE')

        ax2.set_xlabel('Days After Entry (Horizon)', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_ylabel('ATR Multiples', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_title('Average Normalized MFE vs MAE Across All Trades',
                      color=Colors.TEXT_PRIMARY, fontsize=11, pad=10)
        ax2.legend(facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                   labelcolor=Colors.TEXT_PRIMARY, fontsize=8)
        ax2.grid(True, alpha=0.3, color=Colors.BORDER)
        ax2.set_xlim(0, max(days) + 1)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.eratio_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        stats_frame = Theme.create_frame(self.eratio_frame)
        stats_frame.pack(fill="x", pady=(Sizes.PAD_S, 0))

        # Find peak E-ratio
        max_eratio = max(ratios)
        max_day = days[ratios.index(max_eratio)]
        max_eratio_display = f"{max_eratio:.2f}" if max_eratio < 100 else "inf"

        # Find first day where E-ratio > 1
        edge_days = [d for d in days if result['e_ratios'][d] > 1]
        first_edge_day = min(edge_days) if edge_days else "N/A"

        # Build stats text with filtered trades info if applicable
        filtered_count = result.get('trades_filtered_insufficient_data', 0)
        stats_parts = [
            f"Trades Analyzed: {result['trades_analyzed']}",
        ]
        if filtered_count > 0:
            stats_parts.append(f"Filtered (insufficient data): {filtered_count}")
        stats_parts.extend([
            f"Errors: {result['trades_with_errors']}",
            f"Peak E-Ratio: {max_eratio_display} (day {max_day})",
            f"First Day with Edge: {first_edge_day}",
        ])

        Theme.create_label(
            stats_frame,
            " | ".join(stats_parts),
            text_color=Colors.TEXT_SECONDARY
        ).pack()

        # Update validation tab with results
        self._update_validation_tab(result)

    def _update_validation_tab(self, eratio_result: Dict[str, Any]):
        """Update validation report tab with results."""
        # Clear existing content
        for widget in self.validation_scroll.winfo_children():
            widget.destroy()

        validation_report = eratio_result.get('validation_report')
        if validation_report is None:
            Theme.create_label(
                self.validation_scroll,
                "No validation data available.",
                text_color=Colors.TEXT_MUTED
            ).pack(expand=True, pady=Sizes.PAD_XL)
            return

        # Summary header
        Theme.create_header(
            self.validation_scroll,
            "Validation Report",
            size="m"
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Overall summary
        summary_frame = Theme.create_frame(self.validation_scroll)
        summary_frame.pack(fill="x", pady=(0, Sizes.PAD_M))

        passed_color = Colors.SUCCESS if validation_report.error_count == 0 else Colors.WARNING
        Theme.create_label(
            summary_frame,
            f"Passed: {validation_report.passed_count} | "
            f"Failed: {validation_report.failed_count} | "
            f"Errors: {validation_report.error_count} | "
            f"Warnings: {validation_report.warning_count}",
            text_color=passed_color,
            font=Fonts.LABEL_BOLD
        ).pack(anchor="w")

        # Create sections for each category
        categories = [
            ("Data Input & Preprocessing", validation_report.data_checks),
            ("Aggregation & E-ratio", validation_report.aggregation_checks),
            ("Bias & Statistical", validation_report.bias_checks),
        ]

        for category_name, checks in categories:
            if not checks:
                continue

            # Section separator
            ctk.CTkFrame(self.validation_scroll, fg_color=Colors.BORDER, height=1).pack(
                fill="x", pady=Sizes.PAD_M
            )

            # Section header
            passed_in_category = sum(1 for c in checks if c.passed)
            header_color = Colors.SUCCESS if passed_in_category == len(checks) else Colors.WARNING

            Theme.create_header(
                self.validation_scroll,
                f"{category_name} ({passed_in_category}/{len(checks)})",
                size="s"
            ).pack(anchor="w", pady=(0, Sizes.PAD_S))

            # Show check results (limit to important ones)
            shown_checks = []
            # First show failures
            failures = [c for c in checks if not c.passed]
            shown_checks.extend(failures[:10])
            # Then show passes (limited)
            passes = [c for c in checks if c.passed]
            shown_checks.extend(passes[:5])

            for check in shown_checks:
                check_frame = Theme.create_frame(self.validation_scroll)
                check_frame.pack(fill="x", pady=2)

                # Status icon
                if check.passed:
                    status_icon = "✓"
                    status_color = Colors.SUCCESS
                elif check.severity == "error":
                    status_icon = "✗"
                    status_color = Colors.ERROR
                else:
                    status_icon = "!"
                    status_color = Colors.WARNING

                Theme.create_label(
                    check_frame,
                    status_icon,
                    text_color=status_color,
                    font=Fonts.LABEL_BOLD,
                    width=20
                ).pack(side="left")

                Theme.create_label(
                    check_frame,
                    check.message,
                    text_color=Colors.TEXT_SECONDARY if check.passed else status_color,
                    font=Fonts.BODY_S
                ).pack(side="left", padx=Sizes.PAD_S)

            if len(failures) > 10:
                Theme.create_label(
                    self.validation_scroll,
                    f"... and {len(failures) - 10} more issues",
                    text_color=Colors.TEXT_MUTED,
                    font=Fonts.BODY_XS
                ).pack(anchor="w", pady=(Sizes.PAD_XS, 0))

        # Outliers section
        if validation_report.outliers:
            ctk.CTkFrame(self.validation_scroll, fg_color=Colors.BORDER, height=1).pack(
                fill="x", pady=Sizes.PAD_M
            )

            Theme.create_header(
                self.validation_scroll,
                f"Outliers Logged ({len(validation_report.outliers)})",
                size="s"
            ).pack(anchor="w", pady=(0, Sizes.PAD_S))

            for outlier in validation_report.outliers[:10]:
                Theme.create_label(
                    self.validation_scroll,
                    f"Trade {outlier['trade_id']} ({outlier['symbol']}): "
                    f"{outlier['field']} = {outlier['value']:.2f} (threshold: {outlier['threshold']})",
                    text_color=Colors.WARNING,
                    font=Fonts.BODY_S
                ).pack(anchor="w", pady=1)

            if len(validation_report.outliers) > 10:
                Theme.create_label(
                    self.validation_scroll,
                    f"... and {len(validation_report.outliers) - 10} more outliers",
                    text_color=Colors.TEXT_MUTED,
                    font=Fonts.BODY_XS
                ).pack(anchor="w", pady=(Sizes.PAD_XS, 0))

        # Trade details summary
        ctk.CTkFrame(self.validation_scroll, fg_color=Colors.BORDER, height=1).pack(
            fill="x", pady=Sizes.PAD_M
        )

        Theme.create_header(
            self.validation_scroll,
            "Analysis Summary",
            size="s"
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Trades per horizon info
        if eratio_result.get('trades_per_horizon'):
            horizons = eratio_result['trades_per_horizon']
            min_trades = min(horizons.values()) if horizons else 0
            max_trades = max(horizons.values()) if horizons else 0

            Theme.create_label(
                self.validation_scroll,
                f"Trades per horizon: {min_trades} (min) to {max_trades} (max)",
                text_color=Colors.TEXT_SECONDARY,
                font=Fonts.BODY_S
            ).pack(anchor="w")

        # E-ratio range info
        if eratio_result.get('e_ratios'):
            eratios = [e for e in eratio_result['e_ratios'].values() if not np.isinf(e)]
            if eratios:
                Theme.create_label(
                    self.validation_scroll,
                    f"E-ratio range: {min(eratios):.3f} to {max(eratios):.3f}",
                    text_color=Colors.TEXT_SECONDARY,
                    font=Fonts.BODY_S
                ).pack(anchor="w")

    def _update_rmultiple_chart(self):
        """Update R-multiple distribution chart."""
        # Clear existing content
        for widget in self.rmultiple_frame.winfo_children():
            widget.destroy()

        if not MATPLOTLIB_AVAILABLE:
            Theme.create_label(
                self.rmultiple_frame,
                "Matplotlib not available. Install it with: pip install matplotlib",
                text_color=Colors.ERROR
            ).pack(expand=True)
            return

        # Get R-multiples from all trades
        r_multiples = [t.r_multiple for t in self.current_trades if t.r_multiple is not None]

        if not r_multiples:
            Theme.create_label(
                self.rmultiple_frame,
                "No R-multiple data available.\nTrades may be missing stop loss information.",
                text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            return

        winning = [r for r in r_multiples if r >= 0]
        losing = [r for r in r_multiples if r < 0]

        # Create figure with two subplots (back to back histograms)
        fig = Figure(figsize=(10, 6), facecolor=Colors.BG_DARK)

        ax1 = fig.add_subplot(121)  # Losing trades
        ax2 = fig.add_subplot(122)  # Winning trades

        for ax in [ax1, ax2]:
            ax.set_facecolor(Colors.SURFACE)
            ax.tick_params(colors=Colors.TEXT_PRIMARY)
            for spine in ax.spines.values():
                spine.set_color(Colors.BORDER)

        # Plot losing trades histogram (left)
        if losing:
            min_r = min(losing)
            bins = np.arange(min_r - 0.25, 0.25, 0.25)
            if len(bins) < 2:
                bins = np.linspace(min_r, 0, 10)
            ax1.hist(losing, bins=bins, color=Colors.CHART_NEGATIVE, edgecolor=Colors.BG_DARK, alpha=0.8)

        ax1.set_xlabel('R-Multiple', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_ylabel('Number of Trades', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax1.set_title(f'Losing Trades\n({len(losing)} trades)',
                      color=Colors.ERROR, fontsize=11, pad=10)
        ax1.axvline(x=0, color=Colors.TEXT_MUTED, linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3, color=Colors.BORDER)

        # Plot winning trades histogram (right)
        if winning:
            max_r = max(winning)
            bins = np.arange(-0.25, max_r + 0.5, 0.25)
            if len(bins) < 2:
                bins = np.linspace(0, max_r, 10)
            ax2.hist(winning, bins=bins, color=Colors.CHART_POSITIVE, edgecolor=Colors.BG_DARK, alpha=0.8)

        ax2.set_xlabel('R-Multiple', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_ylabel('Number of Trades', color=Colors.TEXT_PRIMARY, fontsize=10)
        ax2.set_title(f'Winning Trades\n({len(winning)} trades)',
                      color=Colors.SUCCESS, fontsize=11, pad=10)
        ax2.axvline(x=0, color=Colors.TEXT_MUTED, linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3, color=Colors.BORDER)

        fig.suptitle('R-Multiple Distribution', color=Colors.TEXT_PRIMARY, fontsize=13, y=0.98)
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.rmultiple_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        stats_frame = Theme.create_frame(self.rmultiple_frame)
        stats_frame.pack(fill="x", pady=(Sizes.PAD_S, 0))

        avg_r = sum(r_multiples) / len(r_multiples)
        avg_win_r = sum(winning) / len(winning) if winning else 0
        avg_loss_r = sum(losing) / len(losing) if losing else 0
        win_rate = len(winning) / len(r_multiples) * 100

        Theme.create_label(
            stats_frame,
            f"Win Rate: {win_rate:.1f}% | "
            f"Avg R: {avg_r:.2f} | "
            f"Avg Win: {avg_win_r:.2f}R | "
            f"Avg Loss: {avg_loss_r:.2f}R | "
            f"Total Trades: {len(r_multiples)}",
            text_color=Colors.TEXT_SECONDARY
        ).pack()

    def _update_details_tab(self, trade: TradeLogEntry):
        """Update trade details tab."""
        # Clear existing content
        for widget in self.details_scroll.winfo_children():
            widget.destroy()

        # Trade header
        Theme.create_header(
            self.details_scroll,
            f"Trade {trade.trade_id}",
            size="m"
        ).pack(anchor="w", pady=(0, Sizes.PAD_M))

        # Create details grid
        details = [
            ("Symbol", trade.symbol),
            ("Entry Date", trade.entry_date.strftime('%Y-%m-%d %H:%M') if trade.entry_date else "N/A"),
            ("Entry Price", f"${trade.entry_price:.4f}"),
            ("Exit Date", trade.exit_date.strftime('%Y-%m-%d %H:%M') if trade.exit_date else "N/A"),
            ("Exit Price", f"${trade.exit_price:.4f}"),
            ("Quantity", f"{trade.quantity:.4f}"),
            ("", ""),
            ("Initial Stop Loss", f"${trade.initial_stop_loss:.4f}" if trade.initial_stop_loss else "N/A"),
            ("Final Stop Loss", f"${trade.final_stop_loss:.4f}" if trade.final_stop_loss else "N/A"),
            ("", ""),
            ("P/L", f"${trade.pl:,.2f}"),
            ("P/L %", f"{trade.pl_pct:.2f}%"),
            ("R-Multiple", f"{trade.r_multiple:.2f}R" if trade.r_multiple is not None else "N/A"),
            ("", ""),
            ("Entry Reason", trade.entry_reason or "N/A"),
            ("Exit Reason", trade.exit_reason or "N/A"),
        ]

        for label, value in details:
            if not label:
                ctk.CTkFrame(self.details_scroll, fg_color=Colors.BORDER, height=1).pack(
                    fill="x", pady=Sizes.PAD_S
                )
                continue

            row = Theme.create_frame(self.details_scroll)
            row.pack(fill="x", pady=Sizes.PAD_XS)

            Theme.create_label(row, f"{label}:", width=150, font=Fonts.LABEL_BOLD).pack(side="left")

            # Color P/L values
            text_color = Colors.TEXT_PRIMARY
            if "P/L" in label or "R-Multiple" in label:
                if trade.pl > 0:
                    text_color = Colors.SUCCESS
                elif trade.pl < 0:
                    text_color = Colors.ERROR

            Theme.create_label(row, str(value), text_color=text_color).pack(side="left")


def main():
    """Main entry point for GUI."""
    app = CTkEdgeAnalysisGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
