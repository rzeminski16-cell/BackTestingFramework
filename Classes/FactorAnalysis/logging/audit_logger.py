"""
Audit Logger for Factor Analysis.

Provides comprehensive logging of all decisions, data transformations,
and analysis steps for reproducibility and debugging.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from contextlib import contextmanager

from ..config.factor_config import FactorAnalysisConfig, Verbosity


class LogLevel(Enum):
    """Log levels for audit messages."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditLogger:
    """
    Comprehensive audit trail logger for Factor Analysis.

    Logs every significant decision, data transformation, and analysis step
    to support reproducibility and debugging.

    Features:
    - Timestamped entries
    - Configurable verbosity
    - Section-based organization
    - Configuration snapshot logging
    - File and console output
    - Context managers for nested logging

    Example:
        logger = AuditLogger(output_path="./logs/audit.txt", verbosity=Verbosity.HIGH)
        logger.start_session("momentum_value_2025", config)

        with logger.section("DATA_LOADING"):
            logger.info("Loading trade logs", {"files": 3, "total_rows": 1250})
            logger.warning("Missing data detected", {"trades_affected": 52})

        logger.end_session()
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        verbosity: Verbosity = Verbosity.HIGH,
        console_output: bool = True,
        enabled: bool = True
    ):
        """
        Initialize AuditLogger.

        Args:
            output_path: Path to output file (None for console only)
            verbosity: Logging verbosity level
            console_output: Whether to also print to console
            enabled: Whether logging is enabled
        """
        self.output_path = Path(output_path) if output_path else None
        self.verbosity = verbosity
        self.console_output = console_output
        self.enabled = enabled

        self._file_handle = None
        self._section_stack: List[str] = []
        self._indent_level = 0
        self._session_start: Optional[datetime] = None
        self._profile_name: Optional[str] = None
        self._stats: Dict[str, int] = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'debug': 0
        }

        if self.output_path:
            self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _should_log(self, level: LogLevel) -> bool:
        """Determine if a message should be logged based on verbosity."""
        if not self.enabled:
            return False

        if self.verbosity == Verbosity.LOW:
            return level in [LogLevel.ERROR, LogLevel.CRITICAL]
        elif self.verbosity == Verbosity.MEDIUM:
            return level in [LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.INFO]
        else:  # HIGH
            return True

    def _format_timestamp(self) -> str:
        """Format current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_message(
        self,
        level: LogLevel,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a log message."""
        timestamp = self._format_timestamp()
        indent = "  " * self._indent_level

        # Build the message
        section_prefix = f"[{'/'.join(self._section_stack)}] " if self._section_stack else ""
        formatted = f"[{timestamp}] {level.value}: {indent}{section_prefix}{message}"

        # Add details if present
        if details:
            if self.verbosity == Verbosity.HIGH:
                # Full details on separate lines
                for key, value in details.items():
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, default=str, indent=2)
                        formatted += f"\n  {indent}  {key}: {value_str}"
                    else:
                        formatted += f"\n  {indent}  {key}: {value}"
            else:
                # Compact format
                detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
                formatted += f" ({detail_str})"

        return formatted

    def _write(self, message: str) -> None:
        """Write message to file and/or console."""
        if self.console_output:
            print(message)

        if self._file_handle:
            self._file_handle.write(message + "\n")
            self._file_handle.flush()

    def start_session(
        self,
        profile_name: str,
        config: Optional[FactorAnalysisConfig] = None
    ) -> None:
        """
        Start a new logging session.

        Args:
            profile_name: Name of the analysis profile
            config: Configuration to snapshot
        """
        if not self.enabled:
            return

        self._session_start = datetime.now()
        self._profile_name = profile_name
        self._stats = {'info': 0, 'warning': 0, 'error': 0, 'debug': 0}

        # Open file if output path specified
        if self.output_path:
            self._file_handle = open(self.output_path, 'w')

        # Write session header
        self._write("=" * 80)
        self._write(f"FACTOR ANALYSIS SESSION")
        self._write("=" * 80)
        self._write(f"Started: {self._format_timestamp()}")
        self._write(f"Profile: {profile_name}")

        # Log configuration snapshot if provided
        if config:
            self._write("")
            self._write("CONFIGURATION SNAPSHOT")
            self._write("-" * 40)
            self._write(f"Profile Version: {config.version}")
            self._write(f"Config Hash: {config.compute_hash()}")
            if self.verbosity == Verbosity.HIGH:
                self._write("")
                self._write(json.dumps(config.to_dict(), indent=2, default=str))
            self._write("-" * 40)

        self._write("")

    def end_session(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """
        End the logging session.

        Args:
            summary: Optional summary data to include
        """
        if not self.enabled:
            return

        duration = None
        if self._session_start:
            duration = datetime.now() - self._session_start

        self._write("")
        self._write("=" * 80)
        self._write("SESSION COMPLETE")
        self._write("=" * 80)
        self._write(f"Ended: {self._format_timestamp()}")
        if duration:
            self._write(f"Duration: {duration}")
        self._write("")
        self._write("LOG STATISTICS:")
        for level, count in self._stats.items():
            self._write(f"  {level.upper()}: {count}")

        if summary:
            self._write("")
            self._write("SESSION SUMMARY:")
            for key, value in summary.items():
                self._write(f"  {key}: {value}")

        self._write("=" * 80)

        # Close file handle
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    @contextmanager
    def section(self, name: str):
        """
        Context manager for logging sections.

        Args:
            name: Section name

        Example:
            with logger.section("DATA_LOADING"):
                logger.info("Loading files...")
        """
        self.start_section(name)
        try:
            yield
        finally:
            self.end_section()

    def start_section(self, name: str) -> None:
        """Start a new logging section."""
        if not self.enabled:
            return

        self._section_stack.append(name)
        self._indent_level = len(self._section_stack)

        self._write("")
        self._write(f"{'=' * 60}")
        self._write(f"[{self._format_timestamp()}] === {name} ===")
        self._write(f"{'=' * 60}")

    def end_section(self) -> None:
        """End the current logging section."""
        if not self.enabled or not self._section_stack:
            return

        section_name = self._section_stack.pop()
        self._indent_level = len(self._section_stack)

        self._write(f"[{self._format_timestamp()}] === END {section_name} ===")
        self._write("")

    def debug(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        if self._should_log(LogLevel.DEBUG):
            self._stats['debug'] += 1
            self._write(self._format_message(LogLevel.DEBUG, message, details))

    def info(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        if self._should_log(LogLevel.INFO):
            self._stats['info'] += 1
            self._write(self._format_message(LogLevel.INFO, message, details))

    def warning(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        if self._should_log(LogLevel.WARNING):
            self._stats['warning'] += 1
            self._write(self._format_message(LogLevel.WARNING, message, details))

    def error(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message."""
        if self._should_log(LogLevel.ERROR):
            self._stats['error'] += 1
            self._write(self._format_message(LogLevel.ERROR, message, details))

    def critical(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log a critical error message."""
        if self._should_log(LogLevel.CRITICAL):
            self._write(self._format_message(LogLevel.CRITICAL, message, details))

    def log_data_summary(
        self,
        source_name: str,
        row_count: int,
        column_count: int,
        date_range: Optional[tuple] = None,
        symbols: Optional[List[str]] = None
    ) -> None:
        """Log a data source summary."""
        details = {
            'rows': row_count,
            'columns': column_count
        }
        if date_range:
            details['date_range'] = f"{date_range[0]} to {date_range[1]}"
        if symbols:
            details['symbols'] = symbols if len(symbols) <= 5 else f"{symbols[:5]}... ({len(symbols)} total)"

        self.info(f"Loaded {source_name}", details)

    def log_trade_classification(
        self,
        good_count: int,
        bad_count: int,
        indeterminate_count: int,
        config_used: Dict[str, Any]
    ) -> None:
        """Log trade classification results."""
        total = good_count + bad_count + indeterminate_count
        self.info("Trade classification complete", {
            'good': f"{good_count} ({100*good_count/total:.1f}%)" if total > 0 else "0",
            'bad': f"{bad_count} ({100*bad_count/total:.1f}%)" if total > 0 else "0",
            'indeterminate': f"{indeterminate_count} ({100*indeterminate_count/total:.1f}%)" if total > 0 else "0",
            'total': total,
            'thresholds': config_used
        })

    def log_temporal_alignment(
        self,
        source_name: str,
        trades_matched: int,
        trades_missing: int,
        method: str
    ) -> None:
        """Log temporal alignment results."""
        total = trades_matched + trades_missing
        self.info(f"Temporal alignment for {source_name}", {
            'matched': f"{trades_matched} ({100*trades_matched/total:.1f}%)" if total > 0 else "0",
            'missing': trades_missing,
            'method': method
        })

    def log_factor_engineering(
        self,
        category: str,
        factor_count: int,
        trades_with_data: int,
        total_trades: int
    ) -> None:
        """Log factor engineering results."""
        coverage = 100 * trades_with_data / total_trades if total_trades > 0 else 0
        self.info(f"Factor engineering: {category}", {
            'factors': factor_count,
            'coverage': f"{coverage:.1f}%",
            'trades_with_data': trades_with_data,
            'total_trades': total_trades
        })

    def log_outliers(
        self,
        factor_name: str,
        outlier_count: int,
        total_values: int,
        threshold: float,
        action: str
    ) -> None:
        """Log outlier detection results."""
        if outlier_count > 0:
            self.warning(f"Outliers detected in {factor_name}", {
                'count': outlier_count,
                'percentage': f"{100*outlier_count/total_values:.2f}%",
                'threshold_zscore': threshold,
                'action': action
            })

    def log_analysis_result(
        self,
        analysis_name: str,
        significant_factors: List[str],
        total_factors: int,
        method: str
    ) -> None:
        """Log statistical analysis results."""
        self.info(f"Analysis complete: {analysis_name}", {
            'significant_factors': len(significant_factors),
            'total_factors': total_factors,
            'method': method,
            'top_factors': significant_factors[:5] if significant_factors else []
        })

    def log_scenario(
        self,
        scenario_name: str,
        n_trades: int,
        win_rate: float,
        calmar: float,
        classification: str
    ) -> None:
        """Log scenario analysis result."""
        self.info(f"Scenario: {scenario_name}", {
            'n_trades': n_trades,
            'win_rate': f"{win_rate:.1%}",
            'calmar': f"{calmar:.2f}",
            'classification': classification
        })

    def log_exception(self, exception: Exception, context: str = "") -> None:
        """Log an exception with traceback."""
        import traceback
        self.error(f"Exception in {context}" if context else "Exception occurred", {
            'type': type(exception).__name__,
            'message': str(exception),
            'traceback': traceback.format_exc()
        })

    def log_table(self, title: str, headers: List[str], rows: List[List[Any]]) -> None:
        """Log tabular data."""
        if not self._should_log(LogLevel.INFO):
            return

        self._write(f"\n{title}")
        self._write("-" * len(title))

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Format header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        self._write(header_line)
        self._write("-" * len(header_line))

        # Format rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            self._write(row_line)

        self._write("")

    @classmethod
    def from_config(cls, config: FactorAnalysisConfig) -> "AuditLogger":
        """Create AuditLogger from configuration."""
        audit_config = config.output.audit_log
        return cls(
            output_path=audit_config.output_path,
            verbosity=audit_config.verbosity,
            enabled=audit_config.enabled
        )
