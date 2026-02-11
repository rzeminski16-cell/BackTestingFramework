"""
Data Collection Report Generator.

Generates comprehensive reports after data collection to validate data quality
and provide insights into the collected datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import json

from .validation_engine import ValidationEngine, ValidationReport


@dataclass
class FileStatistics:
    """Statistics for a single data file."""
    file_name: str
    file_path: str
    symbol: str
    data_type: str  # daily, weekly, fundamental, insider, forex, options

    # File metrics
    file_size_kb: float
    num_rows: int
    num_columns: int
    column_names: List[str]

    # Data quality
    total_cells: int
    missing_cells: int
    missing_percentage: float

    # Date range
    date_column: Optional[str] = None
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None
    date_range_days: Optional[int] = None

    # Numeric statistics (for price/value columns)
    numeric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Validation
    validation_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    validation_errors: int = 0
    validation_warnings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CollectionSummary:
    """Summary of the entire data collection session."""
    session_id: str
    timestamp: str
    total_files: int
    total_size_mb: float
    total_rows: int
    total_cells: int

    # By data type
    files_by_type: Dict[str, int] = field(default_factory=dict)
    rows_by_type: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    overall_quality_score: float = 100.0
    files_passed_validation: int = 0
    files_failed_validation: int = 0
    total_errors: int = 0
    total_warnings: int = 0

    # Missing data
    total_missing_cells: int = 0
    overall_missing_percentage: float = 0.0

    # Date coverage
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CollectionReportGenerator:
    """
    Generates comprehensive reports for data collection sessions.

    Analyzes collected data files and generates:
    - File-level statistics
    - Session-level summary
    - Data quality metrics
    - Missing data analysis
    - Value range analysis
    - Human-readable report
    """

    def __init__(
        self,
        data_dir: Path,
        validation_engine: Optional[ValidationEngine] = None
    ):
        """
        Initialize report generator.

        Args:
            data_dir: Base directory containing collected data
            validation_engine: Optional validation engine for quality checks
        """
        self.data_dir = Path(data_dir)
        self.validation_engine = validation_engine

    def generate_report(
        self,
        session_id: Optional[str] = None,
        validate_data: bool = True
    ) -> Tuple[CollectionSummary, List[FileStatistics]]:
        """
        Generate comprehensive collection report.

        Args:
            session_id: Optional session identifier
            validate_data: Whether to run validation on collected data

        Returns:
            Tuple of (CollectionSummary, List[FileStatistics])
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Collect statistics for all files
        file_stats = []

        # Check each data type directory
        data_types = ['daily', 'weekly', 'fundamentals', 'insider_transactions', 'forex', 'options']

        for data_type in data_types:
            type_dir = self.data_dir / 'raw_data' / data_type
            if not type_dir.exists():
                continue

            # Process all CSV files in this directory
            for csv_file in type_dir.glob('**/*.csv'):
                stats = self._analyze_file(csv_file, data_type, validate_data)
                if stats:
                    file_stats.append(stats)

        # Generate summary
        summary = self._generate_summary(session_id, file_stats)

        return summary, file_stats

    def _analyze_file(
        self,
        file_path: Path,
        data_type: str,
        validate: bool
    ) -> Optional[FileStatistics]:
        """
        Analyze a single data file.

        Args:
            file_path: Path to CSV file
            data_type: Type of data (daily, weekly, etc.)
            validate: Whether to run validation

        Returns:
            FileStatistics or None if file cannot be read
        """
        try:
            # Read file
            df = pd.read_csv(file_path)

            # Extract symbol from filename
            symbol = file_path.stem.split('_')[0]

            # File metrics
            file_size_kb = file_path.stat().st_size / 1024
            num_rows = len(df)
            num_columns = len(df.columns)
            column_names = df.columns.tolist()

            # Missing data analysis
            total_cells = num_rows * num_columns
            missing_cells = df.isna().sum().sum()
            missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0

            # Date range analysis
            date_col, earliest, latest, date_range = self._analyze_date_range(df)

            # Numeric statistics
            numeric_stats = self._analyze_numeric_columns(df)

            # Validation
            validation_score = None
            validation_passed = None
            validation_errors = 0
            validation_warnings = 0

            if validate and self.validation_engine:
                report = self._validate_file(df, symbol, data_type)
                if report:
                    validation_score = report.quality_score
                    validation_passed = report.passed
                    validation_errors = len(report.get_errors())
                    validation_warnings = len(report.get_warnings())

            return FileStatistics(
                file_name=file_path.name,
                file_path=str(file_path),
                symbol=symbol,
                data_type=data_type,
                file_size_kb=round(file_size_kb, 2),
                num_rows=num_rows,
                num_columns=num_columns,
                column_names=column_names,
                total_cells=total_cells,
                missing_cells=missing_cells,
                missing_percentage=round(missing_percentage, 2),
                date_column=date_col,
                earliest_date=earliest,
                latest_date=latest,
                date_range_days=date_range,
                numeric_stats=numeric_stats,
                validation_score=validation_score,
                validation_passed=validation_passed,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings
            )

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def _analyze_date_range(
        self,
        df: pd.DataFrame
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        """
        Analyze date range in DataFrame.

        Returns:
            Tuple of (date_column, earliest_date, latest_date, date_range_days)
        """
        # Try to find date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]

        if not date_cols:
            return None, None, None, None

        date_col = date_cols[0]

        try:
            dates = pd.to_datetime(df[date_col])
            earliest = dates.min().strftime('%Y-%m-%d')
            latest = dates.max().strftime('%Y-%m-%d')
            date_range = (dates.max() - dates.min()).days

            return date_col, earliest, latest, date_range

        except Exception:
            return date_col, None, None, None

    def _analyze_numeric_columns(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze numeric columns for min/max/mean/std.

        Returns:
            Dictionary mapping column names to statistics
        """
        numeric_stats = {}

        # Focus on key columns
        key_columns = ['open', 'high', 'low', 'close', 'volume', 'price', 'strike', 'shares']

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in key_columns or df[col].dtype in [np.float64, np.int64]:
                try:
                    numeric_stats[col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()) if len(df) > 1 else 0.0
                    }
                except Exception:
                    continue

        return numeric_stats

    def _validate_file(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str
    ) -> Optional[ValidationReport]:
        """
        Run validation on a data file.

        Returns:
            ValidationReport or None
        """
        if not self.validation_engine:
            return None

        try:
            if data_type == 'daily':
                return self.validation_engine.validate_daily_data(df, symbol)
            elif data_type == 'weekly':
                return self.validation_engine.validate_weekly_data(df, symbol)
            elif data_type == 'fundamentals':
                return self.validation_engine.validate_fundamental_data(df, symbol)
            elif data_type == 'insider_transactions':
                return self.validation_engine.validate_insider_data(df, symbol)
            elif data_type == 'forex':
                return self.validation_engine.validate_forex_data(df, symbol)
            elif data_type == 'options':
                return self.validation_engine.validate_options_data(df, symbol)
            else:
                return None
        except Exception as e:
            print(f"Validation error for {symbol} ({data_type}): {e}")
            return None

    def _generate_summary(
        self,
        session_id: str,
        file_stats: List[FileStatistics]
    ) -> CollectionSummary:
        """
        Generate session summary from file statistics.

        Args:
            session_id: Session identifier
            file_stats: List of file statistics

        Returns:
            CollectionSummary
        """
        if not file_stats:
            return CollectionSummary(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                total_files=0,
                total_size_mb=0.0,
                total_rows=0,
                total_cells=0
            )

        # Aggregate metrics
        total_files = len(file_stats)
        total_size_mb = sum(f.file_size_kb for f in file_stats) / 1024
        total_rows = sum(f.num_rows for f in file_stats)
        total_cells = sum(f.total_cells for f in file_stats)
        total_missing = sum(f.missing_cells for f in file_stats)

        # By type
        files_by_type = {}
        rows_by_type = {}
        for stat in file_stats:
            files_by_type[stat.data_type] = files_by_type.get(stat.data_type, 0) + 1
            rows_by_type[stat.data_type] = rows_by_type.get(stat.data_type, 0) + stat.num_rows

        # Validation metrics
        files_with_validation = [f for f in file_stats if f.validation_score is not None]
        if files_with_validation:
            overall_quality = sum(f.validation_score for f in files_with_validation) / len(files_with_validation)
            files_passed = sum(1 for f in files_with_validation if f.validation_passed)
            files_failed = len(files_with_validation) - files_passed
            total_errors = sum(f.validation_errors for f in files_with_validation)
            total_warnings = sum(f.validation_warnings for f in files_with_validation)
        else:
            overall_quality = 100.0
            files_passed = 0
            files_failed = 0
            total_errors = 0
            total_warnings = 0

        # Date range
        dates_with_range = [(f.earliest_date, f.latest_date) for f in file_stats
                           if f.earliest_date and f.latest_date]
        if dates_with_range:
            earliest_date = min(d[0] for d in dates_with_range)
            latest_date = max(d[1] for d in dates_with_range)
        else:
            earliest_date = None
            latest_date = None

        return CollectionSummary(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            total_files=total_files,
            total_size_mb=round(total_size_mb, 2),
            total_rows=total_rows,
            total_cells=total_cells,
            files_by_type=files_by_type,
            rows_by_type=rows_by_type,
            overall_quality_score=round(overall_quality, 2),
            files_passed_validation=files_passed,
            files_failed_validation=files_failed,
            total_errors=total_errors,
            total_warnings=total_warnings,
            total_missing_cells=total_missing,
            overall_missing_percentage=round((total_missing / total_cells * 100) if total_cells > 0 else 0.0, 2),
            earliest_date=earliest_date,
            latest_date=latest_date
        )

    def save_report(
        self,
        summary: CollectionSummary,
        file_stats: List[FileStatistics],
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save report to JSON and generate human-readable summary.

        Args:
            summary: Collection summary
            file_stats: List of file statistics
            output_dir: Output directory (defaults to data_dir/reports)

        Returns:
            Path to generated report file
        """
        if output_dir is None:
            output_dir = self.data_dir / 'reports'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_data = {
            'summary': summary.to_dict(),
            'files': [f.to_dict() for f in file_stats]
        }

        json_path = output_dir / f"collection_report_{summary.session_id}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate human-readable report
        text_path = output_dir / f"collection_report_{summary.session_id}.txt"
        with open(text_path, 'w') as f:
            f.write(self._generate_text_report(summary, file_stats))

        return text_path

    def _generate_text_report(
        self,
        summary: CollectionSummary,
        file_stats: List[FileStatistics]
    ) -> str:
        """
        Generate human-readable text report.

        Args:
            summary: Collection summary
            file_stats: List of file statistics

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("DATA COLLECTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Session ID: {summary.session_id}")
        lines.append(f"Timestamp: {summary.timestamp}")
        lines.append("")

        # Summary
        lines.append("COLLECTION SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Files: {summary.total_files}")
        lines.append(f"Total Size: {summary.total_size_mb:.2f} MB")
        lines.append(f"Total Rows: {summary.total_rows:,}")
        lines.append(f"Total Data Points: {summary.total_cells:,}")
        lines.append("")

        # By data type
        if summary.files_by_type:
            lines.append("FILES BY DATA TYPE")
            lines.append("-" * 80)
            for data_type, count in sorted(summary.files_by_type.items()):
                rows = summary.rows_by_type.get(data_type, 0)
                lines.append(f"  {data_type:20s}: {count:4d} files, {rows:8,} rows")
            lines.append("")

        # Data quality
        lines.append("DATA QUALITY METRICS")
        lines.append("-" * 80)
        lines.append(f"Overall Quality Score: {summary.overall_quality_score:.1f}/100")
        lines.append(f"Files Passed Validation: {summary.files_passed_validation}")
        lines.append(f"Files Failed Validation: {summary.files_failed_validation}")
        lines.append(f"Total Validation Errors: {summary.total_errors}")
        lines.append(f"Total Validation Warnings: {summary.total_warnings}")
        lines.append("")

        # Missing data
        lines.append("MISSING DATA ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Total Missing Cells: {summary.total_missing_cells:,}")
        lines.append(f"Missing Data Percentage: {summary.overall_missing_percentage:.2f}%")
        lines.append("")

        # Date coverage
        if summary.earliest_date and summary.latest_date:
            lines.append("DATE COVERAGE")
            lines.append("-" * 80)
            lines.append(f"Earliest Date: {summary.earliest_date}")
            lines.append(f"Latest Date: {summary.latest_date}")
            lines.append("")

        # Individual file details
        lines.append("FILE DETAILS")
        lines.append("-" * 80)
        lines.append(f"{'Symbol':<10} {'Type':<15} {'Rows':>8} {'Cols':>5} {'Missing%':>10} {'Quality':>8}")
        lines.append("-" * 80)

        for stat in sorted(file_stats, key=lambda x: (x.data_type, x.symbol)):
            quality_str = f"{stat.validation_score:.1f}" if stat.validation_score is not None else "N/A"
            lines.append(
                f"{stat.symbol:<10} {stat.data_type:<15} {stat.num_rows:8,} "
                f"{stat.num_columns:5d} {stat.missing_percentage:9.2f}% {quality_str:>8}"
            )

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def print_summary(self, summary: CollectionSummary):
        """
        Print summary to console.

        Args:
            summary: Collection summary to print
        """
        print("\n" + "=" * 80)
        print("DATA COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Total Files: {summary.total_files}")
        print(f"Total Size: {summary.total_size_mb:.2f} MB")
        print(f"Total Rows: {summary.total_rows:,}")
        print(f"Overall Quality Score: {summary.overall_quality_score:.1f}/100")
        print(f"Missing Data: {summary.overall_missing_percentage:.2f}%")

        if summary.files_by_type:
            print("\nFiles by Type:")
            for data_type, count in sorted(summary.files_by_type.items()):
                print(f"  {data_type}: {count} files")

        print("=" * 80 + "\n")
