"""
Comprehensive Logging System for Data Collection.

Provides 7 specialized log types:
- Session log: High-level session information
- API log: All API calls, responses, errors
- Validation log: Data validation results
- Decisions log: User decisions made during collection
- Metrics log: Performance metrics (timing, counts)
- Errors log: All errors, exceptions, warnings
- Data issues log: Data quality issues

Also generates a human-readable summary and remembered answers JSON.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import threading


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class APILogEntry:
    """Entry for API activity log."""
    timestamp: datetime
    method: str
    endpoint: str
    symbol: Optional[str]
    status_code: Optional[int]
    response_size_bytes: Optional[int]
    cache_hit: bool
    duration_seconds: float
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class ValidationLogEntry:
    """Entry for validation log."""
    timestamp: datetime
    file_name: str
    check_name: str
    passed: bool
    message: str
    column: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class DecisionLogEntry:
    """Entry for decisions log."""
    timestamp: datetime
    decision_type: str
    question: str
    options: List[str]
    selected: str
    symbol: Optional[str] = None
    remembered: bool = False


@dataclass
class MetricsSnapshot:
    """Metrics snapshot for metrics log."""
    timestamp: datetime
    total_duration_seconds: float = 0.0
    data_collection_seconds: float = 0.0
    validation_seconds: float = 0.0
    file_io_seconds: float = 0.0
    total_api_calls: int = 0
    cached_responses: int = 0
    fresh_fetches: int = 0
    failed_calls: int = 0
    avg_response_time: float = 0.0
    retry_rate: float = 0.0
    calls_per_minute: float = 0.0
    total_rows_processed: int = 0
    total_cells_processed: int = 0
    validation_checks: int = 0
    issues_found: int = 0
    issues_resolved: int = 0
    data_quality_score: float = 100.0
    cache_hit_ratio: float = 0.0


@dataclass
class ErrorLogEntry:
    """Entry for errors log."""
    timestamp: datetime
    level: LogLevel
    error_type: str
    message: str
    symbol: Optional[str] = None
    endpoint: Optional[str] = None
    recovery_action: Optional[str] = None
    impact: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class DataIssueEntry:
    """Entry for data issues log."""
    timestamp: datetime
    symbol: str
    issue_type: str
    description: str
    severity: str  # info, warning, error
    resolution: Optional[str] = None
    field: Optional[str] = None
    affected_rows: int = 0


class SessionLogger:
    """
    Manages all logging for a single data collection session.

    Creates and manages 7 specialized log files plus a summary.
    """

    def __init__(self, logs_dir: Path, session_id: Optional[str] = None):
        """
        Initialize session logger.

        Args:
            logs_dir: Base directory for logs
            session_id: Optional session ID. If not provided, generates one.
        """
        self.logs_dir = Path(logs_dir)
        self.session_id = session_id or self._generate_session_id()
        self.session_dir = self.logs_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Start time for duration tracking
        self.start_time = datetime.now()

        # Thread lock for concurrent access
        self._lock = threading.Lock()

        # In-memory storage for entries
        self._api_entries: List[APILogEntry] = []
        self._validation_entries: List[ValidationLogEntry] = []
        self._decision_entries: List[DecisionLogEntry] = []
        self._error_entries: List[ErrorLogEntry] = []
        self._data_issue_entries: List[DataIssueEntry] = []
        self._metrics = MetricsSnapshot(timestamp=self.start_time)

        # Session metadata
        self.metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "data_types": [],
            "tickers": [],
            "configuration": {},
        }

        # Initialize log files
        self._init_log_files()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        now = datetime.now()
        short_uuid = uuid.uuid4().hex[:4]
        return f"{now.strftime('%Y%m%d_%H%M%S')}_{short_uuid}"

    def _init_log_files(self):
        """Initialize all log files with headers."""
        # Session log
        self._session_log_path = self.session_dir / f"{self.session_id}_session.log"
        self._write_log(self._session_log_path, f"[{self.start_time}] SESSION STARTED\n")
        self._write_log(self._session_log_path, f"  Session ID: {self.session_id}\n\n")

        # API log
        self._api_log_path = self.session_dir / f"{self.session_id}_api.log"
        self._write_log(self._api_log_path, "API ACTIVITY LOG\n" + "=" * 50 + "\n\n")

        # Validation log
        self._validation_log_path = self.session_dir / f"{self.session_id}_validation.log"
        self._write_log(self._validation_log_path, "VALIDATION LOG\n" + "=" * 50 + "\n\n")

        # Decisions log
        self._decisions_log_path = self.session_dir / f"{self.session_id}_decisions.log"
        self._write_log(self._decisions_log_path, "DECISIONS LOG\n" + "=" * 50 + "\n\n")

        # Metrics log
        self._metrics_log_path = self.session_dir / f"{self.session_id}_metrics.log"
        self._write_log(self._metrics_log_path, "SESSION METRICS\n" + "=" * 50 + "\n\n")

        # Errors log
        self._errors_log_path = self.session_dir / f"{self.session_id}_errors.log"
        self._write_log(self._errors_log_path, "ERRORS LOG\n" + "=" * 50 + "\n\n")

        # Data issues log
        self._data_issues_log_path = self.session_dir / f"{self.session_id}_data_issues.log"
        self._write_log(self._data_issues_log_path, "DATA QUALITY REPORT\n" + "=" * 50 + "\n\n")

        # Summary (text file)
        self._summary_path = self.session_dir / f"{self.session_id}_summary.txt"

        # Remembered answers JSON
        self._remembered_path = self.session_dir / f"{self.session_id}_remembered_answers.json"

    def _write_log(self, path: Path, content: str):
        """Thread-safe log writing."""
        with self._lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)

    # === Session Log Methods ===

    def log_session_info(self, message: str):
        """Log high-level session information."""
        timestamp = datetime.now()
        self._write_log(self._session_log_path, f"[{timestamp}] {message}\n")

    def log_configuration(self, config: Dict[str, Any]):
        """Log session configuration."""
        self.metadata["configuration"] = config
        self.log_session_info("CONFIGURATION:")
        for key, value in config.items():
            self.log_session_info(f"  {key}: {value}")

    def log_tickers(self, tickers: List[str]):
        """Log tickers being processed."""
        self.metadata["tickers"] = tickers
        self.log_session_info(f"TICKERS: {', '.join(tickers)}")

    def log_data_types(self, data_types: List[str]):
        """Log data types being collected."""
        self.metadata["data_types"] = data_types
        self.log_session_info(f"DATA TYPES: {', '.join(data_types)}")

    # === API Log Methods ===

    def log_api_call(self, entry: APILogEntry):
        """Log an API call."""
        self._api_entries.append(entry)

        cache_status = "HIT" if entry.cache_hit else "MISS"
        status_str = f"Status: {entry.status_code}" if entry.status_code else "Status: N/A"

        log_line = (
            f"[API] {entry.timestamp} | {entry.method} | {entry.endpoint}\n"
            f"      Symbol: {entry.symbol or 'N/A'}\n"
            f"      {status_str}\n"
            f"      Response size: {entry.response_size_bytes or 0} bytes\n"
            f"      Cache: {cache_status}\n"
            f"      Time: {entry.duration_seconds:.2f}s\n"
        )

        if entry.retry_count > 0:
            log_line += f"      Retry: {entry.retry_count}\n"
        if entry.error_message:
            log_line += f"      Error: {entry.error_message}\n"

        log_line += "\n"
        self._write_log(self._api_log_path, log_line)

    # === Validation Log Methods ===

    def log_validation(self, entry: ValidationLogEntry):
        """Log a validation check result."""
        self._validation_entries.append(entry)

        status = "[PASS]" if entry.passed else "[FAIL]"
        column_str = f" (column: {entry.column})" if entry.column else ""

        log_line = f"[VALIDATION] {entry.file_name}{column_str}\n"
        log_line += f"  {status} {entry.check_name}: {entry.message}\n"

        if entry.details:
            for key, value in entry.details.items():
                log_line += f"    {key}: {value}\n"

        log_line += "\n"
        self._write_log(self._validation_log_path, log_line)

    def log_validation_summary(self, file_name: str, passed: bool, total_checks: int,
                                failed_checks: int, warnings: int = 0):
        """Log validation summary for a file."""
        status = "PASS" if passed else "FAIL"
        summary = (
            f"[VALIDATION SUMMARY] {file_name}\n"
            f"  Status: {status}\n"
            f"  Total checks: {total_checks}\n"
            f"  Passed: {total_checks - failed_checks}\n"
            f"  Failed: {failed_checks}\n"
            f"  Warnings: {warnings}\n\n"
        )
        self._write_log(self._validation_log_path, summary)

    # === Decision Log Methods ===

    def log_decision(self, entry: DecisionLogEntry):
        """Log a user decision."""
        self._decision_entries.append(entry)

        symbol_str = f" | Symbol: {entry.symbol}" if entry.symbol else ""
        remember_str = " | Remembered: YES" if entry.remembered else ""

        log_line = (
            f"[{entry.timestamp}] DECISION | {entry.decision_type}{symbol_str}\n"
            f"  Question: {entry.question}\n"
            f"  Options: {entry.options}\n"
            f"  Selected: {entry.selected}{remember_str}\n\n"
        )
        self._write_log(self._decisions_log_path, log_line)

        # If remembered, save to JSON
        if entry.remembered:
            self._save_remembered_answer(entry)

    def _save_remembered_answer(self, entry: DecisionLogEntry):
        """Save remembered answer to JSON file."""
        remembered = {}
        if self._remembered_path.exists():
            with open(self._remembered_path, 'r') as f:
                remembered = json.load(f)

        key = entry.decision_type
        if entry.symbol:
            key = f"{entry.decision_type}_{entry.symbol}"

        remembered[key] = {
            "question": entry.question,
            "selected": entry.selected,
            "timestamp": entry.timestamp.isoformat(),
        }

        with open(self._remembered_path, 'w') as f:
            json.dump(remembered, f, indent=2)

    def get_remembered_answer(self, decision_type: str, symbol: Optional[str] = None) -> Optional[str]:
        """Get a previously remembered answer."""
        if not self._remembered_path.exists():
            return None

        with open(self._remembered_path, 'r') as f:
            remembered = json.load(f)

        key = f"{decision_type}_{symbol}" if symbol else decision_type
        if key in remembered:
            return remembered[key].get("selected")

        # Try without symbol if symbol-specific not found
        if symbol and decision_type in remembered:
            return remembered[decision_type].get("selected")

        return None

    # === Metrics Log Methods ===

    def update_metrics(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self._metrics, key):
                setattr(self._metrics, key, value)
        self._metrics.timestamp = datetime.now()

    def increment_metric(self, metric_name: str, amount: int = 1):
        """Increment a numeric metric."""
        if hasattr(self._metrics, metric_name):
            current = getattr(self._metrics, metric_name)
            setattr(self._metrics, metric_name, current + amount)

    def log_metrics(self):
        """Write current metrics to log."""
        m = self._metrics

        # Calculate derived metrics
        if m.total_api_calls > 0:
            m.cache_hit_ratio = (m.cached_responses / m.total_api_calls) * 100
            m.retry_rate = (m.failed_calls / m.total_api_calls) * 100

        duration = (datetime.now() - self.start_time).total_seconds()
        m.total_duration_seconds = duration

        if duration > 0:
            m.calls_per_minute = (m.total_api_calls / duration) * 60

        metrics_text = f"""
Timing:
  Total duration: {m.total_duration_seconds:.1f}s
  Data collection: {m.data_collection_seconds:.1f}s
  Validation: {m.validation_seconds:.1f}s
  File I/O: {m.file_io_seconds:.1f}s

API Performance:
  Total calls: {m.total_api_calls}
  Cached responses: {m.cached_responses} ({m.cache_hit_ratio:.1f}%)
  Fresh fetches: {m.fresh_fetches}
  Failed calls (retried): {m.failed_calls}
  Average response time: {m.avg_response_time:.2f}s
  Retry rate: {m.retry_rate:.1f}%
  Calls per minute: {m.calls_per_minute:.1f}

Data Processing:
  Total rows processed: {m.total_rows_processed}
  Total cells processed: {m.total_cells_processed}
  Validation checks: {m.validation_checks}
  Issues found: {m.issues_found}
  Issues resolved: {m.issues_resolved}
  Data quality score: {m.data_quality_score:.1f}%
"""
        self._write_log(self._metrics_log_path, metrics_text)

    # === Error Log Methods ===

    def log_error(self, entry: ErrorLogEntry):
        """Log an error or warning."""
        self._error_entries.append(entry)

        symbol_str = f" | Symbol: {entry.symbol}" if entry.symbol else ""
        endpoint_str = f" | Endpoint: {entry.endpoint}" if entry.endpoint else ""

        log_line = (
            f"[{entry.level.value}] {entry.timestamp} | {entry.error_type}{symbol_str}{endpoint_str}\n"
            f"  Message: {entry.message}\n"
        )

        if entry.recovery_action:
            log_line += f"  Recovery: {entry.recovery_action}\n"
        if entry.impact:
            log_line += f"  Impact: {entry.impact}\n"
        if entry.stack_trace:
            log_line += f"  Stack trace:\n{entry.stack_trace}\n"

        log_line += "\n"
        self._write_log(self._errors_log_path, log_line)

    def log_warning(self, error_type: str, message: str, **kwargs):
        """Convenience method to log a warning."""
        entry = ErrorLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            error_type=error_type,
            message=message,
            **kwargs
        )
        self.log_error(entry)

    def log_exception(self, error_type: str, message: str, exception: Exception, **kwargs):
        """Log an exception with stack trace."""
        import traceback
        entry = ErrorLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            error_type=error_type,
            message=message,
            stack_trace=traceback.format_exc(),
            **kwargs
        )
        self.log_error(entry)

    # === Data Issues Log Methods ===

    def log_data_issue(self, entry: DataIssueEntry):
        """Log a data quality issue."""
        self._data_issue_entries.append(entry)
        self.increment_metric("issues_found")

        severity_symbol = {"info": "i", "warning": "!", "error": "X"}
        sym = severity_symbol.get(entry.severity, "?")

        log_line = (
            f"[{sym}] {entry.symbol}: {entry.issue_type}\n"
            f"    {entry.description}\n"
        )

        if entry.field:
            log_line += f"    Field: {entry.field}\n"
        if entry.affected_rows > 0:
            log_line += f"    Affected rows: {entry.affected_rows}\n"
        if entry.resolution:
            log_line += f"    Resolution: {entry.resolution}\n"
            self.increment_metric("issues_resolved")

        log_line += "\n"
        self._write_log(self._data_issues_log_path, log_line)

    # === Summary Methods ===

    def generate_summary(self) -> str:
        """Generate human-readable session summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        # Count by severity
        errors = sum(1 for e in self._error_entries if e.level == LogLevel.ERROR)
        warnings = sum(1 for e in self._error_entries if e.level == LogLevel.WARNING)

        # Calculate success rate
        total_api = len(self._api_entries)
        successful_api = sum(1 for e in self._api_entries if e.status_code and 200 <= e.status_code < 300)
        api_success_rate = (successful_api / total_api * 100) if total_api > 0 else 100

        # Validation pass rate
        total_val = len(self._validation_entries)
        passed_val = sum(1 for e in self._validation_entries if e.passed)
        val_pass_rate = (passed_val / total_val * 100) if total_val > 0 else 100

        summary = f"""
================================================================================
DATA COLLECTION SESSION SUMMARY
================================================================================

Session ID: {self.session_id}
Start Time: {self.start_time}
End Time:   {end_time}
Duration:   {duration}

SCOPE
-----
Data Types: {', '.join(self.metadata.get('data_types', ['N/A']))}
Tickers:    {len(self.metadata.get('tickers', []))} symbols

API ACTIVITY
------------
Total API Calls:     {total_api}
Successful:          {successful_api} ({api_success_rate:.1f}%)
Cache Hits:          {sum(1 for e in self._api_entries if e.cache_hit)}
Retries:             {sum(e.retry_count for e in self._api_entries)}

VALIDATION
----------
Total Checks:        {total_val}
Passed:              {passed_val} ({val_pass_rate:.1f}%)
Failed:              {total_val - passed_val}

DATA QUALITY
------------
Issues Found:        {len(self._data_issue_entries)}
Issues Resolved:     {sum(1 for e in self._data_issue_entries if e.resolution)}
Quality Score:       {self._metrics.data_quality_score:.1f}%

DECISIONS
---------
Decisions Made:      {len(self._decision_entries)}
Remembered:          {sum(1 for e in self._decision_entries if e.remembered)}

ISSUES
------
Errors:              {errors}
Warnings:            {warnings}

FILES CREATED
-------------
(See session log for details)

================================================================================
"""
        return summary

    def finalize(self, exit_code: int = 0):
        """Finalize logging, write summary, and close session."""
        end_time = datetime.now()

        # Log session end
        self.log_session_info(f"SESSION COMPLETED")
        self.log_session_info(f"  Exit code: {exit_code}")
        self.log_session_info(f"  End time: {end_time}")
        self.log_session_info(f"  Duration: {end_time - self.start_time}")

        # Write final metrics
        self.log_metrics()

        # Write summary
        summary = self.generate_summary()
        with open(self._summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        return summary


class LoggingManager:
    """
    Factory and manager for session loggers.

    Provides easy access to logging across the application.
    """

    def __init__(self, logs_dir: Path):
        """
        Initialize logging manager.

        Args:
            logs_dir: Base directory for all logs
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: Optional[SessionLogger] = None

    def create_session(self, session_id: Optional[str] = None) -> SessionLogger:
        """Create a new logging session."""
        self._current_session = SessionLogger(self.logs_dir, session_id)
        return self._current_session

    @property
    def current_session(self) -> Optional[SessionLogger]:
        """Get the current active session."""
        return self._current_session

    def get_session_logs(self) -> List[Path]:
        """Get all session log directories."""
        return sorted([d for d in self.logs_dir.iterdir() if d.is_dir()], reverse=True)

    def get_session_by_id(self, session_id: str) -> Optional[Path]:
        """Get session directory by ID."""
        session_dir = self.logs_dir / session_id
        return session_dir if session_dir.exists() else None

    def cleanup_old_sessions(self, keep_days: int = 30):
        """Remove sessions older than specified days."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=keep_days)

        for session_dir in self.get_session_logs():
            try:
                # Parse date from session ID (format: YYYYMMDD_HHMMSS_xxxx)
                date_str = session_dir.name[:15]
                session_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")

                if session_date < cutoff:
                    import shutil
                    shutil.rmtree(session_dir)
            except (ValueError, OSError):
                continue
