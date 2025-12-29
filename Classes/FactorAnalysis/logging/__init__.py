"""
Logging module for Factor Analysis.

Provides comprehensive audit trail logging for reproducibility and debugging.
"""

from .audit_logger import AuditLogger, LogLevel

__all__ = ['AuditLogger', 'LogLevel']
