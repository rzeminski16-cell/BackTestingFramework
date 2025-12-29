"""
Output generation module for Factor Analysis.

Provides:
- Professional Excel report generation
- JSON payloads for GUI consumption
- Formatting utilities
"""

from .excel_generator import ExcelReportGenerator
from .json_generator import JsonPayloadGenerator
from .formatters import TableFormatter, ChartFormatter, ResultFormatter

__all__ = [
    'ExcelReportGenerator',
    'JsonPayloadGenerator',
    'TableFormatter',
    'ChartFormatter',
    'ResultFormatter',
]
