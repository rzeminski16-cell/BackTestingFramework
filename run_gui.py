#!/usr/bin/env python3
"""
Main entry point for the Backtesting Framework GUI.

Usage:
    python run_gui.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.gui import launch


if __name__ == "__main__":
    launch()
