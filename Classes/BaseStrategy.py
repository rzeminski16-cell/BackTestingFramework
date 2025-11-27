import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Excel reporting imports
try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.chart import LineChart, BarChart, Reference, PieChart
    from openpyxl.chart.label import DataLabelList
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: openpyxl not installed. Excel reporting disabled. Install with: pip install openpyxl")

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    Subclass this to create custom strategies.
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize strategy with parameters
        
        Args:
            params: Dictionary of strategy-specific parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def required_columns(self) -> List[str]:
        """
        Return list of required column names for this strategy.
        These are the column names that the strategy expects in the data.
        Must include at least ['date', 'close']
        
        Returns:
            List of column names that will be used in generate_signal()
        
        Example:
            return ['date', 'close', 'sma_50', 'rsi_14']
        """
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, position: Dict) -> Optional[Signal]:
        """
        Generate trading signal based on current data and position
        
        Args:
            data: Historical data up to current point (row -1 is current)
            position: Current position info {'size': 0, 'entry_price': 0, 'stop_loss': 0}
        
        Returns:
            Signal object or None
        """
        pass
    
    def position_size(self, capital: float, price: float, signal: Signal) -> float:
        """
        Calculate position size based on available capital
        
        Args:
            capital: Available capital
            price: Current price
            signal: The generated signal
        
        Returns:
            Number of shares to trade
        """
        # Default: use signal.size as percentage of capital
        amount = capital * signal.size
        shares = amount / price
        return int(shares)  # Round down to whole shares