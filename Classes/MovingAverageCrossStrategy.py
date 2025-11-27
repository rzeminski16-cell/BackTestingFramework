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

class MovingAverageCrossStrategy(BaseStrategy):
    """
    Example Strategy: Moving Average Crossover
    Buy when price crosses above MA, sell when crosses below
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'ma_period': 50,
            'position_size_pct': 0.1,  # 10% of capital per trade
            'stop_loss_pct': 0.05,  # 5% stop loss
            'take_profit_pct': 0.15  # 15% take profit
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def required_columns(self) -> List[str]:
        """
        This strategy needs:
        - date: for tracking
        - close: current price
        - sma_50: the 50-period simple moving average (or whatever ma_period is set to)
        """
        return ['date', 'close', f"sma_{self.params['ma_period']}"]
    
    def generate_signal(self, data: pd.DataFrame, position: Dict) -> Optional[Signal]:
        if len(data) < 2:
            return None
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        sma_col = f"sma_{self.params['ma_period']}"
        
        # Check if we have the required data
        if pd.isna(current[sma_col]) or pd.isna(previous[sma_col]):
            return None
        
        # If no position
        if position['size'] == 0:
            # Buy signal: price crosses above MA
            if previous['close'] <= previous[sma_col] and current['close'] > current[sma_col]:
                stop_loss = current['close'] * (1 - self.params['stop_loss_pct'])
                return Signal(
                    action='BUY',
                    size=self.params['position_size_pct'],
                    stop_loss=stop_loss,
                    take_profit=current['close'] * (1 + self.params['take_profit_pct']),
                    reason="Price crossed above MA"
                )
        else:
            # Check stop loss
            if current['close'] <= position['stop_loss']:
                return Signal(
                    action='CLOSE',
                    size=1.0,
                    reason="Stop loss hit"
                )
            
            # Check take profit
            if position.get('take_profit') and current['close'] >= position['take_profit']:
                return Signal(
                    action='CLOSE',
                    size=1.0,
                    reason="Take profit hit"
                )
            
            # Sell signal: price crosses below MA
            if previous['close'] >= previous[sma_col] and current['close'] < current[sma_col]:
                return Signal(
                    action='CLOSE',
                    size=1.0,
                    reason="Price crossed below MA"
                )
        
        return None