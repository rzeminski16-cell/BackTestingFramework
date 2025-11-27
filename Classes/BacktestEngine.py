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

class BacktestEngine:
    """
    Executes backtesting of a strategy on historical data
    """
    
    def __init__(self, 
                 strategy: BaseStrategy,
                 starting_capital: float = 100000,
                 commission_pct: float = 0.001,  # 0.1%
                 slippage_pct: float = 0.0005,   # 0.05%
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        Initialize backtest engine
        
        Args:
            strategy: Strategy to backtest
            starting_capital: Starting capital
            commission_pct: Commission as percentage of trade value
            slippage_pct: Slippage as percentage of price
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
        """
        self.strategy = strategy
        self.starting_capital = starting_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        self.trades_log = []
        self.equity_curve = {}  # symbol -> list of (date, equity)
    
    def _calculate_commission(self, value: float) -> float:
        """Calculate commission on trade value"""
        return value * self.commission_pct
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price"""
        if side == 'BUY':
            return price * (1 + self.slippage_pct)
        else:  # SELL
            return price * (1 - self.slippage_pct)
    
    def backtest_security(self, symbol: str, data: pd.DataFrame) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Backtest strategy on a single security
        
        Args:
            symbol: Security symbol
            data: Prepared historical data
        
        Returns:
            Tuple of (trades list, equity curve dataframe)
        """
        print(f"\nBacktesting {symbol}...")
        
        # Filter by date range
        data = data.copy()
        if self.start_date:
            data = data[data['date'] >= self.start_date]
        if self.end_date:
            data = data[data['date'] <= self.end_date]
        
        if len(data) == 0:
            print(f"No data for {symbol} in specified date range")
            return [], pd.DataFrame()
        
        # Initialize
        capital = self.starting_capital
        position = {
            'size': 0,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': None,
            'current_trade': None
        }
        
        trades = []
        equity_history = []
        
        # Iterate through data
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_row = current_data.iloc[-1]
            current_price = current_row['close']
            current_date = current_row['date']
            
            # Track equity
            position_value = position['size'] * current_price if position['size'] > 0 else 0
            current_equity = capital + position_value
            equity_history.append({
                'date': current_date,
                'equity': current_equity,
                'capital': capital,
                'position_value': position_value
            })
            
            # Generate signal
            signal = self.strategy.generate_signal(current_data, position)
            
            if signal is None:
                continue
            
            # Execute signal
            if signal.action == 'BUY' and position['size'] == 0:
                # Open position
                exec_price = self._apply_slippage(current_price, 'BUY')
                shares = self.strategy.position_size(capital, exec_price, signal)
                
                if shares > 0:
                    trade_value = shares * exec_price
                    commission = self._calculate_commission(trade_value)
                    total_cost = trade_value + commission
                    
                    if total_cost <= capital:
                        capital -= total_cost
                        position['size'] = shares
                        position['entry_price'] = exec_price
                        position['stop_loss'] = signal.stop_loss or 0
                        position['take_profit'] = signal.take_profit
                        
                        trade = Trade(
                            symbol=symbol,
                            entry_date=current_date,
                            entry_price=exec_price,
                            quantity=shares,
                            initial_stop_loss=signal.stop_loss or 0,
                            current_stop_loss=signal.stop_loss or 0,
                            take_profit=signal.take_profit,
                            commission_paid=commission
                        )
                        position['current_trade'] = trade
                        
                        print(f"  BUY {shares} @ ${exec_price:.2f} on {current_date.date()} - {signal.reason}")
            
            elif signal.action in ['SELL', 'CLOSE'] and position['size'] > 0:
                # Close position
                exec_price = self._apply_slippage(current_price, 'SELL')
                shares = position['size']
                trade_value = shares * exec_price
                commission = self._calculate_commission(trade_value)
                proceeds = trade_value - commission
                
                capital += proceeds
                
                # Complete trade record
                trade = position['current_trade']
                trade.exit_date = current_date
                trade.exit_price = exec_price
                trade.pl = proceeds - (shares * trade.entry_price + trade.commission_paid)
                trade.pl_pct = (exec_price / trade.entry_price - 1) * 100
                trade.duration_days = (current_date - trade.entry_date).days
                trade.exit_reason = signal.reason
                trade.commission_paid += commission
                
                trades.append(trade)
                
                print(f"  SELL {shares} @ ${exec_price:.2f} on {current_date.date()} - P/L: ${trade.pl:.2f} ({trade.pl_pct:.2f}%) - {signal.reason}")
                
                # Reset position
                position = {
                    'size': 0,
                    'entry_price': 0,
                    'stop_loss': 0,
                    'take_profit': None,
                    'current_trade': None
                }
        
        equity_df = pd.DataFrame(equity_history)
        
        return trades, equity_df
    
    def backtest_all_securities(self, data_folder: str) -> Dict[str, Tuple[List[Trade], pd.DataFrame]]:
        """
        Backtest strategy on all securities in folder
        
        Args:
            data_folder: Path to folder with prepared data
        
        Returns:
            Dict mapping symbol to (trades, equity_curve)
        """
        results = {}
        data_path = Path(data_folder)
        
        csv_files = list(data_path.glob("*.csv"))
        print(f"\nFound {len(csv_files)} securities to backtest")
        
        for csv_file in csv_files:
            if csv_file.name == "preparation_metadata.json":
                continue
            
            symbol = csv_file.stem
            data = pd.read_csv(csv_file)
            data['date'] = pd.to_datetime(data['date'])
            
            trades, equity_curve = self.backtest_security(symbol, data)
            results[symbol] = (trades, equity_curve)
        
        self.results = results
        return results