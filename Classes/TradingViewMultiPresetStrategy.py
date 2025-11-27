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

class TradingViewMultiPresetStrategy(BaseStrategy):
    """
    Replication of TradingView "Multi-Preset Strategy V2"
    
    This strategy combines multiple technical indicators with three preset configurations:
    - Aggressive: More trades, higher risk/reward
    - Balanced: Moderate approach
    - Conservative: Fewer trades, more selective
    
    Base Rules (always active):
    - New Highs: Close > highest high in past N candles
    - SMA 200: Close >= 200-period SMA
    
    Position sizing based on ATR and risk percentage
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            # Preset selection
            'preset': 'Balanced',  # Options: 'Aggressive', 'Balanced', 'Conservative'
            
            # Base rules
            'new_high_n': 14,
            'new_high_m': 0,
            'new_high_k': 10,
            'sma_200_length': 200,
            
            # Position sizing & risk
            'atr_length': 14,
            'atr_multiplier': 3.0,
            'risk_percent': 0.5,
            
            # CMF parameters
            'cmf_length': 14,
            'cmf_threshold': -0.2,
            
            # ROC parameters
            'roc_sma_length': 200,
            'roc_period': 14,
            'roc_threshold': -1.5,
            
            # RSI parameters
            'rsi_length': 14,
            'rsi_threshold': 60.0,
            
            # Volume parameters
            'vol_period': 30,
            
            # Bollinger Bands parameters
            'bb_length': 20,
            'bb_std_dev': 2.0,
            
            # MA Crossover parameters
            'ma_fast_length': 7,
            'ma_slow_length': 21,
            'ma_lookback_k': 14,
            
            # SAR parameters
            'sar_start': 0.02,
            'sar_increment': 0.02,
            'sar_maximum': 0.2,
            
            # SQZMOM parameters
            'sqz_length': 20,
            'sqz_mult': 2.0,
            'sqz_length_kc': 20,
            'sqz_mult_kc': 1.5,
            
            # EMA Sell parameters
            'ema_length': 14,
            'ema_sell_threshold': 3.0
        }
        
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
        # Validate preset
        if self.params['preset'] not in ['Aggressive', 'Balanced', 'Conservative']:
            raise ValueError("Preset must be 'Aggressive', 'Balanced', or 'Conservative'")
    
    def required_columns(self) -> List[str]:
        """
        All columns needed for this strategy
        """
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Add SMA 200
        cols.append(f"sma_{self.params['sma_200_length']}")
        
        # Add EMA for fast and slow MA crossover
        cols.append(f"ema_{self.params['ma_fast_length']}")
        cols.append(f"ema_{self.params['ma_slow_length']}")
        
        # Add EMA for sell signal
        cols.append(f"ema_{self.params['ema_length']}")
        
        # Add RSI
        cols.append(f"rsi_{self.params['rsi_length']}")
        
        # Add ATR
        cols.append(f"atr_{self.params['atr_length']}")
        
        return cols
    
    def generate_signal(self, data: pd.DataFrame, position: Dict) -> Optional[Signal]:
        """Generate trading signals based on preset configuration"""
        if len(data) < max(self.params['sma_200_length'], self.params['new_high_n']) + 10:
            return None
        
        current = data.iloc[-1]
        preset = self.params['preset']
        
        # ==== BASE RULES (Always required for BUY) ====
        
        # New Highs Rule
        try:
            lookback_start = max(1, len(data) - self.params['new_high_n'] - 1)
            lookback_end = len(data) - 1
            lookback_highs = data['high'].iloc[lookback_start:lookback_end]
            
            if len(lookback_highs) == 0:
                return None
                
            new_high_condition = current['close'] > lookback_highs.max()
            
            # Count new highs in last K candles
            new_highs_count = 0
            for i in range(1, min(self.params['new_high_k'] + 1, len(data))):
                past_idx_start = max(0, len(data) - i - self.params['new_high_n'] - 1)
                past_idx_end = len(data) - i
                
                if past_idx_start < past_idx_end:
                    past_highs = data['high'].iloc[past_idx_start:past_idx_end]
                    past_close = data['close'].iloc[past_idx_end - 1]
                    
                    if len(past_highs) > 0 and past_close > past_highs.max():
                        new_highs_count += 1
            
            new_highs_rule = new_high_condition and new_highs_count >= self.params['new_high_m']
        except Exception as e:
            print(f"Error in new highs calculation: {e}")
            new_highs_rule = False
        
        # SMA 200 Rule
        sma_200_col = f"sma_{self.params['sma_200_length']}"
        if sma_200_col not in data.columns or pd.isna(current[sma_200_col]):
            return None
        sma_200_rule = current['close'] >= current[sma_200_col]
        
        base_rules = new_highs_rule and sma_200_rule
        
        # ==== CHECK SELL SIGNALS (if in position) ====
        if position['size'] > 0:
            # Check stop loss first
            if current['close'] <= position['stop_loss']:
                return Signal(action='CLOSE', size=1.0, reason="Stop loss hit")
            
            # Check sell signals based on preset
            ema_col = f"ema_{self.params['ema_length']}"
            if ema_col in data.columns and not pd.isna(current[ema_col]):
                ema_value = current[ema_col]
                ema_sell_threshold_price = current['close'] * (1 + self.params['ema_sell_threshold'] / 100)
                
                sell_signal = False
                
                if preset == 'Aggressive':
                    sqzmom = self._calculate_sqzmom(data)
                    if sqzmom < 0:
                        sell_signal = True
                        return Signal(action='CLOSE', size=1.0, reason="SQZMOM sell signal")
                    if ema_value > ema_sell_threshold_price:
                        sell_signal = True
                        return Signal(action='CLOSE', size=1.0, reason="EMA sell signal")
                
                elif preset in ['Balanced', 'Conservative']:
                    if ema_value > ema_sell_threshold_price:
                        return Signal(action='CLOSE', size=1.0, reason="EMA sell signal")
        
        # ==== BUY SIGNAL (only if not in position and base rules met) ====
        if position['size'] == 0 and base_rules:
            
            additional_buy_rules = True
            
            if preset == 'Aggressive':
                # MA Crossover Rule
                ema_fast_col = f"ema_{self.params['ma_fast_length']}"
                ema_slow_col = f"ema_{self.params['ma_slow_length']}"
                
                if ema_fast_col not in data.columns or ema_slow_col not in data.columns:
                    return None
                
                ma_rule = False
                # Check for crossover in last K candles
                lookback_range = min(self.params['ma_lookback_k'] + 1, len(data) - 1)
                for i in range(lookback_range):
                    idx = -(i + 1)
                    prev_idx = idx - 1
                    
                    if abs(prev_idx) >= len(data):
                        break
                    
                    curr_fast = data[ema_fast_col].iloc[idx]
                    curr_slow = data[ema_slow_col].iloc[idx]
                    prev_fast = data[ema_fast_col].iloc[prev_idx]
                    prev_slow = data[ema_slow_col].iloc[prev_idx]
                    
                    if not any(pd.isna([curr_fast, curr_slow, prev_fast, prev_slow])):
                        if prev_fast <= prev_slow and curr_fast > curr_slow:
                            ma_rule = True
                            break
                
                additional_buy_rules = ma_rule
            
            elif preset == 'Balanced':
                # MA Crossover Rule
                ema_fast_col = f"ema_{self.params['ma_fast_length']}"
                ema_slow_col = f"ema_{self.params['ma_slow_length']}"
                
                if ema_fast_col not in data.columns or ema_slow_col not in data.columns:
                    return None
                
                ma_rule = False
                lookback_range = min(self.params['ma_lookback_k'] + 1, len(data) - 1)
                for i in range(lookback_range):
                    idx = -(i + 1)
                    prev_idx = idx - 1
                    
                    if abs(prev_idx) >= len(data):
                        break
                    
                    curr_fast = data[ema_fast_col].iloc[idx]
                    curr_slow = data[ema_slow_col].iloc[idx]
                    prev_fast = data[ema_fast_col].iloc[prev_idx]
                    prev_slow = data[ema_slow_col].iloc[prev_idx]
                    
                    if not any(pd.isna([curr_fast, curr_slow, prev_fast, prev_slow])):
                        if prev_fast <= prev_slow and curr_fast > curr_slow:
                            ma_rule = True
                            break
                
                # SAR Buy Rule - simplified (SAR below price = uptrend)
                sar_buy_rule = True  # Simplified - in real implementation calculate actual SAR
                
                additional_buy_rules = ma_rule and sar_buy_rule
            
            elif preset == 'Conservative':
                # SAR Buy Rule - simplified
                sar_buy_rule = True  # Simplified
                additional_buy_rules = sar_buy_rule
            
            # Generate BUY signal if all conditions met
            if additional_buy_rules:
                # Calculate stop loss based on ATR
                atr_col = f"atr_{self.params['atr_length']}"
                if atr_col not in data.columns or pd.isna(current[atr_col]):
                    return None
                
                stop_distance = current[atr_col] * self.params['atr_multiplier']
                stop_loss = current['close'] - stop_distance
                
                return Signal(
                    action='BUY',
                    size=self.params['risk_percent'] / 100,
                    stop_loss=stop_loss,
                    reason=f"{preset} preset buy signal"
                )
        
        return None
    
    def _calculate_sqzmom(self, data: pd.DataFrame) -> float:
        """Calculate Squeeze Momentum LazyBear - simplified version"""
        length = self.params['sqz_length']
        
        if len(data) < length + 10:
            return 0
        
        try:
            df = data.tail(length + 10).copy()
            
            # Calculate momentum using linear regression on price deviations
            highest_high = df['high'].rolling(window=self.params['sqz_length_kc']).max()
            lowest_low = df['low'].rolling(window=self.params['sqz_length_kc']).min()
            sma_close = df['close'].rolling(window=self.params['sqz_length_kc']).mean()
            
            val = df['close'] - ((highest_high + lowest_low) / 2 + sma_close) / 2
            
            # Get last length values
            val_clean = val.dropna()
            if len(val_clean) >= length:
                y = val_clean.tail(length).values
                x = np.arange(len(y))
                
                if len(y) == len(x) and len(y) > 0:
                    # Linear regression
                    slope, _ = np.polyfit(x, y, 1)
                    return slope
            
            return 0
        except Exception as e:
            print(f"Error calculating SQZMOM: {e}")
            return 0
    
    def position_size(self, capital: float, price: float, signal: Signal) -> float:
        """
        Calculate position size based on ATR and risk percentage
        Uses the formula: position_size = (equity * risk%) / stop_distance
        """
        if signal.stop_loss is None or signal.stop_loss <= 0:
            # Fallback to default sizing
            return super().position_size(capital, price, signal)
        
        stop_distance = price - signal.stop_loss
        if stop_distance <= 0:
            return 0
        
        max_loss = capital * self.params['risk_percent'] / 100
        shares = max_loss / stop_distance
        
        return max(int(shares), 0)  # Ensure non-negative