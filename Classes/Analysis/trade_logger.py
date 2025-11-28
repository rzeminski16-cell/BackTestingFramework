"""
Trade logging to CSV files.
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..Models.trade import Trade


class TradeLogger:
    """
    Logs trades to CSV files for detailed analysis.
    """

    def __init__(self, output_directory: Path):
        """
        Initialize trade logger.

        Args:
            output_directory: Directory to save trade logs
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def log_trades(self, symbol: str, strategy_name: str, trades: List[Trade],
                   strategy_params: Optional[Dict[str, Any]] = None) -> Path:
        """
        Log trades to CSV file.

        Args:
            symbol: Security symbol
            strategy_name: Strategy name
            trades: List of trades
            strategy_params: Optional dictionary of strategy parameters

        Returns:
            Path to created CSV file
        """
        if not trades:
            print(f"No trades to log for {symbol}")
            return None

        # Convert trades to list of dicts
        trade_dicts = [trade.to_dict() for trade in trades]

        # Create DataFrame
        df = pd.DataFrame(trade_dicts)

        # Create filename
        filename = f"{strategy_name}_{symbol}_trades.csv"
        filepath = self.output_directory / filename

        # Save to CSV
        df.to_csv(filepath, index=False)

        print(f"Logged {len(trades)} trades to {filepath}")

        # Save strategy parameters if provided
        if strategy_params:
            self.log_strategy_params(symbol, strategy_name, strategy_params)

        return filepath

    def log_strategy_params(self, symbol: str, strategy_name: str,
                           strategy_params: Dict[str, Any]) -> Path:
        """
        Log strategy parameters to JSON file.

        Args:
            symbol: Security symbol
            strategy_name: Strategy name
            strategy_params: Dictionary of strategy parameters

        Returns:
            Path to created JSON file
        """
        # Create filename
        filename = f"{strategy_name}_{symbol}_parameters.json"
        filepath = self.output_directory / filename

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(strategy_params, f, indent=2)

        print(f"Logged strategy parameters to {filepath}")
        return filepath

    def log_multiple(self, trades_dict: dict, strategy_name: str) -> dict:
        """
        Log trades for multiple securities.

        Args:
            trades_dict: Dictionary mapping symbol to trade list
            strategy_name: Strategy name

        Returns:
            Dictionary mapping symbol to filepath
        """
        filepaths = {}

        for symbol, trades in trades_dict.items():
            filepath = self.log_trades(symbol, strategy_name, trades)
            if filepath:
                filepaths[symbol] = filepath

        return filepaths
