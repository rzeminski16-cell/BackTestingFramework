---
tags:
  - template
  - strategy
---

# New Strategy Template

Copy this template when creating a new strategy. See [[Adding a New Strategy]] for the full guide.

---

```python
"""
[Strategy Name]

[Brief description of the strategy logic.]

Entry: [Entry conditions]
Exit: [Exit conditions]
Stop Loss: [Stop loss calculation]
Position Sizing: [Position sizing method]

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - [indicator_column]: [Description]

OPTIMIZABLE PARAMETERS:
    - [param_name]: [Description] (default: [value])
"""
from typing import List, Optional
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class MyNewStrategy(BaseStrategy):
    """
    [Strategy description]
    """

    def __init__(self, atr_multiplier: float = 2.0, risk_percent: float = 2.0):
        super().__init__(
            atr_multiplier=atr_multiplier,
            risk_percent=risk_percent
        )
        self.atr_multiplier = atr_multiplier
        self.risk_percent = risk_percent

    @property
    def trade_direction(self) -> TradeDirection:
        """LONG or SHORT."""
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        """Columns that MUST exist in the CSV data."""
        return [
            'date', 'close',
            # Add your required indicator columns here
            'atr_14_atr',
        ]

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Return Signal.buy() when entry conditions are met, or None."""
        # TODO: Implement entry logic
        #
        # Example:
        # if entry_condition:
        #     return Signal.buy(
        #         size=1.0,
        #         stop_loss=self.calculate_initial_stop_loss(context),
        #         reason="Entry reason",
        #         direction=self.trade_direction
        #     )
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """Return the stop loss price."""
        atr = context.get_indicator_value('atr_14_atr')
        if atr and atr > 0:
            return context.current_price - (atr * self.atr_multiplier)
        return context.current_price * 0.95

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """Return number of shares to buy."""
        equity = context.total_equity
        risk_amount = equity * (self.risk_percent / 100)
        stop_distance = context.current_price - signal.stop_loss

        if stop_distance <= 0:
            return (context.available_capital * 0.1) / context.current_price

        return risk_amount / (stop_distance * context.fx_rate)

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Return Signal.sell() when exit conditions are met, or None."""
        # TODO: Implement exit logic
        #
        # Example:
        # if exit_condition:
        #     return Signal.sell(reason="Exit reason")
        return None

    # --- OPTIONAL: Uncomment and implement if needed ---

    # def _prepare_data_impl(self, data):
    #     """Pre-calculate custom indicators (runs once before backtest)."""
    #     return data

    # def should_adjust_stop(self, context):
    #     """Return new stop price for trailing stop, or None."""
    #     return None

    # def should_partial_exit(self, context):
    #     """Return fraction (0-1) to exit, or None."""
    #     return None

    # def should_pyramid(self, context):
    #     """Return Signal.pyramid() to add to position, or None. Max 1 per trade."""
    #     return None
```
