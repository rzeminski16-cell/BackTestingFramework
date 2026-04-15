"""
AlphaTrendV1 Strategy

AlphaTrendV1 is a trend-following long-only strategy that captures breakouts once
price action confirms a new trend. The entry signal is produced by the
``AlphaMACross`` indicator, which compares a moving average (MA) against a
version of itself that is offset by ``ma_offset`` bars. When the MA crosses
above its offset copy a long entry is triggered.

Because this is the base version of the system, only long signals are taken and
there are only two exit mechanisms:
    - Stop Loss: placed at ``atr_sl`` multiples of the 14-day ATR below the
      entry price and held static for the life of the trade.
    - Time Exit: the position is closed after ``time_exit`` days regardless of
      P/L.

Entry: MA crosses above MA offset by ``ma_offset`` bars.
Exit: Time-based exit after ``time_exit`` days OR stop loss hit.
Stop Loss: ``entry_price - atr_sl * ATR_14`` (static).
Position Sizing: Risk-based. ``risk_perc`` of total equity is risked at the
    stop loss distance.

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - atr_14_atr: 14-period ATR used for stop loss calculation.

OPTIMIZABLE PARAMETERS:
    - ma_offset: Number of bars to offset the comparison MA (default: 2).
    - ema_sma: Moving-average type. Either ``"EMA"`` or ``"SMA"`` (default:
      ``"EMA"``).
    - ma_length: Length of the moving average (default: 20).
    - atr_sl: Multiplier on the 14-day ATR used to compute the initial stop
      loss distance (default: 2.0).
    - time_exit: Number of days to hold before a forced time-based exit
      (default: 30).
    - risk_perc: Percent of total equity to risk per trade at the initial stop
      loss (default: 2.0).
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class AlphaTrendV1Strategy(BaseStrategy):
    """
    AlphaTrendV1 - long-only trend-following strategy using the AlphaMACross
    indicator for entries, an ATR-based stop loss and a time-based exit.
    """

    # Column names used for the computed MA series.
    _MA_COL = "alpha_ma"
    _MA_OFFSET_COL = "alpha_ma_offset"

    def __init__(self,
                 # Indicator parameters
                 ma_offset: int = 2,
                 ema_sma: str = "EMA",
                 ma_length: int = 20,
                 # Strategy parameters
                 atr_sl: float = 2.0,
                 time_exit: int = 30,
                 risk_perc: float = 2.0):
        """Initialise the AlphaTrendV1 strategy.

        Args:
            ma_offset: Number of bars to offset the second MA by.
            ema_sma: Moving average type. ``"EMA"`` or ``"SMA"`` (case
                insensitive).
            ma_length: Window length for the moving average.
            atr_sl: Multiplier applied to the 14-day ATR to get the SL
                distance.
            time_exit: Maximum holding period in days before forced exit.
            risk_perc: Percent of total equity to risk per trade.
        """
        # Store parameters BEFORE super().__init__() which calls
        # _validate_parameters().
        self.ma_offset = int(ma_offset)
        self.ema_sma = str(ema_sma).strip().upper()
        self.ma_length = int(ma_length)
        self.atr_sl = float(atr_sl)
        self.time_exit = int(time_exit)
        self.risk_perc = float(risk_perc)

        super().__init__(
            ma_offset=self.ma_offset,
            ema_sma=self.ema_sma,
            ma_length=self.ma_length,
            atr_sl=self.atr_sl,
            time_exit=self.time_exit,
            risk_perc=self.risk_perc,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.ma_offset < 1:
            raise ValueError(
                f"ma_offset must be >= 1, got {self.ma_offset}"
            )

        if self.ema_sma not in {"EMA", "SMA"}:
            raise ValueError(
                f"ema_sma must be 'EMA' or 'SMA', got {self.ema_sma!r}"
            )

        if self.ma_length < 2:
            raise ValueError(
                f"ma_length must be >= 2, got {self.ma_length}"
            )

        if self.atr_sl <= 0:
            raise ValueError(
                f"atr_sl must be > 0, got {self.atr_sl}"
            )

        if self.time_exit < 1:
            raise ValueError(
                f"time_exit must be >= 1 day, got {self.time_exit}"
            )

        if not 0 < self.risk_perc <= 100:
            raise ValueError(
                f"risk_perc must be between 0 and 100, got {self.risk_perc}"
            )

    # ------------------------------------------------------------------
    # Basic strategy metadata
    # ------------------------------------------------------------------
    @property
    def trade_direction(self) -> TradeDirection:
        """Long-only base version."""
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        """Columns that must exist in the raw CSV data."""
        return ["date", "close", "atr_14_atr"]

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------
    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-calculate the strategy-specific AlphaMACross indicator.

        Adds two causal columns to the dataframe:
            * ``alpha_ma``: the MA of ``close`` using the configured type and
              length.
            * ``alpha_ma_offset``: the same MA shifted forward by
              ``ma_offset`` bars (i.e. the MA value ``ma_offset`` bars ago).

        It also normalises the ATR column name so the rest of the strategy can
        use ``atr_14``.
        """
        df = data.copy()

        # Normalise ATR column so internal references stay short.
        df["atr_14"] = df["atr_14_atr"]

        close = df["close"]
        if self.ema_sma == "EMA":
            ma_series = close.ewm(span=self.ma_length, adjust=False).mean()
        else:  # SMA
            ma_series = close.rolling(window=self.ma_length, min_periods=self.ma_length).mean()

        df[self._MA_COL] = ma_series
        # shift(n>=0) is causal - pushes previously-known MA values forward.
        df[self._MA_OFFSET_COL] = ma_series.shift(self.ma_offset)

        return df

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------
    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Return a BUY signal when the AlphaMACross indicator fires long."""
        # We need at least two bars of MA / MA-offset history to detect a
        # cross-over. Minimum index required for both values to be available
        # is ``ma_length + ma_offset`` (so that the shifted MA at t-1 is
        # defined too).
        min_index = self.ma_length + self.ma_offset
        if context.current_index < min_index:
            return None

        ma_now = context.get_indicator_value(self._MA_COL, offset=0)
        ma_prev = context.get_indicator_value(self._MA_COL, offset=-1)
        ma_off_now = context.get_indicator_value(self._MA_OFFSET_COL, offset=0)
        ma_off_prev = context.get_indicator_value(self._MA_OFFSET_COL, offset=-1)

        # Guard against missing indicator values (warmup / NaN).
        for value in (ma_now, ma_prev, ma_off_now, ma_off_prev):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None

        # Crossover: previous bar MA <= MA_offset AND current bar MA > MA_offset.
        is_long_signal = (ma_prev <= ma_off_prev) and (ma_now > ma_off_now)

        if not is_long_signal:
            return None

        stop_loss = self.calculate_initial_stop_loss(context)

        return Signal.buy(
            size=1.0,
            stop_loss=stop_loss,
            reason=(
                f"AlphaMACross long ({self.ema_sma} {self.ma_length}"
                f" vs offset {self.ma_offset})"
            ),
            direction=self.trade_direction,
        )

    # ------------------------------------------------------------------
    # Stop loss
    # ------------------------------------------------------------------
    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """Return ``entry_price - atr_sl * ATR_14``.

        Falls back to a 5% stop loss if ATR is unavailable.
        """
        current_price = context.current_price
        atr = context.get_indicator_value("atr_14")

        if atr is not None and not (isinstance(atr, float) and np.isnan(atr)) and atr > 0:
            return current_price - (self.atr_sl * atr)

        # Fallback - should almost never happen because atr_14 is required.
        return current_price * 0.95

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """Risk-based sizing.

        positionInBase = (risk_perc * total_equity) / stop_distance_base

        where ``stop_distance_base`` is the per-share loss (price - SL)
        converted to the base currency via ``fx_rate``. This guarantees that
        if the stop loss is hit, exactly ``risk_perc`` percent of equity is
        lost before slippage and commission.
        """
        if signal.stop_loss is None:
            # Defensive fallback - entry signals always have SL set.
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        equity = context.total_equity
        risk_amount = equity * (self.risk_perc / 100.0)

        stop_distance = context.current_price - signal.stop_loss
        if stop_distance <= 0:
            # Invalid stop - fall back to using 10% of capital.
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        stop_distance_base = stop_distance * context.fx_rate
        return risk_amount / stop_distance_base

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Time-based exit after ``time_exit`` days.

        The stop loss is handled by the engine using the price set by
        ``calculate_initial_stop_loss``.
        """
        if not context.has_position:
            return None

        days_in_position = context.position.duration_days(context.current_date)
        if days_in_position >= self.time_exit:
            return Signal.sell(
                reason=f"Time exit ({self.time_exit} days reached)"
            )

        return None
