"""
AlphaTrendV2 Strategy

AlphaTrendV2 is a trend-following long-only strategy identical to
``AlphaTrendV1Strategy`` on entry, stop loss and position sizing. The only
difference is the exit logic: the V1 time-based exit is replaced with a
combined-indicator exit consisting of up to four independently toggleable
sub-rules. When more than one sub-rule is enabled, ALL enabled sub-rules must
simultaneously signal an exit on the current bar before the position is
closed. If no sub-rule is enabled the position exits only on stop loss.

Entry: MA crosses above MA offset by ``ma_offset`` bars (AlphaMACross).
Exit: Combined-indicator exit. For each ENABLED rule the exit fires when:
    - ADX rule:  ``adx_14_adx`` < ``adx_threshold``
    - ROC rule:  ((close - close[n]) / close[n]) * 100 < ``roc_threshold``
                 (n = ``roc_period``; calculated at runtime)
    - MACD rule: ``macd_14_macd_signal`` < 0
    - MFI rule:  ``mfi_14_mfi`` < 50 for at least ``mfi_consecutive_days``
                 consecutive bars including the current one
  A full exit is generated only when EVERY enabled rule is currently in its
  exit state. Disabled rules are ignored.
Stop Loss: ``entry_price - atr_sl * ATR_14`` (static, same as V1).
Position Sizing: Risk-based. ``risk_perc`` of total equity is risked at the
    stop loss distance.

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - atr_14_atr: 14-period ATR used for stop loss calculation.
    - ema_{N}_ema / sma_{N}_sma: Pre-calculated moving average from raw data,
      where ``N`` must match one of the supported ``ma_length`` values.
    - adx_14_adx: ADX 14, required only when ``adx_exit_enabled`` is True.
    - macd_14_macd_signal: MACD signal line, required only when
      ``macd_exit_enabled`` is True.
    - mfi_14_mfi: Money Flow Index 14, required only when
      ``mfi_exit_enabled`` is True.

OPTIMIZABLE PARAMETERS:
    - ma_offset: Number of bars to offset the comparison MA (default: 2).
    - ema_sma: Moving-average type. Either ``"EMA"`` or ``"SMA"`` (default:
      ``"EMA"``).
    - ma_length: Length of the moving average. Must be one of the supported
      values: 7, 14, 20, 30, 50, 90, 200 (default: 20).
    - atr_sl: Multiplier on the 14-day ATR used for the initial stop loss
      (default: 2.0).
    - risk_perc: Percent of total equity to risk per trade at the initial
      stop loss (default: 2.0).
    - adx_exit_enabled: Toggle ADX exit sub-rule (default: True).
    - adx_threshold: ADX level below which the ADX sub-rule exits
      (default: 20.0).
    - roc_exit_enabled: Toggle ROC exit sub-rule (default: True).
    - roc_period: Lookback in bars for ROC calculation (default: 10).
    - roc_threshold: ROC % level below which the ROC sub-rule exits
      (default: 0.0).
    - macd_exit_enabled: Toggle MACD exit sub-rule (default: True).
    - mfi_exit_enabled: Toggle MFI exit sub-rule (default: True).
    - mfi_consecutive_days: Number of consecutive bars (including current)
      during which MFI must be below 50 for the MFI sub-rule to exit
      (default: 3).
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class AlphaTrendV2Strategy(BaseStrategy):
    """
    AlphaTrendV2 - long-only trend-following strategy using the AlphaMACross
    indicator for entries, an ATR-based stop loss, and a combined
    multi-indicator exit (ADX + ROC + MACD + MFI, each independently
    toggleable).

    Moving averages are read directly from pre-calculated raw-data columns
    (e.g. ``ema_20_ema`` or ``sma_50_sma``) to keep backtests deterministic
    and fast.
    """

    SUPPORTED_MA_LENGTHS = (7, 14, 20, 30, 50, 90, 200)

    MFI_EXIT_LEVEL = 50.0
    MACD_EXIT_LEVEL = 0.0

    def __init__(self,
                 # Indicator parameters
                 ma_offset: int = 2,
                 ema_sma: str = "EMA",
                 ma_length: int = 20,
                 # Entry / SL / sizing (same as V1)
                 atr_sl: float = 2.0,
                 risk_perc: float = 2.0,
                 # Combined exit sub-rule toggles + parameters
                 adx_exit_enabled: bool = True,
                 adx_threshold: float = 20.0,
                 roc_exit_enabled: bool = True,
                 roc_period: int = 10,
                 roc_threshold: float = 0.0,
                 macd_exit_enabled: bool = True,
                 mfi_exit_enabled: bool = True,
                 mfi_consecutive_days: int = 3):
        """Initialise the AlphaTrendV2 strategy.

        Args:
            ma_offset: Number of bars to offset the second MA by.
            ema_sma: Moving average type, ``"EMA"`` or ``"SMA"``.
            ma_length: Moving average length. Must be in
                ``SUPPORTED_MA_LENGTHS``.
            atr_sl: Multiplier on ATR(14) for the initial stop loss.
            risk_perc: Percent of total equity risked per trade.
            adx_exit_enabled: Enable the ADX exit sub-rule.
            adx_threshold: ADX < ``adx_threshold`` triggers the ADX sub-rule.
            roc_exit_enabled: Enable the ROC exit sub-rule.
            roc_period: Lookback period ``n`` for the ROC calculation.
            roc_threshold: ROC % < ``roc_threshold`` triggers the ROC sub-rule.
            macd_exit_enabled: Enable the MACD signal-line exit sub-rule.
            mfi_exit_enabled: Enable the MFI exit sub-rule.
            mfi_consecutive_days: Number of consecutive bars (including the
                current bar) during which MFI must stay below 50.
        """
        # Store all parameters BEFORE super().__init__ which calls
        # _validate_parameters().
        self.ma_offset = int(ma_offset)
        self.ema_sma = str(ema_sma).strip().upper()
        self.ma_length = int(ma_length)
        self.atr_sl = float(atr_sl)
        self.risk_perc = float(risk_perc)

        self.adx_exit_enabled = bool(adx_exit_enabled)
        self.adx_threshold = float(adx_threshold)
        self.roc_exit_enabled = bool(roc_exit_enabled)
        self.roc_period = int(roc_period)
        self.roc_threshold = float(roc_threshold)
        self.macd_exit_enabled = bool(macd_exit_enabled)
        self.mfi_exit_enabled = bool(mfi_exit_enabled)
        self.mfi_consecutive_days = int(mfi_consecutive_days)

        self._ma_column = self._resolve_ma_column(self.ema_sma, self.ma_length)
        self._roc_column = f"alpha_roc_{self.roc_period}"

        super().__init__(
            ma_offset=self.ma_offset,
            ema_sma=self.ema_sma,
            ma_length=self.ma_length,
            atr_sl=self.atr_sl,
            risk_perc=self.risk_perc,
            adx_exit_enabled=self.adx_exit_enabled,
            adx_threshold=self.adx_threshold,
            roc_exit_enabled=self.roc_exit_enabled,
            roc_period=self.roc_period,
            roc_threshold=self.roc_threshold,
            macd_exit_enabled=self.macd_exit_enabled,
            mfi_exit_enabled=self.mfi_exit_enabled,
            mfi_consecutive_days=self.mfi_consecutive_days,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_ma_column(ema_sma: str, ma_length: int) -> str:
        """Map (type, length) to the expected raw-data column name."""
        ema_sma = ema_sma.upper()
        if ema_sma == "EMA":
            return f"ema_{int(ma_length)}_ema"
        if ema_sma == "SMA":
            return f"sma_{int(ma_length)}_sma"
        return f"{ema_sma.lower()}_{int(ma_length)}"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.ma_offset < 1:
            raise ValueError(f"ma_offset must be >= 1, got {self.ma_offset}")

        if self.ema_sma not in {"EMA", "SMA"}:
            raise ValueError(
                f"ema_sma must be 'EMA' or 'SMA', got {self.ema_sma!r}"
            )

        if self.ma_length not in self.SUPPORTED_MA_LENGTHS:
            raise ValueError(
                f"ma_length must be one of {list(self.SUPPORTED_MA_LENGTHS)}, "
                f"got {self.ma_length}. The strategy reads MAs directly from "
                f"raw data so only pre-calculated lengths are supported."
            )

        if self.atr_sl <= 0:
            raise ValueError(f"atr_sl must be > 0, got {self.atr_sl}")

        if not 0 < self.risk_perc <= 100:
            raise ValueError(
                f"risk_perc must be between 0 and 100, got {self.risk_perc}"
            )

        if self.roc_period < 1:
            raise ValueError(
                f"roc_period must be >= 1, got {self.roc_period}"
            )

        if self.mfi_consecutive_days < 1:
            raise ValueError(
                f"mfi_consecutive_days must be >= 1, got "
                f"{self.mfi_consecutive_days}"
            )

    # ------------------------------------------------------------------
    # Basic strategy metadata
    # ------------------------------------------------------------------
    @property
    def trade_direction(self) -> TradeDirection:
        """Long-only."""
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        """Columns that must exist in the raw CSV data.

        Only the columns needed by the currently ENABLED exit sub-rules are
        required, so disabling a rule also relaxes the data requirement.
        """
        cols = ["date", "close", "atr_14_atr", self._ma_column]
        if self.adx_exit_enabled:
            cols.append("adx_14_adx")
        if self.macd_exit_enabled:
            cols.append("macd_14_macd_signal")
        if self.mfi_exit_enabled:
            cols.append("mfi_14_mfi")
        # ROC is computed from ``close`` at runtime - no extra raw column.
        return cols

    # ------------------------------------------------------------------
    # Indicator preparation
    # ------------------------------------------------------------------
    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create the aliased columns and the runtime-computed ROC column.

        * ``atr_14``: alias for ``atr_14_atr``.
        * ``alpha_ma_offset``: MA shifted forward by ``ma_offset`` bars.
        * ``alpha_roc_{n}``: percentage change over ``roc_period`` bars,
          calculated causally via ``pct_change`` - only added when the ROC
          sub-rule is enabled.
        """
        df = data.copy()

        df["atr_14"] = df["atr_14_atr"]
        df["alpha_ma_offset"] = df[self._ma_column].shift(self.ma_offset)

        if self.roc_exit_enabled:
            # pct_change is causal: value at bar t depends only on bars
            # [t - roc_period, t]. Multiply by 100 to express as a percent.
            df[self._roc_column] = (
                df["close"].pct_change(self.roc_period) * 100.0
            )

        return df

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------
    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Return a BUY signal when the AlphaMACross indicator fires long."""
        if context.current_index < self.ma_offset + 1:
            return None

        ma_now = context.get_indicator_value(self._ma_column, offset=0)
        ma_prev = context.get_indicator_value(self._ma_column, offset=-1)
        ma_off_now = context.get_indicator_value("alpha_ma_offset", offset=0)
        ma_off_prev = context.get_indicator_value("alpha_ma_offset", offset=-1)

        for value in (ma_now, ma_prev, ma_off_now, ma_off_prev):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None

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

        return current_price * 0.95

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """Risk-based sizing (identical to V1)."""
        if signal.stop_loss is None:
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        equity = context.total_equity
        risk_amount = equity * (self.risk_perc / 100.0)

        stop_distance = context.current_price - signal.stop_loss
        if stop_distance <= 0:
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        stop_distance_base = stop_distance * context.fx_rate
        return risk_amount / stop_distance_base

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid(value) -> bool:
        """True iff ``value`` is a finite number."""
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return True

    def _adx_exit(self, context: StrategyContext) -> Optional[bool]:
        """Return True/False if the ADX rule can be evaluated, else None."""
        adx = context.get_indicator_value("adx_14_adx")
        if not self._is_valid(adx):
            return None
        return bool(adx < self.adx_threshold)

    def _roc_exit(self, context: StrategyContext) -> Optional[bool]:
        """Return True/False if the ROC rule can be evaluated, else None."""
        if context.current_index < self.roc_period:
            return None
        roc = context.get_indicator_value(self._roc_column)
        if not self._is_valid(roc):
            return None
        return bool(roc < self.roc_threshold)

    def _macd_exit(self, context: StrategyContext) -> Optional[bool]:
        """Return True/False if the MACD rule can be evaluated, else None."""
        macd_signal = context.get_indicator_value("macd_14_macd_signal")
        if not self._is_valid(macd_signal):
            return None
        return bool(macd_signal < self.MACD_EXIT_LEVEL)

    def _mfi_exit(self, context: StrategyContext) -> Optional[bool]:
        """Return True/False if the MFI rule can be evaluated, else None.

        Fires when MFI has been below 50 for at least
        ``mfi_consecutive_days`` consecutive bars including the current one.
        """
        if context.current_index < self.mfi_consecutive_days - 1:
            return None

        for k in range(self.mfi_consecutive_days):
            value = context.get_indicator_value("mfi_14_mfi", offset=-k)
            if not self._is_valid(value):
                return None
            if value >= self.MFI_EXIT_LEVEL:
                return False
        return True

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Combined-indicator exit.

        Evaluates every enabled sub-rule. A full exit fires only when every
        enabled sub-rule is currently signalling exit. If any enabled rule
        cannot be evaluated (e.g. warmup / missing data), no exit is issued.
        The stop loss is handled separately by the engine using the price
        from ``calculate_initial_stop_loss``.
        """
        if not context.has_position:
            return None

        checks = []
        if self.adx_exit_enabled:
            checks.append(("ADX", self._adx_exit(context)))
        if self.roc_exit_enabled:
            checks.append(("ROC", self._roc_exit(context)))
        if self.macd_exit_enabled:
            checks.append(("MACD", self._macd_exit(context)))
        if self.mfi_exit_enabled:
            checks.append(("MFI", self._mfi_exit(context)))

        # No enabled rules -> exit only via stop loss.
        if not checks:
            return None

        # Every enabled rule must report a definite exit signal.
        for _, result in checks:
            if result is not True:
                return None

        fired = ", ".join(name for name, _ in checks)
        return Signal.sell(reason=f"Combined exit ({fired})")
