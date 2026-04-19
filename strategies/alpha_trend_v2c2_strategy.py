"""
AlphaTrendV2C2 Strategy

AlphaTrendV2C2 is a trend-following long-only strategy identical to
``AlphaTrendV2Strategy`` on entry, initial stop loss and position sizing. The
exit logic is replaced with a three-stage lifecycle:

1. Peace-period: only the initial ATR-based stop loss can close the trade.
2. Short-term: exit on initial stop loss, OR when the ADX sub-rule fires, OR
   when the MACD sub-rule fires.
3. Long-term: the stop loss starts moving (tightening only) using a linear
   schedule ``Max% - sensitivity * time``, clamped below at ``Min%``. The MFI
   consecutive-days sub-rule is also active. The global vulnerability scorer
   is left untouched and operates at the portfolio level as usual.

Stage boundaries are measured in bars since trade entry:
  bars < peace_period_candles                    -> peace-period
  peace_period_candles <= bars < peace_period_candles + short_term_candles
                                                 -> short-term
  otherwise                                      -> long-term

Entry: MA crosses above MA offset by ``ma_offset`` bars (AlphaMACross).
Stop Loss: ``entry_price - atr_sl * ATR_14`` at entry. In long-term, the SL
    is recomputed every bar as ``close * (1 - sl_pct/100)`` where
    ``sl_pct = max(sl_max_pct - sl_sensitivity * bars_since_entry,
    sl_min_pct)``. The SL never widens: the first long-term tightening only
    happens when the new SL is at or above the current SL, and afterwards the
    SL is updated only when the recomputed SL is tighter than the current one.
Position Sizing: Risk-based, identical to V1/V2.

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - atr_14_atr: 14-period ATR used for initial stop loss.
    - ema_{N}_ema / sma_{N}_sma: Pre-calculated moving average from raw data.
    - adx_14_adx: ADX 14, required when ``adx_exit_enabled`` is True.
    - macd_14_macd_signal: MACD signal line, required when
      ``macd_exit_enabled`` is True.
    - mfi_14_mfi: Money Flow Index 14, required when ``mfi_exit_enabled`` is
      True.

OPTIMIZABLE PARAMETERS:
    - ma_offset: Number of bars to offset the comparison MA.
    - ema_sma: Moving-average type (``"EMA"`` or ``"SMA"``).
    - ma_length: Length of the moving average. Must be one of the supported
      values: 7, 14, 20, 30, 50, 90, 200.
    - atr_sl: Multiplier on the 14-day ATR used for the initial stop loss.
    - risk_perc: Percent of total equity to risk per trade at the initial
      stop loss.
    - peace_period_candles: Length of the peace-period stage in bars.
    - short_term_candles: Length of the short-term stage in bars.
    - adx_exit_enabled: Toggle ADX exit sub-rule (short-term only).
    - adx_threshold: ADX level below which the ADX sub-rule exits.
    - macd_exit_enabled: Toggle MACD exit sub-rule (short-term only).
    - mfi_exit_enabled: Toggle MFI exit sub-rule (long-term only).
    - mfi_threshold: MFI level below which the MFI sub-rule fires.
    - mfi_consecutive_days: Number of consecutive bars (including current)
      during which MFI must stay below ``mfi_threshold``.
    - sl_sensitivity: Slope of the long-term SL schedule (percent per bar).
    - sl_max_pct: Initial SL distance (percent of price) at the start of the
      long-term stage.
    - sl_min_pct: Minimum SL distance (percent of price) the schedule is
      clamped to.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


class AlphaTrendV2C2Strategy(BaseStrategy):
    """
    AlphaTrendV2C2 - long-only trend-following strategy that keeps V2's
    entry/sizing/initial-SL but replaces the combined exit with a three-stage
    exit: peace-period (SL only), short-term (SL OR ADX OR MACD) and
    long-term (moving SL + MFI consecutive-days rule).
    """

    SUPPORTED_MA_LENGTHS = (7, 14, 20, 30, 50, 90, 200)

    MACD_EXIT_LEVEL = 0.0

    def __init__(self,
                 # Indicator parameters (same as V2)
                 ma_offset: int = 2,
                 ema_sma: str = "EMA",
                 ma_length: int = 20,
                 # Entry / initial SL / sizing (same as V2)
                 atr_sl: float = 2.0,
                 risk_perc: float = 2.0,
                 # Stage lengths
                 peace_period_candles: int = 5,
                 short_term_candles: int = 15,
                 # Short-term exit sub-rule toggles / params
                 adx_exit_enabled: bool = True,
                 adx_threshold: float = 20.0,
                 macd_exit_enabled: bool = True,
                 # Long-term MFI exit sub-rule
                 mfi_exit_enabled: bool = True,
                 mfi_threshold: float = 50.0,
                 mfi_consecutive_days: int = 3,
                 # Long-term moving-SL schedule
                 sl_sensitivity: float = 0.5,
                 sl_max_pct: float = 20.0,
                 sl_min_pct: float = 5.0):
        """Initialise the AlphaTrendV2C2 strategy.

        Args:
            ma_offset: Number of bars to offset the second MA by.
            ema_sma: Moving average type, ``"EMA"`` or ``"SMA"``.
            ma_length: Moving average length. Must be in
                ``SUPPORTED_MA_LENGTHS``.
            atr_sl: Multiplier on ATR(14) for the initial stop loss.
            risk_perc: Percent of total equity risked per trade.
            peace_period_candles: Length of the peace-period stage in bars.
            short_term_candles: Length of the short-term stage in bars.
            adx_exit_enabled: Enable the ADX exit sub-rule (short-term only).
            adx_threshold: ADX < ``adx_threshold`` triggers the ADX sub-rule.
            macd_exit_enabled: Enable the MACD signal-line exit sub-rule.
            mfi_exit_enabled: Enable the MFI exit sub-rule (long-term only).
            mfi_threshold: MFI level the MFI sub-rule compares against.
            mfi_consecutive_days: Number of consecutive bars (including the
                current bar) during which MFI must stay below
                ``mfi_threshold``.
            sl_sensitivity: Slope (percent per bar) for the long-term SL
                schedule.
            sl_max_pct: Starting SL distance for the long-term schedule, in
                percent of the current close.
            sl_min_pct: Minimum SL distance the schedule is clamped to, in
                percent of the current close.
        """
        self.ma_offset = int(ma_offset)
        self.ema_sma = str(ema_sma).strip().upper()
        self.ma_length = int(ma_length)
        self.atr_sl = float(atr_sl)
        self.risk_perc = float(risk_perc)

        self.peace_period_candles = int(peace_period_candles)
        self.short_term_candles = int(short_term_candles)

        self.adx_exit_enabled = bool(adx_exit_enabled)
        self.adx_threshold = float(adx_threshold)
        self.macd_exit_enabled = bool(macd_exit_enabled)

        self.mfi_exit_enabled = bool(mfi_exit_enabled)
        self.mfi_threshold = float(mfi_threshold)
        self.mfi_consecutive_days = int(mfi_consecutive_days)

        self.sl_sensitivity = float(sl_sensitivity)
        self.sl_max_pct = float(sl_max_pct)
        self.sl_min_pct = float(sl_min_pct)

        self._ma_column = self._resolve_ma_column(self.ema_sma, self.ma_length)

        # Per-symbol cache of (entry_date, entry_bar_index) and a flag tracking
        # whether the long-term moving SL has been armed (had its first move).
        self._entry_bar_index: Dict[str, Tuple[pd.Timestamp, int]] = {}
        self._long_term_sl_armed: Dict[str, bool] = {}

        super().__init__(
            ma_offset=self.ma_offset,
            ema_sma=self.ema_sma,
            ma_length=self.ma_length,
            atr_sl=self.atr_sl,
            risk_perc=self.risk_perc,
            peace_period_candles=self.peace_period_candles,
            short_term_candles=self.short_term_candles,
            adx_exit_enabled=self.adx_exit_enabled,
            adx_threshold=self.adx_threshold,
            macd_exit_enabled=self.macd_exit_enabled,
            mfi_exit_enabled=self.mfi_exit_enabled,
            mfi_threshold=self.mfi_threshold,
            mfi_consecutive_days=self.mfi_consecutive_days,
            sl_sensitivity=self.sl_sensitivity,
            sl_max_pct=self.sl_max_pct,
            sl_min_pct=self.sl_min_pct,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_ma_column(ema_sma: str, ma_length: int) -> str:
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
        if self.ma_offset < 1:
            raise ValueError(f"ma_offset must be >= 1, got {self.ma_offset}")

        if self.ema_sma not in {"EMA", "SMA"}:
            raise ValueError(
                f"ema_sma must be 'EMA' or 'SMA', got {self.ema_sma!r}"
            )

        if self.ma_length not in self.SUPPORTED_MA_LENGTHS:
            raise ValueError(
                f"ma_length must be one of {list(self.SUPPORTED_MA_LENGTHS)}, "
                f"got {self.ma_length}."
            )

        if self.atr_sl <= 0:
            raise ValueError(f"atr_sl must be > 0, got {self.atr_sl}")

        if not 0 < self.risk_perc <= 100:
            raise ValueError(
                f"risk_perc must be between 0 and 100, got {self.risk_perc}"
            )

        if self.peace_period_candles < 0:
            raise ValueError(
                f"peace_period_candles must be >= 0, got "
                f"{self.peace_period_candles}"
            )
        if self.short_term_candles < 0:
            raise ValueError(
                f"short_term_candles must be >= 0, got "
                f"{self.short_term_candles}"
            )

        if self.mfi_consecutive_days < 1:
            raise ValueError(
                f"mfi_consecutive_days must be >= 1, got "
                f"{self.mfi_consecutive_days}"
            )

        if self.sl_max_pct <= 0 or self.sl_max_pct >= 100:
            raise ValueError(
                f"sl_max_pct must be in (0, 100), got {self.sl_max_pct}"
            )
        if self.sl_min_pct <= 0 or self.sl_min_pct >= 100:
            raise ValueError(
                f"sl_min_pct must be in (0, 100), got {self.sl_min_pct}"
            )
        if self.sl_min_pct > self.sl_max_pct:
            raise ValueError(
                f"sl_min_pct ({self.sl_min_pct}) must be <= "
                f"sl_max_pct ({self.sl_max_pct})"
            )
        if self.sl_sensitivity < 0:
            raise ValueError(
                f"sl_sensitivity must be >= 0, got {self.sl_sensitivity}"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        cols = ["date", "close", "atr_14_atr", self._ma_column]
        if self.adx_exit_enabled:
            cols.append("adx_14_adx")
        if self.macd_exit_enabled:
            cols.append("macd_14_macd_signal")
        if self.mfi_exit_enabled:
            cols.append("mfi_14_mfi")
        return cols

    # ------------------------------------------------------------------
    # Indicator prep
    # ------------------------------------------------------------------
    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["atr_14"] = df["atr_14_atr"]
        df["alpha_ma_offset"] = df[self._ma_column].shift(self.ma_offset)
        return df

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
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

        # Reset long-term SL state for this symbol on any new entry.
        self._long_term_sl_armed[context.symbol] = False
        # Invalidate any stale entry-bar cache from a previous trade.
        self._entry_bar_index.pop(context.symbol, None)

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
        current_price = context.current_price
        atr = context.get_indicator_value("atr_14")
        if atr is not None and not (isinstance(atr, float) and np.isnan(atr)) and atr > 0:
            return current_price - (self.atr_sl * atr)
        return current_price * 0.95

    # ------------------------------------------------------------------
    # Sizing (same as V2)
    # ------------------------------------------------------------------
    def position_size(self, context: StrategyContext, signal: Signal) -> float:
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
    # Stage / bars-since-entry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid(value) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return True

    def _bars_since_entry(self, context: StrategyContext) -> Optional[int]:
        """Count bars elapsed since the position's entry bar (inclusive of
        the current bar but exclusive of the entry bar itself)."""
        if not context.has_position:
            return None

        symbol = context.symbol
        entry_date = context.position.entry_date

        cached = self._entry_bar_index.get(symbol)
        if cached is None or cached[0] != entry_date:
            data = context.data
            found = None
            entry_ts = pd.Timestamp(entry_date)
            # Linear scan backwards - typically hits within a few bars.
            for i in range(context.current_index, -1, -1):
                bar_date = data.iloc[i]["date"]
                if pd.Timestamp(bar_date) == entry_ts:
                    found = i
                    break
            if found is None:
                return None
            self._entry_bar_index[symbol] = (entry_date, found)

        return context.current_index - self._entry_bar_index[symbol][1]

    def _stage(self, bars_since_entry: int) -> str:
        if bars_since_entry < self.peace_period_candles:
            return "peace"
        if bars_since_entry < self.peace_period_candles + self.short_term_candles:
            return "short"
        return "long"

    # ------------------------------------------------------------------
    # Sub-rules
    # ------------------------------------------------------------------
    def _adx_exit(self, context: StrategyContext) -> Optional[bool]:
        adx = context.get_indicator_value("adx_14_adx")
        if not self._is_valid(adx):
            return None
        return bool(adx < self.adx_threshold)

    def _macd_exit(self, context: StrategyContext) -> Optional[bool]:
        macd_signal = context.get_indicator_value("macd_14_macd_signal")
        if not self._is_valid(macd_signal):
            return None
        return bool(macd_signal < self.MACD_EXIT_LEVEL)

    def _mfi_exit(self, context: StrategyContext) -> Optional[bool]:
        if context.current_index < self.mfi_consecutive_days - 1:
            return None
        for k in range(self.mfi_consecutive_days):
            value = context.get_indicator_value("mfi_14_mfi", offset=-k)
            if not self._is_valid(value):
                return None
            if value >= self.mfi_threshold:
                return False
        return True

    # ------------------------------------------------------------------
    # Long-term moving SL
    # ------------------------------------------------------------------
    def _long_term_sl_price(self, context: StrategyContext,
                            bars_since_entry: int) -> float:
        """Compute the current long-term SL price from the linear schedule.

        ``sl_pct = max(sl_max_pct - sl_sensitivity * bars_since_entry,
        sl_min_pct)`` then ``SL = close * (1 - sl_pct/100)`` for a long
        position.
        """
        sl_pct = self.sl_max_pct - self.sl_sensitivity * float(bars_since_entry)
        if sl_pct < self.sl_min_pct:
            sl_pct = self.sl_min_pct
        return context.current_price * (1.0 - sl_pct / 100.0)

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """Move the SL every bar in the long-term stage, tightening only.

        The first move only happens when the computed SL is at least the
        current SL (i.e. there is enough room to tighten). The engine also
        enforces monotonic tightening for LONG positions.
        """
        if not context.has_position:
            return None

        bars = self._bars_since_entry(context)
        if bars is None:
            return None
        if self._stage(bars) != "long":
            return None

        current_stop = context.position.stop_loss
        new_stop = self._long_term_sl_price(context, bars)

        symbol = context.symbol
        armed = self._long_term_sl_armed.get(symbol, False)

        if not armed:
            if current_stop is not None and new_stop < current_stop:
                # Not enough room yet - wait for a future bar.
                return None
            self._long_term_sl_armed[symbol] = True
            return new_stop

        # Already armed - only tighten.
        if current_stop is not None and new_stop <= current_stop:
            return None
        return new_stop

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------
    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if not context.has_position:
            return None

        bars = self._bars_since_entry(context)
        if bars is None:
            return None

        stage = self._stage(bars)

        if stage == "peace":
            # Only the initial stop loss (handled by the engine) can exit.
            return None

        if stage == "short":
            fired: List[str] = []
            if self.adx_exit_enabled and self._adx_exit(context) is True:
                fired.append("ADX")
            if self.macd_exit_enabled and self._macd_exit(context) is True:
                fired.append("MACD")
            if fired:
                return Signal.sell(
                    reason=f"Short-term exit ({', '.join(fired)})"
                )
            return None

        # Long-term: only the MFI sub-rule can generate a strategy exit.
        # (The moving SL is handled in should_adjust_stop; the vulnerability
        # scorer continues to run at the portfolio level as usual.)
        if self.mfi_exit_enabled and self._mfi_exit(context) is True:
            return Signal.sell(reason="Long-term exit (MFI)")

        return None
