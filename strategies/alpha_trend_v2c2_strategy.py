"""
AlphaTrendV2C2 Strategy

AlphaTrendV2C2 is a trend-following long-only strategy identical to
``AlphaTrendV2Strategy`` on entry, initial stop loss and position sizing. The
exit logic is replaced with a three-stage lifecycle:

1. Peace-period: only the initial ATR-based stop loss can close the trade.
2. Short-term: initial SL is always active. If the short-phase exit rule is
   enabled, the trade is also closed when the close price stays strictly below
   the entry price (break-even) for ``break_even_consecutive_bars`` consecutive
   bars (including the current bar).
3. Long-term: a trailing stop loss is maintained ``trailing_sl_pct`` percent
   below the current close and only moves up (never down). The portfolio-level
   vulnerability scorer continues to run as usual and can retire the position;
   no other strategy-level exit is active in this stage.

Stage boundaries are measured in bars since trade entry:
  bars < peace_period_candles                    -> peace-period
  peace_period_candles <= bars < peace_period_candles + short_term_candles
                                                 -> short-term
  otherwise                                      -> long-term

Entry: MA crosses above MA offset by ``ma_offset`` bars (AlphaMACross).
Stop Loss: ``entry_price - atr_sl * ATR_14`` at entry. In the long-term
    stage, the SL is recomputed every bar as
    ``close * (1 - trailing_sl_pct/100)`` and applied only when it is tighter
    (higher) than the current SL.
Position Sizing: Risk-based, identical to V1/V2.

RAW DATA INDICATORS (NOT OPTIMIZABLE):
    - atr_14_atr: 14-period ATR used for initial stop loss.
    - ema_{N}_ema / sma_{N}_sma: Pre-calculated moving average from raw data.

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
    - short_phase_exit_enabled: Toggle the short-phase break-even exit rule
      (SL remains active regardless).
    - break_even_consecutive_bars: Number of consecutive bars the close must
      stay strictly below the entry price for the short-phase rule to fire.
    - trailing_sl_pct: Distance of the long-term trailing SL as a percent of
      the current close.
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
    exit: peace-period (SL only), short-term (SL + optional break-even
    consecutive-bars rule) and long-term (trailing SL only, plus the
    portfolio-level vulnerability scorer).
    """

    SUPPORTED_MA_LENGTHS = (7, 14, 20, 30, 50, 90, 200)

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
                 # Short-phase break-even exit rule
                 short_phase_exit_enabled: bool = True,
                 break_even_consecutive_bars: int = 7,
                 # Long-phase trailing SL
                 trailing_sl_pct: float = 20.0):
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
            short_phase_exit_enabled: Enable the short-phase break-even
                consecutive-bars exit rule. The initial SL is always active
                regardless of this flag.
            break_even_consecutive_bars: Number of consecutive bars
                (including the current bar) during which the close must stay
                strictly below the entry price for the short-phase rule to
                fire.
            trailing_sl_pct: Distance of the long-term trailing SL as a
                percent of the current close. The SL only moves up.
        """
        self.ma_offset = int(ma_offset)
        self.ema_sma = str(ema_sma).strip().upper()
        self.ma_length = int(ma_length)
        self.atr_sl = float(atr_sl)
        self.risk_perc = float(risk_perc)

        self.peace_period_candles = int(peace_period_candles)
        self.short_term_candles = int(short_term_candles)

        self.short_phase_exit_enabled = bool(short_phase_exit_enabled)
        self.break_even_consecutive_bars = int(break_even_consecutive_bars)

        self.trailing_sl_pct = float(trailing_sl_pct)

        self._ma_column = self._resolve_ma_column(self.ema_sma, self.ma_length)

        # Per-symbol cache of (entry_date, entry_bar_index).
        self._entry_bar_index: Dict[str, Tuple[pd.Timestamp, int]] = {}

        super().__init__(
            ma_offset=self.ma_offset,
            ema_sma=self.ema_sma,
            ma_length=self.ma_length,
            atr_sl=self.atr_sl,
            risk_perc=self.risk_perc,
            peace_period_candles=self.peace_period_candles,
            short_term_candles=self.short_term_candles,
            short_phase_exit_enabled=self.short_phase_exit_enabled,
            break_even_consecutive_bars=self.break_even_consecutive_bars,
            trailing_sl_pct=self.trailing_sl_pct,
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

        if self.break_even_consecutive_bars < 1:
            raise ValueError(
                f"break_even_consecutive_bars must be >= 1, got "
                f"{self.break_even_consecutive_bars}"
            )

        if self.trailing_sl_pct <= 0 or self.trailing_sl_pct >= 100:
            raise ValueError(
                f"trailing_sl_pct must be in (0, 100), got "
                f"{self.trailing_sl_pct}"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ["date", "close", "atr_14_atr", self._ma_column]

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
    def _break_even_exit(self, context: StrategyContext,
                         bars_since_entry: int) -> bool:
        """Return True if the close has stayed strictly below the entry
        price for ``break_even_consecutive_bars`` consecutive bars
        (including the current bar).

        The rule cannot look further back than the entry bar itself, so at
        least that many bars must have elapsed since entry.
        """
        n = self.break_even_consecutive_bars
        # The rule checks the current bar plus (n-1) prior bars; we only
        # consider bars from the entry bar onwards (exclusive of entry bar).
        if bars_since_entry < n:
            return False

        entry_price = float(context.position.entry_price)
        for k in range(n):
            close = context.get_indicator_value("close", offset=-k)
            if not self._is_valid(close):
                return False
            if float(close) >= entry_price:
                return False
        return True

    # ------------------------------------------------------------------
    # Long-term trailing SL
    # ------------------------------------------------------------------
    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        """Maintain a trailing SL at ``trailing_sl_pct`` below close during
        the long-term stage. The SL only moves up: the engine also enforces
        monotonic tightening for LONG positions.
        """
        if not context.has_position:
            return None

        bars = self._bars_since_entry(context)
        if bars is None:
            return None
        if self._stage(bars) != "long":
            return None

        new_stop = context.current_price * (1.0 - self.trailing_sl_pct / 100.0)
        current_stop = context.position.stop_loss

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
            # Initial SL is always active (engine-handled). The break-even
            # consecutive-bars rule is optional.
            if self.short_phase_exit_enabled and self._break_even_exit(context, bars):
                return Signal.sell(
                    reason=(
                        f"Short-term exit (close below entry for "
                        f"{self.break_even_consecutive_bars} bars)"
                    )
                )
            return None

        # Long-term: no strategy-level exit beyond the trailing SL (handled
        # in should_adjust_stop) and the portfolio-level vulnerability
        # scorer.
        return None
