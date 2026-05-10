"""
AlphaTrendV2C3 Strategy

AlphaTrendV2C3 is identical to ``AlphaTrendV2C2Strategy`` in every respect
(initial stop loss, position sizing, three-stage exit lifecycle, long-term
trailing SL) except for the entry condition. The MA-cross signal from V2/V2C2
must now be confirmed by an n-bar closing-high before a trade is opened:

Entry rule:
  1. Detect AlphaMACross long signal: ``MA`` crosses above ``MA`` shifted by
     ``ma_offset`` bars (same as V2C2).
  2. The trade is only opened on a bar whose close is strictly greater than the
     maximum close over the previous ``high_lookback_candles`` bars (the cross
     bar itself is excluded from that maximum).
  3. If the cross fires but the close is not a new n-bar high on the cross bar,
     the signal becomes "pending": each of the next ``entry_wait_candles`` bars
     re-checks the same high condition. The trade is opened on the first bar
     within that window where the high condition is met.
  4. If a new MA-cross fires while a previous signal is still pending, the
     wait window is reset from the new cross bar.
  5. If the wait window expires without confirmation, the pending signal is
     discarded; a new MA-cross is required.

OPTIMIZABLE PARAMETERS (in addition to all V2C2 parameters):
    - high_lookback_candles: Number of prior closes (excluding the current
      bar) used to evaluate the n-bar closing high.
    - entry_wait_candles: Number of bars after the cross during which the
      strategy keeps watching for the high confirmation. ``0`` means the
      high must be reached on the cross bar itself.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal

from strategies.alpha_trend_v2c2_strategy import AlphaTrendV2C2Strategy


class AlphaTrendV2C3Strategy(AlphaTrendV2C2Strategy):
    """
    AlphaTrendV2C3 - V2C2 with an n-bar closing-high confirmation gate on
    entries. The cross signal opens a wait window of ``entry_wait_candles``
    bars after the cross bar; the trade is opened on the first bar within the
    window whose close is strictly greater than the max close over the prior
    ``high_lookback_candles`` bars. A new cross resets the wait window.
    """

    def __init__(self,
                 # All V2C2 parameters
                 ma_offset: int = 2,
                 ema_sma: str = "EMA",
                 ma_length: int = 20,
                 atr_sl: float = 2.0,
                 risk_perc: float = 2.0,
                 peace_period_candles: int = 5,
                 short_term_candles: int = 15,
                 short_phase_exit_enabled: bool = True,
                 break_even_consecutive_bars: int = 7,
                 trailing_sl_pct: float = 20.0,
                 # New V2C3 parameters
                 high_lookback_candles: int = 10,
                 entry_wait_candles: int = 3):
        """Initialise the AlphaTrendV2C3 strategy.

        Args:
            high_lookback_candles: Number of prior bars (excluding the current
                bar) used to compute the closing-high reference. The current
                close must be strictly greater than the maximum close over
                these bars for an entry to be confirmed.
            entry_wait_candles: Number of bars after the cross during which
                the strategy keeps re-evaluating the high condition. ``0``
                requires the high to be reached on the cross bar itself.

        See ``AlphaTrendV2C2Strategy`` for the remaining parameters.
        """
        # Set V2C3-specific attrs before super().__init__ so that
        # _validate_parameters (called inside super) can see them.
        self.high_lookback_candles = int(high_lookback_candles)
        self.entry_wait_candles = int(entry_wait_candles)

        # Per-symbol cache of (cross_date, cross_bar_index) for the active
        # pending entry signal awaiting high confirmation.
        self._pending_cross: Dict[str, Tuple[pd.Timestamp, int]] = {}

        super().__init__(
            ma_offset=ma_offset,
            ema_sma=ema_sma,
            ma_length=ma_length,
            atr_sl=atr_sl,
            risk_perc=risk_perc,
            peace_period_candles=peace_period_candles,
            short_term_candles=short_term_candles,
            short_phase_exit_enabled=short_phase_exit_enabled,
            break_even_consecutive_bars=break_even_consecutive_bars,
            trailing_sl_pct=trailing_sl_pct,
        )

        # Make the V2C3-specific params introspectable on ``self.params``.
        self.params["high_lookback_candles"] = self.high_lookback_candles
        self.params["entry_wait_candles"] = self.entry_wait_candles

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        if self.high_lookback_candles < 1:
            raise ValueError(
                f"high_lookback_candles must be >= 1, got "
                f"{self.high_lookback_candles}"
            )
        if self.entry_wait_candles < 0:
            raise ValueError(
                f"entry_wait_candles must be >= 0, got {self.entry_wait_candles}"
            )

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def _is_n_bar_high_close(self, context: StrategyContext) -> bool:
        """Return True if today's close is strictly greater than the maximum
        close over the previous ``high_lookback_candles`` bars."""
        n = self.high_lookback_candles
        today_close = context.get_indicator_value("close", offset=0)
        if not self._is_valid(today_close):
            return False
        prior_max: Optional[float] = None
        for k in range(1, n + 1):
            prior_close = context.get_indicator_value("close", offset=-k)
            if not self._is_valid(prior_close):
                return False
            value = float(prior_close)
            if prior_max is None or value > prior_max:
                prior_max = value
        if prior_max is None:
            return False
        return float(today_close) > prior_max

    def _detect_ma_cross(self, context: StrategyContext) -> bool:
        """Same MA-cross detection as V2C2's entry rule."""
        ma_now = context.get_indicator_value(self._ma_column, offset=0)
        ma_prev = context.get_indicator_value(self._ma_column, offset=-1)
        ma_off_now = context.get_indicator_value("alpha_ma_offset", offset=0)
        ma_off_prev = context.get_indicator_value("alpha_ma_offset", offset=-1)
        for value in (ma_now, ma_prev, ma_off_now, ma_off_prev):
            if not self._is_valid(value):
                return False
        return (ma_prev <= ma_off_prev) and (ma_now > ma_off_now)

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index < self.ma_offset + 1:
            return None
        # Need enough history for the high lookback window.
        if context.current_index < self.high_lookback_candles:
            return None

        if self._detect_ma_cross(context):
            # New cross: (re)start the wait window from this bar.
            self._pending_cross[context.symbol] = (
                context.current_date,
                context.current_index,
            )

        pending = self._pending_cross.get(context.symbol)
        if pending is None:
            return None

        _, cross_index = pending
        bars_elapsed = context.current_index - cross_index
        if bars_elapsed < 0 or bars_elapsed > self.entry_wait_candles:
            # Stale or expired pending - clear and wait for a new cross.
            self._pending_cross.pop(context.symbol, None)
            return None

        if not self._is_n_bar_high_close(context):
            return None

        # Confirmation reached: clear caches and emit the buy.
        self._pending_cross.pop(context.symbol, None)
        self._entry_bar_index.pop(context.symbol, None)

        stop_loss = self.calculate_initial_stop_loss(context)
        return Signal.buy(
            size=1.0,
            stop_loss=stop_loss,
            reason=(
                f"AlphaMACross long + {self.high_lookback_candles}-bar high "
                f"close confirm ({self.ema_sma} {self.ma_length} vs offset "
                f"{self.ma_offset})"
            ),
            direction=self.trade_direction,
        )
