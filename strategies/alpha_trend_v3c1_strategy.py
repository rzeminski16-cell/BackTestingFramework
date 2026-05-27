"""
AlphaTrendV3C1 Strategy

AlphaTrendV3C1 is identical to ``AlphaTrendV2C3Strategy`` (entry cross +
n-bar closing-high confirmation, initial ATR stop loss, position sizing,
three-stage exit lifecycle with an ATR-based long-term trailing stop) except
for two additional, independently toggleable rules:

  1. Additional ENTRY rule (MA-regime filter): a pre-calculated ``EMA`` of
     length ``entry_ema_length`` must be strictly above a pre-calculated
     ``SMA`` of length ``entry_sma_length``. Both lengths are chosen from the
     lengths available in the raw data (7, 14, 20, 30, 50, 90, 200). The
     filter is evaluated on the bar the trade would be opened (the
     confirmation bar) and acts as an AND gate on top of the inherited V2C3
     entry. If the V2C3 confirmation fires but the filter does not yet hold,
     the pending wait window is preserved so a later bar within the window can
     still open the trade.

  2. Additional EXIT rule (overextension exit): when the close is at least
     ``exit_ema_distance_pct`` percent ABOVE a pre-calculated ``EMA`` of
     length ``exit_ema_length`` (i.e. the EMA sits below the close by at least
     that percentage, measured relative to the EMA), the position is closed.
     The percentage distance is ``(close - EMA) / EMA * 100``. This exit is
     evaluated only in the short-term and long-term stages (it never fires
     during the peace period, where only the initial SL can close the trade).

Both new rules can be turned off via their enable flags, in which case the
strategy behaves exactly like V2C3.

OPTIMIZABLE PARAMETERS (V3C1-specific additions vs V2C3):
    - entry_ema_sma_filter_enabled: Toggle the EMA>SMA entry filter.
    - entry_ema_length: Length of the EMA used by the entry filter
      (pre-calculated raw-data column ``ema_{N}_ema``).
    - entry_sma_length: Length of the SMA used by the entry filter
      (pre-calculated raw-data column ``sma_{N}_sma``).
    - ema_distance_exit_enabled: Toggle the EMA-distance overextension exit.
    - exit_ema_length: Length of the EMA used by the exit rule
      (pre-calculated raw-data column ``ema_{N}_ema``).
    - exit_ema_distance_pct: Minimum percentage the close must exceed the exit
      EMA by for the exit to fire, measured as ``(close - EMA) / EMA * 100``.
"""
from typing import List, Optional

from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal

from strategies.alpha_trend_v2c3_strategy import AlphaTrendV2C3Strategy


class AlphaTrendV3C1Strategy(AlphaTrendV2C3Strategy):
    """
    AlphaTrendV3C1 - V2C3 plus two toggleable rules: an EMA>SMA regime filter
    on entries and an EMA-distance overextension exit (skips the peace period).
    """

    def __init__(self,
                 # All V2C3 parameters
                 ma_offset: int = 2,
                 ema_sma: str = "EMA",
                 ma_length: int = 20,
                 atr_sl: float = 2.0,
                 risk_perc: float = 2.0,
                 peace_period_candles: int = 5,
                 short_term_candles: int = 15,
                 short_phase_exit_enabled: bool = True,
                 break_even_consecutive_bars: int = 7,
                 trailing_atr_mult: float = 2.0,
                 high_lookback_candles: int = 10,
                 entry_wait_candles: int = 3,
                 # New V3C1 parameters
                 entry_ema_sma_filter_enabled: bool = True,
                 entry_ema_length: int = 20,
                 entry_sma_length: int = 50,
                 ema_distance_exit_enabled: bool = True,
                 exit_ema_length: int = 20,
                 exit_ema_distance_pct: float = 10.0):
        """Initialise the AlphaTrendV3C1 strategy.

        Args:
            entry_ema_sma_filter_enabled: Enable the EMA>SMA entry filter.
            entry_ema_length: EMA length for the entry filter. Must be in
                ``SUPPORTED_MA_LENGTHS`` (read from raw data).
            entry_sma_length: SMA length for the entry filter. Must be in
                ``SUPPORTED_MA_LENGTHS`` (read from raw data).
            ema_distance_exit_enabled: Enable the EMA-distance exit.
            exit_ema_length: EMA length for the exit rule. Must be in
                ``SUPPORTED_MA_LENGTHS`` (read from raw data).
            exit_ema_distance_pct: The exit fires when
                ``(close - EMA) / EMA * 100 >= exit_ema_distance_pct``.

        See ``AlphaTrendV2C3Strategy`` for the remaining parameters.
        """
        # Set V3C1-specific attrs before super().__init__ so that
        # _validate_parameters and required_columns (both reached inside
        # super().__init__) can see them.
        self.entry_ema_sma_filter_enabled = bool(entry_ema_sma_filter_enabled)
        self.entry_ema_length = int(entry_ema_length)
        self.entry_sma_length = int(entry_sma_length)
        self.ema_distance_exit_enabled = bool(ema_distance_exit_enabled)
        self.exit_ema_length = int(exit_ema_length)
        self.exit_ema_distance_pct = float(exit_ema_distance_pct)

        self._entry_ema_column = self._resolve_ma_column("EMA", self.entry_ema_length)
        self._entry_sma_column = self._resolve_ma_column("SMA", self.entry_sma_length)
        self._exit_ema_column = self._resolve_ma_column("EMA", self.exit_ema_length)

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
            trailing_atr_mult=trailing_atr_mult,
            high_lookback_candles=high_lookback_candles,
            entry_wait_candles=entry_wait_candles,
        )

        # Surface the new params alongside the inherited ones.
        self.params["entry_ema_sma_filter_enabled"] = self.entry_ema_sma_filter_enabled
        self.params["entry_ema_length"] = self.entry_ema_length
        self.params["entry_sma_length"] = self.entry_sma_length
        self.params["ema_distance_exit_enabled"] = self.ema_distance_exit_enabled
        self.params["exit_ema_length"] = self.exit_ema_length
        self.params["exit_ema_distance_pct"] = self.exit_ema_distance_pct

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        for name, value in (
            ("entry_ema_length", self.entry_ema_length),
            ("entry_sma_length", self.entry_sma_length),
            ("exit_ema_length", self.exit_ema_length),
        ):
            if value not in self.SUPPORTED_MA_LENGTHS:
                raise ValueError(
                    f"{name} must be one of {list(self.SUPPORTED_MA_LENGTHS)}, "
                    f"got {value}. The strategy reads MAs directly from raw "
                    f"data so only pre-calculated lengths are supported."
                )
        if self.exit_ema_distance_pct <= 0:
            raise ValueError(
                f"exit_ema_distance_pct must be > 0, got "
                f"{self.exit_ema_distance_pct}"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def required_columns(self) -> List[str]:
        cols = list(super().required_columns())
        if self.entry_ema_sma_filter_enabled:
            for col in (self._entry_ema_column, self._entry_sma_column):
                if col not in cols:
                    cols.append(col)
        if self.ema_distance_exit_enabled:
            if self._exit_ema_column not in cols:
                cols.append(self._exit_ema_column)
        return cols

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def _entry_ema_sma_filter_pass(self, context: StrategyContext) -> bool:
        """Return True if the EMA>SMA entry filter is satisfied (or disabled).

        When enabled, requires the entry EMA to be strictly greater than the
        entry SMA on the current bar. Missing/NaN values fail the filter.
        """
        if not self.entry_ema_sma_filter_enabled:
            return True
        ema = context.get_indicator_value(self._entry_ema_column, offset=0)
        sma = context.get_indicator_value(self._entry_sma_column, offset=0)
        if not self._is_valid(ema) or not self._is_valid(sma):
            return False
        return float(ema) > float(sma)

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index < self.ma_offset + 1:
            return None
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
            self._pending_cross.pop(context.symbol, None)
            return None

        if not self._is_n_bar_high_close(context):
            return None

        # Additional EMA>SMA gate. On failure we deliberately keep the pending
        # signal alive so a later bar within the wait window can still confirm
        # once both the high close and the filter hold on the same bar.
        if not self._entry_ema_sma_filter_pass(context):
            return None

        # Both conditions met: clear caches and emit the buy.
        self._pending_cross.pop(context.symbol, None)
        self._entry_bar_index.pop(context.symbol, None)

        stop_loss = self.calculate_initial_stop_loss(context)
        filter_note = ""
        if self.entry_ema_sma_filter_enabled:
            filter_note = (
                f" + EMA{self.entry_ema_length} > SMA{self.entry_sma_length}"
            )
        return Signal.buy(
            size=1.0,
            stop_loss=stop_loss,
            reason=(
                f"AlphaMACross long + {self.high_lookback_candles}-bar high "
                f"close confirm ({self.ema_sma} {self.ma_length} vs offset "
                f"{self.ma_offset}){filter_note}"
            ),
            direction=self.trade_direction,
        )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------
    def _ema_distance_exit(self, context: StrategyContext) -> bool:
        """Return True when the close is at least ``exit_ema_distance_pct``
        percent above the exit EMA, measured as ``(close - EMA) / EMA * 100``.
        """
        if not self.ema_distance_exit_enabled:
            return False
        ema = context.get_indicator_value(self._exit_ema_column, offset=0)
        if not self._is_valid(ema) or float(ema) <= 0:
            return False
        close = context.get_indicator_value("close", offset=0)
        if not self._is_valid(close):
            return False
        distance_pct = (float(close) - float(ema)) / float(ema) * 100.0
        return distance_pct >= self.exit_ema_distance_pct

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if not context.has_position:
            return None

        bars = self._bars_since_entry(context)
        if bars is None:
            return None

        stage = self._stage(bars)
        if stage == "peace":
            # Only the initial stop loss (engine-handled) can exit.
            return None

        # Short-term and long-term stages: the EMA-distance overextension exit
        # is allowed to fire (the peace period is intentionally skipped).
        if self._ema_distance_exit(context):
            return Signal.sell(
                reason=(
                    f"EMA-distance exit (close >= "
                    f"{self.exit_ema_distance_pct}% above EMA"
                    f"{self.exit_ema_length})"
                )
            )

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

        # Long-term: no strategy-level exit beyond the EMA-distance rule above,
        # the trailing SL (should_adjust_stop) and the portfolio vulnerability
        # scorer.
        return None
