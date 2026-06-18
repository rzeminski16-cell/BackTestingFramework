"""
ShortOnlyBase Strategy

ShortOnlyBase is a SHORT-only strategy that only trades when a market-wide
regime gate confirms a weak tape, then shorts individual names either as they
break down to new lows (trend continuation) or as they blow off to the upside
(overextension fade).

REGIME GATE (all three must hold, otherwise no entries are taken):
    - benchmark_close < benchmark 100-day MA  (broad market below its 100DMA)
    - breadth_pct_above_50dma < 40%           (few stocks above their 50DMA)
    - VIX close > VIX 60-day SMA               (elevated / rising volatility)

The framework feeds a strategy only its own ticker's data, so the regime series
are loaded directly from disk and merged onto the symbol by date (the same way
``raw_data/benchmarks`` already stores the benchmark and VIX histories):
    - benchmark + VIX: ``raw_data/benchmarks/SPX_daily.csv`` and
      ``raw_data/benchmarks/VIX_daily.csv`` (their MAs are computed at run time).
    - breadth: ``raw_data/benchmarks/BREADTH_daily.csv``, produced by
      ``scripts/compute_breadth.py`` (% of the daily universe above its 50DMA).

UNIVERSE FILTERS (applied as entry guards):
    - close >= ``min_price``
    - Avg_daily_volume_20D >= ``min_adv_20``

ENTRY (close-based) - Block A OR Block B:
    Block A (trend breakdown):
        EMA_7 < EMA_20 < EMA_50           (EMA_7 substitutes for EMA_5; the raw
                                           data has no precomputed EMA_5)
        AND return_20d < 0
        AND return_60d < 0
        AND close < lowest_close over the prior 20 days (new 20-day low)
        AND RSI_14 > ``rsi_floor`` (default 20; avoid already-washed-out names)
    Block B (overextension fade):
        ( 3-day rally > ``rally_atr_mult`` * ATR_14
          OR gap_up > ``gap_atr_mult`` * ATR_14 )
        AND close > upper Bollinger band

EXECUTION: enter on close (engine fills at the signal bar's close).

EXIT (cover):
    - close > EMA_20                                   (technical cover)
    - close > entry_price + ``atr_sl`` * ATR_14        (protective stop, set at
      entry and enforced intrabar by the engine via ``calculate_initial_stop_loss``)
    - holding_days >= ``time_exit``                    (time stop)

POSITION SIZING: risk-based. ``risk_perc`` percent of total equity is risked at
the initial stop distance (stop_loss - entry for a short).

RAW DATA INDICATORS (NOT OPTIMIZABLE - period baked into the column name):
    - atr_14_atr, rsi_14_rsi, ema_7_ema, ema_20_ema, ema_50_ema,
      ``bbands_20_real upper band``.

OPTIMIZABLE PARAMETERS (computed at run time):
    - benchmark_ma_period: Benchmark MA length for the regime gate (default 100).
    - vix_sma_period: VIX SMA length for the regime gate (default 60).
    - breadth_threshold: Max breadth %% to allow entries (default 40).
    - rsi_floor: Minimum RSI_14 for Block A (default 20).
    - rally_atr_mult: ATR multiple for the 3-day rally trigger (default 2.0).
    - gap_atr_mult: ATR multiple for the gap-up trigger (default 1.5).
    - atr_sl: ATR multiple for the protective stop above entry (default 2.5).
    - time_exit: Maximum holding period in days (default 10).
    - risk_perc: Percent of equity risked per trade (default 2.0).
    - min_price: Minimum close price to trade (default 5.0).
    - min_adv_20: Minimum 20-day average daily volume (default 5,000,000).
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection

# Repo root = parent of the ``strategies`` package directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BENCHMARK_DIR = _PROJECT_ROOT / "raw_data" / "benchmarks"


@lru_cache(maxsize=None)
def _load_benchmark_ma(path_str: str, period: int,
                       value_name: str, ma_name: str) -> pd.DataFrame:
    """Load a benchmark OHLCV file and attach a causal rolling MA of its close.

    Cached per ``(path, period, names)`` so the file is read once per process.

    Returns a frame sorted ascending by ``date`` with columns
    ``[date, value_name, ma_name]``.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"ShortOnlyBase regime data missing: {path}. Collect the benchmark "
            f"history (see config/benchmarks.json) before running this strategy."
        )
    df = pd.read_csv(path, usecols=["date", "close"])
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df[ma_name] = df["close"].rolling(period).mean()
    df = df.rename(columns={"close": value_name})
    return df[["date", value_name, ma_name]]


@lru_cache(maxsize=None)
def _load_breadth(path_str: str) -> pd.DataFrame:
    """Load the precomputed breadth series (% of universe above its 50DMA).

    Returns a frame sorted ascending by ``date`` with columns
    ``[date, breadth_pct_above_50dma]``.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"ShortOnlyBase breadth data missing: {path}. Generate it with "
            f"`python scripts/compute_breadth.py` before running this strategy."
        )
    df = pd.read_csv(path, usecols=["date", "breadth_pct_above_50dma"])
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    df = df.dropna(subset=["breadth_pct_above_50dma"])
    return df.sort_values("date").reset_index(drop=True)


class ShortOnlyBaseStrategy(BaseStrategy):
    """SHORT-only, regime-gated breakdown / overextension-fade strategy."""

    def __init__(self,
                 # Regime gate
                 benchmark_ma_period: int = 100,
                 vix_sma_period: int = 60,
                 breadth_threshold: float = 40.0,
                 # Entry
                 rsi_floor: float = 20.0,
                 rally_atr_mult: float = 2.0,
                 gap_atr_mult: float = 1.5,
                 # Stop / exit
                 atr_sl: float = 2.5,
                 time_exit: int = 10,
                 # Position sizing
                 risk_perc: float = 2.0,
                 # Universe
                 min_price: float = 5.0,
                 min_adv_20: float = 5_000_000.0,
                 # Regime data sources (plumbing - not optimizable)
                 benchmark_dir: Optional[str] = None,
                 benchmark_symbol: str = "SPX",
                 vix_symbol: str = "VIX",
                 breadth_filename: str = "BREADTH_daily.csv"):
        """Initialise the ShortOnlyBase strategy.

        Args:
            benchmark_ma_period: Length of the benchmark MA used by the regime gate.
            vix_sma_period: Length of the VIX SMA used by the regime gate.
            breadth_threshold: Entries only allowed when breadth %% is below this.
            rsi_floor: Minimum RSI_14 for the Block A breakdown entry.
            rally_atr_mult: ATR_14 multiple for the Block B 3-day rally trigger.
            gap_atr_mult: ATR_14 multiple for the Block B gap-up trigger.
            atr_sl: ATR_14 multiple for the protective stop placed above entry.
            time_exit: Maximum holding period in days before a forced cover.
            risk_perc: Percent of total equity to risk per trade.
            min_price: Minimum close price for the tradable universe.
            min_adv_20: Minimum 20-day average daily volume for the universe.
            benchmark_dir: Directory holding the benchmark/VIX/breadth CSVs.
                Defaults to ``raw_data/benchmarks``.
            benchmark_symbol: Benchmark file stem (``<symbol>_daily.csv``).
            vix_symbol: VIX file stem (``<symbol>_daily.csv``).
            breadth_filename: Breadth file name within ``benchmark_dir``.
        """
        # Store parameters BEFORE super().__init__() which calls
        # _validate_parameters().
        self.benchmark_ma_period = int(benchmark_ma_period)
        self.vix_sma_period = int(vix_sma_period)
        self.breadth_threshold = float(breadth_threshold)
        self.rsi_floor = float(rsi_floor)
        self.rally_atr_mult = float(rally_atr_mult)
        self.gap_atr_mult = float(gap_atr_mult)
        self.atr_sl = float(atr_sl)
        self.time_exit = int(time_exit)
        self.risk_perc = float(risk_perc)
        self.min_price = float(min_price)
        self.min_adv_20 = float(min_adv_20)

        bench_dir = Path(benchmark_dir) if benchmark_dir else _DEFAULT_BENCHMARK_DIR
        self._benchmark_path = str(bench_dir / f"{benchmark_symbol}_daily.csv")
        self._vix_path = str(bench_dir / f"{vix_symbol}_daily.csv")
        self._breadth_path = str(bench_dir / breadth_filename)

        super().__init__(
            benchmark_ma_period=self.benchmark_ma_period,
            vix_sma_period=self.vix_sma_period,
            breadth_threshold=self.breadth_threshold,
            rsi_floor=self.rsi_floor,
            rally_atr_mult=self.rally_atr_mult,
            gap_atr_mult=self.gap_atr_mult,
            atr_sl=self.atr_sl,
            time_exit=self.time_exit,
            risk_perc=self.risk_perc,
            min_price=self.min_price,
            min_adv_20=self.min_adv_20,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        if self.benchmark_ma_period < 1:
            raise ValueError(
                f"benchmark_ma_period must be >= 1, got {self.benchmark_ma_period}"
            )
        if self.vix_sma_period < 1:
            raise ValueError(
                f"vix_sma_period must be >= 1, got {self.vix_sma_period}"
            )
        if not 0 < self.breadth_threshold <= 100:
            raise ValueError(
                f"breadth_threshold must be in (0, 100], got {self.breadth_threshold}"
            )
        if not 0 <= self.rsi_floor <= 100:
            raise ValueError(
                f"rsi_floor must be in [0, 100], got {self.rsi_floor}"
            )
        if self.rally_atr_mult <= 0:
            raise ValueError(
                f"rally_atr_mult must be > 0, got {self.rally_atr_mult}"
            )
        if self.gap_atr_mult <= 0:
            raise ValueError(
                f"gap_atr_mult must be > 0, got {self.gap_atr_mult}"
            )
        if self.atr_sl <= 0:
            raise ValueError(f"atr_sl must be > 0, got {self.atr_sl}")
        if self.time_exit < 1:
            raise ValueError(f"time_exit must be >= 1 day, got {self.time_exit}")
        if not 0 < self.risk_perc <= 100:
            raise ValueError(
                f"risk_perc must be between 0 and 100, got {self.risk_perc}"
            )
        if self.min_price < 0:
            raise ValueError(f"min_price must be >= 0, got {self.min_price}")
        if self.min_adv_20 < 0:
            raise ValueError(f"min_adv_20 must be >= 0, got {self.min_adv_20}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def trade_direction(self) -> TradeDirection:
        """SHORT-only base version."""
        return TradeDirection.SHORT

    def required_columns(self) -> List[str]:
        return [
            "date", "close", "open", "volume",
            "atr_14_atr", "rsi_14_rsi",
            "ema_7_ema", "ema_20_ema", "ema_50_ema",
            "bbands_20_real upper band",
        ]

    # ------------------------------------------------------------------
    # Indicator preparation
    # ------------------------------------------------------------------
    def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add short aliases, causal derived series and the merged regime gate.

        All derived columns use causal operations only (``.rolling``,
        ``.pct_change``, ``.shift(n>=0)``); the regime series are merged with a
        backward as-of join so each bar sees the most recent known value.
        """
        df = data.copy()

        # Short aliases for awkward raw-data column names.
        df["atr_14"] = df["atr_14_atr"]
        df["rsi_14"] = df["rsi_14_rsi"]
        df["bb_upper"] = df["bbands_20_real upper band"]

        # Causal derived series.
        df["return_20d"] = df["close"].pct_change(20)
        df["return_60d"] = df["close"].pct_change(60)
        # Lowest close over the PRIOR 20 days (shifted so "today" is excluded);
        # a close below it is a genuine new 20-day low.
        df["lowest_close_20d"] = df["close"].rolling(20).min().shift(1)
        df["adv_20"] = df["volume"].rolling(20).mean()
        df["rally_3d"] = df["close"] - df["close"].shift(3)
        df["gap_up"] = df["open"] - df["close"].shift(1)

        df = self._merge_regime(df)
        return df

    def _merge_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge benchmark MA, VIX SMA and breadth onto the symbol by date.

        Uses ``merge_asof`` (backward) so calendar gaps between the symbol and
        the regime series never introduce look-ahead and always resolve to the
        most recent known regime value.
        """
        bench = _load_benchmark_ma(
            self._benchmark_path, self.benchmark_ma_period,
            "benchmark_close", "benchmark_ma",
        )
        vix = _load_benchmark_ma(
            self._vix_path, self.vix_sma_period,
            "vix_close", "vix_sma",
        )
        breadth = _load_breadth(self._breadth_path)

        df = df.sort_values("date").reset_index(drop=True)
        df = pd.merge_asof(df, bench, on="date", direction="backward")
        df = pd.merge_asof(df, vix, on="date", direction="backward")
        df = pd.merge_asof(df, breadth, on="date", direction="backward")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _valid(value: Optional[float]) -> bool:
        """True if a value is present and not NaN."""
        return value is not None and not (isinstance(value, float) and np.isnan(value))

    def _regime_ok(self, context: StrategyContext) -> bool:
        """Return True when all three regime conditions hold on the current bar."""
        g = context.get_indicator_value
        bench_close = g("benchmark_close")
        bench_ma = g("benchmark_ma")
        vix_close = g("vix_close")
        vix_sma = g("vix_sma")
        breadth = g("breadth_pct_above_50dma")

        for value in (bench_close, bench_ma, vix_close, vix_sma, breadth):
            if not self._valid(value):
                return False

        return (
            bench_close < bench_ma
            and breadth < self.breadth_threshold
            and vix_close > vix_sma
        )

    def _block_a(self, context: StrategyContext) -> bool:
        """Trend-breakdown entry block."""
        g = context.get_indicator_value
        ema_fast = g("ema_7_ema")   # EMA_5 proxy (no raw EMA_5 column exists)
        ema_mid = g("ema_20_ema")
        ema_slow = g("ema_50_ema")
        return_20d = g("return_20d")
        return_60d = g("return_60d")
        lowest_20d = g("lowest_close_20d")
        rsi = g("rsi_14")

        for value in (ema_fast, ema_mid, ema_slow, return_20d,
                      return_60d, lowest_20d, rsi):
            if not self._valid(value):
                return False

        return (
            ema_fast < ema_mid < ema_slow
            and return_20d < 0
            and return_60d < 0
            and context.current_price < lowest_20d
            and rsi > self.rsi_floor
        )

    def _block_b(self, context: StrategyContext) -> bool:
        """Overextension-fade entry block."""
        g = context.get_indicator_value
        atr = g("atr_14")
        bb_upper = g("bb_upper")
        if not self._valid(atr) or not self._valid(bb_upper):
            return False

        rally = g("rally_3d")
        gap = g("gap_up")
        spike = (
            (self._valid(rally) and rally > self.rally_atr_mult * atr)
            or (self._valid(gap) and gap > self.gap_atr_mult * atr)
        )
        return spike and context.current_price > bb_upper

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------
    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Short on the close when the regime gate and an entry block agree."""
        # Warmup: return_60d needs 60 prior bars; the regime MAs guard
        # themselves via NaN (handled in _regime_ok).
        if context.current_index < 60:
            return None

        close = context.current_price

        # Universe filters.
        if close < self.min_price:
            return None
        adv = context.get_indicator_value("adv_20")
        if not self._valid(adv) or adv < self.min_adv_20:
            return None

        # Regime gate.
        if not self._regime_ok(context):
            return None

        # Entry blocks.
        block_a = self._block_a(context)
        block_b = self._block_b(context)
        if not (block_a or block_b):
            return None

        stop_loss = self.calculate_initial_stop_loss(context)
        # A SHORT requires stop strictly above entry; skip if ATR is degenerate.
        if stop_loss is None or stop_loss <= close:
            return None

        reason = "trend breakdown" if block_a else "overextension fade"
        return Signal.buy(
            size=1.0,
            stop_loss=stop_loss,
            reason=f"ShortOnlyBase short ({reason})",
            direction=self.trade_direction,
        )

    # ------------------------------------------------------------------
    # Stop loss
    # ------------------------------------------------------------------
    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        """Return ``entry_price + atr_sl * ATR_14`` (stop sits above a short).

        Falls back to a 5% stop above price if ATR is unavailable.
        """
        current_price = context.current_price
        atr = context.get_indicator_value("atr_14")
        if self._valid(atr) and atr > 0:
            return current_price + (self.atr_sl * atr)
        return current_price * 1.05

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        """Risk-based sizing: risk ``risk_perc`` of equity at the stop distance.

        For a short the per-share risk is ``stop_loss - entry`` (the stop is
        above the entry), converted to the base currency via ``fx_rate``.
        """
        if signal.stop_loss is None:
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        equity = context.total_equity
        risk_amount = equity * (self.risk_perc / 100.0)

        stop_distance = signal.stop_loss - context.current_price
        if stop_distance <= 0:
            capital_to_use = context.available_capital * 0.1
            return capital_to_use / context.current_price

        stop_distance_base = stop_distance * context.fx_rate
        return risk_amount / stop_distance_base

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------
    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        """Cover on a close back above EMA_20 or after ``time_exit`` days.

        The ATR protective stop (``entry_price + atr_sl * ATR_14``) is enforced
        intrabar by the engine via the price from ``calculate_initial_stop_loss``.
        """
        if not context.has_position:
            return None

        close = context.current_price
        ema_20 = context.get_indicator_value("ema_20_ema")
        if self._valid(ema_20) and close > ema_20:
            return Signal.sell(reason="Cover: close > EMA_20")

        holding_days = context.position.duration_days(context.current_date)
        if holding_days >= self.time_exit:
            return Signal.sell(
                reason=f"Cover: holding_days >= {self.time_exit}"
            )

        return None
