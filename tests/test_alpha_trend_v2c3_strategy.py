"""
Tests for AlphaTrendV2C3Strategy.

Covers:
- Parameter validation (V2C3-specific + inherited V2C2)
- Required columns
- Entry rule:
    * Cross + high on cross bar -> buy on cross bar
    * Cross without high on cross bar -> no buy, pending
    * High confirmation within wait window -> buy on confirming bar
    * Window expiry without high -> no buy
    * Strictly-greater rule (equal-to-max does not confirm)
    * No high without a prior cross -> no buy
    * New cross resets the wait window cache
- Exit logic still inherits the V2C2 short-phase rule
- Registration with StrategyConfig
"""
import unittest
from typing import Sequence

import numpy as np
import pandas as pd

from strategies.alpha_trend_v2c3_strategy import AlphaTrendV2C3Strategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.position import Position
from Classes.Models.trade_direction import TradeDirection
from config.strategy_config import StrategyConfig


def _make_data(
    closes,
    atr=2.0,
    start_date="2020-01-01",
    ma_lengths: Sequence[int] = (7, 14, 20, 30, 50, 90, 200),
):
    n = len(closes)
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    closes = np.asarray(closes, dtype=float)
    data = {
        "date": dates,
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(n, 1_000_000.0),
        "atr_14_atr": np.full(n, float(atr)),
    }
    series = pd.Series(closes)
    for length in ma_lengths:
        data[f"ema_{length}_ema"] = series.ewm(span=length, adjust=False).mean().values
        data[f"sma_{length}_sma"] = series.rolling(window=length, min_periods=length).mean().values
    return pd.DataFrame(data)


def _patch_ma_for_single_cross(df, cross_index, ma_col="ema_20_ema",
                                low=10.0, high=11.0):
    """Patch ``ma_col`` so a single AlphaMACross long signal fires at
    ``cross_index`` (with ``ma_offset=2``). Values are flat at ``low`` before
    and flat at ``high`` from ``cross_index`` onwards.
    """
    df = df.copy()
    n = len(df)
    values = np.full(n, float(low))
    values[cross_index:] = float(high)
    df[ma_col] = values
    return df


def _patch_ma_for_two_crosses(df, cross1_index, cross2_index,
                               ma_col="ema_20_ema"):
    """Patch ``ma_col`` so AlphaMACross fires at ``cross1_index`` and again at
    ``cross2_index``. Requires ``cross2_index - cross1_index >= 4`` so the MA
    can dip and rise again cleanly.
    """
    if cross2_index - cross1_index < 4:
        raise ValueError("crosses must be at least 4 bars apart")
    df = df.copy()
    n = len(df)
    values = np.full(n, 10.0)
    # First plateau: 11 from cross1 up to (cross2 - 3)
    values[cross1_index:cross2_index - 3] = 11.0
    # Three bars at 10 to reset the comparison
    values[cross2_index - 3:cross2_index] = 10.0
    # Second plateau: 12 from cross2 onwards
    values[cross2_index:] = 12.0
    df[ma_col] = values
    return df


def _make_context(df, index, current_position=None, equity=100_000.0,
                  capital=100_000.0, fx_rate=1.0, symbol="TEST"):
    return StrategyContext(
        data=df,
        current_index=index,
        current_price=float(df["close"].iloc[index]),
        current_date=pd.Timestamp(df["date"].iloc[index]),
        position=current_position,
        available_capital=capital,
        total_equity=equity,
        symbol=symbol,
        fx_rate=fx_rate,
    )


def _open_position_at(df, entry_index, stop=None, price=None, size=10.0,
                      symbol="TEST"):
    entry_date = pd.Timestamp(df["date"].iloc[entry_index])
    if price is None:
        price = float(df["close"].iloc[entry_index])
    if stop is None:
        stop = price * 0.9
    return Position(
        symbol=symbol,
        entry_date=entry_date,
        entry_price=float(price),
        initial_quantity=float(size),
        current_quantity=float(size),
        stop_loss=float(stop),
        direction=TradeDirection.LONG,
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation(unittest.TestCase):
    def test_default_construction(self):
        s = AlphaTrendV2C3Strategy()
        self.assertEqual(s.high_lookback_candles, 10)
        self.assertEqual(s.entry_wait_candles, 3)
        # V2C3 replaces V2C2's trailing_sl_pct with an ATR-based multiplier.
        self.assertEqual(s.trailing_atr_mult, 2.0)
        # V2C2 defaults still hold for the unchanged params.
        self.assertEqual(s.peace_period_candles, 5)
        self.assertEqual(s.short_term_candles, 15)
        self.assertEqual(s.break_even_consecutive_bars, 7)
        self.assertEqual(s.trade_direction, TradeDirection.LONG)

    def test_invalid_high_lookback_candles(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C3Strategy(high_lookback_candles=0)
        with self.assertRaises(ValueError):
            AlphaTrendV2C3Strategy(high_lookback_candles=-3)

    def test_invalid_entry_wait_candles(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C3Strategy(entry_wait_candles=-1)

    def test_zero_wait_window_is_valid(self):
        # entry_wait_candles=0 means high must be reached on the cross bar.
        s = AlphaTrendV2C3Strategy(entry_wait_candles=0)
        self.assertEqual(s.entry_wait_candles, 0)

    def test_invalid_trailing_atr_mult(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C3Strategy(trailing_atr_mult=0.0)
        with self.assertRaises(ValueError):
            AlphaTrendV2C3Strategy(trailing_atr_mult=-1.0)

    def test_inherited_validation_still_applies(self):
        # ma_length must be one of the supported values.
        with self.assertRaises(ValueError):
            AlphaTrendV2C3Strategy(ma_length=21)
        # atr_sl must be > 0 (inherited from V2C2).
        with self.assertRaises(ValueError):
            AlphaTrendV2C3Strategy(atr_sl=0.0)


class TestRequiredColumns(unittest.TestCase):
    def test_required_columns(self):
        s = AlphaTrendV2C3Strategy()
        cols = s.required_columns()
        for c in ("date", "close", "atr_14_atr", "ema_20_ema"):
            self.assertIn(c, cols)


# ---------------------------------------------------------------------------
# Entry rule
# ---------------------------------------------------------------------------

class TestEntryConfirmation(unittest.TestCase):
    """Tests for the cross + n-bar-high confirmation entry gate."""

    def _strat(self, **kwargs):
        defaults = dict(
            ma_offset=2,
            ema_sma="EMA",
            ma_length=20,
            high_lookback_candles=10,
            entry_wait_candles=3,
        )
        defaults.update(kwargs)
        return AlphaTrendV2C3Strategy(**defaults)

    def test_buys_when_cross_and_high_coincide(self):
        """Cross fires on a bar whose close is a new 10-bar high -> buy."""
        n = 50
        cross_idx = 30
        # Closes flat at 100 except a spike to 105 on the cross bar.
        closes = [100.0] * n
        closes[cross_idx] = 105.0
        s = self._strat()
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), cross_idx))
        ctx = _make_context(df, index=cross_idx)
        signal = s.generate_entry_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("AlphaMACross", signal.reason)
        # Pending must be cleared on a confirmed entry.
        self.assertNotIn("TEST", s._pending_cross)

    def test_no_buy_when_cross_without_high_on_cross_bar(self):
        """Cross fires but close is not a new high -> no buy, pending state."""
        n = 50
        cross_idx = 30
        closes = [100.0] * n  # never breaks the prior 10-bar high.
        s = self._strat()
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), cross_idx))
        ctx = _make_context(df, index=cross_idx)
        signal = s.generate_entry_signal(ctx)
        self.assertIsNone(signal)
        # Pending should now hold the cross bar.
        pending = s._pending_cross.get("TEST")
        self.assertIsNotNone(pending)
        _, cached_idx = pending
        self.assertEqual(cached_idx, cross_idx)

    def test_buys_within_wait_window_on_confirmation(self):
        """Cross fails on cross bar, but a new high within the next x bars
        triggers the buy on that bar."""
        n = 50
        cross_idx = 30
        x = 3
        closes = [100.0] * n
        # Spike to 110 at cross_idx + 2 (still within window 1..3).
        confirm_idx = cross_idx + 2
        closes[confirm_idx] = 110.0
        s = self._strat(entry_wait_candles=x)
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), cross_idx))

        # Walk bar by bar: cross_idx (no buy), cross_idx+1 (no buy),
        # cross_idx+2 (buy).
        ctx0 = _make_context(df, index=cross_idx)
        self.assertIsNone(s.generate_entry_signal(ctx0))
        ctx1 = _make_context(df, index=cross_idx + 1)
        self.assertIsNone(s.generate_entry_signal(ctx1))
        ctx2 = _make_context(df, index=confirm_idx)
        signal = s.generate_entry_signal(ctx2)
        self.assertIsNotNone(signal)
        self.assertNotIn("TEST", s._pending_cross)

    def test_buys_at_window_edge(self):
        """Confirmation on the very last bar of the window still buys."""
        n = 50
        cross_idx = 30
        x = 3
        closes = [100.0] * n
        confirm_idx = cross_idx + x  # last bar of window
        closes[confirm_idx] = 110.0
        s = self._strat(entry_wait_candles=x)
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), cross_idx))

        for offset in range(x):
            ctx = _make_context(df, index=cross_idx + offset)
            self.assertIsNone(s.generate_entry_signal(ctx),
                              f"unexpected buy at offset {offset}")
        ctx_edge = _make_context(df, index=confirm_idx)
        self.assertIsNotNone(s.generate_entry_signal(ctx_edge))

    def test_no_buy_after_window_expires(self):
        """A high that arrives one bar past the window is ignored."""
        n = 50
        cross_idx = 30
        x = 3
        closes = [100.0] * n
        # Set a new high at cross_idx + x + 1 (one bar past the window).
        late_idx = cross_idx + x + 1
        closes[late_idx] = 110.0
        s = self._strat(entry_wait_candles=x)
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), cross_idx))

        # Walk through the entire window: no buy.
        for offset in range(x + 1):
            ctx = _make_context(df, index=cross_idx + offset)
            self.assertIsNone(s.generate_entry_signal(ctx),
                              f"unexpected buy at offset {offset}")
        # One bar later: window has expired, no buy even with the high.
        ctx_late = _make_context(df, index=late_idx)
        self.assertIsNone(s.generate_entry_signal(ctx_late))
        self.assertNotIn("TEST", s._pending_cross)

    def test_high_must_be_strictly_greater(self):
        """Close equal to the max of the prior n closes does NOT confirm."""
        n = 50
        cross_idx = 30
        # Prior 10 closes (cross_idx-10..cross_idx-1) max = 100; close on
        # cross_idx = 100 -> not strictly greater -> no buy.
        closes = [100.0] * n
        s = self._strat()
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), cross_idx))
        ctx = _make_context(df, index=cross_idx)
        self.assertIsNone(s.generate_entry_signal(ctx))

    def test_no_buy_without_a_prior_cross(self):
        """A new n-bar high alone, with no cross, must not trigger a buy."""
        n = 50
        idx = 30
        closes = [100.0] * n
        closes[idx] = 110.0  # new high but no MA cross
        s = self._strat()
        # MA flat at 10 throughout -> no cross is ever detected.
        raw = _make_data(closes)
        raw["ema_20_ema"] = np.full(n, 10.0)
        df = s.prepare_data(raw)
        ctx = _make_context(df, index=idx)
        self.assertIsNone(s.generate_entry_signal(ctx))
        self.assertNotIn("TEST", s._pending_cross)

    def test_zero_wait_window_requires_same_bar_confirmation(self):
        """With entry_wait_candles=0, a high must arrive on the cross bar."""
        n = 50
        cross_idx = 30
        # No high on cross bar; high one bar later (would be too late).
        closes = [100.0] * n
        closes[cross_idx + 1] = 110.0
        s = self._strat(entry_wait_candles=0)
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), cross_idx))

        ctx0 = _make_context(df, index=cross_idx)
        self.assertIsNone(s.generate_entry_signal(ctx0))
        ctx1 = _make_context(df, index=cross_idx + 1)
        self.assertIsNone(s.generate_entry_signal(ctx1))

    def test_new_cross_resets_window(self):
        """A second cross while pending restarts the wait window from the
        new cross bar."""
        n = 60
        cross1 = 25
        cross2 = 35  # 10 bars later, well past the original window
        x = 3
        s = self._strat(entry_wait_candles=x)
        # Closes flat (no high anywhere) so the only thing that can change
        # the pending state is a new cross.
        closes = [100.0] * n
        df = s.prepare_data(_patch_ma_for_two_crosses(_make_data(closes),
                                                      cross1, cross2))

        # First cross fires -> pending at cross1.
        ctx_c1 = _make_context(df, index=cross1)
        self.assertIsNone(s.generate_entry_signal(ctx_c1))
        self.assertEqual(s._pending_cross["TEST"][1], cross1)

        # Walk through original window without any high -> window expires.
        for offset in range(1, x + 2):
            ctx = _make_context(df, index=cross1 + offset)
            self.assertIsNone(s.generate_entry_signal(ctx))
        # After the original window expires, pending should be cleared.
        self.assertNotIn("TEST", s._pending_cross)

        # Second cross fires -> pending now anchored at cross2.
        ctx_c2 = _make_context(df, index=cross2)
        self.assertIsNone(s.generate_entry_signal(ctx_c2))
        self.assertEqual(s._pending_cross["TEST"][1], cross2)

    def test_new_cross_overwrites_pending_within_window(self):
        """If a new cross fires while pending is still active, the cache is
        overwritten with the new cross bar (extending the window)."""
        n = 60
        cross1 = 28
        cross2 = 32  # within original cross1's window (cross1 + 4)
        x = 5
        s = self._strat(entry_wait_candles=x)
        closes = [100.0] * n
        df = s.prepare_data(_patch_ma_for_two_crosses(_make_data(closes),
                                                      cross1, cross2))

        # First cross -> pending.
        ctx_c1 = _make_context(df, index=cross1)
        self.assertIsNone(s.generate_entry_signal(ctx_c1))
        self.assertEqual(s._pending_cross["TEST"][1], cross1)

        # Bars between cross1 and cross2 (no new cross, no high) -> still
        # pending at cross1.
        for offset in range(1, cross2 - cross1):
            ctx = _make_context(df, index=cross1 + offset)
            self.assertIsNone(s.generate_entry_signal(ctx))
            self.assertEqual(s._pending_cross["TEST"][1], cross1)

        # Second cross -> pending re-anchored.
        ctx_c2 = _make_context(df, index=cross2)
        self.assertIsNone(s.generate_entry_signal(ctx_c2))
        self.assertEqual(s._pending_cross["TEST"][1], cross2)

    def test_insufficient_history_no_buy(self):
        """If current_index < high_lookback_candles, can't evaluate the high."""
        n = 30
        s = self._strat(high_lookback_candles=10)
        # Set up a "cross" right at index 5 (less than 10 bars of history).
        closes = [100.0] * n
        closes[5] = 200.0
        df = s.prepare_data(_patch_ma_for_single_cross(_make_data(closes), 5))
        ctx = _make_context(df, index=5)
        self.assertIsNone(s.generate_entry_signal(ctx))


# ---------------------------------------------------------------------------
# Exit logic still inherits V2C2
# ---------------------------------------------------------------------------

class TestExitInheritance(unittest.TestCase):
    def test_break_even_consecutive_bars_exit_still_fires(self):
        """V2C3 keeps V2C2's short-phase break-even exit unchanged."""
        n = 60
        s = AlphaTrendV2C3Strategy(
            peace_period_candles=2,
            short_term_candles=10,
            short_phase_exit_enabled=True,
            break_even_consecutive_bars=3,
        )
        closes = [100.0] * 20 + [90.0] * (n - 20)
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0)

        # Offset 3: 3 consecutive below-entry closes -> short-phase exit fires.
        ctx = _make_context(df, index=entry_index + 3, current_position=pos)
        signal = s.generate_exit_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("below entry", signal.reason)

    def test_trailing_sl_uses_atr_multiple(self):
        """Long-term trailing SL is computed as close - trailing_atr_mult * ATR_14."""
        n = 60
        s = AlphaTrendV2C3Strategy(
            peace_period_candles=0,
            short_term_candles=0,
            trailing_atr_mult=2.0,
        )
        # ATR fixed at 2.0 in _make_data; close=100 -> new_stop = 100 - 2*2 = 96.
        df = s.prepare_data(_make_data([100.0] * n, atr=2.0))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=70.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        new_sl = s.should_adjust_stop(ctx)
        self.assertIsNotNone(new_sl)
        self.assertAlmostEqual(new_sl, 96.0, places=6)

    def test_trailing_sl_only_moves_up(self):
        """The trailing SL must not move down: a tighter existing stop wins."""
        n = 60
        s = AlphaTrendV2C3Strategy(
            peace_period_candles=0,
            short_term_candles=0,
            trailing_atr_mult=2.0,
        )
        df = s.prepare_data(_make_data([100.0] * n, atr=2.0))
        entry_index = 20
        # Existing stop already above the would-be new stop (96.0).
        pos = _open_position_at(df, entry_index, stop=98.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        new_sl = s.should_adjust_stop(ctx)
        self.assertIsNone(new_sl)

    def test_trailing_sl_inactive_outside_long_stage(self):
        """Trailing SL must not adjust during peace or short-term stages."""
        n = 60
        s = AlphaTrendV2C3Strategy(
            peace_period_candles=5,
            short_term_candles=10,
            trailing_atr_mult=2.0,
        )
        df = s.prepare_data(_make_data([100.0] * n, atr=2.0))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=70.0)
        # Peace stage (bars_since_entry=2): no adjust.
        ctx_peace = _make_context(df, index=entry_index + 2, current_position=pos)
        self.assertIsNone(s.should_adjust_stop(ctx_peace))
        # Short stage (bars_since_entry=10): no adjust.
        ctx_short = _make_context(df, index=entry_index + 10, current_position=pos)
        self.assertIsNone(s.should_adjust_stop(ctx_short))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):
    def setUp(self):
        # Force re-load so this test reflects the current JSON file.
        StrategyConfig.reload()

    def test_registered_in_strategy_config(self):
        strategies = StrategyConfig.get_strategies()
        self.assertIn("AlphaTrendV2C3Strategy", strategies)

    def test_has_v2c3_parameters(self):
        params = StrategyConfig.get_parameters("AlphaTrendV2C3Strategy")
        # New params present (trailing_atr_mult replaces V2C2's trailing_sl_pct).
        for name in ("high_lookback_candles", "entry_wait_candles",
                     "trailing_atr_mult"):
            self.assertIn(name, params)
        # Inherited V2C2 params still present.
        for name in ("ma_offset", "ma_length", "atr_sl", "risk_perc",
                     "peace_period_candles", "short_term_candles",
                     "short_phase_exit_enabled",
                     "break_even_consecutive_bars"):
            self.assertIn(name, params)
        # The replaced V2C2 parameter must not appear in V2C3.
        self.assertNotIn("trailing_sl_pct", params)

    def test_defaults_match_strategy(self):
        defaults = StrategyConfig.get_defaults("AlphaTrendV2C3Strategy")
        self.assertEqual(defaults["high_lookback_candles"], 10)
        self.assertEqual(defaults["entry_wait_candles"], 3)
        self.assertEqual(defaults["trailing_atr_mult"], 2.0)


if __name__ == "__main__":
    unittest.main()
