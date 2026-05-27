"""
Tests for AlphaTrendV3C1Strategy.

Covers the two V3C1-specific additions on top of the inherited V2C3 behaviour:
- Parameter validation (new dropdown lengths + percent threshold).
- Required columns reflect the enabled rules.
- Entry EMA>SMA filter:
    * Confirmation bar with filter satisfied -> buy.
    * Filter not satisfied -> no buy, pending preserved.
    * Filter fails then passes within the wait window -> buy on the passing bar.
    * Strict comparison (EMA == SMA does not pass).
    * Filter disabled -> behaves like V2C3.
- EMA-distance overextension exit:
    * Fires in the short-term and long-term stages.
    * Never fires during the peace period.
    * Threshold measured as (close - EMA) / EMA * 100.
    * Disabled -> no EMA-distance exit; inherited break-even still works.
- Registration with StrategyConfig.
"""
import unittest
from typing import Sequence

import numpy as np
import pandas as pd

from strategies.alpha_trend_v3c1_strategy import AlphaTrendV3C1Strategy
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
    ``cross_index`` (with ``ma_offset=2``)."""
    df = df.copy()
    n = len(df)
    values = np.full(n, float(low))
    values[cross_index:] = float(high)
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
        s = AlphaTrendV3C1Strategy()
        self.assertTrue(s.entry_ema_sma_filter_enabled)
        self.assertEqual(s.entry_ema_length, 20)
        self.assertEqual(s.entry_sma_length, 50)
        self.assertTrue(s.ema_distance_exit_enabled)
        self.assertEqual(s.exit_ema_length, 20)
        self.assertEqual(s.exit_ema_distance_pct, 10.0)
        # Inherited V2C3 defaults still hold.
        self.assertEqual(s.high_lookback_candles, 10)
        self.assertEqual(s.entry_wait_candles, 3)
        self.assertEqual(s.trailing_atr_mult, 2.0)
        self.assertEqual(s.trade_direction, TradeDirection.LONG)

    def test_invalid_entry_ema_length(self):
        with self.assertRaises(ValueError):
            AlphaTrendV3C1Strategy(entry_ema_length=21)

    def test_invalid_entry_sma_length(self):
        with self.assertRaises(ValueError):
            AlphaTrendV3C1Strategy(entry_sma_length=13)

    def test_invalid_exit_ema_length(self):
        with self.assertRaises(ValueError):
            AlphaTrendV3C1Strategy(exit_ema_length=100)

    def test_invalid_exit_ema_distance_pct(self):
        with self.assertRaises(ValueError):
            AlphaTrendV3C1Strategy(exit_ema_distance_pct=0.0)
        with self.assertRaises(ValueError):
            AlphaTrendV3C1Strategy(exit_ema_distance_pct=-5.0)

    def test_inherited_validation_still_applies(self):
        with self.assertRaises(ValueError):
            AlphaTrendV3C1Strategy(ma_length=21)
        with self.assertRaises(ValueError):
            AlphaTrendV3C1Strategy(trailing_atr_mult=0.0)


class TestRequiredColumns(unittest.TestCase):
    def test_includes_new_columns_when_enabled(self):
        s = AlphaTrendV3C1Strategy(
            entry_ema_length=14, entry_sma_length=50, exit_ema_length=30
        )
        cols = s.required_columns()
        for c in ("date", "close", "atr_14_atr",
                  "ema_14_ema", "sma_50_sma", "ema_30_ema"):
            self.assertIn(c, cols)

    def test_excludes_new_columns_when_disabled(self):
        s = AlphaTrendV3C1Strategy(
            entry_ema_sma_filter_enabled=False,
            ema_distance_exit_enabled=False,
            entry_ema_length=14, entry_sma_length=90, exit_ema_length=30,
        )
        cols = s.required_columns()
        self.assertNotIn("ema_14_ema", cols)
        self.assertNotIn("sma_90_sma", cols)
        self.assertNotIn("ema_30_ema", cols)
        # The inherited cross column (ema_20_ema) is still required.
        self.assertIn("ema_20_ema", cols)


# ---------------------------------------------------------------------------
# Entry EMA>SMA filter
# ---------------------------------------------------------------------------

class TestEntryFilter(unittest.TestCase):
    """The filter is an AND gate on top of the V2C3 cross + n-bar-high gate."""

    def _strat(self, **kwargs):
        defaults = dict(
            ma_offset=2,
            ema_sma="EMA",
            ma_length=20,            # cross uses ema_20_ema
            high_lookback_candles=10,
            entry_wait_candles=3,
            entry_ema_length=14,     # filter uses ema_14_ema...
            entry_sma_length=50,     # ...vs sma_50_sma (distinct from cross)
            ema_distance_exit_enabled=False,
        )
        defaults.update(kwargs)
        return AlphaTrendV3C1Strategy(**defaults)

    @staticmethod
    def _ramp_closes(n):
        # Strictly increasing so every bar is a new n-bar closing high.
        return [100.0 + i for i in range(n)]

    def _prepared(self, s, n, cross_idx, ema14, sma50):
        df = _patch_ma_for_single_cross(
            _make_data(self._ramp_closes(n)), cross_idx
        )
        df = s.prepare_data(df)
        df["ema_14_ema"] = np.asarray(ema14, dtype=float)
        df["sma_50_sma"] = np.asarray(sma50, dtype=float)
        return df

    def test_buys_when_filter_satisfied(self):
        n, cross_idx = 50, 30
        s = self._strat()
        df = self._prepared(
            s, n, cross_idx,
            ema14=np.full(n, 101.0),   # EMA above SMA -> filter passes
            sma50=np.full(n, 100.0),
        )
        ctx = _make_context(df, index=cross_idx)
        signal = s.generate_entry_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("EMA14 > SMA50", signal.reason)
        self.assertNotIn("TEST", s._pending_cross)

    def test_no_buy_when_filter_fails_pending_preserved(self):
        n, cross_idx = 50, 30
        s = self._strat()
        df = self._prepared(
            s, n, cross_idx,
            ema14=np.full(n, 99.0),    # EMA below SMA -> filter fails
            sma50=np.full(n, 100.0),
        )
        ctx = _make_context(df, index=cross_idx)
        self.assertIsNone(s.generate_entry_signal(ctx))
        # The cross is still pending so a later bar can confirm.
        pending = s._pending_cross.get("TEST")
        self.assertIsNotNone(pending)
        self.assertEqual(pending[1], cross_idx)

    def test_filter_fail_then_pass_within_window_buys(self):
        n, cross_idx = 50, 30
        s = self._strat(entry_wait_candles=3)
        ema14 = np.full(n, 99.0)
        ema14[cross_idx + 2:] = 101.0   # filter starts passing at +2
        df = self._prepared(
            s, n, cross_idx, ema14=ema14, sma50=np.full(n, 100.0)
        )
        # cross bar and +1: filter fails -> no buy, pending kept alive.
        self.assertIsNone(s.generate_entry_signal(_make_context(df, cross_idx)))
        self.assertIsNone(s.generate_entry_signal(_make_context(df, cross_idx + 1)))
        self.assertEqual(s._pending_cross["TEST"][1], cross_idx)
        # +2: filter passes -> buy.
        signal = s.generate_entry_signal(_make_context(df, cross_idx + 2))
        self.assertIsNotNone(signal)
        self.assertNotIn("TEST", s._pending_cross)

    def test_filter_requires_strictly_greater(self):
        n, cross_idx = 50, 30
        s = self._strat()
        df = self._prepared(
            s, n, cross_idx,
            ema14=np.full(n, 100.0),   # EMA == SMA -> not strictly greater
            sma50=np.full(n, 100.0),
        )
        self.assertIsNone(s.generate_entry_signal(_make_context(df, cross_idx)))

    def test_disabled_filter_behaves_like_v2c3(self):
        n, cross_idx = 50, 30
        s = self._strat(entry_ema_sma_filter_enabled=False)
        df = self._prepared(
            s, n, cross_idx,
            ema14=np.full(n, 99.0),    # would fail if the filter were on
            sma50=np.full(n, 100.0),
        )
        signal = s.generate_entry_signal(_make_context(df, cross_idx))
        self.assertIsNotNone(signal)
        self.assertNotIn("EMA14 > SMA50", signal.reason)


# ---------------------------------------------------------------------------
# EMA-distance overextension exit
# ---------------------------------------------------------------------------

class TestEmaDistanceExit(unittest.TestCase):
    def _strat(self, **kwargs):
        defaults = dict(
            exit_ema_length=20,        # exit uses ema_20_ema
            entry_ema_sma_filter_enabled=False,
        )
        defaults.update(kwargs)
        return AlphaTrendV3C1Strategy(**defaults)

    def _prepared(self, s, n, close=110.0, exit_ema=100.0, atr=2.0):
        df = s.prepare_data(_make_data([float(close)] * n, atr=atr))
        df["ema_20_ema"] = np.full(n, float(exit_ema))
        return df

    def test_fires_in_short_stage(self):
        n = 60
        # distance = (110 - 100) / 100 * 100 = 10% >= 5% threshold.
        s = self._strat(
            peace_period_candles=0, short_term_candles=10,
            exit_ema_distance_pct=5.0,
        )
        df = self._prepared(s, n, close=110.0, exit_ema=100.0)
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=110.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        signal = s.generate_exit_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("EMA-distance", signal.reason)

    def test_fires_in_long_stage(self):
        n = 60
        s = self._strat(
            peace_period_candles=0, short_term_candles=5,
            exit_ema_distance_pct=5.0,
        )
        df = self._prepared(s, n, close=110.0, exit_ema=100.0)
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=110.0)
        # bars_since_entry = 10 -> long stage (peace 0 + short 5 = 5).
        ctx = _make_context(df, index=entry_index + 10, current_position=pos)
        self.assertIsNotNone(s.generate_exit_signal(ctx))

    def test_does_not_fire_during_peace(self):
        n = 60
        s = self._strat(
            peace_period_candles=5, short_term_candles=10,
            exit_ema_distance_pct=5.0,
        )
        df = self._prepared(s, n, close=110.0, exit_ema=100.0)
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=110.0)
        # bars_since_entry = 2 -> still in the peace period.
        ctx = _make_context(df, index=entry_index + 2, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_does_not_fire_below_threshold(self):
        n = 60
        # distance = 10% < 15% threshold -> no exit.
        s = self._strat(
            peace_period_candles=0, short_term_candles=10,
            exit_ema_distance_pct=15.0,
        )
        df = self._prepared(s, n, close=110.0, exit_ema=100.0)
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=110.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_threshold_measured_relative_to_ema(self):
        n = 60
        # close=120, ema=100 -> (120-100)/100*100 = 20%. Fires at 20, not 21.
        s_fire = self._strat(
            peace_period_candles=0, short_term_candles=10,
            exit_ema_distance_pct=20.0,
        )
        df = self._prepared(s_fire, n, close=120.0, exit_ema=100.0)
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=120.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        self.assertIsNotNone(s_fire.generate_exit_signal(ctx))

        s_no = self._strat(
            peace_period_candles=0, short_term_candles=10,
            exit_ema_distance_pct=21.0,
        )
        df2 = self._prepared(s_no, n, close=120.0, exit_ema=100.0)
        pos2 = _open_position_at(df2, entry_index, price=120.0)
        ctx2 = _make_context(df2, index=entry_index + 1, current_position=pos2)
        self.assertIsNone(s_no.generate_exit_signal(ctx2))

    def test_disabled_exit_does_not_fire(self):
        n = 60
        s = self._strat(
            ema_distance_exit_enabled=False,
            peace_period_candles=0, short_term_candles=10,
            short_phase_exit_enabled=False,
            exit_ema_distance_pct=5.0,
        )
        df = self._prepared(s, n, close=110.0, exit_ema=100.0)
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=110.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_inherited_break_even_still_fires(self):
        """With the EMA-distance exit off, V2C2's short-phase rule still works."""
        n = 60
        s = AlphaTrendV3C1Strategy(
            entry_ema_sma_filter_enabled=False,
            ema_distance_exit_enabled=False,
            peace_period_candles=2,
            short_term_candles=10,
            short_phase_exit_enabled=True,
            break_even_consecutive_bars=3,
        )
        closes = [100.0] * 20 + [90.0] * (n - 20)
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0)
        ctx = _make_context(df, index=entry_index + 3, current_position=pos)
        signal = s.generate_exit_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("below entry", signal.reason)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):
    def setUp(self):
        StrategyConfig.reload()

    def test_registered_in_strategy_config(self):
        self.assertIn("AlphaTrendV3C1Strategy", StrategyConfig.get_strategies())

    def test_has_v3c1_parameters(self):
        params = StrategyConfig.get_parameters("AlphaTrendV3C1Strategy")
        for name in ("entry_ema_sma_filter_enabled", "entry_ema_length",
                     "entry_sma_length", "ema_distance_exit_enabled",
                     "exit_ema_length", "exit_ema_distance_pct"):
            self.assertIn(name, params)
        # Inherited V2C3 params still present.
        for name in ("ma_offset", "ma_length", "high_lookback_candles",
                     "entry_wait_candles", "trailing_atr_mult"):
            self.assertIn(name, params)

    def test_ma_length_params_are_dropdowns(self):
        supported = [7, 14, 20, 30, 50, 90, 200]
        for name in ("entry_ema_length", "entry_sma_length", "exit_ema_length"):
            info = StrategyConfig.get_param_info("AlphaTrendV3C1Strategy", name)
            self.assertEqual(info.get("allowed_values"), supported)

    def test_defaults_match_strategy(self):
        defaults = StrategyConfig.get_defaults("AlphaTrendV3C1Strategy")
        self.assertEqual(defaults["entry_ema_length"], 20)
        self.assertEqual(defaults["entry_sma_length"], 50)
        self.assertEqual(defaults["exit_ema_length"], 20)
        self.assertEqual(defaults["exit_ema_distance_pct"], 10.0)
        self.assertTrue(defaults["entry_ema_sma_filter_enabled"])
        self.assertTrue(defaults["ema_distance_exit_enabled"])


if __name__ == "__main__":
    unittest.main()
