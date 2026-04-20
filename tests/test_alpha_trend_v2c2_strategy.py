"""
Tests for AlphaTrendV2C2Strategy.

Covers:
- Parameter validation
- Required columns
- Stage classification by bars since entry
- Peace-period: no strategy exit
- Short-term: break-even consecutive-bars rule with configurable toggle
- Long-term: no strategy exit beyond trailing SL
- Long-term trailing SL: moves up only, tracks close at trailing_sl_pct
- Registration with StrategyConfig
"""
import unittest
from typing import Sequence

import numpy as np
import pandas as pd

from strategies.alpha_trend_v2c2_strategy import AlphaTrendV2C2Strategy
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
        s = AlphaTrendV2C2Strategy()
        self.assertEqual(s.peace_period_candles, 5)
        self.assertEqual(s.short_term_candles, 15)
        self.assertTrue(s.short_phase_exit_enabled)
        self.assertEqual(s.break_even_consecutive_bars, 7)
        self.assertEqual(s.trailing_sl_pct, 20.0)
        self.assertEqual(s.trade_direction, TradeDirection.LONG)

    def test_invalid_stage_lengths(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(peace_period_candles=-1)
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(short_term_candles=-1)

    def test_invalid_trailing_sl_pct(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(trailing_sl_pct=0.0)
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(trailing_sl_pct=100.0)

    def test_invalid_break_even_consecutive_bars(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(break_even_consecutive_bars=0)

    def test_invalid_ma_length(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(ma_length=21)


class TestRequiredColumns(unittest.TestCase):
    def test_required_columns(self):
        s = AlphaTrendV2C2Strategy()
        cols = s.required_columns()
        for c in ("date", "close", "atr_14_atr", "ema_20_ema"):
            self.assertIn(c, cols)
        # ADX / MACD / MFI are no longer required.
        self.assertNotIn("adx_14_adx", cols)
        self.assertNotIn("macd_14_macd_signal", cols)
        self.assertNotIn("mfi_14_mfi", cols)


# ---------------------------------------------------------------------------
# Peace-period
# ---------------------------------------------------------------------------

class TestPeacePeriod(unittest.TestCase):
    def test_no_exit_during_peace_period(self):
        """Even with price below entry every bar, peace-period returns no exit."""
        n = 60
        s = AlphaTrendV2C2Strategy(
            peace_period_candles=5,
            short_term_candles=15,
            short_phase_exit_enabled=True,
            break_even_consecutive_bars=3,
        )
        # Close stays below entry the whole time.
        closes = [100.0] * 20 + [90.0] * (n - 20)
        df = s.prepare_data(_make_data(closes))

        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0)

        # Bars since entry: 0, 1, 2, 3, 4 -> all peace-period
        for offset in range(5):
            ctx = _make_context(df, index=entry_index + offset,
                                current_position=pos)
            self.assertIsNone(
                s.generate_exit_signal(ctx),
                f"unexpected exit at offset {offset}"
            )


# ---------------------------------------------------------------------------
# Short-term: break-even consecutive-bars rule
# ---------------------------------------------------------------------------

class TestShortTerm(unittest.TestCase):
    def _strat(self, **kwargs):
        defaults = dict(
            peace_period_candles=2,
            short_term_candles=10,
            short_phase_exit_enabled=True,
            break_even_consecutive_bars=3,
        )
        defaults.update(kwargs)
        return AlphaTrendV2C2Strategy(**defaults)

    def test_fires_when_close_below_entry_for_n_bars(self):
        n = 60
        s = self._strat(break_even_consecutive_bars=3)
        # Closes: flat 100 before entry; post-entry all 90 (below entry=100).
        closes = [100.0] * 20 + [90.0] * (n - 20)
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0)

        # Offset 2 is first short-term bar. bars_since_entry=2, need 3
        # below-entry bars (current + 2 prior, scanning back to entry bar + 1).
        # Prior bars are offsets 1 and 2 after entry (both 90), plus current.
        # But the rule only looks at bars after entry -> at offset 2 only 2
        # bars are available so the rule cannot fire.
        ctx = _make_context(df, index=entry_index + 2, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

        # Offset 3: 3 consecutive below-entry closes -> fires.
        ctx = _make_context(df, index=entry_index + 3, current_position=pos)
        signal = s.generate_exit_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("below entry", signal.reason)

    def test_does_not_fire_when_any_bar_at_or_above_entry(self):
        n = 60
        s = self._strat(break_even_consecutive_bars=3)
        # Post-entry closes: 90, 90, 100, 90, 90, 90, ...
        # One bar equal to entry breaks the 3-bar streak at offsets 3 and 4.
        post = [90.0, 90.0, 100.0, 90.0, 90.0] + [90.0] * (n - 25)
        closes = [100.0] * 20 + post
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0)

        # Offset 3: last 3 closes are 90, 100, 90 -> streak broken by 100.
        ctx = _make_context(df, index=entry_index + 3, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

        # Offset 4: last 3 closes are 100, 90, 90 -> streak broken by 100.
        ctx = _make_context(df, index=entry_index + 4, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

        # Offset 5: last 3 closes are 90, 90, 90 -> fires.
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        self.assertIsNotNone(s.generate_exit_signal(ctx))

    def test_disabled_flag_suppresses_rule(self):
        n = 60
        s = self._strat(short_phase_exit_enabled=False,
                        break_even_consecutive_bars=3)
        closes = [100.0] * 20 + [90.0] * (n - 20)
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0)

        # Anywhere in short-term: even with many bars below entry, no exit.
        for offset in (3, 5, 8, 11):
            ctx = _make_context(df, index=entry_index + offset,
                                current_position=pos)
            self.assertIsNone(s.generate_exit_signal(ctx),
                              f"unexpected exit at offset {offset}")

    def test_close_at_entry_does_not_count_as_below(self):
        n = 60
        s = self._strat(break_even_consecutive_bars=3)
        # All closes exactly at entry price -> rule must not fire.
        closes = [100.0] * n
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0)
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))


# ---------------------------------------------------------------------------
# Long-term: no strategy exit beyond trailing SL
# ---------------------------------------------------------------------------

class TestLongTerm(unittest.TestCase):
    def test_no_strategy_exit_in_long_term(self):
        n = 60
        s = AlphaTrendV2C2Strategy(
            peace_period_candles=2,
            short_term_candles=3,
            short_phase_exit_enabled=True,
            break_even_consecutive_bars=3,
        )
        # Closes way below entry even after short-term. Once in long-term
        # the short-phase rule no longer applies -> no strategy exit.
        closes = [100.0] * 20 + [80.0] * (n - 20)
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0, stop=70.0)

        # Offset 5: bars_since_entry=5 -> stage = long (peace=2 + short=3).
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))


# ---------------------------------------------------------------------------
# Long-term trailing SL
# ---------------------------------------------------------------------------

class TestLongTermTrailingSL(unittest.TestCase):
    def _strat(self, **kwargs):
        defaults = dict(
            peace_period_candles=0,
            short_term_candles=0,
            trailing_sl_pct=20.0,
        )
        defaults.update(kwargs)
        return AlphaTrendV2C2Strategy(**defaults)

    def test_no_sl_move_outside_long_term(self):
        n = 60
        s = AlphaTrendV2C2Strategy(
            peace_period_candles=5,
            short_term_candles=10,
            trailing_sl_pct=20.0,
        )
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=70.0)
        # Offset 3: peace. Offset 10: short. Both should NOT move SL.
        for offset in (3, 10):
            ctx = _make_context(df, index=entry_index + offset,
                                current_position=pos)
            self.assertIsNone(s.should_adjust_stop(ctx),
                              f"unexpected SL adjust at offset {offset}")

    def test_trailing_sl_tightens_when_room_exists(self):
        """SL computed at trailing_sl_pct below close; applies when tighter."""
        n = 60
        s = self._strat(trailing_sl_pct=20.0)
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        # Current SL at 70 (30% below) -> computed SL=80 tightens.
        pos = _open_position_at(df, entry_index, stop=70.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        new_sl = s.should_adjust_stop(ctx)
        self.assertIsNotNone(new_sl)
        self.assertAlmostEqual(new_sl, 80.0, places=6)

    def test_sl_never_widens(self):
        """If computed SL is below current SL, no move happens."""
        n = 60
        s = self._strat(trailing_sl_pct=20.0)
        # Current SL at 95 -> computed SL=80 is looser, so no move.
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=95.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        self.assertIsNone(s.should_adjust_stop(ctx))

    def test_sl_follows_rising_price(self):
        """As close rises, trailing SL moves up accordingly."""
        n = 60
        s = self._strat(trailing_sl_pct=20.0)
        # Prices rise from 100 to 120.
        closes = list(np.linspace(100.0, 120.0, n))
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        # Start with a deep SL so all trailing moves are accepted.
        pos = _open_position_at(df, entry_index, price=closes[entry_index],
                                stop=50.0)

        # Offset 1: close = closes[21] -> SL = close * 0.8
        ctx1 = _make_context(df, index=entry_index + 1, current_position=pos)
        sl1 = s.should_adjust_stop(ctx1)
        self.assertAlmostEqual(sl1, closes[entry_index + 1] * 0.8, places=6)
        pos.stop_loss = sl1

        # Offset 5: price is higher -> SL should be higher than sl1.
        ctx5 = _make_context(df, index=entry_index + 5, current_position=pos)
        sl5 = s.should_adjust_stop(ctx5)
        self.assertAlmostEqual(sl5, closes[entry_index + 5] * 0.8, places=6)
        self.assertGreater(sl5, sl1)

    def test_sl_does_not_move_when_price_falls(self):
        """After a tighter SL is set, a later lower price does not move SL down."""
        n = 60
        s = self._strat(trailing_sl_pct=20.0)
        # Price up then down.
        closes = [100.0] * 20 + [120.0] * 5 + [110.0] * (n - 25)
        df = s.prepare_data(_make_data(closes))
        entry_index = 20
        pos = _open_position_at(df, entry_index, price=100.0, stop=50.0)

        # Offset 1 at price 120 -> SL = 96. Apply.
        ctx1 = _make_context(df, index=entry_index + 1, current_position=pos)
        sl1 = s.should_adjust_stop(ctx1)
        self.assertAlmostEqual(sl1, 96.0, places=6)
        pos.stop_loss = sl1

        # Offset 6 at price 110 -> computed SL = 88 which is < 96 -> no move.
        ctx6 = _make_context(df, index=entry_index + 6, current_position=pos)
        self.assertIsNone(s.should_adjust_stop(ctx6))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestMisc(unittest.TestCase):
    def test_no_exit_without_position(self):
        n = 60
        s = AlphaTrendV2C2Strategy()
        df = s.prepare_data(_make_data([100.0] * n))
        ctx = _make_context(df, index=30, current_position=None)
        self.assertIsNone(s.generate_exit_signal(ctx))
        self.assertIsNone(s.should_adjust_stop(ctx))


class TestRegistration(unittest.TestCase):
    def test_registered_in_strategy_config(self):
        strategies = StrategyConfig.get_strategies()
        self.assertIn("AlphaTrendV2C2Strategy", strategies)

    def test_has_parameters(self):
        params = StrategyConfig.get_parameters("AlphaTrendV2C2Strategy")
        for name in ("ma_offset", "ma_length", "atr_sl", "risk_perc",
                     "peace_period_candles", "short_term_candles",
                     "short_phase_exit_enabled", "break_even_consecutive_bars",
                     "trailing_sl_pct"):
            self.assertIn(name, params)
        # Deprecated params should be gone.
        for removed in ("adx_exit_enabled", "adx_threshold",
                        "macd_exit_enabled",
                        "mfi_exit_enabled", "mfi_threshold",
                        "mfi_consecutive_days",
                        "sl_sensitivity", "sl_max_pct", "sl_min_pct"):
            self.assertNotIn(removed, params)


if __name__ == "__main__":
    unittest.main()
