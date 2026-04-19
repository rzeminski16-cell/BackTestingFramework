"""
Tests for AlphaTrendV2C2Strategy.

Covers:
- Parameter validation
- Required columns change with enabled sub-rules
- Stage classification by bars since entry
- Peace-period: no strategy exit
- Short-term: OR semantics between ADX and MACD
- Long-term: MFI consecutive-days rule
- Long-term moving SL: initial move respects "enough room" rule,
  subsequent bars tighten only, clamp at sl_min_pct
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
    adx=None,
    macd_signal=None,
    mfi=None,
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
        "adx_14_adx": (np.full(n, 25.0) if adx is None
                       else np.asarray(adx, dtype=float)),
        "macd_14_macd_signal": (np.full(n, 1.0) if macd_signal is None
                                else np.asarray(macd_signal, dtype=float)),
        "mfi_14_mfi": (np.full(n, 60.0) if mfi is None
                       else np.asarray(mfi, dtype=float)),
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
        self.assertEqual(s.adx_threshold, 20.0)
        self.assertEqual(s.mfi_threshold, 50.0)
        self.assertEqual(s.mfi_consecutive_days, 3)
        self.assertEqual(s.sl_sensitivity, 0.5)
        self.assertEqual(s.sl_max_pct, 20.0)
        self.assertEqual(s.sl_min_pct, 5.0)
        self.assertEqual(s.trade_direction, TradeDirection.LONG)

    def test_invalid_stage_lengths(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(peace_period_candles=-1)
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(short_term_candles=-1)

    def test_invalid_sl_bounds(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(sl_max_pct=0.0)
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(sl_min_pct=0.0)
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(sl_min_pct=30.0, sl_max_pct=20.0)
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(sl_sensitivity=-0.1)

    def test_invalid_mfi_consecutive_days(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(mfi_consecutive_days=0)

    def test_invalid_ma_length(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2C2Strategy(ma_length=21)


class TestRequiredColumns(unittest.TestCase):
    def test_all_rules_enabled_requires_all_columns(self):
        s = AlphaTrendV2C2Strategy()
        cols = s.required_columns()
        for c in ("date", "close", "atr_14_atr", "ema_20_ema",
                  "adx_14_adx", "macd_14_macd_signal", "mfi_14_mfi"):
            self.assertIn(c, cols)

    def test_disabled_rules_drop_their_columns(self):
        s = AlphaTrendV2C2Strategy(
            adx_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=False,
        )
        cols = s.required_columns()
        self.assertNotIn("adx_14_adx", cols)
        self.assertNotIn("macd_14_macd_signal", cols)
        self.assertNotIn("mfi_14_mfi", cols)


# ---------------------------------------------------------------------------
# Peace-period
# ---------------------------------------------------------------------------

class TestPeacePeriod(unittest.TestCase):
    def test_no_exit_during_peace_period(self):
        """Even with ADX/MACD/MFI all firing, peace-period returns no exit."""
        n = 60
        s = AlphaTrendV2C2Strategy(
            peace_period_candles=5,
            short_term_candles=15,
            adx_exit_enabled=True, adx_threshold=20.0,
            macd_exit_enabled=True,
            mfi_exit_enabled=True, mfi_threshold=50.0, mfi_consecutive_days=3,
        )
        df = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 5.0),
            macd_signal=np.full(n, -1.0),
            mfi=np.full(n, 20.0),
        ))

        entry_index = 20
        pos = _open_position_at(df, entry_index)

        # Bars since entry: 0, 1, 2, 3, 4 -> all peace-period
        for offset in range(5):
            ctx = _make_context(df, index=entry_index + offset,
                                current_position=pos)
            self.assertIsNone(
                s.generate_exit_signal(ctx),
                f"unexpected exit at offset {offset}"
            )


# ---------------------------------------------------------------------------
# Short-term: ADX OR MACD
# ---------------------------------------------------------------------------

class TestShortTerm(unittest.TestCase):
    def _strat(self, **kwargs):
        defaults = dict(
            peace_period_candles=2,
            short_term_candles=10,
            adx_exit_enabled=True, adx_threshold=20.0,
            macd_exit_enabled=True,
            mfi_exit_enabled=False,
        )
        defaults.update(kwargs)
        return AlphaTrendV2C2Strategy(**defaults)

    def test_adx_alone_triggers_short_term_exit(self):
        n = 60
        s = self._strat()
        df = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 10.0),          # ADX fires
            macd_signal=np.full(n, 1.0),   # MACD does NOT fire
        ))
        entry_index = 20
        pos = _open_position_at(df, entry_index)
        # Short-term: offset 2..11
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        signal = s.generate_exit_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("ADX", signal.reason)

    def test_macd_alone_triggers_short_term_exit(self):
        n = 60
        s = self._strat()
        df = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 30.0),
            macd_signal=np.full(n, -0.5),
        ))
        entry_index = 20
        pos = _open_position_at(df, entry_index)
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        signal = s.generate_exit_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("MACD", signal.reason)

    def test_neither_fires_no_exit(self):
        n = 60
        s = self._strat()
        df = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 30.0),
            macd_signal=np.full(n, 1.0),
        ))
        entry_index = 20
        pos = _open_position_at(df, entry_index)
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_mfi_disabled_in_short_term(self):
        """MFI below threshold should NOT cause an exit during short-term."""
        n = 60
        s = self._strat(mfi_exit_enabled=True, mfi_threshold=50.0,
                        mfi_consecutive_days=3)
        df = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 30.0),             # ADX does not fire
            macd_signal=np.full(n, 1.0),      # MACD does not fire
            mfi=np.full(n, 20.0),             # MFI fires - but only in long-term
        ))
        entry_index = 20
        pos = _open_position_at(df, entry_index)
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))


# ---------------------------------------------------------------------------
# Long-term: MFI consecutive-days with configurable threshold
# ---------------------------------------------------------------------------

class TestLongTermMfi(unittest.TestCase):
    def test_mfi_fires_only_in_long_term(self):
        n = 60
        s = AlphaTrendV2C2Strategy(
            peace_period_candles=2,
            short_term_candles=3,
            adx_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=True, mfi_threshold=50.0, mfi_consecutive_days=3,
        )
        # MFI stays below 50 everywhere.
        df = s.prepare_data(_make_data([100.0] * n, mfi=np.full(n, 20.0)))

        entry_index = 20
        pos = _open_position_at(df, entry_index)

        # Peace period (offsets 0,1): no exit
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

        # Short term (offsets 2,3,4): no exit (MFI not in short term)
        ctx = _make_context(df, index=entry_index + 3, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

        # Long term (offset 5+): exit
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        signal = s.generate_exit_signal(ctx)
        self.assertIsNotNone(signal)
        self.assertIn("MFI", signal.reason)

    def test_custom_mfi_threshold(self):
        n = 60
        s = AlphaTrendV2C2Strategy(
            peace_period_candles=0,
            short_term_candles=0,
            adx_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=True, mfi_threshold=30.0, mfi_consecutive_days=2,
        )
        # MFI at 40 - below 50 (default) but ABOVE the custom 30 threshold.
        df = s.prepare_data(_make_data([100.0] * n, mfi=np.full(n, 40.0)))
        entry_index = 20
        pos = _open_position_at(df, entry_index)
        ctx = _make_context(df, index=entry_index + 5, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

        # Below 30 -> fires
        df2 = s.prepare_data(_make_data([100.0] * n, mfi=np.full(n, 25.0)))
        ctx2 = _make_context(df2, index=entry_index + 5, current_position=pos)
        self.assertIsNotNone(s.generate_exit_signal(ctx2))


# ---------------------------------------------------------------------------
# Long-term moving SL
# ---------------------------------------------------------------------------

class TestLongTermMovingSL(unittest.TestCase):
    def _strat(self, **kwargs):
        defaults = dict(
            peace_period_candles=0,
            short_term_candles=0,
            adx_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=False,
            sl_sensitivity=0.5,
            sl_max_pct=20.0,
            sl_min_pct=5.0,
        )
        defaults.update(kwargs)
        return AlphaTrendV2C2Strategy(**defaults)

    def test_no_sl_move_outside_long_term(self):
        n = 60
        s = AlphaTrendV2C2Strategy(
            peace_period_candles=5,
            short_term_candles=10,
            adx_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=False,
        )
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=80.0)
        # Offset 3: peace. Offset 10: short. Both should NOT move SL.
        for offset in (3, 10):
            ctx = _make_context(df, index=entry_index + offset,
                                current_position=pos)
            self.assertIsNone(s.should_adjust_stop(ctx),
                              f"unexpected SL adjust at offset {offset}")

    def test_first_move_waits_for_room(self):
        """If the computed SL would WIDEN the stop, no move happens yet."""
        n = 60
        s = self._strat(sl_sensitivity=0.0, sl_max_pct=20.0, sl_min_pct=5.0)
        # Current SL at 95 means SL% is 5% of 100 - very tight.
        # Computed SL% = 20% -> new SL = 80. That would widen, so no move.
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=95.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        self.assertIsNone(s.should_adjust_stop(ctx))

    def test_first_move_fires_when_room_exists(self):
        """If computed SL tightens, the first long-term move fires."""
        n = 60
        s = self._strat(sl_sensitivity=0.0, sl_max_pct=20.0, sl_min_pct=5.0)
        # Current SL at 70 -> distance 30%. Computed 20% -> SL=80 tightens.
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=70.0)
        ctx = _make_context(df, index=entry_index + 1, current_position=pos)
        new_sl = s.should_adjust_stop(ctx)
        self.assertIsNotNone(new_sl)
        self.assertAlmostEqual(new_sl, 80.0, places=6)

    def test_subsequent_moves_only_tighten(self):
        """After the first move, each bar tightens by sl_sensitivity%."""
        n = 60
        s = self._strat(sl_sensitivity=1.0, sl_max_pct=20.0, sl_min_pct=5.0)
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=70.0)

        # Offset 1: SL% = 20 - 1*1 = 19 -> SL = 81
        ctx1 = _make_context(df, index=entry_index + 1, current_position=pos)
        sl1 = s.should_adjust_stop(ctx1)
        self.assertAlmostEqual(sl1, 81.0, places=6)
        pos.stop_loss = sl1

        # Offset 2: SL% = 18 -> SL = 82 (tighter than 81)
        ctx2 = _make_context(df, index=entry_index + 2, current_position=pos)
        sl2 = s.should_adjust_stop(ctx2)
        self.assertAlmostEqual(sl2, 82.0, places=6)

    def test_clamped_at_min_pct(self):
        """SL% is clamped at sl_min_pct once the schedule runs out."""
        n = 80
        s = self._strat(sl_sensitivity=1.0, sl_max_pct=20.0, sl_min_pct=5.0)
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=70.0)

        # After 15+ bars: schedule would give 5%, clamped.
        # At offset 20: 20 - 1*20 = 0 -> clamped to 5 -> SL = 95.
        ctx = _make_context(df, index=entry_index + 20, current_position=pos)
        new_sl = s.should_adjust_stop(ctx)
        self.assertAlmostEqual(new_sl, 95.0, places=6)

    def test_sl_never_widens_after_armed(self):
        """Once armed, a larger-SL-distance recompute should not apply."""
        n = 60
        s = self._strat(sl_sensitivity=5.0, sl_max_pct=20.0, sl_min_pct=5.0)
        df = s.prepare_data(_make_data([100.0] * n))
        entry_index = 20
        pos = _open_position_at(df, entry_index, stop=70.0)

        # First move: offset 1 -> SL% = 15 -> SL=85. Set as current stop.
        ctx1 = _make_context(df, index=entry_index + 1, current_position=pos)
        sl1 = s.should_adjust_stop(ctx1)
        self.assertAlmostEqual(sl1, 85.0, places=6)
        pos.stop_loss = sl1 + 2.0  # Manually set tighter current stop = 87.

        # Offset 2 recompute SL% = 10 -> SL = 90 (tighter than 87) - should apply.
        ctx2 = _make_context(df, index=entry_index + 2, current_position=pos)
        sl2 = s.should_adjust_stop(ctx2)
        self.assertAlmostEqual(sl2, 90.0, places=6)

        # Now artificially bump current stop to 95 (very tight). Next recompute
        # gives 95 again -> equal, should NOT update.
        pos.stop_loss = 95.0
        ctx3 = _make_context(df, index=entry_index + 3, current_position=pos)
        sl3 = s.should_adjust_stop(ctx3)
        # 20 - 5*3 = 5 -> min clamp -> SL = 95. Equal -> no adjust.
        self.assertIsNone(sl3)


# ---------------------------------------------------------------------------
# No exit without position / registration
# ---------------------------------------------------------------------------

class TestMisc(unittest.TestCase):
    def test_no_exit_without_position(self):
        n = 60
        s = AlphaTrendV2C2Strategy()
        df = s.prepare_data(_make_data([100.0] * n,
                                       adx=np.full(n, 5.0),
                                       macd_signal=np.full(n, -1.0),
                                       mfi=np.full(n, 10.0)))
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
                     "adx_exit_enabled", "adx_threshold",
                     "macd_exit_enabled",
                     "mfi_exit_enabled", "mfi_threshold", "mfi_consecutive_days",
                     "sl_sensitivity", "sl_max_pct", "sl_min_pct"):
            self.assertIn(name, params)


if __name__ == "__main__":
    unittest.main()
