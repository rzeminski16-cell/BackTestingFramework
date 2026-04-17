"""
Tests for AlphaTrendV2Strategy.

Covers:
- Parameter validation
- Required columns change with enabled sub-rules
- Entry / SL / sizing parity with V1
- Individual exit sub-rules (ADX, ROC, MACD, MFI)
- Combined (AND) semantics across enabled sub-rules
- No exit when all sub-rules are disabled
- Registration with StrategyConfig
"""
import unittest
from typing import Sequence

import numpy as np
import pandas as pd

from strategies.alpha_trend_v2_strategy import AlphaTrendV2Strategy
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
    """Minimal OHLCV + ATR + precomputed-MA + ADX/MACD/MFI DataFrame."""
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


def _open_position(price=100.0, stop=90.0, size=10.0,
                   date="2020-01-02", symbol="TEST"):
    return Position(
        symbol=symbol,
        entry_date=pd.Timestamp(date),
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
        s = AlphaTrendV2Strategy()
        self.assertEqual(s.ma_offset, 2)
        self.assertEqual(s.ma_length, 20)
        self.assertEqual(s.ema_sma, "EMA")
        self.assertEqual(s.atr_sl, 2.0)
        self.assertEqual(s.risk_perc, 2.0)
        self.assertTrue(s.adx_exit_enabled)
        self.assertEqual(s.adx_threshold, 20.0)
        self.assertTrue(s.roc_exit_enabled)
        self.assertEqual(s.roc_period, 10)
        self.assertEqual(s.roc_threshold, 0.0)
        self.assertTrue(s.macd_exit_enabled)
        self.assertTrue(s.mfi_exit_enabled)
        self.assertEqual(s.mfi_consecutive_days, 3)
        self.assertEqual(s.trade_direction, TradeDirection.LONG)

    def test_invalid_roc_period(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2Strategy(roc_period=0)

    def test_invalid_mfi_consecutive_days(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2Strategy(mfi_consecutive_days=0)

    def test_invalid_ma_length(self):
        with self.assertRaises(ValueError):
            AlphaTrendV2Strategy(ma_length=21)


class TestRequiredColumns(unittest.TestCase):
    def test_all_rules_enabled_requires_all_columns(self):
        s = AlphaTrendV2Strategy()
        cols = s.required_columns()
        for c in ("date", "close", "atr_14_atr", "ema_20_ema",
                  "adx_14_adx", "macd_14_macd_signal", "mfi_14_mfi"):
            self.assertIn(c, cols)

    def test_disabled_rules_drop_their_columns(self):
        s = AlphaTrendV2Strategy(
            adx_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=False,
        )
        cols = s.required_columns()
        self.assertNotIn("adx_14_adx", cols)
        self.assertNotIn("macd_14_macd_signal", cols)
        self.assertNotIn("mfi_14_mfi", cols)


# ---------------------------------------------------------------------------
# prepare_data / ROC calculation
# ---------------------------------------------------------------------------

class TestPrepareData(unittest.TestCase):
    def test_roc_column_added_when_enabled(self):
        closes = np.linspace(100, 120, 30).tolist()
        df = _make_data(closes)
        s = AlphaTrendV2Strategy(roc_period=5, roc_exit_enabled=True)
        prepared = s.prepare_data(df)
        self.assertIn("alpha_roc_5", prepared.columns)
        # Manually compute expected ROC and compare non-NaN values.
        expected = pd.Series(closes).pct_change(5) * 100.0
        actual = prepared["alpha_roc_5"]
        np.testing.assert_allclose(
            actual.iloc[5:].values, expected.iloc[5:].values
        )

    def test_roc_column_absent_when_disabled(self):
        closes = [100.0] * 20
        df = _make_data(closes)
        s = AlphaTrendV2Strategy(roc_exit_enabled=False, roc_period=7)
        prepared = s.prepare_data(df)
        self.assertNotIn("alpha_roc_7", prepared.columns)


# ---------------------------------------------------------------------------
# Individual exit sub-rules
# ---------------------------------------------------------------------------

class TestExitSubRules(unittest.TestCase):
    def _prep(self, **kwargs):
        defaults = dict(
            ma_offset=2, ema_sma="EMA", ma_length=14,
            atr_sl=1.0, risk_perc=1.0,
            adx_exit_enabled=False,
            roc_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=False,
        )
        defaults.update(kwargs)
        return AlphaTrendV2Strategy(**defaults)

    def test_adx_only_fires_below_threshold(self):
        n = 40
        adx_below = np.full(n, 10.0)
        adx_above = np.full(n, 30.0)
        s = self._prep(adx_exit_enabled=True, adx_threshold=20.0)

        df_below = s.prepare_data(_make_data([100.0] * n, adx=adx_below))
        df_above = s.prepare_data(_make_data([100.0] * n, adx=adx_above))

        pos = _open_position()
        ctx_below = _make_context(df_below, index=30, current_position=pos)
        ctx_above = _make_context(df_above, index=30, current_position=pos)

        self.assertIsNotNone(s.generate_exit_signal(ctx_below))
        self.assertIsNone(s.generate_exit_signal(ctx_above))

    def test_macd_only_fires_below_zero(self):
        n = 40
        s = self._prep(macd_exit_enabled=True)

        df_neg = s.prepare_data(_make_data([100.0] * n, macd_signal=np.full(n, -0.5)))
        df_pos = s.prepare_data(_make_data([100.0] * n, macd_signal=np.full(n, 0.5)))

        pos = _open_position()
        ctx_neg = _make_context(df_neg, index=30, current_position=pos)
        ctx_pos = _make_context(df_pos, index=30, current_position=pos)

        self.assertIsNotNone(s.generate_exit_signal(ctx_neg))
        self.assertIsNone(s.generate_exit_signal(ctx_pos))

    def test_mfi_requires_consecutive_days(self):
        """MFI must stay below 50 for mfi_consecutive_days bars."""
        n = 40
        s = self._prep(mfi_exit_enabled=True, mfi_consecutive_days=3)

        # Last 3 values below 50 -> exit
        mfi_exit = np.full(n, 60.0)
        mfi_exit[-3:] = [40.0, 30.0, 25.0]
        df_exit = s.prepare_data(_make_data([100.0] * n, mfi=mfi_exit))
        pos = _open_position()
        ctx_exit = _make_context(df_exit, index=n - 1, current_position=pos)
        self.assertIsNotNone(s.generate_exit_signal(ctx_exit))

        # Only last 2 below 50 -> NOT enough
        mfi_short = np.full(n, 60.0)
        mfi_short[-2:] = [40.0, 30.0]
        df_short = s.prepare_data(_make_data([100.0] * n, mfi=mfi_short))
        ctx_short = _make_context(df_short, index=n - 1, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx_short))

        # 3 below but then recover on current bar -> NOT firing
        mfi_recover = np.full(n, 60.0)
        mfi_recover[-4:-1] = [40.0, 30.0, 25.0]
        mfi_recover[-1] = 55.0
        df_rec = s.prepare_data(_make_data([100.0] * n, mfi=mfi_recover))
        ctx_rec = _make_context(df_rec, index=n - 1, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx_rec))

    def test_roc_fires_when_below_threshold(self):
        n = 30
        # Declining prices -> negative ROC
        closes_down = np.linspace(100, 80, n).tolist()
        closes_up = np.linspace(80, 120, n).tolist()
        s = self._prep(roc_exit_enabled=True, roc_period=5, roc_threshold=0.0)

        df_down = s.prepare_data(_make_data(closes_down))
        df_up = s.prepare_data(_make_data(closes_up))

        pos = _open_position()
        ctx_down = _make_context(df_down, index=20, current_position=pos)
        ctx_up = _make_context(df_up, index=20, current_position=pos)

        self.assertIsNotNone(s.generate_exit_signal(ctx_down))
        self.assertIsNone(s.generate_exit_signal(ctx_up))


# ---------------------------------------------------------------------------
# Combined AND semantics
# ---------------------------------------------------------------------------

class TestCombinedExit(unittest.TestCase):
    def test_all_enabled_must_fire(self):
        """If ADX and MACD are both enabled, both must be in exit state."""
        n = 40
        s = AlphaTrendV2Strategy(
            adx_exit_enabled=True, adx_threshold=20.0,
            roc_exit_enabled=False,
            macd_exit_enabled=True,
            mfi_exit_enabled=False,
        )
        pos = _open_position()

        # Both firing
        df_both = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 10.0),
            macd_signal=np.full(n, -0.5),
        ))
        ctx = _make_context(df_both, index=30, current_position=pos)
        self.assertIsNotNone(s.generate_exit_signal(ctx))

        # Only ADX firing, MACD still positive -> no exit
        df_adx_only = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 10.0),
            macd_signal=np.full(n, 0.5),
        ))
        ctx = _make_context(df_adx_only, index=30, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

        # Only MACD firing -> no exit
        df_macd_only = s.prepare_data(_make_data(
            [100.0] * n,
            adx=np.full(n, 30.0),
            macd_signal=np.full(n, -0.5),
        ))
        ctx = _make_context(df_macd_only, index=30, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_no_rules_enabled_no_exit(self):
        n = 40
        s = AlphaTrendV2Strategy(
            adx_exit_enabled=False,
            roc_exit_enabled=False,
            macd_exit_enabled=False,
            mfi_exit_enabled=False,
        )
        df = s.prepare_data(_make_data([100.0] * n))
        pos = _open_position()
        ctx = _make_context(df, index=30, current_position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_no_exit_without_position(self):
        n = 40
        s = AlphaTrendV2Strategy(adx_exit_enabled=True, adx_threshold=20.0,
                                 roc_exit_enabled=False,
                                 macd_exit_enabled=False,
                                 mfi_exit_enabled=False)
        df = s.prepare_data(_make_data([100.0] * n, adx=np.full(n, 5.0)))
        ctx = _make_context(df, index=30, current_position=None)
        self.assertIsNone(s.generate_exit_signal(ctx))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):
    def test_registered_in_strategy_config(self):
        strategies = StrategyConfig.get_strategies()
        self.assertIn("AlphaTrendV2Strategy", strategies)

    def test_has_parameters(self):
        params = StrategyConfig.get_parameters("AlphaTrendV2Strategy")
        for name in ("ma_offset", "ma_length", "atr_sl", "risk_perc",
                     "adx_exit_enabled", "adx_threshold",
                     "roc_exit_enabled", "roc_period", "roc_threshold",
                     "macd_exit_enabled",
                     "mfi_exit_enabled", "mfi_consecutive_days"):
            self.assertIn(name, params)


if __name__ == "__main__":
    unittest.main()
