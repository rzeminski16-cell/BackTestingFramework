"""
Tests for AlphaTrendV1Strategy.

Covers:
- Parameter validation (including the restricted ma_length set)
- Required columns include the right raw-data MA column
- AlphaMACross crossover detection from precomputed MA columns
- Correct causal offset handling
- Stop loss, position sizing and time-based exit
- End-to-end integration with SingleSecurityEngine
- Registration with StrategyConfig
"""
import unittest
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from strategies.alpha_trend_v1_strategy import AlphaTrendV1Strategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal, SignalType
from Classes.Models.position import Position
from Classes.Models.trade_direction import TradeDirection
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from config.strategy_config import StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(closes, atr=2.0, start_date="2020-01-01",
               ma_lengths: Sequence[int] = (7, 14, 20, 30, 50, 90, 200)):
    """Build a minimal OHLCV + ATR + precomputed-MA DataFrame for testing.

    The MAs are populated in the same way the real raw-data pipeline does:
    causal EMA via ``ewm(adjust=False)`` and causal SMA via ``rolling()``.
    """
    dates = pd.date_range(start=start_date, periods=len(closes), freq="B")
    closes = np.asarray(closes, dtype=float)
    data = {
        "date": dates,
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(len(closes), 1_000_000.0),
        "atr_14_atr": np.full(len(closes), float(atr)),
    }
    series = pd.Series(closes)
    for length in ma_lengths:
        data[f"ema_{length}_ema"] = series.ewm(span=length, adjust=False).mean().values
        data[f"sma_{length}_sma"] = series.rolling(window=length, min_periods=length).mean().values
    return pd.DataFrame(data)


def _make_context(df, index, current_position=None, equity=100_000.0,
                  capital=100_000.0, fx_rate=1.0, symbol="TEST"):
    """Create a StrategyContext pointing at a specific bar index."""
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


def _create_config(capital=100_000.0):
    return BacktestConfig(
        initial_capital=capital,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0,
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestParameterValidation(unittest.TestCase):
    def test_default_construction(self):
        s = AlphaTrendV1Strategy()
        self.assertEqual(s.ma_offset, 2)
        self.assertEqual(s.ema_sma, "EMA")
        self.assertEqual(s.ma_length, 20)
        self.assertEqual(s.atr_sl, 2.0)
        self.assertEqual(s.time_exit, 30)
        self.assertEqual(s.risk_perc, 2.0)
        self.assertEqual(s.trade_direction, TradeDirection.LONG)
        self.assertEqual(s._ma_column, "ema_20_ema")

    def test_sma_resolves_correct_column(self):
        s = AlphaTrendV1Strategy(ema_sma="SMA", ma_length=50)
        self.assertEqual(s._ma_column, "sma_50_sma")

    def test_ema_sma_case_insensitive(self):
        s = AlphaTrendV1Strategy(ema_sma="sma", ma_length=20)
        self.assertEqual(s.ema_sma, "SMA")

    def test_invalid_ma_offset(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(ma_offset=0)

    def test_invalid_ema_sma(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(ema_sma="WMA")

    def test_invalid_ma_length_not_in_supported(self):
        for bad in (5, 15, 21, 100, 500):
            with self.assertRaises(ValueError):
                AlphaTrendV1Strategy(ma_length=bad)

    def test_valid_ma_lengths(self):
        for length in AlphaTrendV1Strategy.SUPPORTED_MA_LENGTHS:
            # Should not raise
            AlphaTrendV1Strategy(ma_length=length)

    def test_invalid_atr_sl(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(atr_sl=0)

    def test_invalid_time_exit(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(time_exit=0)

    def test_invalid_risk_perc(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(risk_perc=0)
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(risk_perc=150)

    def test_required_columns_includes_ma_column(self):
        s = AlphaTrendV1Strategy(ema_sma="EMA", ma_length=14)
        cols = s.required_columns()
        self.assertIn("date", cols)
        self.assertIn("close", cols)
        self.assertIn("atr_14_atr", cols)
        self.assertIn("ema_14_ema", cols)

        s = AlphaTrendV1Strategy(ema_sma="SMA", ma_length=200)
        self.assertIn("sma_200_sma", s.required_columns())


# ---------------------------------------------------------------------------
# Raw-data column consumption + offset handling
# ---------------------------------------------------------------------------

class TestRawDataUsage(unittest.TestCase):
    def test_uses_precomputed_ma_column(self):
        """The strategy should return the raw-data MA unchanged - i.e. it does
        not recalculate the MA itself."""
        closes = np.linspace(100, 120, 60).tolist()
        df = _make_data(closes)
        # Overwrite the EMA column with a sentinel sequence so we can verify
        # the strategy reads straight from that column.
        sentinel = np.arange(100, 100 + len(df), dtype=float)
        df["ema_20_ema"] = sentinel

        s = AlphaTrendV1Strategy(ema_sma="EMA", ma_length=20, ma_offset=3)
        prepared = s.prepare_data(df)

        # Sentinel values in the MA column must be preserved exactly.
        np.testing.assert_array_equal(prepared["ema_20_ema"].values, sentinel)

        # Offset column must equal the MA column shifted forward by ``ma_offset``.
        expected_offset = pd.Series(sentinel).shift(3)
        np.testing.assert_array_equal(
            prepared["alpha_ma_offset"].values, expected_offset.values
        )

    def test_missing_ma_column_raises(self):
        """If the raw data lacks the required MA column, validation should
        fail fast with a clear error."""
        closes = [100.0] * 30
        df = _make_data(closes, ma_lengths=(20,))  # No EMA_14 in data.
        s = AlphaTrendV1Strategy(ema_sma="EMA", ma_length=14)
        with self.assertRaises(ValueError) as ctx:
            s.prepare_data(df)
        self.assertIn("ema_14_ema", str(ctx.exception))

    def test_offset_is_causal(self):
        """The offset column should be derived via a forward shift (causal)
        and so must never look ahead."""
        closes = np.linspace(100, 150, 50).tolist()
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(ema_sma="SMA", ma_length=20, ma_offset=5)
        prepared = s.prepare_data(df)
        # At any bar ``t`` (with t >= ma_offset) the offset column must equal
        # the MA value at bar ``t - ma_offset`` (NaN-aware comparison for the
        # MA warmup period).
        np.testing.assert_array_equal(
            prepared["alpha_ma_offset"].iloc[5:].values,
            prepared["sma_20_sma"].iloc[:-5].values,
        )


# ---------------------------------------------------------------------------
# Entry signal generation
# ---------------------------------------------------------------------------

class TestEntrySignal(unittest.TestCase):
    def _build(self, closes, ma_length=14, ma_offset=2, ema_sma="EMA"):
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(ema_sma=ema_sma, ma_length=ma_length,
                                 ma_offset=ma_offset, atr_sl=1.0, time_exit=10,
                                 risk_perc=1.0)
        df = s.prepare_data(df)
        return s, df

    def test_no_signal_during_warmup(self):
        closes = list(range(1, 30))
        s, df = self._build(closes, ma_length=14, ma_offset=2)
        ctx = _make_context(df, index=1)
        self.assertIsNone(s.generate_entry_signal(ctx))

    def test_crossover_triggers_long_signal(self):
        """Build a series whose MA crosses back up after a dip and confirm
        the strategy generates a long signal."""
        closes = (
            list(np.linspace(100, 80, 30))   # down leg
            + list(np.linspace(80, 140, 30))  # up leg - eventually crosses
        )
        s, df = self._build(closes, ma_length=14, ma_offset=3, ema_sma="SMA")
        signals = []
        for idx in range(s.ma_offset + 1, len(df)):
            ctx = _make_context(df, idx)
            sig = s.generate_entry_signal(ctx)
            if sig:
                signals.append(idx)
        self.assertTrue(signals, "Expected at least one crossover signal on "
                                 "an up-trend reversal.")

    def test_downtrend_no_long_signal(self):
        closes = np.linspace(200, 10, 80).tolist()
        s, df = self._build(closes, ma_length=20, ma_offset=2, ema_sma="SMA")
        for idx in range(s.ma_offset + 1, len(df)):
            ctx = _make_context(df, idx)
            self.assertIsNone(
                s.generate_entry_signal(ctx),
                msg=f"Unexpected long signal on bar {idx} of a falling series",
            )

    def test_signal_has_stop_loss_and_correct_direction(self):
        closes = (
            list(np.linspace(100, 80, 30))
            + list(np.linspace(80, 140, 30))
        )
        s, df = self._build(closes, ma_length=14, ma_offset=3, ema_sma="SMA")
        sig = None
        ctx = None
        for idx in range(s.ma_offset + 1, len(df)):
            c = _make_context(df, idx)
            result = s.generate_entry_signal(c)
            if result:
                sig, ctx = result, c
                break
        self.assertIsNotNone(sig)
        self.assertEqual(sig.type, SignalType.BUY)
        self.assertEqual(sig.direction, TradeDirection.LONG)
        self.assertIsNotNone(sig.stop_loss)
        self.assertLess(sig.stop_loss, ctx.current_price)


# ---------------------------------------------------------------------------
# Stop loss
# ---------------------------------------------------------------------------

class TestStopLoss(unittest.TestCase):
    def test_atr_based_stop(self):
        closes = [100.0] * 60
        df = _make_data(closes, atr=2.5)
        s = AlphaTrendV1Strategy(atr_sl=3.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=55)
        sl = s.calculate_initial_stop_loss(ctx)
        self.assertAlmostEqual(sl, 92.5)

    def test_zero_atr_falls_back_to_percentage(self):
        closes = [100.0] * 60
        df = _make_data(closes, atr=0.0)
        s = AlphaTrendV1Strategy(atr_sl=2.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=55)
        sl = s.calculate_initial_stop_loss(ctx)
        self.assertAlmostEqual(sl, 95.0)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing(unittest.TestCase):
    def test_risk_based_sizing(self):
        closes = [100.0] * 60
        df = _make_data(closes, atr=2.0)
        s = AlphaTrendV1Strategy(atr_sl=2.0, risk_perc=2.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=55, equity=100_000.0, fx_rate=1.0)
        signal = Signal.buy(size=1.0, stop_loss=96.0, direction=TradeDirection.LONG)
        shares = s.position_size(ctx, signal)
        self.assertAlmostEqual(shares, 500.0)

    def test_fx_rate_adjusts_sizing(self):
        closes = [100.0] * 60
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(risk_perc=2.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=55, equity=100_000.0, fx_rate=0.5)
        signal = Signal.buy(size=1.0, stop_loss=96.0, direction=TradeDirection.LONG)
        shares = s.position_size(ctx, signal)
        self.assertAlmostEqual(shares, 1000.0)

    def test_invalid_stop_falls_back(self):
        closes = [100.0] * 60
        df = _make_data(closes)
        s = AlphaTrendV1Strategy()
        df = s.prepare_data(df)
        ctx = _make_context(df, index=55, equity=100_000.0, capital=50_000.0)
        signal = Signal.buy(size=1.0, stop_loss=110.0, direction=TradeDirection.LONG)
        shares = s.position_size(ctx, signal)
        self.assertAlmostEqual(shares, 0.1 * 50_000.0 / 100.0)


# ---------------------------------------------------------------------------
# Time-based exit
# ---------------------------------------------------------------------------

class TestTimeExit(unittest.TestCase):
    def _make_position(self, entry_date):
        return Position(
            symbol="TEST",
            entry_date=entry_date,
            entry_price=100.0,
            initial_quantity=100.0,
            current_quantity=100.0,
            direction=TradeDirection.LONG,
            stop_loss=95.0,
        )

    def test_no_exit_before_time(self):
        closes = [100.0] * 80
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(time_exit=30)
        df = s.prepare_data(df)
        entry_date = pd.Timestamp(df["date"].iloc[5])
        position = self._make_position(entry_date)
        ctx = _make_context(df, index=15, current_position=position)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_exit_after_time_elapsed(self):
        closes = [100.0] * 80
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(time_exit=30)
        df = s.prepare_data(df)
        entry_date = pd.Timestamp(df["date"].iloc[0])
        target_index = next(
            i for i in range(len(df))
            if (pd.Timestamp(df["date"].iloc[i]) - entry_date).days >= 30
        )
        position = self._make_position(entry_date)
        ctx = _make_context(df, index=target_index, current_position=position)
        sig = s.generate_exit_signal(ctx)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.type, SignalType.SELL)


# ---------------------------------------------------------------------------
# End-to-end engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration(unittest.TestCase):
    def test_runs_end_to_end_on_trending_data(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.3, 200)
        closes = np.concatenate([
            np.linspace(100, 80, 50),
            np.linspace(80, 140, 150),
        ]) + noise
        df = _make_data(closes.tolist(), atr=2.0)

        strategy = AlphaTrendV1Strategy(
            ma_offset=2, ema_sma="EMA", ma_length=14,
            atr_sl=2.0, time_exit=20, risk_perc=2.0,
        )
        engine = SingleSecurityEngine(_create_config())
        result = engine.run("TEST", df, strategy)

        self.assertGreaterEqual(len(result.trades), 1)
        for trade in result.trades:
            self.assertEqual(str(trade.side).upper(), "LONG")
        self.assertEqual(len(result.equity_curve), len(df))

    def test_no_trades_on_flat_data(self):
        closes = [100.0] * 150
        df = _make_data(closes, atr=2.0)
        strategy = AlphaTrendV1Strategy(ma_offset=2, ema_sma="SMA", ma_length=14)
        engine = SingleSecurityEngine(_create_config())
        result = engine.run("TEST", df, strategy)
        self.assertEqual(len(result.trades), 0)

    def test_runs_on_real_raw_data(self):
        """Smoke test: load an actual raw-data CSV and run a full backtest to
        confirm the strategy wires up with real columns."""
        df = pd.read_csv("raw_data/daily/AAPL_daily.csv")
        df["date"] = pd.to_datetime(df["date"])
        strategy = AlphaTrendV1Strategy()
        engine = SingleSecurityEngine(_create_config())
        result = engine.run("AAPL", df, strategy)
        # Just make sure the run completes and returns sensible artefacts.
        self.assertEqual(len(result.equity_curve), len(df))


# ---------------------------------------------------------------------------
# Config registration
# ---------------------------------------------------------------------------

class TestStrategyConfigRegistration(unittest.TestCase):
    def test_registered_in_strategy_config(self):
        StrategyConfig.reload()
        strategies = StrategyConfig.get_strategies()
        self.assertIn("AlphaTrendV1Strategy", strategies)

    def test_defaults_match_strategy(self):
        StrategyConfig.reload()
        defaults = StrategyConfig.get_defaults("AlphaTrendV1Strategy")
        strategy = AlphaTrendV1Strategy(**defaults)
        self.assertEqual(strategy.ma_length, defaults["ma_length"])
        self.assertEqual(strategy.ma_offset, defaults["ma_offset"])
        self.assertEqual(strategy.ema_sma, defaults["ema_sma"])
        self.assertAlmostEqual(strategy.atr_sl, defaults["atr_sl"])
        self.assertEqual(strategy.time_exit, defaults["time_exit"])
        self.assertAlmostEqual(strategy.risk_perc, defaults["risk_perc"])

    def test_optimization_ranges_defined(self):
        StrategyConfig.reload()
        opt = StrategyConfig.get_optimization_params("AlphaTrendV1Strategy")
        for name in ("ma_offset", "ma_length", "atr_sl", "time_exit", "risk_perc"):
            self.assertIn(name, opt, f"Missing optimization range for {name}")

    def test_ma_length_optimization_grid_is_discrete(self):
        StrategyConfig.reload()
        grid = StrategyConfig.get_optimization_grid(
            "AlphaTrendV1Strategy", "ma_length"
        )
        self.assertEqual(grid, [7, 14, 20, 30, 50, 90, 200])

    def test_config_validates_ma_length_allowed_values(self):
        StrategyConfig.reload()
        errors_ok = StrategyConfig.validate_params(
            "AlphaTrendV1Strategy", {"ma_length": 20}
        )
        self.assertEqual(errors_ok, [])
        errors_bad = StrategyConfig.validate_params(
            "AlphaTrendV1Strategy", {"ma_length": 25}
        )
        self.assertTrue(
            any("ma_length" in e for e in errors_bad),
            f"Expected validation failure for ma_length=25, got: {errors_bad}",
        )


if __name__ == "__main__":
    unittest.main()
