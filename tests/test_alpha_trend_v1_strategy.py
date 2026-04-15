"""
Tests for AlphaTrendV1Strategy.

Covers:
- Parameter validation
- AlphaMACross indicator calculation (EMA vs SMA, offset shift)
- Entry signal generation (crossover / no-cross)
- Stop loss calculation
- Position sizing (risk-based)
- Time-based exit
- Integration with SingleSecurityEngine
- Registration with StrategyConfig
"""
import math
import unittest
from datetime import datetime
from typing import Optional

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

def _make_data(closes, atr=2.0, start_date="2020-01-01"):
    """Build a minimal OHLCV + ATR DataFrame used for testing."""
    dates = pd.date_range(start=start_date, periods=len(closes), freq="B")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(len(closes), 1_000_000.0),
        "atr_14_atr": np.full(len(closes), float(atr)),
    })


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

    def test_ema_sma_case_insensitive(self):
        s = AlphaTrendV1Strategy(ema_sma="sma")
        self.assertEqual(s.ema_sma, "SMA")

    def test_invalid_ma_offset(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(ma_offset=0)

    def test_invalid_ema_sma(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(ema_sma="WMA")

    def test_invalid_ma_length(self):
        with self.assertRaises(ValueError):
            AlphaTrendV1Strategy(ma_length=1)

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

    def test_required_columns(self):
        s = AlphaTrendV1Strategy()
        cols = s.required_columns()
        self.assertIn("date", cols)
        self.assertIn("close", cols)
        self.assertIn("atr_14_atr", cols)


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

class TestAlphaMACrossIndicator(unittest.TestCase):
    def test_sma_matches_manual(self):
        closes = list(range(1, 31))  # 1..30, strictly increasing
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(ema_sma="SMA", ma_length=5, ma_offset=2)
        prepared = s.prepare_data(df)

        expected_sma = pd.Series(closes, dtype=float).rolling(5).mean()
        pd.testing.assert_series_equal(
            prepared[s._MA_COL].reset_index(drop=True),
            expected_sma.reset_index(drop=True),
            check_names=False,
        )

        # MA offset should be the same MA shifted by 2 bars (causal).
        expected_offset = expected_sma.shift(2)
        pd.testing.assert_series_equal(
            prepared[s._MA_OFFSET_COL].reset_index(drop=True),
            expected_offset.reset_index(drop=True),
            check_names=False,
        )

    def test_ema_matches_manual(self):
        closes = np.linspace(100, 120, 40)
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(ema_sma="EMA", ma_length=10, ma_offset=3)
        prepared = s.prepare_data(df)

        expected_ema = pd.Series(closes).ewm(span=10, adjust=False).mean()
        np.testing.assert_allclose(
            prepared[s._MA_COL].values, expected_ema.values, rtol=1e-9
        )
        np.testing.assert_allclose(
            prepared[s._MA_OFFSET_COL].iloc[3:].values,
            expected_ema.iloc[:-3].values,
            rtol=1e-9,
        )

    def test_prepare_data_no_lookahead(self):
        """The offset column should be shifted forward (shift(+n)), not
        backward - no future information ever leaks."""
        closes = np.linspace(100, 120, 40)
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(ema_sma="SMA", ma_length=5, ma_offset=3)
        prepared = s.prepare_data(df)
        # The first ma_offset values of the offset column must be NaN (no past
        # data available to shift from).
        self.assertTrue(prepared[s._MA_OFFSET_COL].iloc[:3].isna().all())
        # Last rows should NOT be NaN for a causal, populated MA.
        self.assertFalse(prepared[s._MA_OFFSET_COL].iloc[-1:].isna().all())


# ---------------------------------------------------------------------------
# Entry signal generation
# ---------------------------------------------------------------------------

class TestEntrySignal(unittest.TestCase):
    def _build(self, closes, ma_length=3, ma_offset=2):
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(ema_sma="SMA", ma_length=ma_length,
                                 ma_offset=ma_offset, atr_sl=1.0, time_exit=10,
                                 risk_perc=1.0)
        df = s.prepare_data(df)
        return s, df

    def test_no_signal_during_warmup(self):
        closes = [10, 11, 12, 13, 14, 15]
        s, df = self._build(closes)
        # Before warmup: index < ma_length + ma_offset = 5
        ctx = _make_context(df, index=3)
        self.assertIsNone(s.generate_entry_signal(ctx))

    def test_crossover_triggers_long_signal(self):
        # Construct a series where the MA crosses the offset MA on the last
        # bar. Using SMA(3), offset=2.
        # Build a decreasing-then-increasing sequence so that MA(t-1) <=
        # MA_offset(t-1) and MA(t) > MA_offset(t).
        closes = [20, 19, 18, 17, 16, 15, 20, 25]
        s, df = self._build(closes, ma_length=3, ma_offset=2)
        min_idx = s.ma_length + s.ma_offset  # 5
        # Check each bar after warmup for a signal.
        signals_found = []
        for idx in range(min_idx, len(df)):
            ctx = _make_context(df, idx)
            sig = s.generate_entry_signal(ctx)
            if sig is not None:
                signals_found.append(idx)
        self.assertTrue(len(signals_found) >= 1,
                        "Expected at least one crossover signal on the uptrend reversal")

    def test_downtrend_no_long_signal(self):
        closes = np.linspace(100, 10, 40).tolist()  # steadily falling
        s, df = self._build(closes, ma_length=5, ma_offset=2)
        min_idx = s.ma_length + s.ma_offset
        for idx in range(min_idx, len(df)):
            ctx = _make_context(df, idx)
            sig = s.generate_entry_signal(ctx)
            self.assertIsNone(sig, f"Unexpected long signal on bar {idx} of a falling series")

    def test_signal_has_stop_loss_and_correct_direction(self):
        closes = [20, 19, 18, 17, 16, 15, 20, 25, 30, 35]
        s, df = self._build(closes, ma_length=3, ma_offset=2)
        idx = len(df) - 1
        ctx = _make_context(df, idx)
        # Force a crossover by scanning back until we find one.
        sig = None
        for i in range(s.ma_length + s.ma_offset, len(df)):
            c = _make_context(df, i)
            s_ = s.generate_entry_signal(c)
            if s_:
                sig = s_
                ctx = c
                break
        self.assertIsNotNone(sig)
        self.assertEqual(sig.type, SignalType.BUY)
        self.assertEqual(sig.direction, TradeDirection.LONG)
        self.assertIsNotNone(sig.stop_loss)
        # Stop loss must be below current price.
        self.assertLess(sig.stop_loss, ctx.current_price)


# ---------------------------------------------------------------------------
# Stop loss calculation
# ---------------------------------------------------------------------------

class TestStopLoss(unittest.TestCase):
    def test_atr_based_stop(self):
        closes = [100.0] * 30
        df = _make_data(closes, atr=2.5)
        s = AlphaTrendV1Strategy(atr_sl=3.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=25)
        sl = s.calculate_initial_stop_loss(ctx)
        # SL = price - atr_sl * ATR = 100 - 3 * 2.5 = 92.5
        self.assertAlmostEqual(sl, 92.5)

    def test_zero_atr_falls_back_to_percentage(self):
        closes = [100.0] * 30
        df = _make_data(closes, atr=0.0)
        s = AlphaTrendV1Strategy(atr_sl=2.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=25)
        sl = s.calculate_initial_stop_loss(ctx)
        self.assertAlmostEqual(sl, 95.0)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing(unittest.TestCase):
    def test_risk_based_sizing(self):
        closes = [100.0] * 30
        df = _make_data(closes, atr=2.0)
        s = AlphaTrendV1Strategy(atr_sl=2.0, risk_perc=2.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=25, equity=100_000.0, fx_rate=1.0)
        signal = Signal.buy(size=1.0, stop_loss=96.0, direction=TradeDirection.LONG)
        shares = s.position_size(ctx, signal)
        # risk_amount = 100000 * 0.02 = 2000
        # stop_distance = 100 - 96 = 4, fx=1 -> shares = 2000/4 = 500
        self.assertAlmostEqual(shares, 500.0)

    def test_fx_rate_adjusts_sizing(self):
        closes = [100.0] * 30
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(risk_perc=2.0)
        df = s.prepare_data(df)
        ctx = _make_context(df, index=25, equity=100_000.0, fx_rate=0.5)
        signal = Signal.buy(size=1.0, stop_loss=96.0, direction=TradeDirection.LONG)
        shares = s.position_size(ctx, signal)
        # stop_distance_base = 4 * 0.5 = 2 -> shares = 2000/2 = 1000
        self.assertAlmostEqual(shares, 1000.0)

    def test_invalid_stop_falls_back(self):
        closes = [100.0] * 30
        df = _make_data(closes)
        s = AlphaTrendV1Strategy()
        df = s.prepare_data(df)
        ctx = _make_context(df, index=25, equity=100_000.0, capital=50_000.0)
        # Stop loss above price -> invalid for LONG.
        signal = Signal.buy(size=1.0, stop_loss=110.0, direction=TradeDirection.LONG)
        shares = s.position_size(ctx, signal)
        # Falls back to using 10% of available capital.
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
        closes = [100.0] * 60
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(time_exit=30)
        df = s.prepare_data(df)
        entry_date = pd.Timestamp(df["date"].iloc[5])
        position = self._make_position(entry_date)
        ctx = _make_context(df, index=15, current_position=position)
        self.assertIsNone(s.generate_exit_signal(ctx))

    def test_exit_after_time_elapsed(self):
        # Use a long date range so we reliably cross the 30 calendar day mark.
        closes = [100.0] * 80
        df = _make_data(closes)
        s = AlphaTrendV1Strategy(time_exit=30)
        df = s.prepare_data(df)
        entry_date = pd.Timestamp(df["date"].iloc[0])
        # Find the first index whose date is >= entry_date + 30 days.
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
        # Build a series with enough length and a clear uptrend reversal so the
        # strategy can trade.
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.3, 200)
        closes = np.concatenate([
            np.linspace(100, 80, 50),      # down trend
            np.linspace(80, 140, 150),     # strong up trend
        ]) + noise
        df = _make_data(closes.tolist(), atr=2.0)

        strategy = AlphaTrendV1Strategy(
            ma_offset=2, ema_sma="EMA", ma_length=10,
            atr_sl=2.0, time_exit=20, risk_perc=2.0,
        )
        engine = SingleSecurityEngine(_create_config())
        result = engine.run("TEST", df, strategy)

        # The strategy should have executed at least one trade on this data.
        self.assertGreaterEqual(len(result.trades), 1)

        # All trades should be LONG.
        for trade in result.trades:
            self.assertEqual(str(trade.side).upper(), "LONG")

        # Equity curve should match data length.
        self.assertEqual(len(result.equity_curve), len(df))

    def test_no_trades_on_flat_data(self):
        # Perfectly flat prices produce no crossover, so no trades.
        closes = [100.0] * 100
        df = _make_data(closes, atr=2.0)
        strategy = AlphaTrendV1Strategy(ma_offset=2, ema_sma="SMA", ma_length=5)
        engine = SingleSecurityEngine(_create_config())
        result = engine.run("TEST", df, strategy)
        self.assertEqual(len(result.trades), 0)


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
        # Create a strategy with the config defaults to make sure they are
        # valid.
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


if __name__ == "__main__":
    unittest.main()
