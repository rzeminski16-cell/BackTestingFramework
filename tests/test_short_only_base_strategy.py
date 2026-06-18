"""
Tests for ShortOnlyBaseStrategy.

Covers:
- Parameter validation
- Required columns / SHORT trade direction
- Causal derived series (returns, prior 20-day low, ADV, rally, gap)
- Regime gate (benchmark MA, VIX SMA, breadth) merged from disk
- Universe filters (price / ADV)
- Block A (trend breakdown) and Block B (overextension fade) entries
- ATR stop placed ABOVE entry, risk-based sizing for shorts
- EMA_20 cover and time-based cover exits
- Missing regime data raises a clear error
- End-to-end integration with SingleSecurityEngine on real raw data
- Registration with StrategyConfig
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from strategies.short_only_base_strategy import ShortOnlyBaseStrategy
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

def _make_symbol_data(closes, atr=2.0, rsi=30.0, volume=10_000_000.0,
                      bb_upper=None, opens=None, start_date="2020-01-01"):
    """Build a minimal OHLCV + indicator frame with the raw columns the
    strategy consumes (EMAs computed causally, like the real pipeline)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    opens = closes.copy() if opens is None else np.asarray(opens, dtype=float)
    # Default upper band far above price so Block B never fires unless asked.
    bb_upper = (closes + 1e6) if bb_upper is None else np.asarray(bb_upper, dtype=float)
    series = pd.Series(closes)
    data = {
        "date": dates,
        "open": opens,
        "high": np.maximum(closes, opens) * 1.01,
        "low": np.minimum(closes, opens) * 0.99,
        "close": closes,
        "volume": np.full(n, float(volume)),
        "atr_14_atr": np.full(n, float(atr)),
        "rsi_14_rsi": np.full(n, float(rsi)) if np.isscalar(rsi) else np.asarray(rsi, float),
        "ema_7_ema": series.ewm(span=7, adjust=False).mean().values,
        "ema_20_ema": series.ewm(span=20, adjust=False).mean().values,
        "ema_50_ema": series.ewm(span=50, adjust=False).mean().values,
        "bbands_20_real upper band": bb_upper,
    }
    return pd.DataFrame(data)


def _write_regime(dirpath, dates, passing=True):
    """Write SPX / VIX / breadth CSVs covering ``dates``.

    When ``passing`` the three regime conditions are satisfied:
    benchmark below its MA (declining), VIX above its SMA (rising), breadth low.
    """
    n = len(dates)
    bench_close = np.linspace(200, 100, n) if passing else np.linspace(100, 200, n)
    vix_close = np.linspace(10, 40, n) if passing else np.linspace(40, 10, n)
    breadth = np.full(n, 30.0 if passing else 50.0)

    pd.DataFrame({
        "date": dates, "open": bench_close, "high": bench_close,
        "low": bench_close, "close": bench_close,
        "volume": np.zeros(n), "symbol": "SPX",
    }).to_csv(os.path.join(dirpath, "SPX_daily.csv"), index=False)

    pd.DataFrame({
        "date": dates, "open": vix_close, "high": vix_close,
        "low": vix_close, "close": vix_close,
        "volume": np.zeros(n), "symbol": "VIX",
    }).to_csv(os.path.join(dirpath, "VIX_daily.csv"), index=False)

    pd.DataFrame({
        "date": dates,
        "breadth_pct_above_50dma": breadth,
    }).to_csv(os.path.join(dirpath, "BREADTH_daily.csv"), index=False)


def _make_context(df, index, position=None, equity=100_000.0,
                  capital=100_000.0, fx_rate=1.0, symbol="TEST"):
    return StrategyContext(
        data=df,
        current_index=index,
        current_price=float(df["close"].iloc[index]),
        current_date=pd.Timestamp(df["date"].iloc[index]),
        position=position,
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


class _RegimeTempMixin:
    """Per-test temp dir for regime CSVs (unique path avoids the module-level
    file cache returning stale data between tests)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="short_only_regime_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _strategy(self, **overrides):
        params = dict(
            benchmark_ma_period=5,
            vix_sma_period=5,
            benchmark_dir=self.tmp,
        )
        params.update(overrides)
        return ShortOnlyBaseStrategy(**params)


# ---------------------------------------------------------------------------
# Parameter validation / metadata
# ---------------------------------------------------------------------------

class TestParameterValidation(unittest.TestCase):
    def test_default_construction(self):
        s = ShortOnlyBaseStrategy()
        self.assertEqual(s.trade_direction, TradeDirection.SHORT)
        self.assertEqual(s.benchmark_ma_period, 100)
        self.assertEqual(s.vix_sma_period, 60)
        self.assertEqual(s.breadth_threshold, 40.0)
        self.assertEqual(s.atr_sl, 2.5)
        self.assertEqual(s.time_exit, 10)
        self.assertEqual(s.min_adv_20, 5_000_000.0)

    def test_required_columns(self):
        cols = ShortOnlyBaseStrategy().required_columns()
        for c in ("date", "close", "open", "volume", "atr_14_atr", "rsi_14_rsi",
                  "ema_7_ema", "ema_20_ema", "ema_50_ema",
                  "bbands_20_real upper band"):
            self.assertIn(c, cols)

    def test_invalid_params_raise(self):
        for kwargs in (
            {"benchmark_ma_period": 0},
            {"vix_sma_period": 0},
            {"breadth_threshold": 0},
            {"breadth_threshold": 150},
            {"rsi_floor": -1},
            {"rally_atr_mult": 0},
            {"gap_atr_mult": 0},
            {"atr_sl": 0},
            {"time_exit": 0},
            {"risk_perc": 0},
            {"risk_perc": 150},
            {"min_price": -1},
            {"min_adv_20": -1},
        ):
            with self.assertRaises(ValueError, msg=f"expected ValueError for {kwargs}"):
                ShortOnlyBaseStrategy(**kwargs)


# ---------------------------------------------------------------------------
# Derived series (causal) + regime merge
# ---------------------------------------------------------------------------

class TestPrepareData(_RegimeTempMixin, unittest.TestCase):
    def test_derived_columns_are_causal(self):
        closes = np.linspace(150, 90, 80).tolist()
        df = _make_symbol_data(closes)
        _write_regime(self.tmp, df["date"])
        s = self._strategy()
        prepared = s.prepare_data(df)

        # return_20d / return_60d match pct_change.
        np.testing.assert_allclose(
            prepared["return_20d"].values,
            pd.Series(closes).pct_change(20).values,
            equal_nan=True,
        )
        # lowest_close_20d excludes today (prior 20-day low).
        expected_low = pd.Series(closes).rolling(20).min().shift(1)
        np.testing.assert_allclose(
            prepared["lowest_close_20d"].values, expected_low.values, equal_nan=True
        )
        # gap_up = open - prev close (causal shift).
        expected_gap = df["open"].values - pd.Series(closes).shift(1).values
        np.testing.assert_allclose(
            prepared["gap_up"].values, expected_gap, equal_nan=True
        )

    def test_regime_columns_merged(self):
        closes = np.linspace(150, 90, 80).tolist()
        df = _make_symbol_data(closes)
        _write_regime(self.tmp, df["date"], passing=True)
        prepared = self._strategy().prepare_data(df)
        for col in ("benchmark_close", "benchmark_ma", "vix_close", "vix_sma",
                    "breadth_pct_above_50dma"):
            self.assertIn(col, prepared.columns)
        # On a late bar (past warmup) the regime values are populated.
        self.assertFalse(np.isnan(prepared["benchmark_ma"].iloc[-1]))
        self.assertFalse(np.isnan(prepared["vix_sma"].iloc[-1]))

    def test_missing_regime_data_raises(self):
        closes = np.linspace(150, 90, 80).tolist()
        df = _make_symbol_data(closes)
        empty_dir = tempfile.mkdtemp(prefix="short_only_empty_")
        self.addCleanup(shutil.rmtree, empty_dir, ignore_errors=True)
        s = ShortOnlyBaseStrategy(benchmark_ma_period=5, vix_sma_period=5,
                                  benchmark_dir=empty_dir)
        with self.assertRaises(FileNotFoundError):
            s.prepare_data(df)


# ---------------------------------------------------------------------------
# Regime gate
# ---------------------------------------------------------------------------

class TestRegimeGate(_RegimeTempMixin, unittest.TestCase):
    def test_breakdown_entry_when_regime_passes(self):
        closes = np.linspace(150, 90, 80).tolist()  # steady decline => Block A
        df = _make_symbol_data(closes, rsi=30.0)
        _write_regime(self.tmp, df["date"], passing=True)
        s = self._strategy()
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=len(prepared) - 1)
        sig = s.generate_entry_signal(ctx)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.type, SignalType.BUY)
        self.assertEqual(sig.direction, TradeDirection.SHORT)
        # Short stop sits ABOVE entry.
        self.assertGreater(sig.stop_loss, ctx.current_price)

    def test_no_entry_when_regime_fails(self):
        closes = np.linspace(150, 90, 80).tolist()
        df = _make_symbol_data(closes, rsi=30.0)
        _write_regime(self.tmp, df["date"], passing=False)  # gate closed
        s = self._strategy()
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=len(prepared) - 1)
        self.assertIsNone(s.generate_entry_signal(ctx))

    def test_breadth_threshold_blocks_entry(self):
        closes = np.linspace(150, 90, 80).tolist()
        df = _make_symbol_data(closes, rsi=30.0)
        _write_regime(self.tmp, df["date"], passing=True)  # breadth = 30
        # Tighten threshold below the breadth value -> gate closes.
        s = self._strategy(breadth_threshold=25.0)
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=len(prepared) - 1)
        self.assertIsNone(s.generate_entry_signal(ctx))


# ---------------------------------------------------------------------------
# Universe filters
# ---------------------------------------------------------------------------

class TestUniverseFilters(_RegimeTempMixin, unittest.TestCase):
    def test_low_volume_blocks_entry(self):
        closes = np.linspace(150, 90, 80).tolist()
        df = _make_symbol_data(closes, rsi=30.0, volume=1_000_000.0)  # < 5M ADV
        _write_regime(self.tmp, df["date"], passing=True)
        s = self._strategy()
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=len(prepared) - 1)
        self.assertIsNone(s.generate_entry_signal(ctx))

    def test_low_price_blocks_entry(self):
        closes = np.linspace(8, 3, 80).tolist()  # ends below min_price=5
        df = _make_symbol_data(closes, rsi=30.0)
        _write_regime(self.tmp, df["date"], passing=True)
        s = self._strategy()
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=len(prepared) - 1)
        self.assertIsNone(s.generate_entry_signal(ctx))


# ---------------------------------------------------------------------------
# Block B overextension fade
# ---------------------------------------------------------------------------

class TestBlockB(_RegimeTempMixin, unittest.TestCase):
    def test_gap_up_above_band_triggers_short(self):
        # Flat-ish base so Block A (new low / EMA stack) does NOT fire, then a
        # final blow-off bar that gaps up above the upper band.
        base = list(np.linspace(100, 105, 79))
        closes = base + [130.0]
        opens = list(np.linspace(100, 105, 79)) + [125.0]  # big gap up on last bar
        bb_upper = np.full(80, 1e6)
        bb_upper[-1] = 120.0  # last close (130) > band (120)
        df = _make_symbol_data(closes, atr=2.0, rsi=70.0, opens=opens, bb_upper=bb_upper)
        _write_regime(self.tmp, df["date"], passing=True)
        s = self._strategy(gap_atr_mult=1.5)  # gap (~25) > 1.5*ATR (3)
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=len(prepared) - 1)
        sig = s.generate_entry_signal(ctx)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, TradeDirection.SHORT)
        self.assertIn("overextension", sig.reason)


# ---------------------------------------------------------------------------
# Stop loss + sizing
# ---------------------------------------------------------------------------

class TestStopAndSizing(_RegimeTempMixin, unittest.TestCase):
    def test_stop_above_price(self):
        df = _make_symbol_data([100.0] * 70, atr=2.0)
        _write_regime(self.tmp, df["date"])
        s = self._strategy(atr_sl=2.5)
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=65)
        self.assertAlmostEqual(s.calculate_initial_stop_loss(ctx), 105.0)

    def test_zero_atr_fallback(self):
        df = _make_symbol_data([100.0] * 70, atr=0.0)
        _write_regime(self.tmp, df["date"])
        s = self._strategy(atr_sl=2.5)
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=65)
        self.assertAlmostEqual(s.calculate_initial_stop_loss(ctx), 105.0)

    def test_risk_based_sizing_short(self):
        df = _make_symbol_data([100.0] * 70, atr=2.0)
        _write_regime(self.tmp, df["date"])
        s = self._strategy(risk_perc=2.0)
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=65, equity=100_000.0)
        sig = Signal.buy(size=1.0, stop_loss=104.0, direction=TradeDirection.SHORT)
        # risk 2000 / stop distance 4 = 500 shares
        self.assertAlmostEqual(s.position_size(ctx, sig), 500.0)

    def test_invalid_stop_distance_fallback(self):
        df = _make_symbol_data([100.0] * 70)
        _write_regime(self.tmp, df["date"])
        s = self._strategy()
        prepared = s.prepare_data(df)
        ctx = _make_context(prepared, index=65, capital=50_000.0)
        # stop below price is invalid for a short -> 10% capital fallback
        sig = Signal.buy(size=1.0, stop_loss=96.0, direction=TradeDirection.SHORT)
        self.assertAlmostEqual(s.position_size(ctx, sig), 0.1 * 50_000.0 / 100.0)


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

class TestExits(_RegimeTempMixin, unittest.TestCase):
    def _short_position(self, entry_date, entry_price=100.0):
        return Position(
            symbol="TEST", entry_date=entry_date, entry_price=entry_price,
            initial_quantity=100.0, current_quantity=100.0,
            direction=TradeDirection.SHORT, stop_loss=entry_price + 5.0,
        )

    def test_cover_when_close_above_ema20(self):
        # Rising series so the latest close is above EMA_20.
        closes = np.linspace(80, 140, 80).tolist()
        df = _make_symbol_data(closes)
        _write_regime(self.tmp, df["date"])
        s = self._strategy()
        prepared = s.prepare_data(df)
        idx = len(prepared) - 1
        self.assertGreater(prepared["close"].iloc[idx], prepared["ema_20_ema"].iloc[idx])
        pos = self._short_position(pd.Timestamp(prepared["date"].iloc[idx - 2]))
        ctx = _make_context(prepared, index=idx, position=pos)
        sig = s.generate_exit_signal(ctx)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.type, SignalType.SELL)
        self.assertIn("EMA_20", sig.reason)

    def test_time_exit(self):
        closes = np.linspace(150, 90, 80).tolist()  # falling: no EMA20 cover
        df = _make_symbol_data(closes)
        _write_regime(self.tmp, df["date"])
        s = self._strategy(time_exit=10)
        prepared = s.prepare_data(df)
        entry_date = pd.Timestamp(prepared["date"].iloc[0])
        idx = next(i for i in range(len(prepared))
                   if (pd.Timestamp(prepared["date"].iloc[i]) - entry_date).days >= 10)
        pos = self._short_position(entry_date)
        ctx = _make_context(prepared, index=idx, position=pos)
        sig = s.generate_exit_signal(ctx)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.type, SignalType.SELL)
        self.assertIn("holding_days", sig.reason)

    def test_no_exit_while_below_ema20_within_time(self):
        closes = np.linspace(150, 90, 80).tolist()
        df = _make_symbol_data(closes)
        _write_regime(self.tmp, df["date"])
        s = self._strategy(time_exit=30)
        prepared = s.prepare_data(df)
        entry_date = pd.Timestamp(prepared["date"].iloc[60])
        pos = self._short_position(entry_date)
        ctx = _make_context(prepared, index=62, position=pos)
        self.assertIsNone(s.generate_exit_signal(ctx))


# ---------------------------------------------------------------------------
# End-to-end engine integration (real raw + regime data)
# ---------------------------------------------------------------------------

class TestEngineIntegration(unittest.TestCase):
    def test_runs_on_real_raw_data(self):
        """Smoke test against real raw data and the shipped regime files."""
        if not os.path.exists("raw_data/daily/AAPL_daily.csv"):
            self.skipTest("raw data not available")
        if not os.path.exists("raw_data/benchmarks/BREADTH_daily.csv"):
            self.skipTest("breadth file not generated")
        df = pd.read_csv("raw_data/daily/AAPL_daily.csv")
        df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
        strategy = ShortOnlyBaseStrategy()
        engine = SingleSecurityEngine(_create_config())
        result = engine.run("AAPL", df, strategy)
        self.assertEqual(len(result.equity_curve), len(df))
        for trade in result.trades:
            self.assertEqual(str(trade.side).upper(), "SHORT")


# ---------------------------------------------------------------------------
# Config registration
# ---------------------------------------------------------------------------

class TestStrategyConfigRegistration(unittest.TestCase):
    def test_registered(self):
        StrategyConfig.reload()
        self.assertIn("ShortOnlyBaseStrategy", StrategyConfig.get_strategies())

    def test_defaults_construct_strategy(self):
        StrategyConfig.reload()
        defaults = StrategyConfig.get_defaults("ShortOnlyBaseStrategy")
        s = ShortOnlyBaseStrategy(**defaults)
        self.assertEqual(s.breadth_threshold, defaults["breadth_threshold"])
        self.assertEqual(s.atr_sl, defaults["atr_sl"])
        self.assertEqual(s.time_exit, defaults["time_exit"])

    def test_optimization_ranges_defined(self):
        StrategyConfig.reload()
        opt = StrategyConfig.get_optimization_params("ShortOnlyBaseStrategy")
        for name in ("breadth_threshold", "atr_sl", "time_exit", "risk_perc",
                     "rally_atr_mult", "gap_atr_mult"):
            self.assertIn(name, opt)


if __name__ == "__main__":
    unittest.main()
