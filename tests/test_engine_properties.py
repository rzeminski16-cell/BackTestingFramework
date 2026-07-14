"""
Property-based engine invariants (hypothesis).

Instead of hand-picked scenarios, these tests generate random price paths,
entry/exit schedules, cost settings, and stop-adjustment sequences, and
assert the properties that must hold for EVERY backtest:

1. Ledger conservation: final_equity - initial_capital == sum(trade.pl)
2. Position quantity is never negative; partial exits cannot oversell
3. Stop monotonicity: a LONG trade's stop only ever moves up; SHORT down
4. Intrabar exit fills always lie inside the bar's traded range
5. Monte Carlo: same seed -> identical paths; equity never negative
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hs

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Engine.position_manager import PositionManager
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Models.position import Position
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext

SETTINGS = dict(max_examples=30, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #

@hs.composite
def price_paths(draw, min_bars=20, max_bars=50):
    """Random-walk close series, strictly positive, with OHLC around it."""
    n = draw(hs.integers(min_bars, max_bars))
    seed = draw(hs.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.02, n)
    closes = 100.0 * np.exp(np.cumsum(steps))
    dates = pd.date_range("2022-01-03", periods=n, freq="D")
    spread = np.abs(rng.normal(0.005, 0.003, n))
    return pd.DataFrame({
        "date": dates,
        "close": closes,
        "open": closes * (1 + rng.normal(0, 0.004, n)),
        "high": closes * (1 + spread),
        "low": closes * (1 - spread),
        "volume": 1000,
    })


class RandomScheduleStrategy(BaseStrategy):
    """Entries/exits/partials at hypothesis-chosen bars; wide safe stop."""

    _validate_on_init = False

    def __init__(self, direction, entry_bars, exit_bars, partial_bars,
                 stop_adjustments, **params):
        self._direction = direction
        self._entries = set(entry_bars)
        self._exits = set(exit_bars)
        self._partials = set(partial_bars)
        # bar -> arbitrary proposed stop multiplier (engine must keep stops
        # monotone regardless of what we propose)
        self._stop_adjustments = dict(stop_adjustments)
        super().__init__(**params)

    @property
    def trade_direction(self):
        return self._direction

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index in self._entries:
            return Signal.buy(size=1.0,
                              stop_loss=self.calculate_initial_stop_loss(context),
                              direction=self._direction, reason="prop entry")
        return None

    def calculate_initial_stop_loss(self, context):
        if self._direction == TradeDirection.SHORT:
            return context.current_price * 10.0
        return context.current_price * 0.05

    def generate_exit_signal(self, context):
        if context.current_index in self._exits:
            return Signal.sell(reason="prop exit")
        return None

    def should_partial_exit(self, context):
        if context.current_index in self._partials:
            return 0.3
        return None

    def should_adjust_stop(self, context):
        mult = self._stop_adjustments.get(context.current_index)
        if mult is None:
            return None
        # Deliberately propose stops on BOTH sides of the current price;
        # the engine's direction-aware guard must filter them.
        return context.current_price * mult

    def position_size(self, context, signal):
        return (context.available_capital * 0.4) / context.current_price


@hs.composite
def scenarios(draw):
    data = draw(price_paths())
    n = len(data)
    direction = draw(hs.sampled_from([TradeDirection.LONG, TradeDirection.SHORT]))
    entry_bars = draw(hs.sets(hs.integers(1, n - 2), min_size=1, max_size=4))
    exit_bars = draw(hs.sets(hs.integers(1, n - 1), max_size=4))
    partial_bars = draw(hs.sets(hs.integers(1, n - 1), max_size=2))
    stop_adjustments = draw(hs.dictionaries(
        hs.integers(1, n - 1), hs.floats(0.5, 1.5), max_size=5))
    commission = draw(hs.sampled_from([0.0, 0.001, 0.01]))
    slippage = draw(hs.sampled_from([0.0, 0.1, 0.5]))
    intrabar = draw(hs.booleans())
    return (data, direction, entry_bars, exit_bars, partial_bars,
            stop_adjustments, commission, slippage, intrabar)


def _run_scenario(scenario):
    (data, direction, entry_bars, exit_bars, partial_bars,
     stop_adjustments, commission, slippage, intrabar) = scenario
    config = BacktestConfig(
        initial_capital=100_000.0,
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE,
                                    value=commission),
        slippage_percent=slippage,
        intrabar_stops=intrabar,
    )
    strategy = RandomScheduleStrategy(direction, entry_bars, exit_bars,
                                      partial_bars, stop_adjustments)
    engine = SingleSecurityEngine(config)
    result = engine.run("PROP", data, strategy)
    return result


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #

class TestLedgerConservation:
    @given(scenario=scenarios())
    @settings(**SETTINGS)
    def test_trades_explain_final_equity(self, scenario):
        result = _run_scenario(scenario)
        ledger = result.final_equity - 100_000.0
        trades = sum(t.pl for t in result.trades)
        assert trades == pytest.approx(ledger, abs=0.05), (
            f"ledger {ledger:.4f} != trades {trades:.4f} "
            f"({len(result.trades)} trades)")

    @given(scenario=scenarios())
    @settings(**SETTINGS)
    def test_equity_curve_is_finite(self, scenario):
        result = _run_scenario(scenario)
        equity = result.equity_curve["equity"]
        assert np.isfinite(equity).all()
        # equity must always equal cash + marked position value (recorded
        # columns must be self-consistent)
        recomputed = (result.equity_curve["capital"]
                      + result.equity_curve["position_value"])
        assert np.allclose(equity, recomputed)


class TestStopMonotonicity:
    @given(scenario=scenarios())
    @settings(**SETTINGS)
    def test_final_stop_never_loosens(self, scenario):
        """Whatever stops the strategy proposes, a LONG trade's final stop
        must be >= its initial stop and a SHORT's <= (engine guard)."""
        result = _run_scenario(scenario)
        for t in result.trades:
            if t.initial_stop_loss is None or t.final_stop_loss is None:
                continue
            if t.side == "SHORT":
                assert t.final_stop_loss <= t.initial_stop_loss + 1e-9
            else:
                assert t.final_stop_loss >= t.initial_stop_loss - 1e-9


class TestQuantityInvariants:
    @given(fractions=hs.lists(hs.floats(0.05, 0.95), min_size=1, max_size=6))
    @settings(**SETTINGS)
    def test_partial_exits_never_oversell(self, fractions):
        pos = Position(symbol="P", entry_date=pd.Timestamp("2024-01-01"),
                       entry_price=100.0, initial_quantity=100.0,
                       current_quantity=100.0,
                       direction=TradeDirection.LONG, stop_loss=50.0)
        from Classes.Models.position import PartialExit
        for i, frac in enumerate(fractions):
            qty = pos.current_quantity * frac
            pos.add_partial_exit(PartialExit(
                date=pd.Timestamp("2024-01-02"), quantity=qty, price=101.0))
            assert pos.current_quantity >= 0

    def test_overselling_raises(self):
        from Classes.Models.position import PartialExit
        pos = Position(symbol="P", entry_date=pd.Timestamp("2024-01-01"),
                       entry_price=100.0, initial_quantity=10.0,
                       current_quantity=10.0,
                       direction=TradeDirection.LONG, stop_loss=50.0)
        with pytest.raises(ValueError):
            pos.add_partial_exit(PartialExit(
                date=pd.Timestamp("2024-01-02"), quantity=11.0, price=101.0))


class TestIntrabarFillBounds:
    @given(
        entry=hs.floats(50.0, 150.0),
        stop_frac=hs.floats(0.5, 0.99),
        tp_frac=hs.floats(1.01, 1.5),
        bar_seed=hs.integers(0, 2**31 - 1),
    )
    @settings(**SETTINGS)
    def test_long_fill_inside_traded_range(self, entry, stop_frac, tp_frac,
                                           bar_seed):
        rng = np.random.default_rng(bar_seed)
        bar_open = entry * float(rng.uniform(0.7, 1.3))
        bar_high = max(bar_open, entry * float(rng.uniform(0.9, 1.4)))
        bar_low = min(bar_open, entry * float(rng.uniform(0.6, 1.1)))

        pm = PositionManager()
        pm.position = Position(
            symbol="F", entry_date=pd.Timestamp("2024-01-01"),
            entry_price=entry, initial_quantity=10.0, current_quantity=10.0,
            direction=TradeDirection.LONG,
            stop_loss=entry * stop_frac, take_profit=entry * tp_frac)

        fill = pm.check_exits_intrabar(bar_open, bar_high, bar_low)
        if fill is not None:
            price, _reason = fill
            assert bar_low - 1e-9 <= price <= bar_high + 1e-9, (
                f"fill {price} outside bar [{bar_low}, {bar_high}]")


class TestMonteCarloProperties:
    @given(seed=hs.integers(0, 2**31 - 1),
           risk=hs.floats(0.005, 0.05))
    @settings(max_examples=15, deadline=None)
    def test_seed_reproducible_and_non_negative(self, seed, risk):
        from Classes.MonteCarloAnalysis.config import SimulationConfig
        from Classes.MonteCarloAnalysis.simulator import MonteCarloSimulator
        pool = np.random.default_rng(7).normal(0.001, 0.05, 200)
        cfg = SimulationConfig(num_simulations=50, num_trades=60,
                               initial_capital=10_000.0,
                               risk_per_trade=risk, random_seed=seed)
        a = MonteCarloSimulator(cfg).run(pool)
        b = MonteCarloSimulator(cfg).run(pool)
        np.testing.assert_array_equal(a.equity_curves, b.equity_curves)
        assert (a.equity_curves >= 0).all()
