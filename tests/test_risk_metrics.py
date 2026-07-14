"""
Tests for the P2 risk metrics: exposure, VaR/CVaR, rolling helpers, and
engine-tracked MAE/MFE per trade.
"""
import numpy as np
import pandas as pd
import pytest

from Classes.Config.config import BacktestConfig, CommissionConfig, CommissionMode
from Classes.Core.performance_metrics import CentralizedPerformanceMetrics as CPM
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Models.position import Position
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------

class TestExposure:
    def test_half_invested(self):
        curve = pd.DataFrame({
            "equity": [100.0] * 10,
            "position_value": [0, 0, 50, 50, 50, 0, 0, 50, 50, 0],
        })
        assert CPM.calculate_exposure_pct(curve) == pytest.approx(50.0)

    def test_missing_column_is_zero(self):
        curve = pd.DataFrame({"equity": [100.0, 101.0]})
        assert CPM.calculate_exposure_pct(curve) == 0.0


class TestVarCvar:
    def test_hand_computed_tail(self):
        # 20 returns: eighteen 0%, one -1%, one -3%.
        equity = [100.0]
        rets = [0.0] * 18 + [-0.01, -0.03]
        for r in rets:
            equity.append(equity[-1] * (1 + r))
        curve = pd.DataFrame({"equity": equity})
        var_pct, cvar_pct = CPM.calculate_var_cvar(curve, confidence=0.95)
        # 5% quantile of the return sample sits between -3% and -1%; CVaR
        # (mean beyond) must be at least VaR and at most the worst loss.
        assert 1.0 <= var_pct <= 3.0
        assert cvar_pct >= var_pct
        assert cvar_pct <= 3.0 + 1e-9

    def test_all_positive_returns_zero_var(self):
        curve = pd.DataFrame({"equity": np.linspace(100, 120, 30)})
        var_pct, cvar_pct = CPM.calculate_var_cvar(curve)
        assert var_pct == 0.0
        assert cvar_pct == 0.0


class TestRollingHelpers:
    def test_rolling_sharpe_shape_and_warmup(self):
        rng = np.random.default_rng(3)
        equity = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 200))
        curve = pd.DataFrame({"equity": equity})
        rs = CPM.calculate_rolling_sharpe(curve, window=63)
        assert len(rs) == len(curve)
        assert rs.iloc[:62].isna().all()
        assert rs.iloc[63:].notna().any()

    def test_rolling_volatility_positive(self):
        rng = np.random.default_rng(4)
        equity = 100 * np.cumprod(1 + rng.normal(0, 0.01, 120))
        curve = pd.DataFrame({"equity": equity})
        rv = CPM.calculate_rolling_volatility(curve, window=30)
        assert (rv.dropna() > 0).all()


# ---------------------------------------------------------------------------
# Position-level excursion tracking
# ---------------------------------------------------------------------------

class TestPositionExcursions:
    def _pos(self, direction):
        stop = 150.0 if direction == TradeDirection.SHORT else 50.0
        return Position(symbol="T", entry_date=pd.Timestamp("2024-01-01"),
                        entry_price=100.0, initial_quantity=10.0,
                        current_quantity=10.0, direction=direction,
                        stop_loss=stop)

    def test_long_excursions(self):
        pos = self._pos(TradeDirection.LONG)
        pos.update_excursions(high=105.0, low=97.0)   # +5% / -3%
        pos.update_excursions(high=110.0, low=99.0)   # +10% / -1%
        pos.update_excursions(high=104.0, low=95.0)   # +4% / -5%
        assert pos.max_favorable_excursion_pct == pytest.approx(10.0)
        assert pos.max_adverse_excursion_pct == pytest.approx(-5.0)

    def test_short_excursions_mirrored(self):
        pos = self._pos(TradeDirection.SHORT)
        pos.update_excursions(high=105.0, low=97.0)   # fav +3% / adv -5%
        pos.update_excursions(high=112.0, low=99.0)   # fav +1% / adv -12%
        assert pos.max_favorable_excursion_pct == pytest.approx(3.0)
        assert pos.max_adverse_excursion_pct == pytest.approx(-12.0)

    def test_never_positive_mae_or_negative_mfe(self):
        pos = self._pos(TradeDirection.LONG)
        pos.update_excursions(high=101.0, low=100.5)  # never adverse
        assert pos.max_adverse_excursion_pct == 0.0
        assert pos.max_favorable_excursion_pct == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Engine integration: trades carry MAE/MFE, metrics aggregate them
# ---------------------------------------------------------------------------

class OneLongStrategy(BaseStrategy):
    _validate_on_init = False

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index == 1:
            return Signal.buy(size=1.0, stop_loss=context.current_price * 0.5,
                              direction=TradeDirection.LONG, reason="entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return context.current_price * 0.5

    def generate_exit_signal(self, context):
        if context.current_index == 6:
            return Signal.sell(reason="exit")
        return None

    def position_size(self, context, signal):
        return (context.available_capital * 0.5) / context.current_price


class TestEngineTracksExcursions:
    def _run(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        close = [100, 100, 102, 104, 103, 105, 104, 104]
        high = [c * 1.02 for c in close]   # bar 5 high = 107.1 -> +7.1% MFE
        low = [c * 0.95 for c in close]    # bar 2 low = 96.9  -> -3.1% MAE
        data = pd.DataFrame({"date": dates, "close": close, "open": close,
                             "high": high, "low": low, "volume": 1000})
        config = BacktestConfig(
            initial_capital=100_000.0,
            commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
            slippage_percent=0.0,
        )
        return SingleSecurityEngine(config).run("T", data, OneLongStrategy())

    def test_trade_has_excursions(self):
        result = self._run()
        trade = result.trades[0]
        # Entry at bar-1 close (100). Max high after entry: 105*1.02=107.1.
        assert trade.mfe_pct == pytest.approx(7.1, abs=0.01)
        # Min low after entry: 102*0.95=96.9 -> -3.1%.
        assert trade.mae_pct == pytest.approx(-3.1, abs=0.01)
        # MFE must bound the realised return from above.
        assert trade.mfe_pct >= trade.pl_pct

    def test_metrics_aggregate_excursions(self):
        result = self._run()
        m = CPM.calculate_all_metrics(result.equity_curve, result.trades)
        assert m["avg_mfe_pct"] == pytest.approx(7.1, abs=0.01)
        assert m["avg_mae_pct"] == pytest.approx(-3.1, abs=0.01)
        assert m["exposure_pct"] > 0

    def test_trade_to_dict_exports_excursions(self):
        result = self._run()
        d = result.trades[0].to_dict()
        assert "mfe_pct" in d and "mae_pct" in d
