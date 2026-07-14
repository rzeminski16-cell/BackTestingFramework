"""
Tests for the ``btf`` unified CLI.

The backtest test is hermetic: it registers a scripted strategy and runs on
synthetic data in a temp directory, then asserts on the produced trade-log
CSV and metrics JSON. Nothing touches the repo's real data.
"""
import json

import pandas as pd
import pytest

from btf.__main__ import build_parser, main
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy
from strategies.registry import STRATEGY_REGISTRY, get_strategy_class


class CliScriptedStrategy(BaseStrategy):
    """Buy bar 2, sell bar 8 — deterministic for CLI end-to-end tests."""

    _validate_on_init = False

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def generate_entry_signal(self, context):
        if context.current_index == 2:
            return Signal.buy(size=1.0, stop_loss=context.current_price * 0.5,
                              direction=TradeDirection.LONG, reason="cli entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return context.current_price * 0.5

    def generate_exit_signal(self, context):
        if context.current_index == 8:
            return Signal.sell(reason="cli exit")
        return None

    def position_size(self, context, signal):
        return (context.available_capital * 0.5) / context.current_price


@pytest.fixture
def data_dir(tmp_path):
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    prices = [100 + i for i in range(12)]
    df = pd.DataFrame({"date": dates, "close": prices, "open": prices,
                       "high": [p * 1.01 for p in prices],
                       "low": [p * 0.99 for p in prices],
                       "volume": [1000] * 12})
    d = tmp_path / "daily"
    d.mkdir()
    df.to_csv(d / "TSTA.csv", index=False)
    df.to_csv(d / "TSTB.csv", index=False)
    return d


@pytest.fixture
def scripted_registry(monkeypatch):
    monkeypatch.setitem(STRATEGY_REGISTRY, "CliScriptedStrategy",
                        CliScriptedStrategy)
    return "CliScriptedStrategy"


class TestParser:
    def test_version_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "btf" in capsys.readouterr().out

    def test_missing_command_errors(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0

    def test_all_subcommands_registered(self):
        parser = build_parser()
        help_text = parser.format_help()
        for cmd in ("list", "backtest", "optimize", "montecarlo", "dashboard"):
            assert cmd in help_text


class TestRegistry:
    def test_known_strategy(self):
        cls = get_strategy_class("AlphaTrendV1Strategy")
        assert cls.__name__ == "AlphaTrendV1Strategy"

    def test_unknown_strategy_lists_available(self):
        with pytest.raises(KeyError, match="AlphaTrendV1Strategy"):
            get_strategy_class("NopeStrategy")


class TestList:
    def test_list_strategies(self, capsys):
        assert main(["list", "strategies"]) == 0
        out = capsys.readouterr().out
        assert "AlphaTrendV1Strategy" in out
        assert "ShortOnlyBaseStrategy" in out

    def test_list_securities_custom_dir(self, capsys, data_dir):
        assert main(["list", "securities", "--data-dir", str(data_dir)]) == 0
        out = capsys.readouterr().out
        assert "TSTA" in out and "TSTB" in out


class TestBacktest:
    def test_unknown_strategy_exit_code(self, capsys):
        assert main(["backtest", "--strategy", "Nope", "--symbols", "X"]) == 2
        assert "Unknown strategy" in capsys.readouterr().err

    def test_requires_symbols_or_basket(self, capsys, scripted_registry):
        assert main(["backtest", "--strategy", scripted_registry]) == 2
        assert "--symbols" in capsys.readouterr().err

    def test_single_end_to_end(self, capsys, tmp_path, data_dir, scripted_registry):
        trades_csv = tmp_path / "out" / "trades.csv"
        metrics_json = tmp_path / "out" / "metrics.json"
        code = main([
            "backtest", "--strategy", scripted_registry,
            "--symbols", "TSTA",
            "--data-dir", str(data_dir), "--no-fx",
            "--commission", "0.0", "--slippage", "0.0",
            "--trades-csv", str(trades_csv),
            "--json", str(metrics_json),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "final_equity" in out

        trades = pd.read_csv(trades_csv)
        assert len(trades) == 1
        # Entry bar-2 close = 102, exit bar-8 close = 108.
        assert trades["entry_price"].iloc[0] == pytest.approx(102.0)
        assert trades["exit_price"].iloc[0] == pytest.approx(108.0)

        metrics = json.loads(metrics_json.read_text())
        assert metrics["total_trades"] == 1
        assert metrics["final_equity"] > 100_000.0

    def test_portfolio_end_to_end(self, tmp_path, data_dir, scripted_registry):
        trades_csv = tmp_path / "ptrades.csv"
        code = main([
            "backtest", "--strategy", scripted_registry,
            "--symbols", "TSTA", "TSTB",
            "--data-dir", str(data_dir), "--no-fx",
            "--commission", "0.0", "--slippage", "0.0",
            "--trades-csv", str(trades_csv),
        ])
        assert code == 0
        trades = pd.read_csv(trades_csv)
        assert len(trades) == 2
        assert set(trades["symbol"]) == {"TSTA", "TSTB"}

    def test_next_bar_open_flag(self, tmp_path, data_dir, scripted_registry):
        trades_csv = tmp_path / "nbo.csv"
        code = main([
            "backtest", "--strategy", scripted_registry,
            "--symbols", "TSTA",
            "--data-dir", str(data_dir), "--no-fx",
            "--commission", "0.0", "--slippage", "0.0",
            "--next-bar-open",
            "--trades-csv", str(trades_csv),
        ])
        assert code == 0
        trades = pd.read_csv(trades_csv)
        # Signal on bar 2 fills at bar-3 open (=103), exit bar 8 -> bar-9 open.
        assert trades["entry_price"].iloc[0] == pytest.approx(103.0)
        assert trades["exit_price"].iloc[0] == pytest.approx(109.0)

    def test_param_flag_parses_and_runs(self, data_dir, scripted_registry):
        # BaseStrategy accepts free-form params by design; the flag must
        # parse name=value and not break the run.
        code = main([
            "backtest", "--strategy", scripted_registry,
            "--symbols", "TSTA", "--data-dir", str(data_dir), "--no-fx",
            "--commission", "0.0", "--slippage", "0.0",
            "--param", "some_knob=1.5", "--param", "flag=true",
        ])
        assert code == 0

    def test_malformed_param_exits_nonzero(self, data_dir, scripted_registry):
        with pytest.raises(SystemExit, match="name=value"):
            main([
                "backtest", "--strategy", scripted_registry,
                "--symbols", "TSTA", "--data-dir", str(data_dir), "--no-fx",
                "--param", "broken",
            ])


class TestMonteCarlo:
    def _trade_log(self, tmp_path):
        df = pd.DataFrame({
            "trade_id": [f"T{i:03d}" for i in range(40)],
            "symbol": ["TST"] * 40,
            "entry_date": ["2024-01-01"] * 40,
            "entry_price": [100.0] * 40,
            "exit_date": ["2024-01-10"] * 40,
            "exit_price": [101.0] * 40,
            "quantity": [10.0] * 40,
            "side": ["LONG"] * 40,
            "initial_stop_loss": [95.0] * 40,
            "final_stop_loss": [95.0] * 40,
            "pl": [10.0, -5.0] * 20,
            "pl_pct": [1.0, -0.5] * 20,
        })
        path = tmp_path / "trades.csv"
        df.to_csv(path, index=False)
        return path

    def test_from_trade_log(self, capsys, tmp_path):
        log = self._trade_log(tmp_path)
        out_json = tmp_path / "mc.json"
        code = main([
            "montecarlo", "--trade-log", str(log),
            "--simulations", "200", "--steps", "50", "--seed", "1",
            "--json", str(out_json),
        ])
        assert code == 0
        metrics = json.loads(out_json.read_text())
        assert metrics["num_simulations"] == 200
        assert "probability_of_loss" in metrics

    def test_from_daily_curve_sets_annualized(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        equity = [100_000 * (1.001 ** i) for i in range(120)]
        curve = tmp_path / "equity.csv"
        pd.DataFrame({"date": dates, "equity": equity}).to_csv(curve, index=False)
        out_json = tmp_path / "mc_daily.json"
        code = main([
            "montecarlo", "--daily-curve", str(curve),
            "--simulations", "100", "--steps", "60", "--seed", "2",
            "--json", str(out_json),
        ])
        assert code == 0
        metrics = json.loads(out_json.read_text())
        assert metrics["median_annualized_sharpe"] is not None

    def test_requires_a_source(self, capsys):
        assert main(["montecarlo"]) == 2
        assert "--trade-log or --daily-curve" in capsys.readouterr().err
