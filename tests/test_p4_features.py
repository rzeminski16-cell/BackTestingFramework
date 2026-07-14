"""
Tests for the P4 features: strategy auto-discovery + scaffold, the Parquet
data store + ingest, and the live-paper signal bridge.
"""
import importlib.util
import inspect
import time

import pandas as pd
import pytest

from btf.__main__ import main
from Classes.Data.data_loader import DataLoader
from Classes.Data.parquet_store import ingest_directory, ingest_file
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy
from strategies.registry import (
    STRATEGY_REGISTRY, build_registry, get_strategy_class,
)


# --------------------------------------------------------------------------- #
# Auto-discovery
# --------------------------------------------------------------------------- #

KNOWN_STRATEGIES = {
    "AlphaTrendV1Strategy", "AlphaTrendV2Strategy", "AlphaTrendV2C2Strategy",
    "AlphaTrendV2C3Strategy", "AlphaTrendV3C1Strategy",
    "RandomControlStrategy", "ShortOnlyBaseStrategy",
}


class TestDiscovery:
    def test_all_known_strategies_discovered(self):
        assert KNOWN_STRATEGIES <= set(STRATEGY_REGISTRY)

    def test_no_abstract_classes(self):
        for cls in STRATEGY_REGISTRY.values():
            assert not inspect.isabstract(cls)
            assert issubclass(cls, BaseStrategy)

    def test_rebuild_is_stable(self):
        assert set(build_registry()) == set(STRATEGY_REGISTRY)

    def test_unknown_name_lists_available(self):
        with pytest.raises(KeyError, match="AlphaTrendV1Strategy"):
            get_strategy_class("MissingStrategy")


# --------------------------------------------------------------------------- #
# Scaffold generator
# --------------------------------------------------------------------------- #

class TestScaffold:
    def _generate(self, tmp_path, name="DemoAlpha", extra=()):
        code = main(["new-strategy", name, "--output-dir", str(tmp_path),
                     "--indicators", "atr_14", "rsi_14", *extra])
        assert code == 0
        files = list(tmp_path.glob("*.py"))
        assert len(files) == 1
        return files[0]

    def _load_class(self, path, class_name):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    def test_generates_working_strategy(self, tmp_path):
        path = self._generate(tmp_path)
        assert path.name == "demo_alpha_strategy.py"
        cls = self._load_class(path, "DemoAlphaStrategy")
        assert issubclass(cls, BaseStrategy)
        instance = cls()  # scaffold must instantiate out of the box
        assert instance.trade_direction == TradeDirection.LONG
        assert "atr_14" in instance.required_columns()
        assert "rsi_14" in instance.required_columns()

    def test_short_direction(self, tmp_path):
        path = self._generate(tmp_path, name="DemoBear", extra=("--direction", "short"))
        cls = self._load_class(path, "DemoBearStrategy")
        assert cls().trade_direction == TradeDirection.SHORT

    def test_refuses_overwrite_without_force(self, tmp_path, capsys):
        self._generate(tmp_path)
        code = main(["new-strategy", "DemoAlpha", "--output-dir", str(tmp_path),
                     "--indicators", "atr_14", "rsi_14"])
        assert code == 2
        assert "already exists" in capsys.readouterr().err

    def test_rejects_bad_identifier(self, capsys):
        assert main(["new-strategy", "bad-name"]) == 2
        assert "Capitalized" in capsys.readouterr().err

    def test_scaffold_stop_is_on_correct_side(self, tmp_path):
        path = self._generate(tmp_path)
        cls = self._load_class(path, "DemoAlphaStrategy")
        strategy = cls()

        class Ctx:
            current_price = 100.0
            current_index = 0
            data = pd.DataFrame({"close": [100.0], "atr_14": [2.0],
                                 "rsi_14": [50.0]})
        stop = strategy.calculate_initial_stop_loss(Ctx())
        assert stop < 100.0  # LONG stop must sit below entry


# --------------------------------------------------------------------------- #
# Parquet store
# --------------------------------------------------------------------------- #

def _write_csv(directory, name="TPQ.csv", rows=None):
    directory.mkdir(parents=True, exist_ok=True)
    if rows is None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = pd.DataFrame({"date": dates,
                             "close": [100.0 + i for i in range(10)],
                             "open": [99.5 + i for i in range(10)],
                             "volume": [1000] * 10})
    path = directory / name
    rows.to_csv(path, index=False)
    return path


class TestParquetStore:
    def test_ingest_writes_parquet(self, tmp_path):
        _write_csv(tmp_path)
        result = ingest_directory(tmp_path)
        assert result.ok
        assert len(result.written) == 1
        assert (tmp_path / "TPQ.parquet").exists()

    def test_loader_prefers_parquet_and_values_match(self, tmp_path):
        _write_csv(tmp_path)
        from_csv = DataLoader(tmp_path).load_csv("TPQ")

        ingest_directory(tmp_path)
        loader = DataLoader(tmp_path)
        assert loader._find_data_file("TPQ").suffix == ".parquet"
        from_parquet = loader.load_csv("TPQ")

        pd.testing.assert_frame_equal(
            from_csv.reset_index(drop=True),
            from_parquet[from_csv.columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_stale_parquet_falls_back_to_csv(self, tmp_path):
        csv_path = _write_csv(tmp_path)
        ingest_directory(tmp_path)
        # CSV edited after ingest -> parquet is stale -> CSV wins.
        time.sleep(0.02)
        csv_path.touch()
        assert DataLoader(tmp_path)._find_data_file("TPQ").suffix == ".csv"

    def test_up_to_date_parquet_skipped(self, tmp_path):
        _write_csv(tmp_path)
        assert len(ingest_directory(tmp_path).written) == 1
        again = ingest_directory(tmp_path)
        assert len(again.written) == 0
        assert len(again.skipped_up_to_date) == 1
        assert len(ingest_directory(tmp_path, force=True).written) == 1

    def test_validation_collapses_duplicates_and_bad_close(self, tmp_path):
        rows = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03",
                     "2024-01-04"],
            "close": [100.0, 101.0, 101.5, -5.0, 103.0],
        })
        path = _write_csv(tmp_path, rows=rows)
        result = ingest_file(path)
        assert result.ok
        assert any("duplicate" in w for w in result.warnings)
        assert any("non-positive close" in w for w in result.warnings)
        df = pd.read_parquet(tmp_path / "TPQ.parquet")
        assert len(df) == 3  # 01, 02(last), 04
        assert (df["close"] > 0).all()

    def test_unusable_csv_reports_failure(self, tmp_path):
        path = _write_csv(tmp_path, rows=pd.DataFrame({"foo": [1, 2]}))
        result = ingest_file(path)
        assert not result.ok
        assert "date" in result.failed[0]

    def test_cli_ingest(self, tmp_path, capsys):
        _write_csv(tmp_path)
        assert main(["ingest", "--data-dir", str(tmp_path)]) == 0
        assert "written:    1" in capsys.readouterr().out

    def test_symbols_listed_for_parquet_only(self, tmp_path):
        _write_csv(tmp_path)
        ingest_directory(tmp_path)
        (tmp_path / "TPQ.csv").unlink()
        loader = DataLoader(tmp_path)
        assert "TPQ" in loader.get_available_symbols()
        df = loader.load_csv("TPQ")
        assert len(df) == 10


# --------------------------------------------------------------------------- #
# Signal bridge (btf signals)
# --------------------------------------------------------------------------- #

class BridgeStrategy(BaseStrategy):
    """Enters on `entry_bar`, exits on `exit_bar` (indices; -1 = last)."""

    _validate_on_init = False
    entry_bar = None
    exit_bar = None

    @property
    def trade_direction(self):
        return TradeDirection.LONG

    def required_columns(self):
        return ["date", "close"]

    def _resolve(self, idx, context):
        n = context.data._df_len if hasattr(context.data, "_df_len") else None
        return idx

    def generate_entry_signal(self, context):
        if self.entry_bar is not None and context.current_index == self.entry_bar:
            return Signal.buy(size=1.0, stop_loss=context.current_price * 0.5,
                              direction=TradeDirection.LONG, reason="bridge entry")
        return None

    def calculate_initial_stop_loss(self, context):
        return context.current_price * 0.5

    def generate_exit_signal(self, context):
        if self.exit_bar is not None and context.current_index == self.exit_bar:
            return Signal.sell(reason="bridge exit")
        return None

    def position_size(self, context, signal):
        return (context.available_capital * 0.5) / context.current_price


def _bridge_class(entry_bar, exit_bar):
    return type("BridgeCase", (BridgeStrategy,),
                {"entry_bar": entry_bar, "exit_bar": exit_bar})


@pytest.fixture
def signals_data_dir(tmp_path):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = [100.0 + i for i in range(10)]
    df = pd.DataFrame({"date": dates, "close": prices, "open": prices,
                       "high": prices, "low": prices, "volume": 100})
    d = tmp_path / "daily"
    d.mkdir()
    df.to_csv(d / "SIG.csv", index=False)
    return d


class TestSignalBridge:
    def _run(self, monkeypatch, data_dir, tmp_path, entry_bar, exit_bar):
        import json
        monkeypatch.setitem(STRATEGY_REGISTRY, "BridgeCase",
                            _bridge_class(entry_bar, exit_bar))
        out = tmp_path / "signals.json"
        code = main(["signals", "--strategy", "BridgeCase",
                     "--symbols", "SIG", "--data-dir", str(data_dir),
                     "--no-fx", "--json", str(out)])
        assert code == 0
        return json.loads(out.read_text())[0]

    def test_enter_on_latest_bar(self, monkeypatch, signals_data_dir, tmp_path):
        sig = self._run(monkeypatch, signals_data_dir, tmp_path,
                        entry_bar=9, exit_bar=None)
        assert sig["action"] == "ENTER"
        assert sig["as_of"] == "2024-01-10"

    def test_holding_open_position(self, monkeypatch, signals_data_dir, tmp_path):
        sig = self._run(monkeypatch, signals_data_dir, tmp_path,
                        entry_bar=2, exit_bar=None)
        assert sig["action"] == "HOLDING"
        assert "2024-01-03" in sig["detail"]

    def test_exit_on_latest_bar(self, monkeypatch, signals_data_dir, tmp_path):
        sig = self._run(monkeypatch, signals_data_dir, tmp_path,
                        entry_bar=2, exit_bar=9)
        assert sig["action"] == "EXIT"

    def test_flat(self, monkeypatch, signals_data_dir, tmp_path):
        sig = self._run(monkeypatch, signals_data_dir, tmp_path,
                        entry_bar=None, exit_bar=None)
        assert sig["action"] == "FLAT"

    def test_unknown_symbol_is_error_action(self, monkeypatch, signals_data_dir,
                                            tmp_path, capsys):
        import json
        monkeypatch.setitem(STRATEGY_REGISTRY, "BridgeCase",
                            _bridge_class(None, None))
        out = tmp_path / "signals.json"
        code = main(["signals", "--strategy", "BridgeCase",
                     "--symbols", "NOPE", "--data-dir", str(signals_data_dir),
                     "--no-fx", "--json", str(out)])
        assert code == 1
        assert json.loads(out.read_text())[0]["action"] == "ERROR"
