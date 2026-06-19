"""
Tests for the Modelling & Evaluation stage.

Builds a small synthetic run package on disk (trades with overlapping label
windows + an index panel) and exercises loading, leakage-safe feature joins, the
purged/embargoed splitter, the Adjusted RAR% reuse, an end-to-end controller run,
and the exportable scoring-function round-trip.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from Classes.Modelling import (ModellingConfig, ModellingController, ModellingView,
                               RunPackageLoader, ScoringFunction, TargetKind)
from Classes.Modelling.adjusted_rar import (adjusted_rar_from_equity,
                                            build_daily_equity_curve)
from Classes.Modelling.features import FeatureBuilder
from Classes.Modelling.validation_split import ChronologicalSplitter


# --------------------------------------------------------------------------- #
# Synthetic run-package fixture.
# --------------------------------------------------------------------------- #
def _provenance(df, family, freq="daily"):
    df = df.copy()
    df["run_id"] = "synthetic"
    df["family"] = family
    df["native_frequency"] = freq
    df["source_function"] = "TEST"
    df["source_vendor"] = "test"
    df["retrieved_at"] = df["available_ts"]
    df["quality_flag"] = "ok"
    return df


def build_synthetic_package(root, run_id="synthetic", n_trades=160, seed=7):
    rng = np.random.default_rng(seed)
    path = os.path.join(root, run_id)
    os.makedirs(path, exist_ok=True)

    # Daily index panel (the only feature family).
    dates = pd.date_range("2021-01-01", periods=520, freq="D")
    level = 100 + np.cumsum(rng.normal(0.05, 1.0, len(dates)))
    index_panel = pd.DataFrame({
        "entity_id": "SPX",
        "observation_date": dates,
        "available_ts": dates,         # same-day availability
        "close": level,
    })
    index_panel = _provenance(index_panel, "index_panel")
    index_panel.to_parquet(os.path.join(path, "index_panel.parquet"))

    # Trades with overlapping holding windows; outcome depends on the index trend
    # in the days before entry (a learnable, point-in-time signal) plus noise.
    idx_series = pd.Series(level, index=dates)
    entries = pd.to_datetime(rng.choice(dates[20:480], size=n_trades, replace=True))
    entries = pd.DatetimeIndex(sorted(entries))
    rows = []
    for i, ent in enumerate(entries):
        hold = int(rng.integers(3, 21))
        ex = ent + pd.Timedelta(days=hold)
        trail = idx_series.loc[:ent].tail(10)
        signal = (trail.iloc[-1] / trail.iloc[0] - 1.0) if len(trail) >= 2 else 0.0
        pl_pct = 100 * (0.6 * signal + rng.normal(0, 0.01))
        rows.append({
            "trade_id": f"T{i:06d}", "symbol": "AAA",
            "entry_date": ent, "exit_date": ex,
            "entry_price": 100.0, "quantity": 10.0, "side": "LONG",
            "security_currency": "GBP", "concurrent_positions": int(rng.integers(0, 5)),
            "pl": pl_pct * 10.0, "pl_pct": pl_pct,
        })
    trades = pd.DataFrame(rows)
    trades.to_parquet(os.path.join(path, "selected_trades.parquet"))

    entity_map = pd.DataFrame({"trade_id": trades["trade_id"], "symbol": "AAA",
                               "benchmarks": "SPX", "currency": "GBP",
                               "base_currency": "GBP", "needs_fx_conversion": False,
                               "sector": "Test"})
    entity_map.to_parquet(os.path.join(path, "entity_mapping.parquet"))

    manifest = {
        "run_name": run_id, "run_id": run_id, "generated_at": "2024-01-01T00:00:00Z",
        "base_currency": "GBP", "modelling_frequency": "daily",
        "family_toggles": {"index_panel": True},
        "timing_policies": {"index_panel": {"availability_rule": "same_day",
                                            "carry_forward_tolerance_days": 7}},
        "table_row_counts": {"selected_trades": len(trades)},
    }
    with open(os.path.join(path, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    contract = {"leakage_sensitive_fields": ["available_ts", "observation_date"],
                "join_semantics": "merge_asof backward on available_ts"}
    with open(os.path.join(path, "data_contract.json"), "w") as fh:
        json.dump(contract, fh)
    return path


@pytest.fixture
def package_root(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    build_synthetic_package(str(root))
    return str(root)


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
def test_loader_and_readiness(package_root):
    loader = RunPackageLoader(package_root)
    discovered = loader.discover()
    assert any(d["run_id"] == "synthetic" for d in discovered)
    pkg = loader.load("synthetic")
    assert not pkg.trades.empty
    assert "index_panel" in pkg.available_families()
    readiness = loader.check_readiness(pkg)
    assert readiness.ok, readiness.errors


def test_features_are_point_in_time(package_root):
    pkg = RunPackageLoader(package_root).load("synthetic")
    fm = FeatureBuilder(pkg).build_per_trade()
    # The freshness feature is (entry_date - available_ts).days and must be >= 0:
    # a negative age would mean a value from the future leaked into the feature.
    age_cols = [c for c in fm.X.columns if c.endswith("__age_days")]
    assert age_cols, "expected an as-of age feature for the index panel"
    for col in age_cols:
        ages = fm.X[col].dropna()
        assert (ages >= 0).all(), f"leakage: negative age in {col}"
    # The attached index value must be present for most trades.
    val_cols = [c for c in fm.X.columns if c.startswith("index_panel") and c.endswith("__close")]
    assert val_cols and fm.X[val_cols[0]].notna().mean() > 0.8


def test_purged_embargo_drops_overlap():
    # Two clusters of label windows; the splitter must purge train rows whose
    # window overlaps the embargoed test window.
    n = 40
    starts = pd.date_range("2021-01-01", periods=n, freq="3D")
    ends = starts + pd.Timedelta(days=10)  # heavy overlap
    sp = ChronologicalSplitter(n_splits=3, min_train_size=5, embargo_days=5,
                               label_start=pd.Series(starts), label_end=pd.Series(ends))
    ls, le = starts.values, ends.values
    folds = list(sp.split(np.zeros((n, 1))))
    assert folds
    for train, test in folds:
        t_lo = ls[test].min() - np.timedelta64(5, "D")
        t_hi = le[test].max() + np.timedelta64(5, "D")
        overlap = (ls[train] <= t_hi) & (le[train] >= t_lo)
        assert not overlap.any(), "purge/embargo left an overlapping train row"


def test_adjusted_rar_matches_house_metric():
    from Classes.Core.stable_metrics import StableMetricsCalculator
    dates = pd.date_range("2021-01-01", periods=365, freq="D")
    equity = 100_000 * np.exp(np.linspace(0, 0.2, len(dates)))
    curve = pd.DataFrame({"date": dates, "equity": equity})
    ours = adjusted_rar_from_equity(curve)
    house = StableMetricsCalculator.calculate_all(curve).rar_adjusted
    assert abs(ours - house) < 1e-6


def test_build_daily_equity_curve():
    trades = [{"exit_date": "2021-01-05", "pl": 100.0},
              {"exit_date": "2021-01-10", "pl": -40.0}]
    curve = build_daily_equity_curve(trades, initial_capital=1000.0)
    assert not curve.empty
    assert curve["equity"].iloc[-1] == pytest.approx(1060.0)


def test_end_to_end_controller_and_export(package_root):
    config = ModellingConfig(model_run_name="smoke", runs_root=package_root,
                             view=ModellingView.PER_TRADE)
    config.validation.n_splits = 3
    config.ladder.run_ensemble = False
    controller = ModellingController(config)
    readiness = controller.load_package("synthetic")
    assert readiness.ok

    results = controller.run()
    assert results.leaderboard, "no models evaluated"
    assert results.risk_register
    # A non-baseline model should have produced OOS predictions.
    assert any(ev.oos_predictions.shape[0] > 0 for ev in results.leaderboard)

    written = controller.export(results)
    assert os.path.isfile(written["model_leaderboard"])
    assert os.path.isfile(written["research_report"])
    assert os.path.isfile(written["risk_register"])


def test_scoring_function_roundtrip(package_root, tmp_path):
    config = ModellingConfig(model_run_name="sf", runs_root=package_root,
                             view=ModellingView.PER_TRADE)
    config.validation.n_splits = 3
    controller = ModellingController(config)
    controller.load_package("synthetic")
    results = controller.run()
    assert results.scoring_function is not None

    fm = results.views[0].feature_matrix
    out_dir = str(tmp_path / "sf_out")
    results.scoring_function.save(out_dir)
    loaded = ScoringFunction.load(out_dir)
    scored = loaded.apply(fm.X.head(10))
    assert {"good_score", "decision", "size_multiplier"}.issubset(scored.columns)
    assert set(scored["decision"].unique()).issubset({"allow", "reduce", "block"})
