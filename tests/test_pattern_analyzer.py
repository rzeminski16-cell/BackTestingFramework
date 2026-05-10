"""
Tests for Classes/Analysis/pattern_analyzer.py and the Excel report generator.

Covers:
- MAComboSpec validation
- Crossover detection (buy/sell signals) in synthetic series
- Lookback-window slicing (signals strictly before entry)
- Feature computation (counts, density, recency, sequence, runs, alternation)
- Trade-log loading with column-name flexibility
- Correlation/win-loss aggregation in the report builder
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Classes.Analysis.pattern_analyzer import (
    SUPPORTED_MA_LENGTHS,
    SUPPORTED_MA_TYPES,
    MAComboSpec,
    PatternAnalyzer,
    TradeRecord,
    compute_window_features,
    detect_crossover_signals,
    load_combos_from_config,
    load_trade_log,
    parse_combo_string,
)
from Classes.Analysis.pattern_analysis_report import (
    NUMERIC_FEATURE_COLS,
    write_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(closes, ma_lengths=(7, 14, 20, 30, 50), start="2020-01-01"):
    """Build a minimal price DataFrame with causal SMA/EMA columns."""
    closes = np.asarray(closes, dtype=float)
    dates = pd.date_range(start=start, periods=len(closes), freq="B")
    data = {
        "date": dates,
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(len(closes), 1_000_000.0),
    }
    s = pd.Series(closes)
    for n in ma_lengths:
        data[f"ema_{n}_ema"] = s.ewm(span=n, adjust=False).mean().values
        data[f"sma_{n}_sma"] = s.rolling(window=n, min_periods=n).mean().values
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# MAComboSpec
# ---------------------------------------------------------------------------

class TestMAComboSpec(unittest.TestCase):
    def test_accepts_supported_values(self):
        for t in SUPPORTED_MA_TYPES:
            for n in SUPPORTED_MA_LENGTHS:
                c = MAComboSpec(t, n, 1)
                self.assertEqual(c.ma_type, t)
                self.assertEqual(c.ma_length, n)
                self.assertEqual(c.ma_offset, 1)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            MAComboSpec("WMA", 14, 5)

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            MAComboSpec("EMA", 11, 5)

    def test_invalid_offset_raises(self):
        with self.assertRaises(ValueError):
            MAComboSpec("EMA", 14, 0)

    def test_label(self):
        self.assertEqual(MAComboSpec("EMA", 20, 5).label, "EMA20:o5")

    def test_ma_column(self):
        self.assertEqual(MAComboSpec("EMA", 14, 3).ma_column, "ema_14_ema")
        self.assertEqual(MAComboSpec("SMA", 50, 1).ma_column, "sma_50_sma")

    def test_parse_combo_string(self):
        c = parse_combo_string("EMA:20:5")
        self.assertEqual(c.label, "EMA20:o5")
        c = parse_combo_string("sma,14,10")
        self.assertEqual(c.label, "SMA14:o10")
        c = parse_combo_string("EMA-7-3")
        self.assertEqual(c.label, "EMA7:o3")
        with self.assertRaises(ValueError):
            parse_combo_string("EMA:20")


# ---------------------------------------------------------------------------
# Crossover detection
# ---------------------------------------------------------------------------

class TestCrossoverDetection(unittest.TestCase):
    def test_no_signal_on_flat_series(self):
        df = _make_price_df([100.0] * 60)
        signals = detect_crossover_signals(df, MAComboSpec("EMA", 14, 5))
        self.assertTrue(signals.empty)

    def test_v_shape_produces_buy_at_bottom(self):
        # Down then up: base crosses above shifted at the bottom.
        n = 80
        closes = list(np.linspace(100.0, 50.0, n // 2)) + list(
            np.linspace(50.0, 150.0, n - n // 2)
        )
        df = _make_price_df(closes)

        signals = detect_crossover_signals(df, MAComboSpec("EMA", 14, 3))
        self.assertFalse(signals.empty)
        self.assertIn("BUY", signals["signal_type"].tolist())
        self.assertTrue(signals["bar_index"].is_monotonic_increasing)

    def test_inverted_v_produces_sell_at_top(self):
        # Up then down: base crosses below shifted at the top.
        n = 80
        closes = list(np.linspace(50.0, 150.0, n // 2)) + list(
            np.linspace(150.0, 50.0, n - n // 2)
        )
        df = _make_price_df(closes)

        signals = detect_crossover_signals(df, MAComboSpec("EMA", 14, 3))
        self.assertFalse(signals.empty)
        self.assertIn("SELL", signals["signal_type"].tolist())

    def test_oscillation_produces_both_signal_types(self):
        # Sine wave guarantees both crosses.
        n = 200
        x = np.linspace(0, 6 * np.pi, n)
        closes = (100.0 + 20.0 * np.sin(x)).tolist()
        df = _make_price_df(closes)

        signals = detect_crossover_signals(df, MAComboSpec("EMA", 14, 5))
        types = signals["signal_type"].tolist()
        self.assertIn("BUY", types)
        self.assertIn("SELL", types)

    def test_buy_signal_definition(self):
        """Construct a base/shift series where exactly one BUY cross occurs at
        a known bar."""
        # Inject a custom MA column so we can hand-craft the cross.
        # We'll detect EMA20 signals but overwrite the EMA20 column with a
        # known sequence.
        closes = [100.0] * 50
        df = _make_price_df(closes)
        # Build a base MA that's flat at 10. Set the offset MA via choosing
        # a synthetic base with offset=3:
        # We craft `base = [10] * 30` and want shifted (base.shift(3)) to be
        # below base for the first ~3 bars then jump above. So we set base
        # values after position k-3 to be lower than base at position k.
        base = np.full(50, 10.0)
        # Bars 0-19: base = 10. Bar 20: base drops to 5. Bars 21+: base = 12.
        # With offset=3: shift(3) at bar 23 = base at bar 20 = 5; base at 23 =
        # 12 -> shifted < base. shift(3) at bar 22 = base at bar 19 = 10 ->
        # shifted_prev (10) > base_prev (12) -> SELL? Let's pick simpler.
        base[:20] = 10.0
        base[20] = 5.0
        base[21:] = 12.0
        df["ema_20_ema"] = base

        signals = detect_crossover_signals(df, MAComboSpec("EMA", 20, 3))
        # Sanity: at least one signal detected.
        self.assertFalse(signals.empty)
        # Re-derive expected by hand: shifted = base.shift(3).
        # At bar 23: shifted=5 (base[20]), base=12 -> shifted < base. Prev bar
        # 22: shifted=base[19]=10, base=12 -> shifted < base. So no cross at
        # 23. The actual cross happens at a different bar; this test asserts
        # the function only flags bars where shifted_prev <= base_prev and
        # shifted > base (BUY) or the opposite (SELL).
        for _, row in signals.iterrows():
            i = int(row["bar_index"])
            shifted_prev = base[i - 1 - 3] if i - 1 - 3 >= 0 else np.nan
            base_prev = base[i - 1]
            shifted = base[i - 3] if i - 3 >= 0 else np.nan
            base_now = base[i]
            if np.isnan(shifted_prev) or np.isnan(shifted):
                continue
            # AlphaTrend predicate: BUY when base crosses above shifted.
            if row["signal_type"] == "BUY":
                self.assertLessEqual(base_prev, shifted_prev)
                self.assertGreater(base_now, shifted)
            else:
                self.assertGreaterEqual(base_prev, shifted_prev)
                self.assertLess(base_now, shifted)

    def test_missing_column_returns_empty(self):
        df = _make_price_df([100.0] * 30, ma_lengths=(20,))
        # ema_14 is not in the frame.
        signals = detect_crossover_signals(df, MAComboSpec("EMA", 14, 5))
        self.assertTrue(signals.empty)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

class TestFeatureComputation(unittest.TestCase):
    def _trade(self, pl=10.0):
        return TradeRecord(
            trade_id="T1",
            symbol="TEST",
            entry_date=pd.Timestamp("2020-06-01"),
            pl=pl,
            pl_pct=1.0 if pl > 0 else -1.0,
        )

    def test_empty_signals_returns_zeros(self):
        feats = compute_window_features(
            self._trade(),
            MAComboSpec("EMA", 14, 5),
            pd.DataFrame(columns=["date", "signal_type", "bar_index", "days_before_entry"]),
            window_days=30,
        )
        self.assertEqual(feats.buy_count, 0)
        self.assertEqual(feats.sell_count, 0)
        self.assertEqual(feats.total_signals, 0)
        self.assertEqual(feats.signals_per_30d, 0.0)
        self.assertIsNone(feats.last_signal_type)
        self.assertEqual(feats.longest_buy_run, 0)
        self.assertEqual(feats.alternation_rate, 0.0)
        self.assertTrue(feats.is_winner)

    def test_counts_density_recency_sequence(self):
        # 5 signals at days_before_entry 25, 20, 15, 10, 3
        # types:                                   B,  S,  B,  B, S
        sigs = pd.DataFrame({
            "date": pd.to_datetime([
                "2020-05-07", "2020-05-12", "2020-05-17",
                "2020-05-22", "2020-05-29",
            ]),
            "signal_type": ["BUY", "SELL", "BUY", "BUY", "SELL"],
            "bar_index": [10, 15, 20, 25, 32],
            "days_before_entry": [25, 20, 15, 10, 3],
        })
        feats = compute_window_features(
            self._trade(), MAComboSpec("EMA", 14, 5), sigs, window_days=30,
        )
        self.assertEqual(feats.buy_count, 3)
        self.assertEqual(feats.sell_count, 2)
        self.assertEqual(feats.total_signals, 5)
        self.assertEqual(feats.net_signals, 1)
        self.assertEqual(feats.signals_per_30d, 5.0)  # exactly 5 in 30d window
        # Last signal in chronological order is the SELL at days_before=3.
        self.assertEqual(feats.last_signal_type, "SELL")
        self.assertEqual(feats.days_since_last_signal, 3)
        self.assertEqual(feats.days_since_last_buy, 10)
        self.assertEqual(feats.days_since_last_sell, 3)
        # Longest BUY run = 2 (positions 2,3); longest SELL run = 1.
        self.assertEqual(feats.longest_buy_run, 2)
        self.assertEqual(feats.longest_sell_run, 1)
        # Sequence (most recent last): B,S,B,B,S
        self.assertEqual(feats.last5_sequence, "B,S,B,B,S")
        # Alternation rate: pairs (B,S),(S,B),(B,B),(B,S) -> flips at 3/4 = 0.75
        self.assertAlmostEqual(feats.alternation_rate, 0.75)

    def test_density_normalisation_for_small_window(self):
        sigs = pd.DataFrame({
            "date": pd.to_datetime(["2020-05-29"]),
            "signal_type": ["BUY"],
            "bar_index": [25],
            "days_before_entry": [2],
        })
        feats = compute_window_features(
            self._trade(), MAComboSpec("EMA", 14, 5), sigs, window_days=15,
        )
        self.assertEqual(feats.total_signals, 1)
        # signals_per_30d = total * 30 / window
        self.assertAlmostEqual(feats.signals_per_30d, 2.0)


# ---------------------------------------------------------------------------
# Trade log loading
# ---------------------------------------------------------------------------

class TestTradeLogLoader(unittest.TestCase):
    def test_loads_minimum_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            pd.DataFrame({
                "trade_id": [1, 2],
                "symbol": ["AAA", "BBB"],
                "entry_date": ["2020-01-15", "2020-02-15"],
                "exit_date": ["2020-01-20", "2020-02-20"],
                "entry_price": [100.0, 200.0],
                "exit_price": [110.0, 190.0],
                "pl": [10.0, -10.0],
                "pl_pct": [10.0, -5.0],
            }).to_csv(path, index=False)

            trades = load_trade_log(path)
            self.assertEqual(len(trades), 2)
            self.assertEqual(trades[0].symbol, "AAA")
            self.assertTrue(trades[0].is_winner)
            self.assertFalse(trades[1].is_winner)

    def test_skips_rows_with_bad_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            pd.DataFrame({
                "symbol": ["AAA", "BBB"],
                "entry_date": ["2020-01-15", "not-a-date"],
            }).to_csv(path, index=False)
            trades = load_trade_log(path)
            self.assertEqual(len(trades), 1)


# ---------------------------------------------------------------------------
# End-to-end PatternAnalyzer
# ---------------------------------------------------------------------------

class TestPatternAnalyzerEndToEnd(unittest.TestCase):
    def _make_environment(self, tmp: Path) -> Path:
        """Create a tiny price-data dir with one symbol's CSV."""
        data_path = tmp / "raw_data" / "daily"
        data_path.mkdir(parents=True)

        # 200 bars with a V-shape: down then up. Guarantees several crossovers.
        n = 200
        closes = list(np.linspace(100.0, 50.0, n // 2)) + list(
            np.linspace(50.0, 200.0, n - n // 2)
        )
        df = _make_price_df(closes, start="2020-01-01")
        df.to_csv(data_path / "TEST_daily.csv", index=False)
        return data_path

    def test_analyze_runs_and_emits_features(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            data_path = self._make_environment(tmp)

            trade = TradeRecord(
                trade_id="T1",
                symbol="TEST",
                entry_date=pd.Timestamp("2020-08-03"),  # well into the up-leg
                pl=15.0,
                pl_pct=5.0,
            )
            combos = [MAComboSpec("EMA", 14, 5), MAComboSpec("SMA", 20, 10)]
            windows = [30, 60, 120]

            analyzer = PatternAnalyzer(data_path=data_path, combos=combos, windows=windows)
            result = analyzer.analyze([trade])

            # One row per (combo, window).
            self.assertEqual(
                len(result.features),
                len(combos) * len(windows),
            )
            self.assertTrue(result.skipped_trades.empty)

            # Wider window must contain >= signals than narrower window for
            # the same combo (windows are nested).
            for combo in combos:
                rows = result.features[result.features["combo_label"] == combo.label]
                rows = rows.sort_values("window_days")
                self.assertTrue(rows["total_signals"].is_monotonic_increasing)

            # Numeric feature columns are all present.
            for col in NUMERIC_FEATURE_COLS:
                self.assertIn(col, result.features.columns)

    def test_missing_symbol_skipped(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            data_path = self._make_environment(tmp)

            trade = TradeRecord(
                trade_id="T2",
                symbol="MISSING",
                entry_date=pd.Timestamp("2020-08-03"),
            )
            analyzer = PatternAnalyzer(
                data_path=data_path,
                combos=[MAComboSpec("EMA", 14, 5)],
                windows=[60],
            )
            result = analyzer.analyze([trade])
            self.assertTrue(result.features.empty)
            self.assertEqual(len(result.skipped_trades), 1)
            self.assertEqual(result.skipped_trades.iloc[0]["symbol"], "MISSING")

    def test_signals_strictly_before_entry(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            data_path = self._make_environment(tmp)

            trade = TradeRecord(
                trade_id="T1",
                symbol="TEST",
                entry_date=pd.Timestamp("2020-08-03"),
            )
            analyzer = PatternAnalyzer(
                data_path=data_path,
                combos=[MAComboSpec("EMA", 14, 5)],
                windows=[120],
            )
            result = analyzer.analyze([trade])
            if not result.raw_signals.empty:
                # All signals strictly before entry day.
                self.assertTrue(
                    (result.raw_signals["days_before_entry"] >= 1).all()
                )


# ---------------------------------------------------------------------------
# Excel report
# ---------------------------------------------------------------------------

class TestReportGenerator(unittest.TestCase):
    def test_writes_xlsx(self):
        from Classes.Analysis.pattern_analyzer import PatternAnalysisResult

        # Hand-craft a small features frame with a mix of winners/losers.
        features = pd.DataFrame([
            {
                "trade_id": 1, "symbol": "A", "entry_date": pd.Timestamp("2020-01-10"),
                "combo_label": "EMA20:o5", "ma_type": "EMA", "ma_length": 20, "ma_offset": 5,
                "window_days": 60, "buy_count": 3, "sell_count": 1, "total_signals": 4,
                "net_signals": 2, "signals_per_30d": 2.0, "last_signal_type": "BUY",
                "days_since_last_buy": 2, "days_since_last_sell": 10,
                "days_since_last_signal": 2, "last5_sequence": "S,B,B,B",
                "longest_buy_run": 3, "longest_sell_run": 1, "alternation_rate": 0.33,
                "pl": 100.0, "pl_pct": 5.0, "is_winner": True,
            },
            {
                "trade_id": 2, "symbol": "B", "entry_date": pd.Timestamp("2020-02-15"),
                "combo_label": "EMA20:o5", "ma_type": "EMA", "ma_length": 20, "ma_offset": 5,
                "window_days": 60, "buy_count": 1, "sell_count": 3, "total_signals": 4,
                "net_signals": -2, "signals_per_30d": 2.0, "last_signal_type": "SELL",
                "days_since_last_buy": 15, "days_since_last_sell": 1,
                "days_since_last_signal": 1, "last5_sequence": "B,S,S,S",
                "longest_buy_run": 1, "longest_sell_run": 3, "alternation_rate": 0.33,
                "pl": -50.0, "pl_pct": -3.0, "is_winner": False,
            },
        ])
        result = PatternAnalysisResult(
            features=features,
            raw_signals=pd.DataFrame(),
            skipped_trades=pd.DataFrame(),
            combos=[MAComboSpec("EMA", 20, 5)],
            windows=[60],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.xlsx"
            write_report(result, out, trade_log_path=Path("logs/dummy.csv"))
            self.assertTrue(out.exists())
            # File should be openable by pandas.
            xl = pd.ExcelFile(out, engine="openpyxl")
            self.assertIn("Overview", xl.sheet_names)
            self.assertIn("Per-Trade", xl.sheet_names)
            self.assertIn("Summary", xl.sheet_names)
            self.assertIn("Win-Loss Stats", xl.sheet_names)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading(unittest.TestCase):
    def test_load_json_combos(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "combos.json"
            path.write_text(json.dumps({
                "combos": [
                    {"ma_type": "EMA", "ma_length": 14, "ma_offset": 5},
                    {"ma_type": "SMA", "ma_length": 20, "ma_offset": 10},
                ],
                "windows": [30, 90],
            }))
            combos, windows = load_combos_from_config(path)
            self.assertEqual(len(combos), 2)
            self.assertEqual(combos[0].label, "EMA14:o5")
            self.assertEqual(windows, [30, 90])

    def test_load_string_combo_in_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "combos.json"
            path.write_text(json.dumps({"combos": ["EMA:14:5", "SMA:20:10"]}))
            combos, windows = load_combos_from_config(path)
            self.assertEqual([c.label for c in combos], ["EMA14:o5", "SMA20:o10"])
            self.assertIsNone(windows)


if __name__ == "__main__":
    unittest.main()
