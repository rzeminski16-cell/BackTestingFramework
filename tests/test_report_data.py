"""
Tests for Classes/Analysis/report_data.py — the single source of truth for
report chart series. Verifies correctness/identities that the reports rely on.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Classes.Analysis import report_data as rd


def _trade(pl, pl_pct, exit_date, entry_price=100.0, stop=90.0, exit_price=None,
           side="LONG", duration_days=5):
    if exit_price is None:
        exit_price = entry_price * (1 + pl_pct / 100.0)
    return SimpleNamespace(
        pl=pl, pl_pct=pl_pct, exit_date=pd.Timestamp(exit_date),
        entry_price=entry_price, exit_price=exit_price, initial_stop_loss=stop,
        side=side, duration_days=duration_days,
    )


def _equity_curve(n=400, start="2020-01-01", seed=1, drift=0.0006, vol=0.01, cap_frac=0.3):
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    equity = 100000 * np.cumprod(1 + rets)
    capital = equity * cap_frac
    return pd.DataFrame({
        "date": dates, "equity": equity, "capital": capital,
        "position_value": equity - capital,
    })


class TestEquityDrawdown(unittest.TestCase):
    def test_drawdown_math(self):
        df = pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=4),
                           "equity": [100.0, 110.0, 99.0, 105.0]})
        out = rd.compute_equity_drawdown(df)
        self.assertEqual(out["high_water_mark"], [100, 110, 110, 110])
        # drawdown at index 2: (99-110)/110*100 = -10%
        self.assertAlmostEqual(out["drawdown_pct"][2], -10.0, places=6)
        self.assertEqual(out["drawdown_pct"][0], 0.0)


class TestDistribution(unittest.TestCase):
    def test_counts_sum_to_n(self):
        trades = [_trade(10, 5.0, "2021-01-05"), _trade(-5, -2.5, "2021-01-06"),
                  _trade(20, 8.0, "2021-01-07")] * 5
        out = rd.compute_trade_return_distribution(trades)
        self.assertEqual(out["n"], len(trades))
        self.assertEqual(sum(out["counts"]), len(trades))  # every trade lands in a bin
        self.assertEqual(len(out["labels"]), len(out["counts"]))


class TestRMultiple(unittest.TestCase):
    def test_known_r_multiples(self):
        # entry 100, stop 90 -> R=10. exit 120 -> +2R; exit 80 -> -2R; exit 110 -> +1R
        trades = [
            _trade(20, 20, "2021-01-05", exit_price=120),
            _trade(-20, -20, "2021-01-06", exit_price=80),
            _trade(10, 10, "2021-01-07", exit_price=110),
        ]
        out = rd.compute_r_multiple_distribution(trades)
        self.assertTrue(out["available"])
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["n_win"], 2)
        self.assertEqual(out["n_loss"], 1)
        self.assertEqual(sum(out["counts"]), 3)  # all trades binned
        self.assertAlmostEqual(out["avg_r"], (2 + -2 + 1) / 3, places=6)
        self.assertAlmostEqual(out["win_rate"], 2 / 3 * 100, places=6)


class TestMonthlyReturns(unittest.TestCase):
    def test_structure_and_totals(self):
        out = rd.compute_monthly_returns(_equity_curve(n=500))
        self.assertEqual(out["months"], rd.MONTH_NAMES)
        self.assertEqual(len(out["rows"]), len(out["years"]))
        self.assertEqual(len(out["year_totals"]), len(out["years"]))
        for row in out["rows"]:
            self.assertEqual(len(row), 12)
        # year total == sum of present months
        for row, total in zip(out["rows"], out["year_totals"]):
            present = [v for v in row if v is not None]
            self.assertAlmostEqual(total, sum(present), places=6)


class TestRollingMetrics(unittest.TestCase):
    def test_series_aligned(self):
        out = rd.compute_rolling_metrics(_equity_curve(n=400), window=90)
        self.assertTrue(out["available"])
        n = len(out["dates"])
        self.assertEqual(len(out["sharpe"]), n)
        self.assertEqual(len(out["sortino"]), n)
        self.assertEqual(len(out["volatility"]), n)

    def test_too_short(self):
        out = rd.compute_rolling_metrics(_equity_curve(n=50), window=90)
        self.assertFalse(out["available"])


class TestWinRateOverTime(unittest.TestCase):
    def test_window_and_caps(self):
        trades = [_trade(10 if i % 2 else -5, 5 if i % 2 else -2.5, f"2021-{(i%12)+1:02d}-01")
                  for i in range(40)]
        out = rd.compute_win_rate_over_time(trades, window=20)
        self.assertTrue(out["available"])
        self.assertEqual(len(out["win_rates"]), len(out["dates"]))
        self.assertTrue(all(0 <= w <= 100 for w in out["win_rates"]))
        self.assertTrue(all(pf <= 10 for pf in out["profit_factors"]))


class TestContribution(unittest.TestCase):
    def test_sums(self):
        out = rd.compute_contribution({"AAA": 100.0, "BBB": -40.0, "CCC": 60.0})
        self.assertAlmostEqual(sum(out["pnls"]), out["total"], places=6)
        self.assertAlmostEqual(sum(out["shares"]), 100.0, places=4)
        self.assertEqual(out["symbols"][0], "AAA")  # largest |contribution| first


class TestCapitalUtilization(unittest.TestCase):
    def test_stacks_to_equity(self):
        df = _equity_curve(n=20)
        out = rd.compute_capital_utilization(df)
        self.assertTrue(out["available"])
        for cash, invested, eq in zip(out["cash"], out["invested"], df["equity"]):
            self.assertAlmostEqual(cash + invested, eq, places=4)

    def test_missing_columns(self):
        df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5),
                           "equity": [1, 2, 3, 4, 5]})
        self.assertFalse(rd.compute_capital_utilization(df)["available"])


class TestStreaks(unittest.TestCase):
    def test_known_sequence(self):
        # W W L W L L  -> streaks: 2W, 1L, 1W, 2L
        seq = [10, 10, -5, 10, -5, -5]
        trades = [_trade(p, p, f"2021-01-{i+1:02d}") for i, p in enumerate(seq)]
        out = rd.compute_streaks(trades)
        self.assertEqual(out["sequence_lengths"], [2, -1, 1, -2])
        self.assertEqual(out["stats"]["max_win_streak"], 2)
        self.assertEqual(out["stats"]["max_loss_streak"], 2)
        self.assertEqual(out["stats"]["total_streaks"], 4)


if __name__ == "__main__":
    unittest.main()
