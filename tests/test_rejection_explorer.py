"""
Tests for the rejection-counterfactual analysis used by the vulnerability
score visualizer's Exploration tab.

Covers the three pure functions on which the GUI tab depends:

* :func:`load_rejection_log` (in vulnerability_trace.py) -- both CSV formats
* :func:`open_positions_at` (in vulnerability_trace.py)
* :func:`simulate_forced_entry` (in rejection_explorer.py) -- stop, TP, exit
  signal, end-of-data
* :func:`evaluate_rejection_at_config` (in rejection_explorer.py) -- a swap
  decision that flips from accepted to rejected when ``min_trade_age_days``
  is raised
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from Classes.Analysis.rejection_explorer import (
    OUTCOME_ACCEPTED,
    OUTCOME_NO_SWAP_CANDIDATES,
    OUTCOME_STILL_REJECTED,
    evaluate_rejection_at_config,
    simulate_forced_entry,
)
from Classes.Analysis.vulnerability_trace import (
    RejectionRecord,
    TradeRecord,
    load_rejection_log,
    open_positions_at,
)
from Classes.Config.capital_contention import VulnerabilityScoreConfig
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext


# --------------------------------------------------------------- test helpers

class _StubStrategy(BaseStrategy):
    """A bare strategy whose stop / TP / exit hooks can be configured per-test.

    Skips the base-class validation since we are running it free of a
    configured strategy registry.
    """

    _validate_on_init = False

    def __init__(self, stop_price: Optional[float] = None,
                 take_profit_price: Optional[float] = None,
                 exit_at_index: Optional[int] = None,
                 trailing_stop: Optional[float] = None):
        self._stop = stop_price
        self._tp = take_profit_price
        self._exit_idx = exit_at_index
        self._trail = trailing_stop
        super().__init__()

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        return None  # entry is forced

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        # Default well below entry price so it doesn't trip on creation.
        return self._stop if self._stop is not None else context.current_price * 0.5

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return 1.0

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if self._exit_idx is not None and context.current_index == self._exit_idx:
            return Signal.sell(reason='Stub exit')
        return None

    def should_adjust_stop(self, context: StrategyContext) -> Optional[float]:
        return self._trail


def _price_data(values: List[float], start: str = '2024-01-01') -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq='D')
    return pd.DataFrame({
        'date': dates,
        'open': values,
        'high': [v + 0.5 for v in values],
        'low': [v - 0.5 for v in values],
        'close': values,
        'volume': [1000] * len(values),
    })


# --------------------------------------------------- load_rejection_log tests

def test_load_rejection_log_tracking_format(tmp_path: Path):
    path = tmp_path / 'rejection_position_tracking.csv'
    rows = [
        ['rejection_date', 'rejected_symbol', 'rejected_signal_date',
         'rejected_close_price', 'rejection_reason',
         'position_symbol', 'position_open_date',
         'position_duration_days_at_rejection',
         'position_pl_pct_at_rejection',
         'position_score_at_rejection',
         'position_final_duration_days',
         'position_final_pl_pct',
         'position_open_at_end'],
        ['2024-01-05', 'GOOG', '2024-01-05', '102.5',
         'Insufficient capital',
         'AAPL', '2024-01-03', '2', '1.5', '', '8', '4.2', 'False'],
    ]
    with path.open('w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    records = load_rejection_log(path)
    assert len(records) == 1
    r = records[0]
    assert r.rejected_symbol == 'GOOG'
    assert r.rejection_date.date().isoformat() == '2024-01-05'
    assert r.rejected_close_price == 102.5
    assert r.position_symbol == 'AAPL'
    assert r.position_duration_days_at_rejection == 2
    assert r.position_final_pl_pct == 4.2
    assert r.position_open_at_end is False
    # Score column was blank — should round-trip as None, not NaN.
    assert r.position_score_at_rejection is None


def test_load_rejection_log_legacy_format(tmp_path: Path):
    path = tmp_path / 'signal_rejections.csv'
    rows = [
        ['date', 'symbol', 'signal_type', 'reason',
         'available_capital', 'required_capital'],
        ['2024-01-05', 'GOOG', 'BUY', 'No capital', '100.0', '5000.0'],
    ]
    with path.open('w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    records = load_rejection_log(path)
    assert len(records) == 1
    r = records[0]
    assert r.rejected_symbol == 'GOOG'
    assert r.rejection_reason == 'No capital'
    assert r.available_capital == 100.0
    assert r.required_capital == 5000.0
    # Tracking-only fields stay None.
    assert r.position_symbol is None


def test_load_rejection_log_unknown_format(tmp_path: Path):
    path = tmp_path / 'wrong.csv'
    path.write_text('foo,bar\n1,2\n')
    with pytest.raises(ValueError):
        load_rejection_log(path)


# -------------------------------------------------------- open_positions_at

def _trade(tid: str, symbol: str, entry: str, exit_: str) -> TradeRecord:
    return TradeRecord(
        trade_id=tid, symbol=symbol,
        entry_date=datetime.fromisoformat(entry),
        exit_date=datetime.fromisoformat(exit_),
        entry_price=100.0, exit_price=110.0,
        quantity=1.0, side='LONG',
        pl=10.0, pl_pct=10.0, duration_days=0,
    )


def test_open_positions_at_covers_overlap_boundaries():
    trades = [
        _trade('t_before', 'A', '2024-01-01', '2024-01-04'),  # closes BEFORE
        _trade('t_during', 'B', '2024-01-01', '2024-01-08'),  # overlaps
        _trade('t_after',  'C', '2024-01-06', '2024-01-10'),  # opens AFTER
        _trade('t_close_same', 'D', '2024-01-01', '2024-01-05'),  # closes on date
        _trade('t_open_same',  'E', '2024-01-05', '2024-01-09'),  # opens on date
    ]
    out = open_positions_at(datetime(2024, 1, 5), trades)
    ids = {t.trade_id for t in out}
    # entry_date <= date < exit_date semantics:
    # - t_before: exit=01-04 < 01-05 -> not open  -> excluded
    # - t_during: 01-01 <= 01-05 < 01-08 -> open  -> included
    # - t_after:  entry=01-06 > 01-05 -> not yet  -> excluded
    # - t_close_same: exit=01-05 NOT > 01-05 -> already closed -> excluded
    # - t_open_same:  entry=01-05 <= 01-05 < 01-09 -> open -> included
    assert ids == {'t_during', 't_open_same'}


# ------------------------------------------------ simulate_forced_entry tests

def test_simulate_forced_entry_stop_loss_hit():
    # Prices fall after entry; stop is 99, entry is 100.
    data = _price_data([100, 99.5, 98.5, 97.0, 96.0])
    strat = _StubStrategy(stop_price=99.0)
    result = simulate_forced_entry(
        'X', data, strat, datetime(2024, 1, 1), 100.0,
    )
    assert result is not None
    assert result.exit_reason == 'Stop loss hit'
    # Bar 1 (close=99.5) doesn't hit (99.5 > 99). Bar 2 (98.5) does.
    assert result.exit_price == 98.5
    assert result.pl_pct == pytest.approx(-1.5)


def test_simulate_forced_entry_take_profit_hit():
    data = _price_data([100, 101, 105, 108])

    class _TPStrat(_StubStrategy):
        def calculate_initial_stop_loss(self, context):
            return 90.0  # safely below entry

    strat = _TPStrat()
    # Inject a take_profit manually after entry: stub strategy doesn't open
    # the position itself, but we can ask the engine to honor TP by setting
    # take_profit_price on the synthetic position via should_adjust_stop. The
    # cleanest path is to simulate via a strategy whose generate_exit_signal
    # fires when the price crosses a known level.

    class _TPStrat2(_StubStrategy):
        def calculate_initial_stop_loss(self, context):
            return 90.0

        def generate_exit_signal(self, context):
            if context.current_price >= 105:
                return Signal.sell(reason='TP via exit signal')
            return None

    result = simulate_forced_entry(
        'X', data, _TPStrat2(), datetime(2024, 1, 1), 100.0,
    )
    assert result is not None
    assert result.exit_reason == 'TP via exit signal'
    assert result.exit_price == 105.0
    assert result.pl_pct == pytest.approx(5.0)


def test_simulate_forced_entry_exit_signal_index():
    data = _price_data([100, 101, 102, 103, 104])
    # Exit on the 3rd bar (index 3, i.e. day 4).
    strat = _StubStrategy(exit_at_index=3)
    result = simulate_forced_entry(
        'X', data, strat, datetime(2024, 1, 1), 100.0,
    )
    assert result is not None
    assert result.exit_reason == 'Stub exit'
    assert result.exit_price == 103.0
    assert result.pl_pct == pytest.approx(3.0)


def test_simulate_forced_entry_end_of_data():
    data = _price_data([100, 101, 102, 103])
    # Never fires anything.
    strat = _StubStrategy()
    result = simulate_forced_entry(
        'X', data, strat, datetime(2024, 1, 1), 100.0,
    )
    assert result is not None
    assert 'End of data' in result.exit_reason
    assert result.exit_price == 103.0
    assert result.pl_pct == pytest.approx(3.0)


def test_simulate_forced_entry_returns_none_when_no_future_bars():
    data = _price_data([100, 101], start='2024-01-01')
    strat = _StubStrategy()
    # Entry date is the LAST bar — nothing to walk forward.
    result = simulate_forced_entry(
        'X', data, strat, datetime(2024, 1, 2), 101.0,
    )
    assert result is None


# ------------------------------------------ evaluate_rejection_at_config test

def test_evaluate_rejection_flips_with_min_trade_age():
    """A position that's vulnerable at min_trade_age_days=1 becomes immune at
    min_trade_age_days=365, flipping the rejection from accepted back to
    rejected."""
    # Build a price series where the underwater position drops 30% from
    # entry, so its vulnerability score is high once it's eligible.
    bad_dates = pd.date_range('2024-01-01', periods=15, freq='D')
    bad_prices = [100.0] + [100.0 - i * 2.0 for i in range(1, 15)]
    bad_data = pd.DataFrame({
        'date': bad_dates,
        'open': bad_prices, 'high': [p + 0.5 for p in bad_prices],
        'low': [p - 0.5 for p in bad_prices],
        'close': bad_prices, 'volume': [1000] * 15,
    })
    # A flat series for the rejected symbol
    good_dates = pd.date_range('2024-01-01', periods=15, freq='D')
    good_prices = [50.0] * 15
    good_data = pd.DataFrame({
        'date': good_dates,
        'open': good_prices, 'high': [p + 0.5 for p in good_prices],
        'low': [p - 0.5 for p in good_prices],
        'close': good_prices, 'volume': [1000] * 15,
    })

    trades = [
        # Open position in BAD from day 1; closes well after the rejection.
        TradeRecord(
            trade_id='bad1', symbol='BAD',
            entry_date=datetime(2024, 1, 1), exit_date=datetime(2024, 1, 14),
            entry_price=100.0, exit_price=72.0, quantity=1.0, side='LONG',
            pl=-28.0, pl_pct=-28.0, duration_days=13,
        ),
    ]
    rejection = RejectionRecord(
        rejection_date=datetime(2024, 1, 10),
        rejected_symbol='NEW',
        rejection_reason='Insufficient capital',
        rejected_close_price=50.0,
    )
    data_dict = {'BAD': bad_data, 'NEW': good_data}

    # Young threshold = 1 day: BAD is 9 days old and well below target.
    cfg_low = VulnerabilityScoreConfig(min_trade_age_days=1,
                                       target_monthly_growth=0.05,
                                       alpha=1.0, beta=1.0,
                                       avg_window_days=3,
                                       pullback_window_days=3)
    strat = _StubStrategy()  # any sim outcome is fine; we only assert outcome bucket
    out_low = evaluate_rejection_at_config(rejection, trades, data_dict,
                                           cfg_low, strat)
    assert out_low.outcome == OUTCOME_ACCEPTED, out_low.swap_decision_reason

    # Young threshold = 365: every position is immune for years.
    cfg_high = VulnerabilityScoreConfig(min_trade_age_days=365,
                                        target_monthly_growth=0.05,
                                        alpha=1.0, beta=1.0,
                                        avg_window_days=3,
                                        pullback_window_days=3)
    out_high = evaluate_rejection_at_config(rejection, trades, data_dict,
                                            cfg_high, strat)
    assert out_high.outcome == OUTCOME_STILL_REJECTED
    assert 'immune' in out_high.swap_decision_reason.lower()


def test_evaluate_rejection_no_open_positions():
    """A rejection that happens with no open positions ought to be reported
    as no-swap-candidates, not crash."""
    data_dict = {'NEW': _price_data([50.0] * 10)}
    rejection = RejectionRecord(
        rejection_date=datetime(2024, 1, 5),
        rejected_symbol='NEW',
        rejection_reason='No capital',
    )
    cfg = VulnerabilityScoreConfig()
    out = evaluate_rejection_at_config(rejection, trades=[], data_dict=data_dict,
                                       config=cfg, strategy=None)
    assert out.outcome == OUTCOME_NO_SWAP_CANDIDATES
