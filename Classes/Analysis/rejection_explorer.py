"""
Counterfactual rejection analysis for the vulnerability-score visualizer.

Given a portfolio backtest's trade log + rejection log + a chosen strategy
preset, this module answers the question:

    "At a given vulnerability-score config, how many previously-rejected
     signals would now be accepted, and what is the aggregate hypothetical
     P/L of those simulated trades?"

The two main entry points are :func:`evaluate_rejection_at_config` (single
config) and :func:`aggregate_exploration` (run every rejection through one
config and roll up the per-rejection outcomes into a single summary). The
side-by-side A/B presentation in the GUI calls :func:`aggregate_exploration`
twice and shows the delta.

No portfolio state is tracked. Re-entered positions are simulated with a
*forced-entry* mini-engine that bypasses the strategy's entry signal (we
already know when the entry would have happened) and walks the strategy's
stop / trailing / exit rules forward bar-by-bar until something fires.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..Config.capital_contention import VulnerabilityScoreConfig
from ..Data.historical_data_view import HistoricalDataView
from ..Engine.vulnerability_score import VulnerabilityScoreCalculator
from ..Models.position import Position
from ..Models.signal import SignalType
from ..Models.trade_direction import TradeDirection
from ..Strategy.base_strategy import BaseStrategy
from ..Strategy.strategy_context import StrategyContext
from .vulnerability_trace import (
    RejectionRecord,
    TradeRecord,
    build_position,
    open_positions_at,
)


# --------------------------------------------------------------------- results

@dataclass
class ForcedTrade:
    """Outcome of a forced-entry simulation."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    exit_reason: str
    pl_pct: float
    duration_days: int


# Each rejection outcome falls into exactly one of these buckets.
OUTCOME_STILL_REJECTED = 'still_rejected'
OUTCOME_NO_SWAP_CANDIDATES = 'no_swap_candidates'
OUTCOME_SIM_FAILED = 'sim_failed'
OUTCOME_ACCEPTED = 'accepted'
OUTCOME_SKIPPED_NO_DATA = 'skipped_no_data'


@dataclass
class RejectionOutcome:
    """Per-rejection counterfactual outcome at one config."""
    rejection: RejectionRecord
    outcome: str  # one of OUTCOME_* constants
    swap_decision_reason: str = ''
    weakest_symbol: Optional[str] = None  # position chosen for the hypothetical swap
    forced_trade: Optional[ForcedTrade] = None


@dataclass
class ExplorationResult:
    """Aggregate counterfactual stats at one vulnerability config."""
    config: VulnerabilityScoreConfig
    n_rejections: int = 0
    n_accepted: int = 0
    n_sim_failed: int = 0
    n_skipped_no_data: int = 0
    n_no_swap_candidates: int = 0
    n_still_rejected: int = 0
    # Stats over forced trades that actually simulated to an exit
    n_simulated: int = 0
    pl_pcts: List[float] = field(default_factory=list)

    @property
    def aggregate_pl_pct(self) -> float:
        return float(sum(self.pl_pcts))

    @property
    def mean_pl_pct(self) -> float:
        return float(statistics.fmean(self.pl_pcts)) if self.pl_pcts else 0.0

    @property
    def median_pl_pct(self) -> float:
        return float(statistics.median(self.pl_pcts)) if self.pl_pcts else 0.0

    @property
    def win_rate(self) -> float:
        if not self.pl_pcts:
            return 0.0
        wins = sum(1 for x in self.pl_pcts if x > 0)
        return wins / len(self.pl_pcts)


# ----------------------------------------------------------------- swap eval

def _price_on(data: pd.DataFrame, target_date: datetime) -> Optional[float]:
    """Return the close price for ``target_date`` from a price DataFrame.

    Falls back to the latest close on or before the target so rejection
    dates that fall on missing bars (e.g., holidays) still produce a price.
    """
    if data is None or data.empty:
        return None
    dates = pd.to_datetime(data['date'])
    target = pd.Timestamp(target_date)
    mask = dates <= target
    if not mask.any():
        return None
    return float(data.loc[mask, 'close'].iloc[-1])


def evaluate_rejection_at_config(
    rejection: RejectionRecord,
    trades: List[TradeRecord],
    data_dict: Dict[str, pd.DataFrame],
    config: VulnerabilityScoreConfig,
    strategy: Optional[BaseStrategy],
) -> RejectionOutcome:
    """
    Counterfactual evaluation of a single rejection at one vuln config.

    The rejected symbol's price data must be in ``data_dict`` (otherwise the
    outcome is ``OUTCOME_SKIPPED_NO_DATA``). All currently-open positions are
    inferred from ``trades`` using :func:`open_positions_at`. The vulnerability
    calculator's :meth:`should_swap` decides whether the rejected signal
    would now displace a weak position.

    If the swap is positive, ``strategy`` is required and the rejected entry
    is simulated bar-by-bar via :func:`simulate_forced_entry`. If the
    simulator cannot construct a valid entry (no row at ``rejection_date``,
    invalid stop, etc.), the outcome is ``OUTCOME_SIM_FAILED``.
    """
    rejected_symbol = rejection.rejected_symbol
    if rejected_symbol not in data_dict:
        return RejectionOutcome(rejection=rejection,
                                outcome=OUTCOME_SKIPPED_NO_DATA)

    open_trades = open_positions_at(rejection.rejection_date, trades)
    # Only those positions whose price data we can resolve.
    open_trades = [t for t in open_trades if t.symbol in data_dict]

    if not open_trades:
        return RejectionOutcome(
            rejection=rejection,
            outcome=OUTCOME_NO_SWAP_CANDIDATES,
            swap_decision_reason="No open positions at rejection date",
        )

    positions: Dict[str, Position] = {t.symbol: build_position(t) for t in open_trades}
    current_prices: Dict[str, float] = {}
    for t in open_trades:
        price = _price_on(data_dict[t.symbol], rejection.rejection_date)
        if price is not None:
            current_prices[t.symbol] = price
    # Drop any positions for which we couldn't find a price.
    positions = {s: p for s, p in positions.items() if s in current_prices}
    if not positions:
        return RejectionOutcome(
            rejection=rejection,
            outcome=OUTCOME_NO_SWAP_CANDIDATES,
            swap_decision_reason="No open positions had prices on rejection date",
        )

    calc = VulnerabilityScoreCalculator(config, data_dict)
    decision = calc.should_swap(positions, current_prices,
                                rejection.rejection_date, rejected_symbol)

    if not decision.should_swap:
        return RejectionOutcome(
            rejection=rejection,
            outcome=OUTCOME_STILL_REJECTED,
            swap_decision_reason=decision.reason,
        )

    # Rejection becomes accepted at this config. Simulate the new trade.
    if strategy is None:
        return RejectionOutcome(
            rejection=rejection,
            outcome=OUTCOME_SIM_FAILED,
            swap_decision_reason=decision.reason,
            weakest_symbol=decision.position_to_close,
        )

    entry_price = rejection.rejected_close_price
    if entry_price is None:
        entry_price = _price_on(data_dict[rejected_symbol],
                                rejection.rejection_date)
    if entry_price is None or entry_price <= 0:
        return RejectionOutcome(
            rejection=rejection,
            outcome=OUTCOME_SIM_FAILED,
            swap_decision_reason=decision.reason,
            weakest_symbol=decision.position_to_close,
        )

    forced = simulate_forced_entry(
        symbol=rejected_symbol,
        data=data_dict[rejected_symbol],
        strategy=strategy,
        entry_date=rejection.rejection_date,
        entry_price=float(entry_price),
    )
    if forced is None:
        return RejectionOutcome(
            rejection=rejection,
            outcome=OUTCOME_SIM_FAILED,
            swap_decision_reason=decision.reason,
            weakest_symbol=decision.position_to_close,
        )

    return RejectionOutcome(
        rejection=rejection,
        outcome=OUTCOME_ACCEPTED,
        swap_decision_reason=decision.reason,
        weakest_symbol=decision.position_to_close,
        forced_trade=forced,
    )


def aggregate_exploration(
    rejections: List[RejectionRecord],
    trades: List[TradeRecord],
    data_dict: Dict[str, pd.DataFrame],
    config: VulnerabilityScoreConfig,
    strategy: Optional[BaseStrategy],
) -> Tuple[ExplorationResult, List[RejectionOutcome]]:
    """Run :func:`evaluate_rejection_at_config` over every rejection and
    return the aggregate plus the per-rejection outcomes.
    """
    result = ExplorationResult(config=config, n_rejections=len(rejections))
    outcomes: List[RejectionOutcome] = []
    for rej in rejections:
        outc = evaluate_rejection_at_config(
            rej, trades, data_dict, config, strategy
        )
        outcomes.append(outc)
        if outc.outcome == OUTCOME_ACCEPTED:
            result.n_accepted += 1
            if outc.forced_trade is not None:
                result.n_simulated += 1
                result.pl_pcts.append(float(outc.forced_trade.pl_pct))
        elif outc.outcome == OUTCOME_SIM_FAILED:
            result.n_sim_failed += 1
        elif outc.outcome == OUTCOME_SKIPPED_NO_DATA:
            result.n_skipped_no_data += 1
        elif outc.outcome == OUTCOME_NO_SWAP_CANDIDATES:
            result.n_no_swap_candidates += 1
        else:  # still rejected
            result.n_still_rejected += 1
    return result, outcomes


# ---------------------------------------------------------- forced simulator

def simulate_forced_entry(
    symbol: str,
    data: pd.DataFrame,
    strategy: BaseStrategy,
    entry_date: datetime,
    entry_price: float,
) -> Optional[ForcedTrade]:
    """Simulate a hypothetical trade entered at ``entry_date``/``entry_price``
    and walk it forward using only the strategy's stop / take-profit /
    trailing-stop / exit-signal rules.

    Mirrors the control flow of :meth:`SingleSecurityEngine.run` (entry
    bypassed since the rejection record already names the entry point).
    Returns ``None`` if a sensible entry cannot be constructed (no bar at
    ``entry_date``, invalid initial stop, no future bars).
    """
    if data is None or data.empty:
        return None
    if 'date' not in data.columns or 'close' not in data.columns:
        return None

    try:
        prepared = strategy.prepare_data(data)
    except Exception:  # noqa: BLE001
        # A missing indicator or other prep failure means the strategy can't
        # be applied here. Caller bucketises this as sim_failed.
        return None

    prepared = prepared.reset_index(drop=True)
    dates = pd.to_datetime(prepared['date'])
    target = pd.Timestamp(entry_date)
    on_or_after = dates >= target
    if not on_or_after.any():
        return None
    entry_idx = int(on_or_after.idxmax())

    direction = getattr(strategy, 'trade_direction', TradeDirection.LONG)

    # Need at least one bar AFTER the entry to walk forward.
    if entry_idx >= len(prepared) - 1:
        # No future bars to simulate against — fail out so the caller can
        # report it under sim_failed rather than silently zeroing P/L.
        return None

    # Build initial-stop context using data up to the entry bar.
    entry_view = HistoricalDataView(prepared, valid_end_index=entry_idx)
    entry_bar_date = prepared.iloc[entry_idx]['date']
    entry_context = StrategyContext(
        data=entry_view,
        current_index=entry_idx,
        current_price=entry_price,
        current_date=entry_bar_date,
        position=None,
        available_capital=0.0,
        total_equity=0.0,
        symbol=symbol,
        fx_rate=1.0,
    )

    try:
        initial_stop = float(strategy.calculate_initial_stop_loss(entry_context))
    except Exception:  # noqa: BLE001
        return None

    # Construct a synthetic position with quantity=1 so P/L is purely
    # price-driven (we report aggregate P/L %, not dollars).
    try:
        position = Position(
            symbol=symbol,
            entry_date=pd.Timestamp(entry_bar_date).to_pydatetime(),
            entry_price=float(entry_price),
            initial_quantity=1.0,
            current_quantity=1.0,
            direction=direction,
            stop_loss=initial_stop,
        )
    except ValueError:
        # Bad stop side / non-positive stop — strategy parameters incompatible
        # with this entry point. Counted as sim_failed.
        return None

    # Walk forward bar-by-bar, in the same priority order the single-security
    # engine uses: stop -> take profit -> trailing -> exit signal.
    for i in range(entry_idx + 1, len(prepared)):
        bar = prepared.iloc[i]
        current_price = float(bar['close'])
        view = HistoricalDataView(prepared, valid_end_index=i)
        context = StrategyContext(
            data=view,
            current_index=i,
            current_price=current_price,
            current_date=bar['date'],
            position=position,
            available_capital=0.0,
            total_equity=0.0,
            symbol=symbol,
            fx_rate=1.0,
        )

        # Stop loss
        if strategy.should_check_stop_loss(context) and position.stop_loss is not None:
            if direction == TradeDirection.LONG:
                hit = current_price <= position.stop_loss
            else:
                hit = current_price >= position.stop_loss
            if hit:
                return _build_forced_trade(symbol, position, bar, 'Stop loss hit')

        # Take profit
        if strategy.should_check_take_profit(context) and position.take_profit is not None:
            if direction == TradeDirection.LONG:
                tp_hit = current_price >= position.take_profit
            else:
                tp_hit = current_price <= position.take_profit
            if tp_hit:
                return _build_forced_trade(symbol, position, bar, 'Take profit hit')

        # Trailing stop
        try:
            new_stop = strategy.should_adjust_stop(context)
        except Exception:  # noqa: BLE001
            new_stop = None
        if new_stop is not None:
            if direction == TradeDirection.LONG:
                if position.stop_loss is None or new_stop > position.stop_loss:
                    position.stop_loss = float(new_stop)
            else:
                if position.stop_loss is None or new_stop < position.stop_loss:
                    position.stop_loss = float(new_stop)

        # Exit signal
        try:
            signal = strategy.generate_exit_signal(context)
        except Exception:  # noqa: BLE001
            signal = None
        if signal is not None and signal.type == SignalType.SELL:
            return _build_forced_trade(symbol, position, bar,
                                       signal.reason or 'Strategy exit')

    # Nothing fired — close at the final bar.
    last_bar = prepared.iloc[-1]
    return _build_forced_trade(symbol, position, last_bar,
                               'End of data (still open)')


def _build_forced_trade(symbol: str, position: Position,
                        exit_bar: pd.Series, reason: str) -> ForcedTrade:
    exit_price = float(exit_bar['close'])
    exit_date = pd.Timestamp(exit_bar['date']).to_pydatetime()
    if position.direction == TradeDirection.SHORT:
        pl_pct = (position.entry_price - exit_price) / position.entry_price * 100.0
    else:
        pl_pct = (exit_price - position.entry_price) / position.entry_price * 100.0
    return ForcedTrade(
        symbol=symbol,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=reason,
        pl_pct=float(pl_pct),
        duration_days=(exit_date - position.entry_date).days,
    )
