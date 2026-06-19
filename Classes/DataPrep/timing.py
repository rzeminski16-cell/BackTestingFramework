"""
Timing & missing-data policies -- the heart of point-in-time correctness.

Every family converts a raw ``observation_date`` (the period a value describes)
into an ``available_ts`` (the earliest moment the value could have been known in
live trading) via an :class:`AvailabilityRule`. The modelling stage then joins
features to trades as-of ``available_ts``, never by calendar equality.

The module also defines :class:`MissingDataPolicy` (family-specific fill rules --
deliberately *not* one universal rule) and the per-family default policy table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd

from .schema import Family


class AvailabilityRule(str, Enum):
    """How an ``observation_date`` becomes an ``available_ts``."""
    SAME_DAY = "same_day"            # known same day (e.g. close, for close-executed trades)
    NEXT_SESSION = "next_session"    # known next trading session (conservative close handling)
    PUBLICATION_LAG = "publication_lag"  # observation_date + N calendar days (macro/commodity release)
    REPORT_DATE = "report_date"      # explicit report/release date column (fundamentals)
    REALTIME = "realtime"            # the observation timestamp is itself the availability


class MissingDataPolicy(str, Enum):
    """Family-specific missing-data / resampling behaviour.

    There is intentionally no single universal rule; mixing frequencies safely
    requires per-family choices (see the data-prep spec).
    """
    NONE = "none"                              # no fill beyond schedule (intraday)
    CARRY_SESSION = "carry_session"            # carry until next trading session
    HOLD_UNTIL_SUPERSEDED = "hold_until_superseded"  # fundamentals: hold from release
    STEP_FORWARD = "step_forward"              # macro: step-function carry from available_ts only

    @property
    def description(self) -> str:
        return _MISSING_DESCRIPTIONS[self]


_MISSING_DESCRIPTIONS: Dict[MissingDataPolicy, str] = {
    MissingDataPolicy.NONE:
        "No forward-fill beyond the native schedule. Gaps stay as gaps.",
    MissingDataPolicy.CARRY_SESSION:
        "Carry the last close-anchored value until the next trading session.",
    MissingDataPolicy.HOLD_UNTIL_SUPERSEDED:
        "Hold each released value constant until the next release supersedes it.",
    MissingDataPolicy.STEP_FORWARD:
        "Step-function carry-forward from availability only (no back-fill).",
}


@dataclass
class TimingPolicy:
    """Per-family timing & missing-data configuration.

    Attributes:
        availability_rule: How observation_date maps to available_ts.
        publication_lag_days: Calendar-day lag for PUBLICATION_LAG / REPORT_DATE
            fallback (used when an exact release date is unavailable).
        freshness_budget_days: A value older than this (relative to a trade's
            anchor) is flagged ``stale``; None disables the check.
        missing_data_policy: Fill behaviour for gaps.
        carry_forward_tolerance_days: Maximum age a carried value may reach
            before it is no longer joined (the as-of merge tolerance).
        allow_same_day_close: Whether close-based values may be used for same-day
            decisions. When False, SAME_DAY is hardened to NEXT_SESSION.
    """
    availability_rule: AvailabilityRule = AvailabilityRule.SAME_DAY
    publication_lag_days: int = 0
    freshness_budget_days: Optional[int] = None
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.CARRY_SESSION
    carry_forward_tolerance_days: int = 5
    allow_same_day_close: bool = True

    def effective_rule(self) -> AvailabilityRule:
        """Resolve SAME_DAY to NEXT_SESSION when same-day close is disallowed."""
        if self.availability_rule == AvailabilityRule.SAME_DAY and not self.allow_same_day_close:
            return AvailabilityRule.NEXT_SESSION
        return self.availability_rule

    def to_dict(self) -> Dict[str, object]:
        return {
            "availability_rule": self.availability_rule.value,
            "publication_lag_days": self.publication_lag_days,
            "freshness_budget_days": self.freshness_budget_days,
            "missing_data_policy": self.missing_data_policy.value,
            "carry_forward_tolerance_days": self.carry_forward_tolerance_days,
            "allow_same_day_close": self.allow_same_day_close,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TimingPolicy":
        data = dict(data or {})
        if "availability_rule" in data:
            data["availability_rule"] = AvailabilityRule(data["availability_rule"])
        if "missing_data_policy" in data:
            data["missing_data_policy"] = MissingDataPolicy(data["missing_data_policy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def compute_available_ts(
    observation_date: pd.Series,
    policy: TimingPolicy,
    report_date: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Vectorised conversion of ``observation_date`` to ``available_ts``.

    Args:
        observation_date: Series of period/observation timestamps.
        policy: The family's timing policy.
        report_date: Optional explicit release dates (used by REPORT_DATE).

    Returns:
        A datetime64 Series of earliest-usable timestamps, aligned to the input.
    """
    obs = pd.to_datetime(observation_date, errors="coerce")
    rule = policy.effective_rule()

    if rule == AvailabilityRule.SAME_DAY or rule == AvailabilityRule.REALTIME:
        return obs

    if rule == AvailabilityRule.NEXT_SESSION:
        return obs + pd.tseries.offsets.BDay(1)

    if rule == AvailabilityRule.PUBLICATION_LAG:
        return obs + pd.Timedelta(days=int(policy.publication_lag_days))

    if rule == AvailabilityRule.REPORT_DATE:
        fallback = obs + pd.Timedelta(days=int(policy.publication_lag_days))
        if report_date is None:
            return fallback
        rd = pd.to_datetime(report_date, errors="coerce")
        # Use the explicit report date where present, else the lagged fallback.
        return rd.where(rd.notna(), fallback)

    return obs  # pragma: no cover - exhaustive above


# Per-family default policies tuned for a daily-cadence run. The GUI surfaces and
# lets the user override every value; these are conservative, finance-aware
# starting points drawn from the data-prep spec's alignment table.
DEFAULT_FAMILY_TIMING: Dict[Family, TimingPolicy] = {
    Family.EQUITY_PRICES: TimingPolicy(
        availability_rule=AvailabilityRule.SAME_DAY,
        freshness_budget_days=5,
        missing_data_policy=MissingDataPolicy.CARRY_SESSION,
        carry_forward_tolerance_days=5,
        allow_same_day_close=True,
    ),
    Family.CORPORATE_ACTIONS: TimingPolicy(
        availability_rule=AvailabilityRule.SAME_DAY,
        freshness_budget_days=None,
        missing_data_policy=MissingDataPolicy.NONE,
        carry_forward_tolerance_days=0,
        allow_same_day_close=True,
    ),
    Family.FUNDAMENTALS: TimingPolicy(
        availability_rule=AvailabilityRule.REPORT_DATE,
        publication_lag_days=1,
        freshness_budget_days=120,
        missing_data_policy=MissingDataPolicy.HOLD_UNTIL_SUPERSEDED,
        carry_forward_tolerance_days=200,
        allow_same_day_close=False,
    ),
    Family.INDEX: TimingPolicy(
        availability_rule=AvailabilityRule.SAME_DAY,
        freshness_budget_days=5,
        missing_data_policy=MissingDataPolicy.CARRY_SESSION,
        carry_forward_tolerance_days=5,
        allow_same_day_close=True,
    ),
    Family.FX: TimingPolicy(
        availability_rule=AvailabilityRule.SAME_DAY,
        freshness_budget_days=7,
        missing_data_policy=MissingDataPolicy.CARRY_SESSION,
        carry_forward_tolerance_days=7,
        allow_same_day_close=True,
    ),
    Family.COMMODITIES: TimingPolicy(
        availability_rule=AvailabilityRule.PUBLICATION_LAG,
        publication_lag_days=0,
        freshness_budget_days=95,
        missing_data_policy=MissingDataPolicy.STEP_FORWARD,
        carry_forward_tolerance_days=95,
        allow_same_day_close=True,
    ),
    Family.MACRO: TimingPolicy(
        availability_rule=AvailabilityRule.PUBLICATION_LAG,
        publication_lag_days=30,
        freshness_budget_days=400,
        missing_data_policy=MissingDataPolicy.STEP_FORWARD,
        carry_forward_tolerance_days=400,
        allow_same_day_close=False,
    ),
    Family.UTILITIES: TimingPolicy(
        availability_rule=AvailabilityRule.REALTIME,
        freshness_budget_days=None,
        missing_data_policy=MissingDataPolicy.NONE,
        carry_forward_tolerance_days=0,
        allow_same_day_close=True,
    ),
}


def default_timing_for(family: Family) -> TimingPolicy:
    """Return an independent copy of the default timing policy for a family."""
    return TimingPolicy.from_dict(DEFAULT_FAMILY_TIMING[family].to_dict())
