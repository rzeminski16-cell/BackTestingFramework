"""
Family normalisation -- stamp every family panel with the canonical contract.

Each family arrives in its own natural shape (OHLC for prices, single-value for
commodities/macro, events for corporate actions). ``normalise_family_panel``
converts the raw ``observation_date`` into a point-in-time ``available_ts`` using
the family's timing policy and stamps the shared provenance columns
(:data:`~Classes.DataPrep.schema.PROVENANCE_COLUMNS`) plus a row-level
``quality_flag``. The result is a panel the modelling stage can join as-of.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .schema import Family, PROVENANCE_COLUMNS, QualityFlag
from .timing import TimingPolicy, AvailabilityRule, compute_available_ts


def stamp_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder so the canonical provenance columns lead, keeping the rest after.

    Missing provenance columns are created as null so every family table has a
    uniform leading block (the contract the modelling stage relies on).
    """
    df = df.copy()
    for col in PROVENANCE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    rest = [c for c in df.columns if c not in PROVENANCE_COLUMNS]
    return df[PROVENANCE_COLUMNS + rest]


def normalise_family_panel(
    df: pd.DataFrame,
    *,
    family: Family,
    run_id: str,
    timing: TimingPolicy,
    entity_id_col: str = "entity_id",
    observation_date_col: str = "observation_date",
    value_col: Optional[str] = None,
    report_date_col: Optional[str] = None,
    native_frequency: Optional[str] = None,
    source_function: Optional[str] = None,
    source_vendor: str = "alpha_vantage",
    retrieved_at: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalise one family's raw panel into the canonical contract.

    Args:
        df: Raw/tidy family data; must contain the entity id and observation date
            columns (named via ``entity_id_col`` / ``observation_date_col``).
        family: Which :class:`Family` this panel is.
        run_id: The owning run identifier (stamped on every row).
        timing: The family's timing policy (drives ``available_ts``).
        value_col: If given, rows with a null value are flagged ``missing``.
        report_date_col: Explicit release-date column (used by REPORT_DATE rule).
        native_frequency / source_function / retrieved_at: Defaults applied when
            the corresponding column is absent.

    Returns:
        A DataFrame with the provenance block first, then the family's own
        columns. Empty input yields an empty (but correctly-typed) frame.
    """
    if df is None or len(df) == 0:
        return stamp_provenance(pd.DataFrame())

    out = df.copy()

    # Entity id.
    if entity_id_col != "entity_id":
        if entity_id_col not in out.columns:
            raise ValueError(f"entity id column '{entity_id_col}' not found")
        out = out.rename(columns={entity_id_col: "entity_id"})
    if "entity_id" not in out.columns:
        raise ValueError("normalise_family_panel requires an 'entity_id' column")

    # Observation date.
    if observation_date_col != "observation_date":
        if observation_date_col not in out.columns:
            raise ValueError(f"observation date column '{observation_date_col}' not found")
        out = out.rename(columns={observation_date_col: "observation_date"})
    if "observation_date" not in out.columns:
        raise ValueError("normalise_family_panel requires an 'observation_date' column")
    out["observation_date"] = pd.to_datetime(out["observation_date"], errors="coerce")

    # Availability timestamp from the timing policy.
    report_series = out[report_date_col] if (report_date_col and report_date_col in out.columns) else None
    out["available_ts"] = compute_available_ts(out["observation_date"], timing, report_series)

    # Identity / provenance stamps.
    out["run_id"] = run_id
    out["family"] = family.value
    if "native_frequency" not in out.columns:
        out["native_frequency"] = native_frequency
    if "source_function" not in out.columns:
        out["source_function"] = source_function
    if "source_vendor" not in out.columns:
        out["source_vendor"] = source_vendor
    if "retrieved_at" not in out.columns:
        out["retrieved_at"] = retrieved_at or datetime.now(timezone.utc).isoformat()

    out["quality_flag"] = _quality_flags(out, family, timing, value_col, report_series)

    return stamp_provenance(out)


def _quality_flags(
    df: pd.DataFrame,
    family: Family,
    timing: TimingPolicy,
    value_col: Optional[str],
    report_series: Optional[pd.Series],
) -> pd.Series:
    """Row-intrinsic quality flags (priority: missing > inferred_ts > revision_risk > ok).

    Staleness is NOT set here -- it is relative to a trade's anchor and is
    evaluated at join/validation time, not on the standalone panel.
    """
    n = len(df)
    flags = pd.Series([QualityFlag.OK.value] * n, index=df.index, dtype=object)

    # Revision risk (macro latest-history values, or any explicit flag column).
    if "revision_risk_flag" in df.columns:
        rr = df["revision_risk_flag"].fillna(False).astype(bool)
        flags = flags.mask(rr, QualityFlag.REVISION_RISK.value)
    elif family == Family.MACRO:
        flags = QualityFlag.REVISION_RISK.value

    # Inferred timestamp: REPORT_DATE rule but no explicit report date present.
    if timing.effective_rule() == AvailabilityRule.REPORT_DATE:
        if report_series is None:
            flags = pd.Series([QualityFlag.INFERRED_TS.value] * n, index=df.index, dtype=object)
        else:
            missing_rd = pd.to_datetime(report_series, errors="coerce").isna()
            flags = flags.mask(missing_rd, QualityFlag.INFERRED_TS.value)

    # Missing value takes top priority.
    if value_col and value_col in df.columns:
        missing_val = df[value_col].isna()
        flags = flags.mask(missing_val, QualityFlag.MISSING.value)

    return flags
