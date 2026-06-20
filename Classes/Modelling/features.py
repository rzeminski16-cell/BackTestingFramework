"""
Feature engineering — turn the package's raw normalised panels into a
model-ready matrix via leakage-safe as-of joins.

The data-prep package ships *raw* family panels (prices, index, FX, commodities,
macro, fundamentals, …), not engineered features. This module builds the trade ×
feature matrix (per-trade view) and the period × feature matrix (per-period view)
by joining each panel **backward** on ``available_ts`` relative to the trade entry
time / period end, bounded by the family's ``carry_forward_tolerance_days``.

Leakage safety is structural: a ``merge_asof(direction="backward")`` can only
attach a value whose ``available_ts`` is on/before the join key, so nothing the
model sees was unknowable at decision time. Outcome-derived trade columns
(``pl``, ``exit_*``, ``duration_days``, …) are explicitly blocklisted from the
feature set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .run_package import RunPackage

# Columns that are provenance/identity, not model features.
_NON_FEATURE_PANEL_COLS = {
    "run_id", "family", "entity_id", "observation_date", "available_ts",
    "native_frequency", "source_function", "source_vendor", "retrieved_at",
    "quality_flag", "report_date", "revision_risk_flag", "geo_scope",
    "symbol", "role", "market_type", "region", "series_id", "unit",
}

# Trade columns that are outcomes or known only at/after exit — never features.
_OUTCOME_TRADE_COLS = {
    "pl", "pl_pct", "security_pl", "fx_pl", "exit_date", "exit_price",
    "exit_fx_rate", "exit_reason", "duration_days", "partial_exits",
    "final_stop_loss", "is_winner",
}

# Families whose entity_id is the trade's own symbol (joined per-symbol).
_SYMBOL_SPECIFIC_FAMILIES = {"equity_prices", "corporate_actions", "fundamentals_pit"}

# Families whose primary value lives in a specific column.
_PRIMARY_VALUE_COL = {
    "equity_prices": "close",
    "index_panel": "close",
    "fx_panel": "rate",
}

# Guards against feature explosion. Market-wide families create one column set per
# entity, so we cap how many distinct entities a single family may expand into;
# beyond this we keep the most-populated series and record a warning.
MAX_MARKET_ENTITIES = 40
# Soft ceiling on total feature width (a tripwire that records a warning).
MAX_TOTAL_FEATURES = 600


@dataclass
class FeatureMatrix:
    """A built feature matrix plus the metadata needed downstream."""
    X: pd.DataFrame                       # index = trade_id (or period ts)
    index_name: str                       # "trade_id" or "period"
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    feature_family: Dict[str, str] = field(default_factory=dict)  # feature -> family
    keys: pd.DataFrame = field(default_factory=pd.DataFrame)      # id, time key, symbol
    warnings: List[str] = field(default_factory=list)            # feature-build warnings

    @property
    def feature_names(self) -> List[str]:
        return self.numeric_features + self.categorical_features


class FeatureBuilder:
    """Builds per-trade and per-period feature matrices from a run package."""

    def __init__(self, package: RunPackage):
        self.package = package

    # -- helpers ------------------------------------------------------------ #
    def _families_to_use(self, requested: Optional[List[str]]) -> List[str]:
        present = self.package.available_families()
        if not requested:
            return present
        return [f for f in requested if f in present]

    @staticmethod
    def _value_columns(panel: pd.DataFrame, family: str,
                       allow: Optional[List[str]] = None) -> List[str]:
        numeric = [
            c for c in panel.columns
            if c not in _NON_FEATURE_PANEL_COLS
            and pd.api.types.is_numeric_dtype(panel[c])
        ]
        if allow:
            numeric = [c for c in numeric if c in allow]
        elif family in _PRIMARY_VALUE_COL and _PRIMARY_VALUE_COL[family] in numeric:
            # Keep the primary value (e.g. close / rate); drop the rest of OHLCV
            # to avoid near-duplicate collinear columns by default.
            numeric = [_PRIMARY_VALUE_COL[family]]
        return numeric

    @staticmethod
    def _prep_panel_series(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """One entity's (available_ts, value) series with trailing-return derivations."""
        cols = ["available_ts", "observation_date", value_col]
        cols = [c for c in cols if c in panel.columns]
        s = panel[cols].dropna(subset=["available_ts"]).copy()
        order = "observation_date" if "observation_date" in s.columns else "available_ts"
        s = s.sort_values(order)
        # Trailing percentage changes (computed on the panel's own history → no leakage).
        s[f"{value_col}__ret1"] = s[value_col].pct_change(1)
        s[f"{value_col}__ret5"] = s[value_col].pct_change(5)
        return s.sort_values("available_ts")

    @staticmethod
    def _add_numeric(fm: FeatureMatrix, name: str, values, family: str) -> None:
        """Assign a numeric feature exactly once (never duplicate a column name)."""
        if name in fm.feature_family:
            return
        fm.X[name] = values
        fm.numeric_features.append(name)
        fm.feature_family[name] = family

    @staticmethod
    def _add_categorical(fm: FeatureMatrix, name: str, values, family: str) -> None:
        if name in fm.feature_family:
            return
        fm.X[name] = values
        fm.categorical_features.append(name)
        fm.feature_family[name] = family

    @staticmethod
    def _limited_entities(panel: pd.DataFrame, family: str,
                          warnings: List[str]) -> List[str]:
        """Distinct entity_ids for a market-wide family, capped to the most-populated."""
        if "entity_id" not in panel.columns:
            return [None]
        counts = panel["entity_id"].dropna().value_counts()
        entities = list(counts.index)
        if len(entities) > MAX_MARKET_ENTITIES:
            kept = entities[:MAX_MARKET_ENTITIES]
            warnings.append(
                f"Family '{family}' has {len(entities)} series; kept the "
                f"{MAX_MARKET_ENTITIES} most-populated to bound feature width.")
            return kept
        return entities

    # -- per-trade ---------------------------------------------------------- #
    def build_per_trade(self, feature_families: Optional[List[str]] = None,
                        feature_columns: Optional[Dict[str, List[str]]] = None
                        ) -> FeatureMatrix:
        feature_columns = feature_columns or {}
        trades = self.package.trades.copy()
        if trades.empty:
            raise ValueError("Run package has no trades to build features from.")
        trades = trades.dropna(subset=["entry_date"]).sort_values("entry_date")

        fm = FeatureMatrix(
            X=pd.DataFrame(index=trades["trade_id"]),
            index_name="trade_id",
        )
        fm.X.index.name = "trade_id"
        fm.keys = trades[[c for c in ("trade_id", "symbol", "entry_date") if c in trades.columns]].copy()

        # 1) Trade-intrinsic features known at entry.
        self._add_trade_intrinsics(trades, fm)

        # 2) As-of family features.
        for family in self._families_to_use(feature_families):
            panel = self.package.panels.get(family)
            if panel is None or panel.empty or "available_ts" not in panel.columns:
                continue
            tol = self.package.carry_forward_tolerance(family)
            allow = feature_columns.get(family)
            self._join_family_per_trade(trades, panel, family, tol, allow, fm)

        self._drop_dead_columns(fm)
        self._guard_width(fm)
        return fm

    @staticmethod
    def _drop_dead_columns(fm: FeatureMatrix) -> None:
        """Sanitise ±inf to NaN, then remove all-NaN numeric features.

        Trailing-return derivations can produce ±inf when a prior value is 0
        (common in fundamentals); sklearn imputers treat NaN as missing but
        reject inf, so we normalise here.
        """
        num = [c for c in fm.numeric_features if c in fm.X.columns]
        if num:
            fm.X[num] = fm.X[num].replace([np.inf, -np.inf], np.nan)
        dead = [c for c in fm.numeric_features
                if c in fm.X.columns and fm.X[c].isna().all()]
        if not dead:
            return
        fm.X = fm.X.drop(columns=dead)
        dead_set = set(dead)
        fm.numeric_features = [c for c in fm.numeric_features if c not in dead_set]
        for c in dead:
            fm.feature_family.pop(c, None)

    @staticmethod
    def _guard_width(fm: FeatureMatrix) -> None:
        if len(fm.feature_names) > MAX_TOTAL_FEATURES:
            fm.warnings.append(
                f"Feature matrix is wide ({len(fm.feature_names)} features); "
                "consider an explicit feature_columns allow-list per family.")

    def _add_trade_intrinsics(self, trades: pd.DataFrame, fm: FeatureMatrix) -> None:
        idx = trades["trade_id"].values
        # Numeric intrinsics known at entry.
        for col in ("concurrent_positions", "entry_capital_required",
                    "entry_capital_available", "entry_equity", "entry_price",
                    "quantity", "commission_paid"):
            if col in trades.columns and pd.api.types.is_numeric_dtype(trades[col]):
                self._add_numeric(fm, f"trade__{col}",
                                  pd.Series(trades[col].values, index=idx), "trade")
        # Capital pressure ratio (known at entry).
        if {"entry_capital_required", "entry_capital_available"}.issubset(trades.columns):
            denom = trades["entry_capital_available"].replace(0, np.nan).values
            self._add_numeric(fm, "trade__capital_pressure",
                              pd.Series(trades["entry_capital_required"].values / denom, index=idx),
                              "trade")
        # Calendar features of the entry timestamp.
        ed = pd.to_datetime(trades["entry_date"])
        for name, vals in (("trade__entry_month", ed.dt.month),
                           ("trade__entry_dayofweek", ed.dt.dayofweek),
                           ("trade__entry_quarter", ed.dt.quarter)):
            self._add_numeric(fm, name, pd.Series(vals.values, index=idx), "trade")
        # Categoricals.
        for col, fname in (("symbol", "trade__symbol"),
                           ("side", "trade__side"),
                           ("security_currency", "trade__currency"),
                           ("entry_reason", "trade__entry_reason")):
            if col in trades.columns:
                self._add_categorical(fm, fname,
                                      pd.Series(trades[col].astype(str).values, index=idx),
                                      "trade")

    def _join_family_per_trade(self, trades: pd.DataFrame, panel: pd.DataFrame,
                              family: str, tol_days: int,
                              allow: Optional[List[str]], fm: FeatureMatrix) -> None:
        value_cols = self._value_columns(panel, family, allow)
        if not value_cols:
            return
        tolerance = pd.Timedelta(days=tol_days) if tol_days and tol_days > 0 else None
        symbol_specific = (family in _SYMBOL_SPECIFIC_FAMILIES
                           and "entity_id" in panel.columns
                           and "symbol" in trades.columns)
        entry_by_id = pd.to_datetime(trades.set_index("trade_id")["entry_date"]).reindex(fm.X.index)

        if symbol_specific:
            # Single pass: attach the trade's OWN symbol's value via merge_asof(by).
            # Looping entities here previously produced one duplicate feature name
            # per symbol, exploding the matrix; one by-merge is correct AND bounded.
            left = trades[["trade_id", "entry_date"]].copy()
            left["symbol"] = trades["symbol"].astype(str).values
            order = "observation_date" if "observation_date" in panel.columns else "available_ts"
            g = panel.sort_values(["entity_id", order]).copy()
            derived_all: List[str] = []
            for vc in value_cols:
                g[f"{vc}__ret1"] = g.groupby("entity_id")[vc].pct_change(1)
                g[f"{vc}__ret5"] = g.groupby("entity_id")[vc].pct_change(5)
                derived_all += [vc, f"{vc}__ret1", f"{vc}__ret5"]
            right = g[["available_ts", "entity_id"] + derived_all].dropna(subset=["available_ts"])
            right = right.rename(columns={"entity_id": "symbol"})
            right["symbol"] = right["symbol"].astype(str)
            merged = pd.merge_asof(
                left.sort_values("entry_date"), right.sort_values("available_ts"),
                left_on="entry_date", right_on="available_ts", by="symbol",
                direction="backward", tolerance=tolerance,
            ).set_index("trade_id").reindex(fm.X.index)
            self._assign_merged(fm, merged, family, "", derived_all, entry_by_id)
            return

        # Market-wide family: one bounded column set per (capped) entity.
        entities = self._limited_entities(panel, family, fm.warnings)
        for ent in entities:
            sub = panel if ent is None else panel[panel["entity_id"] == ent]
            suffix = f"_{ent}" if ent is not None else ""
            derived_all: List[str] = []
            merged_parts = {}
            for value_col in value_cols:
                series = self._prep_panel_series(sub, value_col)
                derived = [c for c in (value_col, f"{value_col}__ret1", f"{value_col}__ret5")
                           if c in series.columns]
                right = series[["available_ts"] + derived].dropna(subset=["available_ts"])
                if right.empty:
                    continue
                merged = pd.merge_asof(
                    trades[["trade_id", "entry_date"]].sort_values("entry_date"),
                    right.sort_values("available_ts"),
                    left_on="entry_date", right_on="available_ts",
                    direction="backward", tolerance=tolerance,
                ).set_index("trade_id").reindex(fm.X.index)
                for col in derived:
                    merged_parts[col] = merged[col]
                    derived_all.append(col)
                merged_parts["available_ts"] = merged.get("available_ts")
            if derived_all:
                self._assign_merged(fm, pd.DataFrame(merged_parts), family, suffix,
                                    derived_all, entry_by_id)

    def _assign_merged(self, fm: FeatureMatrix, merged: pd.DataFrame, family: str,
                       suffix: str, derived: List[str], entry_by_id: pd.Series) -> None:
        """Assign as-of values + a freshness 'age_days' feature (deduped)."""
        if "available_ts" in merged.columns:
            age = (entry_by_id.values - pd.to_datetime(merged["available_ts"]))
            self._add_numeric(fm, f"{family}{suffix}__age_days",
                              age.dt.days.values, family)
        for col in derived:
            self._add_numeric(fm, f"{family}{suffix}__{col}", merged[col].values, family)

    # -- per-period --------------------------------------------------------- #
    def build_per_period(self, period_freq: str = "W",
                        feature_families: Optional[List[str]] = None,
                        feature_columns: Optional[Dict[str, List[str]]] = None
                        ) -> Tuple[FeatureMatrix, pd.DataFrame]:
        """Build a regular period grid with regime features + portfolio state.

        Returns ``(feature_matrix, period_frame)`` where ``period_frame`` carries
        the portfolio-process aggregates (open trades, realised P/L) the
        per-period targets are derived from.
        """
        feature_columns = feature_columns or {}
        trades = self.package.trades.copy()
        if trades.empty:
            raise ValueError("Run package has no trades to build a period panel from.")
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])

        start = trades["entry_date"].min().normalize()
        end = trades["exit_date"].max().normalize()
        grid = pd.date_range(start=start, end=end, freq=period_freq)
        if len(grid) < 3:
            grid = pd.date_range(start=start, end=end, freq="D")

        pl_col = "pl" if "pl" in trades.columns else None
        rows = []
        for i, ts in enumerate(grid):
            prev = grid[i - 1] if i > 0 else start - pd.Timedelta(days=1)
            open_mask = (trades["entry_date"] <= ts) & (trades["exit_date"] > ts)
            closed_mask = (trades["exit_date"] > prev) & (trades["exit_date"] <= ts)
            rows.append({
                "period": ts,
                "open_trades": int(open_mask.sum()),
                "closed_trades": int(closed_mask.sum()),
                "period_realised_pl": float(trades.loc[closed_mask, pl_col].sum()) if pl_col else 0.0,
            })
        period_frame = pd.DataFrame(rows).set_index("period")

        fm = FeatureMatrix(X=pd.DataFrame(index=period_frame.index), index_name="period")
        fm.X.index.name = "period"
        fm.keys = period_frame.reset_index()[["period"]].copy()

        # Portfolio-state features (known at period end).
        for col in ("open_trades", "closed_trades", "period_realised_pl"):
            fm.X[col] = period_frame[col].values
            fm.numeric_features.append(col)
            fm.feature_family[col] = "portfolio"

        # As-of market-wide family features at each period end.
        period_keys = period_frame.reset_index()[["period"]].copy()
        for family in self._families_to_use(feature_families):
            if family in _SYMBOL_SPECIFIC_FAMILIES:
                continue  # symbol-specific families are not regime-wide
            panel = self.package.panels.get(family)
            if panel is None or panel.empty or "available_ts" not in panel.columns:
                continue
            tol = self.package.carry_forward_tolerance(family)
            allow = feature_columns.get(family)
            self._join_family_per_period(period_keys, panel, family, tol, allow, fm)

        self._drop_dead_columns(fm)
        self._guard_width(fm)
        return fm, period_frame

    def _join_family_per_period(self, period_keys: pd.DataFrame, panel: pd.DataFrame,
                               family: str, tol_days: int, allow: Optional[List[str]],
                               fm: FeatureMatrix) -> None:
        value_cols = self._value_columns(panel, family, allow)
        if not value_cols:
            return
        tolerance = pd.Timedelta(days=tol_days) if tol_days and tol_days > 0 else None
        for ent in self._limited_entities(panel, family, fm.warnings):
            sub = panel if ent is None else panel[panel["entity_id"] == ent]
            suffix = f"_{ent}" if ent is not None else ""
            for value_col in value_cols:
                series = self._prep_panel_series(sub, value_col)
                derived = [c for c in (value_col, f"{value_col}__ret1", f"{value_col}__ret5")
                           if c in series.columns]
                right = series[["available_ts"] + derived].dropna(subset=["available_ts"])
                if right.empty:
                    continue
                merged = pd.merge_asof(
                    period_keys.sort_values("period"),
                    right.sort_values("available_ts"),
                    left_on="period", right_on="available_ts",
                    direction="backward", tolerance=tolerance,
                ).set_index("period").reindex(fm.X.index)
                for col in derived:
                    self._add_numeric(fm, f"{family}{suffix}__{col}",
                                      merged[col].values, family)
