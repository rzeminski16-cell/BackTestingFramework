"""
Target construction — economic-first, not aesthetic-first.

Per-trade targets default to a **cost-aware** success threshold (a trade is
"good" only if it beats estimated costs plus a buffer), a clipped continuous
net-return utility, and a first-class tail-loss flag. Per-period targets describe
the strategy as a portfolio process: next-period return, next-period Adjusted
RAR%, and a favourable / neutral / hostile regime label.

Every target carries its **label interval** ``[label_start, label_end]`` so the
purged/embargoed splitter can drop training rows whose outcome window overlaps a
test window — essential because trades may all be open simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .adjusted_rar import adjusted_rar_from_equity
from .config import AdjustedRARConfig, TargetKind, TargetSpec, WeightMode


@dataclass
class TargetSet:
    """A constructed target plus the metadata modelling needs."""
    primary: TargetKind
    y: pd.Series                                  # the primary target, index-aligned
    targets: Dict[str, pd.Series] = field(default_factory=dict)
    label_start: pd.Series = None                 # label-window open (purge key)
    label_end: pd.Series = None                   # label-window close (purge key)
    sample_weight: pd.Series = None
    is_classification: bool = True
    class_balance: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _resolve_pl_pct(trades: pd.DataFrame) -> pd.Series:
    """Best-effort percentage P/L per trade."""
    if "pl_pct" in trades.columns:
        return pd.to_numeric(trades["pl_pct"], errors="coerce")
    if {"pl", "entry_price", "quantity"}.issubset(trades.columns):
        notional = (trades["entry_price"] * trades["quantity"]).replace(0, np.nan)
        return pd.to_numeric(trades["pl"], errors="coerce") / notional * 100.0
    if "pl" in trades.columns:
        # Fall back to sign-only information scaled to percent-ish units.
        return pd.to_numeric(trades["pl"], errors="coerce")
    raise ValueError("Cannot derive trade returns: need 'pl_pct' or 'pl'.")


class TargetBuilder:
    """Builds per-trade and per-period targets from package data."""

    def __init__(self, spec: TargetSpec, weight_mode: WeightMode = WeightMode.EQUAL,
                 adjusted_rar: Optional[AdjustedRARConfig] = None,
                 initial_capital: float = 100_000.0):
        self.spec = spec
        self.weight_mode = weight_mode
        self.adjusted_rar = adjusted_rar or AdjustedRARConfig()
        self.initial_capital = initial_capital

    # -- per-trade ---------------------------------------------------------- #
    def build_per_trade(self, trades: pd.DataFrame) -> TargetSet:
        trades = trades.dropna(subset=["entry_date"]).sort_values("entry_date")
        idx = trades["trade_id"]
        pl_pct = _resolve_pl_pct(trades).values

        targets: Dict[str, pd.Series] = {}
        good = (pl_pct > self.spec.cost_buffer_pct).astype(int)
        targets[TargetKind.BINARY_GOOD_TRADE.value] = pd.Series(good, index=idx)
        clipped = np.clip(pl_pct, -self.spec.return_clip_pct, self.spec.return_clip_pct)
        targets[TargetKind.CONTINUOUS_NET_RETURN.value] = pd.Series(clipped, index=idx)
        tail = (pl_pct <= self.spec.tail_loss_pct).astype(int)
        targets[TargetKind.BINARY_TAIL_LOSS.value] = pd.Series(tail, index=idx)

        primary = self.spec.primary
        if primary.value not in targets:
            primary = TargetKind.BINARY_GOOD_TRADE
        y = targets[primary.value]

        label_start = pd.Series(pd.to_datetime(trades["entry_date"]).values, index=idx)
        label_end = pd.Series(
            pd.to_datetime(trades["exit_date"]).values if "exit_date" in trades.columns
            else pd.to_datetime(trades["entry_date"]).values,
            index=idx,
        )

        ts = TargetSet(
            primary=primary, y=y, targets=targets,
            label_start=label_start, label_end=label_end,
            is_classification=primary.is_classification,
        )
        ts.sample_weight = self._sample_weight(y, pl_pct, idx, ts.is_classification)
        ts.class_balance = self._class_balance(targets, primary)
        return ts

    # -- per-period --------------------------------------------------------- #
    def build_per_period(self, period_frame: pd.DataFrame) -> TargetSet:
        """``period_frame`` indexed by period ts with ``period_realised_pl``."""
        pf = period_frame.copy()
        idx = pf.index
        realised = pd.to_numeric(pf["period_realised_pl"], errors="coerce").fillna(0.0)

        # Next-period return (fraction of capital), the forward label.
        next_ret = realised.shift(-1) / self.initial_capital
        targets: Dict[str, pd.Series] = {}
        targets[TargetKind.NEXT_PERIOD_RETURN.value] = next_ret

        # Next-period Adjusted RAR% over a short forward window of the equity path.
        equity = self.initial_capital + realised.cumsum()
        equity_df = pd.DataFrame({"date": idx, "equity": equity.values})
        window = max(3, min(6, len(pf) // 4))
        fwd_rar = []
        for i in range(len(pf)):
            seg = equity_df.iloc[i: i + window + 1]
            fwd_rar.append(adjusted_rar_from_equity(seg, self.adjusted_rar) if len(seg) >= 2 else np.nan)
        targets[TargetKind.NEXT_PERIOD_ADJ_RAR.value] = pd.Series(fwd_rar, index=idx)

        # Regime label from next-period return: favourable / neutral / hostile.
        fav = self.spec.regime_favourable_pct / 100.0
        hos = self.spec.regime_hostile_pct / 100.0
        regime = pd.Series("neutral", index=idx, dtype=object)
        regime[next_ret > fav] = "favourable"
        regime[next_ret < hos] = "hostile"
        regime[next_ret.isna()] = np.nan
        targets[TargetKind.REGIME_LABEL.value] = regime

        primary = self.spec.primary
        if primary.value not in targets:
            primary = TargetKind.NEXT_PERIOD_RETURN
        if primary == TargetKind.REGIME_LABEL:
            # Model the 3-class regime as a binary "favourable vs rest" target — the
            # actionable question is whether the strategy is in a favourable regime,
            # and the binary form drives the calibrated probability / exposure overlay.
            # The full favourable/neutral/hostile label is kept in `targets` for the
            # descriptive timeline.
            y = pd.Series(np.where(regime.values == "favourable", 1.0, 0.0), index=idx)
            y[regime.isna().values] = np.nan
            is_classification = True
        else:
            y = targets[primary.value]
            is_classification = primary.is_classification

        # Label window for period t is [t, t+1] (the next period the label measures).
        period_index = pd.Series(idx, index=idx)
        label_start = period_index
        label_end = pd.Series(idx, index=idx).shift(-1).fillna(idx[-1])

        ts = TargetSet(
            primary=primary, y=y, targets=targets,
            label_start=label_start, label_end=label_end,
            is_classification=is_classification,
        )
        # Drop the final period (no next-period label).
        valid = y.notna()
        ts.y = y[valid]
        ts.label_start = label_start[valid]
        ts.label_end = label_end[valid]
        ts.targets = {k: v[valid] if hasattr(v, "__getitem__") else v for k, v in targets.items()}
        ts.sample_weight = self._sample_weight(ts.y, ts.y.values, ts.y.index, ts.is_classification)
        ts.class_balance = self._class_balance(ts.targets, primary)
        return ts

    # -- weighting & balance ------------------------------------------------ #
    def _sample_weight(self, y: pd.Series, economic_value: np.ndarray,
                       idx: pd.Index, is_classification: bool) -> pd.Series:
        n = len(y)
        if self.weight_mode == WeightMode.EQUAL:
            return pd.Series(np.ones(n), index=y.index)
        if self.weight_mode == WeightMode.CLASS_BALANCED and is_classification:
            counts = y.value_counts()
            n_classes = len(counts)
            w = y.map(lambda c: n / (n_classes * counts.get(c, 1)))
            return pd.Series(w.values, index=y.index)
        if self.weight_mode == WeightMode.ECONOMIC:
            mag = np.abs(np.asarray(economic_value, dtype=float))
            mag = np.where(np.isfinite(mag), mag, 0.0)
            scale = mag.mean() if mag.mean() > 0 else 1.0
            return pd.Series(np.clip(mag / scale, 0.05, 20.0), index=y.index)
        return pd.Series(np.ones(n), index=y.index)

    @staticmethod
    def _class_balance(targets: Dict[str, pd.Series], primary: TargetKind) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, series in targets.items():
            kind = TargetKind(name)
            if kind.is_classification:
                vc = series.dropna().value_counts()
                total = int(vc.sum()) or 1
                out[name] = {str(k): {"count": int(v), "pct": round(100.0 * v / total, 1)}
                             for k, v in vc.items()}
            else:
                s = pd.to_numeric(series, errors="coerce").dropna()
                out[name] = {"mean": float(s.mean()) if len(s) else 0.0,
                             "std": float(s.std()) if len(s) else 0.0,
                             "n": int(len(s))}
        out["_primary"] = primary.value
        return out
