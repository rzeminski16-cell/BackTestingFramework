"""
Pattern Analysis Excel Report Generator.

Consumes a ``PatternAnalysisResult`` from
``Classes.Analysis.pattern_analyzer`` and writes a multi-sheet Excel report:

    1. Overview        - run configuration, trade counts, sheet guide.
    2. Per-Trade       - one row per (trade, combo, window) with all features.
    3. Summary         - aggregates per (combo, window) and split by win/loss.
    4. Win-Loss Stats  - per-feature mean/median split by trade outcome with a
                          simple ``win - loss`` delta (the larger the
                          magnitude, the more discriminative the feature).
    5. Correlations    - Pearson correlation between each numeric feature
                          and trade pl / pl_pct, computed per (combo, window).
    6. Raw Signals     - every signal detected within the widest window for
                          every (trade, combo). Useful for ad-hoc analysis.
    7. Skipped Trades  - trades dropped during analysis with a reason.

The report intentionally separates information by combo and window so the
user can spot which (combo, window) pair is most predictive without losing
the per-trade detail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .pattern_analyzer import PatternAnalysisResult


# Numeric feature columns we summarise / correlate.
NUMERIC_FEATURE_COLS: List[str] = [
    "buy_count",
    "sell_count",
    "total_signals",
    "net_signals",
    "signals_per_30d",
    "days_since_last_buy",
    "days_since_last_sell",
    "days_since_last_signal",
    "longest_buy_run",
    "longest_sell_run",
    "alternation_rate",
]


def write_report(
    result: PatternAnalysisResult,
    output_path: Path,
    *,
    trade_log_path: Optional[Path] = None,
) -> Path:
    """Write the full Excel report to ``output_path``.

    Args:
        result: Output of ``PatternAnalyzer.analyze``.
        output_path: Destination ``.xlsx`` path.
        trade_log_path: Optional path to the source trade log (for the
            overview sheet).

    Returns:
        The path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = result.features.copy()
    raw_signals = result.raw_signals.copy()
    skipped = result.skipped_trades.copy()

    overview = _build_overview(result, trade_log_path)
    summary = _build_summary(features)
    win_loss = _build_win_loss_stats(features)
    correlations = _build_correlations(features)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="Overview", index=False)
        if not features.empty:
            features.to_excel(writer, sheet_name="Per-Trade", index=False)
        if not summary.empty:
            summary.to_excel(writer, sheet_name="Summary", index=False)
        if not win_loss.empty:
            win_loss.to_excel(writer, sheet_name="Win-Loss Stats", index=False)
        if not correlations.empty:
            correlations.to_excel(writer, sheet_name="Correlations", index=False)
        if not raw_signals.empty:
            raw_signals.to_excel(writer, sheet_name="Raw Signals", index=False)
        if not skipped.empty:
            skipped.to_excel(writer, sheet_name="Skipped Trades", index=False)

    return output_path


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def _build_overview(
    result: PatternAnalysisResult,
    trade_log_path: Optional[Path],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rows.append({"Field": "Trade log", "Value": str(trade_log_path) if trade_log_path else ""})
    rows.append({"Field": "Combos analyzed", "Value": ", ".join(c.label for c in result.combos)})
    rows.append({"Field": "Lookback windows (days)", "Value": ", ".join(str(w) for w in result.windows)})

    if not result.features.empty:
        unique_trades = result.features[["trade_id", "symbol", "entry_date"]].drop_duplicates()
        rows.append({"Field": "Trades analyzed", "Value": int(len(unique_trades))})
        if "is_winner" in result.features.columns:
            outcomes = (
                result.features.drop_duplicates(["trade_id", "symbol", "entry_date"])
                ["is_winner"]
            )
            rows.append({"Field": "Winners", "Value": int((outcomes == True).sum())})
            rows.append({"Field": "Losers", "Value": int((outcomes == False).sum())})
            rows.append({"Field": "Outcome unknown", "Value": int(outcomes.isna().sum())})
    else:
        rows.append({"Field": "Trades analyzed", "Value": 0})

    rows.append({"Field": "Trades skipped", "Value": int(len(result.skipped_trades))})
    rows.append({"Field": "", "Value": ""})

    rows.append({"Field": "Sheet", "Value": "Description"})
    rows.append({"Field": "Per-Trade",
                 "Value": "One row per (trade, combo, window) with all features."})
    rows.append({"Field": "Summary",
                 "Value": "Mean/median features per (combo, window). Outcome breakdown."})
    rows.append({"Field": "Win-Loss Stats",
                 "Value": "Mean of each feature for winners vs losers; delta = win - loss."})
    rows.append({"Field": "Correlations",
                 "Value": "Pearson correlation of each numeric feature vs pl / pl_pct."})
    rows.append({"Field": "Raw Signals",
                 "Value": "Every signal detected within the widest window."})
    rows.append({"Field": "Skipped Trades",
                 "Value": "Trades dropped (missing data, unmatched entry date, etc.)."})

    return pd.DataFrame(rows)


def _build_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()

    grouped = features.groupby(["combo_label", "window_days"], dropna=False)

    rows: List[Dict[str, object]] = []
    for (combo_label, window), group in grouped:
        outcomes = group["is_winner"]
        n_total = len(group)
        n_winners = int((outcomes == True).sum())
        n_losers = int((outcomes == False).sum())
        win_rate = n_winners / (n_winners + n_losers) if (n_winners + n_losers) else np.nan

        row: Dict[str, object] = {
            "combo_label": combo_label,
            "window_days": int(window),
            "n_trades": int(n_total),
            "n_winners": n_winners,
            "n_losers": n_losers,
            "win_rate": round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
            "avg_pl": _safe_mean(group["pl"]),
            "avg_pl_pct": _safe_mean(group["pl_pct"]),
        }
        for col in NUMERIC_FEATURE_COLS:
            row[f"avg_{col}"] = _safe_mean(group[col])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.sort_values(["combo_label", "window_days"], inplace=True)
    return summary.reset_index(drop=True)


def _build_win_loss_stats(features: pd.DataFrame) -> pd.DataFrame:
    """For each (combo, window, feature), compare winners vs losers.

    Output columns:
        combo_label, window_days, feature, n_winners, n_losers,
        winner_mean, loser_mean, delta_mean, winner_median, loser_median
    """
    if features.empty or "is_winner" not in features.columns:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    grouped = features.groupby(["combo_label", "window_days"], dropna=False)

    for (combo_label, window), group in grouped:
        winners = group[group["is_winner"] == True]
        losers = group[group["is_winner"] == False]
        if winners.empty and losers.empty:
            continue
        for col in NUMERIC_FEATURE_COLS:
            wm = _safe_mean(winners[col])
            lm = _safe_mean(losers[col])
            delta = (wm - lm) if (not _isnan(wm) and not _isnan(lm)) else np.nan
            rows.append({
                "combo_label": combo_label,
                "window_days": int(window),
                "feature": col,
                "n_winners": int(len(winners)),
                "n_losers": int(len(losers)),
                "winner_mean": wm,
                "loser_mean": lm,
                "delta_mean": delta,
                "winner_median": _safe_median(winners[col]),
                "loser_median": _safe_median(losers[col]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Sort with most discriminative features on top per (combo, window).
    df["abs_delta"] = df["delta_mean"].abs()
    df.sort_values(
        ["combo_label", "window_days", "abs_delta"],
        ascending=[True, True, False],
        inplace=True,
    )
    df.drop(columns=["abs_delta"], inplace=True)
    return df.reset_index(drop=True)


def _build_correlations(features: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of each feature against pl and pl_pct.

    Output columns:
        combo_label, window_days, feature, n, corr_pl, corr_pl_pct
    """
    if features.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    grouped = features.groupby(["combo_label", "window_days"], dropna=False)

    for (combo_label, window), group in grouped:
        for col in NUMERIC_FEATURE_COLS:
            corr_pl = _safe_corr(group[col], group["pl"])
            corr_pct = _safe_corr(group[col], group["pl_pct"])
            n = int(group[[col, "pl"]].dropna().shape[0])
            rows.append({
                "combo_label": combo_label,
                "window_days": int(window),
                "feature": col,
                "n": n,
                "corr_pl": corr_pl,
                "corr_pl_pct": corr_pct,
            })

    df = pd.DataFrame(rows)
    df.sort_values(
        ["combo_label", "window_days", "feature"], inplace=True
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.mean())


def _safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.median())


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    a = df.iloc[:, 0]
    b = df.iloc[:, 1]
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return float("nan")
    return float(a.corr(b))


def _isnan(v: float) -> bool:
    try:
        return np.isnan(v)
    except TypeError:
        return False
