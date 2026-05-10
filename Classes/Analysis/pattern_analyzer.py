"""
Pattern Analyzer - MA crossover signal pattern analysis around trade entries.

For each trade in a trade log, this module looks at the N days prior to entry
and detects buy/sell signals defined as crossovers between a moving average
(SMA or EMA, length L) and the same moving average shifted by ``offset`` bars.

Signal definition (mirrors the AlphaTrend strategy entry logic exactly):
    - The "shifted MA" is ``MA(L)`` shifted forward by ``offset`` bars (i.e.
      ``MA.shift(offset)``); it represents the MA's value ``offset`` bars ago.
    - BUY signal: the unshifted MA crosses from below the shifted MA to above
      it. Concretely on bar ``t``:
          ``base_prev <= shifted_prev`` and ``base_now > shifted_now``.
      In a rising market the unshifted MA tends to sit above the shifted one;
      a fresh BUY cross marks the transition into the rising regime. This is
      the same predicate used by ``AlphaTrendV*Strategy.generate_entry_signal``.
    - SELL signal: the unshifted MA crosses from above to below the shifted
      one. Concretely on bar ``t``:
          ``base_prev >= shifted_prev`` and ``base_now < shifted_now``.

The module is independent of the engine/strategy classes; it only needs the
precomputed ``ema_{N}_ema`` / ``sma_{N}_sma`` columns already present in
``raw_data/daily/{SYMBOL}_daily.csv`` files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

#: Supported MA lengths exposed to the user (matches AlphaTrend strategy).
SUPPORTED_MA_LENGTHS: Tuple[int, ...] = (7, 14, 20, 30, 50)

#: Supported MA types.
SUPPORTED_MA_TYPES: Tuple[str, ...] = ("SMA", "EMA")


@dataclass(frozen=True)
class MAComboSpec:
    """A single (type, length, offset) configuration for signal detection."""

    ma_type: str          # "SMA" or "EMA"
    ma_length: int        # 7, 14, 20, 30, 50
    ma_offset: int        # >= 1

    def __post_init__(self) -> None:
        ma_type = self.ma_type.upper()
        if ma_type not in SUPPORTED_MA_TYPES:
            raise ValueError(
                f"ma_type must be one of {SUPPORTED_MA_TYPES}, got {self.ma_type!r}"
            )
        if self.ma_length not in SUPPORTED_MA_LENGTHS:
            raise ValueError(
                f"ma_length must be one of {list(SUPPORTED_MA_LENGTHS)}, "
                f"got {self.ma_length}"
            )
        if self.ma_offset < 1:
            raise ValueError(f"ma_offset must be >= 1, got {self.ma_offset}")
        # Re-assign normalised type via object.__setattr__ since frozen.
        object.__setattr__(self, "ma_type", ma_type)

    @property
    def label(self) -> str:
        """Short label e.g. 'EMA20:o5'."""
        return f"{self.ma_type}{self.ma_length}:o{self.ma_offset}"

    @property
    def ma_column(self) -> str:
        """Column name in the daily-data CSV for the unshifted MA."""
        if self.ma_type == "EMA":
            return f"ema_{self.ma_length}_ema"
        return f"sma_{self.ma_length}_sma"


@dataclass
class Signal:
    """A single crossover signal detected on a bar."""

    date: pd.Timestamp
    signal_type: str       # "BUY" or "SELL"
    days_before_entry: int  # >= 0; 0 == on the entry bar itself
    bar_index: int          # absolute index in the price dataframe


@dataclass
class TradeRecord:
    """Trade-log entry consumed by the analyzer."""

    trade_id: Any
    symbol: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pl: Optional[float] = None
    pl_pct: Optional[float] = None
    side: str = "LONG"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_winner(self) -> Optional[bool]:
        """Return True if pl > 0, False if pl < 0, None if pl unknown."""
        if self.pl is None or pd.isna(self.pl):
            return None
        return self.pl > 0


@dataclass
class WindowFeatures:
    """All pattern features for one (trade, combo, window) tuple."""

    trade_id: Any
    symbol: str
    entry_date: pd.Timestamp
    combo_label: str
    window_days: int

    # Counts / density
    buy_count: int = 0
    sell_count: int = 0
    total_signals: int = 0
    net_signals: int = 0           # buy - sell
    signals_per_30d: float = 0.0   # density normalized to 30-day rate

    # Recency
    last_signal_type: Optional[str] = None       # "BUY" / "SELL" / None
    days_since_last_buy: Optional[int] = None
    days_since_last_sell: Optional[int] = None
    days_since_last_signal: Optional[int] = None

    # Sequence / pattern
    last5_sequence: str = ""              # e.g. "B,S,B,B,S" (most recent last)
    longest_buy_run: int = 0
    longest_sell_run: int = 0
    alternation_rate: float = 0.0         # fraction of adjacent pairs that flip

    # Outcome (carried through for win/loss correlation; analyzer fills these)
    pl: Optional[float] = None
    pl_pct: Optional[float] = None
    is_winner: Optional[bool] = None


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def detect_crossover_signals(
    price_df: pd.DataFrame,
    combo: MAComboSpec,
) -> pd.DataFrame:
    """Detect BUY/SELL crossovers for a single MA combo across the whole frame.

    Args:
        price_df: DataFrame with at minimum a ``date`` column and the MA
            column referenced by ``combo.ma_column``. Must be sorted ascending
            by date.
        combo: MA configuration.

    Returns:
        A DataFrame with one row per detected signal. Columns:
        ``date``, ``signal_type``, ``bar_index``. Empty if no signals or if
        the MA column is missing.
    """
    if combo.ma_column not in price_df.columns:
        return pd.DataFrame(columns=["date", "signal_type", "bar_index"])

    base = price_df[combo.ma_column].astype(float)
    shifted = base.shift(combo.ma_offset)

    base_prev = base.shift(1)
    shifted_prev = shifted.shift(1)

    valid = base.notna() & shifted.notna() & base_prev.notna() & shifted_prev.notna()

    # Same predicate as AlphaTrend: BUY when the unshifted MA crosses above
    # the shifted MA, SELL on the inverse cross.
    buy_mask = valid & (base_prev <= shifted_prev) & (base > shifted)
    sell_mask = valid & (base_prev >= shifted_prev) & (base < shifted)

    signal_types = np.where(buy_mask, "BUY", np.where(sell_mask, "SELL", None))
    mask = buy_mask | sell_mask
    if not mask.any():
        return pd.DataFrame(columns=["date", "signal_type", "bar_index"])

    idx = np.flatnonzero(mask.to_numpy())
    return pd.DataFrame({
        "date": pd.to_datetime(price_df["date"].iloc[idx].values),
        "signal_type": [signal_types[i] for i in idx],
        "bar_index": idx.astype(int),
    })


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _build_sequence_str(signals: Sequence[str], n: int = 5) -> str:
    """Compact tail-of-sequence string like 'B,S,B' (most recent last)."""
    if not signals:
        return ""
    tail = signals[-n:]
    return ",".join("B" if s == "BUY" else "S" for s in tail)


def _longest_run(signals: Sequence[str], target: str) -> int:
    """Longest consecutive run of ``target`` in ``signals``."""
    longest = current = 0
    for s in signals:
        if s == target:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _alternation_rate(signals: Sequence[str]) -> float:
    """Fraction of adjacent pairs (i, i+1) where signal type flips."""
    if len(signals) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(signals, signals[1:]) if a != b)
    return flips / (len(signals) - 1)


def compute_window_features(
    trade: TradeRecord,
    combo: MAComboSpec,
    signals_in_window: pd.DataFrame,
    window_days: int,
) -> WindowFeatures:
    """Compute pattern features from signals already filtered to the window.

    ``signals_in_window`` must have the columns produced by
    ``detect_crossover_signals`` plus ``days_before_entry`` (an int >= 0,
    where 0 means the signal is on the entry bar).
    """
    feats = WindowFeatures(
        trade_id=trade.trade_id,
        symbol=trade.symbol,
        entry_date=trade.entry_date,
        combo_label=combo.label,
        window_days=window_days,
        pl=trade.pl,
        pl_pct=trade.pl_pct,
        is_winner=trade.is_winner,
    )

    if signals_in_window.empty:
        return feats

    # Signals are sorted ascending by date (oldest -> newest). Most recent is
    # the one with the smallest days_before_entry.
    sigs = signals_in_window.sort_values("date").reset_index(drop=True)
    types = sigs["signal_type"].tolist()

    feats.buy_count = int((sigs["signal_type"] == "BUY").sum())
    feats.sell_count = int((sigs["signal_type"] == "SELL").sum())
    feats.total_signals = feats.buy_count + feats.sell_count
    feats.net_signals = feats.buy_count - feats.sell_count
    if window_days > 0:
        feats.signals_per_30d = feats.total_signals * (30.0 / window_days)

    feats.last_signal_type = types[-1]
    feats.days_since_last_signal = int(sigs["days_before_entry"].iloc[-1])

    buys = sigs[sigs["signal_type"] == "BUY"]
    if not buys.empty:
        feats.days_since_last_buy = int(buys["days_before_entry"].iloc[-1])
    sells = sigs[sigs["signal_type"] == "SELL"]
    if not sells.empty:
        feats.days_since_last_sell = int(sells["days_before_entry"].iloc[-1])

    feats.last5_sequence = _build_sequence_str(types, n=5)
    feats.longest_buy_run = _longest_run(types, "BUY")
    feats.longest_sell_run = _longest_run(types, "SELL")
    feats.alternation_rate = _alternation_rate(types)

    return feats


# ---------------------------------------------------------------------------
# Trade-log loading
# ---------------------------------------------------------------------------

def load_trade_log(filepath: Path) -> List[TradeRecord]:
    """Load a trade log CSV into a list of ``TradeRecord``s.

    Expected columns (best-effort tolerance for variants):
        trade_id, symbol, entry_date, exit_date, entry_price, exit_price,
        pl, pl_pct, side
    """
    df = pd.read_csv(filepath)

    def col(*names: str) -> Optional[str]:
        lower = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    trade_id_c = col("trade_id", "id")
    symbol_c = col("symbol", "ticker")
    entry_date_c = col("entry_date", "entry date", "date")
    exit_date_c = col("exit_date", "exit date")
    entry_price_c = col("entry_price", "entry price")
    exit_price_c = col("exit_price", "exit price")
    pl_c = col("pl", "pnl", "p/l", "profit_loss")
    pl_pct_c = col("pl_pct", "pnl_pct", "p/l %", "pl_percent")
    side_c = col("side", "direction")

    if symbol_c is None or entry_date_c is None:
        raise ValueError(
            f"Trade log missing required columns 'symbol' and 'entry_date'. "
            f"Found: {df.columns.tolist()}"
        )

    trades: List[TradeRecord] = []
    for i, row in df.iterrows():
        try:
            entry_date = pd.to_datetime(row[entry_date_c], errors="coerce")
            if pd.isna(entry_date):
                continue
            symbol = str(row[symbol_c]).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            trade = TradeRecord(
                trade_id=row[trade_id_c] if trade_id_c else i,
                symbol=symbol,
                entry_date=entry_date,
                exit_date=pd.to_datetime(row[exit_date_c], errors="coerce") if exit_date_c else None,
                entry_price=_safe_float(row.get(entry_price_c)) if entry_price_c else None,
                exit_price=_safe_float(row.get(exit_price_c)) if exit_price_c else None,
                pl=_safe_float(row.get(pl_c)) if pl_c else None,
                pl_pct=_safe_float(row.get(pl_pct_c)) if pl_pct_c else None,
                side=str(row.get(side_c, "LONG")).strip().upper() if side_c else "LONG",
                raw=row.to_dict(),
            )
            trades.append(trade)
        except Exception:
            continue
    return trades


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Analyzer orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PatternAnalysisResult:
    """Aggregate result returned by ``PatternAnalyzer.analyze``."""

    features: pd.DataFrame                        # one row per (trade, combo, window)
    raw_signals: pd.DataFrame                     # one row per signal
    skipped_trades: pd.DataFrame                  # trades dropped (with reason)
    combos: List[MAComboSpec]
    windows: List[int]


class PatternAnalyzer:
    """Run pattern analysis across trades, MA combos, and lookback windows.

    Args:
        data_path: Directory holding ``{SYMBOL}_daily.csv`` files with the
            precomputed ``ema_*_ema`` / ``sma_*_sma`` columns.
        combos: List of ``MAComboSpec`` to evaluate.
        windows: Sorted list of lookback windows in days (e.g. [30, 60, 90,
            120]). Each is used as a side-by-side analysis horizon.
    """

    def __init__(
        self,
        data_path: Path,
        combos: Sequence[MAComboSpec],
        windows: Sequence[int],
    ):
        self.data_path = Path(data_path)
        self.combos = list(combos)
        self.windows = sorted(int(w) for w in windows)
        if not self.combos:
            raise ValueError("At least one MAComboSpec is required.")
        if not self.windows:
            raise ValueError("At least one lookback window (in days) is required.")
        if any(w <= 0 for w in self.windows):
            raise ValueError("All lookback windows must be > 0 days.")

        self._price_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        trades: Sequence[TradeRecord],
        progress: Optional[callable] = None,
    ) -> PatternAnalysisResult:
        """Compute features for every (trade, combo, window).

        Args:
            trades: Trade records to analyze.
            progress: Optional callback ``progress(done, total, message)``
                invoked after each trade is processed.

        Returns:
            ``PatternAnalysisResult`` with three DataFrames.
        """
        feature_rows: List[Dict[str, Any]] = []
        signal_rows: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        total = len(trades)
        for i, trade in enumerate(trades):
            try:
                price_df = self._get_price_data(trade.symbol)
            except FileNotFoundError as exc:
                skipped.append({
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "entry_date": trade.entry_date,
                    "reason": f"Price data not found: {exc}",
                })
                if progress:
                    progress(i + 1, total, f"Skipped {trade.symbol}: no data")
                continue

            entry_pos = self._locate_entry_index(price_df, trade.entry_date)
            if entry_pos is None:
                skipped.append({
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "entry_date": trade.entry_date,
                    "reason": "Entry date not found in price data",
                })
                if progress:
                    progress(i + 1, total, f"Skipped {trade.symbol}: bad date")
                continue

            for combo in self.combos:
                signals = detect_crossover_signals(price_df, combo)
                if signals.empty:
                    # Still emit empty-feature rows so the report is rectangular.
                    for w in self.windows:
                        feats = WindowFeatures(
                            trade_id=trade.trade_id,
                            symbol=trade.symbol,
                            entry_date=trade.entry_date,
                            combo_label=combo.label,
                            window_days=w,
                            pl=trade.pl,
                            pl_pct=trade.pl_pct,
                            is_winner=trade.is_winner,
                        )
                        feature_rows.append(_features_to_dict(feats, combo))
                    continue

                # days_before_entry counts trading-bars between signal and
                # entry. Signals strictly before the entry bar are kept; the
                # entry bar itself is excluded so we don't bake in the entry
                # signal as an "early warning".
                signals = signals[signals["bar_index"] < entry_pos].copy()
                if not signals.empty:
                    signals["days_before_entry"] = entry_pos - signals["bar_index"]

                    # Record full signal log per trade/combo (across the
                    # widest window so the user can analyze finer-grained
                    # cuts later).
                    widest = max(self.windows)
                    raw_keep = signals[signals["days_before_entry"] <= widest]
                    for _, srow in raw_keep.iterrows():
                        signal_rows.append({
                            "trade_id": trade.trade_id,
                            "symbol": trade.symbol,
                            "entry_date": trade.entry_date,
                            "combo_label": combo.label,
                            "signal_date": srow["date"],
                            "signal_type": srow["signal_type"],
                            "days_before_entry": int(srow["days_before_entry"]),
                        })

                for w in self.windows:
                    if signals.empty:
                        windowed = signals
                    else:
                        windowed = signals[signals["days_before_entry"] <= w]
                    feats = compute_window_features(trade, combo, windowed, w)
                    feature_rows.append(_features_to_dict(feats, combo))

            if progress:
                progress(i + 1, total, f"Analyzed {trade.symbol} {trade.entry_date.date()}")

        features_df = pd.DataFrame(feature_rows)
        raw_signals_df = pd.DataFrame(signal_rows)
        skipped_df = pd.DataFrame(skipped)

        return PatternAnalysisResult(
            features=features_df,
            raw_signals=raw_signals_df,
            skipped_trades=skipped_df,
            combos=self.combos,
            windows=self.windows,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_price_data(self, symbol: str) -> pd.DataFrame:
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        path = self.data_path / f"{symbol}_daily.csv"
        if not path.exists():
            raise FileNotFoundError(str(path))

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        self._price_cache[symbol] = df
        return df

    @staticmethod
    def _locate_entry_index(price_df: pd.DataFrame, entry_date: pd.Timestamp) -> Optional[int]:
        target = pd.Timestamp(entry_date).normalize()
        dates = price_df["date"].dt.normalize()
        match = dates == target
        if match.any():
            return int(np.flatnonzero(match.to_numpy())[0])

        # Fallback: closest bar within +/- 3 calendar days, prefer the
        # earliest bar at or after the entry date.
        diffs = (dates - target).dt.days
        within = diffs.between(0, 3) | diffs.between(-3, 0)
        if within.any():
            candidate = diffs[within].abs().idxmin()
            return int(candidate)
        return None


def _features_to_dict(feats: WindowFeatures, combo: MAComboSpec) -> Dict[str, Any]:
    """Flatten ``WindowFeatures`` to a dict suitable for a DataFrame row."""
    return {
        "trade_id": feats.trade_id,
        "symbol": feats.symbol,
        "entry_date": feats.entry_date,
        "combo_label": feats.combo_label,
        "ma_type": combo.ma_type,
        "ma_length": combo.ma_length,
        "ma_offset": combo.ma_offset,
        "window_days": feats.window_days,
        "buy_count": feats.buy_count,
        "sell_count": feats.sell_count,
        "total_signals": feats.total_signals,
        "net_signals": feats.net_signals,
        "signals_per_30d": round(feats.signals_per_30d, 3),
        "last_signal_type": feats.last_signal_type,
        "days_since_last_buy": feats.days_since_last_buy,
        "days_since_last_sell": feats.days_since_last_sell,
        "days_since_last_signal": feats.days_since_last_signal,
        "last5_sequence": feats.last5_sequence,
        "longest_buy_run": feats.longest_buy_run,
        "longest_sell_run": feats.longest_sell_run,
        "alternation_rate": round(feats.alternation_rate, 3),
        "pl": feats.pl,
        "pl_pct": feats.pl_pct,
        "is_winner": feats.is_winner,
    }


# ---------------------------------------------------------------------------
# Combo parsing helpers (used by CLI and GUI)
# ---------------------------------------------------------------------------

def parse_combo_string(spec: str) -> MAComboSpec:
    """Parse a CLI-style combo string into a ``MAComboSpec``.

    Accepted forms (case-insensitive, separator can be ``:`` or ``,`` or ``-``):
        ``EMA:20:5``       -> EMA length 20, offset 5
        ``sma,14,10``
        ``EMA-7-3``
    """
    parts = [p for p in spec.replace(",", ":").replace("-", ":").split(":") if p]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid combo spec {spec!r}. Expected 'TYPE:LENGTH:OFFSET' "
            f"e.g. 'EMA:20:5'."
        )
    return MAComboSpec(
        ma_type=parts[0].strip().upper(),
        ma_length=int(parts[1]),
        ma_offset=int(parts[2]),
    )


def load_combos_from_config(path: Path) -> Tuple[List[MAComboSpec], Optional[List[int]]]:
    """Load combos (and optionally windows) from a YAML or JSON config file.

    Expected schema (YAML or JSON):

        combos:
          - {ma_type: EMA, ma_length: 20, ma_offset: 5}
          - {ma_type: SMA, ma_length: 14, ma_offset: 10}
        windows: [30, 60, 90, 120]    # optional
    """
    path = Path(path)
    text = path.read_text()
    data: Dict[str, Any]
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read .yaml/.yml configs. "
                "Use a .json file or install pyyaml."
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        import json
        data = json.loads(text)

    combos_raw = data.get("combos") or []
    if not combos_raw:
        raise ValueError(f"Config {path} has no 'combos' entries.")

    combos: List[MAComboSpec] = []
    for c in combos_raw:
        if isinstance(c, str):
            combos.append(parse_combo_string(c))
        elif isinstance(c, dict):
            combos.append(MAComboSpec(
                ma_type=c["ma_type"],
                ma_length=int(c["ma_length"]),
                ma_offset=int(c["ma_offset"]),
            ))
        else:
            raise ValueError(f"Unsupported combo entry: {c!r}")

    windows = data.get("windows")
    if windows is not None:
        windows = [int(w) for w in windows]
    return combos, windows


def default_combos() -> List[MAComboSpec]:
    """A sensible starter set of combos for users who don't pass any."""
    return [
        MAComboSpec("EMA", 14, 5),
        MAComboSpec("EMA", 20, 5),
        MAComboSpec("EMA", 50, 10),
        MAComboSpec("SMA", 14, 5),
        MAComboSpec("SMA", 20, 10),
    ]


def default_windows() -> List[int]:
    """Default lookback windows in days."""
    return [30, 60, 90, 120]
