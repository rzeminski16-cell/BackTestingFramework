"""
Chronological, leakage-aware validation splitters.

All designs are time-ordered — never shuffled. For regular period panels an
expanding (or rolling) walk-forward mirrors learning through time. For irregular
per-trade data with overlapping outcomes, the purged & embargoed splitter is the
default: if a training trade's label window overlaps a test window's label window
it is removed from training, and an embargo around each test block prevents
nearby observations leaking through shared/serial outcomes.

Splitters are scikit-learn compatible (``split(X)`` yields integer
``(train_idx, test_idx)`` over the **time-sorted** row order) so they drop into
``GridSearchCV``/``cross_val_predict`` and the nested tuning loop. Callers must
sort their data by ``label_start`` first; :func:`sort_by_label_time` helps.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ValidationDesign, ValidationConfig


def sort_by_label_time(label_start: pd.Series) -> np.ndarray:
    """Return the positional order that sorts rows by label-window open time."""
    return np.argsort(pd.to_datetime(label_start).values, kind="stable")


class ChronologicalSplitter:
    """Base walk-forward splitter producing contiguous, time-ordered test blocks."""

    def __init__(self, n_splits: int = 5, min_train_size: int = 30,
                 rolling: bool = False, window_size: Optional[int] = None,
                 embargo_days: int = 0,
                 label_start: Optional[pd.Series] = None,
                 label_end: Optional[pd.Series] = None):
        self.n_splits = max(1, int(n_splits))
        self.min_train_size = max(1, int(min_train_size))
        self.rolling = rolling
        self.window_size = window_size
        self.embargo_days = max(0, int(embargo_days))
        self._label_start = (pd.to_datetime(label_start).values
                             if label_start is not None else None)
        self._label_end = (pd.to_datetime(label_end).values
                           if label_end is not None else None)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _purge_embargo(self, train: np.ndarray, test: np.ndarray) -> np.ndarray:
        """Drop train rows whose label window overlaps the embargoed test span."""
        if self._label_start is None or self._label_end is None or len(test) == 0:
            return train
        ls, le = self._label_start, self._label_end
        t_lo = np.min(ls[test])
        t_hi = np.max(le[test])
        emb = np.timedelta64(self.embargo_days, "D")
        lo, hi = t_lo - emb, t_hi + emb
        # Exclude train rows that overlap [lo, hi].
        overlap = (ls[train] <= hi) & (le[train] >= lo)
        return train[~overlap]

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X) if not hasattr(X, "shape") else X.shape[0]
        if n < self.min_train_size + self.n_splits:
            # Degenerate: yield a single train/test split if at all possible.
            cut = max(self.min_train_size, int(n * 0.7))
            if cut < n:
                train = np.arange(0, cut)
                test = np.arange(cut, n)
                train = self._purge_embargo(train, test)
                if len(train) >= 1 and len(test) >= 1:
                    yield train, test
            return

        fold = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            test_start = (i + 1) * fold
            test_end = n if i == self.n_splits - 1 else (i + 2) * fold
            test = np.arange(test_start, test_end)
            if self.rolling:
                win = self.window_size or fold * 1
                train_start = max(0, test_start - win)
                train = np.arange(train_start, test_start)
            else:
                train = np.arange(0, test_start)
            train = self._purge_embargo(train, test)
            if len(train) < self.min_train_size or len(test) == 0:
                continue
            yield train, test


def make_splitter(config: ValidationConfig,
                  label_start: Optional[pd.Series] = None,
                  label_end: Optional[pd.Series] = None) -> ChronologicalSplitter:
    """Build the outer splitter for a validation design."""
    if config.design == ValidationDesign.ROLLING_WALK_FORWARD:
        return ChronologicalSplitter(
            n_splits=config.n_splits, min_train_size=config.min_train_size,
            rolling=True, embargo_days=config.embargo_days,
            label_start=label_start, label_end=label_end,
        )
    if config.design == ValidationDesign.PURGED_EMBARGOED:
        return ChronologicalSplitter(
            n_splits=config.n_splits, min_train_size=config.min_train_size,
            rolling=False, embargo_days=config.embargo_days,
            label_start=label_start, label_end=label_end,
        )
    # Expanding walk-forward (default for regular period panels).
    return ChronologicalSplitter(
        n_splits=config.n_splits, min_train_size=config.min_train_size,
        rolling=False, embargo_days=0,
        label_start=label_start, label_end=label_end,
    )


def make_inner_splitter(config: ValidationConfig,
                        label_start: Optional[pd.Series] = None,
                        label_end: Optional[pd.Series] = None) -> ChronologicalSplitter:
    """Smaller inner splitter for nested hyper-parameter search."""
    return ChronologicalSplitter(
        n_splits=max(2, config.inner_splits),
        min_train_size=max(5, config.min_train_size // 2),
        rolling=False, embargo_days=config.embargo_days,
        label_start=label_start, label_end=label_end,
    )


def fold_preview(splitter: ChronologicalSplitter, n_rows: int,
                 label_start: Optional[pd.Series] = None) -> List[dict]:
    """Human-readable summary of the folds (for the GUI's fold preview)."""
    out: List[dict] = []
    X = np.zeros((n_rows, 1))
    ts = pd.to_datetime(label_start).values if label_start is not None else None
    for k, (train, test) in enumerate(splitter.split(X), start=1):
        row = {"fold": k, "n_train": int(len(train)), "n_test": int(len(test))}
        if ts is not None and len(test):
            row["test_start"] = str(pd.Timestamp(ts[test].min()).date())
            row["test_end"] = str(pd.Timestamp(ts[test].max()).date())
        out.append(row)
    return out
