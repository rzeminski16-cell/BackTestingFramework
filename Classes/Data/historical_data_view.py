"""
Historical Data View - Look-ahead bias protection without copying.

This module provides a lightweight wrapper around pandas DataFrames that:
1. Prevents access to future data (look-ahead bias protection)
2. Avoids expensive copy operations on every bar
3. Maintains the same API as pandas DataFrame for backward compatibility

CRITICAL: This class is designed to prevent data leakage. It should be
impossible for strategies to access data beyond the valid_end_index.
"""
import numbers
import pandas as pd
import numpy as np
from typing import Any, Optional, Union, List


class HistoricalDataView:
    """
    A read-only view of a DataFrame that enforces look-ahead protection.

    This wrapper prevents strategies from accidentally (or intentionally)
    accessing future data while avoiding the O(n²) cost of copying the
    DataFrame on every bar.

    SAFETY GUARANTEES:
    - iloc[i] where i > valid_end_index raises IndexError
    - len() returns valid_end_index + 1 (not full DataFrame length)
    - Iteration only covers valid range
    - All access methods enforce the index limit

    Usage:
        view = HistoricalDataView(df, valid_end_index=100)
        # Can access df.iloc[0:100] but NOT df.iloc[101:]
    """

    __slots__ = ('_data', '_valid_end_index', '_columns')

    def __init__(self, data: pd.DataFrame, valid_end_index: int):
        """
        Create a historical data view.

        Args:
            data: Full DataFrame (not copied, just referenced)
            valid_end_index: Maximum accessible index (inclusive)
                            Indices 0 to valid_end_index are accessible.
                            Any access beyond this raises IndexError.

        Raises:
            ValueError: If valid_end_index is out of bounds
        """
        if valid_end_index < 0:
            raise ValueError(f"valid_end_index must be >= 0, got {valid_end_index}")
        if valid_end_index >= len(data):
            raise ValueError(
                f"valid_end_index ({valid_end_index}) must be < len(data) ({len(data)})"
            )

        self._data = data
        self._valid_end_index = valid_end_index
        self._columns = data.columns

    def __len__(self) -> int:
        """Return the number of accessible rows (not full DataFrame length)."""
        return self._valid_end_index + 1

    def __getitem__(self, key: str) -> pd.Series:
        """
        Get a column by name, limited to valid range.

        Args:
            key: Column name

        Returns:
            Series containing only valid historical data
        """
        if key not in self._columns:
            raise KeyError(f"Column '{key}' not found")
        # Return a view of the valid portion only
        return self._data[key].iloc[:self._valid_end_index + 1]

    @property
    def iloc(self) -> '_ILocIndexer':
        """Provide iloc accessor with look-ahead protection."""
        return _ILocIndexer(self._data, self._valid_end_index)

    @property
    def loc(self) -> '_LocIndexer':
        """Provide loc accessor with look-ahead protection."""
        return _LocIndexer(self._data, self._valid_end_index)

    @property
    def columns(self) -> pd.Index:
        """Return column names."""
        return self._columns

    @property
    def shape(self) -> tuple:
        """Return shape limited to valid data."""
        return (self._valid_end_index + 1, len(self._columns))

    @property
    def values(self) -> np.ndarray:
        """Return values as numpy array, limited to valid range."""
        return self._data.iloc[:self._valid_end_index + 1].values

    def __contains__(self, key: str) -> bool:
        """Check if column exists."""
        return key in self._columns

    def __iter__(self):
        """Iterate over column names."""
        return iter(self._columns)

    def iterrows(self):
        """Iterate over rows, limited to valid range."""
        for i in range(self._valid_end_index + 1):
            yield i, self._data.iloc[i]

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first n rows (limited to valid range)."""
        n = min(n, self._valid_end_index + 1)
        return self._data.iloc[:n].copy()

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Return last n rows of valid data."""
        start = max(0, self._valid_end_index + 1 - n)
        return self._data.iloc[start:self._valid_end_index + 1].copy()

    def copy(self) -> pd.DataFrame:
        """Return a copy of the valid data (for compatibility)."""
        return self._data.iloc[:self._valid_end_index + 1].copy()

    def get_valid_end_index(self) -> int:
        """Return the maximum accessible index."""
        return self._valid_end_index

    def __repr__(self) -> str:
        return (f"HistoricalDataView(rows={len(self)}, "
                f"columns={len(self._columns)}, "
                f"valid_end_index={self._valid_end_index})")


class _ILocIndexer:
    """
    iloc indexer with look-ahead protection.

    Enforces that no access is allowed beyond valid_end_index.
    """

    __slots__ = ('_data', '_valid_end_index')

    def __init__(self, data: pd.DataFrame, valid_end_index: int):
        self._data = data
        self._valid_end_index = valid_end_index

    def __getitem__(self, key) -> Union[pd.Series, pd.DataFrame]:
        """
        Get rows/columns by integer position with look-ahead protection.

        Args:
            key: Integer index, slice, or tuple

        Returns:
            Row(s) as Series or DataFrame

        Raises:
            IndexError: If attempting to access beyond valid_end_index
        """
        if isinstance(key, numbers.Integral):
            # Single row access (handles both Python int and numpy.int64)
            key = int(key)  # Convert to Python int for safety
            if key < 0:
                # Convert negative index to positive
                key = self._valid_end_index + 1 + key
            if key > self._valid_end_index:
                raise IndexError(
                    f"Index {key} is beyond valid historical data "
                    f"(max: {self._valid_end_index}). "
                    f"Accessing future data is not allowed."
                )
            if key < 0:
                raise IndexError(f"Index {key} is out of bounds")
            return self._data.iloc[key]

        elif isinstance(key, slice):
            # Slice access - enforce limits
            start, stop, step = key.start, key.stop, key.step

            # Handle None values
            if start is None:
                start = 0
            if stop is None:
                stop = self._valid_end_index + 1
            if step is None:
                step = 1

            # Handle negative indices
            if start < 0:
                start = max(0, self._valid_end_index + 1 + start)
            if stop < 0:
                stop = self._valid_end_index + 1 + stop

            # Enforce look-ahead protection
            if stop > self._valid_end_index + 1:
                stop = self._valid_end_index + 1
            if start > self._valid_end_index:
                # Accessing entirely future data - return empty
                return self._data.iloc[0:0]

            return self._data.iloc[start:stop:step]

        elif isinstance(key, tuple):
            # Row and column selection
            row_key, col_key = key

            # First apply row restrictions
            if isinstance(row_key, numbers.Integral):
                # Handle both Python int and numpy.int64
                row_key = int(row_key)
                if row_key < 0:
                    row_key = self._valid_end_index + 1 + row_key
                if row_key > self._valid_end_index:
                    raise IndexError(
                        f"Row index {row_key} is beyond valid historical data "
                        f"(max: {self._valid_end_index})"
                    )
                return self._data.iloc[row_key, col_key]

            elif isinstance(row_key, slice):
                start, stop, step = row_key.start, row_key.stop, row_key.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = self._valid_end_index + 1
                if step is None:
                    step = 1

                if start < 0:
                    start = max(0, self._valid_end_index + 1 + start)
                if stop < 0:
                    stop = self._valid_end_index + 1 + stop
                if stop > self._valid_end_index + 1:
                    stop = self._valid_end_index + 1

                return self._data.iloc[start:stop:step, col_key]

            else:
                # List or array of indices
                valid_indices = [
                    i if i >= 0 else self._valid_end_index + 1 + i
                    for i in row_key
                ]
                for i in valid_indices:
                    if i > self._valid_end_index:
                        raise IndexError(
                            f"Index {i} is beyond valid historical data "
                            f"(max: {self._valid_end_index})"
                        )
                return self._data.iloc[valid_indices, col_key]

        else:
            # List or array of indices
            valid_indices = []
            for i in key:
                if i < 0:
                    i = self._valid_end_index + 1 + i
                if i > self._valid_end_index:
                    raise IndexError(
                        f"Index {i} is beyond valid historical data "
                        f"(max: {self._valid_end_index})"
                    )
                valid_indices.append(i)
            return self._data.iloc[valid_indices]


class _LocIndexer:
    """
    loc indexer with look-ahead protection.

    Enforces that no access is allowed beyond valid_end_index.
    """

    __slots__ = ('_data', '_valid_end_index')

    def __init__(self, data: pd.DataFrame, valid_end_index: int):
        self._data = data
        self._valid_end_index = valid_end_index

    def __getitem__(self, key) -> Union[pd.Series, pd.DataFrame]:
        """
        Get rows/columns by label with look-ahead protection.

        For DataFrames with integer index, this behaves similarly to iloc.
        For other index types, it filters based on the valid range.
        """
        # Get the valid data slice and use loc on that
        valid_data = self._data.iloc[:self._valid_end_index + 1]
        return valid_data.loc[key]
