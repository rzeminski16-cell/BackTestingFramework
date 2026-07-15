"""
Reject-cooldown tracking for interactive runs.

Semantics (documented in the plan and README):

- DEFER remembers nothing; any later qualifying signal prompts again.
- REJECT starts a suppression window for (symbol, signal_type). A
  subsequent firing is *continuous* iff its per-symbol bar index is
  exactly one past the last seen firing. While continuous and fewer than
  ``cooldown_bars`` firings have been suppressed, the signal is
  auto-suppressed (logged, no prompt). Once the cooldown is exhausted
  under a sustained signal — or the signal stops firing and later
  re-fires fresh — the entry is cleared and the user is prompted again.
- Entries are cleared when the symbol's position state changes (a BUY
  suppression when a position opens; exit-type suppressions when the
  position closes).

State is derived only: it is rebuilt through the same code path when a
run is resumed by replay, so it can never diverge from the decision log.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_EXIT_TYPES = ("SELL", "PARTIAL_EXIT", "PYRAMID")


@dataclass
class _SuppressionEntry:
    reject_decision_id: int
    last_bar_index: int
    suppressed_count: int = 0


class RejectCooldownTracker:
    """Tracks rejected-signal suppression windows per (symbol, signal_type)."""

    def __init__(self, cooldown_bars: int = 21):
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative")
        self.cooldown_bars = cooldown_bars
        self._entries: Dict[Tuple[str, str], _SuppressionEntry] = {}

    def on_reject(self, symbol: str, signal_type: str,
                  decision_id: int, bar_index: int) -> None:
        """Start (or restart) a suppression window after a user REJECT."""
        self._entries[(symbol, signal_type)] = _SuppressionEntry(
            reject_decision_id=decision_id,
            last_bar_index=bar_index,
        )

    def check(self, symbol: str, signal_type: str, bar_index: int) -> Optional[int]:
        """
        Called when a qualifying signal fires. Returns the originating
        REJECT's decision_id if this firing should be auto-suppressed,
        or None if the user should be prompted.
        """
        key = (symbol, signal_type)
        entry = self._entries.get(key)
        if entry is None:
            return None

        continuous = bar_index == entry.last_bar_index + 1
        if not continuous:
            # The signal died and re-fired fresh: prompt again.
            del self._entries[key]
            return None

        if entry.suppressed_count >= self.cooldown_bars:
            # Cooldown exhausted under a sustained signal: prompt again.
            del self._entries[key]
            return None

        entry.last_bar_index = bar_index
        entry.suppressed_count += 1
        return entry.reject_decision_id

    def clear(self, symbol: str, signal_type: str) -> None:
        self._entries.pop((symbol, signal_type), None)

    def on_position_opened(self, symbol: str) -> None:
        """A position opened: BUY suppression for the symbol is moot."""
        self.clear(symbol, "BUY")

    def on_position_closed(self, symbol: str) -> None:
        """The position closed: exit-type suppressions are moot."""
        for signal_type in _EXIT_TYPES:
            self.clear(symbol, signal_type)
