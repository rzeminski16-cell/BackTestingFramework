"""
Entity mapping -- resolve each trade to its benchmark(s), currency and sector.

The benchmark/currency/sector mapping must not be hidden in code: it is surfaced
in the GUI and written to ``entity_mapping.parquet`` (and echoed in the
manifest). This module turns the run's mapping rules into a concrete one-row-per-
trade table and reports any unmapped trades so validation can flag them.

Benchmark resolution order for a trade is most-specific first:
    symbol  ->  asset_class  ->  "default"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class EntityMapping:
    """The resolved trade->entity mapping table plus diagnostics."""
    table: pd.DataFrame
    unmapped_benchmarks: List[str] = field(default_factory=list)   # trade_ids w/o benchmark
    unmapped_currencies: List[str] = field(default_factory=list)   # trade_ids w/o currency
    benchmark_symbols: List[str] = field(default_factory=list)     # distinct benchmarks used

    @property
    def empty(self) -> bool:
        return self.table is None or self.table.empty


class EntityMapper:
    """Builds the per-trade entity mapping from run mapping rules."""

    def __init__(
        self,
        benchmark_map: Optional[Dict[str, List[str]]] = None,
        currency_map: Optional[Dict[str, str]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        base_currency: str = "GBP",
    ):
        self.benchmark_map = benchmark_map or {}
        self.currency_map = currency_map or {}
        self.sector_map = sector_map or {}
        self.base_currency = base_currency

    def _benchmarks_for(self, symbol: str, asset_class: str) -> List[str]:
        for key in (symbol, asset_class, "default"):
            if key and key in self.benchmark_map:
                return list(self.benchmark_map[key])
        return []

    def build(self, trades: pd.DataFrame) -> EntityMapping:
        """
        Build the entity-mapping table from the trade universe.

        Currency is taken from the trade's own ``currency`` column where present
        (the framework already tracks per-trade security currency), otherwise the
        ``currency_map`` override, otherwise the base currency.
        """
        if trades is None or trades.empty:
            return EntityMapping(pd.DataFrame())

        rows: List[Dict[str, Any]] = []
        unmapped_bm: List[str] = []
        unmapped_ccy: List[str] = []
        all_benchmarks: set = set()

        for _, t in trades.iterrows():
            trade_id = t.get("trade_id")
            symbol = t.get("symbol")
            asset_class = t.get("asset_class") or "equity"

            benchmarks = self._benchmarks_for(symbol, asset_class)
            if not benchmarks:
                unmapped_bm.append(str(trade_id))
            all_benchmarks.update(benchmarks)

            # Currency: trade column -> override map -> base currency.
            currency = t.get("currency")
            if not isinstance(currency, str) or not currency:
                currency = self.currency_map.get(symbol)
            if not currency:
                currency = self.base_currency
                if symbol not in self.currency_map and "currency" not in trades.columns:
                    unmapped_ccy.append(str(trade_id))

            sector = self.sector_map.get(symbol, "")

            rows.append({
                "trade_id": trade_id,
                "symbol": symbol,
                "asset_class": asset_class,
                "benchmarks": ",".join(benchmarks),
                "currency": currency,
                "base_currency": self.base_currency,
                "needs_fx_conversion": bool(currency and currency != self.base_currency),
                "sector": sector,
            })

        table = pd.DataFrame(rows)
        return EntityMapping(
            table=table,
            unmapped_benchmarks=unmapped_bm,
            unmapped_currencies=unmapped_ccy,
            benchmark_symbols=sorted(all_benchmarks),
        )
