"""
Panel source builders -- assemble each family's normalised panel.

Bridges the raw inputs (the framework's local ``raw_data``/``processed_data``
CSVs and the Alpha Vantage collectors) to the canonical, point-in-time family
panels the run package needs. Each builder returns an already-normalised panel
(provenance + ``available_ts`` stamped via the run's timing policy) or an empty
frame, plus human-readable warnings for anything that could not be assembled.

Kept separate from the GUI so the assembly logic is unit-testable headlessly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .schema import Family
from .run import RunConfig
from .families import normalise_family_panel

# Local price-family layout: directory + filename suffix + native frequency.
_PRICE_SOURCES = {
    Family.EQUITY_PRICES: ("daily", "_daily.csv", "daily"),
    Family.INDEX: ("benchmarks", "_daily.csv", "daily"),
}
_PRICE_FIELDS = ["open", "high", "low", "close", "volume"]


class PanelSourceBuilder:
    """Builds normalised family panels from local CSVs and AV collectors."""

    def __init__(
        self,
        raw_data_dir: str = "raw_data",
        processed_dir: str = "processed_data",
        av_client: Any = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.av_client = av_client

    # -- public ------------------------------------------------------------- #
    def build_all(
        self,
        config: RunConfig,
        symbols: List[str],
        currencies: Optional[List[str]] = None,
    ) -> Tuple[Dict[Family, pd.DataFrame], List[str]]:
        """
        Build every included family's panel.

        Returns ``(panels, warnings)`` where ``panels`` maps family to a non-empty
        normalised DataFrame and ``warnings`` lists families/sources that came up
        empty or unavailable.
        """
        panels: Dict[Family, pd.DataFrame] = {}
        warnings: List[str] = []

        for fam in config.included_families():
            try:
                df = self._build_one(fam, config, symbols, currencies or [])
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"{fam.label}: {exc}")
                continue
            if df is None or df.empty:
                warnings.append(f"{fam.label}: no data assembled.")
            else:
                panels[fam] = df
        return panels, warnings

    # -- dispatch ----------------------------------------------------------- #
    def _build_one(
        self, fam: Family, config: RunConfig, symbols: List[str], currencies: List[str]
    ) -> pd.DataFrame:
        if fam in _PRICE_SOURCES:
            return self._build_price_panel(fam, config, symbols)
        if fam == Family.FX:
            return self._build_fx_panel(config, currencies)
        if fam == Family.FUNDAMENTALS:
            return self._build_fundamentals_panel(config, symbols)
        if fam == Family.COMMODITIES:
            return self._build_commodities_panel(config)
        if fam == Family.MACRO:
            return self._build_macro_panel(config)
        if fam == Family.CORPORATE_ACTIONS:
            return self._build_corporate_actions_panel(config, symbols)
        if fam == Family.UTILITIES:
            return self._build_utilities_panel(config)
        return pd.DataFrame()

    def _normalise(self, df: pd.DataFrame, fam: Family, config: RunConfig, **kw: Any) -> pd.DataFrame:
        return normalise_family_panel(
            df, family=fam, run_id=config.run_id,
            timing=config.families[fam].timing, **kw,
        )

    # -- price-like families (equity / index) ------------------------------ #
    def _build_price_panel(self, fam: Family, config: RunConfig, symbols: List[str]) -> pd.DataFrame:
        subdir, suffix, native_freq = _PRICE_SOURCES[fam]
        directory = self.raw_data_dir / subdir

        if fam == Family.INDEX:
            symbols = sorted({s for syms in config.benchmark_map.values() for s in syms})

        scope = config.families[fam].field_scope or _PRICE_FIELDS
        keep = [c for c in _PRICE_FIELDS if c in scope]

        frames: List[pd.DataFrame] = []
        for sym in symbols:
            path = directory / f"{sym}{suffix}"
            if not path.exists():
                continue
            try:
                raw = pd.read_csv(path)
            except Exception:
                continue
            if "date" not in raw.columns:
                continue
            cols = ["date"] + [c for c in keep if c in raw.columns]
            sub = raw[cols].copy()
            sub = sub.rename(columns={"date": "observation_date"})
            sub["entity_id"] = sym
            frames.append(sub)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        return self._normalise(
            combined, fam, config,
            value_col="close" if "close" in combined.columns else None,
            native_frequency=native_freq,
            source_function="local_csv", source_vendor="local",
        )

    # -- FX ----------------------------------------------------------------- #
    def _build_fx_panel(self, config: RunConfig, currencies: List[str]) -> pd.DataFrame:
        directory = self.raw_data_dir / "forex"
        if not directory.exists():
            return pd.DataFrame()

        # Prefer pairs explicitly configured; else any local forex file.
        configured = config.families[Family.FX].series
        files = []
        if configured:
            for pair in configured:
                files.extend(directory.glob(f"{pair}*.csv"))
        else:
            files = sorted(directory.glob("*.csv"))

        frames: List[pd.DataFrame] = []
        for path in files:
            try:
                raw = pd.read_csv(path)
            except Exception:
                continue
            if "date" not in raw.columns or "close" not in raw.columns:
                continue
            pair = raw["symbol"].iloc[0] if "symbol" in raw.columns and len(raw) else path.stem.split("_")[0]
            native_freq = "weekly" if "weekly" in path.stem else ("daily" if "daily" in path.stem else "weekly")
            sub = raw[["date", "close"] + [c for c in ("open", "high", "low") if c in raw.columns]].copy()
            sub = sub.rename(columns={"date": "observation_date", "close": "rate"})
            sub["entity_id"] = pair
            sub["native_frequency"] = native_freq
            sub["role"] = "conversion"  # conversion vs regime kept distinct logically
            frames.append(sub)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        return self._normalise(
            combined, Family.FX, config,
            value_col="rate",
            source_function="FX_LOCAL", source_vendor="local",
        )

    # -- fundamentals ------------------------------------------------------- #
    def _fundamentals_dir(self):
        """Locate the fundamentals store: raw_data/fundamentals is where the
        collector writes; processed_data/fundamentals is a legacy fallback."""
        for d in (self.raw_data_dir / "fundamentals", self.processed_dir / "fundamentals"):
            if d.exists() and any(d.glob("*_fundamental.csv")):
                return d
        return None

    def _build_fundamentals_panel(self, config: RunConfig, symbols: List[str]) -> pd.DataFrame:
        directory = self._fundamentals_dir()
        if directory is None:
            return pd.DataFrame()

        # Case-insensitive symbol -> file lookup (trade symbols may differ in case).
        available = {p.stem.lower(): p for p in directory.glob("*_fundamental.csv")}

        frames: List[pd.DataFrame] = []
        for sym in symbols:
            path = available.get(f"{sym.lower()}_fundamental")
            if path is None:
                continue
            try:
                raw = pd.read_csv(path)
            except Exception:
                continue
            raw.columns = [str(c).lower().strip() for c in raw.columns]
            if "fiscaldateending" not in raw.columns:
                continue
            raw = raw.rename(columns={
                "fiscaldateending": "observation_date",
                "symbol": "entity_id",
            })
            if "entity_id" not in raw.columns:
                raw["entity_id"] = sym
            frames.append(raw)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        report_col = "report_date" if "report_date" in combined.columns else None
        value_col = "reported_eps" if "reported_eps" in combined.columns else None
        return self._normalise(
            combined, Family.FUNDAMENTALS, config,
            report_date_col=report_col, value_col=value_col,
            native_frequency="quarterly",
            source_function="EARNINGS", source_vendor="alpha_vantage",
        )

    # -- commodities (local store first, then Alpha Vantage) --------------- #
    def _build_commodities_panel(self, config: RunConfig) -> pd.DataFrame:
        cfg = config.families[Family.COMMODITIES]
        combined = self._read_local_series("commodities", "series_id", cfg.series)
        if combined.empty and self.av_client is not None:
            combined = self._fetch_commodities_live(cfg)
        if combined.empty:
            return pd.DataFrame()
        combined = combined.rename(columns={"native_function": "source_function"})
        return self._normalise(
            combined, Family.COMMODITIES, config,
            entity_id_col="series_id", value_col="value",
        )

    def _fetch_commodities_live(self, cfg) -> pd.DataFrame:
        from Classes.DataCollection.commodity_collector import CommodityCollector
        results = CommodityCollector(self.av_client).collect_many(
            series_keys=cfg.series or None,
            intervals=cfg.options.get("intervals") or {},
        )
        frames = [r.df for r in results.values() if not r.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # -- macro (local store first, then Alpha Vantage) --------------------- #
    def _build_macro_panel(self, config: RunConfig) -> pd.DataFrame:
        cfg = config.families[Family.MACRO]
        # Stored macro carries native_function (the AV function) for filtering;
        # cfg.series holds those function names.
        combined = self._read_local_series("macro", "native_function", cfg.series)
        if combined.empty and self.av_client is not None:
            combined = self._fetch_macro_live(cfg)
        if combined.empty:
            return pd.DataFrame()
        combined = combined.rename(columns={"native_function": "source_function"})
        return self._normalise(
            combined, Family.MACRO, config,
            entity_id_col="series_id", value_col="value",
        )

    def _fetch_macro_live(self, cfg) -> pd.DataFrame:
        from Classes.DataCollection.macro_collector import MacroCollector
        results = MacroCollector(self.av_client).collect_many(
            series_keys=cfg.series or None,
            intervals=cfg.options.get("intervals") or {},
            treasury_maturities=cfg.options.get("treasury_maturities") or ["10year"],
        )
        frames = [r.df for r in results.values() if not r.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _read_local_series(self, subdir: str, filter_col: str, values: List[str]) -> pd.DataFrame:
        """Read + concat normalised series CSVs from raw_data/<subdir>/, filtered."""
        directory = self.raw_data_dir / subdir
        if not directory.exists():
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for path in sorted(directory.glob("*.csv")):
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if values and filter_col in df.columns:
            df = df[df[filter_col].isin(values)]
        return df.reset_index(drop=True)

    # -- corporate actions (local store first, then Alpha Vantage) --------- #
    def _build_corporate_actions_panel(self, config: RunConfig, symbols: List[str]) -> pd.DataFrame:
        cfg = config.families[Family.CORPORATE_ACTIONS]
        combined = self._read_local_corporate_actions(symbols)
        if combined.empty and self.av_client is not None:
            combined = self._fetch_corporate_actions_live(cfg, symbols)
        if combined.empty:
            return pd.DataFrame()
        return self._normalise(
            combined, Family.CORPORATE_ACTIONS, config,
            entity_id_col="symbol",
            native_frequency="event", source_vendor="alpha_vantage",
        )

    def _read_local_corporate_actions(self, symbols: List[str]) -> pd.DataFrame:
        directory = self.raw_data_dir / "corporate_actions"
        if not directory.exists():
            return pd.DataFrame()
        # Case-insensitive lookup of {symbol}_{kind}.csv files.
        available = {p.stem.lower(): p for p in directory.glob("*.csv")}
        specs = [("dividends", "ex_dividend_date", "amount", "DIVIDENDS"),
                 ("splits", "effective_date", "split_factor", "SPLITS")]
        rows: List[pd.DataFrame] = []
        for sym in symbols:
            for kind, date_col, val_col, func in specs:
                path = available.get(f"{sym.lower()}_{kind}")
                if path is None:
                    continue
                try:
                    raw = pd.read_csv(path)
                except Exception:
                    continue
                if date_col not in raw.columns:
                    continue
                raw = raw.rename(columns={date_col: "observation_date"})
                keep = ["symbol", "observation_date"] + ([val_col] if val_col in raw.columns else [])
                sub = raw[[c for c in keep if c in raw.columns]].copy()
                if "symbol" not in sub.columns:
                    sub["symbol"] = sym
                sub["event_type"] = "dividend" if kind == "dividends" else "split"
                sub["source_function"] = func
                rows.append(sub)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _fetch_corporate_actions_live(self, cfg, symbols: List[str]) -> pd.DataFrame:
        from Classes.DataCollection.corporate_actions_collector import CorporateActionsCollector
        results = CorporateActionsCollector(self.av_client).collect_many(
            symbols,
            include_dividends=cfg.options.get("include_dividends", True),
            include_splits=cfg.options.get("include_splits", True),
        )
        rows: List[pd.DataFrame] = []
        for res in results.values():
            if not res.dividends.empty:
                d = res.dividends.rename(columns={"ex_dividend_date": "observation_date"})
                d = d[["symbol", "observation_date", "amount"]].copy()
                d["event_type"] = "dividend"
                d["source_function"] = "DIVIDENDS"
                rows.append(d)
            if not res.splits.empty:
                s = res.splits.rename(columns={"effective_date": "observation_date"})
                s = s[["symbol", "observation_date", "split_factor"]].copy()
                s["event_type"] = "split"
                s["source_function"] = "SPLITS"
                rows.append(s)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # -- utilities (local store first, then Alpha Vantage market status) ---- #
    def _build_utilities_panel(self, config: RunConfig) -> pd.DataFrame:
        local = self.raw_data_dir / "utilities" / "market_status.csv"
        status = pd.DataFrame()
        if local.exists():
            try:
                status = pd.read_csv(local)
            except Exception:
                status = pd.DataFrame()
        if status.empty and self.av_client is not None:
            from Classes.DataCollection.utilities_collector import UtilitiesCollector
            status = UtilitiesCollector(self.av_client).collect_market_status()
        if status is None or status.empty:
            return pd.DataFrame()
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        status = status.copy()
        status["observation_date"] = today
        market_type = (status["market_type"].astype(str)
                       if "market_type" in status.columns else "market")
        region = (status["region"].astype(str)
                  if "region" in status.columns else "")
        status["entity_id"] = market_type + ":" + region
        return self._normalise(
            status, Family.UTILITIES, config,
            native_frequency="snapshot",
            source_function="MARKET_STATUS", source_vendor="alpha_vantage",
        )
