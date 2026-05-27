#!/usr/bin/env python3
"""
Re-collect point-in-time fundamentals into the tidy panel format.

Headless equivalent of the Data Collection GUI's Fundamental tab: fetches the
Alpha Vantage statement/earnings/overview endpoints (via the cached, rate-limited
client), assembles the joined point-in-time panel with FundamentalCollector,
validates it, and writes raw_data/fundamentals/{SYMBOL}_fundamental.csv plus the
separate OVERVIEW snapshot.

Examples:
    python scripts/recollect_fundamentals.py --symbols ADBE COF CSCO
    python scripts/recollect_fundamentals.py --symbols AAPL --frequency quarterly

Use a large --cache-expiry-hours to force a cache-only (offline) run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Classes.DataCollection.config import APIConfig, CacheConfig, ValidationConfig
from Classes.DataCollection.alpha_vantage_client import AlphaVantageClient
from Classes.DataCollection.file_manager import FileManager
from Classes.DataCollection.validation_engine import ValidationEngine
from Classes.DataCollection.fundamental_collector import FundamentalCollector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', required=True, help='Ticker symbols to collect')
    parser.add_argument('--frequency', default='both', choices=['both', 'quarterly', 'annual'])
    parser.add_argument('--config', default='config/data_collection/settings.json')
    parser.add_argument('--output-dir', default='raw_data')
    parser.add_argument(
        '--cache-expiry-hours', type=int, default=None,
        help='Override cache expiry (use a large value for an offline, cache-only run)'
    )
    args = parser.parse_args()

    settings = json.loads(Path(args.config).read_text())
    api = APIConfig.from_dict(settings.get('api', {}))
    cache = CacheConfig.from_dict(settings.get('cache', {}))

    # settings.json may store a Windows-style cache path; normalize separators.
    cache.cache_dir = Path(str(cache.cache_dir).replace('\\', '/'))
    if args.cache_expiry_hours is not None:
        cache.cache_expiry_hours = args.cache_expiry_hours

    client = AlphaVantageClient(api, cache)
    file_manager = FileManager(Path(args.output_dir))
    validator = ValidationEngine(ValidationConfig.from_dict(settings.get('validation', {})))
    collector = FundamentalCollector(client)

    failures = 0
    for symbol in args.symbols:
        panel, snapshot = collector.collect(symbol, frequency=args.frequency)
        if panel.empty:
            print(f"{symbol}: no data")
            failures += 1
            continue

        report = validator.validate_fundamental_data(panel, symbol)
        meta = file_manager.write_fundamental_data(panel, symbol)
        file_manager.write_overview_snapshot(snapshot, symbol)

        errors = report.get_errors()
        warnings = report.get_warnings()
        status = 'OK' if report.passed else f"PARTIAL ({len(errors)} err, {len(warnings)} warn)"
        print(f"{symbol}: {status} - {meta.rows} rows, {meta.columns} cols -> {meta.file_path}")
        for result in errors + warnings:
            print(f"    [{result.severity.value}] {result.check_name}: {result.message}")
        if errors:
            failures += 1

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
