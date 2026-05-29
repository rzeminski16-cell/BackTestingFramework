#!/usr/bin/env python3
"""
Collect benchmark / index price data from Alpha Vantage (INDEX_DATA).

Fetches the indices in config/benchmarks.json (or those given with --symbols)
via the premium INDEX_DATA endpoint and writes raw_data/benchmarks/{SYMBOL}_{interval}.csv.
Those files are picked up automatically by reports through BenchmarkLoader.

Examples:
    python scripts/collect_benchmarks.py                 # all registry benchmarks
    python scripts/collect_benchmarks.py --symbols SPX DJI
    python scripts/collect_benchmarks.py --symbols SPX --interval weekly

Note: INDEX_DATA is a premium Alpha Vantage endpoint; a premium API key is
required. Network access to alphavantage.co is also required.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Classes.DataCollection.config import APIConfig, CacheConfig
from Classes.DataCollection.alpha_vantage_client import AlphaVantageClient
from Classes.DataCollection.file_manager import FileManager
from Classes.DataCollection.benchmark_collector import (
    BenchmarkCollector,
    load_benchmark_registry,
    resolve_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', help='Index symbols (default: all in the registry)')
    parser.add_argument('--interval', default=None, choices=['daily', 'weekly', 'monthly'],
                        help='Override the interval for all collected symbols')
    parser.add_argument('--outputsize', default='full', choices=['compact', 'full'])
    parser.add_argument('--config', default='config/data_collection/settings.json')
    parser.add_argument('--registry', default='config/benchmarks.json')
    parser.add_argument('--output-dir', default='raw_data')
    args = parser.parse_args()

    settings = json.loads(Path(args.config).read_text())
    api = APIConfig.from_dict(settings.get('api', {}))
    cache = CacheConfig.from_dict(settings.get('cache', {}))
    cache.cache_dir = Path(str(cache.cache_dir).replace('\\', '/'))

    client = AlphaVantageClient(api, cache)
    file_manager = FileManager(Path(args.output_dir))
    collector = BenchmarkCollector(client)
    registry = load_benchmark_registry(Path(args.registry))

    # Build the work list of (symbol, interval).
    jobs = []
    if args.symbols:
        for sym in args.symbols:
            resolved = resolve_benchmark(sym, registry)
            interval = args.interval or (resolved[1].get('interval', 'daily') if resolved else 'daily')
            symbol = resolved[1].get('symbol', sym) if resolved else sym
            jobs.append((symbol, interval))
    else:
        for _name, entry in registry.get('benchmarks', {}).items():
            jobs.append((entry['symbol'], args.interval or entry.get('interval', 'daily')))

    failures = 0
    for symbol, interval in jobs:
        result = collector.collect(symbol, interval=interval, outputsize=args.outputsize)
        if result.empty:
            print(f"{symbol} ({interval}): FAILED - {result.error}")
            failures += 1
            continue
        meta = file_manager.write_benchmark_data(result.df, symbol, interval=interval)
        span = f"{meta.date_range[0]} .. {meta.date_range[1]}" if meta.date_range else "?"
        print(f"{symbol} ({interval}) via {result.source} ({result.symbol_used}): "
              f"{meta.rows} rows [{span}] -> {meta.file_path}")

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
