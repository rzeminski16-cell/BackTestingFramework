#!/usr/bin/env python3
"""
Sync security metadata from tickers.json.

This script ensures all tickers in config/data_collection/tickers.json
are present in config/security_metadata.json with appropriate metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, Set, Tuple

# UK/GBP tickers - common UK listings
UK_TICKERS = {
    'LLOY', 'BARC', 'HSBA', 'VOD', 'BT.A', 'BP', 'SHEL', 'RIO', 'GSK', 'AZN',
    'ULVR', 'DGE', 'RR.', 'BA.', 'STAN', 'NWG', 'SSE', 'NG.', 'GLEN', 'AAL',
    'IMB', 'BATS', 'LSEG', 'EXPN', 'REL', 'CRH', 'III', 'RKT',  # Some may be dual-listed
}

# EUR tickers - common European listings
EUR_TICKERS = {
    'SCMN',  # Swisscom (CHF actually, but grouped with EUR for simplicity)
}


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    """Save JSON file with pretty formatting."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def extract_tickers_with_info(tickers_data: Dict) -> Dict[str, Dict[str, Any]]:
    """
    Extract all tickers from tickers.json with sector and market_cap info.

    Returns dict mapping ticker to {sector, market_cap, type}
    """
    result = {}
    categories = tickers_data.get('categories', {})

    # Process stocks
    stocks = categories.get('stocks', {})
    for sector_name, sector_data in stocks.items():
        if isinstance(sector_data, dict):
            for cap_category in ['large_cap', 'mid_cap', 'small_cap']:
                tickers = sector_data.get(cap_category, [])
                for ticker in tickers:
                    if ticker not in result:
                        result[ticker] = {
                            'sector': sector_name,
                            'market_cap': cap_category,
                            'type': 'stock'
                        }

    # Process ETFs
    etfs = categories.get('etfs', {})
    for etf_category, etf_data in etfs.items():
        if isinstance(etf_data, dict):
            for sub_category, tickers in etf_data.items():
                if isinstance(tickers, list):
                    for ticker in tickers:
                        if ticker not in result:
                            result[ticker] = {
                                'sector': f'etf_{etf_category}',
                                'market_cap': sub_category,  # passive/active
                                'type': 'etf'
                            }

    # Process crypto
    crypto = categories.get('crypto', [])
    for ticker in crypto:
        if ticker not in result:
            result[ticker] = {
                'sector': 'crypto',
                'market_cap': None,
                'type': 'crypto'
            }

    return result


def determine_currency(ticker: str) -> str:
    """Determine currency based on ticker."""
    # UK stocks
    if ticker in UK_TICKERS:
        return 'GBP'

    # UK suffixes
    if ticker.endswith('.L') or ticker.endswith('.A'):
        return 'GBP'

    # EUR tickers
    if ticker in EUR_TICKERS:
        return 'EUR'

    # Crypto
    if ticker.endswith('USD'):
        return 'USD'

    # Default to USD for US stocks
    return 'USD'


def sync_metadata(tickers_path: Path, metadata_path: Path) -> Tuple[int, int, int]:
    """
    Sync tickers from tickers.json to security_metadata.json.

    Returns (added_count, updated_count, total_count)
    """
    # Load files
    tickers_data = load_json(tickers_path)

    if metadata_path.exists():
        existing_metadata = load_json(metadata_path)
    else:
        existing_metadata = {}

    # Extract ticker info
    ticker_info = extract_tickers_with_info(tickers_data)

    added = 0
    updated = 0

    # Update metadata
    for ticker, info in ticker_info.items():
        if ticker not in existing_metadata:
            # New ticker - add with all info
            existing_metadata[ticker] = {
                'type': info['type'],
                'sector': info['sector'],
                'market_cap': info['market_cap'],
                'currency': determine_currency(ticker),
                'description': ''
            }
            added += 1
        else:
            # Existing ticker - update sector and market_cap if missing
            entry = existing_metadata[ticker]
            changed = False

            if 'sector' not in entry or not entry['sector']:
                entry['sector'] = info['sector']
                changed = True

            if 'market_cap' not in entry:
                entry['market_cap'] = info['market_cap']
                changed = True

            if 'currency' not in entry or not entry['currency']:
                entry['currency'] = determine_currency(ticker)
                changed = True

            if changed:
                updated += 1

    # Save updated metadata
    save_json(metadata_path, existing_metadata)

    return added, updated, len(existing_metadata)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    tickers_path = project_root / 'config' / 'data_collection' / 'tickers.json'
    metadata_path = project_root / 'config' / 'security_metadata.json'

    print(f"Syncing tickers from {tickers_path}")
    print(f"Updating metadata at {metadata_path}")
    print()

    if not tickers_path.exists():
        print(f"ERROR: Tickers file not found: {tickers_path}")
        return 1

    added, updated, total = sync_metadata(tickers_path, metadata_path)

    print(f"Results:")
    print(f"  - Added: {added} new tickers")
    print(f"  - Updated: {updated} existing tickers")
    print(f"  - Total: {total} tickers in metadata")

    return 0


if __name__ == '__main__':
    exit(main())
