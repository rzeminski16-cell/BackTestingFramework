"""
Diagnostic script to verify FX/currency handling end-to-end.

Checks that:
- Securities resolve a currency from config/security_metadata.json.
- FX rates load from raw_data/forex (GBP-based pairs are inverted as needed).
- The engine applies a non-trivial FX rate for non-base-currency securities.
- Missing currency / missing FX rates raise instead of silently using 1.0.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from Classes.Data.security_registry import SecurityRegistry, MissingCurrencyError
from Classes.Data.currency_converter import CurrencyConverter, MissingFXRateError
from Classes.Config.config import BacktestConfig
from Classes.Engine.single_security_engine import SingleSecurityEngine

FOREX_DIR = project_root / 'raw_data' / 'forex'
METADATA = project_root / 'config' / 'security_metadata.json'

print("=" * 60)
print("Testing Currency Detection")
print("=" * 60)

# Step 1: Load security registry
print("\n1. Loading security registry...")
security_registry = SecurityRegistry(METADATA)
all_symbols = security_registry.get_all_symbols()
print(f"   Loaded {len(all_symbols)} securities")

# Pick a registered non-base-currency security to demonstrate conversion.
sample_symbol = None
for sym in all_symbols:
    meta = security_registry.get_metadata(sym, warn_on_missing_currency=False)
    if meta and meta.currency and meta.currency != SecurityRegistry.BASE_CURRENCY:
        sample_symbol = sym
        break

if sample_symbol is None:
    print("   No non-base-currency security found in registry; using first symbol.")
    sample_symbol = all_symbols[0] if all_symbols else 'AAPL'

meta = security_registry.get_metadata(sample_symbol, warn_on_missing_currency=False)
print(f"   Sample symbol: {sample_symbol} (currency: {meta.currency if meta else 'N/A'})")

# Step 2: Load currency converter
print("\n2. Loading currency converter...")
currency_converter = CurrencyConverter(base_currency='GBP')
try:
    currency_converter.load_rates_directory(FOREX_DIR)
    print(f"   Loaded FX pairs: {list(currency_converter.rates.keys())}")
    print(f"   Convertible currencies: {currency_converter.get_available_currencies()}")
except Exception as e:
    print(f"   Failed to load FX rates: {e}")

# Step 3: FX rate lookup (should be non-1.0 for a non-base currency)
print("\n3. Testing FX rate lookup...")
test_date = datetime(2025, 1, 6)
if meta and meta.currency != 'GBP':
    rate = currency_converter.get_rate(meta.currency, 'GBP', test_date)
    print(f"   {meta.currency}->GBP on {test_date.date()}: {rate}")
    if rate is None:
        print("   WARNING: no rate found (extend your forex data range)")
    elif abs(rate - 1.0) < 1e-9:
        print("   WARNING: rate is 1.0 - conversion would be a no-op")
    else:
        print("   OK: a real FX rate is being applied")

# Step 4: Engine resolves the same rate
print("\n4. Engine FX rate via SingleSecurityEngine...")
config = BacktestConfig(initial_capital=100000.0, base_currency='GBP')
engine = SingleSecurityEngine(
    config=config,
    currency_converter=currency_converter,
    security_registry=security_registry
)
try:
    fx_rate = engine._get_fx_rate(sample_symbol, test_date)
    print(f"   Engine FX rate for {sample_symbol}: {fx_rate}")
except (MissingCurrencyError, MissingFXRateError) as e:
    print(f"   Raised (as designed): {type(e).__name__}: {str(e).splitlines()[0]}")

# Step 5: Strict errors for missing data
print("\n5. Strict error handling...")
try:
    engine._get_fx_rate('NOT_A_REAL_TICKER_XYZ', test_date)
    print("   WARNING: expected MissingCurrencyError for unknown ticker")
except MissingCurrencyError:
    print("   OK: unknown ticker raises MissingCurrencyError")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
