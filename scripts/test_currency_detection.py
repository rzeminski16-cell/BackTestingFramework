"""
Test script to verify currency detection is working correctly.
"""
from pathlib import Path
from Classes.Data.security_registry import SecurityRegistry
from Classes.Data.currency_converter import CurrencyConverter
from Classes.Config.config import BacktestConfig
from Classes.Engine.single_security_engine import SingleSecurityEngine

print("="*60)
print("Testing Currency Detection")
print("="*60)

# Step 1: Load security registry
print("\n1. Loading security registry...")
security_registry = SecurityRegistry(Path('config/security_metadata.json'))

# Step 2: Check what currency is registered for AAPL
print("\n2. Checking AAPL currency in registry...")
aapl_metadata = security_registry.get_metadata('AAPL')
if aapl_metadata:
    print(f"   ✓ AAPL found in registry")
    print(f"   Currency: {aapl_metadata.currency}")
    print(f"   Type: {aapl_metadata.type}")
    print(f"   Sector: {aapl_metadata.sector}")
else:
    print("   ✗ AAPL NOT found in registry!")

# Step 3: Load currency converter
print("\n3. Loading currency converter...")
currency_converter = CurrencyConverter(base_currency='GBP')
try:
    currency_converter.load_rates_directory(Path('currency_rates/'))
    print(f"   ✓ Loaded FX rates")
    print(f"   Available currencies: {currency_converter.get_available_currencies()}")
except Exception as e:
    print(f"   ✗ Failed to load FX rates: {e}")

# Step 4: Create engine WITH security_registry
print("\n4. Creating engine WITH security_registry...")
config = BacktestConfig(initial_capital=100000.0, base_currency='GBP')
engine_with_registry = SingleSecurityEngine(
    config=config,
    currency_converter=currency_converter,
    security_registry=security_registry
)
print(f"   ✓ Engine created")
print(f"   Engine has security_registry: {engine_with_registry.security_registry is not None}")
print(f"   Engine has currency_converter: {engine_with_registry.currency_converter is not None}")

# Step 5: Test FX rate lookup
print("\n5. Testing FX rate lookup for AAPL...")
from datetime import datetime
test_date = datetime(2020, 1, 2)
fx_rate = engine_with_registry._get_fx_rate('AAPL', test_date)
print(f"   FX rate for AAPL on {test_date.date()}: {fx_rate}")

# Step 6: Create engine WITHOUT security_registry (wrong way)
print("\n6. Creating engine WITHOUT security_registry (WRONG)...")
engine_without_registry = SingleSecurityEngine(config=config)
print(f"   ✓ Engine created")
print(f"   Engine has security_registry: {engine_without_registry.security_registry is not None}")
fx_rate_wrong = engine_without_registry._get_fx_rate('AAPL', test_date)
print(f"   FX rate for AAPL: {fx_rate_wrong} (defaults to 1.0 = no conversion)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Registry loaded: {aapl_metadata is not None}")
print(f"AAPL currency in config: {aapl_metadata.currency if aapl_metadata else 'N/A'}")
print(f"With registry - FX rate: {fx_rate}")
print(f"Without registry - FX rate: {fx_rate_wrong}")
print("\nIf you're seeing GBP, make sure you're passing security_registry!")
print("="*60)
