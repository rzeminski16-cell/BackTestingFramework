"""
Tests for multi-currency FX handling in the backtesting engines.

Covers the fixes that ensure FX actually plays a part in backtesting:
- CurrencyConverter resolves rates from either direction (e.g. GBP/USD data can
  convert USD -> GBP by inverting the rate).
- SecurityRegistry.require_currency raises when currency is unknown.
- Engines apply FX to position values/trades, flag non-base tickers, and raise
  (instead of silently using 1.0) when currency or FX rates are missing.
"""
import unittest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd

from Classes.Config.config import BacktestConfig, PortfolioConfig, CommissionConfig, CommissionMode
from Classes.Data.currency_converter import CurrencyConverter, MissingFXRateError
from Classes.Data.security_registry import SecurityRegistry, MissingCurrencyError
from Classes.Engine.single_security_engine import SingleSecurityEngine
from Classes.Engine.portfolio_engine import PortfolioEngine
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal
from Classes.Models.trade_direction import TradeDirection


# =============================================================================
# Helpers
# =============================================================================

class FixedSharesStrategy(BaseStrategy):
    """Buys a fixed number of shares on one bar, sells on another."""

    _validate_on_init = False

    def __init__(self, buy_bar: int = 1, sell_bar: int = 4, shares: float = 100.0):
        self.buy_bar = buy_bar
        self.sell_bar = sell_bar
        self.shares = shares
        super().__init__()

    @property
    def trade_direction(self) -> TradeDirection:
        return TradeDirection.LONG

    def required_columns(self) -> List[str]:
        return ['date', 'close']

    def generate_entry_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index == self.buy_bar:
            return Signal.buy(size=1.0, reason="entry", direction=TradeDirection.LONG)
        return None

    def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
        return context.current_price * 0.5

    def generate_exit_signal(self, context: StrategyContext) -> Optional[Signal]:
        if context.current_index == self.sell_bar:
            return Signal.sell(reason="exit")
        return None

    def position_size(self, context: StrategyContext, signal: Signal) -> float:
        return self.shares


def _make_forex_dir(pairs: dict) -> tempfile.TemporaryDirectory:
    """
    Create a temp directory of forex CSVs.

    Args:
        pairs: mapping of filename stem (e.g. 'GBPUSD_weekly') to (dates, closes)
    """
    tmp = tempfile.TemporaryDirectory()
    for stem, (dates, closes) in pairs.items():
        df = pd.DataFrame({
            'date': dates,
            'symbol': stem.split('_')[0],
            'open': closes,
            'high': closes,
            'low': closes,
            'close': closes,
        })
        df.to_csv(Path(tmp.name) / f"{stem}.csv", index=False)
    return tmp


def _flat_data(n=8, price=100.0, start='2025-01-06'):
    dates = pd.date_range(start, periods=n, freq='W-MON')
    return pd.DataFrame({
        'date': dates,
        'open': [price] * n,
        'high': [price] * n,
        'low': [price] * n,
        'close': [price] * n,
        'volume': [1000] * n,
    })


def _make_config(**kwargs):
    return BacktestConfig(
        initial_capital=100000.0,
        base_currency='GBP',
        commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
        slippage_percent=0.0,
        **kwargs
    )


# =============================================================================
# CurrencyConverter
# =============================================================================

class TestCurrencyConverterInversion(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='W-MON')
        # GBP-based quotes: 1 GBP = 1.25 USD, 1 GBP = 1.15 EUR
        self.forex = _make_forex_dir({
            'GBPUSD_weekly': (dates, [1.25] * 5),
            'GBPEUR_weekly': (dates, [1.15] * 5),
        })
        self.cc = CurrencyConverter(base_currency='GBP')
        self.cc.load_rates_directory(Path(self.forex.name))
        self.date = dates[-1]

    def tearDown(self):
        self.forex.cleanup()

    def test_pairs_loaded_as_gbp_based(self):
        self.assertIn('GBP/USD', self.cc.rates)
        self.assertIn('GBP/EUR', self.cc.rates)

    def test_same_currency_is_one(self):
        self.assertEqual(self.cc.get_rate('GBP', 'GBP', self.date), 1.0)

    def test_inverse_lookup_usd_to_gbp(self):
        # USD -> GBP must invert the GBP/USD quote: 1 / 1.25 = 0.8
        self.assertAlmostEqual(self.cc.get_rate('USD', 'GBP', self.date), 0.8)

    def test_inverse_lookup_eur_to_gbp(self):
        self.assertAlmostEqual(self.cc.get_rate('EUR', 'GBP', self.date), 1.0 / 1.15)

    def test_direct_lookup_gbp_to_usd(self):
        self.assertAlmostEqual(self.cc.get_rate('GBP', 'USD', self.date), 1.25)

    def test_unknown_pair_returns_none(self):
        self.assertIsNone(self.cc.get_rate('JPY', 'GBP', self.date))

    def test_has_rate_both_directions(self):
        self.assertTrue(self.cc.has_rate('USD'))
        self.assertTrue(self.cc.has_rate('EUR'))
        self.assertTrue(self.cc.has_rate('GBP'))
        self.assertFalse(self.cc.has_rate('JPY'))

    def test_available_currencies_includes_both_sides(self):
        self.assertEqual(self.cc.get_available_currencies(), ['EUR', 'GBP', 'USD'])

    def test_date_before_series_returns_none(self):
        self.assertIsNone(self.cc.get_rate('USD', 'GBP', datetime(2000, 1, 1)))

    def test_convert_amount(self):
        # 1000 USD at 1/1.25 => 800 GBP
        self.assertAlmostEqual(self.cc.convert(1000.0, 'USD', 'GBP', self.date), 800.0)


class TestCurrencyConverterDirectPair(unittest.TestCase):
    """A directly quoted XXX/GBP series should be used as-is (no inversion)."""

    def test_direct_usd_gbp_used_as_is(self):
        dates = pd.date_range('2024-01-01', periods=3, freq='W-MON')
        forex = _make_forex_dir({'USDGBP_weekly': (dates, [0.79] * 3)})
        try:
            cc = CurrencyConverter(base_currency='GBP')
            cc.load_rates_directory(Path(forex.name))
            self.assertIn('USD/GBP', cc.rates)
            self.assertAlmostEqual(cc.get_rate('USD', 'GBP', dates[-1]), 0.79)
        finally:
            forex.cleanup()


# =============================================================================
# SecurityRegistry.require_currency
# =============================================================================

class TestRequireCurrency(unittest.TestCase):
    def setUp(self):
        self.reg = SecurityRegistry()
        self.reg.register('USDSTOCK', type='stock', currency='USD')
        self.reg.register('NOCCY', type='stock', currency='')
        self.reg.register('PLACEHOLDER', type='stock', currency='NA')

    def test_returns_currency(self):
        self.assertEqual(self.reg.require_currency('USDSTOCK'), 'USD')

    def test_unregistered_raises(self):
        with self.assertRaises(MissingCurrencyError):
            self.reg.require_currency('UNKNOWN')

    def test_empty_currency_raises(self):
        with self.assertRaises(MissingCurrencyError):
            self.reg.require_currency('NOCCY')

    def test_placeholder_currency_raises(self):
        # 'NA' (and similar) are manual-fill placeholders, not real currencies.
        with self.assertRaises(MissingCurrencyError):
            self.reg.require_currency('PLACEHOLDER')

    def test_placeholder_currency_flagged_as_missing(self):
        issues = self.reg.validate_securities(['USDSTOCK', 'NOCCY', 'PLACEHOLDER'])
        self.assertIn('NOCCY', issues['missing_currency'])
        self.assertIn('PLACEHOLDER', issues['missing_currency'])
        self.assertNotIn('USDSTOCK', issues['missing_currency'])


# =============================================================================
# Engine integration
# =============================================================================

class TestSingleEngineFX(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2024-12-01', periods=12, freq='W-MON')
        self.forex = _make_forex_dir({
            'GBPUSD_weekly': (dates, [1.25] * 12),  # USD -> GBP = 0.8
        })
        self.cc = CurrencyConverter(base_currency='GBP')
        self.cc.load_rates_directory(Path(self.forex.name))
        self.reg = SecurityRegistry()
        self.reg.register('USDSTOCK', type='stock', currency='USD')
        self.reg.register('GBPSTOCK', type='stock', currency='GBP')
        self.reg.register('JPYSTOCK', type='stock', currency='JPY')
        self.reg.register('NOCCY', type='stock', currency='')

    def tearDown(self):
        self.forex.cleanup()

    def test_fx_applied_to_trade_and_equity(self):
        engine = SingleSecurityEngine(_make_config(),
                                      currency_converter=self.cc,
                                      security_registry=self.reg)
        result = engine.run('USDSTOCK', _flat_data(price=100.0), FixedSharesStrategy(shares=100.0))
        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_fx_rate, 0.8)
        self.assertAlmostEqual(trade.exit_fx_rate, 0.8)
        self.assertEqual(trade.security_currency, 'USD')
        # 100 shares * 100 USD = 10,000 USD -> 8,000 GBP position value
        eq_after_entry = result.equity_curve.iloc[1]
        self.assertAlmostEqual(eq_after_entry['position_value'], 8000.0, places=2)

    def test_base_currency_no_conversion(self):
        engine = SingleSecurityEngine(_make_config(),
                                      currency_converter=self.cc,
                                      security_registry=self.reg)
        result = engine.run('GBPSTOCK', _flat_data(price=100.0), FixedSharesStrategy(shares=100.0))
        self.assertEqual(result.trades[0].entry_fx_rate, 1.0)

    def test_missing_currency_raises(self):
        engine = SingleSecurityEngine(_make_config(),
                                      currency_converter=self.cc,
                                      security_registry=self.reg)
        with self.assertRaises(MissingCurrencyError):
            engine.run('NOCCY', _flat_data(), FixedSharesStrategy())

    def test_unregistered_symbol_raises(self):
        engine = SingleSecurityEngine(_make_config(),
                                      currency_converter=self.cc,
                                      security_registry=self.reg)
        with self.assertRaises(MissingCurrencyError):
            engine.run('UNREGISTERED', _flat_data(), FixedSharesStrategy())

    def test_missing_fx_rate_raises(self):
        engine = SingleSecurityEngine(_make_config(),
                                      currency_converter=self.cc,
                                      security_registry=self.reg)
        with self.assertRaises(MissingFXRateError):
            engine.run('JPYSTOCK', _flat_data(), FixedSharesStrategy())

    def test_no_fx_wiring_defaults_to_one(self):
        # Backward compatible: without converter/registry, no conversion applied.
        engine = SingleSecurityEngine(_make_config())
        result = engine.run('ANYTHING', _flat_data(price=100.0), FixedSharesStrategy(shares=100.0))
        self.assertEqual(result.trades[0].entry_fx_rate, 1.0)


class TestPortfolioEngineFX(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2024-12-01', periods=12, freq='W-MON')
        self.forex = _make_forex_dir({
            'GBPUSD_weekly': (dates, [1.25] * 12),
        })
        self.cc = CurrencyConverter(base_currency='GBP')
        self.cc.load_rates_directory(Path(self.forex.name))
        self.reg = SecurityRegistry()
        self.reg.register('USDSTOCK', type='stock', currency='USD')
        self.reg.register('NOCCY', type='stock', currency='')
        self.reg.register('JPYSTOCK', type='stock', currency='JPY')

    def tearDown(self):
        self.forex.cleanup()

    def _portfolio_config(self):
        return PortfolioConfig(
            initial_capital=100000.0,
            base_currency='GBP',
            commission=CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.0),
            slippage_percent=0.0,
        )

    def test_fx_applied(self):
        engine = PortfolioEngine(self._portfolio_config(),
                                 currency_converter=self.cc,
                                 security_registry=self.reg)
        data_dict = {'USDSTOCK': _flat_data(price=100.0)}
        result = engine.run(data_dict, FixedSharesStrategy(shares=100.0))
        trades = result.symbol_results['USDSTOCK'].trades
        self.assertTrue(trades)
        self.assertAlmostEqual(trades[0].entry_fx_rate, 0.8)

    def test_missing_currency_raises(self):
        engine = PortfolioEngine(self._portfolio_config(),
                                 currency_converter=self.cc,
                                 security_registry=self.reg)
        with self.assertRaises(MissingCurrencyError):
            engine.run({'NOCCY': _flat_data()}, FixedSharesStrategy())

    def test_missing_fx_rate_raises(self):
        engine = PortfolioEngine(self._portfolio_config(),
                                 currency_converter=self.cc,
                                 security_registry=self.reg)
        with self.assertRaises(MissingFXRateError):
            engine.run({'JPYSTOCK': _flat_data()}, FixedSharesStrategy())


if __name__ == '__main__':
    unittest.main()
