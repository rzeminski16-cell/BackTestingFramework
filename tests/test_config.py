"""
Comprehensive tests for configuration classes:
- CommissionConfig
- BacktestConfig
- PortfolioConfig
- OptimizationConfig
- CapitalContentionConfig / VulnerabilityScoreConfig
"""
import pytest
from datetime import datetime

from Classes.Config.config import (
    CommissionConfig, CommissionMode,
    BacktestConfig, PortfolioConfig, OptimizationConfig
)
from Classes.Config.capital_contention import (
    CapitalContentionConfig, CapitalContentionMode,
    VulnerabilityScoreConfig, EnhancedVulnerabilityConfig,
    FeatureWeightConfig
)


# =============================================================================
# CommissionConfig Tests
# =============================================================================

class TestCommissionConfig:
    def test_default_commission(self):
        config = CommissionConfig()
        assert config.mode == CommissionMode.PERCENTAGE
        assert config.value == 0.001

    def test_percentage_commission(self):
        config = CommissionConfig(mode=CommissionMode.PERCENTAGE, value=0.002)
        commission = config.calculate(10000.0)
        assert commission == 20.0

    def test_fixed_commission(self):
        config = CommissionConfig(mode=CommissionMode.FIXED, value=5.0)
        commission = config.calculate(10000.0)
        assert commission == 5.0

    def test_fixed_commission_independent_of_value(self):
        config = CommissionConfig(mode=CommissionMode.FIXED, value=5.0)
        assert config.calculate(1000.0) == config.calculate(100000.0)

    def test_zero_commission(self):
        config = CommissionConfig(value=0.0)
        assert config.calculate(10000.0) == 0.0

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            CommissionConfig(value=-0.001)

    def test_percentage_over_100_rejected(self):
        with pytest.raises(ValueError, match="<= 1.0"):
            CommissionConfig(mode=CommissionMode.PERCENTAGE, value=1.5)

    def test_percentage_at_boundary(self):
        config = CommissionConfig(mode=CommissionMode.PERCENTAGE, value=1.0)
        assert config.calculate(10000.0) == 10000.0


# =============================================================================
# BacktestConfig Tests
# =============================================================================

class TestBacktestConfig:
    def test_default_config(self):
        config = BacktestConfig()
        assert config.initial_capital == 100000.0
        assert config.position_size_limit == 1.0
        assert config.base_currency == "GBP"
        assert config.slippage_percent == 0.1

    def test_custom_capital(self):
        config = BacktestConfig(initial_capital=50000.0)
        assert config.initial_capital == 50000.0

    def test_zero_capital_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            BacktestConfig(initial_capital=0.0)

    def test_negative_capital_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            BacktestConfig(initial_capital=-1000.0)

    def test_position_size_limit_zero_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1.0"):
            BacktestConfig(position_size_limit=0.0)

    def test_position_size_limit_over_1_rejected(self):
        with pytest.raises(ValueError, match="between 0 and 1.0"):
            BacktestConfig(position_size_limit=1.5)

    def test_date_range_validation(self):
        with pytest.raises(ValueError, match="before end date"):
            BacktestConfig(
                start_date=datetime(2024, 12, 1),
                end_date=datetime(2024, 1, 1)
            )

    def test_valid_date_range(self):
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1)
        )
        assert config.start_date == datetime(2024, 1, 1)
        assert config.end_date == datetime(2024, 12, 1)

    def test_negative_slippage_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            BacktestConfig(slippage_percent=-0.1)

    def test_zero_slippage_allowed(self):
        config = BacktestConfig(slippage_percent=0.0)
        assert config.slippage_percent == 0.0


# =============================================================================
# PortfolioConfig Tests
# =============================================================================

class TestPortfolioConfig:
    def test_default_config(self):
        config = PortfolioConfig()
        assert config.initial_capital == 100000.0
        assert config.base_currency == "GBP"

    def test_zero_capital_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PortfolioConfig(initial_capital=0.0)

    def test_invalid_date_range(self):
        with pytest.raises(ValueError, match="before end date"):
            PortfolioConfig(
                start_date=datetime(2025, 1, 1),
                end_date=datetime(2024, 1, 1)
            )

    def test_negative_slippage_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            PortfolioConfig(slippage_percent=-1.0)

    def test_to_dict(self):
        config = PortfolioConfig(initial_capital=50000.0, basket_name="test")
        d = config.to_dict()
        assert d['initial_capital'] == 50000.0
        assert d['basket_name'] == "test"
        assert 'commission' in d
        assert 'capital_contention' in d


# =============================================================================
# OptimizationConfig Tests
# =============================================================================

class TestOptimizationConfig:
    def test_default_config(self):
        config = OptimizationConfig()
        assert config.metric == 'total_return'
        assert config.maximize is True

    def test_valid_metric(self):
        config = OptimizationConfig(metric='sharpe_ratio')
        assert config.metric == 'sharpe_ratio'

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValueError, match="Metric must be one of"):
            OptimizationConfig(metric='invalid_metric')

    def test_negative_min_trades_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            OptimizationConfig(min_trades=-1)

    def test_all_valid_metrics(self):
        valid = ['total_return', 'total_return_pct', 'sharpe_ratio', 'profit_factor',
                 'win_rate', 'max_drawdown', 'avg_win', 'avg_loss', 'num_trades']
        for metric in valid:
            config = OptimizationConfig(metric=metric)
            assert config.metric == metric


# =============================================================================
# VulnerabilityScoreConfig Tests
# =============================================================================

class TestVulnerabilityScoreConfig:
    def test_default_config(self):
        config = VulnerabilityScoreConfig()
        assert config.immunity_days == 7
        assert config.swap_threshold == 50.0
        assert config.base_score == 100.0

    def test_negative_immunity_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            VulnerabilityScoreConfig(immunity_days=-1)

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="between 0 and"):
            VulnerabilityScoreConfig(swap_threshold=-1.0)

    def test_threshold_above_base_rejected(self):
        with pytest.raises(ValueError, match="between 0 and"):
            VulnerabilityScoreConfig(swap_threshold=150.0)

    def test_negative_base_score_rejected(self):
        with pytest.raises(ValueError):
            VulnerabilityScoreConfig(base_score=0.0)

    def test_to_dict_and_from_dict_roundtrip(self):
        original = VulnerabilityScoreConfig(
            immunity_days=10,
            min_profit_threshold=0.05,
            decay_rate_fast=8.0,
            decay_rate_slow=2.0,
            swap_threshold=60.0
        )
        d = original.to_dict()
        restored = VulnerabilityScoreConfig.from_dict(d)
        assert restored.immunity_days == original.immunity_days
        assert restored.min_profit_threshold == original.min_profit_threshold
        assert restored.decay_rate_fast == original.decay_rate_fast
        assert restored.swap_threshold == original.swap_threshold


# =============================================================================
# CapitalContentionConfig Tests
# =============================================================================

class TestCapitalContentionConfig:
    def test_default_mode(self):
        config = CapitalContentionConfig.default_mode()
        assert config.mode == CapitalContentionMode.DEFAULT

    def test_vulnerability_score_mode(self):
        config = CapitalContentionConfig.vulnerability_score_mode(immunity_days=10)
        assert config.mode == CapitalContentionMode.VULNERABILITY_SCORE
        assert config.vulnerability_config.immunity_days == 10

    def test_to_dict_and_from_dict_roundtrip(self):
        original = CapitalContentionConfig.vulnerability_score_mode(
            swap_threshold=65.0
        )
        d = original.to_dict()
        restored = CapitalContentionConfig.from_dict(d)
        assert restored.mode == CapitalContentionMode.VULNERABILITY_SCORE
        assert restored.vulnerability_config.swap_threshold == 65.0


# =============================================================================
# EnhancedVulnerabilityConfig Tests
# =============================================================================

class TestEnhancedVulnerabilityConfig:
    def test_default_features_populated(self):
        config = EnhancedVulnerabilityConfig()
        assert 'days_held' in config.features
        assert 'current_pl_pct' in config.features

    def test_conservative_preset(self):
        config = EnhancedVulnerabilityConfig.conservative_preset()
        assert config.name == "Conservative"
        assert config.immunity_days == 14
        assert config.swap_threshold == 30.0

    def test_aggressive_preset(self):
        config = EnhancedVulnerabilityConfig.aggressive_preset()
        assert config.name == "Aggressive"
        assert config.immunity_days == 3
        assert config.swap_threshold == 70.0

    def test_momentum_focused_preset(self):
        config = EnhancedVulnerabilityConfig.momentum_focused_preset()
        assert config.name == "Momentum Focused"

    def test_invalid_immunity_days(self):
        with pytest.raises(ValueError, match="non-negative"):
            EnhancedVulnerabilityConfig(immunity_days=-1)

    def test_invalid_base_score(self):
        with pytest.raises(ValueError, match="positive"):
            EnhancedVulnerabilityConfig(base_score=0.0)

    def test_to_dict_and_from_dict_roundtrip(self):
        original = EnhancedVulnerabilityConfig.aggressive_preset()
        d = original.to_dict()
        restored = EnhancedVulnerabilityConfig.from_dict(d)
        assert restored.name == original.name
        assert restored.immunity_days == original.immunity_days
        assert len(restored.features) == len(original.features)

    def test_to_simple_config(self):
        config = EnhancedVulnerabilityConfig()
        simple = config.to_simple_config()
        assert isinstance(simple, VulnerabilityScoreConfig)
        assert simple.immunity_days == config.immunity_days


class TestFeatureWeightConfig:
    def test_default(self):
        fw = FeatureWeightConfig()
        assert fw.enabled is True
        assert fw.weight == 1.0

    def test_to_dict_and_from_dict(self):
        original = FeatureWeightConfig(enabled=False, weight=3.5, decay_point=21)
        d = original.to_dict()
        restored = FeatureWeightConfig.from_dict(d)
        assert restored.enabled == False
        assert restored.weight == 3.5
        assert restored.decay_point == 21
