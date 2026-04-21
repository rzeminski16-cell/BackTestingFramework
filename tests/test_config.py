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
    VulnerabilityScoreConfig,
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
        assert config.min_trade_age_days == 100
        assert config.target_monthly_growth == 0.05
        assert config.alpha == 1.0
        assert config.beta == 1.0
        assert config.avg_window_days == 7
        assert config.pullback_window_days == 14

    def test_negative_min_trade_age_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            VulnerabilityScoreConfig(min_trade_age_days=-1)

    def test_monthly_growth_below_minus_one_rejected(self):
        with pytest.raises(ValueError, match="greater than -1"):
            VulnerabilityScoreConfig(target_monthly_growth=-1.5)

    def test_negative_alpha_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            VulnerabilityScoreConfig(alpha=-0.1)

    def test_negative_beta_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            VulnerabilityScoreConfig(beta=-0.1)

    def test_zero_avg_window_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            VulnerabilityScoreConfig(avg_window_days=0)

    def test_zero_pullback_window_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            VulnerabilityScoreConfig(pullback_window_days=0)

    def test_to_dict_and_from_dict_roundtrip(self):
        original = VulnerabilityScoreConfig(
            min_trade_age_days=60,
            target_monthly_growth=0.03,
            alpha=1.5,
            beta=0.8,
            avg_window_days=5,
            pullback_window_days=21,
        )
        d = original.to_dict()
        restored = VulnerabilityScoreConfig.from_dict(d)
        assert restored.min_trade_age_days == original.min_trade_age_days
        assert restored.target_monthly_growth == original.target_monthly_growth
        assert restored.alpha == original.alpha
        assert restored.beta == original.beta
        assert restored.avg_window_days == original.avg_window_days
        assert restored.pullback_window_days == original.pullback_window_days

    def test_from_dict_ignores_unknown_keys(self):
        data = {
            'min_trade_age_days': 80,
            'target_monthly_growth': 0.04,
            'legacy_field': 'ignored',
        }
        config = VulnerabilityScoreConfig.from_dict(data)
        assert config.min_trade_age_days == 80
        assert config.target_monthly_growth == 0.04


# =============================================================================
# CapitalContentionConfig Tests
# =============================================================================

class TestCapitalContentionConfig:
    def test_default_mode(self):
        config = CapitalContentionConfig.default_mode()
        assert config.mode == CapitalContentionMode.DEFAULT

    def test_vulnerability_score_mode(self):
        config = CapitalContentionConfig.vulnerability_score_mode(min_trade_age_days=60)
        assert config.mode == CapitalContentionMode.VULNERABILITY_SCORE
        assert config.vulnerability_config.min_trade_age_days == 60

    def test_to_dict_and_from_dict_roundtrip(self):
        original = CapitalContentionConfig.vulnerability_score_mode(
            target_monthly_growth=0.03, alpha=2.0, beta=0.5,
        )
        d = original.to_dict()
        restored = CapitalContentionConfig.from_dict(d)
        assert restored.mode == CapitalContentionMode.VULNERABILITY_SCORE
        assert restored.vulnerability_config.target_monthly_growth == 0.03
        assert restored.vulnerability_config.alpha == 2.0
        assert restored.vulnerability_config.beta == 0.5
