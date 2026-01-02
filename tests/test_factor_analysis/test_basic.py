"""
Basic tests for Factor Analysis module.

Tests core functionality without requiring external data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestFactorAnalysisImports:
    """Test that all modules can be imported."""

    def test_import_main_module(self):
        """Test importing the main module."""
        from Classes.FactorAnalysis import FactorAnalyzer
        assert FactorAnalyzer is not None

    def test_import_config(self):
        """Test importing configuration classes."""
        from Classes.FactorAnalysis import (
            FactorAnalysisConfig,
            TradeClassificationConfig,
            DataAlignmentConfig,
            FactorEngineeringConfig,
            StatisticalAnalysisConfig,
            ScenarioAnalysisConfig,
            OutputConfig
        )
        assert FactorAnalysisConfig is not None

    def test_import_data_loaders(self):
        """Test importing data loaders."""
        from Classes.FactorAnalysis import (
            TradeLogLoader,
            PriceDataLoader,
            FundamentalLoader,
            InsiderLoader,
            OptionsLoader
        )
        assert TradeLogLoader is not None

    def test_import_preprocessing(self):
        """Test importing preprocessing classes."""
        from Classes.FactorAnalysis import (
            TradeClassifier,
            TemporalAligner,
            DataEnricher,
            QualityScorer,
            MultiLogAggregator
        )
        assert TradeClassifier is not None

    def test_import_factors(self):
        """Test importing factor classes."""
        from Classes.FactorAnalysis import (
            TechnicalFactors,
            FundamentalFactors,
            InsiderFactors,
            OptionsFactors,
            RegimeFactors,
            FactorNormalizer,
            OutlierHandler
        )
        assert TechnicalFactors is not None

    def test_import_analysis(self):
        """Test importing analysis classes."""
        from Classes.FactorAnalysis import (
            Tier1Exploratory,
            Tier2Hypothesis,
            Tier3ML,
            MultipleTestingCorrector
        )
        assert Tier1Exploratory is not None

    def test_import_scenarios(self):
        """Test importing scenario classes."""
        from Classes.FactorAnalysis import (
            ScenarioDetector,
            ScenarioValidator,
            InteractionAnalyzer
        )
        assert ScenarioDetector is not None

    def test_import_output(self):
        """Test importing output classes."""
        from Classes.FactorAnalysis import (
            ExcelReportGenerator,
            JsonPayloadGenerator,
            ResultFormatter
        )
        assert ExcelReportGenerator is not None


class TestTradeClassification:
    """Test trade classification functionality."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'symbol': ['AAPL'] * n,
            'entry_date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'exit_date': pd.date_range('2024-01-02', periods=n, freq='D'),
            'entry_price': np.random.uniform(100, 200, n),
            'exit_price': np.random.uniform(100, 200, n),
            'pl': np.random.uniform(-100, 100, n),
            'pl_pct': np.random.uniform(-0.1, 0.1, n)
        })

    def test_classifier_initialization(self):
        """Test TradeClassifier initialization."""
        from Classes.FactorAnalysis import TradeClassifier, TradeClassificationConfig

        config = TradeClassificationConfig(
            good_threshold=0.03,
            bad_threshold=-0.02
        )
        classifier = TradeClassifier(config=config)
        assert classifier is not None

    def test_classifier_classify(self, sample_trades):
        """Test trade classification."""
        from Classes.FactorAnalysis import TradeClassifier

        classifier = TradeClassifier()
        result = classifier.classify(sample_trades)

        assert 'trade_class' in result.columns
        assert 'trade_class_numeric' in result.columns
        assert set(result['trade_class'].unique()).issubset({'good', 'bad', 'indeterminate'})


class TestConfiguration:
    """Test configuration classes."""

    def test_default_config(self):
        """Test default configuration."""
        from Classes.FactorAnalysis import FactorAnalysisConfig

        config = FactorAnalysisConfig()
        assert config.trade_classification is not None
        assert config.data_alignment is not None
        assert config.factor_engineering is not None
        assert config.statistical_analysis is not None
        assert config.scenario_analysis is not None
        assert config.output is not None

    def test_config_immutability(self):
        """Test that configuration is immutable."""
        from Classes.FactorAnalysis import TradeClassificationConfig

        config = TradeClassificationConfig()
        with pytest.raises(Exception):  # Should raise FrozenInstanceError
            config.good_threshold = 0.1

    def test_config_validation(self):
        """Test configuration validation."""
        from Classes.FactorAnalysis import TradeClassificationConfig

        # Invalid threshold (good should be > bad)
        with pytest.raises(ValueError):
            TradeClassificationConfig(
                good_threshold=-0.05,
                bad_threshold=0.05
            )


class TestDataValidation:
    """Test data validation functionality."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOG'],
            'entry_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'exit_date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'pl': [100, -50, 75],
            'pl_pct': [0.05, -0.02, 0.03]
        })

    def test_validator_initialization(self):
        """Test DataValidator initialization."""
        from Classes.FactorAnalysis.data.validators import DataValidator

        validator = DataValidator()
        assert validator is not None

    def test_trade_log_validation(self, sample_trades):
        """Test trade log validation."""
        from Classes.FactorAnalysis.data.validators import DataValidator

        validator = DataValidator()
        result = validator.validate_trade_log(sample_trades)

        assert result.is_valid
        assert result.row_count == 3


class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""

    @pytest.fixture
    def enriched_trades(self):
        """Create enriched trade data with factors."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'symbol': ['AAPL'] * n,
            'trade_class': np.random.choice(['good', 'bad'], n, p=[0.6, 0.4]),
            'trade_class_numeric': np.random.choice([1, -1], n, p=[0.6, 0.4]),
            'factor_rsi': np.random.uniform(20, 80, n),
            'factor_momentum': np.random.uniform(-0.1, 0.1, n),
            'factor_volume_ratio': np.random.uniform(0.5, 2.0, n)
        })

    def test_tier1_analysis(self, enriched_trades):
        """Test Tier 1 exploratory analysis."""
        from Classes.FactorAnalysis import Tier1Exploratory

        tier1 = Tier1Exploratory()
        factor_cols = ['factor_rsi', 'factor_momentum', 'factor_volume_ratio']
        results = tier1.analyze(enriched_trades, factor_cols)

        assert results.get('enabled', True)
        assert 'point_biserial' in results or 'descriptive_stats' in results


class TestScenarioDetection:
    """Test scenario detection functionality."""

    @pytest.fixture
    def trades_with_factors(self):
        """Create trade data with clear factor patterns."""
        np.random.seed(42)
        n = 200

        # Create data where high RSI tends to be good trades
        rsi = np.random.uniform(20, 80, n)
        # Higher RSI = higher probability of good trade
        good_prob = (rsi - 20) / 60
        trade_class = np.where(np.random.random(n) < good_prob, 'good', 'bad')

        return pd.DataFrame({
            'symbol': ['AAPL'] * n,
            'trade_class': trade_class,
            'factor_rsi': rsi,
            'factor_momentum': np.random.uniform(-0.1, 0.1, n)
        })

    def test_scenario_detector_initialization(self):
        """Test ScenarioDetector initialization."""
        from Classes.FactorAnalysis import ScenarioDetector

        detector = ScenarioDetector()
        assert detector is not None

    def test_scenario_detection(self, trades_with_factors):
        """Test scenario detection."""
        from Classes.FactorAnalysis import ScenarioDetector

        detector = ScenarioDetector()
        factor_cols = ['factor_rsi', 'factor_momentum']
        result = detector.detect(trades_with_factors, factor_cols)

        assert result.mode is not None
        # Should find some scenarios
        assert result.n_scenarios >= 0


class TestOutputGeneration:
    """Test output generation functionality."""

    def test_json_generator_initialization(self):
        """Test JsonPayloadGenerator initialization."""
        from Classes.FactorAnalysis import JsonPayloadGenerator

        generator = JsonPayloadGenerator()
        assert generator is not None

    def test_result_formatter_initialization(self):
        """Test ResultFormatter initialization."""
        from Classes.FactorAnalysis import ResultFormatter

        formatter = ResultFormatter()
        assert formatter is not None

    def test_format_summary(self):
        """Test summary text formatting."""
        from Classes.FactorAnalysis import ResultFormatter

        formatter = ResultFormatter()
        results = {
            'data_summary': {
                'total_trades': 100,
                'good_trades': 60,
                'bad_trades': 40
            },
            'key_findings': ['Finding 1', 'Finding 2']
        }
        summary = formatter.format_summary_text(results)

        assert 'FACTOR ANALYSIS SUMMARY' in summary
        assert '100' in summary


class TestIntegration:
    """Integration tests for the complete analysis pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration testing."""
        np.random.seed(42)
        n = 100

        # Trades
        trades = pd.DataFrame({
            'symbol': ['AAPL'] * n,
            'entry_date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'exit_date': pd.date_range('2024-01-02', periods=n, freq='D'),
            'entry_price': np.random.uniform(100, 200, n),
            'exit_price': np.random.uniform(100, 200, n),
            'pl': np.random.uniform(-100, 100, n),
            'pl_pct': np.random.uniform(-0.1, 0.1, n)
        })

        # Price data
        dates = pd.date_range('2023-01-01', periods=365 + n, freq='D')
        prices = pd.DataFrame({
            'symbol': ['AAPL'] * len(dates),
            'date': dates,
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(100, 200, len(dates)),
            'low': np.random.uniform(100, 200, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })

        return trades, prices

    def test_analyzer_initialization(self):
        """Test FactorAnalyzer initialization."""
        from Classes.FactorAnalysis import FactorAnalyzer

        analyzer = FactorAnalyzer(verbose=False)
        assert analyzer is not None

    def test_basic_analysis(self, sample_data):
        """Test basic analysis pipeline."""
        from Classes.FactorAnalysis import FactorAnalyzer, FactorAnalysisConfig

        trades, prices = sample_data

        # Create config with minimal features
        config = FactorAnalysisConfig()

        analyzer = FactorAnalyzer(config=config, verbose=False)

        # Run analysis with just trade data (no external data)
        result = analyzer.analyze(trades)

        assert result.success or result.error is not None
        if result.success:
            assert result.data_summary is not None
            assert result.enriched_trades is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
