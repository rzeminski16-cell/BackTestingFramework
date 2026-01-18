"""
Main Factor Analyzer for Strategy Performance Analysis.

This is the primary entry point for factor analysis. It orchestrates:
- Data loading and validation
- Trade classification
- Factor engineering
- Statistical analysis (Tier 1, 2, 3)
- Scenario detection
- Report generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
from .config.factor_config import (
    FactorAnalysisConfig,
    TradeClassificationConfig,
    DataAlignmentConfig,
    FactorEngineeringConfig,
    StatisticalAnalysisConfig,
    ScenarioAnalysisConfig,
    OutputConfig
)
from .config.profile_manager import ProfileManager

# Logging
from .logging.audit_logger import AuditLogger

# Data loading
from .data.trade_log_loader import TradeLogLoader
from .data.price_data_loader import PriceDataLoader
from .data.fundamental_loader import FundamentalLoader
from .data.insider_loader import InsiderLoader
from .data.options_loader import OptionsLoader
from .data.validators import DataValidator

# Preprocessing
from .preprocessing.trade_classifier import TradeClassifier
from .preprocessing.temporal_aligner import TemporalAligner
from .preprocessing.data_enricher import DataEnricher
from .preprocessing.quality_scorer import QualityScorer
from .preprocessing.multi_log_aggregator import MultiLogAggregator

# Factor engineering
from .factors.technical_factors import TechnicalFactors
from .factors.fundamental_factors import FundamentalFactors
from .factors.insider_factors import InsiderFactors
from .factors.options_factors import OptionsFactors
from .factors.regime_factors import RegimeFactors
from .factors.normalizers import FactorNormalizer
from .factors.outlier_handler import OutlierHandler

# Analysis
from .analysis.tier1_exploratory import Tier1Exploratory
from .analysis.tier2_hypothesis import Tier2Hypothesis
from .analysis.tier3_ml import Tier3ML
from .analysis.multiple_testing import MultipleTestingCorrector

# Scenarios
from .scenarios.scenario_detector import ScenarioDetector
from .scenarios.scenario_validator import ScenarioValidator
from .scenarios.interaction_analyzer import InteractionAnalyzer

# Output
from .output.excel_generator import ExcelReportGenerator
from .output.json_generator import JsonPayloadGenerator
from .output.formatters import ResultFormatter


@dataclass
class AnalysisInput:
    """Input data for factor analysis."""
    trade_logs: List[pd.DataFrame] = field(default_factory=list)
    log_metadata: Optional[List[Dict]] = None
    price_data: Optional[pd.DataFrame] = None
    fundamental_data: Optional[pd.DataFrame] = None
    insider_data: Optional[pd.DataFrame] = None
    options_data: Optional[pd.DataFrame] = None


@dataclass
class AnalysisOutput:
    """Complete output from factor analysis."""
    success: bool
    timestamp: str
    config_used: Dict
    data_summary: Dict
    quality_score: Dict
    enriched_trades: Optional[pd.DataFrame]
    tier1: Optional[Dict]
    tier2: Optional[Dict]
    tier3: Optional[Any]
    multiple_testing: Optional[Any]
    scenarios: Optional[Any]
    interactions: Optional[Any]
    key_findings: List[str]
    warnings: List[str]
    error: Optional[str] = None


class FactorAnalyzer:
    """
    Main orchestrator for factor analysis.

    This class provides a unified API for:
    1. Loading and validating data
    2. Classifying trades
    3. Engineering factors
    4. Running statistical analysis
    5. Detecting scenarios
    6. Generating reports

    Usage:
        analyzer = FactorAnalyzer(config)
        result = analyzer.analyze(trades_df, price_data=prices_df)
        analyzer.generate_report(result, 'output.xlsx')
    """

    def __init__(
        self,
        config: Optional[FactorAnalysisConfig] = None,
        config_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize FactorAnalyzer.

        Args:
            config: Analysis configuration
            config_path: Path to config YAML file
            verbose: Enable verbose logging
        """
        # Load config
        if config_path:
            profile_manager = ProfileManager()
            self.config = profile_manager.load_profile(config_path)
        else:
            self.config = config or FactorAnalysisConfig()

        self.verbose = verbose

        # Initialize logger
        from .config.factor_config import Verbosity
        self.logger = AuditLogger(
            enabled=True,
            verbosity=Verbosity.HIGH if verbose else Verbosity.LOW,
            console_output=verbose
        )

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize analysis components."""
        # Data loaders
        self.trade_loader = TradeLogLoader(logger=self.logger)
        self.price_loader = PriceDataLoader(logger=self.logger)
        self.fundamental_loader = FundamentalLoader(logger=self.logger)
        self.insider_loader = InsiderLoader(logger=self.logger)
        self.options_loader = OptionsLoader(logger=self.logger)
        self.validator = DataValidator()

        # Preprocessing
        self.classifier = TradeClassifier(
            config=self.config.trade_classification,
            logger=self.logger
        )
        self.aligner = TemporalAligner(
            config=self.config.data_alignment,
            logger=self.logger
        )
        self.enricher = DataEnricher(logger=self.logger)
        self.quality_scorer = QualityScorer(logger=self.logger)
        self.aggregator = MultiLogAggregator(logger=self.logger)

        # Factors
        self.technical_factors = TechnicalFactors(logger=self.logger)
        self.fundamental_factors = FundamentalFactors(logger=self.logger)
        self.insider_factors = InsiderFactors(logger=self.logger)
        self.options_factors = OptionsFactors(logger=self.logger)
        self.regime_factors = RegimeFactors(logger=self.logger)
        self.normalizer = FactorNormalizer(logger=self.logger)
        self.outlier_handler = OutlierHandler(
            config=self.config.factor_engineering.outlier_handling,
            logger=self.logger
        )

        # Analysis
        self.tier1 = Tier1Exploratory(
            config=self.config.statistical_analysis.tier1_exploratory,
            logger=self.logger
        )
        self.tier2 = Tier2Hypothesis(
            config=self.config.statistical_analysis.tier2_hypothesis_tests,
            logger=self.logger
        )
        self.tier3 = Tier3ML(
            config=self.config.statistical_analysis.tier3_ml_analysis,
            logger=self.logger
        )
        self.multiple_testing = MultipleTestingCorrector(logger=self.logger)

        # Scenarios
        self.scenario_detector = ScenarioDetector(
            config=self.config.scenario_analysis,
            logger=self.logger
        )
        self.scenario_validator = ScenarioValidator(logger=self.logger)
        self.interaction_analyzer = InteractionAnalyzer(logger=self.logger)

        # Output
        self.excel_generator = ExcelReportGenerator(logger=self.logger)
        self.json_generator = JsonPayloadGenerator(logger=self.logger)
        self.formatter = ResultFormatter()

    def analyze(
        self,
        trade_data: Union[pd.DataFrame, List[pd.DataFrame], str],
        price_data: Optional[Union[pd.DataFrame, str]] = None,
        fundamental_data: Optional[Union[pd.DataFrame, str]] = None,
        insider_data: Optional[Union[pd.DataFrame, str]] = None,
        options_data: Optional[Union[pd.DataFrame, str]] = None,
        log_metadata: Optional[List[Dict]] = None
    ) -> AnalysisOutput:
        """
        Run complete factor analysis.

        Args:
            trade_data: Trade log DataFrame(s) or path to CSV
            price_data: Price data DataFrame or path
            fundamental_data: Fundamental data DataFrame or path
            insider_data: Insider data DataFrame or path
            options_data: Options data DataFrame or path
            log_metadata: Metadata for each trade log

        Returns:
            AnalysisOutput with complete results
        """
        self.logger.start_section("FACTOR_ANALYSIS")
        timestamp = datetime.now().isoformat()
        warnings = []
        key_findings = []

        try:
            # 1. Load and prepare data
            self.logger.info("Loading and preparing data...")
            input_data = self._prepare_input(
                trade_data, price_data, fundamental_data,
                insider_data, options_data, log_metadata
            )

            # 2. Aggregate multiple logs if needed
            if len(input_data.trade_logs) > 1:
                trades_df, agg_result = self.aggregator.aggregate(
                    input_data.trade_logs,
                    input_data.log_metadata
                )
                warnings.extend(agg_result.warnings)
            else:
                trades_df = input_data.trade_logs[0]

            # 3. Classify trades
            self.logger.info("Classifying trades...")
            trades_df, _ = self.classifier.classify_trades(trades_df)

            # 4. Check class balance
            is_balanced, balance_info = self.aggregator.check_class_balance(trades_df)
            if not is_balanced:
                warnings.append(balance_info.get('warning', 'Class imbalance detected'))

            # 5. Engineer factors
            self.logger.info("Engineering factors...")
            factor_columns = []
            fe_config = self.config.factor_engineering

            # Technical factors
            if fe_config.technical.enabled and input_data.price_data is not None:
                trades_df = self.technical_factors.compute_all(
                    trades_df, input_data.price_data
                )
                factor_columns.extend(self.technical_factors.get_factor_names())

            # Fundamental factors (EPS-only by default, or value/quality/growth if enabled)
            # EPS-only mode is used when fundamental data is mostly missing for whole periods
            eps_only_mode = fe_config.eps_only_fundamentals
            traditional_fundamental_enabled = (
                fe_config.value.enabled or
                fe_config.quality.enabled or
                fe_config.growth.enabled
            )
            fundamental_enabled = eps_only_mode or traditional_fundamental_enabled

            # Debug: Check if fundamental data is available
            has_fundamental = input_data.fundamental_data is not None
            fund_len = len(input_data.fundamental_data) if has_fundamental else 0
            print(f"[DEBUG] Fundamental data available: {has_fundamental}, rows: {fund_len}")
            print(f"[DEBUG] eps_only_mode: {eps_only_mode}, fundamental_enabled: {fundamental_enabled}")

            if fundamental_enabled and input_data.fundamental_data is not None:
                print(f"[DEBUG] Fundamental data columns: {list(input_data.fundamental_data.columns)}")
                aligned_fund = self.aligner.align_fundamentals(
                    trades_df, input_data.fundamental_data
                )
                print(f"[DEBUG] Aligned fundamental data rows: {len(aligned_fund)}, columns: {list(aligned_fund.columns)}")
                trades_df = self.fundamental_factors.compute_all(
                    trades_df, aligned_fund, eps_only=eps_only_mode
                )
                factor_columns.extend(self.fundamental_factors.get_factor_names())
                print(f"[DEBUG] After compute_all, trades_df columns: {[c for c in trades_df.columns if c.startswith('eps_') or c.startswith('composite_')]}")
            else:
                print(f"[DEBUG] Skipping fundamental factors - enabled: {fundamental_enabled}, data available: {has_fundamental}")

            # Insider factors
            if fe_config.insider.enabled and input_data.insider_data is not None:
                aligned_insider = self.aligner.align_insider_data(
                    trades_df, input_data.insider_data
                )
                trades_df = self.insider_factors.compute_all(
                    trades_df, aligned_insider
                )
                factor_columns.extend(self.insider_factors.get_factor_names())

            # Options factors
            if fe_config.options.enabled and input_data.options_data is not None:
                trades_df = self.options_factors.compute_all(
                    trades_df, input_data.options_data
                )
                factor_columns.extend(self.options_factors.get_factor_names())

            # Regime factors (part of technical)
            if fe_config.technical.enabled and input_data.price_data is not None:
                trades_df = self.regime_factors.compute_all(
                    trades_df, input_data.price_data
                )
                factor_columns.extend(self.regime_factors.get_factor_names())

            # Filter factor_columns to only include columns that exist in trades_df
            valid_factor_columns = [c for c in factor_columns if c in trades_df.columns]
            missing_factors = set(factor_columns) - set(valid_factor_columns)
            if missing_factors:
                self.logger.warning(f"Factor columns not found in trades_df: {missing_factors}")
                print(f"[WARNING] Factor columns not found in trades_df: {missing_factors}")
            factor_columns = valid_factor_columns

            self.logger.info(f"Valid factor columns for analysis: {len(factor_columns)}")
            print(f"[INFO] Valid factor columns for analysis ({len(factor_columns)}): {factor_columns}")

            # 6. Handle outliers
            trades_df, outlier_result = self.outlier_handler.handle_outliers(
                trades_df, factor_columns
            )

            # 7. Normalize factors (always apply if there are factor columns)
            if factor_columns:
                trades_df = self.normalizer.normalize(trades_df, factor_columns)

            # 8. Calculate quality score
            trades_df, quality_report = self.quality_scorer.score_all_trades(trades_df)
            quality_score = {
                'avg_quality_score': quality_report.avg_quality_score,
                'total_trades': quality_report.total_trades,
                'trades_excellent': quality_report.trades_excellent,
                'trades_good': quality_report.trades_good,
                'trades_fair': quality_report.trades_fair,
                'trades_poor': quality_report.trades_poor
            }

            # 9. Data summary
            data_summary = self._compute_data_summary(trades_df)

            # 10. Run statistical analysis
            self.logger.info("Running statistical analysis...")

            # Tier 1: Exploratory
            tier1_results = self.tier1.analyze(
                trades_df, factor_columns
            )

            # Tier 2: Hypothesis testing
            tier2_results = self.tier2.analyze(
                trades_df, factor_columns
            )

            # Tier 3: ML
            tier3_results = self.tier3.analyze(
                trades_df, factor_columns
            )

            # Multiple testing correction
            mt_results = self.multiple_testing.correct_from_tier2(tier2_results)

            # 11. Scenario detection
            self.logger.info("Detecting scenarios...")
            scenario_results = self.scenario_detector.detect(
                trades_df, factor_columns
            )

            # Validate scenarios
            all_scenarios = (
                scenario_results.best_scenarios +
                scenario_results.worst_scenarios
            )
            validation_results = self.scenario_validator.validate_all(
                trades_df, all_scenarios
            )

            # 12. Interaction analysis
            interaction_results = self.interaction_analyzer.analyze_interactions(
                trades_df, factor_columns
            )

            # 13. Generate key findings
            key_findings = self._generate_key_findings(
                tier1_results, tier2_results, tier3_results,
                scenario_results, mt_results
            )

            self.logger.end_section()

            return AnalysisOutput(
                success=True,
                timestamp=timestamp,
                config_used=self._config_to_dict(),
                data_summary=data_summary,
                quality_score=quality_score,
                enriched_trades=trades_df,
                tier1=tier1_results,
                tier2=tier2_results,
                tier3=tier3_results,
                multiple_testing=mt_results,
                scenarios=scenario_results,
                interactions=interaction_results,
                key_findings=key_findings,
                warnings=warnings
            )

        except Exception as e:
            self.logger.error("Analysis failed", {'error': str(e)})
            self.logger.end_section()
            return AnalysisOutput(
                success=False,
                timestamp=timestamp,
                config_used=self._config_to_dict(),
                data_summary={},
                quality_score={},
                enriched_trades=None,
                tier1=None,
                tier2=None,
                tier3=None,
                multiple_testing=None,
                scenarios=None,
                interactions=None,
                key_findings=[],
                warnings=warnings,
                error=str(e)
            )

    def _prepare_input(
        self,
        trade_data: Union[pd.DataFrame, List[pd.DataFrame], str],
        price_data: Optional[Union[pd.DataFrame, str]],
        fundamental_data: Optional[Union[pd.DataFrame, str]],
        insider_data: Optional[Union[pd.DataFrame, str]],
        options_data: Optional[Union[pd.DataFrame, str]],
        log_metadata: Optional[List[Dict]]
    ) -> AnalysisInput:
        """Prepare input data from various formats."""
        result = AnalysisInput()

        # Trade data
        if isinstance(trade_data, str):
            df, _, _ = self.trade_loader.load(trade_data)
            result.trade_logs = [df]
        elif isinstance(trade_data, list):
            result.trade_logs = trade_data
        else:
            result.trade_logs = [trade_data]

        result.log_metadata = log_metadata

        # Price data
        if isinstance(price_data, str):
            result.price_data, _, _ = self.price_loader.load(price_data)
        else:
            result.price_data = price_data

        # Fundamental data
        if isinstance(fundamental_data, str):
            result.fundamental_data, _, _ = self.fundamental_loader.load(fundamental_data)
        else:
            result.fundamental_data = fundamental_data

        # Insider data
        if isinstance(insider_data, str):
            result.insider_data, _, _ = self.insider_loader.load(insider_data)
        else:
            result.insider_data = insider_data

        # Options data
        if isinstance(options_data, str):
            result.options_data, _, _ = self.options_loader.load(options_data)
        else:
            result.options_data = options_data

        return result

    def _compute_data_summary(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics for the data."""
        summary = {
            'total_trades': len(df),
            'unique_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0
        }

        if 'trade_class' in df.columns:
            class_counts = df['trade_class'].value_counts()
            summary['good_trades'] = int(class_counts.get('good', 0))
            summary['bad_trades'] = int(class_counts.get('bad', 0))
            summary['indeterminate_trades'] = int(class_counts.get('indeterminate', 0))

        if 'pl_pct' in df.columns:
            summary['avg_return'] = float(df['pl_pct'].mean())
            summary['total_return'] = float(df['pl_pct'].sum())
            summary['win_rate'] = float((df['pl_pct'] > 0).mean())

        if 'entry_date' in df.columns:
            summary['date_range'] = {
                'start': str(df['entry_date'].min()),
                'end': str(df['entry_date'].max())
            }

        return summary

    def _generate_key_findings(
        self,
        tier1: Dict,
        tier2: Dict,
        tier3: Any,
        scenarios: Any,
        mt_results: Any
    ) -> List[str]:
        """Generate key findings from analysis results."""
        findings = []

        # Significant correlations
        if tier1.get('point_biserial'):
            sig_corrs = [c for c in tier1['point_biserial'] if c.significant]
            if sig_corrs:
                top_corr = max(sig_corrs, key=lambda x: abs(x.correlation))
                direction = "positively" if top_corr.correlation > 0 else "negatively"
                findings.append(
                    f"{top_corr.factor} is {direction} correlated with good trades "
                    f"(r={top_corr.correlation:.3f}, p={top_corr.p_value:.4f})"
                )

        # Significant regression coefficients
        if tier2.get('logistic_regression'):
            reg = tier2['logistic_regression']
            sig_factors = reg.get_significant_factors()
            if sig_factors:
                top_factor = sig_factors[0]
                effect = "increases" if top_factor.coefficient > 0 else "decreases"
                findings.append(
                    f"Higher {top_factor.factor_name} {effect} probability of good trade "
                    f"(OR={top_factor.odds_ratio:.2f})"
                )

        # Multiple testing
        if mt_results:
            before = mt_results.n_significant_original
            after = mt_results.n_significant_corrected
            if before > after:
                findings.append(
                    f"After FDR correction, {after} of {before} significant factors remain"
                )

        # Best scenarios
        if scenarios and scenarios.best_scenarios:
            best = scenarios.best_scenarios[0]
            findings.append(
                f"Best scenario '{best.name}' has {best.lift:.1f}x lift "
                f"({best.n_trades} trades, {best.good_trade_rate:.0%} good rate)"
            )

        # Worst scenarios
        if scenarios and scenarios.worst_scenarios:
            worst = scenarios.worst_scenarios[0]
            findings.append(
                f"Worst scenario '{worst.name}' has {worst.lift:.2f}x lift "
                f"({worst.n_trades} trades, {worst.good_trade_rate:.0%} good rate)"
            )

        # ML importance
        if tier3 and tier3.rf_feature_importances:
            top_feat = tier3.rf_feature_importances[0]
            findings.append(
                f"Most important factor in ML model: {top_feat.feature_name} "
                f"(importance={top_feat.importance:.3f})"
            )

        return findings

    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'classification': {
                'good_threshold_pct': self.config.trade_classification.good_threshold_pct,
                'bad_threshold_pct': self.config.trade_classification.bad_threshold_pct
            },
            'alignment': {
                'fundamentals_delay_days': self.config.data_alignment.fundamentals_reporting_delay_days,
                'insiders_delay_days': self.config.data_alignment.insiders_reporting_delay_days
            },
            'factors': {
                'technical': self.config.factor_engineering.technical.enabled,
                'value': self.config.factor_engineering.value.enabled,
                'quality': self.config.factor_engineering.quality.enabled,
                'growth': self.config.factor_engineering.growth.enabled,
                'insider': self.config.factor_engineering.insider.enabled,
                'options': self.config.factor_engineering.options.enabled
            }
        }

    def generate_excel_report(
        self,
        result: AnalysisOutput,
        output_path: str
    ) -> str:
        """
        Generate Excel report from analysis results.

        Args:
            result: AnalysisOutput from analyze()
            output_path: Output file path

        Returns:
            Path to generated file
        """
        results_dict = {
            'data_summary': result.data_summary,
            'quality_score': result.quality_score,
            'tier1': result.tier1,
            'tier2': result.tier2,
            'tier3': result.tier3,
            'scenarios': result.scenarios,
            'key_findings': result.key_findings,
            'enriched_trades': result.enriched_trades
        }

        return self.excel_generator.generate_report(results_dict, output_path)

    def generate_json_payload(
        self,
        result: AnalysisOutput,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate JSON payload from analysis results.

        Args:
            result: AnalysisOutput from analyze()
            output_path: Optional output file path

        Returns:
            JSON-serializable dictionary
        """
        results_dict = {
            'data_summary': result.data_summary,
            'quality_score': result.quality_score,
            'tier1': result.tier1,
            'tier2': result.tier2,
            'tier3': result.tier3,
            'scenarios': result.scenarios,
            'key_findings': result.key_findings
        }

        return self.json_generator.generate(results_dict, output_path)

    def print_summary(self, result: AnalysisOutput) -> None:
        """Print formatted summary to console."""
        results_dict = {
            'data_summary': result.data_summary,
            'tier2': result.tier2,
            'scenarios': result.scenarios,
            'key_findings': result.key_findings
        }
        print(self.formatter.format_summary_text(results_dict))

    def get_audit_log(self) -> List[Dict]:
        """Get the audit log entries."""
        return self.logger.get_log_entries()

    def save_audit_log(self, path: str) -> None:
        """Save audit log to file."""
        self.logger.save_to_file(path)
