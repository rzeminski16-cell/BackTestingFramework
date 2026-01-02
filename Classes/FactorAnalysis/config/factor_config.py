"""
Configuration dataclasses for Factor Analysis Module.

All configurations are immutable (frozen=True) with validation in __post_init__.
Follows the pattern established in Classes/Config/config.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import hashlib
import json


class ThresholdType(Enum):
    """Type of threshold for trade classification."""
    ABSOLUTE = "absolute"
    PERCENTILE = "percentile"


class NormalizationType(Enum):
    """Type of normalization for factors."""
    ZSCORE = "zscore"
    PERCENTILE_RANK = "percentile_rank"
    NONE = "none"


class NullHandlingStrategy(Enum):
    """Strategy for handling null/missing values."""
    SKIP = "skip"          # Exclude trade from analysis
    ZERO = "zero"          # Treat as zero
    IMPUTE = "impute"      # Impute with cross-sectional mean or forward-fill
    EXCLUDE = "exclude"    # Exclude from specific analysis


class OutlierMethod(Enum):
    """Method for handling outliers."""
    FLAG_AND_REPORT = "flag_and_report"
    WINSORIZE = "winsorize"
    EXCLUDE = "exclude"


class AggregationMode(Enum):
    """Mode for multi-log aggregation."""
    POOLED = "pooled"
    STRATIFIED = "stratified"


class CorrectionMethod(Enum):
    """Multiple testing correction method."""
    FDR = "fdr"
    BONFERRONI = "bonferroni"


class ScenarioMode(Enum):
    """Mode for scenario analysis."""
    BINARY = "binary"
    AUTOMATIC_CLUSTERING = "automatic_clustering"
    USER_GUIDED = "user_guided"


class InteractionMode(Enum):
    """Mode for interaction analysis."""
    USER_GUIDED = "user_guided"
    EXHAUSTIVE = "exhaustive"


class Verbosity(Enum):
    """Logging verbosity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class TradeClassificationConfig:
    """Configuration for classifying trades as good/bad/indeterminate."""

    good_threshold_pct: float = 2.0
    bad_threshold_pct: float = -1.0
    indeterminate_max_days: int = 15
    bad_min_days: int = 20
    threshold_type: ThresholdType = ThresholdType.ABSOLUTE

    def __post_init__(self):
        if self.good_threshold_pct <= self.bad_threshold_pct:
            raise ValueError(
                f"good_threshold_pct ({self.good_threshold_pct}) must be > "
                f"bad_threshold_pct ({self.bad_threshold_pct})"
            )
        if self.indeterminate_max_days < 0:
            raise ValueError("indeterminate_max_days must be non-negative")
        if self.bad_min_days < 0:
            raise ValueError("bad_min_days must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "good_threshold_pct": self.good_threshold_pct,
            "bad_threshold_pct": self.bad_threshold_pct,
            "indeterminate_max_days": self.indeterminate_max_days,
            "bad_min_days": self.bad_min_days,
            "threshold_type": self.threshold_type.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeClassificationConfig":
        return cls(
            good_threshold_pct=data.get("good_threshold_pct", 2.0),
            bad_threshold_pct=data.get("bad_threshold_pct", -1.0),
            indeterminate_max_days=data.get("indeterminate_max_days", 15),
            bad_min_days=data.get("bad_min_days", 20),
            threshold_type=ThresholdType(data.get("threshold_type", "absolute"))
        )


@dataclass(frozen=True)
class DataAlignmentConfig:
    """Configuration for temporal alignment and bias prevention."""

    fundamentals_reporting_delay_days: int = 0
    insiders_reporting_delay_days: int = 3
    options_lookback_days: int = 60
    price_forward_fill_allowed: bool = True
    flag_price_gaps: bool = True

    def __post_init__(self):
        if self.fundamentals_reporting_delay_days < 0:
            raise ValueError("fundamentals_reporting_delay_days must be non-negative")
        if self.insiders_reporting_delay_days < 0:
            raise ValueError("insiders_reporting_delay_days must be non-negative")
        if self.options_lookback_days < 0:
            raise ValueError("options_lookback_days must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fundamentals_reporting_delay_days": self.fundamentals_reporting_delay_days,
            "insiders_reporting_delay_days": self.insiders_reporting_delay_days,
            "options_lookback_days": self.options_lookback_days,
            "price_forward_fill_allowed": self.price_forward_fill_allowed,
            "flag_price_gaps": self.flag_price_gaps
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataAlignmentConfig":
        return cls(
            fundamentals_reporting_delay_days=data.get("fundamentals_reporting_delay_days", 0),
            insiders_reporting_delay_days=data.get("insiders_reporting_delay_days", 3),
            options_lookback_days=data.get("options_lookback_days", 60),
            price_forward_fill_allowed=data.get("price_forward_fill_allowed", True),
            flag_price_gaps=data.get("flag_price_gaps", True)
        )


@dataclass(frozen=True)
class FactorCategoryConfig:
    """Configuration for a single factor category."""

    enabled: bool = True
    factors: Optional[tuple] = None  # None means include_all
    include_all: bool = False
    lookback_days: int = 0
    normalization: NormalizationType = NormalizationType.ZSCORE
    aggregation_window_days: Optional[int] = None  # For insider
    metrics: Optional[tuple] = None  # For insider/options

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "lookback_days": self.lookback_days,
            "normalization": self.normalization.value
        }
        if self.factors is not None:
            result["factors"] = list(self.factors)
        if self.include_all:
            result["include_all"] = self.include_all
        if self.aggregation_window_days is not None:
            result["aggregation_window_days"] = self.aggregation_window_days
        if self.metrics is not None:
            result["metrics"] = list(self.metrics)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorCategoryConfig":
        factors = data.get("factors")
        metrics = data.get("metrics")
        return cls(
            enabled=data.get("enabled", True),
            factors=tuple(factors) if factors else None,
            include_all=data.get("include_all", False),
            lookback_days=data.get("lookback_days", 0),
            normalization=NormalizationType(data.get("normalization", "zscore")),
            aggregation_window_days=data.get("aggregation_window_days"),
            metrics=tuple(metrics) if metrics else None
        )


@dataclass(frozen=True)
class NullHandlingConfig:
    """Configuration for handling null/missing values."""

    fundamental_factors: NullHandlingStrategy = NullHandlingStrategy.SKIP
    insider_factors: NullHandlingStrategy = NullHandlingStrategy.ZERO
    options_factors: NullHandlingStrategy = NullHandlingStrategy.IMPUTE
    price_data: NullHandlingStrategy = NullHandlingStrategy.EXCLUDE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fundamental_factors": self.fundamental_factors.value,
            "insider_factors": self.insider_factors.value,
            "options_factors": self.options_factors.value,
            "price_data": self.price_data.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NullHandlingConfig":
        return cls(
            fundamental_factors=NullHandlingStrategy(data.get("fundamental_factors", "skip")),
            insider_factors=NullHandlingStrategy(data.get("insider_factors", "zero")),
            options_factors=NullHandlingStrategy(data.get("options_factors", "impute")),
            price_data=NullHandlingStrategy(data.get("price_data", "exclude"))
        )


@dataclass(frozen=True)
class OutlierHandlingConfig:
    """Configuration for outlier detection and handling."""

    enabled: bool = True
    method: OutlierMethod = OutlierMethod.FLAG_AND_REPORT
    threshold_zscore: float = 3.0
    winsorize_percentile: float = 95.0

    def __post_init__(self):
        if self.threshold_zscore <= 0:
            raise ValueError("threshold_zscore must be positive")
        if not 0 < self.winsorize_percentile < 100:
            raise ValueError("winsorize_percentile must be between 0 and 100")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method.value,
            "threshold_zscore": self.threshold_zscore,
            "winsorize_percentile": self.winsorize_percentile
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutlierHandlingConfig":
        return cls(
            enabled=data.get("enabled", True),
            method=OutlierMethod(data.get("method", "flag_and_report")),
            threshold_zscore=data.get("threshold_zscore", 3.0),
            winsorize_percentile=data.get("winsorize_percentile", 95.0)
        )


@dataclass(frozen=True)
class FactorEngineeringConfig:
    """Configuration for factor engineering."""

    technical: FactorCategoryConfig = field(default_factory=lambda: FactorCategoryConfig(
        enabled=True, include_all=True, normalization=NormalizationType.ZSCORE
    ))
    value: FactorCategoryConfig = field(default_factory=lambda: FactorCategoryConfig(
        enabled=True,
        factors=("pe_ratio", "price_to_book", "price_to_sales_ttm", "peg_ratio", "dividend_yield"),
        normalization=NormalizationType.ZSCORE
    ))
    quality: FactorCategoryConfig = field(default_factory=lambda: FactorCategoryConfig(
        enabled=True,
        factors=("return_on_equity_ttm", "return_on_assets_ttm", "current_ratio", "debt_to_equity"),
        normalization=NormalizationType.ZSCORE
    ))
    growth: FactorCategoryConfig = field(default_factory=lambda: FactorCategoryConfig(
        enabled=True,
        factors=("revenue_growth_yoy", "earnings_growth_yoy", "earnings_surprise"),
        normalization=NormalizationType.ZSCORE
    ))
    insider: FactorCategoryConfig = field(default_factory=lambda: FactorCategoryConfig(
        enabled=True,
        aggregation_window_days=30,
        metrics=("buy_count", "sell_count", "net_shares", "insider_score", "buy_sell_ratio")
    ))
    options: FactorCategoryConfig = field(default_factory=lambda: FactorCategoryConfig(
        enabled=True,
        metrics=("implied_volatility", "put_call_ratio", "iv_percentile"),
        lookback_days=60
    ))
    regime: FactorCategoryConfig = field(default_factory=lambda: FactorCategoryConfig(
        enabled=True,
        metrics=("volatility_regime", "trend_regime")
    ))
    null_handling: NullHandlingConfig = field(default_factory=NullHandlingConfig)
    outlier_handling: OutlierHandlingConfig = field(default_factory=OutlierHandlingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categories": {
                "technical": self.technical.to_dict(),
                "value": self.value.to_dict(),
                "quality": self.quality.to_dict(),
                "growth": self.growth.to_dict(),
                "insider": self.insider.to_dict(),
                "options": self.options.to_dict(),
                "regime": self.regime.to_dict()
            },
            "null_handling": self.null_handling.to_dict(),
            "outlier_handling": self.outlier_handling.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorEngineeringConfig":
        categories = data.get("categories", {})
        return cls(
            technical=FactorCategoryConfig.from_dict(categories.get("technical", {})),
            value=FactorCategoryConfig.from_dict(categories.get("value", {})),
            quality=FactorCategoryConfig.from_dict(categories.get("quality", {})),
            growth=FactorCategoryConfig.from_dict(categories.get("growth", {})),
            insider=FactorCategoryConfig.from_dict(categories.get("insider", {})),
            options=FactorCategoryConfig.from_dict(categories.get("options", {})),
            regime=FactorCategoryConfig.from_dict(categories.get("regime", {})),
            null_handling=NullHandlingConfig.from_dict(data.get("null_handling", {})),
            outlier_handling=OutlierHandlingConfig.from_dict(data.get("outlier_handling", {}))
        )


@dataclass(frozen=True)
class MultiLogAggregationConfig:
    """Configuration for multi-log aggregation."""

    aggregation_mode: AggregationMode = AggregationMode.STRATIFIED
    metadata_tags: tuple = ("strategy", "symbol", "period")
    confounder_controls: tuple = ("control_symbol_effects",)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregation_mode": self.aggregation_mode.value,
            "metadata_tags": list(self.metadata_tags),
            "confounder_controls": list(self.confounder_controls)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiLogAggregationConfig":
        return cls(
            aggregation_mode=AggregationMode(data.get("aggregation_mode", "stratified")),
            metadata_tags=tuple(data.get("metadata_tags", ["strategy", "symbol", "period"])),
            confounder_controls=tuple(data.get("confounder_controls", ["control_symbol_effects"]))
        )


@dataclass(frozen=True)
class Tier1Config:
    """Configuration for Tier 1 exploratory analysis."""

    enabled: bool = True
    descriptive_stats: bool = True
    correlations: bool = True
    distributions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "descriptive_stats": self.descriptive_stats,
            "correlations": self.correlations,
            "distributions": self.distributions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tier1Config":
        return cls(
            enabled=data.get("enabled", True),
            descriptive_stats=data.get("descriptive_stats", True),
            correlations=data.get("correlations", True),
            distributions=data.get("distributions", True)
        )


@dataclass(frozen=True)
class Tier2Config:
    """Configuration for Tier 2 hypothesis testing."""

    enabled: bool = True
    logistic_regression: bool = True
    anova: bool = True
    kruskal_wallis: bool = True
    chi_square: bool = True
    wilcoxon_mannwhitney: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "logistic_regression": self.logistic_regression,
            "anova": self.anova,
            "kruskal_wallis": self.kruskal_wallis,
            "chi_square": self.chi_square,
            "wilcoxon_mannwhitney": self.wilcoxon_mannwhitney
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tier2Config":
        return cls(
            enabled=data.get("enabled", True),
            logistic_regression=data.get("logistic_regression", True),
            anova=data.get("anova", True),
            kruskal_wallis=data.get("kruskal_wallis", True),
            chi_square=data.get("chi_square", True),
            wilcoxon_mannwhitney=data.get("wilcoxon_mannwhitney", True)
        )


@dataclass(frozen=True)
class Tier3Config:
    """Configuration for Tier 3 ML-based analysis."""

    enabled: bool = True
    random_forest_importance: bool = True
    shap_analysis: bool = True
    mutual_information: bool = True
    bayesian_logistic_regression: bool = False
    n_estimators: int = 100
    random_state: int = 42

    def __post_init__(self):
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "random_forest_importance": self.random_forest_importance,
            "shap_analysis": self.shap_analysis,
            "mutual_information": self.mutual_information,
            "bayesian_logistic_regression": self.bayesian_logistic_regression,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tier3Config":
        return cls(
            enabled=data.get("enabled", True),
            random_forest_importance=data.get("random_forest_importance", True),
            shap_analysis=data.get("shap_analysis", True),
            mutual_information=data.get("mutual_information", True),
            bayesian_logistic_regression=data.get("bayesian_logistic_regression", False),
            n_estimators=data.get("n_estimators", 100),
            random_state=data.get("random_state", 42)
        )


@dataclass(frozen=True)
class MultipleTestingConfig:
    """Configuration for multiple testing correction."""

    correction_method: CorrectionMethod = CorrectionMethod.FDR
    alpha_threshold: float = 0.05
    apply_to_all_tiers: bool = True

    def __post_init__(self):
        if not 0 < self.alpha_threshold < 1:
            raise ValueError("alpha_threshold must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correction_method": self.correction_method.value,
            "alpha_threshold": self.alpha_threshold,
            "apply_to_all_tiers": self.apply_to_all_tiers
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultipleTestingConfig":
        return cls(
            correction_method=CorrectionMethod(data.get("correction_method", "fdr")),
            alpha_threshold=data.get("alpha_threshold", 0.05),
            apply_to_all_tiers=data.get("apply_to_all_tiers", True)
        )


@dataclass(frozen=True)
class StatisticalAnalysisConfig:
    """Configuration for statistical analysis."""

    tier1_exploratory: Tier1Config = field(default_factory=Tier1Config)
    tier2_hypothesis_tests: Tier2Config = field(default_factory=Tier2Config)
    tier3_ml_analysis: Tier3Config = field(default_factory=Tier3Config)
    multiple_testing: MultipleTestingConfig = field(default_factory=MultipleTestingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier1_exploratory": self.tier1_exploratory.to_dict(),
            "tier2_hypothesis_tests": self.tier2_hypothesis_tests.to_dict(),
            "tier3_ml_analysis": self.tier3_ml_analysis.to_dict(),
            "multiple_testing": self.multiple_testing.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalAnalysisConfig":
        return cls(
            tier1_exploratory=Tier1Config.from_dict(data.get("tier1_exploratory", {})),
            tier2_hypothesis_tests=Tier2Config.from_dict(data.get("tier2_hypothesis_tests", {})),
            tier3_ml_analysis=Tier3Config.from_dict(data.get("tier3_ml_analysis", {})),
            multiple_testing=MultipleTestingConfig.from_dict(data.get("multiple_testing", {}))
        )


@dataclass(frozen=True)
class ScenarioAnalysisConfig:
    """Configuration for scenario analysis."""

    scenario_mode: ScenarioMode = ScenarioMode.BINARY
    min_trades_per_scenario: int = 20
    metric: str = "calmar_ratio"
    best_scenario_threshold: float = 0.5
    interaction_mode: InteractionMode = InteractionMode.USER_GUIDED
    n_clusters: int = 3
    user_interactions: Optional[tuple] = None

    def __post_init__(self):
        if self.min_trades_per_scenario < 1:
            raise ValueError("min_trades_per_scenario must be at least 1")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "scenario_mode": self.scenario_mode.value,
            "min_trades_per_scenario": self.min_trades_per_scenario,
            "metric": self.metric,
            "best_scenario_threshold": self.best_scenario_threshold,
            "interaction_mode": self.interaction_mode.value,
            "n_clusters": self.n_clusters
        }
        if self.user_interactions:
            result["user_interactions"] = [list(i) for i in self.user_interactions]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioAnalysisConfig":
        interactions = data.get("user_interactions")
        return cls(
            scenario_mode=ScenarioMode(data.get("scenario_mode", "binary")),
            min_trades_per_scenario=data.get("min_trades_per_scenario", 20),
            metric=data.get("metric", "calmar_ratio"),
            best_scenario_threshold=data.get("best_scenario_threshold", 0.5),
            interaction_mode=InteractionMode(data.get("interaction_mode", "user_guided")),
            n_clusters=data.get("n_clusters", 3),
            user_interactions=tuple(tuple(i) for i in interactions) if interactions else None
        )


@dataclass(frozen=True)
class ExcelReportConfig:
    """Configuration for Excel report generation."""

    enabled: bool = True
    include_summary_sheet: bool = True
    include_factor_sheets: bool = True
    include_method_details: bool = True
    include_trade_details: bool = False
    include_charts: bool = True
    include_scenario_analysis: bool = True
    include_data_quality: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "include_summary_sheet": self.include_summary_sheet,
            "include_factor_sheets": self.include_factor_sheets,
            "include_method_details": self.include_method_details,
            "include_trade_details": self.include_trade_details,
            "include_charts": self.include_charts,
            "include_scenario_analysis": self.include_scenario_analysis,
            "include_data_quality": self.include_data_quality
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExcelReportConfig":
        return cls(
            enabled=data.get("enabled", True),
            include_summary_sheet=data.get("include_summary_sheet", True),
            include_factor_sheets=data.get("include_factor_sheets", True),
            include_method_details=data.get("include_method_details", True),
            include_trade_details=data.get("include_trade_details", False),
            include_charts=data.get("include_charts", True),
            include_scenario_analysis=data.get("include_scenario_analysis", True),
            include_data_quality=data.get("include_data_quality", True)
        )


@dataclass(frozen=True)
class GuiPayloadsConfig:
    """Configuration for GUI JSON payloads."""

    enabled: bool = True
    json_output_path: str = "./analysis_output/payloads.json"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "json_output_path": self.json_output_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuiPayloadsConfig":
        return cls(
            enabled=data.get("enabled", True),
            json_output_path=data.get("json_output_path", "./analysis_output/payloads.json")
        )


@dataclass(frozen=True)
class AuditLogConfig:
    """Configuration for audit logging."""

    enabled: bool = True
    output_path: str = "./analysis_output/audit_log.txt"
    verbosity: Verbosity = Verbosity.HIGH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "output_path": self.output_path,
            "verbosity": self.verbosity.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogConfig":
        return cls(
            enabled=data.get("enabled", True),
            output_path=data.get("output_path", "./analysis_output/audit_log.txt"),
            verbosity=Verbosity(data.get("verbosity", "high"))
        )


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output generation."""

    excel_report: ExcelReportConfig = field(default_factory=ExcelReportConfig)
    gui_payloads: GuiPayloadsConfig = field(default_factory=GuiPayloadsConfig)
    audit_log: AuditLogConfig = field(default_factory=AuditLogConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "excel_report": self.excel_report.to_dict(),
            "gui_payloads": self.gui_payloads.to_dict(),
            "audit_log": self.audit_log.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        return cls(
            excel_report=ExcelReportConfig.from_dict(data.get("excel_report", {})),
            gui_payloads=GuiPayloadsConfig.from_dict(data.get("gui_payloads", {})),
            audit_log=AuditLogConfig.from_dict(data.get("audit_log", {}))
        )


@dataclass(frozen=True)
class FactorAnalysisConfig:
    """
    Complete configuration for Factor Analysis.

    This is the main configuration class that contains all settings
    for running a factor analysis session.
    """

    # Profile metadata
    profile_name: str = "default"
    strategy_name: str = "unnamed_strategy"
    description: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    last_modified: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    version: str = "1.0"

    # Configuration sections
    trade_classification: TradeClassificationConfig = field(default_factory=TradeClassificationConfig)
    data_alignment: DataAlignmentConfig = field(default_factory=DataAlignmentConfig)
    factor_engineering: FactorEngineeringConfig = field(default_factory=FactorEngineeringConfig)
    multi_log_aggregation: MultiLogAggregationConfig = field(default_factory=MultiLogAggregationConfig)
    statistical_analysis: StatisticalAnalysisConfig = field(default_factory=StatisticalAnalysisConfig)
    scenario_analysis: ScenarioAnalysisConfig = field(default_factory=ScenarioAnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "profile_name": self.profile_name,
            "strategy_name": self.strategy_name,
            "description": self.description,
            "created_date": self.created_date,
            "last_modified": self.last_modified,
            "version": self.version,
            "trade_classification": self.trade_classification.to_dict(),
            "data_alignment": self.data_alignment.to_dict(),
            "factor_engineering": self.factor_engineering.to_dict(),
            "multi_log_aggregation": self.multi_log_aggregation.to_dict(),
            "statistical_analysis": self.statistical_analysis.to_dict(),
            "scenario_analysis": self.scenario_analysis.to_dict(),
            "output": self.output.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorAnalysisConfig":
        """Create configuration from dictionary."""
        return cls(
            profile_name=data.get("profile_name", "default"),
            strategy_name=data.get("strategy_name", "unnamed_strategy"),
            description=data.get("description", ""),
            created_date=data.get("created_date", datetime.now().strftime("%Y-%m-%d")),
            last_modified=data.get("last_modified", datetime.now().strftime("%Y-%m-%d")),
            version=data.get("version", "1.0"),
            trade_classification=TradeClassificationConfig.from_dict(
                data.get("trade_classification", {})
            ),
            data_alignment=DataAlignmentConfig.from_dict(
                data.get("data_alignment", {})
            ),
            factor_engineering=FactorEngineeringConfig.from_dict(
                data.get("factor_engineering", {})
            ),
            multi_log_aggregation=MultiLogAggregationConfig.from_dict(
                data.get("multi_log_aggregation", {})
            ),
            statistical_analysis=StatisticalAnalysisConfig.from_dict(
                data.get("statistical_analysis", {})
            ),
            scenario_analysis=ScenarioAnalysisConfig.from_dict(
                data.get("scenario_analysis", {})
            ),
            output=OutputConfig.from_dict(
                data.get("output", {})
            )
        )

    def compute_hash(self) -> str:
        """Compute MD5 hash of configuration for reproducibility tracking."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def with_updated_timestamp(self) -> "FactorAnalysisConfig":
        """Return a new config with updated last_modified timestamp."""
        config_dict = self.to_dict()
        config_dict["last_modified"] = datetime.now().strftime("%Y-%m-%d")
        return FactorAnalysisConfig.from_dict(config_dict)

    @classmethod
    def create_default(cls, profile_name: str = "default", strategy_name: str = "unnamed") -> "FactorAnalysisConfig":
        """Create a default configuration with sensible defaults."""
        return cls(
            profile_name=profile_name,
            strategy_name=strategy_name,
            description="Default factor analysis configuration"
        )
