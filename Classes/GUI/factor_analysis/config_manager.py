"""
Factor Analysis Configuration Manager GUI.

A CustomTkinter application for managing Factor Analysis configuration profiles.
Allows users to create, edit, and manage analysis settings.
"""

import customtkinter as ctk
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from dataclasses import asdict

from ..ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info, ask_yes_no
from .components import ConfigSection, ProfileSelector

# Import factor analysis config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
try:
    from Classes.FactorAnalysis.config.factor_config import (
        FactorAnalysisConfig, TradeClassificationConfig, DataAlignmentConfig,
        FactorEngineeringConfig, StatisticalAnalysisConfig, ScenarioAnalysisConfig,
        TechnicalFactorConfig, ValueFactorConfig, QualityFactorConfig,
        GrowthFactorConfig, InsiderFactorConfig, OptionsFactorConfig,
        Tier1Config, Tier2Config, Tier3Config, ScenarioMode
    )
    FACTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    FACTOR_ANALYSIS_AVAILABLE = False


class FactorConfigManagerGUI:
    """
    Configuration Manager for Factor Analysis.

    Provides a wizard-like interface for configuring all aspects
    of factor analysis including:
    - Trade classification settings
    - Data alignment settings
    - Factor engineering settings
    - Statistical analysis settings
    - Scenario analysis settings
    """

    CONFIG_DIR = Path("config/factor_analysis")

    def __init__(self, parent: ctk.CTk = None, on_save: Optional[Callable[[dict], None]] = None):
        """
        Initialize the Configuration Manager.

        Args:
            parent: Parent window (if modal)
            on_save: Callback when config is saved
        """
        Theme.setup()

        self.on_save = on_save
        self.config_dir = self.CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.current_profile: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.modified = False

        # Create window
        if parent:
            self.root = ctk.CTkToplevel(parent)
            self.root.transient(parent)
        else:
            self.root = ctk.CTk()

        self.root.title("Factor Analysis Configuration Manager")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.root.configure(fg_color=Colors.BG_DARK)

        self._create_layout()
        self._load_profiles()

    def _create_layout(self):
        """Create the main layout."""
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self.root, fg_color=Colors.SURFACE, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")

        header_content = Theme.create_frame(header)
        header_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_header(header_content, "Configuration Manager", size="l").pack(side="left")

        # Profile selector
        self.profile_selector = ProfileSelector(
            header_content,
            profiles=[],
            on_profile_change=self._on_profile_change,
            on_create=self._on_create_profile,
            on_edit=self._on_edit_profile,
            on_delete=self._on_delete_profile
        )
        self.profile_selector.pack(side="right")

        # Main content - scrollable
        content_container = Theme.create_frame(self.root)
        content_container.grid(row=1, column=0, sticky="nsew", padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        content_container.grid_columnconfigure(0, weight=1)
        content_container.grid_rowconfigure(0, weight=1)

        self.content = ctk.CTkScrollableFrame(
            content_container,
            fg_color="transparent"
        )
        self.content.grid(row=0, column=0, sticky="nsew")

        # Create configuration sections
        self._create_trade_classification_section()
        self._create_data_alignment_section()
        self._create_factor_engineering_section()
        self._create_statistical_analysis_section()
        self._create_scenario_analysis_section()

        # Footer with action buttons
        footer = ctk.CTkFrame(self.root, fg_color=Colors.SURFACE, corner_radius=0)
        footer.grid(row=2, column=0, sticky="ew")

        footer_content = Theme.create_frame(footer)
        footer_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        Theme.create_button(
            footer_content, "Cancel",
            command=self._on_cancel,
            style="secondary",
            width=100
        ).pack(side="right", padx=(Sizes.PAD_S, 0))

        Theme.create_button(
            footer_content, "Save",
            command=self._on_save,
            style="primary",
            width=100
        ).pack(side="right")

        Theme.create_button(
            footer_content, "Reset to Defaults",
            command=self._on_reset,
            style="ghost",
            width=130
        ).pack(side="left")

    def _create_trade_classification_section(self):
        """Create trade classification settings section."""
        section = ConfigSection(self.content, "Trade Classification", expanded=True)
        section.pack(fill="x", pady=(0, Sizes.PAD_S))

        content = section.get_content_frame()

        # Classification method
        method_frame = Theme.create_frame(content)
        method_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(method_frame, "Classification Method:").pack(side="left")
        self.classification_method_var = ctk.StringVar(value="threshold")
        Theme.create_combobox(
            method_frame,
            values=["threshold", "percentile", "risk_adjusted"],
            variable=self.classification_method_var,
            width=150
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Good threshold
        good_frame = Theme.create_frame(content)
        good_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(good_frame, "Good Trade Threshold (%):").pack(side="left")
        self.good_threshold_var = ctk.StringVar(value="0.0")
        Theme.create_entry(good_frame, width=100).configure(textvariable=self.good_threshold_var)
        good_entry = Theme.create_entry(good_frame, width=100)
        good_entry.configure(textvariable=self.good_threshold_var)
        good_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(good_frame, "(Trades above this return % are 'good')").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Bad threshold
        bad_frame = Theme.create_frame(content)
        bad_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(bad_frame, "Bad Trade Threshold (%):").pack(side="left")
        self.bad_threshold_var = ctk.StringVar(value="0.0")
        bad_entry = Theme.create_entry(bad_frame, width=100)
        bad_entry.configure(textvariable=self.bad_threshold_var)
        bad_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(bad_frame, "(Trades below this return % are 'bad')").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Min holding period
        hold_frame = Theme.create_frame(content)
        hold_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(hold_frame, "Min Holding Period (days):").pack(side="left")
        self.min_holding_var = ctk.StringVar(value="1")
        hold_entry = Theme.create_entry(hold_frame, width=100)
        hold_entry.configure(textvariable=self.min_holding_var)
        hold_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Exclude options
        exclude_frame = Theme.create_frame(content)
        exclude_frame.pack(fill="x", pady=Sizes.PAD_S)

        self.exclude_partial_var = ctk.BooleanVar(value=False)
        Theme.create_checkbox(
            exclude_frame, "Exclude partial exits",
            variable=self.exclude_partial_var
        ).pack(side="left")

        self.exclude_stopped_var = ctk.BooleanVar(value=False)
        Theme.create_checkbox(
            exclude_frame, "Exclude stopped-out trades",
            variable=self.exclude_stopped_var
        ).pack(side="left", padx=(Sizes.PAD_L, 0))

    def _create_data_alignment_section(self):
        """Create data alignment settings section."""
        section = ConfigSection(self.content, "Data Alignment", expanded=True)
        section.pack(fill="x", pady=Sizes.PAD_S)

        content = section.get_content_frame()

        # Temporal mode
        mode_frame = Theme.create_frame(content)
        mode_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(mode_frame, "Alignment Mode:").pack(side="left")
        self.alignment_mode_var = ctk.StringVar(value="strict")
        Theme.create_combobox(
            mode_frame,
            values=["strict", "relaxed", "interpolate"],
            variable=self.alignment_mode_var,
            width=150
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Lookback days
        lookback_frame = Theme.create_frame(content)
        lookback_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(lookback_frame, "Lookback Days:").pack(side="left")
        self.lookback_days_var = ctk.StringVar(value="1")
        lookback_entry = Theme.create_entry(lookback_frame, width=100)
        lookback_entry.configure(textvariable=self.lookback_days_var)
        lookback_entry.pack(side="left", padx=(Sizes.PAD_S, 0))
        Theme.create_hint(lookback_frame, "(Days before trade entry for factor data)").pack(side="left", padx=(Sizes.PAD_S, 0))

        # Min required days
        min_days_frame = Theme.create_frame(content)
        min_days_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(min_days_frame, "Min Required Days:").pack(side="left")
        self.min_required_days_var = ctk.StringVar(value="252")
        min_days_entry = Theme.create_entry(min_days_frame, width=100)
        min_days_entry.configure(textvariable=self.min_required_days_var)
        min_days_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Forward looking prevention
        forward_frame = Theme.create_frame(content)
        forward_frame.pack(fill="x", pady=Sizes.PAD_S)

        self.prevent_forward_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            forward_frame, "Prevent forward-looking bias",
            variable=self.prevent_forward_var
        ).pack(side="left")

    def _create_factor_engineering_section(self):
        """Create factor engineering settings section."""
        section = ConfigSection(self.content, "Factor Engineering", expanded=False)
        section.pack(fill="x", pady=Sizes.PAD_S)

        content = section.get_content_frame()

        # Factor categories
        categories_label = Theme.create_label(
            content, "Enable Factor Categories:",
            font=Fonts.LABEL_BOLD
        )
        categories_label.pack(anchor="w", pady=(0, Sizes.PAD_S))

        cat_frame = Theme.create_frame(content)
        cat_frame.pack(fill="x", pady=Sizes.PAD_S)

        self.factor_category_vars = {}
        categories = ["Technical", "Value", "Quality", "Growth", "Insider", "Options"]

        for i, cat in enumerate(categories):
            var = ctk.BooleanVar(value=True)
            self.factor_category_vars[cat.lower()] = var
            Theme.create_checkbox(
                cat_frame, cat,
                variable=var
            ).grid(row=i // 3, column=i % 3, sticky="w", padx=Sizes.PAD_S, pady=Sizes.PAD_XS)

        # Normalize factors
        normalize_frame = Theme.create_frame(content)
        normalize_frame.pack(fill="x", pady=Sizes.PAD_S)

        self.normalize_factors_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            normalize_frame, "Normalize factors (z-score)",
            variable=self.normalize_factors_var
        ).pack(side="left")

        # Handle outliers
        outlier_frame = Theme.create_frame(content)
        outlier_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(outlier_frame, "Outlier Handling:").pack(side="left")
        self.outlier_method_var = ctk.StringVar(value="winsorize")
        Theme.create_combobox(
            outlier_frame,
            values=["none", "winsorize", "clip", "remove"],
            variable=self.outlier_method_var,
            width=150
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Outlier threshold
        threshold_frame = Theme.create_frame(content)
        threshold_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(threshold_frame, "Outlier Threshold (std):").pack(side="left")
        self.outlier_threshold_var = ctk.StringVar(value="3.0")
        threshold_entry = Theme.create_entry(threshold_frame, width=100)
        threshold_entry.configure(textvariable=self.outlier_threshold_var)
        threshold_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

    def _create_statistical_analysis_section(self):
        """Create statistical analysis settings section."""
        section = ConfigSection(self.content, "Statistical Analysis", expanded=False)
        section.pack(fill="x", pady=Sizes.PAD_S)

        content = section.get_content_frame()

        # Tier 1 - Exploratory
        tier1_label = Theme.create_label(
            content, "Tier 1: Exploratory Analysis",
            font=Fonts.LABEL_BOLD,
            text_color=Colors.PRIMARY_LIGHT
        )
        tier1_label.pack(anchor="w", pady=(0, Sizes.PAD_S))

        tier1_frame = Theme.create_frame(content)
        tier1_frame.pack(fill="x", padx=(Sizes.PAD_M, 0), pady=(0, Sizes.PAD_M))

        self.tier1_enabled_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            tier1_frame, "Enable Tier 1",
            variable=self.tier1_enabled_var
        ).pack(anchor="w")

        corr_frame = Theme.create_frame(tier1_frame)
        corr_frame.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(corr_frame, "Correlation threshold:").pack(side="left")
        self.tier1_corr_threshold_var = ctk.StringVar(value="0.1")
        tier1_entry = Theme.create_entry(corr_frame, width=80)
        tier1_entry.configure(textvariable=self.tier1_corr_threshold_var)
        tier1_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Tier 2 - Hypothesis Testing
        tier2_label = Theme.create_label(
            content, "Tier 2: Hypothesis Testing",
            font=Fonts.LABEL_BOLD,
            text_color=Colors.PRIMARY_LIGHT
        )
        tier2_label.pack(anchor="w", pady=(Sizes.PAD_S, Sizes.PAD_S))

        tier2_frame = Theme.create_frame(content)
        tier2_frame.pack(fill="x", padx=(Sizes.PAD_M, 0), pady=(0, Sizes.PAD_M))

        self.tier2_enabled_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            tier2_frame, "Enable Tier 2",
            variable=self.tier2_enabled_var
        ).pack(anchor="w")

        alpha_frame = Theme.create_frame(tier2_frame)
        alpha_frame.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(alpha_frame, "Significance level (alpha):").pack(side="left")
        self.tier2_alpha_var = ctk.StringVar(value="0.05")
        tier2_entry = Theme.create_entry(alpha_frame, width=80)
        tier2_entry.configure(textvariable=self.tier2_alpha_var)
        tier2_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        self.tier2_fdr_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            tier2_frame, "Apply FDR correction",
            variable=self.tier2_fdr_var
        ).pack(anchor="w", pady=Sizes.PAD_XS)

        # Tier 3 - ML Analysis
        tier3_label = Theme.create_label(
            content, "Tier 3: ML Analysis",
            font=Fonts.LABEL_BOLD,
            text_color=Colors.PRIMARY_LIGHT
        )
        tier3_label.pack(anchor="w", pady=(Sizes.PAD_S, Sizes.PAD_S))

        tier3_frame = Theme.create_frame(content)
        tier3_frame.pack(fill="x", padx=(Sizes.PAD_M, 0))

        self.tier3_enabled_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            tier3_frame, "Enable Tier 3",
            variable=self.tier3_enabled_var
        ).pack(anchor="w")

        trees_frame = Theme.create_frame(tier3_frame)
        trees_frame.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(trees_frame, "Random Forest trees:").pack(side="left")
        self.tier3_n_trees_var = ctk.StringVar(value="100")
        tier3_entry = Theme.create_entry(trees_frame, width=80)
        tier3_entry.configure(textvariable=self.tier3_n_trees_var)
        tier3_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        self.tier3_shap_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            tier3_frame, "Compute SHAP values",
            variable=self.tier3_shap_var
        ).pack(anchor="w", pady=Sizes.PAD_XS)

    def _create_scenario_analysis_section(self):
        """Create scenario analysis settings section."""
        section = ConfigSection(self.content, "Scenario Analysis", expanded=False)
        section.pack(fill="x", pady=Sizes.PAD_S)

        content = section.get_content_frame()

        # Enable/mode
        mode_frame = Theme.create_frame(content)
        mode_frame.pack(fill="x", pady=Sizes.PAD_S)

        self.scenario_enabled_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            mode_frame, "Enable Scenario Analysis",
            variable=self.scenario_enabled_var
        ).pack(side="left")

        # Scenario mode
        scenario_mode_frame = Theme.create_frame(content)
        scenario_mode_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(scenario_mode_frame, "Detection Mode:").pack(side="left")
        self.scenario_mode_var = ctk.StringVar(value="binary")
        Theme.create_combobox(
            scenario_mode_frame,
            values=["binary", "clustering", "rule_based"],
            variable=self.scenario_mode_var,
            width=150
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Min samples per scenario
        samples_frame = Theme.create_frame(content)
        samples_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(samples_frame, "Min samples per scenario:").pack(side="left")
        self.scenario_min_samples_var = ctk.StringVar(value="30")
        samples_entry = Theme.create_entry(samples_frame, width=100)
        samples_entry.configure(textvariable=self.scenario_min_samples_var)
        samples_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Number of clusters (for clustering mode)
        clusters_frame = Theme.create_frame(content)
        clusters_frame.pack(fill="x", pady=Sizes.PAD_S)

        Theme.create_label(clusters_frame, "Number of clusters:").pack(side="left")
        self.scenario_n_clusters_var = ctk.StringVar(value="5")
        clusters_entry = Theme.create_entry(clusters_frame, width=100)
        clusters_entry.configure(textvariable=self.scenario_n_clusters_var)
        clusters_entry.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Validate scenarios
        validate_frame = Theme.create_frame(content)
        validate_frame.pack(fill="x", pady=Sizes.PAD_S)

        self.scenario_validate_var = ctk.BooleanVar(value=True)
        Theme.create_checkbox(
            validate_frame, "Validate scenario significance",
            variable=self.scenario_validate_var
        ).pack(side="left")

    def _load_profiles(self):
        """Load available configuration profiles."""
        profiles = []
        if self.config_dir.exists():
            for f in self.config_dir.glob("*.json"):
                profiles.append(f.stem)

        if not profiles:
            profiles = ["default"]
            self._create_default_profile()

        self.profile_selector.update_profiles(profiles, profiles[0] if profiles else None)
        if profiles:
            self._load_profile(profiles[0])

    def _create_default_profile(self):
        """Create a default profile."""
        default_config = self._get_default_config()
        self._save_profile_to_file("default", default_config)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "trade_classification": {
                "method": "threshold",
                "good_threshold_pct": 0.0,
                "bad_threshold_pct": 0.0,
                "min_holding_period": 1,
                "exclude_partial_exits": False,
                "exclude_stopped_out": False
            },
            "data_alignment": {
                "mode": "strict",
                "lookback_days": 1,
                "min_required_days": 252,
                "prevent_forward_looking": True
            },
            "factor_engineering": {
                "categories": {
                    "technical": True,
                    "value": True,
                    "quality": True,
                    "growth": True,
                    "insider": True,
                    "options": True
                },
                "normalize": True,
                "outlier_method": "winsorize",
                "outlier_threshold": 3.0
            },
            "statistical_analysis": {
                "tier1": {
                    "enabled": True,
                    "correlation_threshold": 0.1
                },
                "tier2": {
                    "enabled": True,
                    "alpha": 0.05,
                    "fdr_correction": True
                },
                "tier3": {
                    "enabled": True,
                    "n_trees": 100,
                    "compute_shap": True
                }
            },
            "scenario_analysis": {
                "enabled": True,
                "mode": "binary",
                "min_samples": 30,
                "n_clusters": 5,
                "validate": True
            }
        }

    def _load_profile(self, profile_name: str):
        """Load a configuration profile."""
        profile_path = self.config_dir / f"{profile_name}.json"

        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    self.config = json.load(f)
                self.current_profile = profile_name
                self._update_ui_from_config()
            except Exception as e:
                show_error(self.root, "Load Error", f"Failed to load profile: {e}")
        else:
            self.config = self._get_default_config()
            self._update_ui_from_config()

    def _update_ui_from_config(self):
        """Update UI widgets from loaded configuration."""
        # Trade classification
        tc = self.config.get("trade_classification", {})
        self.classification_method_var.set(tc.get("method", "threshold"))
        self.good_threshold_var.set(str(tc.get("good_threshold_pct", 0.0)))
        self.bad_threshold_var.set(str(tc.get("bad_threshold_pct", 0.0)))
        self.min_holding_var.set(str(tc.get("min_holding_period", 1)))
        self.exclude_partial_var.set(tc.get("exclude_partial_exits", False))
        self.exclude_stopped_var.set(tc.get("exclude_stopped_out", False))

        # Data alignment
        da = self.config.get("data_alignment", {})
        self.alignment_mode_var.set(da.get("mode", "strict"))
        self.lookback_days_var.set(str(da.get("lookback_days", 1)))
        self.min_required_days_var.set(str(da.get("min_required_days", 252)))
        self.prevent_forward_var.set(da.get("prevent_forward_looking", True))

        # Factor engineering
        fe = self.config.get("factor_engineering", {})
        cats = fe.get("categories", {})
        for cat, var in self.factor_category_vars.items():
            var.set(cats.get(cat, True))
        self.normalize_factors_var.set(fe.get("normalize", True))
        self.outlier_method_var.set(fe.get("outlier_method", "winsorize"))
        self.outlier_threshold_var.set(str(fe.get("outlier_threshold", 3.0)))

        # Statistical analysis
        sa = self.config.get("statistical_analysis", {})
        t1 = sa.get("tier1", {})
        self.tier1_enabled_var.set(t1.get("enabled", True))
        self.tier1_corr_threshold_var.set(str(t1.get("correlation_threshold", 0.1)))

        t2 = sa.get("tier2", {})
        self.tier2_enabled_var.set(t2.get("enabled", True))
        self.tier2_alpha_var.set(str(t2.get("alpha", 0.05)))
        self.tier2_fdr_var.set(t2.get("fdr_correction", True))

        t3 = sa.get("tier3", {})
        self.tier3_enabled_var.set(t3.get("enabled", True))
        self.tier3_n_trees_var.set(str(t3.get("n_trees", 100)))
        self.tier3_shap_var.set(t3.get("compute_shap", True))

        # Scenario analysis
        sc = self.config.get("scenario_analysis", {})
        self.scenario_enabled_var.set(sc.get("enabled", True))
        self.scenario_mode_var.set(sc.get("mode", "binary"))
        self.scenario_min_samples_var.set(str(sc.get("min_samples", 30)))
        self.scenario_n_clusters_var.set(str(sc.get("n_clusters", 5)))
        self.scenario_validate_var.set(sc.get("validate", True))

    def _get_config_from_ui(self) -> Dict[str, Any]:
        """Get configuration from UI widgets."""
        return {
            "trade_classification": {
                "method": self.classification_method_var.get(),
                "good_threshold_pct": float(self.good_threshold_var.get()),
                "bad_threshold_pct": float(self.bad_threshold_var.get()),
                "min_holding_period": int(self.min_holding_var.get()),
                "exclude_partial_exits": self.exclude_partial_var.get(),
                "exclude_stopped_out": self.exclude_stopped_var.get()
            },
            "data_alignment": {
                "mode": self.alignment_mode_var.get(),
                "lookback_days": int(self.lookback_days_var.get()),
                "min_required_days": int(self.min_required_days_var.get()),
                "prevent_forward_looking": self.prevent_forward_var.get()
            },
            "factor_engineering": {
                "categories": {cat: var.get() for cat, var in self.factor_category_vars.items()},
                "normalize": self.normalize_factors_var.get(),
                "outlier_method": self.outlier_method_var.get(),
                "outlier_threshold": float(self.outlier_threshold_var.get())
            },
            "statistical_analysis": {
                "tier1": {
                    "enabled": self.tier1_enabled_var.get(),
                    "correlation_threshold": float(self.tier1_corr_threshold_var.get())
                },
                "tier2": {
                    "enabled": self.tier2_enabled_var.get(),
                    "alpha": float(self.tier2_alpha_var.get()),
                    "fdr_correction": self.tier2_fdr_var.get()
                },
                "tier3": {
                    "enabled": self.tier3_enabled_var.get(),
                    "n_trees": int(self.tier3_n_trees_var.get()),
                    "compute_shap": self.tier3_shap_var.get()
                }
            },
            "scenario_analysis": {
                "enabled": self.scenario_enabled_var.get(),
                "mode": self.scenario_mode_var.get(),
                "min_samples": int(self.scenario_min_samples_var.get()),
                "n_clusters": int(self.scenario_n_clusters_var.get()),
                "validate": self.scenario_validate_var.get()
            }
        }

    def _save_profile_to_file(self, profile_name: str, config: Dict[str, Any]):
        """Save configuration to file."""
        profile_path = self.config_dir / f"{profile_name}.json"
        with open(profile_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _on_profile_change(self, profile_name: str):
        """Handle profile selection change."""
        if self.modified:
            if ask_yes_no(self.root, "Unsaved Changes", "Save changes before switching profiles?"):
                self._on_save()

        self._load_profile(profile_name)

    def _on_create_profile(self):
        """Handle create new profile."""
        dialog = ProfileNameDialog(self.root, "Create Profile", "Enter profile name:")
        name = dialog.get_result()

        if name:
            config = self._get_config_from_ui()
            self._save_profile_to_file(name, config)
            self._load_profiles()
            self.profile_selector.update_profiles(
                [f.stem for f in self.config_dir.glob("*.json")],
                name
            )
            self._load_profile(name)
            show_info(self.root, "Created", f"Profile '{name}' created successfully.")

    def _on_edit_profile(self, profile_name: str):
        """Handle edit profile name."""
        dialog = ProfileNameDialog(
            self.root, "Rename Profile",
            "Enter new name:",
            default=profile_name
        )
        new_name = dialog.get_result()

        if new_name and new_name != profile_name:
            old_path = self.config_dir / f"{profile_name}.json"
            new_path = self.config_dir / f"{new_name}.json"

            if old_path.exists():
                old_path.rename(new_path)
                self._load_profiles()
                self.profile_selector.update_profiles(
                    [f.stem for f in self.config_dir.glob("*.json")],
                    new_name
                )

    def _on_delete_profile(self, profile_name: str):
        """Handle delete profile."""
        if profile_name == "default":
            show_error(self.root, "Cannot Delete", "Cannot delete the default profile.")
            return

        if ask_yes_no(self.root, "Confirm Delete", f"Delete profile '{profile_name}'?"):
            profile_path = self.config_dir / f"{profile_name}.json"
            if profile_path.exists():
                profile_path.unlink()
            self._load_profiles()

    def _on_save(self):
        """Handle save button click."""
        try:
            config = self._get_config_from_ui()

            if self.current_profile:
                self._save_profile_to_file(self.current_profile, config)
                self.modified = False
                show_info(self.root, "Saved", f"Configuration saved to '{self.current_profile}'.")

                if self.on_save:
                    self.on_save(config)
            else:
                self._on_create_profile()
        except ValueError as e:
            show_error(self.root, "Validation Error", str(e))

    def _on_reset(self):
        """Handle reset to defaults."""
        if ask_yes_no(self.root, "Reset", "Reset all settings to defaults?"):
            self.config = self._get_default_config()
            self._update_ui_from_config()
            self.modified = True

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.modified:
            if not ask_yes_no(self.root, "Discard Changes", "Discard unsaved changes?"):
                return
        self.root.destroy()

    def run(self):
        """Run the configuration manager."""
        self.root.mainloop()

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._get_config_from_ui()


class ProfileNameDialog(ctk.CTkToplevel):
    """Dialog for entering profile name."""

    def __init__(self, parent, title: str, prompt: str, default: str = ""):
        super().__init__(parent)

        self.title(title)
        self.geometry("350x150")
        self.resizable(False, False)
        self.configure(fg_color=Colors.BG_DARK)

        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 350) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 150) // 2
        self.geometry(f"+{x}+{y}")

        self._result = None

        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        Theme.create_label(content, prompt).pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.entry = Theme.create_entry(content)
        self.entry.pack(fill="x", pady=Sizes.PAD_S)
        if default:
            self.entry.insert(0, default)
        self.entry.focus()
        self.entry.bind("<Return>", lambda e: self._on_ok())

        btn_frame = Theme.create_frame(content)
        btn_frame.pack(fill="x", pady=(Sizes.PAD_M, 0))

        Theme.create_button(btn_frame, "Cancel", command=self._on_cancel, style="secondary", width=80).pack(side="right")
        Theme.create_button(btn_frame, "OK", command=self._on_ok, width=80).pack(side="right", padx=(0, Sizes.PAD_S))

        self.wait_window()

    def _on_ok(self):
        self._result = self.entry.get().strip()
        self.destroy()

    def _on_cancel(self):
        self._result = None
        self.destroy()

    def get_result(self) -> Optional[str]:
        return self._result


def main():
    """Entry point for Configuration Manager."""
    app = FactorConfigManagerGUI()
    app.run()


if __name__ == "__main__":
    main()
