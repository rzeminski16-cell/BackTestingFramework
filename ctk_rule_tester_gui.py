"""
Rule Tester GUI (CustomTkinter).

A graphical interface for testing the effect of rules on strategy performance.

Features:
- Load multiple trade logs with auto price data loading
- Select 0-3 features from price data
- Build rules with histogram visualization
- Support for entry or exit rule evaluation
- Configurable lookback period (N bars before)
- Compare before/after metrics
- Export filtered trades
"""

import customtkinter as ctk
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import threading
import traceback

from Classes.GUI.ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info
from Classes.RuleTester import (
    Rule, RuleEngine, RuleMode, CompareType, RuleMetricsCalculator,
    extract_ticker_from_filename, load_price_data_for_tickers,
    export_filtered_trades, export_comparison_report,
    StrategyExitRulesRegistry
)
from Classes.FactorAnalysis.data.trade_log_loader import TradeLogLoader

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CTkRuleTesterGUI:
    """Main GUI application for Rule Tester using CustomTkinter."""

    MAX_FEATURES = 3

    def __init__(self):
        """Initialize the GUI."""
        Theme.setup()

        self.root = ctk.CTk()
        self.root.title("Rule Tester - Strategy Filter Analysis")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(fg_color=Colors.BG_DARK)

        # Data storage
        self.trades_df = None
        self.price_data_dict: Dict = {}
        self.rule_engine: Optional[RuleEngine] = None
        self.metrics_calculator = RuleMetricsCalculator()

        # Rule mode (entry or exit)
        self.rule_mode = RuleMode.ENTRY

        # Selected features and rules
        self.selected_features: List[str] = []
        self.rules: List[Rule] = []
        self.feature_widgets: Dict[str, Dict] = {}

        # Results cache
        self.before_metrics: Dict = {}
        self.after_metrics: Dict = {}
        self.comparison: Dict = {}

        # Create UI
        self._create_main_layout()

    def run(self):
        """Start the application."""
        self.root.mainloop()

    def _create_main_layout(self):
        """Create main layout with tabs."""
        # Main container
        main_frame = Theme.create_frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Title row
        title_frame = Theme.create_frame(main_frame)
        title_frame.pack(fill="x", pady=(0, Sizes.PAD_M))

        title_label = ctk.CTkLabel(
            title_frame,
            text="Rule Tester",
            font=Fonts.HEADER_XL,
            text_color=Colors.TEXT_PRIMARY
        )
        title_label.pack(side="left")

        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="  -  Test how rules affect strategy performance",
            font=Fonts.BODY_L,
            text_color=Colors.TEXT_SECONDARY
        )
        subtitle_label.pack(side="left")

        # Entry/Exit Mode Toggle (on the right side of title)
        mode_frame = Theme.create_frame(title_frame)
        mode_frame.pack(side="right")

        ctk.CTkLabel(
            mode_frame,
            text="Rule Mode:",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        self.mode_var = ctk.StringVar(value="Entry")
        self.mode_switch = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Entry", "Exit"],
            variable=self.mode_var,
            command=self._on_mode_changed,
            font=Fonts.BODY_M,
            width=150
        )
        self.mode_switch.pack(side="left")

        # Mode description
        self.mode_desc_label = ctk.CTkLabel(
            mode_frame,
            text="(Rules evaluated at trade entry date)",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        self.mode_desc_label.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Strategy selector frame (shown only in Exit mode)
        self.strategy_frame = Theme.create_frame(main_frame)
        # Don't pack initially - will be shown when Exit mode is selected

        ctk.CTkLabel(
            self.strategy_frame,
            text="Original Strategy:",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        # Get available strategies
        strategy_names = StrategyExitRulesRegistry.get_strategy_names()
        strategy_display = StrategyExitRulesRegistry.get_display_names()
        self._strategy_name_map = {v: k for k, v in strategy_display.items()}  # display_name -> strategy_name
        display_names = list(strategy_display.values())

        self.strategy_var = ctk.StringVar(value=display_names[0] if display_names else "")
        self.strategy_menu = ctk.CTkOptionMenu(
            self.strategy_frame,
            values=display_names,
            variable=self.strategy_var,
            command=self._on_strategy_changed,
            width=250,
            font=Fonts.BODY_M
        )
        self.strategy_menu.pack(side="left", padx=(0, Sizes.PAD_M))

        # Strategy description
        self.strategy_desc_label = ctk.CTkLabel(
            self.strategy_frame,
            text="",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED,
            wraplength=600
        )
        self.strategy_desc_label.pack(side="left")

        # Tabview
        self.tabview = ctk.CTkTabview(
            main_frame,
            fg_color=Colors.SURFACE,
            segmented_button_fg_color=Colors.BG_MEDIUM,
            segmented_button_selected_color=Colors.PRIMARY,
            segmented_button_unselected_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_L
        )
        self.tabview.pack(fill="both", expand=True)

        # Create tabs
        self.tab_data = self.tabview.add("1. Load Data")
        self.tab_features = self.tabview.add("2. Select Features")
        self.tab_rules = self.tabview.add("3. Build Rules")
        self.tab_results = self.tabview.add("4. Results")

        self._create_data_tab()
        self._create_features_tab()
        self._create_rules_tab()
        self._create_results_tab()

    def _on_mode_changed(self, value: str):
        """Handle mode change between Entry and Exit."""
        self.rule_mode = RuleMode.ENTRY if value == "Entry" else RuleMode.EXIT

        if value == "Entry":
            self.mode_desc_label.configure(text="(Rules evaluated at trade entry date)")
            # Hide strategy selector for entry mode
            self.strategy_frame.pack_forget()
        else:
            self.mode_desc_label.configure(text="(Rules evaluated at trade exit date)")
            # Show strategy selector for exit mode
            self.strategy_frame.pack(fill="x", pady=(0, Sizes.PAD_S), before=self.tabview)
            # Update strategy description
            self._on_strategy_changed(self.strategy_var.get())

        # Update rule engine mode if it exists
        if self.rule_engine:
            self.rule_engine.set_mode(self.rule_mode)
            # Set strategy if in exit mode
            if self.rule_mode == RuleMode.EXIT:
                self._on_strategy_changed(self.strategy_var.get())
            # Refresh feature list since statistics might differ
            self._populate_features_list()

    def _on_strategy_changed(self, display_name: str):
        """Handle strategy selection change."""
        # Get the actual strategy name from display name
        strategy_name = self._strategy_name_map.get(display_name, display_name)

        # Update the rule engine with the selected strategy
        if self.rule_engine:
            self.rule_engine.set_strategy(strategy_name)

        # Get strategy config and show description
        config = StrategyExitRulesRegistry.get(strategy_name)
        if config and config.exit_rules:
            rules_desc = ", ".join([r.description for r in config.exit_rules[:3]])
            if len(config.exit_rules) > 3:
                rules_desc += f", ... (+{len(config.exit_rules) - 3} more)"
            self.strategy_desc_label.configure(
                text=f"Exit rules: {rules_desc}"
            )
        elif config and not config.exit_rules:
            self.strategy_desc_label.configure(
                text="No predefined exit rules - only user-defined rules will apply"
            )
        else:
            self.strategy_desc_label.configure(text="")

    # =========================================================================
    # TAB 1: DATA LOADING
    # =========================================================================

    def _create_data_tab(self):
        """Create the data loading tab."""
        tab = self.tab_data

        # Main scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Instructions
        instructions = Theme.create_card(scroll_frame)
        instructions.pack(fill="x", pady=(0, Sizes.PAD_M))

        inst_content = Theme.create_frame(instructions)
        inst_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        ctk.CTkLabel(
            inst_content,
            text="Step 1: Load Trade Logs",
            font=Fonts.HEADER_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        ctk.CTkLabel(
            inst_content,
            text="Select one or more trade log CSV files. Price data will be automatically loaded from raw_data/daily/{ticker}_daily.csv",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY,
            wraplength=800
        ).pack(anchor="w", pady=(Sizes.PAD_S, 0))

        # Load button
        btn_frame = Theme.create_frame(scroll_frame)
        btn_frame.pack(fill="x", pady=(0, Sizes.PAD_M))

        Theme.create_button(
            btn_frame,
            "Load Trade Logs...",
            command=self._load_trade_logs,
            width=200
        ).pack(side="left", padx=(0, Sizes.PAD_M))

        self.load_status_label = ctk.CTkLabel(
            btn_frame,
            text="No data loaded",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY
        )
        self.load_status_label.pack(side="left")

        # Data summary card
        self.summary_card = Theme.create_card(scroll_frame)
        self.summary_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        summary_content = Theme.create_frame(self.summary_card)
        summary_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        ctk.CTkLabel(
            summary_content,
            text="Data Summary",
            font=Fonts.HEADER_S,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.summary_text = ctk.CTkTextbox(
            summary_content,
            height=200,
            font=Fonts.MONO,
            fg_color=Colors.BG_MEDIUM,
            text_color=Colors.TEXT_PRIMARY
        )
        self.summary_text.pack(fill="x")
        self.summary_text.insert("1.0", "Load trade logs to see summary...")
        self.summary_text.configure(state="disabled")

        # Continue button
        continue_frame = Theme.create_frame(scroll_frame)
        continue_frame.pack(fill="x")

        self.continue_to_features_btn = Theme.create_button(
            continue_frame,
            "Continue to Feature Selection ->",
            command=lambda: self.tabview.set("2. Select Features"),
            width=250,
            state="disabled"
        )
        self.continue_to_features_btn.pack(side="right")

    def _load_trade_logs(self):
        """Open file dialog and load trade logs."""
        from tkinter import filedialog

        files = filedialog.askopenfilenames(
            title="Select Trade Log Files",
            initialdir="logs",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not files:
            return

        self.load_status_label.configure(text="Loading...", text_color=Colors.WARNING)
        self.root.update()

        try:
            # Load trade logs
            loader = TradeLogLoader()
            self.trades_df, validations, metadata_list = loader.load_multiple(
                list(files),
                validate=True,
                combine=True
            )

            if self.trades_df is None or len(self.trades_df) == 0:
                show_error(self.root, "Error", "No valid trades found in selected files.")
                return

            # Extract tickers from filenames or trade data
            tickers = set()
            for f in files:
                ticker = extract_ticker_from_filename(f)
                if ticker:
                    tickers.add(ticker)

            # Also try from symbol column
            if 'symbol' in self.trades_df.columns:
                tickers.update(self.trades_df['symbol'].str.upper().unique())

            # Load price data
            self.price_data_dict = load_price_data_for_tickers(
                list(tickers),
                base_path="raw_data/daily"
            )

            # Initialize rule engine with current mode
            self.rule_engine = RuleEngine(self.trades_df, self.price_data_dict, self.rule_mode)

            # Set strategy if in exit mode
            if self.rule_mode == RuleMode.EXIT:
                strategy_name = self._strategy_name_map.get(self.strategy_var.get(), self.strategy_var.get())
                self.rule_engine.set_strategy(strategy_name)

            # Update summary
            self._update_data_summary(files, tickers)

            # Enable continue button
            self.continue_to_features_btn.configure(state="normal")
            self.load_status_label.configure(
                text=f"Loaded {len(self.trades_df)} trades",
                text_color=Colors.SUCCESS
            )

            # Populate features tab
            self._populate_features_list()

        except Exception as e:
            show_error(self.root, "Error Loading Data", str(e))
            self.load_status_label.configure(text="Error loading data", text_color=Colors.ERROR)
            traceback.print_exc()

    def _update_data_summary(self, files: tuple, tickers: set):
        """Update the data summary display."""
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")

        lines = []
        lines.append(f"Trade Logs Loaded: {len(files)}")
        for f in files:
            lines.append(f"  - {Path(f).name}")

        lines.append("")
        lines.append(f"Total Trades: {len(self.trades_df)}")

        if 'entry_date' in self.trades_df.columns:
            date_min = self.trades_df['entry_date'].min()
            date_max = self.trades_df['entry_date'].max()
            lines.append(f"Date Range: {str(date_min)[:10]} to {str(date_max)[:10]}")

        lines.append("")
        lines.append(f"Tickers Found: {len(tickers)}")
        for t in sorted(tickers):
            has_price = t in self.price_data_dict
            status = "OK" if has_price else "MISSING"
            lines.append(f"  - {t}: Price data {status}")

        lines.append("")
        lines.append(f"Price Data Loaded: {len(self.price_data_dict)} tickers")

        # Show available features
        if self.rule_engine:
            features = self.rule_engine.get_available_features()
            lines.append("")
            lines.append(f"Available Features: {len(features)}")
            for f in features[:10]:
                lines.append(f"  - {f}")
            if len(features) > 10:
                lines.append(f"  ... and {len(features) - 10} more")

        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")

    # =========================================================================
    # TAB 2: FEATURE SELECTION
    # =========================================================================

    def _create_features_tab(self):
        """Create the feature selection tab."""
        tab = self.tab_features

        # Main frame
        main_frame = Theme.create_frame(tab)
        main_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Instructions
        instructions = Theme.create_card(main_frame)
        instructions.pack(fill="x", pady=(0, Sizes.PAD_M))

        inst_content = Theme.create_frame(instructions)
        inst_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        ctk.CTkLabel(
            inst_content,
            text="Step 2: Select Features (0-3)",
            font=Fonts.HEADER_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w")

        ctk.CTkLabel(
            inst_content,
            text="Select up to 3 features from the price data to build rules around. Features are evaluated at each trade's entry/exit date (based on mode).",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY,
            wraplength=800
        ).pack(anchor="w", pady=(Sizes.PAD_S, 0))

        # Feature list
        self.features_scroll = ctk.CTkScrollableFrame(
            main_frame,
            fg_color=Colors.BG_MEDIUM,
            corner_radius=Sizes.RADIUS_M,
            height=400
        )
        self.features_scroll.pack(fill="both", expand=True, pady=(0, Sizes.PAD_M))

        self.features_placeholder = ctk.CTkLabel(
            self.features_scroll,
            text="Load trade logs first to see available features.",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY
        )
        self.features_placeholder.pack(pady=Sizes.PAD_XL)

        # Selection summary
        summary_frame = Theme.create_frame(main_frame)
        summary_frame.pack(fill="x", pady=(0, Sizes.PAD_M))

        self.selection_label = ctk.CTkLabel(
            summary_frame,
            text="Selected: 0 features",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        )
        self.selection_label.pack(side="left")

        # Buttons
        btn_frame = Theme.create_frame(main_frame)
        btn_frame.pack(fill="x")

        Theme.create_button(
            btn_frame,
            "<- Back to Data",
            command=lambda: self.tabview.set("1. Load Data"),
            style="secondary",
            width=150
        ).pack(side="left")

        self.continue_to_rules_btn = Theme.create_button(
            btn_frame,
            "Continue to Rule Builder ->",
            command=self._go_to_rules_tab,
            width=200
        )
        self.continue_to_rules_btn.pack(side="right")

        # Store feature checkboxes
        self.feature_checkboxes: Dict[str, ctk.CTkCheckBox] = {}

    def _populate_features_list(self):
        """Populate the features list from loaded data."""
        if not self.rule_engine:
            return

        # Clear existing widgets
        for widget in self.features_scroll.winfo_children():
            widget.destroy()

        # Get available features
        features = self.rule_engine.get_available_features()

        self.feature_checkboxes.clear()

        if not features:
            ctk.CTkLabel(
                self.features_scroll,
                text="No features available in price data.",
                font=Fonts.BODY_M,
                text_color=Colors.TEXT_SECONDARY
            ).pack(pady=Sizes.PAD_XL)
            return

        # Create checkbox for each feature
        for feature in features:
            feature_type = self.rule_engine.detect_feature_type(feature)
            type_label = f"[{feature_type[:4]}]"

            frame = Theme.create_frame(self.features_scroll)
            frame.pack(fill="x", padx=Sizes.PAD_S, pady=Sizes.PAD_XS)

            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                frame,
                text=f"{feature}  {type_label}",
                variable=var,
                command=self._on_feature_selection_changed,
                font=Fonts.BODY_M,
                text_color=Colors.TEXT_PRIMARY
            )
            cb.pack(side="left")

            self.feature_checkboxes[feature] = cb
            cb._variable = var

    def _on_feature_selection_changed(self):
        """Handle feature selection changes."""
        selected = []
        for feature, cb in self.feature_checkboxes.items():
            if cb._variable.get():
                selected.append(feature)

        # Enforce max features
        if len(selected) > self.MAX_FEATURES:
            # Uncheck the last one
            for feature, cb in self.feature_checkboxes.items():
                if cb._variable.get() and feature not in self.selected_features:
                    cb._variable.set(False)
                    break
            selected = selected[:self.MAX_FEATURES]

        self.selected_features = selected
        self.selection_label.configure(text=f"Selected: {len(selected)} feature(s)")

    def _go_to_rules_tab(self):
        """Navigate to rules tab and set up rule builders."""
        self._setup_rule_builders()
        self.tabview.set("3. Build Rules")

    # =========================================================================
    # TAB 3: RULE BUILDER
    # =========================================================================

    def _create_rules_tab(self):
        """Create the rule builder tab."""
        tab = self.tab_rules

        # Main scrollable frame
        self.rules_scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        self.rules_scroll.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Placeholder
        self.rules_placeholder = ctk.CTkLabel(
            self.rules_scroll,
            text="Select features first to build rules.",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY
        )
        self.rules_placeholder.pack(pady=Sizes.PAD_XL)

        # Bottom buttons (outside scroll)
        btn_frame = Theme.create_frame(tab)
        btn_frame.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_button(
            btn_frame,
            "<- Back to Features",
            command=lambda: self.tabview.set("2. Select Features"),
            style="secondary",
            width=150
        ).pack(side="left")

        # Preview
        self.preview_label = ctk.CTkLabel(
            btn_frame,
            text="",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY
        )
        self.preview_label.pack(side="left", padx=Sizes.PAD_XL)

        self.apply_rules_btn = Theme.create_button(
            btn_frame,
            "Apply Rules & Show Results ->",
            command=self._apply_rules_and_show_results,
            width=250
        )
        self.apply_rules_btn.pack(side="right")

    def _setup_rule_builders(self):
        """Set up rule builder widgets for each selected feature."""
        # Clear existing
        for widget in self.rules_scroll.winfo_children():
            widget.destroy()

        self.feature_widgets.clear()
        self.rules.clear()

        # Mode indicator at top
        mode_indicator = Theme.create_card(self.rules_scroll)
        mode_indicator.pack(fill="x", pady=(0, Sizes.PAD_M))

        mode_content = Theme.create_frame(mode_indicator)
        mode_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)

        mode_text = "ENTRY" if self.rule_mode == RuleMode.ENTRY else "EXIT"
        ctk.CTkLabel(
            mode_content,
            text=f"Mode: {mode_text} Rules",
            font=Fonts.HEADER_S,
            text_color=Colors.PRIMARY_LIGHT
        ).pack(side="left")

        ctk.CTkLabel(
            mode_content,
            text=f"  (Features evaluated at trade {mode_text.lower()} date)",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        ).pack(side="left")

        if not self.selected_features:
            # No features selected - skip rules
            no_filter_label = ctk.CTkLabel(
                self.rules_scroll,
                text="No features selected. Click 'Apply Rules' to analyze all trades without filtering.",
                font=Fonts.BODY_M,
                text_color=Colors.TEXT_SECONDARY
            )
            no_filter_label.pack(pady=Sizes.PAD_XL)
            self._update_preview()
            return

        for feature in self.selected_features:
            self._create_rule_builder_card(feature)

        self._update_preview()

    def _create_rule_builder_card(self, feature: str):
        """Create a rule builder card for a feature."""
        if not self.rule_engine:
            return

        card = Theme.create_card(self.rules_scroll)
        card.pack(fill="x", pady=(0, Sizes.PAD_M))

        content = Theme.create_frame(card)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Header row
        header_frame = Theme.create_frame(content)
        header_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        feature_type = self.rule_engine.detect_feature_type(feature)
        type_label = "Continuous" if feature_type == 'continuous' else "Discrete"

        ctk.CTkLabel(
            header_frame,
            text=f"{feature}",
            font=Fonts.HEADER_S,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")

        ctk.CTkLabel(
            header_frame,
            text=f"  ({type_label})",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left")

        # Enable checkbox
        enable_var = ctk.BooleanVar(value=True)
        enable_cb = ctk.CTkCheckBox(
            header_frame,
            text="Enable rule",
            variable=enable_var,
            command=self._update_preview,
            font=Fonts.BODY_S
        )
        enable_cb.pack(side="right")

        # Get feature statistics
        stats = self.rule_engine.get_feature_statistics(feature)

        # Histogram (if matplotlib available)
        if HAS_MATPLOTLIB and HAS_NUMPY and feature_type == 'continuous':
            self._create_histogram(content, feature)

        # Rule controls frame
        controls_frame = Theme.create_frame(content)
        controls_frame.pack(fill="x", pady=Sizes.PAD_S)

        if feature_type == 'continuous':
            widgets = self._create_continuous_controls(controls_frame, feature, stats)
        else:
            widgets = self._create_discrete_controls(controls_frame, feature, stats)

        widgets['enable_var'] = enable_var
        widgets['feature_type'] = feature_type
        self.feature_widgets[feature] = widgets

    def _create_histogram(self, parent, feature: str):
        """Create histogram for a continuous feature."""
        values = self.rule_engine.get_feature_values_at_reference(feature)
        if values is None or len(values) == 0:
            return

        values_clean = values.dropna()
        if len(values_clean) == 0:
            return

        # Create figure
        fig = Figure(figsize=(8, 2.5), dpi=100, facecolor=Colors.BG_MEDIUM)
        ax = fig.add_subplot(111)
        ax.set_facecolor(Colors.BG_MEDIUM)

        # Plot histogram
        ax.hist(values_clean, bins=30, color=Colors.PRIMARY_LIGHT, edgecolor=Colors.BG_DARK, alpha=0.8)

        ax.set_xlabel(feature, color=Colors.TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel('Count', color=Colors.TEXT_SECONDARY, fontsize=9)
        ax.tick_params(colors=Colors.TEXT_SECONDARY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(Colors.BORDER)

        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", pady=Sizes.PAD_S)

    def _create_continuous_controls(self, parent, feature: str, stats: Dict) -> Dict:
        """Create controls for continuous feature."""
        widgets = {}

        # Row 1: Operator and Lookback
        row1_frame = Theme.create_frame(parent)
        row1_frame.pack(fill="x", pady=Sizes.PAD_XS)

        # Operator selection
        ctk.CTkLabel(
            row1_frame,
            text="Operator:",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        operators = ['>', '<', '>=', '<=', 'between']
        op_var = ctk.StringVar(value='>')
        op_menu = ctk.CTkOptionMenu(
            row1_frame,
            values=operators,
            variable=op_var,
            command=lambda _: self._on_operator_changed(feature),
            width=120
        )
        op_menu.pack(side="left", padx=(0, Sizes.PAD_L))
        widgets['operator'] = op_var

        # Lookback bars (N)
        ctk.CTkLabel(
            row1_frame,
            text="Lookback (N):",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        lookback_var = ctk.StringVar(value="0")
        lookback_entry = ctk.CTkEntry(
            row1_frame,
            textvariable=lookback_var,
            width=60,
            font=Fonts.BODY_M
        )
        lookback_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        lookback_entry.bind('<KeyRelease>', lambda e: self._update_preview())
        widgets['lookback'] = lookback_var

        ctk.CTkLabel(
            row1_frame,
            text="(0 = ref date only, N>0 = within N bars before)",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        ).pack(side="left")

        # Row 2: Compare To selector and Value/Feature inputs
        value_frame = Theme.create_frame(parent)
        value_frame.pack(fill="x", pady=Sizes.PAD_XS)

        # Compare To selector (Value or Feature)
        ctk.CTkLabel(
            value_frame,
            text="Compare to:",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        compare_type_var = ctk.StringVar(value="Value")
        compare_type_menu = ctk.CTkSegmentedButton(
            value_frame,
            values=["Value", "Feature"],
            variable=compare_type_var,
            command=lambda _: self._on_compare_type_changed(feature),
            font=Fonts.BODY_S,
            width=140
        )
        compare_type_menu.pack(side="left", padx=(0, Sizes.PAD_M))
        widgets['compare_type'] = compare_type_var

        # Value input frame (shown when "Value" is selected)
        value_input_frame = Theme.create_frame(value_frame)
        value_input_frame.pack(side="left", padx=(0, Sizes.PAD_S))
        widgets['value_input_frame'] = value_input_frame

        # Primary value
        default_val = stats.get('median', 0) if stats else 0
        value1_var = ctk.StringVar(value=str(round(default_val, 4)))
        value1_entry = ctk.CTkEntry(
            value_input_frame,
            textvariable=value1_var,
            width=100,
            font=Fonts.BODY_M
        )
        value1_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        value1_entry.bind('<KeyRelease>', lambda e: self._update_preview())
        widgets['value1'] = value1_var
        widgets['value1_entry'] = value1_entry

        # Second value (for between) - created but hidden initially
        between_label = ctk.CTkLabel(
            value_input_frame,
            text="and",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        )
        widgets['between_label'] = between_label

        value2_var = ctk.StringVar(value=str(round(stats.get('q75', 0), 4)) if stats else "0")
        value2_entry = ctk.CTkEntry(
            value_input_frame,
            textvariable=value2_var,
            width=100,
            font=Fonts.BODY_M
        )
        value2_entry.bind('<KeyRelease>', lambda e: self._update_preview())
        widgets['value2'] = value2_var
        widgets['value2_entry'] = value2_entry

        # Feature selector frame (hidden initially, shown when "Feature" is selected)
        feature_input_frame = Theme.create_frame(value_frame)
        widgets['feature_input_frame'] = feature_input_frame
        # Don't pack initially - will be shown/hidden by _on_compare_type_changed

        # Get all comparable features (other features + price columns)
        comparable_features = self.rule_engine.get_all_comparable_features() if self.rule_engine else []
        # Remove the current feature from the list
        comparable_features = [f for f in comparable_features if f != feature]

        compare_feature_var = ctk.StringVar(value=comparable_features[0] if comparable_features else "")
        if comparable_features:
            compare_feature_menu = ctk.CTkOptionMenu(
                feature_input_frame,
                values=comparable_features,
                variable=compare_feature_var,
                command=lambda _: self._update_preview(),
                width=150
            )
            compare_feature_menu.pack(side="left")
        else:
            ctk.CTkLabel(
                feature_input_frame,
                text="No other features available",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED
            ).pack(side="left")
        widgets['compare_feature'] = compare_feature_var

        # Stats display
        if stats:
            stats_text = f"Min: {stats.get('min', 'N/A'):.2f}  Max: {stats.get('max', 'N/A'):.2f}  Mean: {stats.get('mean', 'N/A'):.2f}"
            ctk.CTkLabel(
                value_frame,
                text=stats_text,
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED
            ).pack(side="right")

        return widgets

    def _create_discrete_controls(self, parent, feature: str, stats: Dict) -> Dict:
        """Create controls for discrete feature."""
        widgets = {}

        # Row 1: Operator and Lookback
        row1_frame = Theme.create_frame(parent)
        row1_frame.pack(fill="x", pady=Sizes.PAD_XS)

        ctk.CTkLabel(
            row1_frame,
            text="Operator:",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        operators = ['==', '!=', 'in']
        op_var = ctk.StringVar(value='==')
        op_menu = ctk.CTkOptionMenu(
            row1_frame,
            values=operators,
            variable=op_var,
            command=lambda _: self._on_operator_changed(feature),
            width=120
        )
        op_menu.pack(side="left", padx=(0, Sizes.PAD_L))
        widgets['operator'] = op_var

        # Lookback bars (N)
        ctk.CTkLabel(
            row1_frame,
            text="Lookback (N):",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        lookback_var = ctk.StringVar(value="0")
        lookback_entry = ctk.CTkEntry(
            row1_frame,
            textvariable=lookback_var,
            width=60,
            font=Fonts.BODY_M
        )
        lookback_entry.pack(side="left", padx=(0, Sizes.PAD_S))
        lookback_entry.bind('<KeyRelease>', lambda e: self._update_preview())
        widgets['lookback'] = lookback_var

        ctk.CTkLabel(
            row1_frame,
            text="(0 = ref date only, N>0 = within N bars before)",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        ).pack(side="left")

        # Row 2: Value selection
        value_frame = Theme.create_frame(parent)
        value_frame.pack(fill="x", pady=Sizes.PAD_XS)

        unique_values = stats.get('unique_values', [])
        str_values = [str(v) for v in unique_values]

        ctk.CTkLabel(
            value_frame,
            text="Value:",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        if str_values:
            value_var = ctk.StringVar(value=str_values[0])
            value_menu = ctk.CTkOptionMenu(
                value_frame,
                values=str_values,
                variable=value_var,
                command=lambda _: self._update_preview(),
                width=150
            )
            value_menu.pack(side="left")
            widgets['value1'] = value_var
            widgets['unique_values'] = unique_values
        else:
            value_var = ctk.StringVar(value="")
            value_entry = ctk.CTkEntry(
                value_frame,
                textvariable=value_var,
                width=150,
                font=Fonts.BODY_M
            )
            value_entry.pack(side="left")
            value_entry.bind('<KeyRelease>', lambda e: self._update_preview())
            widgets['value1'] = value_var

        return widgets

    def _on_operator_changed(self, feature: str):
        """Handle operator change for a feature."""
        widgets = self.feature_widgets.get(feature)
        if not widgets:
            return

        op = widgets['operator'].get()

        # Show/hide second value for 'between'
        if widgets.get('feature_type') == 'continuous':
            # For 'between', force Value mode and show second value
            if op == 'between':
                # Force to Value mode for between operator
                widgets.get('compare_type', ctk.StringVar(value="Value")).set("Value")
                self._on_compare_type_changed(feature)
                widgets.get('between_label').pack(side="left", padx=Sizes.PAD_XS)
                widgets.get('value2_entry').pack(side="left")
            else:
                widgets.get('between_label').pack_forget()
                widgets.get('value2_entry').pack_forget()

        self._update_preview()

    def _on_compare_type_changed(self, feature: str):
        """Handle compare type change (Value vs Feature) for a feature."""
        widgets = self.feature_widgets.get(feature)
        if not widgets:
            return

        compare_type = widgets.get('compare_type', ctk.StringVar(value="Value")).get()
        operator = widgets.get('operator', ctk.StringVar(value="")).get()

        # For 'between' operator, always show value inputs
        if operator == 'between':
            widgets.get('value_input_frame').pack(side="left", padx=(0, Sizes.PAD_S))
            widgets.get('feature_input_frame').pack_forget()
            return

        if compare_type == "Value":
            # Show value input, hide feature selector
            widgets.get('value_input_frame').pack(side="left", padx=(0, Sizes.PAD_S))
            widgets.get('feature_input_frame').pack_forget()
        else:
            # Show feature selector, hide value input
            widgets.get('value_input_frame').pack_forget()
            widgets.get('feature_input_frame').pack(side="left", padx=(0, Sizes.PAD_S))

        self._update_preview()

    def _update_preview(self):
        """Update the preview of trades passing rules."""
        self._build_rules_from_widgets()

        if not self.rule_engine:
            return

        preview = self.rule_engine.get_rule_preview(self.rules)
        total = preview['total_trades']
        mode = preview.get('mode', 'entry').upper()

        if mode == 'ENTRY':
            passing = preview['passing_trades']
            pct = preview['pass_rate']
            self.preview_label.configure(
                text=f"Preview ({mode}): {passing} of {total} trades pass ({pct:.1f}%)"
            )
        else:
            # Exit mode - show modified exits count
            modified = preview.get('modified_exits', 0)
            self.preview_label.configure(
                text=f"Preview ({mode}): {modified} of {total} trades would have earlier exits"
            )

    def _build_rules_from_widgets(self):
        """Build Rule objects from widget values."""
        self.rules = []

        for feature, widgets in self.feature_widgets.items():
            if not widgets.get('enable_var', ctk.BooleanVar(value=True)).get():
                continue

            op = widgets['operator'].get()
            feature_type = widgets.get('feature_type', 'continuous')

            try:
                # Get lookback value
                lookback_str = widgets.get('lookback', ctk.StringVar(value="0")).get()
                lookback = int(lookback_str) if lookback_str.isdigit() else 0
                lookback = max(0, lookback)  # Ensure non-negative

                # Determine compare type and value
                compare_type = CompareType.VALUE
                compare_feature = None
                value = None

                if feature_type == 'continuous':
                    # Check if comparing to another feature (only for non-between operators)
                    compare_type_str = widgets.get('compare_type', ctk.StringVar(value="Value")).get()

                    if op == 'between':
                        # Between always uses static values
                        val1 = float(widgets['value1'].get())
                        val2 = float(widgets['value2'].get())
                        value = (min(val1, val2), max(val1, val2))
                        compare_type = CompareType.VALUE
                    elif compare_type_str == "Feature":
                        # Compare to another feature
                        compare_type = CompareType.FEATURE
                        compare_feature = widgets.get('compare_feature', ctk.StringVar(value="")).get()
                        value = None  # Not used when comparing to feature
                    else:
                        # Compare to static value
                        value = float(widgets['value1'].get())
                        compare_type = CompareType.VALUE
                else:
                    # Discrete features always use static values
                    if op == 'in':
                        # Parse comma-separated values
                        raw = widgets['value1'].get()
                        value = [v.strip() for v in raw.split(',')]
                    else:
                        value = widgets['value1'].get()

                rule = Rule(
                    feature=feature,
                    operator=op,
                    value=value,
                    compare_type=compare_type,
                    compare_feature=compare_feature,
                    lookback_bars=lookback,
                    is_discrete=(feature_type == 'discrete')
                )
                self.rules.append(rule)

            except (ValueError, KeyError):
                pass  # Skip invalid rules

    def _apply_rules_and_show_results(self):
        """Apply rules and navigate to results tab."""
        self._build_rules_from_widgets()
        self._calculate_results()
        self.tabview.set("4. Results")

    # =========================================================================
    # TAB 4: RESULTS
    # =========================================================================

    def _create_results_tab(self):
        """Create the results tab."""
        tab = self.tab_results

        # Main scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Comparison table card
        table_card = Theme.create_card(scroll_frame)
        table_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        table_content = Theme.create_frame(table_card)
        table_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        ctk.CTkLabel(
            table_content,
            text="Before/After Comparison",
            font=Fonts.HEADER_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        self.results_text = ctk.CTkTextbox(
            table_content,
            height=280,
            font=Fonts.MONO,
            fg_color=Colors.BG_MEDIUM,
            text_color=Colors.TEXT_PRIMARY
        )
        self.results_text.pack(fill="x")
        self.results_text.insert("1.0", "Apply rules to see results...")
        self.results_text.configure(state="disabled")

        # Warning label
        self.warning_label = ctk.CTkLabel(
            table_content,
            text="",
            font=Fonts.BODY_M,
            text_color=Colors.WARNING
        )
        self.warning_label.pack(anchor="w", pady=(Sizes.PAD_S, 0))

        # Charts card
        charts_card = Theme.create_card(scroll_frame)
        charts_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        charts_content = Theme.create_frame(charts_card)
        charts_content.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        ctk.CTkLabel(
            charts_content,
            text="Performance Charts",
            font=Fonts.HEADER_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        # Chart buttons
        chart_btn_frame = Theme.create_frame(charts_content)
        chart_btn_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        Theme.create_button(
            chart_btn_frame,
            "Cumulative P/L",
            command=self._show_cumulative_pl_chart,
            width=150
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        Theme.create_button(
            chart_btn_frame,
            "Cumulative RAR%",
            command=self._show_cumulative_rar_chart,
            width=150
        ).pack(side="left")

        # Chart container
        self.chart_frame = Theme.create_frame(charts_content)
        self.chart_frame.pack(fill="both", expand=True)

        # Export card
        export_card = Theme.create_card(scroll_frame)
        export_card.pack(fill="x", pady=(0, Sizes.PAD_M))

        export_content = Theme.create_frame(export_card)
        export_content.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        ctk.CTkLabel(
            export_content,
            text="Export",
            font=Fonts.HEADER_M,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_S))

        export_btn_frame = Theme.create_frame(export_content)
        export_btn_frame.pack(fill="x")

        Theme.create_button(
            export_btn_frame,
            "Export Filtered Trades",
            command=self._export_filtered_trades,
            width=180
        ).pack(side="left", padx=(0, Sizes.PAD_S))

        Theme.create_button(
            export_btn_frame,
            "Export Comparison Report",
            command=self._export_report,
            width=180
        ).pack(side="left")

        self.export_status_label = ctk.CTkLabel(
            export_btn_frame,
            text="",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        )
        self.export_status_label.pack(side="left", padx=Sizes.PAD_M)

        # Back button
        back_frame = Theme.create_frame(scroll_frame)
        back_frame.pack(fill="x")

        Theme.create_button(
            back_frame,
            "<- Back to Rule Builder",
            command=lambda: self.tabview.set("3. Build Rules"),
            style="secondary",
            width=180
        ).pack(side="left")

    def _calculate_results(self):
        """Calculate before/after metrics."""
        if not self.rule_engine or self.trades_df is None:
            return

        # Before metrics (all trades)
        self.before_metrics = self.metrics_calculator.calculate_metrics(self.trades_df)

        # Apply rules
        filtered_df = self.rule_engine.apply_rules(self.rules)

        # After metrics (filtered trades)
        self.after_metrics = self.metrics_calculator.calculate_metrics(filtered_df)

        # Compare
        self.comparison = self.metrics_calculator.compare_metrics(
            self.before_metrics,
            self.after_metrics
        )

        # Update display
        self._update_results_display()

    def _update_results_display(self):
        """Update the results display."""
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")

        # Build table
        lines = []
        mode_text = "ENTRY" if self.rule_mode == RuleMode.ENTRY else "EXIT"
        lines.append(f"Rule Mode: {mode_text}")
        lines.append("")
        lines.append(f"{'Metric':<25} {'Before':>15} {'After':>15} {'Change':>20}")
        lines.append("-" * 75)

        table_data = self.metrics_calculator.get_comparison_table_data(self.comparison)
        for label, before, after, change in table_data:
            lines.append(f"{label:<25} {before:>15} {after:>15} {change:>20}")

        lines.append("")
        lines.append("Rules Applied:")
        if self.rules:
            for i, rule in enumerate(self.rules, 1):
                lines.append(f"  {i}. {rule}")
        else:
            lines.append("  (No rules - showing all trades)")

        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.configure(state="disabled")

        # Update warning
        warnings = []
        if self.comparison.get('after_warning'):
            warnings.append(self.comparison['after_warning'])
        self.warning_label.configure(text="\n".join(warnings))

    def _show_cumulative_pl_chart(self):
        """Show cumulative P/L chart."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            show_error(self.root, "Error", "Matplotlib is required for charts.")
            return

        # Clear chart frame
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        before_pl = self.before_metrics.get('cumulative_pl')
        after_pl = self.after_metrics.get('cumulative_pl')

        if before_pl is None or len(before_pl) == 0:
            return

        fig = Figure(figsize=(10, 4), dpi=100, facecolor=Colors.BG_MEDIUM)
        ax = fig.add_subplot(111)
        ax.set_facecolor(Colors.BG_MEDIUM)

        # Plot before
        if len(before_pl) > 0:
            ax.plot(before_pl['date'], before_pl['cumulative_pl'],
                    color=Colors.TEXT_SECONDARY, label='Before (All Trades)',
                    linewidth=1.5, alpha=0.7)

        # Plot after
        if after_pl is not None and len(after_pl) > 0:
            ax.plot(after_pl['date'], after_pl['cumulative_pl'],
                    color=Colors.PRIMARY_LIGHT, label='After (Filtered)',
                    linewidth=2)

        ax.set_xlabel('Date', color=Colors.TEXT_SECONDARY)
        ax.set_ylabel('Cumulative P/L ($)', color=Colors.TEXT_SECONDARY)
        ax.set_title('Cumulative P/L Comparison', color=Colors.TEXT_PRIMARY)
        ax.legend(loc='upper left', facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                  labelcolor=Colors.TEXT_PRIMARY)
        ax.tick_params(colors=Colors.TEXT_SECONDARY)
        ax.grid(True, color=Colors.BORDER, alpha=0.3)

        for spine in ax.spines.values():
            spine.set_color(Colors.BORDER)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _show_cumulative_rar_chart(self):
        """Show cumulative RAR% chart."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            show_error(self.root, "Error", "Matplotlib is required for charts.")
            return

        # Clear chart frame
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        before_rar = self.before_metrics.get('cumulative_rar')
        after_rar = self.after_metrics.get('cumulative_rar')

        if before_rar is None or len(before_rar) == 0:
            ctk.CTkLabel(
                self.chart_frame,
                text="Not enough data to show cumulative RAR% chart.",
                font=Fonts.BODY_M,
                text_color=Colors.TEXT_SECONDARY
            ).pack(pady=Sizes.PAD_XL)
            return

        fig = Figure(figsize=(10, 4), dpi=100, facecolor=Colors.BG_MEDIUM)
        ax = fig.add_subplot(111)
        ax.set_facecolor(Colors.BG_MEDIUM)

        # Plot before
        if len(before_rar) > 0:
            ax.plot(before_rar['date'], before_rar['cumulative_rar'],
                    color=Colors.TEXT_SECONDARY, label='Before (All Trades)',
                    linewidth=1.5, alpha=0.7)

        # Plot after
        if after_rar is not None and len(after_rar) > 0:
            ax.plot(after_rar['date'], after_rar['cumulative_rar'],
                    color=Colors.SUCCESS, label='After (Filtered)',
                    linewidth=2)

        ax.set_xlabel('Date', color=Colors.TEXT_SECONDARY)
        ax.set_ylabel('Cumulative RAR%', color=Colors.TEXT_SECONDARY)
        ax.set_title('Cumulative RAR% Comparison', color=Colors.TEXT_PRIMARY)
        ax.legend(loc='upper left', facecolor=Colors.SURFACE, edgecolor=Colors.BORDER,
                  labelcolor=Colors.TEXT_PRIMARY)
        ax.tick_params(colors=Colors.TEXT_SECONDARY)
        ax.grid(True, color=Colors.BORDER, alpha=0.3)

        for spine in ax.spines.values():
            spine.set_color(Colors.BORDER)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _export_filtered_trades(self):
        """Export filtered trades to CSV."""
        if not self.rule_engine or self.trades_df is None:
            show_error(self.root, "Error", "No data to export.")
            return

        filtered = self.rule_engine.apply_rules(self.rules)

        if len(filtered) == 0:
            show_error(self.root, "Error", "No trades pass the current rules.")
            return

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "entry" if self.rule_mode == RuleMode.ENTRY else "exit"
        output_path = Path("logs") / f"filtered_trades_{mode_str}_{timestamp}.csv"
        output_path.parent.mkdir(exist_ok=True)

        export_filtered_trades(filtered, str(output_path))
        self.export_status_label.configure(
            text=f"Saved to {output_path}",
            text_color=Colors.SUCCESS
        )

    def _export_report(self):
        """Export comparison report to CSV."""
        if not self.comparison:
            show_error(self.root, "Error", "No results to export. Apply rules first.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "entry" if self.rule_mode == RuleMode.ENTRY else "exit"
        output_path = Path("logs") / f"rule_test_report_{mode_str}_{timestamp}.csv"
        output_path.parent.mkdir(exist_ok=True)

        export_comparison_report(self.comparison, self.rules, str(output_path))
        self.export_status_label.configure(
            text=f"Report saved to {output_path}",
            text_color=Colors.SUCCESS
        )


# Helper function for Theme (if not available in ctk_theme)
def Theme_create_button(parent, text, command=None, style="primary", width=120, state="normal", **kwargs):
    """Create a themed button."""
    if style == "secondary":
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            state=state,
            fg_color=Colors.SURFACE,
            hover_color=Colors.SURFACE_HOVER,
            **kwargs
        )
    else:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            state=state,
            **kwargs
        )


# Monkey-patch if Theme doesn't have create_button
if not hasattr(Theme, 'create_button'):
    Theme.create_button = staticmethod(Theme_create_button)


def main():
    """Main entry point."""
    app = CTkRuleTesterGUI()
    app.run()


if __name__ == "__main__":
    main()
