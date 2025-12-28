#!/usr/bin/env python3
"""
Raw Data Collection System GUI

A comprehensive unified raw data collection system powered by the Alpha Vantage API.
Provides a professional-grade GUI that orchestrates data fetching, validation,
preparation, and logging for all data types required by the backtesting framework.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import customtkinter as ctk
from tkinter import messagebox, filedialog
import tkinter as tk
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime, date
import threading
import json
import pandas as pd

# Import our data collection modules
from Classes.DataCollection import (
    DataCollectionConfig,
    APIConfig,
    CacheConfig,
    ValidationConfig,
    DailyDataConfig,
    WeeklyDataConfig,
    FundamentalDataConfig,
    InsiderDataConfig,
    ForexDataConfig,
    OptionsDataConfig,
    AlphaVantageClient,
    ValidationEngine,
    FileManager,
    LoggingManager,
    SessionLogger,
    AVAILABLE_INDICATORS,
    FUNDAMENTAL_CATEGORIES,
    MissingDataHandling,
    DateRangeType,
)
from Classes.DataCollection.file_manager import DataTransformer
from Classes.DataCollection.logging_manager import (
    DecisionLogEntry,
    DataIssueEntry,
)

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TickerSelector(ctk.CTkFrame):
    """Reusable ticker selection widget with multi-select and search."""

    TICKERS_JSON_PATH = Path("config/data_collection/tickers.json")

    def __init__(self, parent, available_tickers: List[str] = None, **kwargs):
        super().__init__(parent, **kwargs)

        self.available_tickers = available_tickers or []
        self.selected_tickers: set = set()
        self.ticker_vars: Dict[str, tk.BooleanVar] = {}
        self.presets = self._load_presets()

        self._create_widgets()

    def _create_widgets(self):
        # Header with count
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(header_frame, text="Tickers", font=("", 14, "bold")).pack(side="left")
        self.count_label = ctk.CTkLabel(header_frame, text="Selected: 0")
        self.count_label.pack(side="right")

        # Search/Add entry
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=5, pady=5)

        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search or add ticker...")
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.search_entry.bind("<KeyRelease>", self._on_search)
        self.search_entry.bind("<Return>", self._add_ticker)

        ctk.CTkButton(search_frame, text="Add", width=60, command=self._add_ticker).pack(side="right")

        # Scrollable ticker list
        self.ticker_frame = ctk.CTkScrollableFrame(self, height=200)
        self.ticker_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Preset selector
        if self.presets:
            preset_frame = ctk.CTkFrame(self, fg_color="transparent")
            preset_frame.pack(fill="x", padx=5, pady=(0, 5))

            ctk.CTkLabel(preset_frame, text="Load Preset:").pack(side="left", padx=(0, 5))
            preset_names = ["-- Select Preset --"] + list(self.presets.keys())
            self.preset_dropdown = ctk.CTkComboBox(
                preset_frame,
                values=preset_names,
                width=200,
                command=self._on_preset_selected
            )
            self.preset_dropdown.pack(side="left", fill="x", expand=True)

        # Action buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(btn_frame, text="Select All", width=80, command=self._select_all).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Clear All", width=80, command=self._clear_all).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Load from File", width=100, command=self._load_from_file).pack(side="right", padx=2)

        # Populate initial tickers
        self._populate_tickers()

    def _populate_tickers(self, filter_text: str = ""):
        # Clear existing
        for widget in self.ticker_frame.winfo_children():
            widget.destroy()

        filter_upper = filter_text.upper()

        for ticker in sorted(self.available_tickers):
            if filter_text and filter_upper not in ticker.upper():
                continue

            if ticker not in self.ticker_vars:
                self.ticker_vars[ticker] = tk.BooleanVar(value=ticker in self.selected_tickers)

            cb = ctk.CTkCheckBox(
                self.ticker_frame,
                text=ticker,
                variable=self.ticker_vars[ticker],
                command=lambda t=ticker: self._on_ticker_toggle(t)
            )
            cb.pack(anchor="w", pady=1)

    def _on_search(self, event=None):
        self._populate_tickers(self.search_entry.get())

    def _add_ticker(self, event=None):
        ticker = self.search_entry.get().strip().upper()
        if ticker and ticker not in self.available_tickers:
            self.available_tickers.append(ticker)
        if ticker:
            self.selected_tickers.add(ticker)
            if ticker not in self.ticker_vars:
                self.ticker_vars[ticker] = tk.BooleanVar(value=True)
            else:
                self.ticker_vars[ticker].set(True)
        self.search_entry.delete(0, "end")
        self._populate_tickers()
        self._update_count()

    def _on_ticker_toggle(self, ticker: str):
        if self.ticker_vars[ticker].get():
            self.selected_tickers.add(ticker)
        else:
            self.selected_tickers.discard(ticker)
        self._update_count()

    def _select_all(self):
        for ticker in self.available_tickers:
            self.selected_tickers.add(ticker)
            if ticker in self.ticker_vars:
                self.ticker_vars[ticker].set(True)
        self._update_count()

    def _clear_all(self):
        self.selected_tickers.clear()
        for var in self.ticker_vars.values():
            var.set(False)
        self._update_count()

    def _load_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Ticker List",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                # Parse tickers (comma, newline, or space separated)
                import re
                tickers = re.split(r'[,\s\n]+', content)
                tickers = [t.strip().upper() for t in tickers if t.strip()]

                for ticker in tickers:
                    if ticker not in self.available_tickers:
                        self.available_tickers.append(ticker)
                    self.selected_tickers.add(ticker)
                    if ticker not in self.ticker_vars:
                        self.ticker_vars[ticker] = tk.BooleanVar(value=True)
                    else:
                        self.ticker_vars[ticker].set(True)

                self._populate_tickers()
                self._update_count()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def _update_count(self):
        self.count_label.configure(text=f"Selected: {len(self.selected_tickers)}")

    def get_selected(self) -> List[str]:
        return sorted(list(self.selected_tickers))

    def set_tickers(self, tickers: List[str]):
        self.available_tickers = list(set(self.available_tickers + tickers))
        self._populate_tickers()

    def _load_presets(self) -> Dict[str, List[str]]:
        """Load ticker presets from tickers.json file."""
        presets = {}
        try:
            if self.TICKERS_JSON_PATH.exists():
                with open(self.TICKERS_JSON_PATH, 'r') as f:
                    data = json.load(f)

                # Load defined presets
                if "presets" in data:
                    for name, preset_data in data["presets"].items():
                        if isinstance(preset_data, dict) and "tickers" in preset_data:
                            description = preset_data.get("description", "")
                            display_name = f"{name.replace('_', ' ').title()} ({len(preset_data['tickers'])})"
                            presets[display_name] = preset_data["tickers"]
                        elif isinstance(preset_data, list):
                            presets[name.replace('_', ' ').title()] = preset_data

                # Create category-based presets from stocks
                if "categories" in data and "stocks" in data["categories"]:
                    for sector, caps in data["categories"]["stocks"].items():
                        sector_tickers = []
                        for cap_type, tickers in caps.items():
                            sector_tickers.extend(tickers)
                        if sector_tickers:
                            display_name = f"Sector: {sector.replace('_', ' ').title()} ({len(sector_tickers)})"
                            presets[display_name] = sector_tickers

                # Create ETF presets
                if "categories" in data and "etfs" in data["categories"]:
                    all_etfs = []
                    for etf_type, etf_data in data["categories"]["etfs"].items():
                        for style, tickers in etf_data.items():
                            all_etfs.extend(tickers)
                    if all_etfs:
                        presets[f"All ETFs ({len(all_etfs)})"] = all_etfs

                # Add forex pairs if available
                if "forex_pairs" in data:
                    presets[f"Forex Pairs ({len(data['forex_pairs'])})"] = data["forex_pairs"]

        except Exception as e:
            print(f"Warning: Could not load ticker presets: {e}")

        return presets

    def _on_preset_selected(self, preset_name: str):
        """Handle preset selection from dropdown."""
        if preset_name == "-- Select Preset --":
            return

        tickers = self.presets.get(preset_name, [])
        if not tickers:
            return

        # Add all tickers from preset
        for ticker in tickers:
            ticker = ticker.upper()
            if ticker not in self.available_tickers:
                self.available_tickers.append(ticker)
            self.selected_tickers.add(ticker)
            if ticker not in self.ticker_vars:
                self.ticker_vars[ticker] = tk.BooleanVar(value=True)
            else:
                self.ticker_vars[ticker].set(True)

        self._populate_tickers()
        self._update_count()

        # Reset dropdown
        self.preset_dropdown.set("-- Select Preset --")


class DateRangeSelector(ctk.CTkFrame):
    """Reusable date range selection widget."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.range_type = tk.StringVar(value="all")
        self._create_widgets()

    def _create_widgets(self):
        ctk.CTkLabel(self, text="Date Range", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        # Radio buttons for range type
        ctk.CTkRadioButton(
            self, text="All available data", variable=self.range_type,
            value="all", command=self._on_type_change
        ).pack(anchor="w", padx=20, pady=2)

        ctk.CTkRadioButton(
            self, text="Custom range:", variable=self.range_type,
            value="custom", command=self._on_type_change
        ).pack(anchor="w", padx=20, pady=2)

        # Custom range inputs
        self.custom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.custom_frame.pack(fill="x", padx=40, pady=5)

        ctk.CTkLabel(self.custom_frame, text="From:").grid(row=0, column=0, padx=5, pady=2)
        self.from_entry = ctk.CTkEntry(self.custom_frame, width=120, placeholder_text="YYYY-MM-DD")
        self.from_entry.grid(row=0, column=1, padx=5, pady=2)
        self.from_entry.insert(0, "2020-01-01")

        ctk.CTkLabel(self.custom_frame, text="To:").grid(row=1, column=0, padx=5, pady=2)
        self.to_entry = ctk.CTkEntry(self.custom_frame, width=120, placeholder_text="YYYY-MM-DD")
        self.to_entry.grid(row=1, column=1, padx=5, pady=2)
        self.to_entry.insert(0, date.today().strftime("%Y-%m-%d"))

        ctk.CTkRadioButton(
            self, text="Last N days:", variable=self.range_type,
            value="last_n", command=self._on_type_change
        ).pack(anchor="w", padx=20, pady=2)

        self.last_n_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.last_n_frame.pack(fill="x", padx=40, pady=5)

        self.last_n_entry = ctk.CTkEntry(self.last_n_frame, width=80)
        self.last_n_entry.pack(side="left", padx=5)
        self.last_n_entry.insert(0, "365")
        ctk.CTkLabel(self.last_n_frame, text="days").pack(side="left")

        self._on_type_change()

    def _on_type_change(self):
        is_custom = self.range_type.get() == "custom"
        is_last_n = self.range_type.get() == "last_n"

        for widget in self.custom_frame.winfo_children():
            widget.configure(state="normal" if is_custom else "disabled")

        self.last_n_entry.configure(state="normal" if is_last_n else "disabled")

    def get_config(self) -> Dict[str, Any]:
        range_type = self.range_type.get()

        if range_type == "all":
            return {"type": DateRangeType.ALL_AVAILABLE}
        elif range_type == "custom":
            return {
                "type": DateRangeType.CUSTOM_RANGE,
                "from_date": self.from_entry.get(),
                "to_date": self.to_entry.get()
            }
        else:
            return {
                "type": DateRangeType.LAST_N_DAYS,
                "last_n_days": int(self.last_n_entry.get() or 365)
            }


class IndicatorSelector(ctk.CTkFrame):
    """Reusable indicator selection widget with period configuration."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.selected_indicators: Dict[str, List[int]] = {}
        self.indicator_vars: Dict[str, tk.BooleanVar] = {}
        self.period_entries: Dict[str, ctk.CTkEntry] = {}

        self._create_widgets()

    def _create_widgets(self):
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(header_frame, text="Technical Indicators", font=("", 14, "bold")).pack(side="left")
        self.count_label = ctk.CTkLabel(header_frame, text="Selected: 0")
        self.count_label.pack(side="right")

        # Quick select buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(btn_frame, text="All", width=50, command=self._select_all).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="None", width=50, command=self._clear_all).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Common", width=60, command=self._select_common).pack(side="left", padx=2)

        # Scrollable indicator list
        self.indicator_frame = ctk.CTkScrollableFrame(self, height=250)
        self.indicator_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Populate indicators
        self._populate_indicators()

    def _populate_indicators(self):
        # Group indicators by category
        categories = {
            "Moving Averages": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3"],
            "Momentum": ["RSI", "STOCH", "STOCHF", "STOCHRSI", "WILLR", "MOM", "ROC", "ROCR", "CCI", "CMO"],
            "Trend": ["ADX", "ADXR", "AROON", "AROONOSC", "DX", "MINUS_DI", "PLUS_DI", "MACD", "MACDEXT"],
            "Volatility": ["ATR", "NATR", "TRANGE", "BBANDS"],
            "Volume": ["AD", "ADOSC", "OBV", "MFI"],
            "Other": ["APO", "PPO", "BOP", "TRIX", "ULTOSC", "SAR", "MIDPOINT", "MIDPRICE",
                      "HT_TRENDLINE", "HT_SINE", "HT_TRENDMODE", "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "VWAP", "MAMA"]
        }

        default_periods = {
            "SMA": "20,50,200", "EMA": "12,26", "RSI": "14", "ATR": "14",
            "ADX": "14", "BBANDS": "20", "CCI": "20", "MOM": "10",
            "ROC": "10", "WILLR": "14", "MFI": "14", "STOCH": "14",
        }

        for category, indicators in categories.items():
            # Category header
            cat_label = ctk.CTkLabel(self.indicator_frame, text=category, font=("", 12, "bold"))
            cat_label.pack(anchor="w", pady=(10, 5))

            for indicator in indicators:
                if indicator not in AVAILABLE_INDICATORS:
                    continue

                frame = ctk.CTkFrame(self.indicator_frame, fg_color="transparent")
                frame.pack(fill="x", pady=1)

                self.indicator_vars[indicator] = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(
                    frame, text=indicator, width=100,
                    variable=self.indicator_vars[indicator],
                    command=self._update_count
                )
                cb.pack(side="left")

                # Period entry for indicators that need it
                if indicator in ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3",
                                 "RSI", "ATR", "ADX", "BBANDS", "CCI", "MOM", "ROC", "WILLR", "MFI"]:
                    ctk.CTkLabel(frame, text="Period:").pack(side="left", padx=(10, 5))
                    entry = ctk.CTkEntry(frame, width=80, placeholder_text="e.g., 14")
                    entry.pack(side="left")
                    entry.insert(0, default_periods.get(indicator, "14"))
                    self.period_entries[indicator] = entry

    def _select_all(self):
        for var in self.indicator_vars.values():
            var.set(True)
        self._update_count()

    def _clear_all(self):
        for var in self.indicator_vars.values():
            var.set(False)
        self._update_count()

    def _select_common(self):
        common = ["SMA", "EMA", "RSI", "MACD", "ATR", "BBANDS", "ADX"]
        for ind, var in self.indicator_vars.items():
            var.set(ind in common)
        self._update_count()

    def _update_count(self):
        count = sum(1 for v in self.indicator_vars.values() if v.get())
        self.count_label.configure(text=f"Selected: {count}")

    def get_selected(self) -> Dict[str, List[int]]:
        result = {}
        for indicator, var in self.indicator_vars.items():
            if var.get():
                periods = []
                if indicator in self.period_entries:
                    period_str = self.period_entries[indicator].get()
                    if period_str:
                        periods = [int(p.strip()) for p in period_str.split(",") if p.strip().isdigit()]
                result[indicator] = periods if periods else [14]  # Default period
        return result


class MissingDataHandler(ctk.CTkFrame):
    """Widget for selecting missing data handling strategy."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.strategy = tk.StringVar(value="forward_fill")
        self._create_widgets()

    def _create_widgets(self):
        ctk.CTkLabel(self, text="Handle Missing Data", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        strategies = [
            ("forward_fill", "Forward Fill (use last known value)"),
            ("interpolate", "Interpolate (linear interpolation)"),
            ("skip", "Skip (remove rows with missing data)"),
        ]

        for value, text in strategies:
            ctk.CTkRadioButton(
                self, text=text, variable=self.strategy, value=value
            ).pack(anchor="w", padx=20, pady=2)

    def get_strategy(self) -> MissingDataHandling:
        return MissingDataHandling(self.strategy.get())


class ProgressPanel(ctk.CTkFrame):
    """Progress tracking panel with progress bar and message log."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        # Progress bar section
        progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        progress_frame.pack(fill="x", padx=10, pady=10)

        self.status_label = ctk.CTkLabel(progress_frame, text="Ready", font=("", 12))
        self.status_label.pack(anchor="w")

        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)

        self.progress_text = ctk.CTkLabel(progress_frame, text="0%")
        self.progress_text.pack(anchor="e")

        # Message log
        ctk.CTkLabel(self, text="Messages:", font=("", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        self.message_box = ctk.CTkTextbox(self, height=150)
        self.message_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def set_status(self, text: str):
        self.status_label.configure(text=text)

    def set_progress(self, value: float, text: str = None):
        """Set progress (0.0 to 1.0)."""
        self.progress_bar.set(value)
        self.progress_text.configure(text=f"{int(value * 100)}%")
        if text:
            self.set_status(text)

    def log_message(self, message: str, level: str = "info"):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "[i]", "success": "[+]", "warning": "[!]", "error": "[X]"}.get(level, "[*]")

        self.message_box.configure(state="normal")
        self.message_box.insert("end", f"{timestamp} {prefix} {message}\n")
        self.message_box.see("end")
        self.message_box.configure(state="disabled")

    def clear(self):
        self.progress_bar.set(0)
        self.progress_text.configure(text="0%")
        self.set_status("Ready")
        self.message_box.configure(state="normal")
        self.message_box.delete("1.0", "end")
        self.message_box.configure(state="disabled")


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog for API and cache configuration."""

    def __init__(self, parent, config: DataCollectionConfig, on_save: Callable):
        super().__init__(parent)

        self.config = config
        self.on_save = on_save

        self.title("Settings")
        self.geometry("500x600")
        self.transient(parent)
        self.grab_set()

        self._create_widgets()
        self._load_config()

    def _create_widgets(self):
        # API Configuration
        api_frame = ctk.CTkFrame(self)
        api_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(api_frame, text="API Configuration", font=("", 16, "bold")).pack(anchor="w", pady=10)

        # API Key
        ctk.CTkLabel(api_frame, text="API Key:").pack(anchor="w", padx=10)
        self.api_key_entry = ctk.CTkEntry(api_frame, width=400, show="*")
        self.api_key_entry.pack(padx=10, pady=5)

        # Show/Hide API Key
        self.show_key_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            api_frame, text="Show API Key", variable=self.show_key_var,
            command=self._toggle_api_key_visibility
        ).pack(anchor="w", padx=10)

        # Rate Limit
        rate_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
        rate_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(rate_frame, text="Rate Limit:").pack(side="left")
        self.rate_limit_entry = ctk.CTkEntry(rate_frame, width=80)
        self.rate_limit_entry.pack(side="left", padx=10)
        ctk.CTkLabel(rate_frame, text="calls/minute").pack(side="left")

        # Timeout
        timeout_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
        timeout_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(timeout_frame, text="Timeout:").pack(side="left")
        self.timeout_entry = ctk.CTkEntry(timeout_frame, width=80)
        self.timeout_entry.pack(side="left", padx=10)
        ctk.CTkLabel(timeout_frame, text="seconds").pack(side="left")

        # Max Retries
        retry_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
        retry_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(retry_frame, text="Max Retries:").pack(side="left")
        self.retries_entry = ctk.CTkEntry(retry_frame, width=80)
        self.retries_entry.pack(side="left", padx=10)

        # Test Connection Button
        ctk.CTkButton(api_frame, text="Test Connection", command=self._test_connection).pack(pady=10)

        # Cache Configuration
        cache_frame = ctk.CTkFrame(self)
        cache_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(cache_frame, text="Cache Configuration", font=("", 16, "bold")).pack(anchor="w", pady=10)

        # Cache Enabled
        self.cache_enabled_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(cache_frame, text="Enable Caching", variable=self.cache_enabled_var).pack(anchor="w", padx=10)

        # Cache Directory
        cache_dir_frame = ctk.CTkFrame(cache_frame, fg_color="transparent")
        cache_dir_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(cache_dir_frame, text="Cache Directory:").pack(anchor="w")
        self.cache_dir_entry = ctk.CTkEntry(cache_dir_frame, width=350)
        self.cache_dir_entry.pack(side="left", pady=5)
        ctk.CTkButton(cache_dir_frame, text="Browse", width=60, command=self._browse_cache_dir).pack(side="left", padx=5)

        # Cache Expiry
        expiry_frame = ctk.CTkFrame(cache_frame, fg_color="transparent")
        expiry_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(expiry_frame, text="Cache Expiry:").pack(side="left")
        self.cache_expiry_entry = ctk.CTkEntry(expiry_frame, width=80)
        self.cache_expiry_entry.pack(side="left", padx=10)
        ctk.CTkLabel(expiry_frame, text="hours").pack(side="left")

        # Clear Cache Button
        ctk.CTkButton(cache_frame, text="Clear Cache", command=self._clear_cache).pack(pady=10)

        # Save/Cancel buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkButton(btn_frame, text="Save", command=self._save).pack(side="right", padx=10)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy).pack(side="right")

    def _load_config(self):
        self.api_key_entry.insert(0, self.config.api.api_key)
        self.rate_limit_entry.insert(0, str(self.config.api.rate_limit_per_minute))
        self.timeout_entry.insert(0, str(self.config.api.timeout_seconds))
        self.retries_entry.insert(0, str(self.config.api.max_retries))

        self.cache_enabled_var.set(self.config.cache.enabled)
        self.cache_dir_entry.insert(0, str(self.config.cache.cache_dir))
        self.cache_expiry_entry.insert(0, str(self.config.cache.cache_expiry_hours))

    def _toggle_api_key_visibility(self):
        self.api_key_entry.configure(show="" if self.show_key_var.get() else "*")

    def _browse_cache_dir(self):
        directory = filedialog.askdirectory(title="Select Cache Directory")
        if directory:
            self.cache_dir_entry.delete(0, "end")
            self.cache_dir_entry.insert(0, directory)

    def _test_connection(self):
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter an API key")
            return

        # Create temporary client for testing
        api_config = APIConfig(api_key=api_key)
        cache_config = CacheConfig(enabled=False)
        client = AlphaVantageClient(api_config, cache_config)

        valid, message = client.validate_api_key()
        if valid:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", message)

    def _clear_cache(self):
        if messagebox.askyesno("Confirm", "Clear all cached data?"):
            cache_dir = Path(self.cache_dir_entry.get())
            if cache_dir.exists():
                import shutil
                for f in cache_dir.glob("*.json"):
                    f.unlink()
                messagebox.showinfo("Success", "Cache cleared")

    def _save(self):
        try:
            self.config.api.api_key = self.api_key_entry.get().strip()
            self.config.api.rate_limit_per_minute = int(self.rate_limit_entry.get())
            self.config.api.timeout_seconds = int(self.timeout_entry.get())
            self.config.api.max_retries = int(self.retries_entry.get())

            self.config.cache.enabled = self.cache_enabled_var.get()
            self.config.cache.cache_dir = Path(self.cache_dir_entry.get())
            self.config.cache.cache_expiry_hours = int(self.cache_expiry_entry.get())

            self.on_save(self.config)
            self.destroy()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")


class BaseTab(ctk.CTkFrame):
    """Base class for data collection tabs."""

    def __init__(self, parent, app: 'DataCollectionApp', **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app

    def get_config(self) -> Dict[str, Any]:
        """Get configuration from this tab. Override in subclasses."""
        raise NotImplementedError

    def validate(self) -> tuple:
        """Validate tab inputs. Returns (is_valid, error_message)."""
        raise NotImplementedError

    def collect_data(self, progress_callback: Callable):
        """Collect data for this tab. Override in subclasses."""
        raise NotImplementedError


class DailyTab(BaseTab):
    """Tab for daily price and indicator data collection."""

    def __init__(self, parent, app: 'DataCollectionApp', **kwargs):
        super().__init__(parent, app, **kwargs)
        self._create_widgets()

    def _create_widgets(self):
        # Create scrollable container
        container = ctk.CTkScrollableFrame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left column: Tickers
        left_frame = ctk.CTkFrame(container)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.ticker_selector = TickerSelector(left_frame, self._get_existing_tickers())
        self.ticker_selector.pack(fill="both", expand=True, pady=5)

        # Date range
        self.date_range = DateRangeSelector(left_frame)
        self.date_range.pack(fill="x", pady=5)

        # Missing data handling
        self.missing_handler = MissingDataHandler(left_frame)
        self.missing_handler.pack(fill="x", pady=5)

        # Options
        options_frame = ctk.CTkFrame(left_frame)
        options_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(options_frame, text="Options", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.adjust_splits_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_frame, text="Adjust for Splits/Dividends", variable=self.adjust_splits_var).pack(anchor="w", padx=20)

        # Right column: Indicators
        right_frame = ctk.CTkFrame(container)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.indicator_selector = IndicatorSelector(right_frame)
        self.indicator_selector.pack(fill="both", expand=True)

    def _get_existing_tickers(self) -> List[str]:
        """Get list of tickers from existing raw_data files."""
        raw_data_path = Path("raw_data")
        tickers = []
        if raw_data_path.exists():
            for f in raw_data_path.glob("*.csv"):
                ticker = f.stem.replace("_daily", "").replace("_weekly", "").upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
        return sorted(tickers)

    def get_config(self) -> Dict[str, Any]:
        date_config = self.date_range.get_config()
        indicators = self.indicator_selector.get_selected()

        return {
            "tickers": self.ticker_selector.get_selected(),
            "date_range": date_config,
            "indicators": indicators,
            "adjust_for_splits": self.adjust_splits_var.get(),
            "missing_handling": self.missing_handler.get_strategy(),
        }

    def validate(self) -> tuple:
        tickers = self.ticker_selector.get_selected()
        if not tickers:
            return False, "Please select at least one ticker"
        return True, ""

    def collect_data(self, progress_callback: Callable) -> Dict[str, Any]:
        """Collect daily data for selected tickers."""
        config = self.get_config()
        tickers = config["tickers"]
        indicators = config["indicators"]
        results = {"success": [], "failed": [], "partial": []}

        client = self.app.api_client
        file_manager = self.app.file_manager
        validator = self.app.validator
        logger = self.app.logger

        total_steps = len(tickers) * (1 + sum(len(periods) for periods in indicators.values()))
        current_step = 0

        for ticker in tickers:
            if self.app.cancelled:
                break

            try:
                progress_callback(f"Fetching {ticker} daily prices...", current_step / total_steps)
                logger.log_session_info(f"FETCHING {ticker}")

                # Get price data
                price_response = client.get_daily_prices(ticker, adjusted=config["adjust_for_splits"])
                current_step += 1

                if not price_response.success:
                    results["failed"].append((ticker, price_response.error_message))
                    logger.log_warning("API Error", price_response.error_message, symbol=ticker)
                    continue

                # Transform price data
                price_df = DataTransformer.transform_daily_prices(price_response.data, config["adjust_for_splits"])

                if price_df.empty:
                    results["failed"].append((ticker, "No price data returned"))
                    continue

                # Fetch indicators
                indicator_dfs = []
                for ind_name, periods in indicators.items():
                    for period in periods:
                        if self.app.cancelled:
                            break

                        progress_callback(f"Fetching {ticker} {ind_name}({period})...", current_step / total_steps)

                        ind_response = client.get_indicator(ind_name, ticker, time_period=period)
                        current_step += 1

                        if ind_response.success:
                            ind_df = DataTransformer.transform_indicator(ind_response.data, f"{ind_name}_{period}")
                            if not ind_df.empty:
                                indicator_dfs.append(ind_df)
                        else:
                            logger.log_warning("Indicator Error", f"Failed to fetch {ind_name}({period})", symbol=ticker)

                # Merge price and indicator data
                final_df = DataTransformer.merge_price_and_indicators(price_df, indicator_dfs)

                # Filter by date range
                final_df = self._filter_date_range(final_df, config["date_range"])

                # Validate
                indicator_cols = [c for c in final_df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]
                validation_report = validator.validate_daily_data(final_df, ticker, indicator_cols)

                # Write file
                file_meta = file_manager.write_daily_data(
                    final_df, ticker,
                    indicators=indicator_cols,
                    missing_handling=config["missing_handling"]
                )

                if validation_report.passed:
                    results["success"].append((ticker, file_meta))
                else:
                    results["partial"].append((ticker, file_meta, validation_report))

                progress_callback(f"Completed {ticker}", current_step / total_steps)

            except Exception as e:
                results["failed"].append((ticker, str(e)))
                logger.log_exception("Collection Error", str(e), e, symbol=ticker)

        return results

    def _filter_date_range(self, df: pd.DataFrame, date_config: Dict) -> pd.DataFrame:
        """Filter DataFrame by date range configuration."""
        if df.empty or 'date' not in df.columns:
            return df

        df['date'] = pd.to_datetime(df['date'])

        if date_config["type"] == DateRangeType.ALL_AVAILABLE:
            return df
        elif date_config["type"] == DateRangeType.CUSTOM_RANGE:
            from_date = pd.to_datetime(date_config.get("from_date"))
            to_date = pd.to_datetime(date_config.get("to_date"))
            return df[(df['date'] >= from_date) & (df['date'] <= to_date)]
        elif date_config["type"] == DateRangeType.LAST_N_DAYS:
            from datetime import timedelta
            days = date_config.get("last_n_days", 365)
            cutoff = datetime.now() - timedelta(days=days)
            return df[df['date'] >= cutoff]

        return df


class WeeklyTab(BaseTab):
    """Tab for weekly price and indicator data collection."""

    def __init__(self, parent, app: 'DataCollectionApp', **kwargs):
        super().__init__(parent, app, **kwargs)
        self._create_widgets()

    def _create_widgets(self):
        container = ctk.CTkScrollableFrame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left column
        left_frame = ctk.CTkFrame(container)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.ticker_selector = TickerSelector(left_frame, self._get_existing_tickers())
        self.ticker_selector.pack(fill="both", expand=True, pady=5)

        self.date_range = DateRangeSelector(left_frame)
        self.date_range.pack(fill="x", pady=5)

        self.missing_handler = MissingDataHandler(left_frame)
        self.missing_handler.pack(fill="x", pady=5)

        options_frame = ctk.CTkFrame(left_frame)
        options_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(options_frame, text="Options", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.adjust_splits_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_frame, text="Adjust for Splits/Dividends", variable=self.adjust_splits_var).pack(anchor="w", padx=20)

        # Right column
        right_frame = ctk.CTkFrame(container)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.indicator_selector = IndicatorSelector(right_frame)
        self.indicator_selector.pack(fill="both", expand=True)

    def _get_existing_tickers(self) -> List[str]:
        raw_data_path = Path("raw_data")
        tickers = []
        if raw_data_path.exists():
            for f in raw_data_path.glob("*.csv"):
                ticker = f.stem.replace("_daily", "").replace("_weekly", "").upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
        return sorted(tickers)

    def get_config(self) -> Dict[str, Any]:
        date_config = self.date_range.get_config()
        indicators = self.indicator_selector.get_selected()

        return {
            "tickers": self.ticker_selector.get_selected(),
            "date_range": date_config,
            "indicators": indicators,
            "adjust_for_splits": self.adjust_splits_var.get(),
            "missing_handling": self.missing_handler.get_strategy(),
        }

    def validate(self) -> tuple:
        tickers = self.ticker_selector.get_selected()
        if not tickers:
            return False, "Please select at least one ticker"
        return True, ""

    def collect_data(self, progress_callback: Callable) -> Dict[str, Any]:
        """Collect weekly data for selected tickers."""
        config = self.get_config()
        tickers = config["tickers"]
        indicators = config["indicators"]
        results = {"success": [], "failed": [], "partial": []}

        client = self.app.api_client
        file_manager = self.app.file_manager
        validator = self.app.validator
        logger = self.app.logger

        total_steps = len(tickers) * (1 + sum(len(periods) for periods in indicators.values()))
        current_step = 0

        for ticker in tickers:
            if self.app.cancelled:
                break

            try:
                progress_callback(f"Fetching {ticker} weekly prices...", current_step / total_steps)
                logger.log_session_info(f"FETCHING {ticker} WEEKLY")

                price_response = client.get_weekly_prices(ticker, adjusted=config["adjust_for_splits"])
                current_step += 1

                if not price_response.success:
                    results["failed"].append((ticker, price_response.error_message))
                    continue

                price_df = DataTransformer.transform_weekly_prices(price_response.data, config["adjust_for_splits"])

                if price_df.empty:
                    results["failed"].append((ticker, "No price data returned"))
                    continue

                # Fetch indicators with weekly interval
                indicator_dfs = []
                for ind_name, periods in indicators.items():
                    for period in periods:
                        if self.app.cancelled:
                            break

                        progress_callback(f"Fetching {ticker} {ind_name}({period})...", current_step / total_steps)

                        ind_response = client.get_indicator(ind_name, ticker, interval="weekly", time_period=period)
                        current_step += 1

                        if ind_response.success:
                            ind_df = DataTransformer.transform_indicator(ind_response.data, f"{ind_name}_{period}")
                            if not ind_df.empty:
                                indicator_dfs.append(ind_df)

                final_df = DataTransformer.merge_price_and_indicators(price_df, indicator_dfs)

                indicator_cols = [c for c in final_df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]
                validation_report = validator.validate_weekly_data(final_df, ticker, indicator_cols)

                file_meta = file_manager.write_weekly_data(
                    final_df, ticker,
                    indicators=indicator_cols,
                    missing_handling=config["missing_handling"]
                )

                if validation_report.passed:
                    results["success"].append((ticker, file_meta))
                else:
                    results["partial"].append((ticker, file_meta, validation_report))

                progress_callback(f"Completed {ticker}", current_step / total_steps)

            except Exception as e:
                results["failed"].append((ticker, str(e)))
                logger.log_exception("Collection Error", str(e), e, symbol=ticker)

        return results


class FundamentalTab(BaseTab):
    """Tab for fundamental data collection."""

    def __init__(self, parent, app: 'DataCollectionApp', **kwargs):
        super().__init__(parent, app, **kwargs)
        self._create_widgets()

    def _create_widgets(self):
        container = ctk.CTkScrollableFrame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Ticker selection
        self.ticker_selector = TickerSelector(container, self._get_existing_tickers())
        self.ticker_selector.pack(fill="x", pady=5)

        # Date range
        self.date_range = DateRangeSelector(container)
        self.date_range.pack(fill="x", pady=5)

        # Categories
        cat_frame = ctk.CTkFrame(container)
        cat_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(cat_frame, text="Data Categories", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.category_vars = {}
        for category in FUNDAMENTAL_CATEGORIES.keys():
            display_name = category.replace("_", " ").title()
            self.category_vars[category] = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(cat_frame, text=display_name, variable=self.category_vars[category]).pack(anchor="w", padx=20, pady=2)

        # Options
        options_frame = ctk.CTkFrame(container)
        options_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(options_frame, text="Options", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.include_estimates_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(options_frame, text="Include Analyst Estimates", variable=self.include_estimates_var).pack(anchor="w", padx=20)

        # Data frequency
        ctk.CTkLabel(options_frame, text="Frequency:").pack(anchor="w", padx=20, pady=(10, 5))

        self.frequency_var = tk.StringVar(value="both")
        ctk.CTkRadioButton(options_frame, text="Both Quarterly & Annual", variable=self.frequency_var, value="both").pack(anchor="w", padx=40)
        ctk.CTkRadioButton(options_frame, text="Quarterly Only", variable=self.frequency_var, value="quarterly").pack(anchor="w", padx=40)
        ctk.CTkRadioButton(options_frame, text="Annual Only", variable=self.frequency_var, value="annual").pack(anchor="w", padx=40)

        self.missing_handler = MissingDataHandler(container)
        self.missing_handler.pack(fill="x", pady=5)

    def _get_existing_tickers(self) -> List[str]:
        raw_data_path = Path("raw_data")
        tickers = []
        if raw_data_path.exists():
            for f in raw_data_path.glob("*.csv"):
                ticker = f.stem.replace("_daily", "").replace("_weekly", "").replace("_fundamental", "").upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
        return sorted(tickers)

    def get_config(self) -> Dict[str, Any]:
        selected_categories = [cat for cat, var in self.category_vars.items() if var.get()]

        return {
            "tickers": self.ticker_selector.get_selected(),
            "date_range": self.date_range.get_config(),
            "categories": selected_categories,
            "include_estimates": self.include_estimates_var.get(),
            "frequency": self.frequency_var.get(),
            "missing_handling": self.missing_handler.get_strategy(),
        }

    def validate(self) -> tuple:
        tickers = self.ticker_selector.get_selected()
        if not tickers:
            return False, "Please select at least one ticker"

        categories = [cat for cat, var in self.category_vars.items() if var.get()]
        if not categories:
            return False, "Please select at least one data category"

        return True, ""

    def collect_data(self, progress_callback: Callable) -> Dict[str, Any]:
        """Collect fundamental data for selected tickers."""
        config = self.get_config()
        tickers = config["tickers"]
        results = {"success": [], "failed": [], "partial": []}

        client = self.app.api_client
        file_manager = self.app.file_manager
        validator = self.app.validator
        logger = self.app.logger

        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            if self.app.cancelled:
                break

            try:
                progress_callback(f"Fetching {ticker} fundamentals...", i / total_tickers)
                logger.log_session_info(f"FETCHING {ticker} FUNDAMENTAL")

                all_data = []

                # Get company overview
                overview_response = client.get_company_overview(ticker)
                if overview_response.success:
                    overview_df = DataTransformer.transform_fundamental_overview(overview_response.data)
                    if not overview_df.empty:
                        all_data.append(overview_df)

                # Get earnings
                earnings_response = client.get_earnings(ticker)
                if earnings_response.success:
                    earnings_df = DataTransformer.transform_earnings(earnings_response.data)
                    if not earnings_df.empty:
                        all_data.append(earnings_df)

                # Get income statement, balance sheet, cash flow
                for endpoint, transform_func in [
                    (client.get_income_statement, "income_statement"),
                    (client.get_balance_sheet, "balance_sheet"),
                    (client.get_cash_flow, "cash_flow"),
                ]:
                    response = endpoint(ticker)
                    if response.success and response.data:
                        # These endpoints return quarterly and annual reports
                        quarterly = response.data.get("quarterlyReports", [])
                        annual = response.data.get("annualReports", [])

                        if config["frequency"] in ["both", "quarterly"] and quarterly:
                            df = pd.DataFrame(quarterly)
                            df["report_type"] = "quarterly"
                            all_data.append(df)

                        if config["frequency"] in ["both", "annual"] and annual:
                            df = pd.DataFrame(annual)
                            df["report_type"] = "annual"
                            all_data.append(df)

                if not all_data:
                    results["failed"].append((ticker, "No fundamental data returned"))
                    continue

                # Combine all data (this is simplified - real implementation would merge properly)
                combined_df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]

                # Add symbol column
                combined_df["symbol"] = ticker

                # Validate and write
                validation_report = validator.validate_fundamental_data(combined_df, ticker)

                file_meta = file_manager.write_fundamental_data(
                    combined_df, ticker,
                    missing_handling=config["missing_handling"]
                )

                if validation_report.passed:
                    results["success"].append((ticker, file_meta))
                else:
                    results["partial"].append((ticker, file_meta, validation_report))

                progress_callback(f"Completed {ticker}", (i + 1) / total_tickers)

            except Exception as e:
                results["failed"].append((ticker, str(e)))
                logger.log_exception("Collection Error", str(e), e, symbol=ticker)

        return results


class InsiderTab(BaseTab):
    """Tab for insider transaction data collection."""

    def __init__(self, parent, app: 'DataCollectionApp', **kwargs):
        super().__init__(parent, app, **kwargs)
        self._create_widgets()

    def _create_widgets(self):
        container = ctk.CTkScrollableFrame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        self.ticker_selector = TickerSelector(container, self._get_existing_tickers())
        self.ticker_selector.pack(fill="x", pady=5)

        self.date_range = DateRangeSelector(container)
        self.date_range.pack(fill="x", pady=5)

        # Transaction types
        type_frame = ctk.CTkFrame(container)
        type_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(type_frame, text="Transaction Types", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.trans_type_var = tk.StringVar(value="all")
        ctk.CTkRadioButton(type_frame, text="All Transactions", variable=self.trans_type_var, value="all").pack(anchor="w", padx=20)
        ctk.CTkRadioButton(type_frame, text="Buys Only", variable=self.trans_type_var, value="buys").pack(anchor="w", padx=20)
        ctk.CTkRadioButton(type_frame, text="Sells Only", variable=self.trans_type_var, value="sells").pack(anchor="w", padx=20)

        # Filters
        filter_frame = ctk.CTkFrame(container)
        filter_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(filter_frame, text="Filters", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        min_val_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        min_val_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(min_val_frame, text="Min Transaction Value: $").pack(side="left")
        self.min_value_entry = ctk.CTkEntry(min_val_frame, width=100)
        self.min_value_entry.pack(side="left", padx=5)
        self.min_value_entry.insert(0, "0")

    def _get_existing_tickers(self) -> List[str]:
        raw_data_path = Path("raw_data")
        tickers = []
        if raw_data_path.exists():
            for f in raw_data_path.glob("*.csv"):
                ticker = f.stem.split("_")[0].upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
        return sorted(tickers)

    def get_config(self) -> Dict[str, Any]:
        return {
            "tickers": self.ticker_selector.get_selected(),
            "date_range": self.date_range.get_config(),
            "transaction_type": self.trans_type_var.get(),
            "min_value": float(self.min_value_entry.get() or 0),
        }

    def validate(self) -> tuple:
        tickers = self.ticker_selector.get_selected()
        if not tickers:
            return False, "Please select at least one ticker"
        return True, ""

    def collect_data(self, progress_callback: Callable) -> Dict[str, Any]:
        """Collect insider transaction data."""
        config = self.get_config()
        tickers = config["tickers"]
        results = {"success": [], "failed": [], "partial": []}

        client = self.app.api_client
        file_manager = self.app.file_manager
        validator = self.app.validator
        logger = self.app.logger

        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            if self.app.cancelled:
                break

            try:
                progress_callback(f"Fetching {ticker} insider transactions...", i / total_tickers)
                logger.log_session_info(f"FETCHING {ticker} INSIDER")

                response = client.get_insider_transactions(ticker)

                if not response.success:
                    results["failed"].append((ticker, response.error_message))
                    continue

                df = DataTransformer.transform_insider_transactions(response.data)

                if df.empty:
                    results["failed"].append((ticker, "No insider transaction data returned"))
                    logger.log_data_issue(DataIssueEntry(
                        timestamp=datetime.now(),
                        symbol=ticker,
                        issue_type="No Data",
                        description="No insider transactions found",
                        severity="info"
                    ))
                    continue

                # Filter by transaction type
                if config["transaction_type"] == "buys":
                    df = df[df["transaction_type"] == "BUY"]
                elif config["transaction_type"] == "sells":
                    df = df[df["transaction_type"] == "SELL"]

                # Filter by minimum value
                if config["min_value"] > 0 and "value" in df.columns:
                    df = df[df["value"] >= config["min_value"]]

                if df.empty:
                    results["failed"].append((ticker, "No transactions match filters"))
                    continue

                validation_report = validator.validate_insider_data(df, ticker)
                file_meta = file_manager.write_insider_data(df, ticker)

                if validation_report.passed:
                    results["success"].append((ticker, file_meta))
                else:
                    results["partial"].append((ticker, file_meta, validation_report))

                progress_callback(f"Completed {ticker}", (i + 1) / total_tickers)

            except Exception as e:
                results["failed"].append((ticker, str(e)))
                logger.log_exception("Collection Error", str(e), e, symbol=ticker)

        return results


class ForexTab(BaseTab):
    """Tab for forex data collection."""

    def __init__(self, parent, app: 'DataCollectionApp', **kwargs):
        super().__init__(parent, app, **kwargs)
        self._create_widgets()

    def _create_widgets(self):
        container = ctk.CTkScrollableFrame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Currency pairs
        pairs_frame = ctk.CTkFrame(container)
        pairs_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(pairs_frame, text="Currency Pairs", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.gbpusd_var = tk.BooleanVar(value=True)
        self.gbpeur_var = tk.BooleanVar(value=True)

        ctk.CTkCheckBox(pairs_frame, text="GBP/USD", variable=self.gbpusd_var).pack(anchor="w", padx=20)
        ctk.CTkCheckBox(pairs_frame, text="GBP/EUR", variable=self.gbpeur_var).pack(anchor="w", padx=20)

        # Custom pairs
        custom_frame = ctk.CTkFrame(pairs_frame, fg_color="transparent")
        custom_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(custom_frame, text="Custom pairs (comma-separated, e.g., EUR/USD, USD/JPY):").pack(anchor="w")
        self.custom_pairs_entry = ctk.CTkEntry(custom_frame, width=300)
        self.custom_pairs_entry.pack(anchor="w", pady=5)

        # Date range
        self.date_range = DateRangeSelector(container)
        self.date_range.pack(fill="x", pady=5)

        # Week definition
        week_frame = ctk.CTkFrame(container)
        week_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(week_frame, text="Week Definition", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.week_def_var = tk.StringVar(value="friday")
        ctk.CTkRadioButton(week_frame, text="Friday Close (UTC)", variable=self.week_def_var, value="friday").pack(anchor="w", padx=20)
        ctk.CTkRadioButton(week_frame, text="Monday Open", variable=self.week_def_var, value="monday").pack(anchor="w", padx=20)

        self.missing_handler = MissingDataHandler(container)
        self.missing_handler.pack(fill="x", pady=5)

    def get_config(self) -> Dict[str, Any]:
        pairs = []
        if self.gbpusd_var.get():
            pairs.append("GBPUSD")
        if self.gbpeur_var.get():
            pairs.append("GBPEUR")

        # Add custom pairs
        custom = self.custom_pairs_entry.get().strip()
        if custom:
            custom_pairs = [p.strip().upper().replace("/", "") for p in custom.split(",") if p.strip()]
            pairs.extend(custom_pairs)

        return {
            "pairs": pairs,
            "date_range": self.date_range.get_config(),
            "week_definition": self.week_def_var.get(),
            "missing_handling": self.missing_handler.get_strategy(),
        }

    def validate(self) -> tuple:
        config = self.get_config()
        if not config["pairs"]:
            return False, "Please select at least one currency pair"
        return True, ""

    def collect_data(self, progress_callback: Callable) -> Dict[str, Any]:
        """Collect forex data."""
        config = self.get_config()
        pairs = config["pairs"]
        results = {"success": [], "failed": [], "partial": []}

        client = self.app.api_client
        file_manager = self.app.file_manager
        validator = self.app.validator
        logger = self.app.logger

        total_pairs = len(pairs)

        for i, pair in enumerate(pairs):
            if self.app.cancelled:
                break

            try:
                progress_callback(f"Fetching {pair} forex data...", i / total_pairs)
                logger.log_session_info(f"FETCHING {pair} FOREX")

                # Parse from/to currencies
                from_currency = pair[:3]
                to_currency = pair[3:]

                response = client.get_forex_weekly(from_currency, to_currency)

                if not response.success:
                    results["failed"].append((pair, response.error_message))
                    continue

                df = DataTransformer.transform_forex_weekly(response.data, pair)

                if df.empty:
                    results["failed"].append((pair, "No forex data returned"))
                    continue

                validation_report = validator.validate_forex_data(df, pair)

                file_meta = file_manager.write_forex_data(
                    df, pair,
                    missing_handling=config["missing_handling"]
                )

                if validation_report.passed:
                    results["success"].append((pair, file_meta))
                else:
                    results["partial"].append((pair, file_meta, validation_report))

                progress_callback(f"Completed {pair}", (i + 1) / total_pairs)

            except Exception as e:
                results["failed"].append((pair, str(e)))
                logger.log_exception("Collection Error", str(e), e, symbol=pair)

        return results


class OptionsTab(BaseTab):
    """Tab for options data collection."""

    def __init__(self, parent, app: 'DataCollectionApp', **kwargs):
        super().__init__(parent, app, **kwargs)
        self._create_widgets()

    def _create_widgets(self):
        container = ctk.CTkScrollableFrame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        self.ticker_selector = TickerSelector(container, self._get_existing_tickers())
        self.ticker_selector.pack(fill="x", pady=5)

        # Options type
        type_frame = ctk.CTkFrame(container)
        type_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(type_frame, text="Option Types", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.opt_type_var = tk.StringVar(value="both")
        ctk.CTkRadioButton(type_frame, text="Both Calls & Puts", variable=self.opt_type_var, value="both").pack(anchor="w", padx=20)
        ctk.CTkRadioButton(type_frame, text="Calls Only", variable=self.opt_type_var, value="calls").pack(anchor="w", padx=20)
        ctk.CTkRadioButton(type_frame, text="Puts Only", variable=self.opt_type_var, value="puts").pack(anchor="w", padx=20)

        # Date selection
        date_frame = ctk.CTkFrame(container)
        date_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(date_frame, text="Historical Date (Optional)", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        date_input_frame = ctk.CTkFrame(date_frame, fg_color="transparent")
        date_input_frame.pack(fill="x", padx=20)

        ctk.CTkLabel(date_input_frame, text="Date (YYYY-MM-DD):").pack(side="left")
        self.date_entry = ctk.CTkEntry(date_input_frame, width=120, placeholder_text="Leave empty for latest")
        self.date_entry.pack(side="left", padx=10)

        # Data fields
        fields_frame = ctk.CTkFrame(container)
        fields_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(fields_frame, text="Data Fields", font=("", 14, "bold")).pack(anchor="w", padx=5, pady=5)

        self.field_vars = {}
        fields = ["strike", "bid", "ask", "last_price", "volume", "open_interest",
                  "implied_volatility", "delta", "gamma", "theta", "vega"]

        for field in fields:
            self.field_vars[field] = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(fields_frame, text=field.replace("_", " ").title(), variable=self.field_vars[field]).pack(anchor="w", padx=20)

    def _get_existing_tickers(self) -> List[str]:
        raw_data_path = Path("raw_data")
        tickers = []
        if raw_data_path.exists():
            for f in raw_data_path.glob("*.csv"):
                ticker = f.stem.split("_")[0].upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
        return sorted(tickers)

    def get_config(self) -> Dict[str, Any]:
        selected_fields = [f for f, var in self.field_vars.items() if var.get()]

        return {
            "tickers": self.ticker_selector.get_selected(),
            "option_type": self.opt_type_var.get(),
            "date": self.date_entry.get().strip() or None,
            "fields": selected_fields,
        }

    def validate(self) -> tuple:
        tickers = self.ticker_selector.get_selected()
        if not tickers:
            return False, "Please select at least one ticker"

        date_str = self.date_entry.get().strip()
        if date_str:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return False, "Invalid date format. Use YYYY-MM-DD"

        return True, ""

    def collect_data(self, progress_callback: Callable) -> Dict[str, Any]:
        """Collect options data."""
        config = self.get_config()
        tickers = config["tickers"]
        results = {"success": [], "failed": [], "partial": [], "not_available": []}

        client = self.app.api_client
        file_manager = self.app.file_manager
        validator = self.app.validator
        logger = self.app.logger

        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            if self.app.cancelled:
                break

            try:
                progress_callback(f"Fetching {ticker} options data...", i / total_tickers)
                logger.log_session_info(f"FETCHING {ticker} OPTIONS")

                response = client.get_historical_options(ticker, config["date"])

                if not response.success:
                    if "not available" in (response.error_message or "").lower():
                        results["not_available"].append((ticker, response.error_message))
                        logger.log_data_issue(DataIssueEntry(
                            timestamp=datetime.now(),
                            symbol=ticker,
                            issue_type="Data Not Available",
                            description=f"Options data not available: {response.error_message}",
                            severity="warning"
                        ))
                    else:
                        results["failed"].append((ticker, response.error_message))
                    continue

                df = DataTransformer.transform_options(response.data, ticker)

                if df.empty:
                    results["not_available"].append((ticker, "No options data returned"))
                    continue

                # Filter by option type
                if config["option_type"] == "calls":
                    df = df[df["option_type"] == "CALL"]
                elif config["option_type"] == "puts":
                    df = df[df["option_type"] == "PUT"]

                if df.empty:
                    results["failed"].append((ticker, "No options match type filter"))
                    continue

                # Group by expiration and write separate files
                if "expiration_date" in df.columns:
                    for expiration, exp_df in df.groupby("expiration_date"):
                        exp_str = pd.to_datetime(expiration).strftime("%Y%m%d")

                        if config["option_type"] in ["both", "calls"]:
                            calls_df = exp_df[exp_df["option_type"] == "CALL"]
                            if not calls_df.empty:
                                file_manager.write_options_data(calls_df, ticker, exp_str, "calls")

                        if config["option_type"] in ["both", "puts"]:
                            puts_df = exp_df[exp_df["option_type"] == "PUT"]
                            if not puts_df.empty:
                                file_manager.write_options_data(puts_df, ticker, exp_str, "puts")

                results["success"].append((ticker, len(df)))
                progress_callback(f"Completed {ticker}", (i + 1) / total_tickers)

            except Exception as e:
                results["failed"].append((ticker, str(e)))
                logger.log_exception("Collection Error", str(e), e, symbol=ticker)

        return results


class DataCollectionApp(ctk.CTk):
    """Main application window for the Data Collection System."""

    CONFIG_PATH = Path("config/data_collection/settings.json")

    def __init__(self):
        super().__init__()

        self.title("Raw Data Collection System")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Initialize configuration
        self.config = self._load_config()

        # Initialize components (will be set when collection starts)
        self.api_client: Optional[AlphaVantageClient] = None
        self.file_manager: Optional[FileManager] = None
        self.validator: Optional[ValidationEngine] = None
        self.logger: Optional[SessionLogger] = None
        self.logging_manager = LoggingManager(self.config.logs_dir)

        # State
        self.cancelled = False
        self.collection_thread: Optional[threading.Thread] = None

        # Create UI
        self._create_widgets()

        # Check API key on startup
        self.after(500, self._check_api_key)

    def _load_config(self) -> DataCollectionConfig:
        """Load configuration from file or create default."""
        if self.CONFIG_PATH.exists():
            try:
                return DataCollectionConfig.load(self.CONFIG_PATH)
            except Exception:
                pass

        # Create default config
        config = DataCollectionConfig(
            api=APIConfig(rate_limit_per_minute=75),
            cache=CacheConfig(),
            validation=ValidationConfig(),
            output_dir=Path("raw_data"),
            logs_dir=Path("logs/data_collection"),
        )

        # Ensure directories exist
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)

        return config

    def _save_config(self, config: DataCollectionConfig):
        """Save configuration to file."""
        self.config = config
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.save(self.CONFIG_PATH)

    def _check_api_key(self):
        """Check if API key is configured."""
        if not self.config.api.api_key:
            messagebox.showwarning(
                "API Key Required",
                "Please configure your Alpha Vantage API key in Settings before collecting data."
            )

    def _create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="Raw Data Collection System",
            font=("", 24, "bold")
        ).pack(side="left")

        # Header buttons
        btn_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        btn_frame.pack(side="right")

        ctk.CTkButton(btn_frame, text="Settings", width=100, command=self._open_settings).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Clear Cache", width=100, command=self._clear_cache).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="View Logs", width=100, command=self._view_logs).pack(side="left", padx=5)

        # Tabview for data types
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.pack(fill="both", expand=True, pady=10)

        # Create tabs
        self.tabs = {}

        self.tabview.add("Daily")
        self.tabs["daily"] = DailyTab(self.tabview.tab("Daily"), self)
        self.tabs["daily"].pack(fill="both", expand=True)

        self.tabview.add("Weekly")
        self.tabs["weekly"] = WeeklyTab(self.tabview.tab("Weekly"), self)
        self.tabs["weekly"].pack(fill="both", expand=True)

        self.tabview.add("Fundamental")
        self.tabs["fundamental"] = FundamentalTab(self.tabview.tab("Fundamental"), self)
        self.tabs["fundamental"].pack(fill="both", expand=True)

        self.tabview.add("Insider")
        self.tabs["insider"] = InsiderTab(self.tabview.tab("Insider"), self)
        self.tabs["insider"].pack(fill="both", expand=True)

        self.tabview.add("Forex")
        self.tabs["forex"] = ForexTab(self.tabview.tab("Forex"), self)
        self.tabs["forex"].pack(fill="both", expand=True)

        self.tabview.add("Options")
        self.tabs["options"] = OptionsTab(self.tabview.tab("Options"), self)
        self.tabs["options"].pack(fill="both", expand=True)

        # Bottom section: Progress and controls
        bottom_frame = ctk.CTkFrame(main_frame)
        bottom_frame.pack(fill="x", pady=10)

        # Control buttons
        control_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        control_frame.pack(fill="x", padx=10, pady=10)

        self.run_button = ctk.CTkButton(
            control_frame, text="Run Collection", width=150,
            fg_color="green", hover_color="darkgreen",
            command=self._start_collection
        )
        self.run_button.pack(side="left", padx=10)

        self.cancel_button = ctk.CTkButton(
            control_frame, text="Cancel", width=100,
            fg_color="red", hover_color="darkred",
            command=self._cancel_collection, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=10)

        ctk.CTkButton(
            control_frame, text="Exit", width=100,
            command=self._on_exit
        ).pack(side="right", padx=10)

        # Progress panel
        self.progress_panel = ProgressPanel(bottom_frame)
        self.progress_panel.pack(fill="both", expand=True, padx=10, pady=10)

    def _open_settings(self):
        """Open settings dialog."""
        SettingsDialog(self, self.config, self._save_config)

    def _clear_cache(self):
        """Clear the API cache."""
        if messagebox.askyesno("Confirm", "Clear all cached API responses?"):
            try:
                cache_dir = self.config.cache.cache_dir
                if cache_dir.exists():
                    for f in cache_dir.glob("*.json"):
                        f.unlink()
                messagebox.showinfo("Success", "Cache cleared successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear cache: {e}")

    def _view_logs(self):
        """Open logs directory."""
        import subprocess
        import sys

        logs_dir = self.config.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            subprocess.run(["explorer", str(logs_dir)])
        elif sys.platform == "darwin":
            subprocess.run(["open", str(logs_dir)])
        else:
            subprocess.run(["xdg-open", str(logs_dir)])

    def _get_current_tab(self) -> tuple:
        """Get the current active tab and its name."""
        tab_name = self.tabview.get()
        tab_key = tab_name.lower()
        return tab_key, self.tabs.get(tab_key)

    def _start_collection(self):
        """Start data collection for the current tab."""
        # Check API key
        if not self.config.api.api_key:
            messagebox.showerror("Error", "Please configure your API key in Settings first.")
            return

        # Get current tab
        tab_key, current_tab = self._get_current_tab()

        if not current_tab:
            messagebox.showerror("Error", "Invalid tab selected")
            return

        # Validate tab inputs
        is_valid, error_msg = current_tab.validate()
        if not is_valid:
            messagebox.showerror("Validation Error", error_msg)
            return

        # Confirm start
        config = current_tab.get_config()
        item_count = len(config.get("tickers", config.get("pairs", [])))
        if not messagebox.askyesno(
            "Confirm",
            f"Start collecting {tab_key} data for {item_count} items?\n\n"
            f"This may take several minutes depending on the number of items and indicators selected."
        ):
            return

        # Initialize components
        self.cancelled = False
        self.logger = self.logging_manager.create_session()
        self.api_client = AlphaVantageClient(self.config.api, self.config.cache, self.logger)
        self.file_manager = FileManager(self.config.output_dir, self.logger)
        self.validator = ValidationEngine(self.config.validation, self.logger)

        # Log session start
        self.logger.log_session_info(f"Data Type: {tab_key.title()}")
        self.logger.log_configuration(config)

        # Update UI
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_panel.clear()
        self.progress_panel.log_message(f"Starting {tab_key} data collection...", "info")

        # Start collection in background thread
        self.collection_thread = threading.Thread(
            target=self._run_collection,
            args=(current_tab, tab_key),
            daemon=True
        )
        self.collection_thread.start()

    def _run_collection(self, tab: BaseTab, tab_key: str):
        """Run collection in background thread."""
        try:
            def progress_callback(message: str, progress: float):
                self.after(0, lambda: self._update_progress(message, progress))

            # Run collection
            results = tab.collect_data(progress_callback)

            # Finalize
            self.after(0, lambda: self._collection_complete(results, tab_key))

        except Exception as e:
            self.after(0, lambda: self._collection_error(str(e)))

    def _update_progress(self, message: str, progress: float):
        """Update progress UI (called from main thread)."""
        self.progress_panel.set_progress(progress, message)
        self.progress_panel.log_message(message, "info")

    def _collection_complete(self, results: Dict[str, Any], tab_key: str):
        """Handle collection completion."""
        # Re-enable buttons
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

        # Log results
        success_count = len(results.get("success", []))
        failed_count = len(results.get("failed", []))
        partial_count = len(results.get("partial", []))
        not_available_count = len(results.get("not_available", []))

        self.progress_panel.set_progress(1.0, "Collection complete!")

        # Log summary
        self.progress_panel.log_message(f"Collection completed for {tab_key} data", "success")
        self.progress_panel.log_message(f"  Successful: {success_count}", "info")

        if partial_count > 0:
            self.progress_panel.log_message(f"  Partial (with warnings): {partial_count}", "warning")

        if failed_count > 0:
            self.progress_panel.log_message(f"  Failed: {failed_count}", "error")
            for item, error in results.get("failed", [])[:5]:
                self.progress_panel.log_message(f"    - {item}: {error}", "error")

        if not_available_count > 0:
            self.progress_panel.log_message(f"  Data not available: {not_available_count}", "warning")
            for item, msg in results.get("not_available", [])[:5]:
                self.progress_panel.log_message(f"    - {item}: {msg}", "warning")

        # Finalize logging
        if self.logger:
            summary = self.logger.finalize(exit_code=0 if failed_count == 0 else 1)
            self.progress_panel.log_message(f"Logs saved to: {self.logger.session_dir}", "info")

        # Save metadata
        if self.file_manager:
            session_meta = self.file_manager.create_session_metadata(
                self.logger.session_id if self.logger else "unknown",
                [tab_key],
                [item[0] for item in results.get("success", [])]
            )
            self.file_manager.save_session_metadata(session_meta)

        # Show summary dialog
        messagebox.showinfo(
            "Collection Complete",
            f"Data collection completed!\n\n"
            f"Successful: {success_count}\n"
            f"Partial: {partial_count}\n"
            f"Failed: {failed_count}\n"
            f"Not Available: {not_available_count}\n\n"
            f"Files saved to: {self.config.output_dir}"
        )

    def _collection_error(self, error: str):
        """Handle collection error."""
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

        self.progress_panel.log_message(f"Collection failed: {error}", "error")

        if self.logger:
            self.logger.finalize(exit_code=1)

        messagebox.showerror("Collection Error", f"An error occurred:\n\n{error}")

    def _cancel_collection(self):
        """Cancel ongoing collection."""
        if messagebox.askyesno("Confirm", "Cancel the current collection? Progress will be saved."):
            self.cancelled = True
            if self.api_client:
                self.api_client.cancel()

            self.progress_panel.log_message("Cancellation requested...", "warning")
            self.cancel_button.configure(state="disabled")

    def _on_exit(self):
        """Handle application exit."""
        if self.collection_thread and self.collection_thread.is_alive():
            if not messagebox.askyesno("Confirm", "Collection is in progress. Exit anyway?"):
                return
            self.cancelled = True
            if self.api_client:
                self.api_client.cancel()

        self.destroy()


def main():
    """Main entry point."""
    app = DataCollectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
