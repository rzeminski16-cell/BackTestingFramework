"""
Factor Analysis Data Upload GUI.

A CustomTkinter application for uploading and preparing data for factor analysis.
Supports trade logs, price data, fundamental data, insider data, and options data.
"""

import customtkinter as ctk
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime
import threading
import queue

from ..ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info, ask_yes_no, ProgressPanel
from .components import DataQualityIndicator


class FileUploadPanel(ctk.CTkFrame):
    """
    Panel for uploading a single data file.

    Features:
    - File path entry
    - Browse button
    - File type indicator
    - Preview button
    - Status indicator
    """

    def __init__(
        self,
        parent,
        title: str,
        file_types: List[Tuple[str, str]] = None,
        required: bool = True,
        on_file_selected: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.title = title
        self.file_types = file_types or [("CSV files", "*.csv"), ("All files", "*.*")]
        self.required = required
        self.on_file_selected = on_file_selected
        self.file_path: Optional[str] = None
        self.data: Optional[pd.DataFrame] = None

        self._create_widgets()

    def _create_widgets(self):
        """Create panel widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, self.title, size="s").pack(side="left")

        if self.required:
            Theme.create_label(
                header, "(Required)",
                font=Fonts.BODY_XS,
                text_color=Colors.ERROR
            ).pack(side="left", padx=(Sizes.PAD_S, 0))
        else:
            Theme.create_label(
                header, "(Optional)",
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            ).pack(side="left", padx=(Sizes.PAD_S, 0))

        self.status_label = Theme.create_label(
            header, "",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        )
        self.status_label.pack(side="right")

        # File path row
        path_frame = Theme.create_frame(self)
        path_frame.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)

        self.path_var = ctk.StringVar()
        self.path_entry = Theme.create_entry(path_frame, placeholder="Select file...")
        self.path_entry.configure(textvariable=self.path_var)
        self.path_entry.pack(side="left", fill="x", expand=True)

        Theme.create_button(
            path_frame, "Browse",
            command=self._browse,
            style="secondary",
            width=80,
            height=32
        ).pack(side="left", padx=(Sizes.PAD_S, 0))

        # Actions row
        actions_frame = Theme.create_frame(self)
        actions_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self.preview_btn = Theme.create_button(
            actions_frame, "Preview",
            command=self._preview,
            style="ghost",
            width=80,
            height=28
        )
        self.preview_btn.pack(side="left")
        self.preview_btn.configure(state="disabled")

        self.clear_btn = Theme.create_button(
            actions_frame, "Clear",
            command=self._clear,
            style="ghost",
            width=60,
            height=28
        )
        self.clear_btn.pack(side="left", padx=(Sizes.PAD_S, 0))
        self.clear_btn.configure(state="disabled")

        # Info label
        self.info_label = Theme.create_label(
            actions_frame, "",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        )
        self.info_label.pack(side="right")

    def _browse(self):
        """Open file browser dialog."""
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title=f"Select {self.title}",
            filetypes=self.file_types
        )

        if file_path:
            self.set_file(file_path)

    def set_file(self, file_path: str):
        """Set the file path and load preview."""
        self.file_path = file_path
        self.path_var.set(file_path)

        # Try to load and validate
        try:
            self.data = pd.read_csv(file_path, nrows=1000)  # Preview first 1000 rows
            rows = len(pd.read_csv(file_path))
            cols = len(self.data.columns)

            self.status_label.configure(text="Loaded", text_color=Colors.SUCCESS)
            self.info_label.configure(text=f"{rows:,} rows, {cols} columns")
            self.preview_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")

            if self.on_file_selected:
                self.on_file_selected(file_path)

        except Exception as e:
            self.status_label.configure(text="Error", text_color=Colors.ERROR)
            self.info_label.configure(text=str(e)[:50])
            self.data = None

    def _preview(self):
        """Show data preview dialog."""
        if self.data is not None:
            PreviewDialog(self.winfo_toplevel(), self.title, self.data)

    def _clear(self):
        """Clear the selected file."""
        self.file_path = None
        self.data = None
        self.path_var.set("")
        self.status_label.configure(text="", text_color=Colors.TEXT_MUTED)
        self.info_label.configure(text="")
        self.preview_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")

    def get_file_path(self) -> Optional[str]:
        """Get the selected file path."""
        return self.file_path

    def is_valid(self) -> bool:
        """Check if file is valid (or not required)."""
        if not self.required:
            return True
        return self.file_path is not None and self.data is not None


class ColumnMappingPanel(ctk.CTkFrame):
    """
    Panel for mapping data columns to expected fields.

    Features:
    - Auto-detection of columns
    - Manual mapping dropdowns
    - Required field indicators
    """

    def __init__(
        self,
        parent,
        title: str,
        required_fields: List[str],
        optional_fields: List[str] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.title = title
        self.required_fields = required_fields
        self.optional_fields = optional_fields or []
        self.available_columns: List[str] = []
        self.mapping_vars: Dict[str, ctk.StringVar] = {}

        self._create_widgets()

    def _create_widgets(self):
        """Create panel widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, self.title, size="s").pack(side="left")

        self.auto_detect_btn = Theme.create_button(
            header, "Auto-Detect",
            command=self._auto_detect,
            style="ghost",
            width=100,
            height=28
        )
        self.auto_detect_btn.pack(side="right")

        # Mapping grid
        self.mapping_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            height=200
        )
        self.mapping_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        self._create_mappings()

    def _create_mappings(self):
        """Create mapping rows for all fields."""
        # Required fields
        if self.required_fields:
            Theme.create_label(
                self.mapping_frame, "Required Fields",
                font=Fonts.LABEL_BOLD,
                text_color=Colors.PRIMARY_LIGHT
            ).pack(anchor="w", pady=(0, Sizes.PAD_S))

            for field in self.required_fields:
                self._create_mapping_row(field, required=True)

        # Optional fields
        if self.optional_fields:
            Theme.create_label(
                self.mapping_frame, "Optional Fields",
                font=Fonts.LABEL_BOLD,
                text_color=Colors.TEXT_SECONDARY
            ).pack(anchor="w", pady=(Sizes.PAD_M, Sizes.PAD_S))

            for field in self.optional_fields:
                self._create_mapping_row(field, required=False)

    def _create_mapping_row(self, field: str, required: bool):
        """Create a single mapping row."""
        row = Theme.create_frame(self.mapping_frame)
        row.pack(fill="x", pady=Sizes.PAD_XS)

        label_text = f"{field}:"
        if required:
            label_text += " *"

        Theme.create_label(
            row, label_text,
            font=Fonts.BODY_S,
            width=150
        ).pack(side="left")

        var = ctk.StringVar(value="(not mapped)")
        self.mapping_vars[field] = var

        combo = Theme.create_combobox(
            row,
            values=["(not mapped)"] + self.available_columns,
            variable=var,
            width=180
        )
        combo.pack(side="left", padx=(Sizes.PAD_S, 0))

    def set_columns(self, columns: List[str]):
        """Set available columns from loaded data."""
        self.available_columns = columns

        # Update all comboboxes
        for field, var in self.mapping_vars.items():
            # Find the combobox widget - need to traverse children
            pass  # Will be updated when columns change

        # Rebuild mappings
        for widget in self.mapping_frame.winfo_children():
            widget.destroy()
        self._create_mappings()

        # Try auto-detection
        self._auto_detect()

    def _auto_detect(self):
        """Auto-detect column mappings based on column names."""
        if not self.available_columns:
            return

        # Common mapping patterns
        patterns = {
            'date': ['date', 'timestamp', 'time', 'dt', 'trade_date'],
            'symbol': ['symbol', 'ticker', 'security', 'stock'],
            'entry_price': ['entry_price', 'entry', 'buy_price', 'open_price'],
            'exit_price': ['exit_price', 'exit', 'sell_price', 'close_price'],
            'quantity': ['quantity', 'qty', 'shares', 'size', 'volume'],
            'direction': ['direction', 'side', 'type', 'action'],
            'pnl': ['pnl', 'pl', 'profit', 'profit_loss', 'return'],
            'return_pct': ['return_pct', 'return_percent', 'pct_return', 'pct_pnl'],
        }

        for field, var in self.mapping_vars.items():
            field_lower = field.lower().replace(' ', '_')

            # Check patterns
            if field_lower in patterns:
                for pattern in patterns[field_lower]:
                    for col in self.available_columns:
                        if pattern in col.lower():
                            var.set(col)
                            break
                    if var.get() != "(not mapped)":
                        break
            else:
                # Try direct match
                for col in self.available_columns:
                    if field_lower in col.lower() or col.lower() in field_lower:
                        var.set(col)
                        break

    def get_mapping(self) -> Dict[str, Optional[str]]:
        """Get the current column mapping."""
        mapping = {}
        for field, var in self.mapping_vars.items():
            value = var.get()
            mapping[field] = None if value == "(not mapped)" else value
        return mapping

    def is_valid(self) -> Tuple[bool, List[str]]:
        """Check if all required fields are mapped."""
        errors = []
        for field in self.required_fields:
            var = self.mapping_vars.get(field)
            if not var or var.get() == "(not mapped)":
                errors.append(f"Required field '{field}' is not mapped")

        return len(errors) == 0, errors


class PreviewDialog(ctk.CTkToplevel):
    """Dialog for previewing data."""

    def __init__(self, parent, title: str, data: pd.DataFrame):
        super().__init__(parent)

        self.title(f"Preview: {title}")
        self.geometry("800x500")
        self.configure(fg_color=Colors.BG_DARK)

        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 800) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 500) // 2
        self.geometry(f"+{x}+{y}")

        self._create_widgets(data)

    def _create_widgets(self, data: pd.DataFrame):
        """Create dialog widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        Theme.create_label(
            header,
            f"Showing first {len(data)} rows of {len(data.columns)} columns",
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left")

        Theme.create_button(
            header, "Close",
            command=self.destroy,
            style="secondary",
            width=80
        ).pack(side="right")

        # Column headers
        cols_frame = Theme.create_frame(self)
        cols_frame.pack(fill="x", padx=Sizes.PAD_M)

        Theme.create_label(
            cols_frame,
            "Columns: " + ", ".join(data.columns[:10]) + ("..." if len(data.columns) > 10 else ""),
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED,
            wraplength=750
        ).pack(anchor="w")

        # Data preview
        preview_frame = ctk.CTkScrollableFrame(self, fg_color=Colors.SURFACE)
        preview_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=Sizes.PAD_M)

        # Create text representation
        preview_text = ctk.CTkTextbox(
            preview_frame,
            font=Fonts.MONO_S,
            fg_color=Colors.BG_DARK
        )
        preview_text.pack(fill="both", expand=True)

        # Add data as text table
        text = data.head(50).to_string()
        preview_text.insert("1.0", text)
        preview_text.configure(state="disabled")


class FactorDataUploadGUI:
    """
    Data Upload and Preparation GUI for Factor Analysis.

    Provides interface for:
    - Uploading trade log data
    - Uploading supplementary data (price, fundamental, insider, options)
    - Column mapping
    - Data quality validation
    - Data preparation and export
    """

    def __init__(
        self,
        parent: ctk.CTk = None,
        on_data_ready: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the Data Upload GUI.

        Args:
            parent: Parent window (if modal)
            on_data_ready: Callback when data is prepared and ready
        """
        Theme.setup()

        self.on_data_ready = on_data_ready
        self.prepared_data: Dict[str, Any] = {}

        # Create window
        if parent:
            self.root = ctk.CTkToplevel(parent)
            self.root.transient(parent)
        else:
            self.root = ctk.CTk()

        self.root.title("Factor Analysis - Data Upload")
        self.root.geometry("950x750")
        self.root.minsize(800, 600)
        self.root.configure(fg_color=Colors.BG_DARK)

        self._create_layout()

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

        Theme.create_header(header_content, "Data Upload & Preparation", size="l").pack(side="left")

        Theme.create_label(
            header_content,
            "Upload trade logs and supplementary data for factor analysis",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(Sizes.PAD_L, 0))

        # Main content - two columns
        content_container = Theme.create_frame(self.root)
        content_container.grid(row=1, column=0, sticky="nsew", padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        content_container.grid_columnconfigure(0, weight=2)
        content_container.grid_columnconfigure(1, weight=1)
        content_container.grid_rowconfigure(0, weight=1)

        # Left column - File uploads
        left_column = ctk.CTkScrollableFrame(content_container, fg_color="transparent")
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PAD_S))

        self._create_upload_panels(left_column)

        # Right column - Quality & mapping
        right_column = Theme.create_frame(content_container)
        right_column.grid(row=0, column=1, sticky="nsew")

        self._create_quality_panel(right_column)
        self._create_mapping_panel(right_column)

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
            footer_content, "Prepare Data",
            command=self._on_prepare,
            style="primary",
            width=120
        ).pack(side="right")

        Theme.create_button(
            footer_content, "Validate",
            command=self._on_validate,
            style="secondary",
            width=100
        ).pack(side="right", padx=(0, Sizes.PAD_S))

    def _create_upload_panels(self, parent):
        """Create file upload panels."""
        # Trade Log (Required)
        self.trade_log_panel = FileUploadPanel(
            parent,
            title="Trade Log",
            file_types=[("CSV files", "*.csv"), ("JSON files", "*.json")],
            required=True,
            on_file_selected=self._on_trade_log_selected
        )
        self.trade_log_panel.pack(fill="x", pady=(0, Sizes.PAD_S))

        # Price Data (Optional)
        self.price_data_panel = FileUploadPanel(
            parent,
            title="Price Data",
            file_types=[("CSV files", "*.csv")],
            required=False,
            on_file_selected=self._on_price_data_selected
        )
        self.price_data_panel.pack(fill="x", pady=Sizes.PAD_S)

        # Fundamental Data (Optional)
        self.fundamental_panel = FileUploadPanel(
            parent,
            title="Fundamental Data",
            file_types=[("CSV files", "*.csv")],
            required=False,
            on_file_selected=self._on_fundamental_selected
        )
        self.fundamental_panel.pack(fill="x", pady=Sizes.PAD_S)

        # Insider Data (Optional)
        self.insider_panel = FileUploadPanel(
            parent,
            title="Insider Data",
            file_types=[("CSV files", "*.csv")],
            required=False,
            on_file_selected=self._on_insider_selected
        )
        self.insider_panel.pack(fill="x", pady=Sizes.PAD_S)

        # Options Data (Optional)
        self.options_panel = FileUploadPanel(
            parent,
            title="Options Data",
            file_types=[("CSV files", "*.csv")],
            required=False,
            on_file_selected=self._on_options_selected
        )
        self.options_panel.pack(fill="x", pady=Sizes.PAD_S)

    def _create_quality_panel(self, parent):
        """Create data quality panel."""
        self.quality_indicator = DataQualityIndicator(parent)
        self.quality_indicator.pack(fill="x", pady=(0, Sizes.PAD_S))

        # Initialize with unknown status
        self.quality_indicator.set_quality(
            score=0,
            status="No data loaded",
            metrics={"Files loaded": "0", "Total records": "0"}
        )

    def _create_mapping_panel(self, parent):
        """Create column mapping panel."""
        self.mapping_panel = ColumnMappingPanel(
            parent,
            title="Trade Log Column Mapping",
            required_fields=["date", "symbol", "entry_price", "exit_price", "quantity"],
            optional_fields=["direction", "pnl", "return_pct", "entry_date", "exit_date", "stop_loss", "take_profit"]
        )
        self.mapping_panel.pack(fill="both", expand=True, pady=Sizes.PAD_S)

    def _on_trade_log_selected(self, file_path: str):
        """Handle trade log file selection."""
        if self.trade_log_panel.data is not None:
            columns = list(self.trade_log_panel.data.columns)
            self.mapping_panel.set_columns(columns)
            self._update_quality()

    def _on_price_data_selected(self, file_path: str):
        """Handle price data file selection."""
        self._update_quality()

    def _on_fundamental_selected(self, file_path: str):
        """Handle fundamental data file selection."""
        self._update_quality()

    def _on_insider_selected(self, file_path: str):
        """Handle insider data file selection."""
        self._update_quality()

    def _on_options_selected(self, file_path: str):
        """Handle options data file selection."""
        self._update_quality()

    def _update_quality(self):
        """Update data quality indicator."""
        files_loaded = 0
        total_records = 0
        warnings = []

        # Count files and records
        if self.trade_log_panel.data is not None:
            files_loaded += 1
            total_records += len(self.trade_log_panel.data)

        if self.price_data_panel.data is not None:
            files_loaded += 1
            total_records += len(self.price_data_panel.data)

        if self.fundamental_panel.data is not None:
            files_loaded += 1
            total_records += len(self.fundamental_panel.data)

        if self.insider_panel.data is not None:
            files_loaded += 1
            total_records += len(self.insider_panel.data)

        if self.options_panel.data is not None:
            files_loaded += 1
            total_records += len(self.options_panel.data)

        # Calculate quality score
        if files_loaded == 0:
            score = 0
            status = "No data loaded"
        else:
            score = min(100, files_loaded * 20 + 20)

            # Check mapping validity
            is_valid, errors = self.mapping_panel.is_valid()
            if not is_valid:
                warnings.extend(errors)
                score -= len(errors) * 10

            if score >= 80:
                status = "Good"
            elif score >= 60:
                status = "Acceptable"
            else:
                status = "Needs attention"

        self.quality_indicator.set_quality(
            score=max(0, score),
            status=status,
            metrics={
                "Files loaded": str(files_loaded),
                "Total records": f"{total_records:,}"
            },
            warnings=warnings if warnings else None
        )

    def _on_validate(self):
        """Validate all uploaded data."""
        errors = []

        # Check required files
        if not self.trade_log_panel.is_valid():
            errors.append("Trade log file is required")

        # Check column mapping
        is_valid, mapping_errors = self.mapping_panel.is_valid()
        errors.extend(mapping_errors)

        if errors:
            show_error(self.root, "Validation Failed", "\n".join(errors))
        else:
            show_info(self.root, "Validation Passed", "All data is valid and ready for preparation.")
            self._update_quality()

    def _on_prepare(self):
        """Prepare data for analysis."""
        # Validate first
        errors = []

        if not self.trade_log_panel.is_valid():
            errors.append("Trade log file is required")

        is_valid, mapping_errors = self.mapping_panel.is_valid()
        errors.extend(mapping_errors)

        if errors:
            show_error(self.root, "Cannot Prepare", "\n".join(errors))
            return

        # Show progress dialog
        progress = PrepareProgressDialog(self.root)

        def prepare_thread():
            try:
                # Load and prepare data
                progress.update(0.1, "Loading trade log...")
                trade_data = self._load_trade_log()

                progress.update(0.3, "Loading price data...")
                price_data = self._load_optional_data(self.price_data_panel)

                progress.update(0.5, "Loading fundamental data...")
                fundamental_data = self._load_optional_data(self.fundamental_panel)

                progress.update(0.7, "Loading insider data...")
                insider_data = self._load_optional_data(self.insider_panel)

                progress.update(0.9, "Loading options data...")
                options_data = self._load_optional_data(self.options_panel)

                self.prepared_data = {
                    'trade_data': trade_data,
                    'price_data': price_data,
                    'fundamental_data': fundamental_data,
                    'insider_data': insider_data,
                    'options_data': options_data,
                    'column_mapping': self.mapping_panel.get_mapping()
                }

                progress.complete()

                # Call callback
                if self.on_data_ready:
                    self.root.after(100, lambda: self.on_data_ready(self.prepared_data))

            except Exception as e:
                progress.error(str(e))

        # Run in thread
        thread = threading.Thread(target=prepare_thread, daemon=True)
        thread.start()

    def _load_trade_log(self) -> pd.DataFrame:
        """Load and transform trade log with column mapping."""
        file_path = self.trade_log_panel.get_file_path()
        if not file_path:
            raise ValueError("No trade log file selected")

        data = pd.read_csv(file_path)

        # Apply column mapping
        mapping = self.mapping_panel.get_mapping()
        rename_map = {}
        for target, source in mapping.items():
            if source and source in data.columns:
                rename_map[source] = target

        if rename_map:
            data = data.rename(columns=rename_map)

        return data

    def _load_optional_data(self, panel: FileUploadPanel) -> Optional[pd.DataFrame]:
        """Load optional data file."""
        file_path = panel.get_file_path()
        if not file_path:
            return None

        return pd.read_csv(file_path)

    def _on_cancel(self):
        """Handle cancel button click."""
        if ask_yes_no(self.root, "Cancel", "Discard uploaded data?"):
            self.root.destroy()

    def run(self):
        """Run the data upload GUI."""
        self.root.mainloop()

    def get_prepared_data(self) -> Dict[str, Any]:
        """Get the prepared data."""
        return self.prepared_data


class PrepareProgressDialog(ctk.CTkToplevel):
    """Progress dialog for data preparation."""

    def __init__(self, parent):
        super().__init__(parent)

        self.title("Preparing Data")
        self.geometry("400x180")
        self.resizable(False, False)
        self.configure(fg_color=Colors.BG_DARK)

        self.transient(parent)
        self.grab_set()

        # Center
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 180) // 2
        self.geometry(f"+{x}+{y}")

        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", lambda: None)

    def _create_widgets(self):
        """Create dialog widgets."""
        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        self.message_label = Theme.create_label(
            content, "Initializing...",
            font=Fonts.BODY_M
        )
        self.message_label.pack(pady=(0, Sizes.PAD_M))

        self.progress_bar = Theme.create_progressbar(content)
        self.progress_bar.pack(fill="x", pady=Sizes.PAD_S)
        self.progress_bar.set(0)

        self.percent_label = Theme.create_label(
            content, "0%",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        self.percent_label.pack()

    def update(self, progress: float, message: str):
        """Update progress and message."""
        self.progress_bar.set(progress)
        self.percent_label.configure(text=f"{int(progress * 100)}%")
        self.message_label.configure(text=message)
        self.update_idletasks()

    def complete(self):
        """Mark as complete and close."""
        self.progress_bar.set(1)
        self.percent_label.configure(text="100%")
        self.message_label.configure(text="Complete!", text_color=Colors.SUCCESS)
        self.after(500, self.destroy)

    def error(self, message: str):
        """Show error and close."""
        self.message_label.configure(text=f"Error: {message}", text_color=Colors.ERROR)
        self.after(2000, self.destroy)


def main():
    """Entry point for Data Upload GUI."""
    app = FactorDataUploadGUI()
    app.run()


if __name__ == "__main__":
    main()
