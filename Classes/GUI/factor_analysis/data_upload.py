"""
Factor Analysis Data Upload GUI.

A CustomTkinter application for uploading and preparing data for factor analysis.
Supports multiple trade logs, automatic raw data discovery, price data,
fundamental data, insider data, and options data.
"""

import customtkinter as ctk
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from pathlib import Path
import pandas as pd
from datetime import datetime
import threading
import queue
import os

from ..ctk_theme import Theme, Colors, Fonts, Sizes, show_error, show_info, ask_yes_no, ProgressPanel
from .components import DataQualityIndicator


# Default raw data folder
RAW_DATA_FOLDER = Path(__file__).parent.parent.parent.parent / "raw_data"


class MultiFileUploadPanel(ctk.CTkFrame):
    """
    Panel for uploading multiple trade log files.

    Features:
    - Add/remove multiple files
    - File status indicators
    - Preview button for each file
    - Symbol extraction
    """

    def __init__(
        self,
        parent,
        title: str,
        file_types: List[Tuple[str, str]] = None,
        on_files_changed: Optional[Callable[[List[str]], None]] = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.title = title
        self.file_types = file_types or [("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        self.on_files_changed = on_files_changed
        self.files: List[Dict[str, Any]] = []  # List of {path, data, symbols, status}

        self._create_widgets()

    def _create_widgets(self):
        """Create panel widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, self.title, size="s").pack(side="left")

        self.file_count_label = Theme.create_label(
            header, "(0 files)",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        )
        self.file_count_label.pack(side="left", padx=(Sizes.PAD_S, 0))

        # Add file button
        Theme.create_button(
            header, "+ Add Files",
            command=self._browse,
            style="secondary",
            width=100,
            height=28
        ).pack(side="right")

        # File list container
        self.file_list_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            height=150
        )
        self.file_list_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Empty state message
        self.empty_label = Theme.create_label(
            self.file_list_frame,
            "No trade log files added. Click '+ Add Files' to add trade logs.",
            font=Fonts.BODY_S,
            text_color=Colors.TEXT_MUTED
        )
        self.empty_label.pack(pady=Sizes.PAD_L)

    def _browse(self):
        """Open file browser dialog for multiple files."""
        from tkinter import filedialog

        file_paths = filedialog.askopenfilenames(
            title=f"Select {self.title}",
            filetypes=self.file_types
        )

        if file_paths:
            for path in file_paths:
                self.add_file(path)

    def add_file(self, file_path: str):
        """Add a trade log file."""
        # Check if already added
        for f in self.files:
            if f['path'] == file_path:
                return

        try:
            # Load preview data
            data = pd.read_csv(file_path, nrows=1000)
            full_data = pd.read_csv(file_path)

            # Extract symbols from the data
            symbols = set()
            for col in ['symbol', 'ticker', 'Symbol', 'Ticker', 'SYMBOL', 'TICKER']:
                if col in full_data.columns:
                    symbols = set(full_data[col].dropna().unique())
                    break

            file_info = {
                'path': file_path,
                'data': data,
                'full_data': full_data,
                'symbols': symbols,
                'rows': len(full_data),
                'status': 'loaded'
            }
            self.files.append(file_info)
            self._refresh_list()

            if self.on_files_changed:
                self.on_files_changed([f['path'] for f in self.files])

        except Exception as e:
            show_error(self.winfo_toplevel(), "Error Loading File", f"Could not load {Path(file_path).name}:\n{str(e)}")

    def _refresh_list(self):
        """Refresh the file list display."""
        # Clear existing
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()

        if not self.files:
            self.empty_label = Theme.create_label(
                self.file_list_frame,
                "No trade log files added. Click '+ Add Files' to add trade logs.",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED
            )
            self.empty_label.pack(pady=Sizes.PAD_L)
            self.file_count_label.configure(text="(0 files)")
            return

        self.file_count_label.configure(text=f"({len(self.files)} files)")

        for i, file_info in enumerate(self.files):
            row = ctk.CTkFrame(self.file_list_frame, fg_color=Colors.BG_MEDIUM, corner_radius=Sizes.RADIUS_S)
            row.pack(fill="x", pady=2)

            # File name and info
            info_frame = Theme.create_frame(row)
            info_frame.pack(side="left", fill="x", expand=True, padx=Sizes.PAD_S, pady=Sizes.PAD_XS)

            name = Path(file_info['path']).name
            Theme.create_label(info_frame, name, font=Fonts.BODY_S).pack(anchor="w")

            details = f"{file_info['rows']:,} rows | {len(file_info['symbols'])} symbols"
            Theme.create_label(
                info_frame, details,
                font=Fonts.BODY_XS,
                text_color=Colors.TEXT_MUTED
            ).pack(anchor="w")

            # Status indicator
            status_color = Colors.SUCCESS if file_info['status'] == 'loaded' else Colors.ERROR
            Theme.create_label(
                row, "●",
                font=Fonts.BODY_S,
                text_color=status_color
            ).pack(side="right", padx=Sizes.PAD_XS)

            # Preview button
            Theme.create_button(
                row, "Preview",
                command=lambda f=file_info: self._preview(f),
                style="ghost",
                width=60,
                height=24
            ).pack(side="right", padx=Sizes.PAD_XS)

            # Remove button
            Theme.create_button(
                row, "✕",
                command=lambda idx=i: self._remove_file(idx),
                style="ghost",
                width=30,
                height=24
            ).pack(side="right")

    def _preview(self, file_info: Dict):
        """Show file preview."""
        PreviewDialog(self.winfo_toplevel(), Path(file_info['path']).name, file_info['data'])

    def _remove_file(self, index: int):
        """Remove a file from the list."""
        if 0 <= index < len(self.files):
            self.files.pop(index)
            self._refresh_list()
            if self.on_files_changed:
                self.on_files_changed([f['path'] for f in self.files])

    def get_files(self) -> List[Dict[str, Any]]:
        """Get all loaded files."""
        return self.files

    def get_all_symbols(self) -> Set[str]:
        """Get all unique symbols from all loaded files."""
        symbols = set()
        for f in self.files:
            symbols.update(f['symbols'])
        return symbols

    def is_valid(self) -> bool:
        """Check if at least one file is loaded."""
        return len(self.files) > 0


class RawDataDiscoveryPanel(ctk.CTkFrame):
    """
    Panel for discovering and displaying raw data availability.

    Features:
    - Automatic discovery of raw data files
    - Status per symbol
    - Missing data highlighting
    """

    def __init__(
        self,
        parent,
        raw_data_folder: Path = None,
        **kwargs
    ):
        super().__init__(parent, fg_color=Colors.SURFACE, corner_radius=Sizes.RADIUS_L, **kwargs)

        self.raw_data_folder = raw_data_folder or RAW_DATA_FOLDER
        self.symbols: Set[str] = set()
        self.discovered_data: Dict[str, Dict[str, Any]] = {}

        self._create_widgets()

    def _create_widgets(self):
        """Create panel widgets."""
        # Header
        header = Theme.create_frame(self)
        header.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

        Theme.create_header(header, "Raw Data Discovery", size="s").pack(side="left")

        self.status_label = Theme.create_label(
            header, "No symbols",
            font=Fonts.BODY_XS,
            text_color=Colors.TEXT_MUTED
        )
        self.status_label.pack(side="right")

        # Refresh button
        Theme.create_button(
            header, "Refresh",
            command=self._discover,
            style="ghost",
            width=70,
            height=24
        ).pack(side="right", padx=Sizes.PAD_S)

        # Data availability grid
        self.grid_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            height=200
        )
        self.grid_frame.pack(fill="both", expand=True, padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

        # Column headers
        self._create_headers()

    def _create_headers(self):
        """Create column headers."""
        header_row = Theme.create_frame(self.grid_frame)
        header_row.pack(fill="x", pady=(0, Sizes.PAD_XS))

        Theme.create_label(header_row, "Symbol", font=Fonts.LABEL_BOLD, width=80).pack(side="left")
        Theme.create_label(header_row, "Daily", font=Fonts.LABEL_BOLD, width=60).pack(side="left")
        Theme.create_label(header_row, "Weekly", font=Fonts.LABEL_BOLD, width=60).pack(side="left")
        Theme.create_label(header_row, "Fundamental", font=Fonts.LABEL_BOLD, width=80).pack(side="left")
        Theme.create_label(header_row, "Insider", font=Fonts.LABEL_BOLD, width=60).pack(side="left")
        Theme.create_label(header_row, "Options", font=Fonts.LABEL_BOLD, width=60).pack(side="left")

        # Separator
        ctk.CTkFrame(self.grid_frame, fg_color=Colors.BORDER, height=1).pack(fill="x", pady=Sizes.PAD_XS)

    def set_symbols(self, symbols: Set[str]):
        """Set symbols to discover data for."""
        self.symbols = symbols
        self._discover()

    def _discover(self):
        """Discover available raw data for all symbols."""
        self.discovered_data.clear()

        for symbol in self.symbols:
            self.discovered_data[symbol] = {
                'daily': self._find_file('daily', symbol),
                'weekly': self._find_file('weekly', symbol),
                'fundamental': self._find_file('fundamentals', symbol, '_fundamental'),
                'insider': self._find_file('insider_transactions', symbol, '_insider'),
                'options': self._find_options(symbol)
            }

        self._refresh_display()

    def _find_file(self, folder: str, symbol: str, suffix: str = '') -> Optional[Dict[str, Any]]:
        """Find a data file for a symbol."""
        folder_path = self.raw_data_folder / folder
        if not folder_path.exists():
            return None

        # Try different naming patterns
        patterns = [
            f"{symbol}_daily.csv",
            f"{symbol}_weekly.csv",
            f"{symbol}{suffix}.csv",
            f"{symbol}.csv",
            f"{symbol.upper()}_daily.csv",
            f"{symbol.upper()}_weekly.csv",
            f"{symbol.upper()}{suffix}.csv",
            f"{symbol.upper()}.csv",
        ]

        for pattern in patterns:
            file_path = folder_path / pattern
            if file_path.exists():
                try:
                    # Get row count without loading entire file
                    with open(file_path, 'r') as f:
                        row_count = sum(1 for _ in f) - 1  # Subtract header
                    return {
                        'path': str(file_path),
                        'rows': row_count,
                        'exists': True
                    }
                except Exception:
                    pass

        return None

    def _find_options(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Find options data for a symbol."""
        options_folder = self.raw_data_folder / 'options' / symbol
        if not options_folder.exists():
            options_folder = self.raw_data_folder / 'options' / symbol.upper()

        if not options_folder.exists():
            return None

        files = list(options_folder.glob("*.csv"))
        if not files:
            return None

        total_rows = 0
        for f in files:
            try:
                with open(f, 'r') as file:
                    total_rows += sum(1 for _ in file) - 1
            except Exception:
                pass

        return {
            'path': str(options_folder),
            'files': len(files),
            'rows': total_rows,
            'exists': True
        }

    def _refresh_display(self):
        """Refresh the data grid display."""
        # Clear existing (except headers)
        children = self.grid_frame.winfo_children()
        for widget in children[2:]:  # Skip header and separator
            widget.destroy()

        if not self.symbols:
            Theme.create_label(
                self.grid_frame,
                "Load trade logs to discover available raw data",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.PAD_M)
            self.status_label.configure(text="No symbols")
            return

        # Count available data
        total = len(self.symbols)
        with_all = sum(1 for s in self.symbols if all(
            self.discovered_data.get(s, {}).get(k)
            for k in ['daily', 'weekly', 'fundamental', 'insider']
        ))

        self.status_label.configure(text=f"{with_all}/{total} complete")

        # Create rows for each symbol
        for symbol in sorted(self.symbols):
            data = self.discovered_data.get(symbol, {})
            row = Theme.create_frame(self.grid_frame)
            row.pack(fill="x", pady=1)

            Theme.create_label(row, symbol, font=Fonts.BODY_S, width=80).pack(side="left")

            for key in ['daily', 'weekly', 'fundamental', 'insider', 'options']:
                info = data.get(key)
                if info and info.get('exists'):
                    text = "✓"
                    color = Colors.SUCCESS
                else:
                    text = "✗"
                    color = Colors.ERROR

                width = 80 if key == 'fundamental' else 60
                Theme.create_label(row, text, font=Fonts.BODY_S, text_color=color, width=width).pack(side="left")

    def get_discovered_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all discovered data paths."""
        return self.discovered_data

    def get_missing_data_report(self) -> Dict[str, List[str]]:
        """Get a report of missing data per symbol."""
        report = {}
        for symbol in self.symbols:
            data = self.discovered_data.get(symbol, {})
            missing = []
            for key in ['daily', 'weekly', 'fundamental', 'insider']:
                if not data.get(key):
                    missing.append(key)
            if missing:
                report[symbol] = missing
        return report


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
    - Uploading multiple trade log files
    - Automatic discovery of raw data files
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
        self.root.geometry("1100x800")
        self.root.minsize(900, 700)
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
        # Multiple Trade Logs (Required)
        self.trade_log_panel = MultiFileUploadPanel(
            parent,
            title="Trade Logs",
            file_types=[("CSV files", "*.csv"), ("JSON files", "*.json")],
            on_files_changed=self._on_trade_logs_changed
        )
        self.trade_log_panel.pack(fill="x", pady=(0, Sizes.PAD_S))

        # Raw Data Discovery
        self.raw_data_panel = RawDataDiscoveryPanel(parent)
        self.raw_data_panel.pack(fill="x", pady=Sizes.PAD_S)

        # Additional Data Files (Optional) - for manual override
        Theme.create_label(
            parent,
            "Manual Data Override (Optional)",
            font=Fonts.LABEL_BOLD,
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(Sizes.PAD_M, Sizes.PAD_S))

        # Price Data (Optional)
        self.price_data_panel = FileUploadPanel(
            parent,
            title="Price Data Override",
            file_types=[("CSV files", "*.csv")],
            required=False,
            on_file_selected=self._on_price_data_selected
        )
        self.price_data_panel.pack(fill="x", pady=Sizes.PAD_S)

        # Fundamental Data (Optional)
        self.fundamental_panel = FileUploadPanel(
            parent,
            title="Fundamental Data Override",
            file_types=[("CSV files", "*.csv")],
            required=False,
            on_file_selected=self._on_fundamental_selected
        )
        self.fundamental_panel.pack(fill="x", pady=Sizes.PAD_S)

        # Insider Data (Optional)
        self.insider_panel = FileUploadPanel(
            parent,
            title="Insider Data Override",
            file_types=[("CSV files", "*.csv")],
            required=False,
            on_file_selected=self._on_insider_selected
        )
        self.insider_panel.pack(fill="x", pady=Sizes.PAD_S)

        # Options Data (Optional)
        self.options_panel = FileUploadPanel(
            parent,
            title="Options Data Override",
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

    def _on_trade_logs_changed(self, file_paths: List[str]):
        """Handle trade log files change."""
        # Update column mapping from first file
        if self.trade_log_panel.files:
            first_file = self.trade_log_panel.files[0]
            if first_file.get('data') is not None:
                columns = list(first_file['data'].columns)
                self.mapping_panel.set_columns(columns)

        # Update raw data discovery with symbols from all files
        symbols = self.trade_log_panel.get_all_symbols()
        self.raw_data_panel.set_symbols(symbols)

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

        # Count trade log files
        trade_files = self.trade_log_panel.get_files()
        files_loaded += len(trade_files)
        for f in trade_files:
            total_records += f.get('rows', 0)

        # Count discovered raw data
        discovered = self.raw_data_panel.get_discovered_data()
        for symbol, data in discovered.items():
            for key, info in data.items():
                if info and info.get('exists'):
                    files_loaded += 1

        # Count manual override files
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

        # Add warnings for missing data
        missing_report = self.raw_data_panel.get_missing_data_report()
        if missing_report:
            missing_count = len(missing_report)
            warnings.append(f"{missing_count} symbols missing some raw data")

        # Calculate quality score
        if len(trade_files) == 0:
            score = 0
            status = "No data loaded"
        else:
            # Base score from having trade logs
            score = 40

            # Add points for discovered data completeness
            symbols = self.trade_log_panel.get_all_symbols()
            if symbols:
                complete_symbols = len(symbols) - len(missing_report)
                completeness = complete_symbols / len(symbols)
                score += int(completeness * 40)

            # Add points for mapping validity
            is_valid, errors = self.mapping_panel.is_valid()
            if is_valid:
                score += 20
            else:
                warnings.extend(errors)

            if score >= 80:
                status = "Good"
            elif score >= 60:
                status = "Acceptable"
            else:
                status = "Needs attention"

        self.quality_indicator.set_quality(
            score=max(0, min(100, score)),
            status=status,
            metrics={
                "Files loaded": str(files_loaded),
                "Trade logs": str(len(trade_files)),
                "Symbols": str(len(self.trade_log_panel.get_all_symbols())),
                "Total records": f"{total_records:,}"
            },
            warnings=warnings if warnings else None
        )

    def _on_validate(self):
        """Validate all uploaded data."""
        errors = []

        # Check required files
        if not self.trade_log_panel.is_valid():
            errors.append("At least one trade log file is required")

        # Check column mapping
        is_valid, mapping_errors = self.mapping_panel.is_valid()
        errors.extend(mapping_errors)

        if errors:
            show_error(self.root, "Validation Failed", "\n".join(errors))
        else:
            # Show missing data warnings
            missing = self.raw_data_panel.get_missing_data_report()
            if missing:
                msg = "Validation passed with warnings:\n\n"
                msg += "Missing raw data for some symbols:\n"
                for symbol, missing_types in list(missing.items())[:5]:
                    msg += f"  • {symbol}: {', '.join(missing_types)}\n"
                if len(missing) > 5:
                    msg += f"  ... and {len(missing) - 5} more symbols\n"
                show_info(self.root, "Validation Passed", msg)
            else:
                show_info(self.root, "Validation Passed", "All data is valid and ready for preparation.")
            self._update_quality()

    def _on_prepare(self):
        """Prepare data for analysis."""
        # Validate first
        errors = []

        if not self.trade_log_panel.is_valid():
            errors.append("At least one trade log file is required")

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
                progress.update(0.1, "Loading trade logs...")
                trade_data = self._load_trade_logs()

                progress.update(0.25, "Loading daily price data...")
                daily_price_data = self._load_raw_data('daily')

                progress.update(0.35, "Loading weekly price data...")
                weekly_price_data = self._load_raw_data('weekly')

                progress.update(0.5, "Loading fundamental data...")
                fundamental_data = self._load_raw_data('fundamental')
                if self.fundamental_panel.data is not None:
                    fundamental_data = self.fundamental_panel.data

                progress.update(0.65, "Loading insider data...")
                insider_data = self._load_raw_data('insider')
                if self.insider_panel.data is not None:
                    insider_data = self.insider_panel.data

                progress.update(0.8, "Loading options data...")
                options_data = self._load_raw_data('options')
                if self.options_panel.data is not None:
                    options_data = self.options_panel.data

                progress.update(0.95, "Finalizing...")

                self.prepared_data = {
                    'trade_data': trade_data,
                    'daily_price_data': daily_price_data,
                    'weekly_price_data': weekly_price_data,
                    'price_data': daily_price_data,  # For backwards compatibility
                    'fundamental_data': fundamental_data,
                    'insider_data': insider_data,
                    'options_data': options_data,
                    'column_mapping': self.mapping_panel.get_mapping(),
                    'symbols': list(self.trade_log_panel.get_all_symbols()),
                    'discovered_data': self.raw_data_panel.get_discovered_data()
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

    def _load_trade_logs(self) -> pd.DataFrame:
        """Load and combine all trade logs with column mapping."""
        all_trades = []
        mapping = self.mapping_panel.get_mapping()

        for file_info in self.trade_log_panel.get_files():
            file_path = file_info['path']
            data = pd.read_csv(file_path)

            # Apply column mapping
            rename_map = {}
            for target, source in mapping.items():
                if source and source in data.columns:
                    rename_map[source] = target

            if rename_map:
                data = data.rename(columns=rename_map)

            # Add source file column
            data['_source_file'] = Path(file_path).name

            all_trades.append(data)

        if all_trades:
            return pd.concat(all_trades, ignore_index=True)
        return pd.DataFrame()

    def _load_raw_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """Load raw data from discovered files."""
        discovered = self.raw_data_panel.get_discovered_data()
        all_data = []

        type_key_map = {
            'daily': 'daily',
            'weekly': 'weekly',
            'fundamental': 'fundamental',
            'insider': 'insider',
            'options': 'options'
        }

        key = type_key_map.get(data_type)
        if not key:
            return None

        for symbol, data_info in discovered.items():
            file_info = data_info.get(key)
            if file_info and file_info.get('exists'):
                try:
                    path = file_info.get('path')
                    if os.path.isdir(path):
                        # Options folder - combine all files
                        for csv_file in Path(path).glob("*.csv"):
                            df = pd.read_csv(csv_file)
                            df['symbol'] = symbol
                            all_data.append(df)
                    else:
                        df = pd.read_csv(path)
                        if 'symbol' not in df.columns:
                            df['symbol'] = symbol
                        all_data.append(df)
                except Exception:
                    pass

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

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
