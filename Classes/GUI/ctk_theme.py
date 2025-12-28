"""
Unified CustomTkinter Theme and Styling Module.

This module provides consistent theming, colors, fonts, and widget factory
functions for all GUI components in the BackTestingFramework.

Usage:
    from Classes.GUI.ctk_theme import Theme, create_styled_button, create_header_label

    # Initialize theme (call once at app start)
    Theme.setup()

    # Use factory functions for consistent widgets
    btn = Theme.create_button(parent, "Click Me", command=callback)
    label = Theme.create_header(parent, "Section Title")
"""

import customtkinter as ctk
from typing import Optional, Callable, Tuple, List, Dict, Any


class Colors:
    """Centralized color definitions for the application."""

    # Primary colors (blue theme)
    PRIMARY = "#1f538d"
    PRIMARY_HOVER = "#164175"
    PRIMARY_LIGHT = "#3b7dd8"

    # Secondary colors
    SECONDARY = "#2E5994"
    SECONDARY_HOVER = "#1e4a7d"

    # Background colors
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    BG_CARD = "#242442"

    # Surface colors for frames/cards
    SURFACE = "#2b2b40"
    SURFACE_HOVER = "#363654"

    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b0b0b0"
    TEXT_MUTED = "#808080"
    TEXT_LINK = "#4da6ff"

    # Status colors
    SUCCESS = "#27AE60"
    SUCCESS_HOVER = "#1e8449"
    WARNING = "#f39c12"
    WARNING_HOVER = "#d68910"
    ERROR = "#E74C3C"
    ERROR_HOVER = "#c0392b"
    INFO = "#3498db"
    INFO_HOVER = "#2980b9"

    # Chart/Visualization colors
    CHART_LINE = "#2E5994"
    CHART_POSITIVE = "#27AE60"
    CHART_NEGATIVE = "#E74C3C"
    CHART_NEUTRAL = "#95A5A6"
    CHART_GRID = "#404060"
    CHART_BG = "#1a1a2e"

    # Wizard-specific colors
    WIZARD_STEP_CURRENT = "#1f538d"
    WIZARD_STEP_COMPLETE = "#27AE60"
    WIZARD_STEP_PENDING = "#808080"
    WIZARD_PROGRESS_BG = "#2b2b40"

    # Border colors
    BORDER = "#404060"
    BORDER_LIGHT = "#505070"
    BORDER_FOCUS = "#1f538d"

    # Immunity/vulnerability colors (for charts)
    IMMUNITY_REGION = "#2d5a35"
    VULNERABLE_REGION = "#5a2d2d"


class Fonts:
    """Centralized font definitions."""

    # Font families (empty string uses system default)
    FAMILY = ""
    FAMILY_MONO = "Consolas" if True else "Courier"  # Will be system-dependent

    # Header fonts
    HEADER_XL = ("", 20, "bold")
    HEADER_L = ("", 16, "bold")
    HEADER_M = ("", 14, "bold")
    HEADER_S = ("", 12, "bold")

    # Body fonts
    BODY_L = ("", 14)
    BODY_M = ("", 12)
    BODY_S = ("", 11)
    BODY_XS = ("", 10)

    # Special fonts
    LABEL = ("", 12)
    LABEL_BOLD = ("", 12, "bold")
    BUTTON = ("", 12)
    HINT = ("", 10)
    MONO = ("Consolas", 11)
    MONO_S = ("Consolas", 10)


class Sizes:
    """Centralized size definitions."""

    # Padding
    PAD_XS = 2
    PAD_S = 5
    PAD_M = 10
    PAD_L = 15
    PAD_XL = 20

    # Border radius
    RADIUS_S = 4
    RADIUS_M = 8
    RADIUS_L = 12

    # Button sizes
    BTN_HEIGHT = 32
    BTN_HEIGHT_L = 40
    BTN_WIDTH = 120
    BTN_WIDTH_L = 150

    # Entry sizes
    ENTRY_HEIGHT = 32
    ENTRY_WIDTH = 200

    # Widget defaults
    CORNER_RADIUS = 8
    BORDER_WIDTH = 1

    # Wizard sizes
    WIZARD_WIDTH = 1000
    WIZARD_HEIGHT = 700
    WIZARD_STEP_SIZE = 24

    # Dialog sizes
    DIALOG_S = (400, 300)
    DIALOG_M = (600, 450)
    DIALOG_L = (800, 600)
    DIALOG_XL = (1000, 750)


class Theme:
    """Main theme manager class with factory methods for styled widgets."""

    _initialized = False

    @classmethod
    def setup(cls, appearance_mode: str = "dark", color_theme: str = "blue"):
        """
        Initialize the CustomTkinter theme.

        Args:
            appearance_mode: "dark", "light", or "system"
            color_theme: "blue", "green", or "dark-blue"
        """
        if not cls._initialized:
            ctk.set_appearance_mode(appearance_mode)
            ctk.set_default_color_theme(color_theme)
            cls._initialized = True

    # === Frame Factory Methods ===

    @classmethod
    def create_frame(cls, parent, **kwargs) -> ctk.CTkFrame:
        """Create a styled frame."""
        defaults = {
            "fg_color": "transparent",
            "corner_radius": Sizes.CORNER_RADIUS,
        }
        defaults.update(kwargs)
        return ctk.CTkFrame(parent, **defaults)

    @classmethod
    def create_card(cls, parent, **kwargs) -> ctk.CTkFrame:
        """Create a card-style frame with background."""
        defaults = {
            "fg_color": Colors.SURFACE,
            "corner_radius": Sizes.RADIUS_L,
            "border_width": 1,
            "border_color": Colors.BORDER,
        }
        defaults.update(kwargs)
        return ctk.CTkFrame(parent, **defaults)

    @classmethod
    def create_scrollable_frame(cls, parent, **kwargs) -> ctk.CTkScrollableFrame:
        """Create a styled scrollable frame."""
        defaults = {
            "fg_color": "transparent",
            "corner_radius": Sizes.CORNER_RADIUS,
        }
        defaults.update(kwargs)
        return ctk.CTkScrollableFrame(parent, **defaults)

    # === Label Factory Methods ===

    @classmethod
    def create_label(cls, parent, text: str, **kwargs) -> ctk.CTkLabel:
        """Create a standard label."""
        defaults = {
            "font": Fonts.BODY_M,
            "text_color": Colors.TEXT_PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkLabel(parent, text=text, **defaults)

    @classmethod
    def create_header(cls, parent, text: str, size: str = "m", **kwargs) -> ctk.CTkLabel:
        """
        Create a header label.

        Args:
            parent: Parent widget
            text: Header text
            size: "xl", "l", "m", or "s"
        """
        font_map = {
            "xl": Fonts.HEADER_XL,
            "l": Fonts.HEADER_L,
            "m": Fonts.HEADER_M,
            "s": Fonts.HEADER_S,
        }
        defaults = {
            "font": font_map.get(size, Fonts.HEADER_M),
            "text_color": Colors.TEXT_PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkLabel(parent, text=text, **defaults)

    @classmethod
    def create_hint(cls, parent, text: str, **kwargs) -> ctk.CTkLabel:
        """Create a hint/helper text label."""
        defaults = {
            "font": Fonts.HINT,
            "text_color": Colors.TEXT_MUTED,
        }
        defaults.update(kwargs)
        return ctk.CTkLabel(parent, text=text, **defaults)

    # === Button Factory Methods ===

    @classmethod
    def create_button(cls, parent, text: str, command: Optional[Callable] = None,
                      style: str = "primary", **kwargs) -> ctk.CTkButton:
        """
        Create a styled button.

        Args:
            parent: Parent widget
            text: Button text
            command: Click callback
            style: "primary", "secondary", "success", "danger", "ghost"
        """
        style_configs = {
            "primary": {
                "fg_color": Colors.PRIMARY,
                "hover_color": Colors.PRIMARY_HOVER,
                "text_color": Colors.TEXT_PRIMARY,
            },
            "secondary": {
                "fg_color": Colors.SURFACE,
                "hover_color": Colors.SURFACE_HOVER,
                "text_color": Colors.TEXT_PRIMARY,
                "border_width": 1,
                "border_color": Colors.BORDER,
            },
            "success": {
                "fg_color": Colors.SUCCESS,
                "hover_color": Colors.SUCCESS_HOVER,
                "text_color": Colors.TEXT_PRIMARY,
            },
            "danger": {
                "fg_color": Colors.ERROR,
                "hover_color": Colors.ERROR_HOVER,
                "text_color": Colors.TEXT_PRIMARY,
            },
            "ghost": {
                "fg_color": "transparent",
                "hover_color": Colors.SURFACE,
                "text_color": Colors.TEXT_LINK,
            },
        }

        defaults = style_configs.get(style, style_configs["primary"])
        defaults.update({
            "font": Fonts.BUTTON,
            "corner_radius": Sizes.RADIUS_M,
            "height": Sizes.BTN_HEIGHT,
        })
        defaults.update(kwargs)

        return ctk.CTkButton(parent, text=text, command=command, **defaults)

    # === Entry Factory Methods ===

    @classmethod
    def create_entry(cls, parent, placeholder: str = "", **kwargs) -> ctk.CTkEntry:
        """Create a styled entry field."""
        defaults = {
            "font": Fonts.BODY_M,
            "height": Sizes.ENTRY_HEIGHT,
            "corner_radius": Sizes.RADIUS_M,
            "border_width": 1,
            "border_color": Colors.BORDER,
            "fg_color": Colors.SURFACE,
            "text_color": Colors.TEXT_PRIMARY,
            "placeholder_text_color": Colors.TEXT_MUTED,
        }
        defaults.update(kwargs)
        return ctk.CTkEntry(parent, placeholder_text=placeholder, **defaults)

    @classmethod
    def create_textbox(cls, parent, **kwargs) -> ctk.CTkTextbox:
        """Create a styled multi-line textbox."""
        defaults = {
            "font": Fonts.MONO,
            "corner_radius": Sizes.RADIUS_M,
            "border_width": 1,
            "border_color": Colors.BORDER,
            "fg_color": Colors.SURFACE,
            "text_color": Colors.TEXT_PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkTextbox(parent, **defaults)

    # === Selection Widget Factory Methods ===

    @classmethod
    def create_checkbox(cls, parent, text: str, variable: Optional[ctk.Variable] = None,
                        command: Optional[Callable] = None, **kwargs) -> ctk.CTkCheckBox:
        """Create a styled checkbox."""
        defaults = {
            "font": Fonts.BODY_M,
            "text_color": Colors.TEXT_PRIMARY,
            "fg_color": Colors.PRIMARY,
            "hover_color": Colors.PRIMARY_HOVER,
            "border_color": Colors.BORDER,
            "checkmark_color": Colors.TEXT_PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkCheckBox(parent, text=text, variable=variable, command=command, **defaults)

    @classmethod
    def create_radiobutton(cls, parent, text: str, variable: ctk.Variable,
                           value: Any, command: Optional[Callable] = None, **kwargs) -> ctk.CTkRadioButton:
        """Create a styled radio button."""
        defaults = {
            "font": Fonts.BODY_M,
            "text_color": Colors.TEXT_PRIMARY,
            "fg_color": Colors.PRIMARY,
            "hover_color": Colors.PRIMARY_HOVER,
            "border_color": Colors.BORDER,
        }
        defaults.update(kwargs)
        return ctk.CTkRadioButton(parent, text=text, variable=variable, value=value,
                                  command=command, **defaults)

    @classmethod
    def create_switch(cls, parent, text: str = "", variable: Optional[ctk.Variable] = None,
                      command: Optional[Callable] = None, **kwargs) -> ctk.CTkSwitch:
        """Create a styled switch."""
        defaults = {
            "font": Fonts.BODY_M,
            "text_color": Colors.TEXT_PRIMARY,
            "fg_color": Colors.SURFACE,
            "progress_color": Colors.PRIMARY,
            "button_color": Colors.TEXT_PRIMARY,
            "button_hover_color": Colors.TEXT_SECONDARY,
        }
        defaults.update(kwargs)
        return ctk.CTkSwitch(parent, text=text, variable=variable, command=command, **defaults)

    @classmethod
    def create_combobox(cls, parent, values: List[str], variable: Optional[ctk.Variable] = None,
                        command: Optional[Callable] = None, **kwargs) -> ctk.CTkComboBox:
        """Create a styled combobox/dropdown."""
        defaults = {
            "font": Fonts.BODY_M,
            "height": Sizes.ENTRY_HEIGHT,
            "corner_radius": Sizes.RADIUS_M,
            "border_width": 1,
            "border_color": Colors.BORDER,
            "fg_color": Colors.SURFACE,
            "text_color": Colors.TEXT_PRIMARY,
            "button_color": Colors.PRIMARY,
            "button_hover_color": Colors.PRIMARY_HOVER,
            "dropdown_fg_color": Colors.SURFACE,
            "dropdown_hover_color": Colors.SURFACE_HOVER,
            "dropdown_text_color": Colors.TEXT_PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkComboBox(parent, values=values, variable=variable, command=command, **defaults)

    @classmethod
    def create_optionmenu(cls, parent, values: List[str], variable: Optional[ctk.Variable] = None,
                          command: Optional[Callable] = None, **kwargs) -> ctk.CTkOptionMenu:
        """Create a styled option menu."""
        defaults = {
            "font": Fonts.BODY_M,
            "height": Sizes.ENTRY_HEIGHT,
            "corner_radius": Sizes.RADIUS_M,
            "fg_color": Colors.PRIMARY,
            "button_color": Colors.PRIMARY_HOVER,
            "button_hover_color": Colors.PRIMARY_LIGHT,
            "text_color": Colors.TEXT_PRIMARY,
            "dropdown_fg_color": Colors.SURFACE,
            "dropdown_hover_color": Colors.SURFACE_HOVER,
            "dropdown_text_color": Colors.TEXT_PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkOptionMenu(parent, values=values, variable=variable, command=command, **defaults)

    # === Progress/Status Factory Methods ===

    @classmethod
    def create_progressbar(cls, parent, **kwargs) -> ctk.CTkProgressBar:
        """Create a styled progress bar."""
        defaults = {
            "height": 12,
            "corner_radius": 6,
            "fg_color": Colors.SURFACE,
            "progress_color": Colors.PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkProgressBar(parent, **defaults)

    @classmethod
    def create_slider(cls, parent, from_: float = 0, to: float = 100,
                      variable: Optional[ctk.Variable] = None,
                      command: Optional[Callable] = None, **kwargs) -> ctk.CTkSlider:
        """Create a styled slider."""
        defaults = {
            "height": 16,
            "fg_color": Colors.SURFACE,
            "progress_color": Colors.PRIMARY,
            "button_color": Colors.PRIMARY,
            "button_hover_color": Colors.PRIMARY_HOVER,
        }
        defaults.update(kwargs)
        return ctk.CTkSlider(parent, from_=from_, to=to, variable=variable,
                            command=command, **defaults)

    # === Tabview Factory Method ===

    @classmethod
    def create_tabview(cls, parent, **kwargs) -> ctk.CTkTabview:
        """Create a styled tab view."""
        defaults = {
            "corner_radius": Sizes.RADIUS_L,
            "fg_color": Colors.SURFACE,
            "segmented_button_fg_color": Colors.BG_CARD,
            "segmented_button_selected_color": Colors.PRIMARY,
            "segmented_button_selected_hover_color": Colors.PRIMARY_HOVER,
            "segmented_button_unselected_color": Colors.BG_CARD,
            "segmented_button_unselected_hover_color": Colors.SURFACE_HOVER,
            "text_color": Colors.TEXT_PRIMARY,
        }
        defaults.update(kwargs)
        return ctk.CTkTabview(parent, **defaults)

    # === Toplevel/Dialog Factory Method ===

    @classmethod
    def create_toplevel(cls, parent, title: str = "", size: Tuple[int, int] = None,
                        **kwargs) -> ctk.CTkToplevel:
        """Create a styled toplevel window."""
        defaults = {
            "fg_color": Colors.BG_DARK,
        }
        defaults.update(kwargs)
        toplevel = ctk.CTkToplevel(parent, **defaults)
        if title:
            toplevel.title(title)
        if size:
            toplevel.geometry(f"{size[0]}x{size[1]}")
        return toplevel


class ProgressPanel(ctk.CTkFrame):
    """
    Reusable progress panel with status, progress bar, and log.

    Similar to the data collection GUI's ProgressPanel.
    """

    def __init__(self, parent, show_log: bool = True, log_height: int = 150, **kwargs):
        """
        Initialize the progress panel.

        Args:
            parent: Parent widget
            show_log: Whether to show the message log
            log_height: Height of the log textbox
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.show_log = show_log

        # Status frame
        status_frame = Theme.create_frame(self)
        status_frame.pack(fill="x", pady=(0, Sizes.PAD_S))

        self.status_label = Theme.create_label(status_frame, "Ready", font=Fonts.LABEL_BOLD)
        self.status_label.pack(side="left")

        self.percentage_label = Theme.create_label(status_frame, "", font=Fonts.BODY_S)
        self.percentage_label.pack(side="right")

        # Progress bar
        self.progress_bar = Theme.create_progressbar(self)
        self.progress_bar.pack(fill="x", pady=Sizes.PAD_S)
        self.progress_bar.set(0)

        # Log textbox
        if show_log:
            self.log_textbox = Theme.create_textbox(self, height=log_height)
            self.log_textbox.pack(fill="both", expand=True, pady=(Sizes.PAD_S, 0))
            self.log_textbox.configure(state="disabled")

    def update_progress(self, progress: float, status: str = None):
        """
        Update progress bar and status.

        Args:
            progress: Progress value between 0 and 1
            status: Optional status text
        """
        self.progress_bar.set(progress)
        self.percentage_label.configure(text=f"{int(progress * 100)}%")
        if status:
            self.status_label.configure(text=status)

    def log(self, message: str, level: str = "info"):
        """
        Add a message to the log.

        Args:
            message: Message text
            level: "info", "success", "warning", or "error"
        """
        if not self.show_log:
            return

        self.log_textbox.configure(state="normal")

        # Color coding based on level
        color_map = {
            "info": Colors.TEXT_PRIMARY,
            "success": Colors.SUCCESS,
            "warning": Colors.WARNING,
            "error": Colors.ERROR,
        }

        # Insert message with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_textbox.insert("end", f"[{timestamp}] {message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def clear(self):
        """Clear the log and reset progress."""
        self.progress_bar.set(0)
        self.status_label.configure(text="Ready")
        self.percentage_label.configure(text="")

        if self.show_log:
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.configure(state="disabled")

    def set_complete(self, message: str = "Complete"):
        """Mark progress as complete."""
        self.progress_bar.set(1)
        self.status_label.configure(text=message)
        self.percentage_label.configure(text="100%")


class MessageDialog(ctk.CTkToplevel):
    """A styled message dialog."""

    def __init__(self, parent, title: str, message: str,
                 dialog_type: str = "info", buttons: List[str] = None):
        """
        Create a message dialog.

        Args:
            parent: Parent widget
            title: Dialog title
            message: Message to display
            dialog_type: "info", "warning", "error", "success", or "question"
            buttons: List of button labels (default: ["OK"])
        """
        super().__init__(parent)

        self.title(title)
        self.geometry("400x200")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 200) // 2
        self.geometry(f"+{x}+{y}")

        self.result = None

        # Icon colors
        icon_colors = {
            "info": Colors.INFO,
            "warning": Colors.WARNING,
            "error": Colors.ERROR,
            "success": Colors.SUCCESS,
            "question": Colors.PRIMARY,
        }

        # Icon symbols
        icon_symbols = {
            "info": "i",
            "warning": "!",
            "error": "X",
            "success": "✓",
            "question": "?",
        }

        # Content frame
        content_frame = Theme.create_frame(self)
        content_frame.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        # Icon
        icon_color = icon_colors.get(dialog_type, Colors.INFO)
        icon_symbol = icon_symbols.get(dialog_type, "i")
        icon_label = ctk.CTkLabel(
            content_frame,
            text=icon_symbol,
            font=("", 32, "bold"),
            text_color=icon_color,
            width=50
        )
        icon_label.pack(side="left", padx=(0, Sizes.PAD_L))

        # Message
        message_label = Theme.create_label(
            content_frame,
            message,
            wraplength=300,
            justify="left"
        )
        message_label.pack(side="left", fill="both", expand=True)

        # Buttons
        if buttons is None:
            buttons = ["OK"]

        button_frame = Theme.create_frame(self)
        button_frame.pack(fill="x", padx=Sizes.PAD_L, pady=(0, Sizes.PAD_L))

        for btn_text in buttons:
            style = "primary" if btn_text in ["OK", "Yes", "Confirm"] else "secondary"
            btn = Theme.create_button(
                button_frame,
                btn_text,
                command=lambda t=btn_text: self._on_button(t),
                style=style,
                width=80
            )
            btn.pack(side="right", padx=(Sizes.PAD_S, 0))

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", lambda: self._on_button(buttons[-1] if buttons else "Cancel"))

        # Wait for dialog to close
        self.wait_window()

    def _on_button(self, button: str):
        """Handle button click."""
        self.result = button
        self.destroy()


def show_info(parent, title: str, message: str):
    """Show an info message dialog."""
    return MessageDialog(parent, title, message, "info", ["OK"]).result


def show_warning(parent, title: str, message: str):
    """Show a warning message dialog."""
    return MessageDialog(parent, title, message, "warning", ["OK"]).result


def show_error(parent, title: str, message: str):
    """Show an error message dialog."""
    return MessageDialog(parent, title, message, "error", ["OK"]).result


def show_success(parent, title: str, message: str):
    """Show a success message dialog."""
    return MessageDialog(parent, title, message, "success", ["OK"]).result


def ask_yes_no(parent, title: str, message: str) -> bool:
    """Show a yes/no question dialog. Returns True if Yes was clicked."""
    result = MessageDialog(parent, title, message, "question", ["No", "Yes"]).result
    return result == "Yes"


def ask_ok_cancel(parent, title: str, message: str) -> bool:
    """Show an OK/Cancel dialog. Returns True if OK was clicked."""
    result = MessageDialog(parent, title, message, "question", ["Cancel", "OK"]).result
    return result == "OK"
