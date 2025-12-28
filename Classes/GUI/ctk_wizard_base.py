"""
CustomTkinter Wizard Base Framework

A modern wizard framework for step-by-step configuration flows using CustomTkinter.
Provides consistent navigation, progress tracking, and selection summaries with
a dark/blue theme matching the data collection GUI.
"""

import customtkinter as ctk
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod

from .ctk_theme import Theme, Colors, Fonts, Sizes


class CTkWizardStep(ABC):
    """Base class for individual wizard steps."""

    def __init__(self, wizard: 'CTkWizardBase', title: str):
        """
        Initialize a wizard step.

        Args:
            wizard: Parent wizard instance
            title: Step title displayed in header and progress
        """
        self.wizard = wizard
        self.title = title
        self.frame: Optional[ctk.CTkFrame] = None
        self.validation_errors: List[str] = []

    @abstractmethod
    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        """
        Create the widgets for this step.

        Args:
            parent: Parent frame to create widgets in
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, str]:
        """
        Get a summary of selections made in this step.

        Returns:
            Dict mapping field names to their string values
        """
        pass

    def validate(self) -> bool:
        """
        Validate the current step.

        Override for custom validation. Sets self.validation_errors
        with any error messages.

        Returns:
            True if valid, False otherwise
        """
        self.validation_errors = []
        return True

    def on_enter(self) -> None:
        """Called when entering this step. Override for custom behavior."""
        pass

    def on_leave(self) -> None:
        """Called when leaving this step. Override for custom behavior."""
        pass


class CTkWizardBase:
    """
    Base wizard class providing step-by-step navigation with progress indicator.

    Layout:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Progress Indicator: Step 1 of 5: Mode & Securities  [● ● ● ○ ○]       │
    ├────────────────────────────────────────────┬────────────────────────────┤
    │                                            │                            │
    │  Step Content                              │  Current Selections        │
    │  (scrollable)                              │  (Summary Sidebar)         │
    │                                            │                            │
    ├────────────────────────────────────────────┴────────────────────────────┤
    │  Validation Messages (if any)                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                              [Cancel]  [← Back]  [Next →]               │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        root: ctk.CTk = None,
        title: str = "Wizard",
        width: int = 1100,
        height: int = 750,
        is_toplevel: bool = False,
        parent: ctk.CTk = None
    ):
        """
        Initialize the wizard.

        Args:
            root: Existing CTk root window (if is_toplevel=False)
            title: Window title
            width: Window width
            height: Window height
            is_toplevel: If True, create as CTkToplevel dialog
            parent: Parent window for toplevel mode
        """
        # Initialize theme
        Theme.setup()

        # Create window
        if is_toplevel:
            if parent is None:
                raise ValueError("Parent required for toplevel wizard")
            self.root = ctk.CTkToplevel(parent)
            self.root.transient(parent)
            self.root.grab_set()
        else:
            if root is None:
                self.root = ctk.CTk()
            else:
                self.root = root

        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(900, 600)

        # Configure dark theme background
        self.root.configure(fg_color=Colors.BG_DARK)

        self.steps: List[CTkWizardStep] = []
        self.current_step_index = 0
        self.step_frames: Dict[int, ctk.CTkFrame] = {}

        # Callbacks
        self.on_complete: Optional[Callable[[], None]] = None
        self.on_cancel: Optional[Callable[[], None]] = None

        # Create main layout
        self._create_layout()

    def _create_layout(self):
        """Create the main wizard layout."""
        # Configure root grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Main container
        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=Sizes.PAD_M, pady=Sizes.PAD_M)
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # === Progress Indicator (Top) ===
        self.progress_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_L
        )
        self.progress_frame.grid(
            row=0, column=0, columnspan=2,
            sticky="ew", pady=(0, Sizes.PAD_M)
        )

        # Progress content
        progress_content = Theme.create_frame(self.progress_frame)
        progress_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        self.progress_title_label = Theme.create_header(
            progress_content,
            "Step 1 of 1: Loading...",
            size="m"
        )
        self.progress_title_label.pack(side="left")

        # Step indicator container (right side)
        self.step_indicator_frame = Theme.create_frame(progress_content)
        self.step_indicator_frame.pack(side="right")
        self.step_indicators: List[ctk.CTkLabel] = []

        # === Content Area (Left) ===
        self.content_container = ctk.CTkFrame(
            self.main_frame,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_L,
            border_width=1,
            border_color=Colors.BORDER
        )
        self.content_container.grid(
            row=1, column=0,
            sticky="nsew", padx=(0, Sizes.PAD_M)
        )
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(1, weight=1)

        # Content title
        self.content_title = Theme.create_header(
            self.content_container,
            "",
            size="s"
        )
        self.content_title.grid(
            row=0, column=0,
            sticky="w", padx=Sizes.PAD_L, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        # Content frame (scrollable)
        self.content_frame = ctk.CTkScrollableFrame(
            self.content_container,
            fg_color="transparent",
            corner_radius=0
        )
        self.content_frame.grid(
            row=1, column=0,
            sticky="nsew", padx=Sizes.PAD_S, pady=(0, Sizes.PAD_S)
        )

        # === Summary Sidebar (Right) ===
        self.summary_container = ctk.CTkFrame(
            self.main_frame,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_L,
            border_width=1,
            border_color=Colors.BORDER,
            width=280
        )
        self.summary_container.grid(row=1, column=1, sticky="nsew")
        self.summary_container.grid_columnconfigure(0, weight=1)
        self.summary_container.grid_rowconfigure(1, weight=1)

        # Summary title
        Theme.create_header(
            self.summary_container,
            "Current Selections",
            size="s"
        ).grid(
            row=0, column=0,
            sticky="w", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S)
        )

        # Summary content (scrollable)
        self.summary_frame = ctk.CTkScrollableFrame(
            self.summary_container,
            fg_color="transparent",
            corner_radius=0
        )
        self.summary_frame.grid(
            row=1, column=0,
            sticky="nsew", padx=Sizes.PAD_S, pady=(0, Sizes.PAD_S)
        )

        # === Validation Message Area ===
        self.validation_frame = Theme.create_frame(self.main_frame)
        self.validation_frame.grid(
            row=2, column=0, columnspan=2,
            sticky="ew", pady=(Sizes.PAD_S, 0)
        )

        self.validation_label = ctk.CTkLabel(
            self.validation_frame,
            text="",
            font=Fonts.BODY_S,
            text_color=Colors.ERROR,
            wraplength=800,
            justify="left"
        )
        self.validation_label.pack(anchor="w")

        # === Navigation Buttons (Bottom) ===
        self.nav_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=Colors.SURFACE,
            corner_radius=Sizes.RADIUS_L
        )
        self.nav_frame.grid(
            row=3, column=0, columnspan=2,
            sticky="ew", pady=(Sizes.PAD_M, 0)
        )

        nav_content = Theme.create_frame(self.nav_frame)
        nav_content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)

        # Cancel button (left)
        self.cancel_button = Theme.create_button(
            nav_content,
            "Cancel",
            command=self._on_cancel,
            style="secondary",
            width=100
        )
        self.cancel_button.pack(side="left")

        # Next/Finish button (right)
        self.next_button = Theme.create_button(
            nav_content,
            "Next",
            command=self._go_next,
            style="primary",
            width=120
        )
        self.next_button.pack(side="right", padx=(Sizes.PAD_S, 0))

        # Back button (right, before next)
        self.back_button = Theme.create_button(
            nav_content,
            "Back",
            command=self._go_back,
            style="secondary",
            width=100
        )
        self.back_button.pack(side="right")

    def add_step(self, step: CTkWizardStep) -> None:
        """Add a step to the wizard."""
        self.steps.append(step)

    def start(self) -> None:
        """Start the wizard - call after adding all steps."""
        if not self.steps:
            raise ValueError("No steps added to wizard")

        # Create step indicator circles
        self._create_step_indicators()

        # Create frames for all steps
        for i, step in enumerate(self.steps):
            frame = Theme.create_frame(self.content_frame)
            step.frame = frame
            step.create_widgets(frame)
            self.step_frames[i] = frame

        # Show first step
        self._show_step(0)

    def run(self) -> None:
        """Run the wizard main loop."""
        self.root.mainloop()

    def _create_step_indicators(self):
        """Create the step indicator circles."""
        # Clear existing indicators
        for widget in self.step_indicator_frame.winfo_children():
            widget.destroy()
        self.step_indicators.clear()

        for i in range(len(self.steps)):
            # Create indicator frame for each step
            indicator_frame = Theme.create_frame(self.step_indicator_frame)
            indicator_frame.pack(side="left", padx=Sizes.PAD_XS)

            # Step number circle
            indicator = ctk.CTkLabel(
                indicator_frame,
                text=str(i + 1),
                font=Fonts.BODY_S,
                width=Sizes.WIZARD_STEP_SIZE,
                height=Sizes.WIZARD_STEP_SIZE,
                corner_radius=Sizes.WIZARD_STEP_SIZE // 2,
                fg_color=Colors.WIZARD_STEP_PENDING,
                text_color=Colors.TEXT_PRIMARY
            )
            indicator.pack()
            self.step_indicators.append(indicator)

            # Connector line (except for last step)
            if i < len(self.steps) - 1:
                connector = ctk.CTkFrame(
                    self.step_indicator_frame,
                    fg_color=Colors.WIZARD_STEP_PENDING,
                    height=2,
                    width=20
                )
                connector.pack(side="left", pady=Sizes.WIZARD_STEP_SIZE // 2 - 1)

    def _update_step_indicators(self):
        """Update the step indicators to show current progress."""
        for i, indicator in enumerate(self.step_indicators):
            if i < self.current_step_index:
                # Completed step
                indicator.configure(
                    fg_color=Colors.WIZARD_STEP_COMPLETE,
                    text="✓"
                )
            elif i == self.current_step_index:
                # Current step
                indicator.configure(
                    fg_color=Colors.WIZARD_STEP_CURRENT,
                    text=str(i + 1)
                )
            else:
                # Pending step
                indicator.configure(
                    fg_color=Colors.WIZARD_STEP_PENDING,
                    text=str(i + 1)
                )

    def _show_step(self, index: int):
        """Show the step at the given index."""
        if index < 0 or index >= len(self.steps):
            return

        # Hide current step
        if self.current_step_index < len(self.step_frames):
            current_frame = self.step_frames.get(self.current_step_index)
            if current_frame:
                self.steps[self.current_step_index].on_leave()
                current_frame.pack_forget()

        # Update index
        self.current_step_index = index
        step = self.steps[index]

        # Show new step
        frame = self.step_frames[index]
        frame.pack(fill="both", expand=True, padx=Sizes.PAD_S, pady=Sizes.PAD_S)
        step.on_enter()

        # Update progress indicator
        self.progress_title_label.configure(
            text=f"Step {index + 1} of {len(self.steps)}: {step.title}"
        )
        self._update_step_indicators()

        # Update content title
        self.content_title.configure(text=step.title)

        # Update navigation buttons
        self._update_navigation()

        # Update summary
        self._update_summary()

        # Clear validation messages
        self.validation_label.configure(text="")

    def _update_navigation(self):
        """Update navigation button states and text."""
        # Back button
        if self.current_step_index == 0:
            self.back_button.configure(state="disabled")
        else:
            self.back_button.configure(state="normal")

        # Next button - changes text on last step
        if self.current_step_index == len(self.steps) - 1:
            self.next_button.configure(text=self._get_final_button_text())
        else:
            self.next_button.configure(text="Next")

    def _get_final_button_text(self) -> str:
        """Override to customize the final step button text."""
        return "Finish"

    def _go_next(self):
        """Navigate to the next step or complete the wizard."""
        current_step = self.steps[self.current_step_index]

        # Validate current step
        if not current_step.validate():
            # Show validation errors
            errors = "\n".join(current_step.validation_errors)
            self.validation_label.configure(text=errors)
            return

        # Clear validation messages
        self.validation_label.configure(text="")

        if self.current_step_index < len(self.steps) - 1:
            # Go to next step
            self._show_step(self.current_step_index + 1)
        else:
            # Complete wizard
            self._on_complete()

    def _go_back(self):
        """Navigate to the previous step."""
        if self.current_step_index > 0:
            self._show_step(self.current_step_index - 1)

    def _on_cancel(self):
        """Handle cancel button click."""
        if self.on_cancel:
            self.on_cancel()
        else:
            self.root.destroy()

    def _on_complete(self):
        """Handle wizard completion."""
        if self.on_complete:
            self.on_complete()

    def _update_summary(self):
        """Update the summary sidebar with current selections."""
        # Clear existing summary
        for widget in self.summary_frame.winfo_children():
            widget.destroy()

        # Add summary from completed steps
        for i in range(self.current_step_index + 1):
            step = self.steps[i]
            summary = step.get_summary()

            if summary:
                # Step section
                section = Theme.create_frame(self.summary_frame)
                section.pack(fill="x", pady=(Sizes.PAD_S, 0))

                # Step title
                Theme.create_label(
                    section,
                    step.title,
                    font=Fonts.LABEL_BOLD,
                    text_color=Colors.PRIMARY_LIGHT
                ).pack(anchor="w")

                # Summary items
                for key, value in summary.items():
                    item_frame = Theme.create_frame(section)
                    item_frame.pack(fill="x", padx=(Sizes.PAD_S, 0), pady=Sizes.PAD_XS)

                    Theme.create_label(
                        item_frame,
                        f"{key}:",
                        font=Fonts.BODY_XS,
                        text_color=Colors.TEXT_SECONDARY
                    ).pack(anchor="w")

                    # Truncate long values
                    display_value = str(value)
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."

                    Theme.create_label(
                        item_frame,
                        display_value,
                        font=Fonts.BODY_XS,
                        text_color=Colors.TEXT_MUTED,
                        wraplength=200
                    ).pack(anchor="w", padx=(Sizes.PAD_S, 0))

                # Separator between steps
                if i < self.current_step_index:
                    ctk.CTkFrame(
                        self.summary_frame,
                        fg_color=Colors.BORDER,
                        height=1
                    ).pack(fill="x", pady=(Sizes.PAD_S, 0))

    def go_to_step(self, index: int):
        """Go to a specific step (used by Review step edit buttons)."""
        if 0 <= index < len(self.steps):
            self._show_step(index)

    def get_all_summaries(self) -> Dict[str, Dict[str, str]]:
        """Get summaries from all steps for the review step."""
        summaries = {}
        for step in self.steps:
            summaries[step.title] = step.get_summary()
        return summaries

    def close(self):
        """Close the wizard window."""
        self.root.destroy()


class CTkReviewStep(CTkWizardStep):
    """
    A special step that shows a review of all selections before final action.
    Provides [Edit] buttons to jump back to previous steps.
    """

    def __init__(self, wizard: CTkWizardBase, title: str = "Review & Confirm"):
        super().__init__(wizard, title)
        self.scrollable_frame: Optional[ctk.CTkScrollableFrame] = None

    def create_widgets(self, parent: ctk.CTkFrame) -> None:
        """Create the review widgets."""
        # Create scrollable frame for review content
        self.scrollable_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color="transparent"
        )
        self.scrollable_frame.pack(fill="both", expand=True)

    def on_enter(self) -> None:
        """Refresh the review content when entering this step."""
        self._refresh_review()

    def _refresh_review(self):
        """Refresh the review content."""
        if not self.scrollable_frame:
            return

        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Title/instruction
        Theme.create_label(
            self.scrollable_frame,
            "Please review your selections before proceeding:",
            font=Fonts.BODY_M,
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(0, Sizes.PAD_L))

        # Add sections for each previous step
        for i, step in enumerate(self.wizard.steps[:-1]):  # Exclude review step itself
            summary = step.get_summary()
            if not summary:
                continue

            # Section card
            section = Theme.create_card(self.scrollable_frame)
            section.pack(fill="x", pady=Sizes.PAD_S)

            # Header with title and edit button
            header_frame = Theme.create_frame(section)
            header_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(Sizes.PAD_M, Sizes.PAD_S))

            Theme.create_header(
                header_frame,
                step.title,
                size="s"
            ).pack(side="left")

            # Edit button
            step_index = i
            Theme.create_button(
                header_frame,
                "Edit",
                command=lambda idx=step_index: self.wizard.go_to_step(idx),
                style="ghost",
                width=60,
                height=28
            ).pack(side="right")

            # Content area
            content_frame = Theme.create_frame(section)
            content_frame.pack(fill="x", padx=Sizes.PAD_M, pady=(0, Sizes.PAD_M))

            # Summary items in grid layout
            row = 0
            for key, value in summary.items():
                # Key label
                key_label = Theme.create_label(
                    content_frame,
                    f"{key}:",
                    font=Fonts.LABEL_BOLD,
                    text_color=Colors.TEXT_SECONDARY
                )
                key_label.grid(row=row, column=0, sticky="nw", pady=Sizes.PAD_XS)

                # Value label
                value_label = Theme.create_label(
                    content_frame,
                    str(value),
                    font=Fonts.BODY_M,
                    wraplength=500,
                    justify="left"
                )
                value_label.grid(row=row, column=1, sticky="w", padx=(Sizes.PAD_M, 0), pady=Sizes.PAD_XS)

                row += 1

    def get_summary(self) -> Dict[str, str]:
        """Review step doesn't have its own summary."""
        return {}

    def validate(self) -> bool:
        """Review step is always valid if we got here."""
        return True


class CTkProgressDialog(ctk.CTkToplevel):
    """
    A modal progress dialog for long-running operations.

    Features:
    - Progress bar (determinate or indeterminate)
    - Status message
    - Optional cancel button
    - Auto-center on parent
    """

    def __init__(
        self,
        parent: ctk.CTk,
        title: str = "Please Wait",
        message: str = "Processing...",
        determinate: bool = True,
        cancellable: bool = True
    ):
        """
        Initialize progress dialog.

        Args:
            parent: Parent window
            title: Dialog title
            message: Initial status message
            determinate: If True, show percentage. If False, show indeterminate progress.
            cancellable: If True, show cancel button
        """
        super().__init__(parent)

        self.title(title)
        self.geometry("400x150")
        self.resizable(False, False)
        self.configure(fg_color=Colors.BG_DARK)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 150) // 2
        self.geometry(f"+{x}+{y}")

        self.cancelled = False
        self.determinate = determinate
        self.cancellable = cancellable

        self._create_widgets(message)

        # Prevent closing via X button
        self.protocol("WM_DELETE_WINDOW", self._on_cancel if cancellable else lambda: None)

    def _create_widgets(self, message: str):
        """Create dialog widgets."""
        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_L, pady=Sizes.PAD_L)

        # Message
        self.message_label = Theme.create_label(
            content,
            message,
            wraplength=350
        )
        self.message_label.pack(pady=(0, Sizes.PAD_M))

        # Progress bar
        self.progress_bar = Theme.create_progressbar(content)
        self.progress_bar.pack(fill="x", pady=Sizes.PAD_S)

        if not self.determinate:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
        else:
            self.progress_bar.set(0)

        # Percentage label (only for determinate)
        if self.determinate:
            self.percentage_label = Theme.create_label(
                content,
                "0%",
                font=Fonts.BODY_S,
                text_color=Colors.TEXT_MUTED
            )
            self.percentage_label.pack()

        # Cancel button
        if self.cancellable:
            Theme.create_button(
                content,
                "Cancel",
                command=self._on_cancel,
                style="secondary",
                width=80
            ).pack(pady=(Sizes.PAD_M, 0))

    def update(self, progress: float = None, message: str = None):
        """
        Update progress and/or message.

        Args:
            progress: Progress value between 0 and 1
            message: Optional new status message
        """
        if progress is not None and self.determinate:
            self.progress_bar.set(progress)
            if hasattr(self, 'percentage_label'):
                self.percentage_label.configure(text=f"{int(progress * 100)}%")

        if message is not None:
            self.message_label.configure(text=message)

        self.update_idletasks()

    def close(self):
        """Close the dialog."""
        if not self.determinate:
            self.progress_bar.stop()
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.close()
