"""
Wizard Base Framework

A reusable wizard framework for step-by-step configuration flows.
Provides consistent navigation, progress tracking, and selection summaries.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod


class WizardStep(ABC):
    """Base class for individual wizard steps."""

    def __init__(self, wizard: 'WizardBase', title: str):
        self.wizard = wizard
        self.title = title
        self.frame: Optional[ttk.Frame] = None
        self.validation_errors: List[str] = []

    @abstractmethod
    def create_widgets(self, parent: ttk.Frame) -> None:
        """Create the widgets for this step."""
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, str]:
        """Get a summary of selections made in this step."""
        pass

    def validate(self) -> bool:
        """
        Validate the current step. Override for custom validation.
        Returns True if valid, False otherwise.
        Sets self.validation_errors with any error messages.
        """
        self.validation_errors = []
        return True

    def on_enter(self) -> None:
        """Called when entering this step. Override for custom behavior."""
        pass

    def on_leave(self) -> None:
        """Called when leaving this step. Override for custom behavior."""
        pass


class WizardBase:
    """
    Base wizard class providing step-by-step navigation with progress indicator.

    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Progress Indicator: Step 1 of 5: Mode & Securities  [●○○○○]   │
    ├────────────────────────────────────┬────────────────────────────┤
    │                                    │                            │
    │  Step Content                      │  Current Selections        │
    │                                    │  (Summary Sidebar)         │
    │                                    │                            │
    ├────────────────────────────────────┴────────────────────────────┤
    │  Validation Messages (if any)                                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                            [Cancel]  [← Back]  [Next →]         │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, root: tk.Tk, title: str, width: int = 1000, height: int = 700):
        self.root = root
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        self.steps: List[WizardStep] = []
        self.current_step_index = 0
        self.step_frames: Dict[int, ttk.Frame] = {}

        # Callbacks
        self.on_complete: Optional[Callable[[], None]] = None
        self.on_cancel: Optional[Callable[[], None]] = None

        # Create main layout
        self._create_layout()

    def _create_layout(self):
        """Create the main wizard layout."""
        # Configure root grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # === Progress Indicator (Top) ===
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.progress_title_label = ttk.Label(
            self.progress_frame,
            text="Step 1 of 1: Loading...",
            font=('TkDefaultFont', 12, 'bold')
        )
        self.progress_title_label.pack(side=tk.LEFT, padx=10)

        # Step indicator circles
        self.step_indicator_frame = ttk.Frame(self.progress_frame)
        self.step_indicator_frame.pack(side=tk.RIGHT, padx=10)
        self.step_indicators: List[ttk.Label] = []

        # === Content Area (Left) ===
        self.content_frame = ttk.LabelFrame(self.main_frame, text="", padding="15")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)

        # === Summary Sidebar (Right) ===
        self.summary_frame = ttk.LabelFrame(self.main_frame, text="Current Selections", padding="10")
        self.summary_frame.grid(row=1, column=1, sticky="nsew")
        self.summary_frame.columnconfigure(0, weight=1)

        # Summary content (scrollable)
        self.summary_canvas = tk.Canvas(self.summary_frame, highlightthickness=0, width=250)
        self.summary_scrollbar = ttk.Scrollbar(self.summary_frame, orient="vertical",
                                                command=self.summary_canvas.yview)
        self.summary_inner_frame = ttk.Frame(self.summary_canvas)

        self.summary_inner_frame.bind(
            "<Configure>",
            lambda e: self.summary_canvas.configure(scrollregion=self.summary_canvas.bbox("all"))
        )

        self.summary_canvas.create_window((0, 0), window=self.summary_inner_frame, anchor="nw")
        self.summary_canvas.configure(yscrollcommand=self.summary_scrollbar.set)

        self.summary_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === Validation Message Area ===
        self.validation_frame = ttk.Frame(self.main_frame)
        self.validation_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.validation_label = ttk.Label(
            self.validation_frame,
            text="",
            foreground="red",
            wraplength=800
        )
        self.validation_label.pack(anchor=tk.W)

        # === Navigation Buttons (Bottom) ===
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        self.cancel_button = ttk.Button(self.nav_frame, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(side=tk.LEFT)

        self.next_button = ttk.Button(self.nav_frame, text="Next →", command=self._go_next)
        self.next_button.pack(side=tk.RIGHT, padx=(5, 0))

        self.back_button = ttk.Button(self.nav_frame, text="← Back", command=self._go_back)
        self.back_button.pack(side=tk.RIGHT)

    def add_step(self, step: WizardStep) -> None:
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
            frame = ttk.Frame(self.content_frame)
            step.frame = frame
            step.create_widgets(frame)
            self.step_frames[i] = frame

        # Show first step
        self._show_step(0)

    def _create_step_indicators(self):
        """Create the step indicator circles."""
        for widget in self.step_indicator_frame.winfo_children():
            widget.destroy()
        self.step_indicators.clear()

        for i in range(len(self.steps)):
            # Use unicode circles: ● (filled) and ○ (empty)
            indicator = ttk.Label(
                self.step_indicator_frame,
                text="○",
                font=('TkDefaultFont', 14)
            )
            indicator.pack(side=tk.LEFT, padx=2)
            self.step_indicators.append(indicator)

    def _update_step_indicators(self):
        """Update the step indicators to show current progress."""
        for i, indicator in enumerate(self.step_indicators):
            if i < self.current_step_index:
                indicator.config(text="●", foreground="green")
            elif i == self.current_step_index:
                indicator.config(text="●", foreground="#2E5994")
            else:
                indicator.config(text="○", foreground="gray")

    def _show_step(self, index: int):
        """Show the step at the given index."""
        if index < 0 or index >= len(self.steps):
            return

        # Hide current step
        if self.current_step_index < len(self.step_frames):
            current_frame = self.step_frames.get(self.current_step_index)
            if current_frame:
                self.steps[self.current_step_index].on_leave()
                current_frame.grid_forget()

        # Update index
        self.current_step_index = index
        step = self.steps[index]

        # Show new step
        frame = self.step_frames[index]
        frame.grid(row=0, column=0, sticky="nsew")
        step.on_enter()

        # Update progress indicator
        self.progress_title_label.config(
            text=f"Step {index + 1} of {len(self.steps)}: {step.title}"
        )
        self._update_step_indicators()

        # Update content frame title
        self.content_frame.config(text=step.title)

        # Update navigation buttons
        self._update_navigation()

        # Update summary
        self._update_summary()

        # Clear validation messages
        self.validation_label.config(text="")

    def _update_navigation(self):
        """Update navigation button states and text."""
        # Back button
        if self.current_step_index == 0:
            self.back_button.config(state=tk.DISABLED)
        else:
            self.back_button.config(state=tk.NORMAL)

        # Next button - changes to action button on last step
        if self.current_step_index == len(self.steps) - 1:
            self.next_button.config(text=self._get_final_button_text())
        else:
            self.next_button.config(text="Next →")

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
            self.validation_label.config(text=errors)
            return

        # Clear validation messages
        self.validation_label.config(text="")

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

    def _on_complete(self):
        """Handle wizard completion."""
        if self.on_complete:
            self.on_complete()

    def _update_summary(self):
        """Update the summary sidebar with current selections."""
        # Clear existing summary
        for widget in self.summary_inner_frame.winfo_children():
            widget.destroy()

        # Add summary from completed steps
        for i in range(self.current_step_index + 1):
            step = self.steps[i]
            summary = step.get_summary()

            if summary:
                # Step title
                step_label = ttk.Label(
                    self.summary_inner_frame,
                    text=step.title,
                    font=('TkDefaultFont', 9, 'bold'),
                    foreground='#2E5994'
                )
                step_label.pack(anchor=tk.W, pady=(10, 2))

                # Summary items
                for key, value in summary.items():
                    item_frame = ttk.Frame(self.summary_inner_frame)
                    item_frame.pack(fill=tk.X, padx=(10, 0))

                    ttk.Label(
                        item_frame,
                        text=f"{key}:",
                        font=('TkDefaultFont', 8)
                    ).pack(anchor=tk.W)

                    ttk.Label(
                        item_frame,
                        text=str(value),
                        font=('TkDefaultFont', 8, 'italic'),
                        foreground='#555555',
                        wraplength=200
                    ).pack(anchor=tk.W, padx=(5, 0))

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


class ReviewStep(WizardStep):
    """
    A special step that shows a review of all selections before final action.
    Provides [Edit] buttons to jump back to previous steps.
    """

    def __init__(self, wizard: WizardBase, title: str = "Review & Confirm"):
        super().__init__(wizard, title)
        self.edit_callbacks: Dict[int, Callable] = {}

    def create_widgets(self, parent: ttk.Frame) -> None:
        """Create the review widgets."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Create scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.canvas = canvas

    def on_enter(self) -> None:
        """Refresh the review content when entering this step."""
        self._refresh_review()

    def _refresh_review(self):
        """Refresh the review content."""
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Title
        title_label = ttk.Label(
            self.scrollable_frame,
            text="Please review your selections before proceeding:",
            font=('TkDefaultFont', 10, 'italic')
        )
        title_label.pack(anchor=tk.W, pady=(0, 15))

        # Add sections for each previous step
        for i, step in enumerate(self.wizard.steps[:-1]):  # Exclude review step itself
            summary = step.get_summary()
            if not summary:
                continue

            # Section frame
            section_frame = ttk.LabelFrame(self.scrollable_frame, text=step.title, padding="10")
            section_frame.pack(fill=tk.X, pady=5, padx=5)

            # Content frame (for items)
            content_frame = ttk.Frame(section_frame)
            content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Summary items
            for key, value in summary.items():
                item_frame = ttk.Frame(content_frame)
                item_frame.pack(fill=tk.X, pady=2)

                ttk.Label(
                    item_frame,
                    text=f"{key}:",
                    font=('TkDefaultFont', 9, 'bold'),
                    width=25,
                    anchor=tk.W
                ).pack(side=tk.LEFT)

                ttk.Label(
                    item_frame,
                    text=str(value),
                    font=('TkDefaultFont', 9),
                    wraplength=400,
                    anchor=tk.W
                ).pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Edit button
            step_index = i
            edit_btn = ttk.Button(
                section_frame,
                text="Edit",
                width=6,
                command=lambda idx=step_index: self.wizard.go_to_step(idx)
            )
            edit_btn.pack(side=tk.RIGHT, padx=(10, 0))

    def get_summary(self) -> Dict[str, str]:
        """Review step doesn't have its own summary."""
        return {}

    def validate(self) -> bool:
        """Review step is always valid if we got here."""
        return True
