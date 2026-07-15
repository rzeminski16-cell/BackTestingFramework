"""
Signal Decision Panel for interactive backtests.

Appears when the engine pauses on an actionable signal. Shows the
signal header, a price chart with EMAs around the signal date (built
only from the look-ahead-safe slice the engine provided), the portfolio
snapshot, a time-bounded research prompt generator (copy to clipboard,
paste findings back), and the decision controls (accept / modify /
reject / defer plus quick actions and a rationale box).

Threading model: the engine's worker thread is blocked on
``reply_queue.get()``. This panel runs on the Tk main thread; submitting
puts a DecisionResponse on the reply queue and destroys the window.
Closing the panel confirms and sends ABORT, which pauses the run (all
decisions so far are already saved).

CAPITAL_RESOLUTION requests reuse the same window with the positions
table switched into a pick-list (close / trim) plus reduce-size and
reject choices.
"""
import queue
import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

from Classes.Interactive.models import (
    DecisionAction,
    DecisionRequest,
    DecisionResponse,
    DecisionSource,
)
from Classes.Interactive.prompt_generator import generate_research_prompt
from Classes.Interactive.store import find_resumable_runs
from .ctk_theme import Colors, Fonts, Sizes, Theme, ask_yes_no, show_error

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    MATPLOTLIB_AVAILABLE = False

EMA_PALETTE = [Colors.WARNING, Colors.SUCCESS, Colors.INFO,
               Colors.ERROR, Colors.PRIMARY_LIGHT, Colors.CHART_NEUTRAL]


class CTkSignalDecisionPanel(ctk.CTkToplevel):
    """Modal-style decision panel; unblocks the engine via reply_queue."""

    def __init__(self, parent, request: DecisionRequest,
                 reply_queue: "queue.Queue",
                 default_horizon: int = 90,
                 on_done: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.request = request
        self.reply_queue = reply_queue
        self.default_horizon = default_horizon
        self.on_done = on_done or (lambda: None)
        self._answered = False
        self._prompt_text = ""

        event = request.event
        self.title(f"Decision: {event.symbol} {event.signal_type} "
                   f"on {event.bar_date}")
        self.geometry("1060x860")
        self.configure(fg_color=Colors.BG_DARK)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.lift()
        self.focus_force()

        body = Theme.create_scrollable_frame(self)
        body.pack(fill="both", expand=True, padx=Sizes.PAD_M,
                  pady=Sizes.PAD_M)

        self._build_header(body)
        if len(request.day_batch or []) > 1:
            self._build_day_batch(body)
        self._build_chart(body)
        self._build_portfolio(body)
        if request.kind == "CAPITAL_RESOLUTION":
            self._build_capital_controls(body)
        else:
            self._build_research(body)
            self._build_decision_controls(body)
        self._build_action_row(body)

    # ------------------------------------------------------------- sections
    def _card(self, parent, title):
        card = Theme.create_card(parent)
        card.pack(fill="x", pady=(0, Sizes.PAD_M))
        content = Theme.create_frame(card)
        content.pack(fill="x", padx=Sizes.PAD_L, pady=Sizes.PAD_M)
        Theme.create_header(content, title, size="s").pack(
            anchor="w", pady=(0, Sizes.PAD_S))
        return content

    def _build_header(self, parent):
        event = self.request.event
        content = self._card(parent, "Signal")

        direction = "LONG" if event.direction == "LONG" else "SHORT"
        price = (event.technical_snapshot or {}).get('close')
        title = (f"{event.symbol}   {event.signal_type} ({direction})   "
                 f"{event.bar_date}"
                 + (f"   @ {price:,.4f}" if price is not None else ""))
        Theme.create_label(content, title, font=Fonts.LABEL_BOLD).pack(anchor="w")
        if event.signal_reason:
            Theme.create_label(content, f"Strategy reason: {event.signal_reason}",
                               text_color=Colors.TEXT_SECONDARY).pack(anchor="w")

        proposal = []
        if event.proposed_size:
            proposal.append(f"size {event.proposed_size:g}")
        if event.proposed_stop_loss is not None:
            proposal.append(f"stop {event.proposed_stop_loss:,.4f}")
        if event.proposed_take_profit is not None:
            proposal.append(f"take-profit {event.proposed_take_profit:,.4f}")
        if proposal:
            Theme.create_label(content, "Proposed: " + ", ".join(proposal),
                               text_color=Colors.TEXT_SECONDARY).pack(anchor="w")

        snapshot = event.portfolio_snapshot or {}
        required = snapshot.get('required_capital')
        available = snapshot.get('available_capital')
        if required is not None and available is not None:
            short = required > available
            Theme.create_label(
                content,
                f"Capital: required {required:,.2f} / available {available:,.2f}"
                + ("  — INSUFFICIENT (resolution options below)" if short else ""),
                text_color=(Colors.ERROR if short else Colors.TEXT_SECONDARY),
            ).pack(anchor="w")

        if self.request.warning:
            Theme.create_label(content, f"Note: {self.request.warning}",
                               text_color=Colors.WARNING).pack(anchor="w")

    def _build_day_batch(self, parent):
        batch = self.request.day_batch
        index = self.request.batch_index
        content = self._card(
            parent, f"Today's signals ({index + 1} of {len(batch)})")
        header = ["", "Symbol", "Type", "Dir", "Price", "Stop",
                  "Capital needed"]
        grid = Theme.create_frame(content)
        grid.pack(fill="x")
        for col, text in enumerate(header):
            Theme.create_label(grid, text, font=Fonts.LABEL_BOLD,
                               text_color=Colors.TEXT_SECONDARY).grid(
                row=0, column=col, sticky="w", padx=(0, Sizes.PAD_M))
        for row, item in enumerate(batch, start=1):
            position = row - 1
            if position < index:
                marker, color = "done", Colors.TEXT_SECONDARY
            elif position == index:
                marker, color = "► now", Colors.WARNING
            else:
                marker, color = "queued", Colors.TEXT_PRIMARY
            stop = item.get('stop_loss')
            required = item.get('required_capital')
            values = [
                marker,
                item.get('symbol', ''),
                item.get('signal_type', ''),
                item.get('direction', ''),
                f"{item.get('price', 0):,.2f}",
                (f"{stop:,.2f}" if stop is not None else "-"),
                (f"{required:,.0f}" if required is not None else "-"),
            ]
            for col, value in enumerate(values):
                Theme.create_label(grid, value, text_color=color).grid(
                    row=row, column=col, sticky="w", padx=(0, Sizes.PAD_M))
        Theme.create_hint(
            content,
            "All of today's signals are shown; each is decided in turn so "
            "the capital effect of every decision carries into the next one."
        ).pack(anchor="w", pady=(Sizes.PAD_XS, 0))

    def _build_chart(self, parent):
        chart_data = self.request.chart_data
        if (not MATPLOTLIB_AVAILABLE or chart_data is None
                or len(chart_data) == 0 or 'close' not in chart_data.columns):
            return
        content = self._card(
            parent, f"Price ({len(chart_data)} bars up to the signal date "
                    f"— no future data shown)")

        fig = Figure(figsize=(9.6, 3.4), dpi=100, facecolor=Colors.BG_DARK)
        ax = fig.add_subplot(111)
        ax.set_facecolor(Colors.BG_MEDIUM)
        for spine in ax.spines.values():
            spine.set_color(Colors.CHART_GRID)
        ax.tick_params(colors=Colors.TEXT_SECONDARY, labelsize=8)
        ax.grid(True, color=Colors.CHART_GRID, alpha=0.35, linewidth=0.6)

        x = (chart_data['date'] if 'date' in chart_data.columns
             else range(len(chart_data)))
        ax.plot(x, chart_data['close'], color=Colors.PRIMARY_LIGHT,
                linewidth=1.6, label='Close')
        ema_cols = [c for c in chart_data.columns if c.startswith('ema_')]
        for i, col in enumerate(ema_cols[:len(EMA_PALETTE)]):
            ax.plot(x, chart_data[col], color=EMA_PALETTE[i],
                    linewidth=1.0, alpha=0.9, label=col)
        try:
            signal_x = (chart_data['date'].iloc[-1]
                        if 'date' in chart_data.columns
                        else len(chart_data) - 1)
            ax.axvline(signal_x, color=Colors.WARNING, linestyle='--',
                       linewidth=1.2)
        except Exception:
            pass
        legend = ax.legend(loc='upper left', fontsize=7,
                           facecolor=Colors.BG_MEDIUM,
                           edgecolor=Colors.CHART_GRID)
        for text in legend.get_texts():
            text.set_color(Colors.TEXT_SECONDARY)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=content)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")

    def _build_portfolio(self, parent):
        snapshot = self.request.event.portfolio_snapshot or {}
        positions = snapshot.get('positions') or []
        content = self._card(parent, "Portfolio")
        cash = snapshot.get('available_capital')
        equity = snapshot.get('total_equity')
        summary = []
        if cash is not None:
            summary.append(f"Cash: {cash:,.2f}")
        if equity is not None:
            summary.append(f"Total equity: {equity:,.2f}")
        summary.append(f"Open positions: {len(positions)}")
        Theme.create_label(content, "   |   ".join(summary)).pack(anchor="w")

        if not positions:
            return
        header = ["Symbol", "Dir", "Entry", "Qty", "P&L %", "Weight %",
                  "Vuln", "Frees"]
        grid = Theme.create_frame(content)
        grid.pack(fill="x", pady=(Sizes.PAD_S, 0))
        for col, text in enumerate(header):
            Theme.create_label(grid, text, font=Fonts.LABEL_BOLD,
                               text_color=Colors.TEXT_SECONDARY).grid(
                row=0, column=col, sticky="w", padx=(0, Sizes.PAD_M))
        for row, position in enumerate(positions, start=1):
            pl = position.get('unrealized_pl_pct')
            values = [
                position.get('symbol', ''),
                position.get('direction', ''),
                f"{position.get('entry_price', 0):,.2f}",
                f"{position.get('quantity', 0):,.2f}",
                (f"{pl:+.2f}" if pl is not None else "-"),
                (f"{position['weight_pct']:.1f}"
                 if position.get('weight_pct') is not None else "-"),
                (f"{position['vulnerability_score']:.1f}"
                 if position.get('vulnerability_score') is not None else "-"),
                (f"{position['est_capital_freed_close']:,.0f}"
                 if position.get('est_capital_freed_close') is not None else "-"),
            ]
            color = (Colors.SUCCESS if (pl or 0) >= 0 else Colors.ERROR)
            for col, value in enumerate(values):
                Theme.create_label(
                    grid, value,
                    text_color=(color if col == 4 else Colors.TEXT_PRIMARY),
                ).grid(row=row, column=col, sticky="w", padx=(0, Sizes.PAD_M))

    def _build_research(self, parent):
        content = self._card(parent, "Fundamental Research (no look-ahead)")
        Theme.create_hint(
            content,
            "Generate a time-bounded research prompt, run it in Perplexity "
            "yourself, then paste the key findings back before deciding."
        ).pack(anchor="w")

        button_row = Theme.create_frame(content)
        button_row.pack(fill="x", pady=Sizes.PAD_S)
        Theme.create_button(button_row, "Generate Research Prompt",
                            command=self._generate_prompt,
                            width=200).pack(side="left")
        self.copy_button = Theme.create_button(
            button_row, "Copy to Clipboard", command=self._copy_prompt,
            style="secondary", width=160)
        self.copy_button.pack(side="left", padx=(Sizes.PAD_S, 0))

        self.prompt_box = Theme.create_textbox(content, height=110)
        self.prompt_box.pack(fill="x", pady=Sizes.PAD_S)
        self.prompt_box.configure(state="disabled")

        Theme.create_label(content, "Findings summary (pasted from your "
                                    "research; stored with the decision):"
                           ).pack(anchor="w")
        self.response_box = Theme.create_textbox(content, height=70)
        self.response_box.pack(fill="x", pady=(Sizes.PAD_XS, 0))

    def _generate_prompt(self):
        self._prompt_text = generate_research_prompt(
            self.request.event, horizon_days=self.default_horizon)
        self.prompt_box.configure(state="normal")
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", self._prompt_text)
        self.prompt_box.configure(state="disabled")

    def _copy_prompt(self):
        if not self._prompt_text:
            self._generate_prompt()
        self.clipboard_clear()
        self.clipboard_append(self._prompt_text)
        self.copy_button.configure(text="Copied!")
        self.after(1500, lambda: self._safe_button_reset())

    def _safe_button_reset(self):
        try:
            self.copy_button.configure(text="Copy to Clipboard")
        except tk.TclError:
            pass

    def _build_decision_controls(self, parent):
        event = self.request.event
        content = self._card(parent, "Decision")

        self.action_var = ctk.StringVar(value="accept")
        radio_row = Theme.create_frame(content)
        radio_row.pack(fill="x", pady=Sizes.PAD_XS)
        for value, label in [("accept", "Accept"), ("modify", "Modify"),
                             ("reject", "Reject"), ("defer", "Defer")]:
            Theme.create_radiobutton(radio_row, label,
                                     variable=self.action_var, value=value,
                                     command=self._on_action_change).pack(
                side="left", padx=(0, Sizes.PAD_L))
        Theme.create_hint(
            content,
            "Reject suppresses re-prompts while the same signal keeps firing "
            "(~1 month); Defer asks again on the next firing."
        ).pack(anchor="w")

        self.modify_frame = Theme.create_frame(content)
        self.size_var = ctk.StringVar(value="100")
        self.stop_var = ctk.StringVar(
            value=("" if event.proposed_stop_loss is None
                   else f"{event.proposed_stop_loss:g}"))
        self.tp_var = ctk.StringVar(
            value=("" if event.proposed_take_profit is None
                   else f"{event.proposed_take_profit:g}"))

        row = Theme.create_frame(self.modify_frame)
        row.pack(fill="x", pady=Sizes.PAD_XS)
        Theme.create_label(row, "Size (% of proposed):", width=170).pack(side="left")
        Theme.create_entry(row, width=90, textvariable=self.size_var).pack(side="left")
        if event.signal_type in ("BUY",):
            row2 = Theme.create_frame(self.modify_frame)
            row2.pack(fill="x", pady=Sizes.PAD_XS)
            Theme.create_label(row2, "Stop loss:", width=170).pack(side="left")
            Theme.create_entry(row2, width=90, textvariable=self.stop_var).pack(side="left")
            Theme.create_label(row2, "  Take profit:", width=110).pack(side="left")
            Theme.create_entry(row2, width=90, textvariable=self.tp_var).pack(side="left")

        Theme.create_label(content, "Rationale (required for Modify/Reject):"
                           ).pack(anchor="w", pady=(Sizes.PAD_S, 0))
        self.rationale_box = Theme.create_textbox(content, height=70)
        self.rationale_box.pack(fill="x", pady=(Sizes.PAD_XS, 0))

    def _on_action_change(self):
        if self.action_var.get() == "modify":
            self.modify_frame.pack(fill="x", pady=Sizes.PAD_XS)
        else:
            self.modify_frame.pack_forget()

    def _build_capital_controls(self, parent):
        options = self.request.capital_options
        content = self._card(parent, "Capital Resolution")
        Theme.create_label(
            content,
            f"This entry needs {options.required_capital:,.2f} but only "
            f"{options.available_capital:,.2f} is available "
            f"({options.affordable_fraction * 100:.0f}% affordable).",
            text_color=Colors.WARNING).pack(anchor="w")

        self.capital_choice_var = ctk.StringVar(value="reduce_size")
        Theme.create_radiobutton(
            content,
            f"Reduce position to what fits "
            f"(~{options.affordable_fraction * 100:.0f}% of proposed)",
            variable=self.capital_choice_var, value="reduce_size").pack(
            anchor="w", pady=Sizes.PAD_XS)
        Theme.create_radiobutton(
            content, "Free capital by closing/trimming positions below",
            variable=self.capital_choice_var, value="free_capital").pack(
            anchor="w", pady=Sizes.PAD_XS)
        Theme.create_radiobutton(
            content, "Reject this entry",
            variable=self.capital_choice_var, value="reject").pack(
            anchor="w", pady=Sizes.PAD_XS)

        self.free_action_vars = []
        if options.positions:
            grid = Theme.create_frame(content)
            grid.pack(fill="x", pady=Sizes.PAD_S)
            for col, text in enumerate(["Position", "P&L %", "Frees (close)",
                                        "Action", "Trim %"]):
                Theme.create_label(grid, text, font=Fonts.LABEL_BOLD,
                                   text_color=Colors.TEXT_SECONDARY).grid(
                    row=0, column=col, sticky="w", padx=(0, Sizes.PAD_M))
            for row, position in enumerate(options.positions, start=1):
                symbol = position.get('symbol', '')
                pl = position.get('unrealized_pl_pct')
                action_var = ctk.StringVar(value="keep")
                trim_var = ctk.StringVar(value="50")
                self.free_action_vars.append((symbol, action_var, trim_var))
                Theme.create_label(grid, symbol).grid(
                    row=row, column=0, sticky="w", padx=(0, Sizes.PAD_M))
                Theme.create_label(
                    grid, f"{pl:+.2f}" if pl is not None else "-",
                    text_color=(Colors.SUCCESS if (pl or 0) >= 0
                                else Colors.ERROR)).grid(
                    row=row, column=1, sticky="w", padx=(0, Sizes.PAD_M))
                frees = position.get('est_capital_freed_close')
                Theme.create_label(
                    grid, f"{frees:,.0f}" if frees is not None else "-").grid(
                    row=row, column=2, sticky="w", padx=(0, Sizes.PAD_M))
                Theme.create_optionmenu(
                    grid, values=["keep", "close", "trim"],
                    variable=action_var, width=90).grid(
                    row=row, column=3, sticky="w", padx=(0, Sizes.PAD_M))
                Theme.create_entry(grid, width=60,
                                   textvariable=trim_var).grid(
                    row=row, column=4, sticky="w")

        Theme.create_label(content, "Rationale:").pack(
            anchor="w", pady=(Sizes.PAD_S, 0))
        self.rationale_box = Theme.create_textbox(content, height=60)
        self.rationale_box.pack(fill="x", pady=(Sizes.PAD_XS, 0))

    def _build_action_row(self, parent):
        row = Theme.create_frame(parent)
        row.pack(fill="x", pady=(0, Sizes.PAD_M))
        if self.request.kind == "CAPITAL_RESOLUTION":
            Theme.create_button(row, "Confirm", command=self._submit_capital,
                                style="primary", width=160).pack(side="right")
        else:
            Theme.create_button(row, "Save && Continue",
                                command=self._submit_decision,
                                style="primary", width=160).pack(side="right")
            Theme.create_button(
                row, "Quick Reject", style="danger", width=130,
                command=lambda: self._finish(DecisionResponse(
                    action=DecisionAction.REJECT,
                    rationale=self._rationale(),
                    source=DecisionSource.QUICK,
                    **self._prompt_fields()))).pack(
                side="right", padx=(0, Sizes.PAD_S))
            Theme.create_button(
                row, "Accept Default", style="success", width=140,
                command=lambda: self._finish(DecisionResponse(
                    action=DecisionAction.ACCEPT,
                    rationale=self._rationale(),
                    source=DecisionSource.QUICK,
                    **self._prompt_fields()))).pack(
                side="right", padx=(0, Sizes.PAD_S))
            Theme.create_button(
                row, "Decide Rest Randomly", style="secondary", width=180,
                command=self._hand_off_random).pack(side="left")

    def _hand_off_random(self):
        if not ask_yes_no(
                self, "Random auto-completion",
                "Decide this signal and every remaining signal in this run "
                "randomly?\n\nEntries are accepted or rejected by a coin "
                "flip; exit signals are always accepted. The run finishes "
                "without prompting again, and every random decision is "
                "logged like any other."):
            return
        self._finish(DecisionResponse(
            action=DecisionAction.ACCEPT,
            rationale="Handed off to random auto-completion",
            source=DecisionSource.USER,
            hand_off_random=True,
            **self._prompt_fields()))

    # ------------------------------------------------------------ submission
    def _rationale(self) -> str:
        try:
            return self.rationale_box.get("1.0", "end").strip()
        except (AttributeError, tk.TclError):
            return ""

    def _prompt_fields(self) -> dict:
        response_summary = ""
        try:
            response_summary = self.response_box.get("1.0", "end").strip()
        except (AttributeError, tk.TclError):
            pass
        if not self._prompt_text and not response_summary:
            return {}
        return {
            'prompt_text': self._prompt_text,
            'prompt_horizon_days': self.default_horizon,
            'response_summary': response_summary,
        }

    def _parse_float(self, value: str, label: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{label} must be a number (got {value!r})")

    def _submit_decision(self):
        event = self.request.event
        action_map = {"accept": DecisionAction.ACCEPT,
                      "modify": DecisionAction.MODIFY,
                      "reject": DecisionAction.REJECT,
                      "defer": DecisionAction.DEFER}
        action = action_map[self.action_var.get()]
        rationale = self._rationale()
        try:
            if action in (DecisionAction.MODIFY, DecisionAction.REJECT) \
                    and not rationale:
                raise ValueError(
                    "Please enter a rationale for Modify/Reject decisions — "
                    "it is what makes the run analysable later.")

            size_factor = None
            stop = None
            take_profit = None
            if action == DecisionAction.MODIFY:
                size_pct = self._parse_float(self.size_var.get(), "Size %")
                if size_pct is not None:
                    if size_pct <= 0:
                        raise ValueError("Size % must be positive.")
                    size_factor = size_pct / 100.0
                stop = self._parse_float(self.stop_var.get(), "Stop loss")
                take_profit = self._parse_float(self.tp_var.get(), "Take profit")
                price = (event.technical_snapshot or {}).get('close')
                if stop is not None and price:
                    if event.direction == "LONG" and stop >= price:
                        raise ValueError(
                            f"Stop {stop} must be below the price {price} "
                            f"for a LONG entry.")
                    if event.direction == "SHORT" and stop <= price:
                        raise ValueError(
                            f"Stop {stop} must be above the price {price} "
                            f"for a SHORT entry.")
                if (stop is not None
                        and event.proposed_stop_loss is not None
                        and stop == event.proposed_stop_loss):
                    stop = None  # unchanged
                if (take_profit is not None
                        and event.proposed_take_profit is not None
                        and take_profit == event.proposed_take_profit):
                    take_profit = None
        except ValueError as exc:
            show_error(self, "Invalid decision", str(exc))
            return

        self._finish(DecisionResponse(
            action=action,
            size_factor=size_factor,
            modified_stop_loss=stop,
            modified_take_profit=take_profit,
            rationale=rationale,
            source=DecisionSource.USER,
            **self._prompt_fields()))

    def _submit_capital(self):
        choice = self.capital_choice_var.get()
        resolution = {'choice': choice}
        if choice == "free_capital":
            free_actions = []
            for symbol, action_var, trim_var in self.free_action_vars:
                kind = action_var.get()
                if kind == "close":
                    free_actions.append({'symbol': symbol, 'action': 'close'})
                elif kind == "trim":
                    try:
                        fraction = float(trim_var.get()) / 100.0
                    except ValueError:
                        show_error(self, "Invalid decision",
                                   f"Trim % for {symbol} must be a number.")
                        return
                    if not 0 < fraction <= 1:
                        show_error(self, "Invalid decision",
                                   f"Trim % for {symbol} must be in (0, 100].")
                        return
                    free_actions.append({'symbol': symbol, 'action': 'trim',
                                         'fraction': fraction})
            if not free_actions:
                show_error(self, "Invalid decision",
                           "Select at least one position to close or trim, "
                           "or choose another option.")
                return
            resolution['free_actions'] = free_actions

        self._finish(DecisionResponse(
            action=(DecisionAction.REJECT if choice == "reject"
                    else DecisionAction.MODIFY),
            rationale=self._rationale(),
            capital_resolution=resolution,
            source=DecisionSource.USER))

    def _finish(self, response: DecisionResponse):
        if self._answered:
            return
        self._answered = True
        self.reply_queue.put(response)
        self.on_done()
        self.destroy()

    def _on_close(self):
        if self._answered:
            self.destroy()
            return
        if ask_yes_no(self, "Pause run?",
                      "Pause this interactive backtest?\n\nAll decisions so "
                      "far are saved; you can resume later from the wizard's "
                      "'Resume interactive run' button."):
            self._finish(DecisionResponse(action=DecisionAction.ABORT))


class CTkResumeRunDialog(ctk.CTkToplevel):
    """Pick a paused/in-progress interactive run to resume."""

    def __init__(self, parent, on_select: Callable[[dict], None]):
        super().__init__(parent)
        self.on_select = on_select
        self.title("Resume Interactive Run")
        self.geometry("640x420")
        self.configure(fg_color=Colors.BG_DARK)
        self.transient(parent)
        self.grab_set()

        content = Theme.create_frame(self)
        content.pack(fill="both", expand=True, padx=Sizes.PAD_L,
                     pady=Sizes.PAD_L)
        Theme.create_header(content, "Paused interactive runs",
                            size="m").pack(anchor="w", pady=(0, Sizes.PAD_M))

        runs = find_resumable_runs()
        if not runs:
            Theme.create_label(content,
                               "No paused or in-progress interactive runs "
                               "found under logs/.").pack(anchor="w")
        else:
            scroll = Theme.create_scrollable_frame(content, height=280)
            scroll.pack(fill="both", expand=True)
            for entry in runs:
                manifest = entry['manifest']
                row = Theme.create_card(scroll)
                row.pack(fill="x", pady=Sizes.PAD_XS)
                inner = Theme.create_frame(row)
                inner.pack(fill="x", padx=Sizes.PAD_M, pady=Sizes.PAD_S)
                label = (f"{manifest.backtest_name}  "
                         f"[{manifest.engine_type}]  {manifest.status}  "
                         f"decisions: {manifest.counts.get('decisions', '?')}"
                         f"  created: {manifest.created_at[:16]}")
                Theme.create_label(inner, label).pack(side="left")
                Theme.create_button(
                    inner, "Resume", width=90,
                    command=lambda e=entry: self._select(e)).pack(side="right")

        Theme.create_button(content, "Close", command=self.destroy,
                            style="secondary", width=100).pack(
            anchor="e", pady=(Sizes.PAD_M, 0))

    def _select(self, entry):
        self.destroy()
        self.on_select(entry)
