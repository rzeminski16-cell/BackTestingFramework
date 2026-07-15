"""
Deterministic, time-bounded research prompt generation.

The generated prompt is a pure function of the SignalEvent (plus an
explicit horizon), so it is unit-testable and replay-safe. Every prompt
embeds a hard time bound: the analyst must assume the signal date is
"today" and use only information from official, reputable sources
published on or before that date — the core no-look-ahead control for
the discretionary research step.

The prompt is designed for copy-paste into Perplexity (or any research
assistant); the response summary the user pastes back is stored on the
PromptRecord.
"""
from typing import Any, Dict, Optional

from .models import SignalEvent

DEFAULT_HORIZON_DAYS = 90


def default_horizon_days(strategy_name: str = "",
                         strategy_params: Optional[Dict[str, Any]] = None) -> int:
    """
    Heuristic holding horizon for the research prompt, overridable in
    the decision panel. Uses an explicit holding-period-like strategy
    parameter when one exists; otherwise a 90-day swing default.
    """
    for key, value in (strategy_params or {}).items():
        lowered = key.lower()
        if ('hold' in lowered or 'horizon' in lowered) and isinstance(value, (int, float)):
            days = int(value)
            if days > 0:
                return days
    return DEFAULT_HORIZON_DAYS


def _price_action_summary(event: SignalEvent) -> str:
    """Describe recent price action using only signal-time snapshot data."""
    tech = event.technical_snapshot or {}
    parts = []
    for n, label in ((5, "1 week"), (21, "1 month"), (63, "3 months")):
        value = tech.get(f'pct_change_{n}d')
        if value is not None:
            parts.append(f"{value:+.1f}% over the last {label}")
    high = tech.get('dist_from_252d_high_pct')
    low = tech.get('dist_from_252d_low_pct')
    if high is not None:
        parts.append(f"{abs(high):.1f}% below its 52-week high")
    if low is not None:
        parts.append(f"{low:+.1f}% above its 52-week low")
    if not parts:
        return ""
    return "Recent price action (for context only): the stock is " + ", ".join(parts) + "."


def generate_research_prompt(event: SignalEvent,
                             horizon_days: int = DEFAULT_HORIZON_DAYS,
                             market: str = "US equities",
                             currency: str = "") -> str:
    """
    Build the time-bounded research prompt for one signal event.

    Deterministic: identical inputs always produce the identical prompt.
    """
    symbol = event.symbol
    date = event.bar_date
    side = "long (buy)" if event.direction == "LONG" else "short (sell)"
    action = {
        'BUY': f"opening a new {side} position",
        'PYRAMID': f"adding to an existing {side} position",
        'SELL': f"closing an existing {side} position",
        'PARTIAL_EXIT': f"partially closing an existing {side} position",
    }.get(event.signal_type, "evaluating a trade")
    currency_note = f" The account is denominated in {currency}." if currency else ""
    price_action = _price_action_summary(event)
    price_action_block = f"\n{price_action}\n" if price_action else "\n"

    return (
        f"You are an equity analyst evaluating {symbol} ({market}) on {date}. "
        f"A systematic strategy has signalled {action} with an intended holding "
        f"horizon of roughly {horizon_days} days.{currency_note}\n"
        f"\n"
        f"CRITICAL TIME CONSTRAINT - NO LOOK-AHEAD:\n"
        f"Assume today is {date}. You do not know anything that happened after "
        f"{date}. Use ONLY fundamental, valuation, and macro information that "
        f"was publicly available from official, reputable sources (company "
        f"filings, earnings releases, regulator publications, major financial "
        f"news) published on or before {date}. Do not reference any later "
        f"earnings, guidance, news, price moves, or outcomes. If a source's "
        f"publication date is unclear or after {date}, exclude it.\n"
        f"{price_action_block}"
        f"\n"
        f"Provide a structured assessment, as of {date}, of:\n"
        f"1. Business model and key revenue drivers.\n"
        f"2. The most recent reported earnings and guidance available on or "
        f"before {date} (no later quarters), and the trajectory they imply.\n"
        f"3. Valuation versus peers using data available at that time.\n"
        f"4. Known upcoming catalysts within the next {horizon_days} days "
        f"(scheduled earnings dates, product events, macro releases) as they "
        f"were known on {date}.\n"
        f"5. Major risks, red flags, or bear-case arguments known at that time.\n"
        f"\n"
        f"Conclude with:\n"
        f"- A one-paragraph summary.\n"
        f"- A single rating of the stock's fundamental attractiveness as of "
        f"{date} for a {horizon_days}-day {side} holding: "
        f"ATTRACTIVE / NEUTRAL / UNATTRACTIVE, with a confidence score 1-5.\n"
        f"- Cite the publication dates of the key sources you relied on."
    )
