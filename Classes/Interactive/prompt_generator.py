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
        f"OUTPUT FORMAT:\n"
        f"Present your entire answer as an organised bullet list grouped under "
        f"the section headings below. Do not write paragraphs of prose - every "
        f"point must be a concise bullet, with nested bullets for supporting "
        f"detail.\n"
        f"\n"
        f"CRITICAL TIME CONSTRAINT - NO LOOK-AHEAD:\n"
        f"Assume today is {date}. Answer exactly as if it is {date}: you do "
        f"not know anything that happened after {date}. Use ONLY information "
        f"that was publicly available from official, reputable sources "
        f"(company filings, earnings releases, regulator publications, major "
        f"financial news) published on or before {date}. Do not reference any "
        f"later earnings, guidance, news, price moves, or outcomes. If a "
        f"source's publication date is unclear or after {date}, exclude it.\n"
        f"{price_action_block}"
        f"\n"
        f"RESEARCH SECTIONS (everything as of {date}):\n"
        f"1. Business overview: a brief overview of the business model and "
        f"current operations - what the company sells, to whom, and how it "
        f"makes money.\n"
        f"2. Financial trend - previous 12 quarters: key financial figures "
        f"and ratios over the previous 12 reported quarters where available "
        f"(revenue, earnings/EPS, margins, free cash flow, debt and liquidity "
        f"ratios, valuation multiples). Give one bullet per metric "
        f"summarising the quarter-by-quarter trend the company has been "
        f"following, with the figures.\n"
        f"3. SWOT analysis: strengths, weaknesses, opportunities, and "
        f"threats, each supported only by facts already known on or before "
        f"{date} - no forward-looking speculation.\n"
        f"4. Intrinsic value: an estimate of intrinsic value per share. "
        f"State the method (e.g. DCF, earnings/multiples), the key "
        f"assumptions, and a value range, using only data available on "
        f"{date}, and compare it with the market price at that time.\n"
        f"5. Catalysts & recent news: any catalysts or notable news recent "
        f"relative to {date} (the weeks leading up to it), plus events "
        f"already scheduled to occur within the next {horizon_days} days as "
        f"they were known on {date}.\n"
        f"\n"
        f"FINISH WITH:\n"
        f"- A single bullet rating the stock's fundamental attractiveness as "
        f"of {date} for a {horizon_days}-day {side} holding: "
        f"ATTRACTIVE / NEUTRAL / UNATTRACTIVE, with a confidence score 1-5.\n"
        f"- Bullets citing the key sources used and their publication dates."
    )
