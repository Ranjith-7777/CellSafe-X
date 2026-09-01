"""
Layout primitives: page header, panels, metric cards and safe HTML helpers.

SECURITY NOTE
-------------
Several helpers in this package emit raw HTML so that the dashboard can look
like a product rather than a form.  Every dynamic value that reaches an HTML
string goes through `esc()` (text) or `num()` / `pct()` (numerics) first, so a
scenario name, an action label or a NaN probability can never break out of its
element or inject markup.  All arithmetic stays in Python, outside the markup.
"""

from __future__ import annotations

import contextlib
import html
import math
from typing import Iterator, Mapping, Sequence

import streamlit as st

from dashboard.styles import ACCENT_CYAN, TEXT_MUTED


# ---------------------------------------------------------------------------
# safety helpers
# ---------------------------------------------------------------------------
def esc(text: object) -> str:
    """Escape any value destined for an HTML text node or attribute."""
    return html.escape(str(text), quote=True)


def num(value: object, digits: int = 1, fallback: str = "—") -> str:
    """Format a number for display, refusing NaN / inf / non-numeric input."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v):
        return fallback
    return f"{v:.{digits}f}"


def pct(value: object, digits: int = 1, fallback: str = "—") -> str:
    """Format a probability in [0, 1] as a percentage string."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v):
        return fallback
    v = min(max(v, 0.0), 1.0)
    return f"{v * 100:.{digits}f}%"


def safe_color(colour: object, fallback: str = TEXT_MUTED) -> str:
    """Only allow a literal `#RRGGBB` colour into a style attribute."""
    s = str(colour)
    if len(s) == 7 and s[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in s[1:]):
        return s
    return fallback


def compact_html(markup: str) -> str:
    """Flatten indented markup into a single line before handing it to Streamlit.

    `st.markdown(..., unsafe_allow_html=True)` still runs the string through a
    Markdown parser first, and Markdown turns any line indented by four or more
    spaces into a literal code block.  Since the markup in this package is
    written indented for readability, a large blob - the battery SVG in
    particular - would otherwise be partially rendered as escaped source text
    instead of as an element.  Whitespace between tags is insignificant in
    HTML and SVG, so stripping it is safe and makes the output deterministic.
    """
    lines = (line.strip() for line in str(markup).splitlines())
    return "".join(line for line in lines if line)


def render_html(markup: str) -> None:
    """Emit raw HTML/SVG safely through Streamlit's markdown renderer."""
    st.markdown(compact_html(markup), unsafe_allow_html=True)


def unit_interval(value: object, fallback: float = 0.0) -> float:
    """Clamp anything to a valid [0, 1] float, for widths and opacities."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v):
        return fallback
    return min(max(v, 0.0), 1.0)


# ---------------------------------------------------------------------------
# page header
# ---------------------------------------------------------------------------
def page_header(
    title: str,
    purpose: str,
    scenario: str,
    minute: float,
    extra_chips: Sequence[str] = (),
) -> None:
    """The header every page shows: what this page is, and where the twin is."""
    chips = "".join(
        f'<span class="csx-chip">{esc(c)}</span>' for c in extra_chips
    )
    render_html(
        f"""
        <div class="csx-pagehead">
          <div>
            <div class="title">{esc(title)}</div>
            <div class="purpose">{esc(purpose)}</div>
          </div>
          <div class="meta">
            <span class="csx-chip cyan">{esc(scenario)}</span>
            <span class="csx-chip">t = {num(minute, 2)} min</span>
            {chips}
            <span class="csx-chip lime"><span class="dot"></span>Bayesian Engine Active</span>
            <span class="csx-chip proto">Simulation-Based Research Prototype</span>
          </div>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def panel(title: str | None = None, tall: bool = False) -> Iterator[None]:
    """Native bordered container so charts and widgets stay inside the card."""
    with st.container(border=True, height="stretch" if tall else "content"):
        if title:
            render_html(f'<div class="csx-panel-title">{esc(title)}</div>')
        yield


def section(title: str) -> None:
    render_html(f'<div class="csx-section">{esc(title)}</div>')


def note(text: str) -> None:
    render_html(f'<p class="csx-note">{esc(text)}</p>')


def muted(text: str) -> None:
    render_html(f'<p class="csx-muted">{esc(text)}</p>')


# ---------------------------------------------------------------------------
# cards
# ---------------------------------------------------------------------------
def metric_card(
    label: str,
    value: str,
    note_text: str = "",
    colour: str = ACCENT_CYAN,
    small: bool = False,
) -> None:
    """A single KPI tile with a coloured top rule."""
    size_class = " sm" if small else ""
    render_html(
        f"""
        <div class="csx-metric" style="border-top-color:{safe_color(colour)};">
          <div class="label">{esc(label)}</div>
          <div class="value{size_class}">{esc(value)}</div>
          <div class="note">{esc(note_text)}</div>
        </div>
        """
    )


def metric_grid(items: Sequence[tuple[str, str, str, str, bool]]) -> None:
    """Responsive KPI row that becomes two columns, then one, at narrow widths."""
    cards = []
    for label, value, note_text, colour, small in items:
        size_class = " sm" if small else ""
        cards.append(
            f'<div class="csx-metric"><div class="label">{esc(label)}</div>'
            f'<div class="value{size_class}" style="color:{safe_color(colour)}">{esc(value)}</div>'
            f'<div class="note">{esc(note_text)}</div></div>'
        )
    render_html(f'<div class="csx-metric-grid">{"".join(cards)}</div>')


def kv_rows(rows: Sequence[tuple[str, str]], colours: Mapping[str, str] | None = None) -> None:
    """A compact key/value list, optionally colouring individual values."""
    colours = colours or {}
    body = []
    for key, value in rows:
        style = ""
        if key in colours:
            style = f' style="color:{safe_color(colours[key])};"'
        body.append(
            f'<div class="csx-kv"><span class="k">{esc(key)}</span>'
            f'<span class="v"{style}>{esc(value)}</span></div>'
        )
    render_html("".join(body))


def pipeline_flow(nodes: Sequence[tuple[str, str]]) -> None:
    """The 'Sensor Evidence -> ... -> Decision' strip.

    `nodes` is a sequence of (title, description) pairs; the numbering and the
    arrows are added here.
    """
    parts = []
    for i, (title, desc) in enumerate(nodes, start=1):
        if i > 1:
            parts.append('<div class="arrow">&rsaquo;</div>')
        parts.append(
            f'<div class="node"><div class="n">0{i}</div>'
            f'<div class="t">{esc(title)}</div>'
            f'<div class="d">{esc(desc)}</div></div>'
        )
    render_html(f'<div class="csx-flow">{"".join(parts)}</div>')
