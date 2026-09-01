"""
Persistent left navigation, shared simulation controls and shared selectors.

The navigation is rendered from Streamlit buttons rather than `st.page_link`
so that the active item can be styled precisely: the active entry is a
`type="primary"` button (lime text, green-tinted background, lime left
indicator, restrained glow) and every inactive entry is a `type="secondary"`
button (muted blue-grey, transparent, subtle hover).  Selecting an entry calls
`st.switch_page`, which is real navigation - the URL changes and the page can be
opened directly later.

WHY EVERY MUTATION IS A CALLBACK
--------------------------------
Widgets here write to shared state through `on_change` / `on_click` callbacks
rather than by mutating state inline and calling `st.rerun()`.  Callbacks run
*before* the script body, so `app.py` builds its `PageContext` from the already
updated state and the page renders correctly on the very same run.  An inline
mutation would need an explicit `st.rerun()`, and under `st.navigation` a bare
rerun re-resolves the route - which silently bounces the user back to the
default page.  Callbacks avoid that entirely.
"""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from config import model_parameters as P
from dashboard.layout import esc, render_html
from state import session_manager as SM

BRAND_MARK_SVG = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
  <rect x="3" y="6.5" width="15" height="11" rx="3.2"
        stroke="#246B4B" stroke-width="1.7"/>
  <rect x="19" y="10" width="2.6" height="4" rx="1" fill="#246B4B"/>
  <path d="M11.4 8.6 L8.2 12.6 h2.6 l-0.9 3 l3.4 -4.2 h-2.6 z" fill="#246B4B"/>
</svg>
"""


# ---------------------------------------------------------------------------
# brand / navigation / status
# ---------------------------------------------------------------------------
def render_brand() -> None:
    with st.sidebar:
        render_html(
            f"""
            <div class="csx-brand">
              <div class="mark">{BRAND_MARK_SVG}</div>
              <div>
                <div class="name">CellSafe-X</div>
                <div class="tag">Battery Safety Twin</div>
              </div>
            </div>
            """
        )


def render_nav(pages: Sequence, active_url_path: str) -> None:
    """Draw the five navigation entries, highlighting the active one."""
    with st.sidebar:
        render_html('<div class="csx-navlabel">Navigate</div>')
    for page in pages:
        is_active = page.url_path == active_url_path
        if st.sidebar.button(
            page.title,
            key=f"nav_{page.url_path or 'home'}",
            type="primary" if is_active else "secondary",
            icon=page.icon,
            width="stretch",
        ):
            if not is_active:
                st.switch_page(page)


def render_status_footer() -> None:
    with st.sidebar:
        render_html(
            f"""
            <div class="csx-navfoot">
              <div class="csx-status-line"><span class="dot"></span>Simulation Active</div>
              <div class="csx-version">Prototype v1.0</div>
            </div>
            """
        )


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------
def _cb_scenario() -> None:
    SM.set_scenario(st.session_state["w_scenario"])


def _cb_affected(labels: Sequence[str]) -> None:
    SM.set_affected_cell(list(labels).index(st.session_state["w_affected"]))


def _cb_seed() -> None:
    SM.set_seed(int(st.session_state["w_seed"]))


def _cb_steps() -> None:
    SM.set_n_steps(int(st.session_state["w_steps"]))


def _cb_step(widget_key: str) -> None:
    SM.set_step(int(st.session_state[widget_key]))


def _cb_focus(widget_key: str, labels: Sequence[str]) -> None:
    SM.set_focus_cell(list(labels).index(st.session_state[widget_key]))


def _cb_nudge_step(delta: int) -> None:
    SM.set_step(int(st.session_state[SM.K_STEP]) + int(delta))


# ---------------------------------------------------------------------------
# shared controls
# ---------------------------------------------------------------------------
def render_controls(ctx: SM.PageContext) -> None:
    """Shared simulation controls.

    These are the only widgets that can change the simulation signature, which
    is why they live in the persistent sidebar rather than on any one page: the
    twin must behave identically no matter which page you are looking at.
    """
    with st.sidebar:
        render_html('<div class="csx-navlabel">Simulation</div>')

    st.session_state.setdefault("w_scenario", ctx.scenario)
    st.sidebar.selectbox(
        "Scenario", list(P.SCENARIOS), key="w_scenario", on_change=_cb_scenario,
    )

    cell_labels = [f"Cell {i + 1}" for i in range(P.N_CELLS)]
    st.session_state.setdefault("w_affected", cell_labels[ctx.affected_cell])
    st.sidebar.selectbox(
        "Affected cell", cell_labels, key="w_affected",
        on_change=_cb_affected, args=(cell_labels,),
        help="Which cell carries the injected fault. Ignored by Normal Operation.",
    )

    with st.sidebar.expander("Seed & length"):
        st.session_state.setdefault("w_seed", int(ctx.seed))
        st.number_input(
            "Random seed", min_value=0, max_value=1_000_000, step=1,
            key="w_seed", on_change=_cb_seed,
            help="The same seed always reproduces exactly the same run.",
        )
        st.session_state.setdefault("w_steps", int(ctx.n_steps))
        st.slider(
            "Simulation length (steps)", min_value=60, max_value=360, step=30,
            key="w_steps", on_change=_cb_steps,
            help=f"One step = {P.DT_SECONDS:.0f} s of simulated time.",
        )
        st.caption(
            f"{int(ctx.n_steps)} steps × {P.DT_SECONDS:.0f} s = "
            f"{int(ctx.n_steps) * P.DT_MINUTES:.1f} simulated minutes"
        )

    c1, c2 = st.sidebar.columns(2)
    c1.button("Regenerate", key="btn_regen", width="stretch", on_click=SM.regenerate,
              help="Draw a fresh random seed and rebuild from t = 0.")
    c2.button("Reset", key="btn_reset", width="stretch", on_click=SM.reset_timeline,
              help="Rewind to t = 0 without changing the scenario or seed.")


# ---------------------------------------------------------------------------
# shared page-level selectors
# ---------------------------------------------------------------------------
def cell_labels_for(ctx: SM.PageContext) -> list[str]:
    """Identical cell labels on every page, so the selection reads consistently."""
    return [
        f"Cell {i + 1}" + (" · fault injected" if i == ctx.affected_cell else "")
        for i in range(ctx.n_cells)
    ]


def _sync_widget(widget_key: str, desired: object, valid: Sequence) -> None:
    """Force a keyed widget back into agreement with the canonical state.

    Needed because a keyed widget ignores its `index`/`value` argument after the
    first render, so a change made on another page (or by Reset) would otherwise
    leave this page's widget showing a stale selection.  On the run where the
    user themselves changed the widget, the on_change callback has already
    updated the canonical state, so `desired` matches and nothing is overwritten.
    """
    current = st.session_state.get(widget_key)
    if current not in valid or current != desired:
        st.session_state[widget_key] = desired


def cell_selector(ctx: SM.PageContext, key: str, label: str = "Diagnostic cell",
                  collapsed: bool = False) -> None:
    """Cell selector bound to the shared focus cell."""
    labels = cell_labels_for(ctx)
    _sync_widget(key, labels[ctx.focus_cell], labels)
    st.selectbox(
        label, labels, key=key, on_change=_cb_focus, args=(key, labels),
        label_visibility="collapsed" if collapsed else "visible",
    )


def render_timeline(ctx: SM.PageContext, key: str = "timeline") -> None:
    """Compact timeline scrubber bound to the shared step index."""
    widget_key = f"{key}_slider"
    _sync_widget(widget_key, int(ctx.step), range(0, ctx.n_steps))
    st.slider(
        "Timeline",
        min_value=0,
        max_value=ctx.max_step,
        key=widget_key,
        on_change=_cb_step,
        args=(widget_key,),
        help=f"Step 0–{ctx.max_step}; one step = {P.DT_SECONDS:.0f} s of simulated time.",
        label_visibility="collapsed",
    )


def step_buttons(ctx: SM.PageContext, key: str = "twin") -> None:
    """Single-step nudge buttons beside the timeline."""
    c1, c2 = st.columns(2, gap="small")
    c1.button("◀ Step", key=f"{key}_back", width="stretch",
              on_click=_cb_nudge_step, args=(-1,), disabled=ctx.step <= 0)
    c2.button("Step ▶", key=f"{key}_fwd", width="stretch",
              on_click=_cb_nudge_step, args=(1,), disabled=ctx.step >= ctx.max_step)
