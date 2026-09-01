"""
Reusable HTML/Streamlit components for the CellSafe-X dashboard.

Every function here is pure presentation: it renders values that were already
computed by the models package.  No probability is invented in this module.
Plotly figures live in `dashboard/charts.py`; this module owns the DOM.

All dynamic values are escaped or numerically validated by `dashboard.layout`
before they reach an HTML string.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from config import model_parameters as P
from dashboard.layout import esc, num, pct, render_html, safe_color
from dashboard.styles import (
    ACCENT_LIME,
    CONFLICT,
    CRITICAL,
    STATE_COLORS,
    TEXT_MUTED,
    hex_to_rgba,
)


# ---------------------------------------------------------------------------
# six-cell miniature overview (used on Home and Sensor Monitoring)
# ---------------------------------------------------------------------------
def cell_strip(ctx, compact: bool = False) -> None:
    """A compact one-line-per-cell overview of the whole pack.

    Deliberately NOT the main digital twin - that lives on the Live Battery Twin
    page as the neon pack model.  This is the at-a-glance list.
    """
    rows = []
    for c in range(ctx.n_cells):
        out = ctx.results[c][ctx.step]
        danger = ctx.cell_danger(c)
        conflicted = float(out.sensor.p_faulty) > 0.5
        state = ctx.state_of(c)
        colour = safe_color(CONFLICT if conflicted else STATE_COLORS.get(state, CONFLICT))

        badge = ""
        if conflicted:
            badge = '<span class="badge conflict">sensor conflict</span>'
        elif c == ctx.affected_cell:
            badge = '<span class="badge">fault injected</span>'

        sel = " sel" if c == ctx.focus_cell else ""
        rows.append(
            f"""
            <div class="csx-strip-row{sel}">
              <span class="bar" style="background:{colour};"></span>
              <span class="id">C{c + 1}</span>
              <span class="temp">{num(out.fused_temp_c, 1)}°C</span>
              <span class="state" style="color:{colour};">{esc(state)}</span>
              <span class="risk">{pct(danger, 1)}</span>
              {badge}
            </div>
            """
        )

    render_html(
        f'<div class="csx-strip{" compact" if compact else ""}">{"".join(rows)}</div>'
    )


# ---------------------------------------------------------------------------
# summary / status
# ---------------------------------------------------------------------------
def status_banner(ctx) -> None:
    """Scenario provenance line. Always states that the data is simulated."""
    render_html(
        f"""
        <div class="csx-banner">
          <span class="tag">SIMULATED</span>
          <span class="txt"><b>{esc(ctx.scenario)}</b> · seed {esc(ctx.seed)} ·
          affected cell #{esc(ctx.affected_cell + 1)} · {esc(ctx.sim.notes)}</span>
        </div>
        """
    )


def recommendation_card(decision, sensor_p_faulty: float | None = None) -> None:
    """The recommended action with its expected loss and its reason."""
    action = decision.recommended_action
    loss = decision.expected_losses[action]
    render_html(
        f"""
        <div class="csx-reco">
          <div class="lab">Recommended action</div>
          <div class="act">{esc(action)}</div>
          <div class="loss">minimum expected loss {num(loss, 2)} ·
          margin {num(decision.margin, 2)} over the runner-up</div>
          <div class="why">{esc(decision.reason)}</div>
        </div>
        """
    )


def evidence_pill(label: str, value: str, colour: str) -> str:
    """Inline pill used inside evidence summaries. Returns markup, does not render."""
    c = safe_color(colour)
    return (
        f'<span class="csx-pill" style="color:{c};'
        f'background:{hex_to_rgba(c, 0.12)};border-color:{hex_to_rgba(c, 0.34)};">'
        f'{esc(label)} <b>{esc(value)}</b></span>'
    )


def evidence_row(pills: Sequence[str]) -> None:
    render_html(f'<div class="csx-pillrow">{"".join(pills)}</div>')


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def cue_table(cues: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """The reliability cues as a readable agreement/disagreement table."""
    rows = []
    for c in cues:
        lr = float(c["log_ratio_faulty"])
        rows.append(
            {
                "Evidence cue": str(c["label"]),
                "Residual": round(float(c["residual"]), 3),
                "Log-ratio (faulty vs reliable)": round(lr, 3),
                "Verdict": "disagrees with primary" if lr > 0.5
                else ("corroborates primary" if lr < -0.5 else "neutral"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# threshold comparison
# ---------------------------------------------------------------------------
def comparison_panel(comparison: Mapping[str, str]) -> None:
    """Side-by-side threshold verdict versus Bayesian verdict."""
    agree = comparison["agreement"] == "Agree"
    accent = ACCENT_LIME if agree else CRITICAL

    left, right = st.columns(2, gap="medium")
    with left:
        render_html(
            f"""
            <div class="csx-compare" style="border-left:3px solid {safe_color(TEXT_MUTED)};">
              <div class="title">Threshold baseline</div>
              <div class="verdict">{esc(comparison['threshold_action'])}</div>
              <div class="detail">Level: <b>{esc(comparison['threshold_level'])}</b><br>
              {esc(comparison['threshold_note'])}</div>
            </div>
            """
        )
    with right:
        render_html(
            f"""
            <div class="csx-compare" style="border-left:3px solid {safe_color(accent)};">
              <div class="title">CellSafe-X Bayesian decision</div>
              <div class="verdict">{esc(comparison['bayesian_action'])}</div>
              <div class="detail">{esc(comparison['bayesian_note'])}</div>
            </div>
            """
        )

    render_html(
        f'<p class="csx-note" style="margin-top:11px;">'
        f'<b style="color:{safe_color(accent)};">{esc(comparison["agreement"])}.</b> '
        f'{esc(comparison["explanation"])}</p>'
    )


# ---------------------------------------------------------------------------
# component-local CSS
# ---------------------------------------------------------------------------
COMPONENTS_CSS = """
<style>
.csx-strip { display: flex; flex-direction: column; gap: 5px; }
.csx-strip-row {
  display: flex; align-items: center; gap: 9px;
  background: #F7FAF8;
  border: 1px solid #E2E8E5;
  border-radius: 10px; padding: 7px 10px;
  transition: border-color .18s ease, background .18s ease;
}
.csx-strip-row.sel {
  border-color: #9CC8B8;
  background: #EAF4F0;
}
.csx-strip-row .bar { width: 3px; height: 20px; border-radius: 2px; flex: 0 0 3px; }
.csx-strip-row .id {
  font-size: .72rem; font-weight: 800; color: #6F7F78; width: 22px; flex: 0 0 22px;
}
.csx-strip-row .temp {
  font-size: .8rem; font-weight: 700; color: #17352A;
  font-variant-numeric: tabular-nums; width: 56px; flex: 0 0 56px;
}
.csx-strip-row .state {
  font-size: .72rem; font-weight: 700; flex: 1 1 auto;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.csx-strip-row .risk {
  font-size: .74rem; color: #6F7F78; font-variant-numeric: tabular-nums;
  flex: 0 0 auto;
}
.csx-strip-row .badge {
  font-size: .58rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: #6F7F78; border: 1px solid #D9E3DE;
  border-radius: 999px; padding: 2px 7px; flex: 0 0 auto;
}
.csx-strip-row .badge.conflict { color: #F7C95C; border-color: rgba(247,201,92,.34); }

.csx-banner {
  display: flex; align-items: center; gap: 10px;
  background: rgba(247,201,92,.055);
  border: 1px solid rgba(247,201,92,.22);
  border-radius: 12px; padding: 8px 13px; margin-bottom: 12px;
}
.csx-banner .tag {
  font-size: .58rem; font-weight: 800; letter-spacing: .12em;
  color: #F7C95C; border: 1px solid rgba(247,201,92,.40);
  border-radius: 999px; padding: 2px 8px; flex: 0 0 auto;
}
.csx-banner .txt { font-size: .74rem; color: #A8B7D4; line-height: 1.5; }

.csx-reco {
  background: #F2F8F5;
  border: 1px solid #D6E9E2;
  border-radius: 9px; padding: 14px 16px;
}
.csx-reco .lab {
  font-size: .6rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: #6F7F78;
}
.csx-reco .act {
  font-size: 1.12rem; font-weight: 800; color: #246B4B;
  margin-top: 4px; line-height: 1.25; letter-spacing: -.01em;
}
.csx-reco .loss {
  font-size: .71rem; color: #6F7F78; margin-top: 4px;
  font-variant-numeric: tabular-nums;
}
.csx-reco .why { font-size: .76rem; color: #6F7F78; margin-top: 8px; line-height: 1.55; }

.csx-pillrow { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 2px 0; }
.csx-pill {
  display: inline-flex; align-items: center; gap: 5px;
  border: 1px solid; border-radius: 999px; padding: 3px 10px;
  font-size: .69rem; font-weight: 600;
}

@media (max-width: 640px) {
  .csx-strip-row { flex-wrap: wrap; gap: 6px 8px; }
  .csx-strip-row .state { flex: 1 1 100%; order: 5; }
  .csx-reco .act { font-size: 1rem; }
}
</style>
"""
