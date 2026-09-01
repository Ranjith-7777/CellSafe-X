"""
All Plotly figures for the CellSafe-X dashboard, on the dark navy/lime theme.

Every figure is built from values the models package already computed.  These
functions never call an inference routine and never invent a number; they take
arrays and dictionaries and draw them.

House rules applied to every chart here:
  * transparent paper and plot background (the panel shows through),
  * faint blue grid lines and light-grey tick labels,
  * project colours only, never Plotly defaults,
  * right-hand margin wide enough that outside bar labels are not clipped,
  * horizontal legend above the plot so it can never overlap the data.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import model_parameters as P
from dashboard.styles import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_LIME,
    CRITICAL,
    HIGH_RISK,
    STATE_COLORS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WARNING,
    hex_to_rgba,
    plotly_layout,
    risk_color,
)

# Six distinct, theme-consistent line colours - one per cell.  Deliberately not
# the risk ramp: these identify *which* cell a trace belongs to, and reusing the
# risk colours here would imply a severity that a line colour does not carry.
CELL_TRACE_COLORS = (ACCENT_CYAN, ACCENT_LIME, "#7BA7FF", ACCENT_GREEN, "#C99BFF", WARNING)


def _cell_palette(n: int) -> list[str]:
    return [CELL_TRACE_COLORS[i % len(CELL_TRACE_COLORS)] for i in range(n)]


# ---------------------------------------------------------------------------
# time series
# ---------------------------------------------------------------------------
def temperature_chart(
    frame: pd.DataFrame,
    current_step: int,
    affected_cell: int,
    column: str = "temp_primary",
    title: str = "Pack temperature — primary sensor (SIMULATED)",
    height: int = 320,
) -> go.Figure:
    """Temperature of every cell over time, with the current step marked."""
    fig = go.Figure()
    cells = sorted(frame["cell"].unique())
    palette = _cell_palette(len(cells))

    for i, cell in enumerate(cells):
        g = frame[frame["cell"] == cell].sort_values("time_min")
        is_aff = int(cell) == int(affected_cell)
        fig.add_trace(
            go.Scatter(
                x=g["time_min"],
                y=g[column],
                mode="lines",
                name=f"Cell {int(cell) + 1}" + (" · affected" if is_aff else ""),
                line={"width": 3.0 if is_aff else 1.5, "color": palette[i]},
                opacity=1.0 if is_aff else 0.62,
                hovertemplate=f"Cell {int(cell) + 1}"
                + "<br>%{x:.2f} min<br>%{y:.1f} °C<extra></extra>",
            )
        )

    rows = frame[frame["step"] == current_step]
    if not rows.empty:
        t_now = float(rows["time_min"].iloc[0])
        fig.add_vline(x=t_now, line_width=1.6, line_dash="dash",
                      line_color=hex_to_rgba(ACCENT_LIME, 0.75))

    fig.add_hline(y=P.THRESHOLD_WARN_C, line_width=1, line_dash="dot",
                  line_color=hex_to_rgba(WARNING, 0.55))
    fig.add_hline(y=P.THRESHOLD_CRITICAL_C, line_width=1, line_dash="dot",
                  line_color=hex_to_rgba(CRITICAL, 0.55))

    fig.update_layout(**plotly_layout(height=height, title=title))
    fig.update_xaxes(title="simulated time (min)")
    fig.update_yaxes(title="°C")
    return fig


def sensor_series_chart(
    frame: pd.DataFrame,
    column: str,
    title: str,
    y_title: str,
    current_step: int,
    affected_cell: int,
    height: int = 240,
    log_y: bool = False,
) -> go.Figure:
    """Generic per-cell sensor channel over time."""
    fig = go.Figure()
    cells = sorted(frame["cell"].unique())
    palette = _cell_palette(len(cells))

    for i, cell in enumerate(cells):
        g = frame[frame["cell"] == cell].sort_values("time_min")
        is_aff = int(cell) == int(affected_cell)
        fig.add_trace(
            go.Scatter(
                x=g["time_min"],
                y=g[column],
                mode="lines",
                name=f"Cell {int(cell) + 1}",
                line={"width": 2.6 if is_aff else 1.4, "color": palette[i]},
                opacity=1.0 if is_aff else 0.58,
                hovertemplate=f"Cell {int(cell) + 1}"
                + "<br>%{x:.2f} min<br>%{y:.4g}<extra></extra>",
            )
        )

    rows = frame[frame["step"] == current_step]
    if not rows.empty:
        fig.add_vline(x=float(rows["time_min"].iloc[0]), line_width=1.4,
                      line_dash="dash", line_color=hex_to_rgba(ACCENT_LIME, 0.7))

    fig.update_layout(**plotly_layout(height=height, title=title))
    fig.update_xaxes(title="simulated time (min)")
    fig.update_yaxes(title=y_title, type="log" if log_y else "linear")
    return fig


def primary_vs_backup_chart(
    frame: pd.DataFrame, cell: int, current_step: int, height: int = 260
) -> go.Figure:
    """The single most diagnostic sensor view: two thermistors on one cell."""
    g = frame[frame["cell"] == cell].sort_values("time_min")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g["time_min"], y=g["temp_primary"], mode="lines", name="primary",
            line={"width": 2.6, "color": HIGH_RISK},
            hovertemplate="primary<br>%{x:.2f} min<br>%{y:.1f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g["time_min"], y=g["temp_backup"], mode="lines", name="backup",
            line={"width": 2.2, "color": ACCENT_CYAN, "dash": "dot"},
            hovertemplate="backup<br>%{x:.2f} min<br>%{y:.1f} °C<extra></extra>",
        )
    )
    rows = frame[frame["step"] == current_step]
    if not rows.empty:
        fig.add_vline(x=float(rows["time_min"].iloc[0]), line_width=1.4,
                      line_dash="dash", line_color=hex_to_rgba(ACCENT_LIME, 0.7))

    fig.update_layout(
        **plotly_layout(height=height,
                        title=f"Cell {int(cell) + 1} — primary versus backup thermistor")
    )
    fig.update_xaxes(title="simulated time (min)")
    fig.update_yaxes(title="°C")
    return fig


# ---------------------------------------------------------------------------
# posteriors
# ---------------------------------------------------------------------------
def posterior_chart(
    posterior: Sequence[float],
    prior: Sequence[float] | None = None,
    height: int = 280,
    title: str = "Hidden-state posterior P(Z_t | Y_1:t)",
) -> go.Figure:
    """Bar chart of the four hidden-state probabilities."""
    posterior = np.asarray(posterior, dtype=float)
    fig = go.Figure()

    if prior is not None:
        fig.add_trace(
            go.Bar(
                x=list(P.STATES),
                y=np.asarray(prior, dtype=float),
                name="prior (after transition)",
                marker_color=hex_to_rgba(TEXT_SECONDARY, 0.30),
                marker_line_width=0,
                hovertemplate="%{x}<br>prior %{y:.4f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Bar(
            x=list(P.STATES),
            y=posterior,
            name="posterior",
            marker_color=[STATE_COLORS[s] for s in P.STATES],
            marker_line_width=0,
            text=[f"{v * 100:.1f}%" for v in posterior],
            textposition="outside",
            textfont={"color": TEXT_SECONDARY, "size": 13},
            hovertemplate="%{x}<br>posterior %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(**plotly_layout(height=height, barmode="group", title=title))
    fig.update_yaxes(range=[0, 1.18], tickformat=".0%", title="probability")
    fig.update_xaxes(tickangle=-12)
    return fig


def posterior_timeline_chart(
    times: Sequence[float],
    history: np.ndarray,
    total_minutes: float | None = None,
    height: int = 280,
) -> go.Figure:
    """Stacked area of the posterior over all four states through time."""
    history = np.asarray(history, dtype=float)
    fig = go.Figure()
    for i, state in enumerate(P.STATES):
        fig.add_trace(
            go.Scatter(
                x=list(times),
                y=history[:, i],
                mode="lines",
                name=state,
                stackgroup="posterior",
                line={"width": 0.6, "color": STATE_COLORS[state]},
                fillcolor=hex_to_rgba(STATE_COLORS[state], 0.55),
                hovertemplate=f"{state}<br>%{{x:.2f}} min<br>%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **plotly_layout(height=height, title="Posterior evolution P(Z_t | Y_1:t)")
    )
    if total_minutes and total_minutes > 0:
        fig.update_xaxes(title="simulated time (min)", range=[0, total_minutes])
    else:
        fig.update_xaxes(title="simulated time (min)")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="probability")
    return fig


def root_cause_chart(
    posterior: Mapping[str, float], height: int = 300
) -> go.Figure:
    """Horizontal probability bars over the six candidate causes."""
    items = sorted(posterior.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [float(v) for _, v in items]
    top = names[-1] if names else None

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[
                ACCENT_LIME if n == top else hex_to_rgba(ACCENT_CYAN, 0.42) for n in names
            ],
            marker_line_width=0,
            text=[f"{v * 100:.1f}%" for v in values],
            textposition="outside",
            textfont={"color": TEXT_SECONDARY, "size": 13},
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_layout(height=height, title="Root cause — P(Cause | Evidence)",
                        margin={"l": 12, "r": 66, "t": 44, "b": 12})
    )
    fig.update_xaxes(range=[0, 1.22], tickformat=".0%", title="posterior")
    return fig


def sensor_reliability_chart(
    p_reliable: float, p_faulty: float, height: int = 220
) -> go.Figure:
    """The binary sensor-reliability posterior."""
    fig = go.Figure(
        go.Bar(
            x=["Reliable", "Faulty"],
            y=[float(p_reliable), float(p_faulty)],
            marker_color=[ACCENT_LIME, CRITICAL],
            marker_line_width=0,
            text=[f"{p_reliable * 100:.2f}%", f"{p_faulty * 100:.2f}%"],
            textposition="outside",
            textfont={"color": TEXT_SECONDARY, "size": 13},
            hovertemplate="%{x}<br>%{y:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_layout(height=height,
                        title="P(primary temperature sensor | evidence)")
    )
    fig.update_yaxes(range=[0, 1.20], tickformat=".0%", title="posterior")
    return fig


def sensor_fault_timeline_chart(
    times: Sequence[float], p_faulty: Sequence[float],
    total_minutes: float | None = None, height: int = 220
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=list(times), y=list(p_faulty), mode="lines", fill="tozeroy",
            line={"width": 2.2, "color": WARNING},
            fillcolor=hex_to_rgba(WARNING, 0.16),
            hovertemplate="%{x:.2f} min<br>P(faulty) %{y:.4f}<extra></extra>",
            name="P(sensor faulty)",
        )
    )
    fig.update_layout(
        **plotly_layout(height=height, title="Sensor-fault posterior over time")
    )
    if total_minutes and total_minutes > 0:
        fig.update_xaxes(title="simulated time (min)", range=[0, total_minutes])
    else:
        fig.update_xaxes(title="simulated time (min)")
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", title="probability")
    return fig


# ---------------------------------------------------------------------------
# risk
# ---------------------------------------------------------------------------
def gauge_chart(value: float, label: str, height: int = 250) -> go.Figure:
    """Circular pack-health gauge.

    The number is 1 - P(dangerous) so that a full ring reads as 'healthy',
    while the ring colour still comes from the risk ramp.
    """
    value = float(np.clip(float(value), 0.0, 1.0))
    health = 1.0 - value
    colour = risk_color(value)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health * 100.0,
            number={
                "suffix": "%",
                "font": {"size": 34, "color": "#F4F8FF"},
                "valueformat": ".1f",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickcolor": TEXT_MUTED,
                    "tickfont": {"size": 9, "color": TEXT_MUTED},
                },
                "bar": {"color": colour, "thickness": 0.30},
                "bgcolor": "rgba(132,169,255,0.07)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": hex_to_rgba(CRITICAL, 0.14)},
                    {"range": [30, 60], "color": hex_to_rgba(HIGH_RISK, 0.11)},
                    {"range": [60, 85], "color": hex_to_rgba(WARNING, 0.10)},
                    {"range": [85, 100], "color": hex_to_rgba(ACCENT_LIME, 0.11)},
                ],
                "threshold": {
                    "line": {"color": colour, "width": 3},
                    "thickness": 0.80,
                    "value": health * 100.0,
                },
            },
            title={"text": label, "font": {"size": 12, "color": TEXT_MUTED}},
        )
    )
    fig.update_layout(
        **plotly_layout(height=height, margin={"l": 22, "r": 22, "t": 44, "b": 6})
    )
    return fig


def forecast_chart(
    rows: Sequence[Mapping[str, float]], current: float, height: int = 250
) -> go.Figure:
    """Dangerous-state probability now and at each exact k-step horizon."""
    labels = ["now"] + [f"+{int(r['horizon_min'])} min" for r in rows]
    values = [float(current)] + [float(r["dangerous_probability"]) for r in rows]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[risk_color(v) for v in values],
            marker_line_width=0,
            text=[f"{v * 100:.1f}%" for v in values],
            textposition="outside",
            textfont={"color": TEXT_SECONDARY, "size": 13},
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_layout(height=height,
                        title="P(Pre-Runaway or Thermal Runaway) — exact k-step forecast")
    )
    fig.update_yaxes(range=[0, 1.18], tickformat=".0%", title="probability")
    return fig


def forecast_trajectory_chart(
    minutes: Sequence[float], danger: Sequence[float], height: int = 220
) -> go.Figure:
    """The forecast as a continuous curve: how fast risk grows if nothing is done."""
    fig = go.Figure(
        go.Scatter(
            x=list(minutes), y=list(danger), mode="lines", fill="tozeroy",
            line={"width": 2.4, "color": HIGH_RISK},
            fillcolor=hex_to_rgba(HIGH_RISK, 0.15),
            hovertemplate="+%{x:.1f} min<br>P(dangerous) %{y:.4f}<extra></extra>",
            name="forecast",
        )
    )
    fig.update_layout(
        **plotly_layout(height=height, title="Forecast trajectory if nothing is done")
    )
    fig.update_xaxes(title="minutes ahead")
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", title="probability")
    return fig


def risk_history_chart(
    times: Sequence[float],
    danger: Sequence[float],
    current_time: float,
    total_minutes: float | None = None,
    height: int = 220,
) -> go.Figure:
    """Filtered dangerous-state probability over the run so far."""
    fig = go.Figure(
        go.Scatter(
            x=list(times), y=list(danger), mode="lines", fill="tozeroy",
            line={"width": 2.4, "color": CRITICAL},
            fillcolor=hex_to_rgba(CRITICAL, 0.15),
            hovertemplate="%{x:.2f} min<br>P(dangerous) %{y:.4f}<extra></extra>",
            name="P(dangerous)",
        )
    )
    fig.add_vline(x=float(current_time), line_width=1.4, line_dash="dash",
                  line_color=hex_to_rgba(ACCENT_LIME, 0.7))
    fig.update_layout(
        **plotly_layout(height=height, title="Bayesian risk evolution")
    )
    if total_minutes and total_minutes > 0:
        fig.update_xaxes(title="simulated time (min)", range=[0, total_minutes])
    else:
        fig.update_xaxes(title="simulated time (min)")
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", title="probability")
    return fig


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------
def expected_loss_chart(
    losses: Mapping[str, float], recommended: str, height: int = 300
) -> go.Figure:
    """Expected loss per action, cheapest at the bottom, winner highlighted."""
    items = sorted(losses.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [float(v) for _, v in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[
                ACCENT_LIME if n == recommended else hex_to_rgba(ACCENT_CYAN, 0.40)
                for n in names
            ],
            marker_line_width=0,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont={"color": TEXT_SECONDARY, "size": 13},
            hovertemplate="%{y}<br>expected loss %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_layout(height=height,
                        title="Expected loss per action (lower is better)",
                        margin={"l": 12, "r": 62, "t": 44, "b": 12})
    )
    fig.update_xaxes(title="E[loss] = Σ P(state) × Loss(action, state)")
    return fig


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def confusion_heatmap(
    cm: np.ndarray, labels: Sequence[str], height: int = 400
) -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig = go.Figure(
        go.Heatmap(
            z=norm,
            x=list(labels),
            y=list(labels),
            colorscale=[
                [0.0, "#F3F7F5"],
                [0.35, "#C8E1D4"],
                [1.0, ACCENT_LIME],
            ],
            zmin=0.0,
            zmax=1.0,
            text=[[f"{norm[i, j]:.2f}<br>({int(cm[i, j])})" for j in range(len(labels))]
                  for i in range(len(labels))],
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#173C2A"},
            hovertemplate="true %{y}<br>predicted %{x}<br>%{z:.4f}<extra></extra>",
            colorbar={
                "title": {"text": "fraction", "font": {"color": TEXT_SECONDARY, "size": 13}},
                "tickfont": {"color": TEXT_SECONDARY, "size": 13},
                "outlinewidth": 0,
            },
        )
    )
    fig.update_layout(
        **plotly_layout(height=height, title="Confusion matrix (row-normalised)",
                        margin={"l": 12, "r": 12, "t": 44, "b": 12})
    )
    fig.update_xaxes(title="predicted hidden state", tickangle=-16)
    fig.update_yaxes(title="true hidden state", autorange="reversed")
    return fig


def calibration_chart(cal: Mapping[str, object], height: int = 330) -> go.Figure:
    """Reliability diagram for P(dangerous) from the saved evaluation."""
    mp = np.array(cal.get("bin_mean_predicted", []), dtype=float)
    of = np.array(cal.get("bin_observed_frequency", []), dtype=float)
    mask = np.isfinite(mp) & np.isfinite(of)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
            line={"width": 1.4, "color": hex_to_rgba(TEXT_MUTED, 0.75), "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mp[mask], y=of[mask], mode="lines+markers", name="CellSafe-X",
            line={"width": 2.6, "color": ACCENT_LIME},
            marker={"size": 8, "color": ACCENT_LIME},
            hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
        )
    )
    ece = cal.get("expected_calibration_error", float("nan"))
    fig.update_layout(
        **plotly_layout(height=height,
                        title=f"Reliability diagram — ECE {float(ece):.4f}")
    )
    fig.update_xaxes(title="predicted P(dangerous)", range=[-0.02, 1.02])
    fig.update_yaxes(title="observed frequency", range=[-0.02, 1.02])
    return fig


def scenario_bar_chart(
    per_scenario: Mapping[str, Mapping[str, float]], height: int = 300
) -> go.Figure:
    """Per-scenario accuracy alongside within-one-band accuracy."""
    names = list(per_scenario.keys())
    acc = [float(per_scenario[k]["accuracy"]) for k in names]
    band = [float(per_scenario[k]["within_one_band_accuracy"]) for k in names]
    order = np.argsort(acc)
    names = [names[i] for i in order]
    acc = [acc[i] for i in order]
    band = [band[i] for i in order]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=band, y=names, orientation="h", name="within one band",
               marker_color=hex_to_rgba(ACCENT_CYAN, 0.32), marker_line_width=0,
               hovertemplate="%{y}<br>±1 band %{x:.3f}<extra></extra>")
    )
    fig.add_trace(
        go.Bar(x=acc, y=names, orientation="h", name="exact accuracy",
               marker_color=ACCENT_LIME, marker_line_width=0,
               text=[f"{v:.3f}" for v in acc], textposition="outside",
               textfont={"color": TEXT_SECONDARY, "size": 13},
               hovertemplate="%{y}<br>accuracy %{x:.3f}<extra></extra>")
    )
    fig.update_layout(
        **plotly_layout(height=height, barmode="overlay",
                        title="Per-scenario hidden-state accuracy",
                        margin={"l": 12, "r": 56, "t": 44, "b": 12})
    )
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title="accuracy")
    return fig


def alarm_comparison_chart(
    alarm: Mapping[str, Mapping[str, float]], height: int = 280
) -> go.Figure:
    """False alarms versus missed dangerous events, per system."""
    label_map = {
        "bayesian": "CellSafe-X Bayesian",
        "threshold_warn": f"Threshold {P.THRESHOLD_WARN_C:.0f}°C",
        "threshold_critical": f"Threshold {P.THRESHOLD_CRITICAL_C:.0f}°C",
    }
    names = [label_map.get(k, k) for k in alarm]
    fa = [float(v["false_alarms"]) for v in alarm.values()]
    miss = [float(v["missed_dangerous"]) for v in alarm.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=fa, name="false alarms",
                         marker_color=WARNING, marker_line_width=0,
                         hovertemplate="%{x}<br>false alarms %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=names, y=miss, name="missed dangerous events",
                         marker_color=CRITICAL, marker_line_width=0,
                         hovertemplate="%{x}<br>missed %{y:,.0f}<extra></extra>"))
    fig.update_layout(
        **plotly_layout(height=height, barmode="group",
                        title="False alarms versus missed dangerous events")
    )
    fig.update_yaxes(title="samples")
    return fig
