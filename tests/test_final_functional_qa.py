"""Final functional QA coverage for simulator, pipeline, charts and artifacts."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import plotly.graph_objects as go
import pytest

from config import model_parameters as P
from dashboard import charts
from dashboard.evaluation_data import EvaluationDataError, load_evaluation_metrics
from dashboard import network_graph
from data.battery_simulator import simulate_pack
from models import dangerous_probability, run_pipeline


REQUIRED_COLUMNS = {
    "step", "time_min", "cell", "scenario", "temp_true", "temp_primary",
    "temp_backup", "neighbour_temp", "gas_ppm", "log_gas", "voltage_dev",
    "voltage_v", "current", "soc", "cooling_eff", "true_state",
}


@pytest.mark.parametrize("scenario", P.SCENARIOS)
@pytest.mark.parametrize("affected_cell", range(P.N_CELLS))
def test_every_scenario_cell_dataset_is_structurally_valid(scenario, affected_cell):
    sim = simulate_pack(scenario, affected_cell=affected_cell, seed=314, n_steps=36)
    frame = sim.frame
    assert REQUIRED_COLUMNS <= set(frame.columns)
    assert len(frame) == 36 * P.N_CELLS
    assert frame.groupby(["step", "cell"]).size().eq(1).all()
    assert frame.groupby("cell")["time_min"].apply(lambda x: x.is_monotonic_increasing).all()
    numeric = frame.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    assert frame["soc"].between(0, 1).all()
    assert frame["cooling_eff"].between(0, 1).all()
    assert frame["gas_ppm"].ge(0).all()
    assert frame["voltage_dev"].ge(0).all()
    assert frame["cell"].nunique() == P.N_CELLS


@pytest.mark.parametrize("scenario", P.SCENARIOS)
def test_full_pipeline_is_normalized_and_decisions_are_argmin(scenario):
    sim = simulate_pack(scenario, affected_cell=2, seed=2718, n_steps=72)
    outputs = run_pipeline(sim.frame)
    changed = False
    for cell, sequence in outputs.items():
        assert len(sequence) == 72
        for index, out in enumerate(sequence):
            for vector in (out.filter_step.prior, out.filter_step.likelihood_norm, out.posterior):
                assert np.isfinite(vector).all()
                assert (vector >= 0).all()
                assert np.isclose(vector.sum(), 1.0)
            assert np.isclose(out.sensor.p_reliable + out.sensor.p_faulty, 1.0)
            assert np.isclose(sum(out.cause.posterior.values()), 1.0)
            assert set(out.cause.posterior) == set(P.CAUSES)
            assert out.cause.top_cause == max(out.cause.posterior, key=out.cause.posterior.get)
            assert out.decision.recommended_action == min(
                out.decision.expected_losses, key=out.decision.expected_losses.get
            )
            assert np.isclose(out.forecast.current_dangerous, dangerous_probability(out.posterior))
            for risk in [out.forecast.current_dangerous, *out.forecast.dangerous_by_horizon.values()]:
                assert np.isfinite(risk) and 0 <= risk <= 1
            if index and not np.allclose(sequence[index - 1].posterior, out.posterior):
                changed = True
    assert changed


def _assert_figure_valid(fig: go.Figure) -> None:
    assert isinstance(fig, go.Figure)
    for trace in fig.data:
        x = list(trace.x) if getattr(trace, "x", None) is not None else None
        y = list(trace.y) if getattr(trace, "y", None) is not None else None
        if x is not None and y is not None:
            assert len(x) == len(y)
        for values in (x, y):
            if values is None:
                continue
            numeric = [v for v in values if isinstance(v, (int, float, np.number))]
            assert np.isfinite(numeric).all()


@pytest.mark.parametrize("step", [0, 1, 30, 71])
def test_every_dashboard_plotly_graph_constructs_at_history_boundaries(step):
    sim = simulate_pack("Internal Short Circuit", affected_cell=2, seed=42, n_steps=72)
    out = run_pipeline(sim.frame)[2]
    current = out[step]
    times = [o.time_min for o in out[: step + 1]]
    posterior_history = np.array([o.posterior for o in out[: step + 1]])
    danger = [dangerous_probability(o.posterior) for o in out[: step + 1]]
    faulty = [o.sensor.p_faulty for o in out[: step + 1]]
    metrics = load_evaluation_metrics("results/metrics/evaluation_metrics.json")
    figures = [
        charts.temperature_chart(sim.frame, step, 2),
        charts.sensor_series_chart(sim.frame, "voltage_dev", "Voltage", "V", step, 2),
        charts.primary_vs_backup_chart(sim.frame, 2, step),
        charts.posterior_chart(current.posterior, current.filter_step.prior),
        charts.posterior_timeline_chart(times, posterior_history),
        charts.root_cause_chart(current.cause.posterior),
        charts.sensor_reliability_chart(current.sensor.p_reliable, current.sensor.p_faulty),
        charts.sensor_fault_timeline_chart(times, faulty),
        charts.gauge_chart(danger[-1], "Health"),
        charts.forecast_chart(current.forecast.as_rows(), current.forecast.current_dangerous),
        charts.risk_history_chart(times, danger, current.time_min),
        charts.expected_loss_chart(current.decision.expected_losses, current.decision.recommended_action),
        charts.confusion_heatmap(np.array(metrics["hidden_state"]["confusion_matrix"]), metrics["hidden_state"]["confusion_matrix_labels"]),
        charts.scenario_bar_chart(metrics["per_scenario"]),
        charts.alarm_comparison_chart(metrics["alarm_comparison"]),
    ]
    for fig in figures:
        _assert_figure_valid(fig)


def test_evaluation_loader_handles_missing_malformed_and_valid_files(monkeypatch):
    with pytest.raises(EvaluationDataError, match="not found"):
        load_evaluation_metrics("results/metrics/does-not-exist.json")
    with monkeypatch.context() as patch:
        patch.setattr("pathlib.Path.read_text", lambda *args, **kwargs: "{not json")
        with pytest.raises(EvaluationDataError, match="could not be read"):
            load_evaluation_metrics("results/metrics/bad.json")
    valid = load_evaluation_metrics("results/metrics/evaluation_metrics.json")
    hs = valid["hidden_state"]
    # Project-level regression guarantees rather than a pinned constant: the
    # exact figure legitimately moves whenever evaluation/evaluate.py is
    # re-run after a justified model or simulator correction (see README
    # Phase-1/Phase-2 notes). These bounds are set well inside the current
    # numbers (accuracy 0.88, within-1-band 0.98, ECE 0.04) so a real
    # regression - e.g. someone weakening the emission model or breaking the
    # transition matrix - still fails the test, while a legitimate small
    # improvement or a re-seeded rerun does not.
    assert 0.80 <= hs["accuracy"] <= 1.0
    assert hs["within_one_band_accuracy"] >= 0.95
    assert valid["calibration"]["expected_calibration_error"] <= 0.10
    # Structural invariant: the confusion matrix must show zero direct
    # Healthy <-> Thermal-Runaway confusion. This is a core design claim (a
    # cell cannot "teleport" between the two extreme bands - see the
    # transition matrix's zero entries) and is far more informative than any
    # single accuracy number, because it cannot be gamed by a lucky reseed.
    cm = np.array(hs["confusion_matrix"])
    healthy_idx, runaway_idx = P.STATES.index("Healthy"), P.STATES.index("Thermal Runaway")
    assert cm[healthy_idx, runaway_idx] == 0
    assert cm[runaway_idx, healthy_idx] == 0
    # Safety invariant: every reported "missed dangerous event" must be a
    # sub-threshold label-flicker artifact, never a sample where the cell was
    # actually hot (see evaluation.evaluate's missed_dangerous_breakdown).
    breakdown = valid["alarm_comparison"]["bayesian"]["missed_dangerous_breakdown"]
    assert breakdown["missed_temp_actually_crossed_pre_runaway_threshold"] == 0


def test_bayesian_svg_is_valid_complete_and_bounded(monkeypatch):
    rendered = []
    monkeypatch.setattr(network_graph, "render_html", rendered.append)
    network_graph.render_bayesian_network()
    assert len(rendered) == 1
    svg = rendered[0][rendered[0].index("<svg"):rendered[0].index("</svg>") + 6]
    root = ET.fromstring(svg)
    assert root.attrib["viewBox"] == "0 0 1400 540"
    assert root.attrib["preserveAspectRatio"] == "xMidYMid meet"
    labels = [node.text for node in root.iter() if node.tag.endswith("text")]
    required = {
        "Previous State", "Current Hidden State", "Root Cause", "Sensor Reliability",
        "Primary Temperature", "Backup Temperature", "Voltage Evidence", "Gas Evidence",
        "Cooling Evidence", "Neighbour Evidence", "Updated Posterior", "Risk Forecast", "Decision",
    }
    assert set(labels) == required
    assert all(labels.count(label) == 1 for label in required)
    assert len([node for node in root.iter() if node.tag.endswith("path")]) >= 19
