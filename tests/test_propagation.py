"""Tests for Phase-4 cross-cell propagation forecasting (models/propagation.py)."""

from __future__ import annotations

import numpy as np
import pytest

from config import model_parameters as P
from models.propagation import forecast_pack_propagation, propagation_pressure

HEALTHY = [0.95, 0.04, 0.008, 0.002]
DANGEROUS = [0.0, 0.10, 0.30, 0.60]


def _pack(dangerous_cell: int, n_cells: int = 6):
    return {c: (DANGEROUS if c == dangerous_cell else HEALTHY) for c in range(n_cells)}


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------
def test_non_neighbour_has_zero_direct_pressure():
    hazard = {0: 0.0, 1: 0.0, 2: 0.9, 3: 0.0, 4: 0.0, 5: 0.0}
    # cell 4 is not adjacent to cell 2 in a linear chain (neighbours of 4 are 3,5)
    assert 2 not in P.pack_neighbours(4)
    pressure = propagation_pressure(4, hazard)
    assert pressure == pytest.approx(0.0)


def test_neighbour_has_nonzero_pressure_from_dangerous_source():
    hazard = {1: 0.9}
    pressure = propagation_pressure(2, hazard)  # 1 is a neighbour of 2
    assert pressure > 0.0


# ---------------------------------------------------------------------------
# zero coupling reproduces independent forecast
# ---------------------------------------------------------------------------
def test_zero_coupling_strength_reproduces_independent_forecast(monkeypatch):
    monkeypatch.setattr(P, "PROPAGATION_COUPLING_STRENGTH", 0.0)
    result = forecast_pack_propagation(_pack(2), horizon_min=15.0)
    for cell, f in result.per_cell.items():
        assert f.future_dangerous_coupled == pytest.approx(f.future_dangerous_independent, abs=1e-9)
        assert f.propagation_increment == pytest.approx(0.0, abs=1e-9)
        assert f.escalation_multiplier == pytest.approx(1.0)


def test_zero_gain_reproduces_independent_forecast(monkeypatch):
    monkeypatch.setattr(P, "PROPAGATION_GAIN", 0.0)
    result = forecast_pack_propagation(_pack(2), horizon_min=15.0)
    for f in result.per_cell.values():
        assert f.future_dangerous_coupled == pytest.approx(f.future_dangerous_independent, abs=1e-9)


# ---------------------------------------------------------------------------
# monotonicity: stronger coupling never lowers neighbour propagation risk
# ---------------------------------------------------------------------------
def test_stronger_coupling_never_lowers_neighbour_risk(monkeypatch):
    pack = _pack(2)
    increments = []
    for strength in (0.0, 0.3, 0.6, 1.0):
        monkeypatch.setattr(P, "PROPAGATION_COUPLING_STRENGTH", strength)
        result = forecast_pack_propagation(pack, horizon_min=15.0)
        increments.append(result.per_cell[1].future_dangerous_coupled)  # neighbour of 2
    assert increments == sorted(increments)


def test_higher_gain_never_lowers_neighbour_risk(monkeypatch):
    pack = _pack(2)
    dangerous_vals = []
    for gain in (0.0, 1.0, 2.0, 4.0):
        monkeypatch.setattr(P, "PROPAGATION_GAIN", gain)
        result = forecast_pack_propagation(pack, horizon_min=15.0)
        dangerous_vals.append(result.per_cell[3].future_dangerous_coupled)  # other neighbour of 2
    assert dangerous_vals == sorted(dangerous_vals)


# ---------------------------------------------------------------------------
# distributions/pack quantities remain valid
# ---------------------------------------------------------------------------
def test_pack_quantities_are_valid():
    result = forecast_pack_propagation(_pack(2), horizon_min=15.0)
    assert 0.0 <= result.pack_at_least_one_dangerous <= 1.0
    assert result.expected_dangerous_cells >= 0.0
    for f in result.per_cell.values():
        assert 0.0 <= f.future_dangerous_independent <= 1.0
        assert 0.0 <= f.future_dangerous_coupled <= 1.0
        assert f.propagation_increment >= -1e-9
        assert f.escalation_multiplier >= 1.0


def test_escalation_multiplier_capped():
    # a pack with every cell fully dangerous maximises pressure on all sides
    pack = {c: [0.0, 0.0, 0.0, 1.0] for c in range(6)}
    result = forecast_pack_propagation(pack, horizon_min=15.0)
    for f in result.per_cell.values():
        assert f.escalation_multiplier <= P.PROPAGATION_MAX_ESCALATION_MULTIPLIER + 1e-9


def test_coupled_transition_matrix_rows_valid_even_at_the_cap():
    from models.propagation import coupled_transition_matrix

    A = coupled_transition_matrix(P.PROPAGATION_MAX_ESCALATION_MULTIPLIER)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert (A >= 0).all()


def test_one_dangerous_cell_does_not_deterministically_infect_the_pack():
    """A single dangerous cell must not push a healthy, non-adjacent cell to
    near-certain danger - propagation should be a modest nudge, not a cascade."""
    result = forecast_pack_propagation(_pack(2), horizon_min=15.0)
    far_cell = result.per_cell[5]  # not adjacent to 2 (chain: 0-1-2-3-4-5)
    assert far_cell.future_dangerous_coupled < 0.3


# ---------------------------------------------------------------------------
# no mutation, determinism
# ---------------------------------------------------------------------------
def test_forecast_does_not_mutate_input_posteriors():
    pack = _pack(2)
    originals = {c: list(b) for c, b in pack.items()}
    forecast_pack_propagation(pack, horizon_min=15.0)
    for c, b in pack.items():
        assert list(b) == originals[c]


def test_forecast_is_deterministic():
    pack = _pack(2)
    r1 = forecast_pack_propagation(pack, horizon_min=15.0)
    r2 = forecast_pack_propagation(pack, horizon_min=15.0)
    for c in pack:
        assert r1.per_cell[c].future_dangerous_coupled == pytest.approx(
            r2.per_cell[c].future_dangerous_coupled
        )


# ---------------------------------------------------------------------------
# intervention integration
# ---------------------------------------------------------------------------
def test_isolation_reduces_source_to_neighbour_contribution():
    pack = _pack(2)
    baseline = forecast_pack_propagation(pack, horizon_min=15.0)
    isolated = forecast_pack_propagation(pack, horizon_min=15.0, isolated_cells=frozenset({2}))
    assert isolated.per_cell[1].future_dangerous_coupled < baseline.per_cell[1].future_dangerous_coupled
    assert isolated.per_cell[3].future_dangerous_coupled < baseline.per_cell[3].future_dangerous_coupled


def test_cooling_boost_reduces_or_holds_neighbour_risk():
    pack = _pack(2)
    baseline = forecast_pack_propagation(pack, horizon_min=15.0)
    cooled = forecast_pack_propagation(pack, horizon_min=15.0, cooling_boost=True)
    assert cooled.per_cell[1].future_dangerous_coupled <= baseline.per_cell[1].future_dangerous_coupled + 1e-9


def test_emergency_shutdown_at_source_does_not_reverse_its_own_runaway():
    """Shutdown must reduce, not eliminate, an already-runaway source cell's
    own forecast - mirrors the Phase 3 design (tiny Runaway self-loop)."""
    pack = {2: [0.0, 0.0, 0.05, 0.95]}
    result = forecast_pack_propagation(pack, horizon_min=15.0, source_interventions={2: "Emergency Shutdown"})
    assert result.per_cell[2].future_dangerous_coupled > 0.5  # still dangerous, not cured


# ---------------------------------------------------------------------------
# most-vulnerable / most-likely-source identification
# ---------------------------------------------------------------------------
def test_most_likely_propagation_source_is_the_dangerous_cell():
    result = forecast_pack_propagation(_pack(2), horizon_min=15.0)
    assert result.most_likely_propagation_source == 2


def test_most_vulnerable_neighbour_is_adjacent_to_the_source():
    result = forecast_pack_propagation(_pack(2), horizon_min=15.0)
    assert result.most_vulnerable_neighbour in P.pack_neighbours(2)
