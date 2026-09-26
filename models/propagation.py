"""
Cross-cell propagation forecasting (Phase 4, Part A).

SCOPE NOTE
----------
Cells are filtered independently (models/__init__.py :: CellPipeline - one
BayesianFilter per cell). There is no joint multi-cell HMM here; building one
would need 4**6 = 4096 states and approximate inference (README lists this as
future work). What follows is a documented FORECASTING LAYER on top of the
existing independent per-cell posteriors: it asks "how much should a
neighbour cell's own forecast be nudged upward given how dangerous the
adjacent cell is projected to become", using the pack's linear-chain topology
(config.model_parameters.pack_neighbours - the same topology function the
simulator's thermal integration uses, so there is exactly one topology
definition in the project).

This is a model-based estimate under documented assumptions, not a validated
physical propagation probability and not derived from real multi-cell abuse
data - see config.model_parameters section "10b. CROSS-CELL PROPAGATION
MODEL" for the exact mechanism and every constant's justification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence

import numpy as np

from config import model_parameters as P
from models.risk_forecast import dangerous_probability, propagate

DEFAULT_HORIZON_MIN: float = float(P.FORECAST_HORIZONS_MIN[0])


# ---------------------------------------------------------------------------
# coupled transition matrix
# ---------------------------------------------------------------------------
def _scale_escalation(base: np.ndarray, multiplier: float) -> np.ndarray:
    """Scale every forward (more-dangerous) entry of `base` by `multiplier`
    (>= 1 here; Phase 3's controlled_transition_matrix handles <= 1 for
    interventions) and recompute the diagonal so rows still sum to 1.

    If the scaled off-diagonal mass would exceed the row (only possible for
    an unusually large multiplier), the increments are scaled back
    proportionally so the diagonal never goes negative - this keeps the
    matrix valid for ANY multiplier rather than asserting it can never
    happen, which is the safety property PROPAGATION_MAX_ESCALATION_MULTIPLIER
    exists to make unnecessary in practice.
    """
    n = base.shape[0]
    A = base.copy()
    for i in range(n):
        row_diag = base[i, i]
        forward_budget = 1.0 - row_diag  # room available for off-diagonal mass
        scaled = base[i].copy()
        for j in range(n):
            if j > i:
                scaled[j] = base[i, j] * multiplier
        off_sum = float(scaled.sum() - scaled[i])
        if off_sum > forward_budget and off_sum > 0:
            shrink = forward_budget / off_sum
            for j in range(n):
                if j > i:
                    scaled[j] *= shrink
            off_sum = forward_budget
        for j in range(n):
            if j != i:
                A[i, j] = scaled[j]
        A[i, i] = 1.0 - float(A[i].sum() - A[i, i])
    A = np.clip(A, 0.0, 1.0)
    return A / A.sum(axis=1, keepdims=True)


def coupled_transition_matrix(escalation_multiplier: float) -> np.ndarray:
    """A(pressure): baseline TRANSITION_MATRIX with escalation entries scaled
    up by `escalation_multiplier` >= 1.0 (1.0 = no coupling, i.e. baseline)."""
    if escalation_multiplier < 1.0:
        raise ValueError("propagation escalation_multiplier must be >= 1.0")
    if escalation_multiplier == 1.0:
        return P.TRANSITION_MATRIX
    return _scale_escalation(P.TRANSITION_MATRIX, escalation_multiplier)


def propagation_pressure(
    target_cell: int,
    neighbour_hazard: Mapping[int, float],
    n_cells: int = P.N_CELLS,
    excluded_sources: FrozenSet[int] = frozenset(),
) -> float:
    """Sum of coupling-weighted hazard from `target_cell`'s physical
    neighbours (config.pack_neighbours). Non-neighbours contribute nothing -
    this is the only place propagation reads topology, so a cell two hops
    away structurally cannot have a direct effect.

    `excluded_sources` lets a source's contribution be severed entirely (used
    by Isolate Affected Module - see forecast_pack_propagation).
    """
    pressure = 0.0
    for j in P.pack_neighbours(target_cell, n_cells):
        if j in excluded_sources:
            continue
        pressure += P.PROPAGATION_COUPLING_STRENGTH * float(neighbour_hazard.get(j, 0.0))
    return pressure


# ---------------------------------------------------------------------------
# per-cell / pack-level result
# ---------------------------------------------------------------------------
@dataclass
class CellPropagationForecast:
    cell: int
    horizon_min: float
    current_dangerous: float
    future_dangerous_independent: float   # no neighbour coupling (baseline forecast)
    future_dangerous_coupled: float       # with neighbour coupling
    propagation_increment: float          # coupled - independent, >= 0 by construction
    escalation_multiplier: float
    highest_risk_source_neighbour: Optional[int]


@dataclass
class PackPropagationForecast:
    horizon_min: float
    per_cell: Dict[int, CellPropagationForecast] = field(default_factory=dict)
    pack_at_least_one_dangerous: float = 0.0   # INDEPENDENCE APPROXIMATION - see note below
    expected_dangerous_cells: float = 0.0      # exact (linearity of expectation, no approximation)
    most_vulnerable_neighbour: Optional[int] = None
    most_likely_propagation_source: Optional[int] = None
    independence_note: str = (
        "pack_at_least_one_dangerous uses 1 - prod(1 - p_i), which assumes the "
        "six coupled per-cell forecasts are independent even though propagation "
        "makes them correlated. This is an explicitly labelled approximation, "
        "kept for the same reason models.risk_forecast.pack_risk uses it: an "
        "exact joint answer needs the 4**6-state coupled DBN that is out of "
        "scope. expected_dangerous_cells is NOT an approximation - it is a sum "
        "of per-cell probabilities, valid by linearity of expectation regardless "
        "of any correlation between cells."
    )


def forecast_pack_propagation(
    cell_posteriors: Mapping[int, Sequence[float]],
    horizon_min: float = DEFAULT_HORIZON_MIN,
    n_cells: int = P.N_CELLS,
    isolated_cells: FrozenSet[int] = frozenset(),
    cooling_boost: bool = False,
    source_interventions: Optional[Mapping[int, str]] = None,
) -> PackPropagationForecast:
    """Forecast every cell's dangerous-state probability at `horizon_min`,
    both without and with neighbour coupling.

    `cell_posteriors` is read-only: every belief is propagated through
    `models.risk_forecast.propagate`, which copies its input, so none of the
    caller's posterior arrays are mutated.

    Intervention integration (Phase 3 reuse, see config section 10b):
      * `isolated_cells`  - cells with "Isolate Affected Module" applied. Their
        outgoing contribution to neighbours' pressure is severed entirely
        (excluded_sources), and their OWN forecast uses Phase 3's
        controlled_transition_matrix for that action.
      * `source_interventions` - {cell: action_name} for any other Phase-3
        action applied to a specific source cell (e.g. Emergency Shutdown);
        that cell's own forecast, AND the hazard it projects onto neighbours,
        both use its post-intervention forecast rather than the baseline one.
        Emergency Shutdown reduces but does not zero a source's own future
        danger (see models/intervention.py), so it also does not zero its
        contribution to neighbours - it only reduces it, honestly reflecting
        "stops making things worse, does not undo damage already done".
      * `cooling_boost` - "Increase Cooling" applied pack-wide: scales
        PROPAGATION_GAIN by that action's own escalation_multiplier from
        Phase 3's INTERVENTION_TRANSITION_EFFECTS (reusing the existing,
        disclosed constant rather than inventing a second cooling number).
    """
    from models.intervention import forecast_under_intervention  # local import: avoid a cycle

    source_interventions = dict(source_interventions or {})
    steps = P.horizon_steps(horizon_min)

    beliefs = {c: np.asarray(b, dtype=float) / np.sum(b) for c, b in cell_posteriors.items()}

    # Step 1: each cell's OWN independent forecast (and, if intervened on,
    # its post-intervention forecast) - this is what drives neighbour pressure.
    independent_dangerous: Dict[int, float] = {}
    for c, b in beliefs.items():
        if c in isolated_cells:
            cf = forecast_under_intervention(b, "Isolate Affected Module", horizon_min)
            independent_dangerous[c] = cf.intervention_dangerous
        elif c in source_interventions:
            cf = forecast_under_intervention(b, source_interventions[c], horizon_min)
            independent_dangerous[c] = cf.intervention_dangerous
        else:
            independent_dangerous[c] = dangerous_probability(propagate(b, steps))

    gain = P.PROPAGATION_GAIN
    if cooling_boost:
        gain *= P.INTERVENTION_TRANSITION_EFFECTS["Increase Cooling"]["escalation_multiplier"]

    per_cell: Dict[int, CellPropagationForecast] = {}
    for c, b in beliefs.items():
        pressure = propagation_pressure(c, independent_dangerous, n_cells, excluded_sources=isolated_cells)
        multiplier = min(1.0 + gain * pressure, P.PROPAGATION_MAX_ESCALATION_MULTIPLIER)

        future_independent = dangerous_probability(propagate(b, steps))
        if multiplier > 1.0:
            A = coupled_transition_matrix(multiplier)
            future_coupled = dangerous_probability(propagate(b, steps, transition=A))
        else:
            future_coupled = future_independent

        neighbours = P.pack_neighbours(c, n_cells)
        contributing = [j for j in neighbours if j not in isolated_cells]
        source = (
            max(contributing, key=lambda j: independent_dangerous.get(j, 0.0))
            if contributing else None
        )

        per_cell[c] = CellPropagationForecast(
            cell=c,
            horizon_min=float(horizon_min),
            current_dangerous=dangerous_probability(b),
            future_dangerous_independent=float(future_independent),
            future_dangerous_coupled=float(future_coupled),
            propagation_increment=float(max(0.0, future_coupled - future_independent)),
            escalation_multiplier=float(multiplier),
            highest_risk_source_neighbour=source,
        )

    probs = [f.future_dangerous_coupled for f in per_cell.values()]
    prod_safe = float(np.prod([1.0 - p for p in probs])) if probs else 1.0
    expected_dangerous = float(sum(probs))

    most_vulnerable = (
        max(per_cell, key=lambda c: per_cell[c].propagation_increment) if per_cell else None
    )
    most_likely_source = (
        max(independent_dangerous, key=lambda c: independent_dangerous[c])
        if independent_dangerous else None
    )

    return PackPropagationForecast(
        horizon_min=float(horizon_min),
        per_cell=per_cell,
        pack_at_least_one_dangerous=1.0 - prod_safe,
        expected_dangerous_cells=expected_dangerous,
        most_vulnerable_neighbour=most_vulnerable,
        most_likely_propagation_source=most_likely_source,
    )
