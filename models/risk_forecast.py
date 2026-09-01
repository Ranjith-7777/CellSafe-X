"""
Exact multi-step risk forecasting.

Given the current filtered belief b_t = P(Z_t | Y_{1:t}) and the transition
matrix A, the k-step-ahead marginal is exactly

    P(Z_{t+k} | Y_{1:t}) = b_t A^k

(no future observations are assumed - this is a pure prior forecast).  The
"dangerous-state probability" at horizon k is then the mass this distribution
places on {Pre-Runaway, Thermal Runaway}.

Nothing here is hand-assigned: every percentage the dashboard shows for 5, 15
and 30 minutes comes out of repeated multiplication by A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np

from config import model_parameters as P


def dangerous_probability(belief: Sequence[float]) -> float:
    """Total probability mass on the dangerous hidden states."""
    b = np.asarray(belief, dtype=float)
    return float(sum(b[s] for s in P.DANGEROUS_STATES))


@lru_cache(maxsize=256)
def _default_transition_power(steps: int) -> np.ndarray:
    """A^k for the default transition matrix, cached.

    The dashboard and the evaluation both call propagate() tens of thousands of
    times with the same handful of horizons, so caching the matrix powers turns
    an O(k) loop per call into a single 4x4 matrix-vector product.  The result is
    numerically identical to repeated multiplication.
    """
    return np.linalg.matrix_power(P.TRANSITION_MATRIX, int(steps))


def propagate(belief: Sequence[float], steps: int, transition: np.ndarray | None = None) -> np.ndarray:
    """b A^k, renormalised to guard against float drift."""
    b = np.asarray(belief, dtype=float).copy()
    s = b.sum()
    if s <= 0:
        raise ValueError("belief must have positive mass")
    b /= s

    steps = max(0, int(steps))
    if steps == 0:
        return b

    if transition is None:
        b = b @ _default_transition_power(steps)
    else:
        A = np.asarray(transition, dtype=float)
        for _ in range(steps):
            b = b @ A
            b /= b.sum()
    return b / b.sum()


@dataclass
class RiskForecast:
    current_dangerous: float                 # P(dangerous now), from the filter
    current_runaway: float                   # P(Thermal Runaway now)
    horizons_min: List[float] = field(default_factory=list)
    dangerous_by_horizon: Dict[float, float] = field(default_factory=dict)
    distribution_by_horizon: Dict[float, np.ndarray] = field(default_factory=dict)

    def as_rows(self) -> List[Dict[str, float]]:
        return [
            {"horizon_min": h, "dangerous_probability": self.dangerous_by_horizon[h]}
            for h in self.horizons_min
        ]


def forecast_risk(
    belief: Sequence[float],
    horizons_min: Sequence[float] = P.FORECAST_HORIZONS_MIN,
    transition: np.ndarray | None = None,
) -> RiskForecast:
    """Full forecast panel for one cell.

    `belief` is the current filtered posterior over the four hidden states.
    Each horizon in minutes is converted to an integer number of model steps via
    the DT_SECONDS time base and propagated exactly.
    """
    b = np.asarray(belief, dtype=float)
    b = b / b.sum()

    fc = RiskForecast(
        current_dangerous=dangerous_probability(b),
        current_runaway=float(b[P.S_RUNAWAY]),
        horizons_min=[float(h) for h in horizons_min],
    )

    for h in horizons_min:
        k = P.horizon_steps(h)
        future = propagate(b, k, transition)
        fc.distribution_by_horizon[float(h)] = future
        fc.dangerous_by_horizon[float(h)] = dangerous_probability(future)

    return fc


def risk_trajectory(
    belief: Sequence[float],
    max_minutes: float = 30.0,
    n_points: int = 31,
    transition: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Dangerous-state probability as a smooth curve over the forecast window.

    Used by the dashboard to plot how the risk grows if nothing is done.
    """
    A = P.TRANSITION_MATRIX if transition is None else np.asarray(transition, dtype=float)
    b = np.asarray(belief, dtype=float)
    b = b / b.sum()

    minutes = np.linspace(0.0, float(max_minutes), int(n_points))
    step_targets = [P.horizon_steps(m) if m > 0 else 0 for m in minutes]

    curve = np.zeros(len(minutes), dtype=float)
    cur = b.copy()
    done = 0
    for i, target in enumerate(step_targets):
        while done < target:
            cur = cur @ A
            cur /= cur.sum()
            done += 1
        curve[i] = dangerous_probability(cur)

    return {"minutes": minutes, "dangerous_probability": curve}


def pack_risk(cell_beliefs: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Aggregate six per-cell beliefs into a pack-level summary.

    Cells are filtered independently, so under that modelling assumption

        P(at least one cell dangerous) = 1 - prod_i (1 - P(cell i dangerous))

    We report that, plus the worst single cell, which is what an operator
    actually acts on.
    """
    per_cell = [dangerous_probability(b) for b in cell_beliefs]
    if not per_cell:
        return {"any_cell_dangerous": 0.0, "worst_cell_dangerous": 0.0, "worst_cell_index": -1}
    prod_safe = float(np.prod([1.0 - p for p in per_cell]))
    worst = int(np.argmax(per_cell))
    return {
        "any_cell_dangerous": 1.0 - prod_safe,
        "worst_cell_dangerous": float(per_cell[worst]),
        "worst_cell_index": worst,
    }
