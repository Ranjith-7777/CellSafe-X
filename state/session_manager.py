"""
Global simulation state shared by every page of the CellSafe-X dashboard.

DESIGN CONTRACT
---------------
The Bayesian pipeline is expensive and, more importantly, *stateful in time*:
`CellPipeline` filters a sequence step by step, so re-running it would be both
slow and a visible reset for the user.  Therefore:

  * The simulation is rebuilt ONLY when its signature
    (scenario, affected cell, seed, number of steps) actually changes, or when
    the user explicitly presses Regenerate / Reset.
  * Simply navigating between pages never changes the signature, so the twin,
    the timeline position and the selected cell all survive a page change.

Everything the pages need is exposed through one immutable `PageContext`
snapshot so that no page has to reach into `st.session_state` directly and no
page can accidentally mutate shared state while rendering.

This module performs NO probabilistic computation of its own.  Every number it
hands to the UI comes from `data.battery_simulator` and the `models` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from config import model_parameters as P
from data.battery_simulator import SimulationResult, simulate_pack
from models import (
    PipelineOutput,
    dangerous_probability,
    pack_risk,
    run_pipeline,
)

# Canonical (non-widget) session keys.  Widget keys are prefixed `w_` and are
# mirrored into these on every run, because Streamlit garbage-collects widget
# state for widgets that were not rendered in the current script run.
K_SCENARIO = "sim_scenario"
K_AFFECTED = "sim_affected_cell"
K_SEED = "sim_seed"
K_STEPS = "sim_n_steps"
K_STEP = "sim_step"
K_FOCUS = "sim_focus_cell"
K_PLAY = "sim_play"
K_SIGNATURE = "sim_signature"
K_TWIN = "sim_twin"

VERSION = "v1.0"


# ---------------------------------------------------------------------------
# build (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_twin(
    scenario: str, affected_cell: int, seed: int, n_steps: int
) -> Tuple[SimulationResult, Dict[int, List[PipelineOutput]]]:
    """Simulate the pack and run the full Bayesian pipeline over every cell.

    Cached as a *resource* rather than data so the PipelineOutput objects are
    handed back by reference instead of being pickled on every rerun.  This is
    the same build function the four-tab prototype used; its behaviour is
    unchanged.
    """
    sim = simulate_pack(
        scenario=scenario, affected_cell=affected_cell, seed=seed, n_steps=n_steps
    )
    results = run_pipeline(sim.frame)
    return sim, results


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------
def init_session() -> None:
    """Populate any missing session keys with safe defaults.

    Called from `app.py` on every run, so opening a deep-linked page directly
    initialises exactly as if the user had arrived through Home.
    """
    st.session_state.setdefault(K_SCENARIO, P.SCENARIOS[0])
    st.session_state.setdefault(K_AFFECTED, 2)
    st.session_state.setdefault(K_SEED, int(P.DEFAULT_SEED))
    st.session_state.setdefault(K_STEPS, int(P.DEFAULT_STEPS))
    st.session_state.setdefault(K_STEP, 0)
    st.session_state.setdefault(K_FOCUS, 2)
    st.session_state.setdefault(K_PLAY, False)


# ---------------------------------------------------------------------------
# mutation helpers (the only sanctioned ways to change shared state)
# ---------------------------------------------------------------------------
def set_scenario(scenario: str) -> None:
    """A deliberate scenario change: rebuild and rewind the timeline."""
    if scenario in P.SCENARIOS and scenario != st.session_state[K_SCENARIO]:
        st.session_state[K_SCENARIO] = scenario
        st.session_state[K_STEP] = 0
        st.session_state[K_PLAY] = False


def set_affected_cell(cell: int) -> None:
    cell = int(np.clip(int(cell), 0, P.N_CELLS - 1))
    if cell != st.session_state[K_AFFECTED]:
        st.session_state[K_AFFECTED] = cell
        st.session_state[K_FOCUS] = cell
        st.session_state[K_STEP] = 0
        st.session_state[K_PLAY] = False


def set_seed(seed: int) -> None:
    """A confirmed new seed: rebuild and rewind."""
    seed = int(np.clip(int(seed), 0, 1_000_000))
    if seed != st.session_state[K_SEED]:
        st.session_state[K_SEED] = seed
        st.session_state[K_STEP] = 0
        st.session_state[K_PLAY] = False


def set_n_steps(n_steps: int) -> None:
    n_steps = int(np.clip(int(n_steps), 30, 720))
    if n_steps != st.session_state[K_STEPS]:
        st.session_state[K_STEPS] = n_steps
        st.session_state[K_STEP] = min(st.session_state[K_STEP], n_steps - 1)
        st.session_state[K_PLAY] = False


def regenerate() -> None:
    """Draw a fresh random seed and rebuild from the start."""
    st.session_state[K_SEED] = int(np.random.default_rng().integers(0, 999_999))
    st.session_state[K_STEP] = 0
    st.session_state[K_PLAY] = False


def reset_timeline() -> None:
    """Rewind to t = 0 without touching the scenario or the seed."""
    st.session_state[K_STEP] = 0
    st.session_state[K_PLAY] = False


def set_step(step: int) -> None:
    max_step = int(st.session_state[K_STEPS]) - 1
    st.session_state[K_STEP] = int(np.clip(int(step), 0, max_step))


def set_focus_cell(cell: int) -> None:
    st.session_state[K_FOCUS] = int(np.clip(int(cell), 0, P.N_CELLS - 1))


def set_play(on: bool) -> None:
    st.session_state[K_PLAY] = bool(on)


# ---------------------------------------------------------------------------
# the snapshot handed to pages
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PageContext:
    """Everything a page needs for the currently selected time step.

    All probabilistic quantities are read straight off the persisted
    `PipelineOutput` objects; nothing is recomputed or approximated here.
    """

    sim: SimulationResult
    results: Dict[int, List[PipelineOutput]]
    scenario: str
    seed: int
    affected_cell: int
    n_steps: int
    step: int
    focus_cell: int
    playing: bool

    # -- convenience --------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(self.sim.n_cells)

    @property
    def max_step(self) -> int:
        return self.n_steps - 1

    @property
    def time_min(self) -> float:
        return float(self.focus_out.time_min)

    @property
    def total_minutes(self) -> float:
        return float(self.sim.frame["time_min"].max())

    @property
    def cell_states(self) -> List[PipelineOutput]:
        """The PipelineOutput for every cell at the current step."""
        return [self.results[c][self.step] for c in range(self.n_cells)]

    @property
    def focus_out(self) -> PipelineOutput:
        return self.results[self.focus_cell][self.step]

    @property
    def pack(self) -> Dict[str, float]:
        """Pack-level aggregation from models.risk_forecast.pack_risk."""
        return pack_risk([o.posterior for o in self.cell_states])

    @property
    def worst_cell(self) -> int:
        return int(self.pack["worst_cell_index"])

    @property
    def worst_out(self) -> PipelineOutput:
        return self.cell_states[self.worst_cell]

    @property
    def pack_danger(self) -> float:
        """P(dangerous) for the worst cell right now."""
        return float(self.pack["worst_cell_dangerous"])

    def cell_danger(self, cell: int) -> float:
        return float(dangerous_probability(self.results[cell][self.step].posterior))

    def state_of(self, cell: int) -> str:
        return P.STATES[int(np.argmax(self.results[cell][self.step].posterior))]

    def confidence_of(self, cell: int) -> float:
        return float(np.max(self.results[cell][self.step].posterior))

    # -- histories (used by the timeline charts) ----------------------------
    def posterior_history(self, cell: int) -> np.ndarray:
        """(step+1, 4) array of filtered posteriors up to the current step."""
        return np.array(
            [self.results[cell][s].posterior for s in range(self.step + 1)],
            dtype=float,
        )

    def danger_history(self, cell: int) -> Tuple[List[float], List[float]]:
        """(times, P(dangerous)) up to the current step."""
        times = [self.results[cell][s].time_min for s in range(self.step + 1)]
        danger = [
            float(dangerous_probability(self.results[cell][s].posterior))
            for s in range(self.step + 1)
        ]
        return times, danger

    def sensor_fault_history(self, cell: int) -> Tuple[List[float], List[float]]:
        times = [self.results[cell][s].time_min for s in range(self.step + 1)]
        pf = [float(self.results[cell][s].sensor.p_faulty) for s in range(self.step + 1)]
        return times, pf


def get_context() -> PageContext:
    """Build (or reuse) the simulation and return an immutable snapshot.

    The twin is rebuilt only when the signature changes.  Navigating between
    pages leaves the signature untouched, so the timeline position, the
    selected cell and every filtered posterior survive the page change.
    """
    init_session()

    scenario = str(st.session_state[K_SCENARIO])
    affected = int(st.session_state[K_AFFECTED])
    seed = int(st.session_state[K_SEED])
    n_steps = int(st.session_state[K_STEPS])
    signature = (scenario, affected, seed, n_steps)

    if st.session_state.get(K_SIGNATURE) != signature:
        with st.spinner("Running the Bayesian pipeline over the pack…"):
            st.session_state[K_TWIN] = build_twin(*signature)
        st.session_state[K_SIGNATURE] = signature

    sim, results = st.session_state[K_TWIN]

    # Clamp indices that a signature change may have invalidated.
    step = int(np.clip(int(st.session_state[K_STEP]), 0, n_steps - 1))
    focus = int(np.clip(int(st.session_state[K_FOCUS]), 0, sim.n_cells - 1))
    st.session_state[K_STEP] = step
    st.session_state[K_FOCUS] = focus

    return PageContext(
        sim=sim,
        results=results,
        scenario=scenario,
        seed=seed,
        affected_cell=int(sim.affected_cell),
        n_steps=n_steps,
        step=step,
        focus_cell=focus,
        playing=bool(st.session_state[K_PLAY]),
    )


# ---------------------------------------------------------------------------
# derived display helpers (formatting only)
# ---------------------------------------------------------------------------
def evidence_consistency(p_faulty: float) -> str:
    """Plain-language reading of the sensor-reliability posterior."""
    p = float(np.clip(float(p_faulty), 0.0, 1.0))
    if p > 0.80:
        return "Conflicting — primary sensor contradicted"
    if p > 0.50:
        return "Conflicting — primary sensor doubtful"
    if p > 0.15:
        return "Partially corroborated"
    return "Corroborated across channels"


def is_conflicted(p_faulty: float) -> bool:
    """True when the primary channel should not be presented as trustworthy."""
    return float(p_faulty) > 0.5


def cell_display_color(out: PipelineOutput, state_colors: Dict[str, str],
                       conflict_color: str) -> str:
    """Colour a cell by its inferred state - unless the evidence conflicts.

    A cell whose primary thermistor is probably lying must NOT be painted red
    just because that thermistor reads hot.  In that case the twin shows the
    neutral 'conflicting evidence' colour instead, which is the honest
    rendering of what the model actually believes.
    """
    if is_conflicted(out.sensor.p_faulty):
        return conflict_color
    return state_colors.get(P.STATES[int(np.argmax(out.posterior))], conflict_color)
