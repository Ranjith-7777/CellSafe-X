"""Shared, page-independent application state for the CellSafe-X dashboard.

The whole point of this package is that navigating between pages must never
regenerate or reset the simulation.  See `state/session_manager.py`.
"""

from state.session_manager import (
    PageContext,
    build_twin,
    evidence_consistency,
    get_context,
    init_session,
    is_conflicted,
    regenerate,
    reset_timeline,
    set_affected_cell,
    set_focus_cell,
    set_n_steps,
    set_play,
    set_scenario,
    set_seed,
    set_step,
)

__all__ = [
    "PageContext",
    "build_twin",
    "evidence_consistency",
    "get_context",
    "init_session",
    "is_conflicted",
    "regenerate",
    "reset_timeline",
    "set_affected_cell",
    "set_focus_cell",
    "set_n_steps",
    "set_play",
    "set_scenario",
    "set_seed",
    "set_step",
]
