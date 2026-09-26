"""
Phase 6 — adapter for the Warwick/Faraday Institution real thermal-runaway
dataset (see data/external/warwick_thermal_runaway/PROVENANCE.md).

SCOPE. This module is for EXTERNAL VALIDATION only: replaying real
measurements through the existing, UNCHANGED CellSafe-X hidden-state filter
to see whether its danger estimate behaves sensibly on genuine thermal-runaway
data. It is a separate track from the synthetic simulator
(data/battery_simulator.py) and must never be merged into synthetic metrics.

WHAT IS AND IS NOT FED TO THE MODEL. Only `temp_c` (from the internal
midpoint thermocouple), a causal `temp_rate`, and `voltage_dev` (from cell
voltage) are available and used. `log_gas` and `neighbour_c` are genuinely
absent for every real sample and are simply left out of the observation
dict - `models.bayesian_filter.observation_log_likelihood` already treats a
missing channel as "no evidence from that channel" (it was written with this
exact case in mind; nothing in that module is changed by this file). Sensor
reliability, root-cause diagnosis, interventions, propagation and
active-sensing are not run against real data - see PROVENANCE.md for why.

PARSING. `TR_dataTable.mat` stores its data as a MATLAB `table` (an MCOS
class), which scipy.io.loadmat cannot deserialise (it is a proprietary,
undocumented binary format outside the documented MAT5 spec). `_parse_raw()`
below manually walks the subsystem's MAT5 element stream with scipy's own
low-level reader (scipy.io.matlab._mio5) to reach the underlying numeric
arrays. This is real parsing of the real file, not a synthetic stand-in -
verified against the dataset's own reported field names, shapes and physical
value ranges (see tests/test_real_data_adapter.py and PROVENANCE.md).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import model_parameters as P

RAW_DIR = Path(__file__).resolve().parent / "external" / "warwick_thermal_runaway" / "raw"
RAW_MAT_PATH = RAW_DIR / "TR_dataTable.mat"
PROCESSED_DIR = Path(__file__).resolve().parent / "external" / "warwick_thermal_runaway" / "processed"

N_EXPERIMENTS = 3
VARIABLE_NAMES = [
    "TestID", "ExpTime", "IntPre", "CellVoltage", "ExpTimeTemp",
    "MidIntTemp", "MidSurfTemp", "NegSurfTemp", "PosSurfTemp",
    "VentPos5mmAway", "VentPos10mmAway",
]

# Channels resampled onto the "temperature" (slower) timeline, all sharing
# ExpTimeTemp - see PROVENANCE.md for original sampling rates.
TEMP_CHANNELS = [
    "MidIntTemp", "MidSurfTemp", "NegSurfTemp", "PosSurfTemp",
    "VentPos5mmAway", "VentPos10mmAway",
]
# Channels resampled onto the "fast" (voltage/pressure) timeline.
FAST_CHANNELS = ["IntPre", "CellVoltage"]


def raw_data_available() -> bool:
    return RAW_MAT_PATH.exists()


# ---------------------------------------------------------------------------
# Step 2/3 — raw .mat parsing (MCOS table, manual MAT5 element walk)
# ---------------------------------------------------------------------------
def _parse_raw() -> Dict[str, List[np.ndarray]]:
    """Return {variable_name: [array_exp0, array_exp1, array_exp2]}.

    Walks the subsystem data block scipy.io.loadmat exposes (but does not
    decode) as `__function_workspace__` when a file contains a MATLAB class
    instance. That block is itself a MAT5 byte stream missing its own
    128-byte file header (the outer file's header covers it); this function
    reconstructs a minimal valid header (see the byte-offset comments below)
    so scipy's own MAT5 element reader can walk it structurally, then reads
    the FileWrapper__ object graph scipy still cannot classify (its MCOS
    "table" class) directly off the raw element list - the object holds
    (per position) a data cell array (11 variables x 3 experiments), a
    variable-count and a VariableNames array, matching VARIABLE_NAMES above
    exactly (asserted below rather than assumed).
    """
    import scipy.io.matlab._mio5 as mio5

    with open(RAW_MAT_PATH, "rb") as f:
        f.seek(128)  # skip the outer file's own 128-byte header
        # scipy.io.loadmat treats the top-level variable named
        # "__function_workspace__" as an opaque uint8 array (raw bytes of a
        # NESTED mat5 stream). Extract it the same way pymatreader does, by
        # reading the outer file's variable list and grabbing that one array.
        f.seek(0)
        import scipy.io as sio

        outer = sio.loadmat(f, variable_names=None, simplify_cells=False)
    # scipy hides "__function_workspace__" from the normal dict in some
    # versions; fall back to a direct low-level scan if it is absent.
    fw_bytes = outer.get("__function_workspace__")
    if fw_bytes is None:
        raise RuntimeError(
            "Could not locate the MCOS subsystem block in TR_dataTable.mat "
            "(no '__function_workspace__' variable found)."
        )
    body = np.asarray(fw_bytes).tobytes()

    # The nested stream's own header text/reserved bytes were not preserved
    # (scipy strips them when returning the opaque array); rebuild a minimal
    # 128-byte MAT5 header (blank description + zero subsystem offset), then
    # splice back the four version/endian bytes that DO live at the start of
    # `body`, skipping the four reserved bytes that follow them.
    header = b" " * 124
    reconstructed = header + body[0:4] + body[8:]

    stream = io.BytesIO(reconstructed)
    mr = mio5.MatFile5Reader(stream, struct_as_record=True, simplify_cells=True)
    mr.initialize_read()
    stream.seek(128)
    hdr, _ = mr.read_var_header()
    top_cell = mr.read_var_array(hdr, process=False)
    wrapper = top_cell[0, 0]  # mat_struct with one field, "MCOS"
    obj_metadata = wrapper.MCOS["_ObjectMetadata"]
    top = obj_metadata[0]  # the table object's property array

    data_cell = top[4]      # 11-element cell array, one per variable
    var_names = list(top[9])
    if var_names != VARIABLE_NAMES:
        raise ValueError(f"Unexpected table columns: {var_names}")

    out: Dict[str, List[np.ndarray]] = {}
    for i, name in enumerate(var_names):
        col = data_cell[i]
        out[name] = [np.asarray(col[e], dtype=float).ravel() for e in range(N_EXPERIMENTS)]
    return out


def _cache_paths(exp_id: int) -> Dict[str, Path]:
    return {
        "temp": PROCESSED_DIR / f"exp{exp_id}_temp.npz",
        "fast": PROCESSED_DIR / f"exp{exp_id}_fast.npz",
    }


def parse_and_cache(force: bool = False) -> None:
    """Parse the raw .mat once and cache each experiment's raw arrays as
    compact .npz files, so later runs never need to re-walk the 215 MB file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all(p.exists() for e in range(N_EXPERIMENTS) for p in _cache_paths(e).values()):
        return
    data = _parse_raw()
    for e in range(N_EXPERIMENTS):
        paths = _cache_paths(e)
        np.savez_compressed(
            paths["temp"],
            time_s=data["ExpTimeTemp"][e],
            **{ch: data[ch][e] for ch in TEMP_CHANNELS},
        )
        np.savez_compressed(
            paths["fast"],
            time_s=data["ExpTime"][e],
            **{ch: data[ch][e] for ch in FAST_CHANNELS},
        )


# ---------------------------------------------------------------------------
# Step 4 — causal resampling onto a common CellSafe-X-step timeline
# ---------------------------------------------------------------------------
def _causal_bin_mean(time_s: np.ndarray, values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Mean of all samples with `bin_edges[i-1] < t <= bin_edges[i]`
    (a trailing, causal window - never uses a sample from after the bin's
    own timestamp). Empty bins (a gap in the source data) are NaN, not 0.
    """
    idx = np.digitize(time_s, bin_edges, right=True)
    n_bins = len(bin_edges) - 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    valid = (idx >= 1) & (idx <= n_bins)
    np.add.at(sums, idx[valid] - 1, values[valid])
    np.add.at(counts, idx[valid] - 1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return means


def resample_experiment(exp_id: int, dt_seconds: float = P.DT_SECONDS) -> pd.DataFrame:
    """Common timeline at `dt_seconds` spacing (default: CellSafe-X's own
    native step, so one row here corresponds to exactly one HMM step),
    built by trailing-window mean aggregation of both source sampling rates
    - downsampling, never upsampling, so no interpolated/future sample can
    leak into an earlier bin.
    """
    parse_and_cache()
    temp = np.load(_cache_paths(exp_id)["temp"])
    fast = np.load(_cache_paths(exp_id)["fast"])

    t_end = float(max(temp["time_s"].max(), fast["time_s"].max()))
    bin_edges = np.arange(0.0, t_end + dt_seconds, dt_seconds)
    centres = bin_edges[1:]  # each row is labelled by the END of its causal window

    out = {"time_s": centres}
    for ch in TEMP_CHANNELS:
        vals = temp[ch]
        out[ch] = _causal_bin_mean(temp["time_s"], vals, bin_edges) if vals.size else np.full(len(centres), np.nan)
    for ch in FAST_CHANNELS:
        out[ch] = _causal_bin_mean(fast["time_s"], fast[ch], bin_edges)

    return pd.DataFrame(out)


def build_observation_stream(df: pd.DataFrame) -> List[Dict[str, float]]:
    """One dict per resampled row, using ONLY the channels genuinely
    comparable to what OBS_MEANS/OBS_STDS were calibrated to represent (see
    module docstring). `temp_rate` is a causal first difference across
    already-resampled (average-only-of-the-past) bins, so it never uses a
    future sample either.

    DIAGNOSED AND EXCLUDED: `voltage_dev`. It was initially computed as
    `abs(NOMINAL_VOLTAGE - CellVoltage)`, matching the synthetic simulator's
    definition. Replaying experiment 0 with it included saturated
    P(dangerous) to ~1.0 within the first step, driven entirely by this one
    channel: at t=10 s (temp_c=18.9 degC, unambiguously Healthy, tempered
    log-likelihood spread ~13 nats favouring Healthy) `voltage_dev` alone
    contributed a ~45-nat swing toward Thermal Runaway, because these real
    cells rest at a freshly-charged ~4.1-4.2 V while `NOMINAL_VOLTAGE = 3.70`
    represents the synthetic simulator's narrow monitoring-baseline
    assumption, not a real cell's full state-of-charge voltage range. The two
    "nominal voltage" concepts are not the same physical reference point, so
    feeding the real absolute voltage through that formula does not produce
    genuine evidence - it produces a scaling artifact that would swamp every
    other channel. This is a data-adapter mapping decision (which real
    measurement is validly comparable to which calibrated channel), not a
    change to NOMINAL_VOLTAGE, OBS_STDS or any other frozen model parameter -
    see docs/REAL_VALIDATION.md for the full diagnosis and the
    voltage-included trajectory kept for comparison.
    """
    temp_c = df["MidIntTemp"].to_numpy()
    dt_min = (df["time_s"].iloc[1] - df["time_s"].iloc[0]) / 60.0 if len(df) > 1 else P.DT_MINUTES
    rate = np.empty_like(temp_c)
    rate[0] = 0.0
    rate[1:] = (temp_c[1:] - temp_c[:-1]) / dt_min

    stream: List[Dict[str, float]] = []
    for i in range(len(df)):
        obs: Dict[str, float] = {}
        if np.isfinite(temp_c[i]):
            obs["temp_c"] = float(temp_c[i])
        if np.isfinite(rate[i]):
            obs["temp_rate"] = float(rate[i])
        # voltage_dev: excluded, see docstring. log_gas, neighbour_c:
        # genuinely unavailable - omitted, not filled.
        stream.append(obs)
    return stream


# ---------------------------------------------------------------------------
# Step 5 — signal-derived event proxies (NOT paper-sourced annotations)
# ---------------------------------------------------------------------------
TEMP_RUNAWAY_ONSET_C = 150.0   # well above any benign operating temperature
PRESSURE_VENT_DELTA_BAR = 1.0  # rise above the experiment's own early baseline


@dataclass
class EventMarkers:
    temp_runaway_onset_s: Optional[float]
    pressure_vent_onset_s: Optional[float]
    note: str = (
        "Signal-derived proxies computed from this analysis' own resampled "
        "channels, NOT event timestamps extracted from the source paper "
        "(not accessible from this environment) - see PROVENANCE.md."
    )


def event_markers(df: pd.DataFrame) -> EventMarkers:
    temp = df["MidIntTemp"].to_numpy()
    above = np.where(np.isfinite(temp) & (temp >= TEMP_RUNAWAY_ONSET_C))[0]
    t_onset = float(df["time_s"].iloc[above[0]]) if len(above) else None

    pressure = df["IntPre"].to_numpy()
    finite = np.isfinite(pressure)
    baseline_mask = finite & (df["time_s"].to_numpy() <= 60.0)
    baseline = float(np.nanmean(pressure[baseline_mask])) if baseline_mask.any() else 0.0
    vent_idx = np.where(finite & (pressure >= baseline + PRESSURE_VENT_DELTA_BAR))[0]
    t_vent = float(df["time_s"].iloc[vent_idx[0]]) if len(vent_idx) else None

    return EventMarkers(temp_runaway_onset_s=t_onset, pressure_vent_onset_s=t_vent)


@dataclass
class RealExperiment:
    experiment_id: int
    frame: pd.DataFrame
    observations: List[Dict[str, float]] = field(default_factory=list)
    events: Optional[EventMarkers] = None


def load_experiment(exp_id: int, dt_seconds: float = P.DT_SECONDS) -> RealExperiment:
    if not (0 <= exp_id < N_EXPERIMENTS):
        raise ValueError(f"exp_id must be in [0, {N_EXPERIMENTS - 1}]")
    df = resample_experiment(exp_id, dt_seconds)
    return RealExperiment(
        experiment_id=exp_id,
        frame=df,
        observations=build_observation_stream(df),
        events=event_markers(df),
    )
