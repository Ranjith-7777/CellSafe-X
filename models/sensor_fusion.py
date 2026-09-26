"""
Overlapping-sensor reliability-aware fusion (Phase 7A).

SCOPE. `models/fault_diagnosis.py :: sensor_reliability` already performs a
Bayesian reliability test, but it is a BINARY, asymmetric test: one
designated "primary" sensor is judged reliable/faulty using the backup,
neighbour, gas and voltage channels as corroborating evidence, and only two
temperatures (primary, backup) are ever blended. This module GENERALISES
that idea (it does not replace or duplicate it - `sensor_reliability` is
unchanged and still drives the main pipeline) to N SYMMETRIC sensors that
all observe the SAME underlying quantity, with no sensor designated as the
trusted reference.

MODEL. For sensors {s_1..s_N} with readings {y_1..y_N}, consider every
non-empty subset R of "which sensors are currently reliable". Under
hypothesis R, the reliable sensors' shared consensus temperature is their
mean, and each sensor's reading is modelled as Gaussian around that
consensus - tight (OVERLAPPING_SENSOR_NOISE_C) if reliable, broad
(OVERLAPPING_SENSOR_FAULT_SIGMA_C) if not, exactly the same
"reliable = tight residual, faulty = broad residual" idea already used by
SENSOR_CUE_MODEL, generalised from one designated sensor to all of them:

    log P(R | y) prop-to  sum_s log P(reliable_s = (s in R))
                         + sum_s log N(y_s ; mean(y_i : i in R), sigma(s, s in R))

`P(reliable_s)` reuses the existing SENSOR_PRIOR for every sensor
independently (iid prior - no sensor is assumed more trustworthy a priori).
Normalising over all non-empty R gives an exact discrete posterior
P(R | y) (2**N - 1 hypotheses; N=3 here, so 7 - trivial to enumerate).

Per-sensor reliability:      P(sensor s reliable) = sum_{R: s in R} P(R | y)
Fused temperature:           E[consensus] = sum_R P(R | y) * mean(y_i : i in R)
                              (a Bayesian MODEL-AVERAGED estimate, not a
                              hard "drop the outlier" rule - every hypothesis,
                              including ones that call every sensor faulty,
                              contributes in proportion to how well it
                              explains the data)
Disagreement measure:        sample standard deviation of the raw readings
                              (reported for diagnostics; does not feed back
                              into the fusion weights - the weights come
                              entirely from the Bayesian posterior above)

This never hard-codes "if two agree, drop the third": the weight given to
any subset is a smooth function of how well it explains ALL the data,
including the prior cost of declaring more sensors faulty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config import model_parameters as P
from models.bayesian_filter import log_gaussian, log_normalise


def _nonempty_subsets(names: Sequence[str]) -> List[tuple]:
    subsets = []
    for k in range(1, len(names) + 1):
        subsets.extend(combinations(names, k))
    return subsets


@dataclass
class OverlappingSensorFusion:
    readings: Dict[str, float]
    reliability: Dict[str, float]          # P(sensor reliable | all readings)
    fused_temp_c: float                    # Bayesian model-averaged consensus
    disagreement_c: float                  # sample std of the raw readings
    subset_posterior: Dict[str, float] = field(default_factory=dict)  # diagnostics


def fuse_overlapping_sensors(
    readings: Dict[str, float],
    noise_std: Dict[str, float] | None = None,
    fault_sigma: float = P.OVERLAPPING_SENSOR_FAULT_SIGMA_C,
    prior_reliable: float = P.SENSOR_PRIOR["reliable"],
) -> OverlappingSensorFusion:
    """Reliability-aware fusion of readings that ALL claim to observe the
    same underlying temperature. `readings` may contain any subset of
    `config.OVERLAPPING_SENSOR_NAMES` - sensors not present are simply not
    part of the hypothesis space (no fabricated reading is ever inserted).
    """
    names = [n for n in P.OVERLAPPING_SENSOR_NAMES if n in readings]
    if len(names) < 1:
        raise ValueError("at least one sensor reading is required")
    noise_std = noise_std or P.OVERLAPPING_SENSOR_NOISE_C
    y = {n: float(readings[n]) for n in names}

    if len(names) == 1:
        only = names[0]
        return OverlappingSensorFusion(
            readings=dict(y),
            reliability={only: prior_reliable},
            fused_temp_c=y[only],
            disagreement_c=0.0,
            subset_posterior={f"({only},)": 1.0},
        )

    subsets = _nonempty_subsets(names)
    log_prior_reliable = float(np.log(prior_reliable))
    log_prior_faulty = float(np.log(1.0 - prior_reliable))

    log_post = np.empty(len(subsets))
    consensus_of = np.empty(len(subsets))
    for k, R in enumerate(subsets):
        consensus = float(np.mean([y[s] for s in R]))
        consensus_of[k] = consensus
        ll = 0.0
        for s in names:
            reliable = s in R
            sigma = noise_std[s] if reliable else fault_sigma
            ll += log_gaussian(y[s], consensus, sigma)
            ll += log_prior_reliable if reliable else log_prior_faulty
        log_post[k] = ll

    post = log_normalise(log_post)

    reliability = {s: 0.0 for s in names}
    fused = 0.0
    subset_posterior: Dict[str, float] = {}
    for k, R in enumerate(subsets):
        p = float(post[k])
        subset_posterior[str(R)] = p
        fused += p * consensus_of[k]
        for s in R:
            reliability[s] += p

    raw_vals = np.array(list(y.values()))
    disagreement = float(np.std(raw_vals)) if len(raw_vals) > 1 else 0.0

    return OverlappingSensorFusion(
        readings=dict(y),
        reliability=reliability,
        fused_temp_c=float(fused),
        disagreement_c=disagreement,
        subset_posterior=subset_posterior,
    )


def select_temperature_evidence(
    reading: Mapping[str, float]
) -> Tuple[Optional[float], Optional[OverlappingSensorFusion]]:
    """Goal-2 entry point used by `models.CellPipeline`: build the fused
    `temp_c` evidence from whichever of `temp_primary`/`temp_surface`/
    `temp_backup` are actually present in `reading`, correctly degrading
    from 3 to 2 to 1 sensors.

    If NONE of the overlapping temperature channels are present, returns
    `(None, None)` - the caller must omit `temp_c` from the observation dict
    entirely (marginalised, exactly like `log_gas`/`neighbour_c` already are
    for real-data replay) rather than inventing a placeholder value.
    """
    available = {}
    for name in P.OVERLAPPING_SENSOR_NAMES:
        key = f"temp_{name}"
        if key in reading:
            v = reading[key]
            if v is not None and np.isfinite(v):
                available[name] = float(v)
    if not available:
        return None, None
    fusion = fuse_overlapping_sensors(available)
    return fusion.fused_temp_c, fusion
