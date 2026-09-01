"""
Two Bayesian inference blocks that sit either side of the HMM:

1. `sensor_reliability` - a binary hypothesis test on the PRIMARY temperature
   sensor.  Answers "is this reading trustworthy?" by asking how surprising the
   observed disagreement between the primary sensor and every other piece of
   evidence would be under each hypothesis.

2. `diagnose_root_cause` - a transparent Bayesian (naive-Bayes style)
   classifier over six candidate physical causes, computed in log space:

       P(Cause | E)  proportional-to  P(Cause) * prod_f P(e_f | Cause)

Neither block uses if/else thresholds to produce its final probabilities; both
return full normalised posteriors from Bayes' rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np

from config import model_parameters as P
from models.bayesian_filter import log_gaussian, log_normalise


# ---------------------------------------------------------------------------
# physical expectation curves (used to build residuals)
# ---------------------------------------------------------------------------
def expected_log_gas(temp_c: float) -> float:
    """How much vent gas we would expect if a cell really were this hot."""
    xs, ys = P.EXPECTED_LOG_GAS_KNOTS
    return float(np.interp(temp_c, xs, ys))


def expected_abs_voltage_dev(temp_c: float) -> float:
    """How much voltage deviation we would expect if a cell really were this hot."""
    xs, ys = P.EXPECTED_ABS_VDEV_KNOTS
    return float(np.interp(temp_c, xs, ys))


# ---------------------------------------------------------------------------
# 1. SENSOR RELIABILITY
# ---------------------------------------------------------------------------
@dataclass
class SensorReliabilityResult:
    p_reliable: float
    p_faulty: float
    cues: List[Dict[str, float | str]]   # per-cue residual and log-likelihood ratio
    fused_temp_c: float                  # posterior-weighted temperature estimate
    explanation: str

    @property
    def posterior(self) -> Dict[str, float]:
        return {"reliable": self.p_reliable, "faulty": self.p_faulty}


def _cue_residuals(reading: Mapping[str, float]) -> Dict[str, float]:
    """Build the five reliability cues from a raw sensor reading.

    Required keys: temp_primary, temp_backup, neighbour_temp, log_gas,
    voltage_dev.  Optional: temp_primary_prev (previous step's primary reading).
    """
    tp = float(reading["temp_primary"])
    tb = float(reading["temp_backup"])
    tn = float(reading["neighbour_temp"])
    lg = float(reading["log_gas"])
    vd = abs(float(reading["voltage_dev"]))
    tp_prev = reading.get("temp_primary_prev", None)

    residuals = {
        "backup_delta": tp - tb,
        "neighbour_delta": tp - tn,
        # One-sided: only a SHORTFALL against the physical signature implied by
        # the claimed temperature counts against the sensor.  An excess of gas
        # or voltage deviation means the cell has an electrical problem, not
        # that its thermistor is lying.  See config.SENSOR_CUE_MODEL.
        "gas_shortfall": min(0.0, lg - expected_log_gas(tp)),
        "volt_shortfall": min(0.0, vd - expected_abs_voltage_dev(tp)),
    }
    if tp_prev is not None and np.isfinite(tp_prev):
        residuals["jump_residual"] = tp - float(tp_prev)
    return residuals


_CUE_LABELS = {
    "backup_delta": "primary vs backup sensor",
    "neighbour_delta": "primary vs neighbouring cell",
    "gas_shortfall": "missing vent gas for the claimed temperature",
    "volt_shortfall": "missing voltage deviation for the claimed temperature",
    "jump_residual": "step-to-step temperature jump",
}


def sensor_reliability(reading: Mapping[str, float]) -> SensorReliabilityResult:
    """Posterior probability that the primary temperature sensor is faulty.

        P(H | cues)  proportional-to  P(H) * prod_c P(cue_c | H)

    Under H = reliable each residual is a *tight* zero-mean Gaussian: a working
    sensor should agree with the backup, with the neighbours, and with the gas
    and voltage signatures that physics implies at that temperature.

    Under H = faulty the primary reading is decoupled from reality, so each
    residual becomes a *broad* zero-mean Gaussian - a large disagreement is now
    unsurprising, while a perfect agreement is comparatively unlikely.

    Consequence (this is the demonstration case, and it is *derived*, not coded
    as a rule): primary = 90 degC with a normal backup, no gas, normal voltage
    and cool neighbours makes every residual huge.  The tight "reliable"
    Gaussians assign that essentially zero density while the broad "faulty"
    Gaussians assign it small-but-finite density, so the posterior swings to
    "faulty" and the system does not declare thermal runaway.
    """
    residuals = _cue_residuals(reading)

    log_post = {
        "reliable": float(np.log(P.SENSOR_PRIOR["reliable"])),
        "faulty": float(np.log(P.SENSOR_PRIOR["faulty"])),
    }

    cue_rows: List[Dict[str, float | str]] = []
    for cue, value in residuals.items():
        m = P.SENSOR_CUE_MODEL[cue]
        ll_r = log_gaussian(value, m["mu_r"], m["sd_r"])
        ll_f = log_gaussian(value, m["mu_f"], m["sd_f"])
        log_post["reliable"] += ll_r
        log_post["faulty"] += ll_f
        cue_rows.append(
            {
                "cue": cue,
                "label": _CUE_LABELS[cue],
                "residual": float(value),
                "log_lik_reliable": float(ll_r),
                "log_lik_faulty": float(ll_f),
                # positive => this cue argues the sensor is FAULTY
                "log_ratio_faulty": float(ll_f - ll_r),
            }
        )

    probs = log_normalise(np.array([log_post["reliable"], log_post["faulty"]]))
    p_reliable, p_faulty = float(probs[0]), float(probs[1])

    fused = fuse_temperature(
        float(reading["temp_primary"]), float(reading["temp_backup"]), p_faulty
    )

    # Human-readable summary: name the cue that moved the posterior most.
    cue_rows.sort(key=lambda r: abs(float(r["log_ratio_faulty"])), reverse=True)
    strongest = cue_rows[0]
    direction = "disagrees with" if float(strongest["log_ratio_faulty"]) > 0 else "is corroborated by"
    explanation = (
        f"Strongest cue: {strongest['label']} (residual {float(strongest['residual']):+.2f}); "
        f"the primary reading {direction} this evidence."
    )

    return SensorReliabilityResult(
        p_reliable=p_reliable,
        p_faulty=p_faulty,
        cues=cue_rows,
        fused_temp_c=fused,
        explanation=explanation,
    )


def fuse_temperature(primary_c: float, backup_c: float, p_faulty: float) -> float:
    """Best single temperature estimate given how much we trust the primary.

    A simple mixture:  E[T] = (1 - p_faulty) * primary + p_faulty * backup.
    When the primary is trusted this is just the primary reading; when it is
    almost certainly faulty the estimate falls back onto the backup channel.
    """
    p_faulty = float(np.clip(p_faulty, 0.0, 1.0))
    return float((1.0 - p_faulty) * primary_c + p_faulty * backup_c)


# ---------------------------------------------------------------------------
# 2. ROOT-CAUSE DIAGNOSIS
# ---------------------------------------------------------------------------
@dataclass
class RootCauseResult:
    posterior: Dict[str, float]
    top_cause: str
    top_probability: float
    explanation: str
    evidence_table: List[Dict[str, float | str]]

    def as_sorted(self) -> List[Tuple[str, float]]:
        return sorted(self.posterior.items(), key=lambda kv: kv[1], reverse=True)


_FEATURE_LABELS = {
    "temp_excess": "temperature above ambient",
    "temp_rate": "heating rate",
    "vdev_anomaly": "voltage deviation beyond what the temperature explains",
    "gas_anomaly": "vent gas beyond what the temperature explains",
    "current": "pack current",
    "soc": "state of charge",
    "cooling_eff": "cooling effectiveness",
    "self_vs_neighbour": "this cell versus its hottest neighbour",
}


def build_cause_features(reading: Mapping[str, float]) -> Dict[str, float]:
    """Assemble the root-cause feature vector from one raw sensor reading.

    Note the deliberate use of the RAW primary temperature: if we substituted a
    sensor-corrected temperature here, "Temperature Sensor Fault" could never be
    diagnosed because its own symptom would have been erased first.

    `temp_rate` is expected to be the slope of the RAW primary channel, supplied
    by the caller (models.CellPipeline computes it); it defaults to 0 so that
    the function can also be used on a single isolated reading.
    """
    tp = float(reading["temp_primary"])
    return {
        "temp_excess": tp - P.AMBIENT_C,
        "temp_rate": float(reading.get("temp_rate", 0.0)),
        "vdev_anomaly": abs(float(reading["voltage_dev"])) - expected_abs_voltage_dev(tp),
        "gas_anomaly": float(reading["log_gas"]) - expected_log_gas(tp),
        "current": float(reading["current"]),
        "soc": float(reading["soc"]),
        "cooling_eff": float(reading["cooling_eff"]),
        "self_vs_neighbour": tp - float(reading["neighbour_temp"]),
    }


def diagnose_root_cause(
    reading: Mapping[str, float],
    p_sensor_faulty: float = 0.0,
) -> RootCauseResult:
    """Posterior over the six candidate causes.

        log P(C | E) = log P(C) + sum_f w_f * log N(e_f ; mu_{C,f}, sd_{C,f})
                       + log P(sensor-fault evidence | C)   + const

    The sensor-reliability posterior enters as *soft* evidence rather than a
    hard flag:

        P(evidence | C) = p_faulty * P(faulty|C) + (1 - p_faulty) * (1 - P(faulty|C))

    so an uncertain reliability estimate only nudges the cause posterior, while
    a confident one dominates it.
    """
    feats = build_cause_features(reading)
    p_sensor_faulty = float(np.clip(p_sensor_faulty, 0.0, 1.0))

    log_scores = np.zeros(len(P.CAUSES), dtype=float)
    per_cause_feature_ll: Dict[str, Dict[str, float]] = {}

    for i, cause in enumerate(P.CAUSES):
        total = float(np.log(P.CAUSE_PRIORS[cause]))
        feature_ll: Dict[str, float] = {}
        for f in P.CAUSE_FEATURES:
            ll = P.CAUSE_LIKELIHOOD_WEIGHTS[f] * log_gaussian(
                feats[f], P.CAUSE_MEANS[cause][f], P.CAUSE_STDS[cause][f]
            )
            feature_ll[f] = ll
            total += ll

        # soft evidence from the sensor-reliability module
        pf = P.P_SENSOR_FAULTY_GIVEN_CAUSE[cause]
        soft = p_sensor_faulty * pf + (1.0 - p_sensor_faulty) * (1.0 - pf)
        total += float(np.log(max(soft, 1e-12)))
        feature_ll["sensor_reliability"] = float(np.log(max(soft, 1e-12)))

        log_scores[i] = total
        per_cause_feature_ll[cause] = feature_ll

    probs = log_normalise(log_scores)
    posterior = {c: float(p) for c, p in zip(P.CAUSES, probs)}
    top_cause = max(posterior, key=posterior.get)

    # Evidence attribution: for the winning cause, how much better did each
    # feature score than the average over the competing causes?  This is a
    # log-likelihood-ratio contribution, so it is directly interpretable.
    rows: List[Dict[str, float | str]] = []
    others = [c for c in P.CAUSES if c != top_cause]
    for f in list(P.CAUSE_FEATURES) + ["sensor_reliability"]:
        top_ll = per_cause_feature_ll[top_cause][f]
        avg_other = float(np.mean([per_cause_feature_ll[c][f] for c in others]))
        rows.append(
            {
                "feature": f,
                "label": _FEATURE_LABELS.get(f, "sensor-reliability evidence"),
                "value": float(feats.get(f, p_sensor_faulty)),
                "log_evidence_for_top": float(top_ll - avg_other),
            }
        )
    rows.sort(key=lambda r: float(r["log_evidence_for_top"]), reverse=True)

    supporting = [r["label"] for r in rows[:3] if float(r["log_evidence_for_top"]) > 0]
    if supporting:
        explanation = (
            f"'{top_cause}' is favoured mainly by: " + ", ".join(str(s) for s in supporting) + "."
        )
    else:
        explanation = (
            f"Evidence is weak and largely non-discriminative; '{top_cause}' is close to its prior."
        )

    return RootCauseResult(
        posterior=posterior,
        top_cause=top_cause,
        top_probability=float(posterior[top_cause]),
        explanation=explanation,
        evidence_table=rows,
    )
