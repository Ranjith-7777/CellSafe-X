"""
CellSafe-X inference models.

This package also hosts the end-to-end *pipeline* that wires the four blocks
together in the order shown in assets/architecture.md:

    raw sensor reading
        -> sensor-reliability inference          (models.fault_diagnosis)
        -> fused temperature + smoothed rate
        -> exact Bayesian filter over Z_t        (models.bayesian_filter)
        -> root-cause posterior                  (models.fault_diagnosis)
        -> multi-step risk forecast              (models.risk_forecast)
        -> minimum-expected-loss action          (models.decision_engine)

The reliability block runs FIRST and deliberately so: the hidden-state filter
must not be fed a temperature it has no reason to trust, otherwise a broken
thermistor alone would drive the pack into a false Thermal Runaway belief.
The root-cause block, by contrast, is given the RAW primary reading, because a
sensor fault is only diagnosable while its symptom is still visible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from config import model_parameters as P
from models.bayesian_filter import (
    BayesianFilter,
    FilterStep,
    log_gaussian,
    log_normalise,
    log_sum_exp,
    observation_likelihood_normalised,
    observation_log_likelihood,
)
from models.decision_engine import (
    DecisionResult,
    ThresholdResult,
    compare_with_threshold,
    expected_losses,
    recommend_action,
    threshold_baseline,
)
from models.fault_diagnosis import (
    RootCauseResult,
    SensorReliabilityResult,
    build_cause_features,
    diagnose_root_cause,
    expected_abs_voltage_dev,
    expected_log_gas,
    fuse_temperature,
    sensor_reliability,
)
from models.risk_forecast import (
    RiskForecast,
    dangerous_probability,
    forecast_risk,
    pack_risk,
    propagate,
    risk_trajectory,
)

__all__ = [
    "BayesianFilter",
    "CellPipeline",
    "DecisionResult",
    "FilterStep",
    "PipelineOutput",
    "RiskForecast",
    "RootCauseResult",
    "SensorReliabilityResult",
    "ThresholdResult",
    "build_cause_features",
    "compare_with_threshold",
    "dangerous_probability",
    "diagnose_root_cause",
    "expected_abs_voltage_dev",
    "expected_log_gas",
    "expected_losses",
    "forecast_risk",
    "fuse_temperature",
    "log_gaussian",
    "log_normalise",
    "log_sum_exp",
    "observation_likelihood_normalised",
    "observation_log_likelihood",
    "pack_risk",
    "propagate",
    "recommend_action",
    "risk_trajectory",
    "run_pipeline",
    "sensor_reliability",
    "threshold_baseline",
]

# Rate estimation: raw 10-second differences of a thermistor with ~0.55 degC of
# noise would have a standard deviation of several degC/min, swamping the real
# signal.  We therefore exponentially smooth the fused temperature and take the
# slope over a one-minute window.
RATE_EMA_ALPHA = 0.35
RATE_WINDOW_STEPS = max(1, int(round(1.0 / P.DT_MINUTES)))  # 1 minute


@dataclass
class PipelineOutput:
    """Everything the pipeline produced for one (cell, step)."""

    step: int
    time_min: float
    cell: int
    reading: Dict[str, float]
    fused_temp_c: float
    smoothed_temp_c: float
    temp_rate: float
    sensor: SensorReliabilityResult
    filter_step: FilterStep
    posterior: np.ndarray
    cause: RootCauseResult
    forecast: RiskForecast
    decision: DecisionResult
    threshold: ThresholdResult


class CellPipeline:
    """Streaming inference for a single cell.  Call `step()` once per time step."""

    def __init__(self, cell: int = 0) -> None:
        self.cell = cell
        self.filter = BayesianFilter()
        self._prev_primary: Optional[float] = None
        self._ema_temp: Optional[float] = None
        self._ema_history: Deque[float] = deque(maxlen=RATE_WINDOW_STEPS + 1)
        # A second, independent rate estimate on the RAW primary channel: the
        # root-cause model works on raw readings, so it needs the raw slope.
        self._ema_raw: Optional[float] = None
        self._ema_raw_history: Deque[float] = deque(maxlen=RATE_WINDOW_STEPS + 1)
        self.history: List[PipelineOutput] = []

    def reset(self) -> None:
        self.filter.reset()
        self._prev_primary = None
        self._ema_temp = None
        self._ema_history.clear()
        self._ema_raw = None
        self._ema_raw_history.clear()
        self.history.clear()

    @staticmethod
    def _slope(history: Deque[float]) -> float:
        """Slope in degC/min across the buffered (already smoothed) samples."""
        if len(history) < 2:
            return 0.0
        span_steps = len(history) - 1
        return (history[-1] - history[0]) / (span_steps * P.DT_MINUTES)

    def step(self, row: Mapping[str, Any]) -> PipelineOutput:
        reading: Dict[str, float] = {
            "temp_primary": float(row["temp_primary"]),
            "temp_backup": float(row["temp_backup"]),
            "neighbour_temp": float(row["neighbour_temp"]),
            "log_gas": float(row["log_gas"]),
            "voltage_dev": float(row["voltage_dev"]),
            "current": float(row["current"]),
            "soc": float(row["soc"]),
            "cooling_eff": float(row["cooling_eff"]),
        }
        if self._prev_primary is not None:
            reading["temp_primary_prev"] = self._prev_primary

        # 1) how much do we trust the primary thermistor?
        sensor = sensor_reliability(reading)
        fused = sensor.fused_temp_c

        # 2) smoothed temperature and heating rate
        self._ema_temp = fused if self._ema_temp is None else (
            RATE_EMA_ALPHA * fused + (1.0 - RATE_EMA_ALPHA) * self._ema_temp
        )
        self._ema_history.append(self._ema_temp)
        rate = self._slope(self._ema_history)

        raw = reading["temp_primary"]
        self._ema_raw = raw if self._ema_raw is None else (
            RATE_EMA_ALPHA * raw + (1.0 - RATE_EMA_ALPHA) * self._ema_raw
        )
        self._ema_raw_history.append(self._ema_raw)
        raw_rate = self._slope(self._ema_raw_history)

        # 3) exact Bayesian filtering on the trusted observation vector
        obs = {
            "temp_c": fused,
            "temp_rate": float(rate),
            "voltage_dev": abs(reading["voltage_dev"]),
            "log_gas": reading["log_gas"],
            "cooling_eff": reading["cooling_eff"],
            "neighbour_c": reading["neighbour_temp"],
        }
        fstep = self.filter.update(obs)

        # 4) root cause, using the RAW primary reading plus soft sensor evidence
        cause_reading = dict(reading)
        cause_reading["temp_rate"] = float(raw_rate)
        cause = diagnose_root_cause(cause_reading, p_sensor_faulty=sensor.p_faulty)

        # 5) forecast and 6) decision
        fc = forecast_risk(fstep.posterior)
        decision = recommend_action(
            fstep.posterior, sensor_fault_probability=sensor.p_faulty
        )
        thr = threshold_baseline(reading["temp_primary"])

        self._prev_primary = reading["temp_primary"]

        out = PipelineOutput(
            step=int(row["step"]),
            time_min=float(row["time_min"]),
            cell=int(row.get("cell", self.cell)),
            reading=reading,
            fused_temp_c=float(fused),
            smoothed_temp_c=float(self._ema_temp),
            temp_rate=float(rate),
            sensor=sensor,
            filter_step=fstep,
            posterior=fstep.posterior,
            cause=cause,
            forecast=fc,
            decision=decision,
            threshold=thr,
        )
        self.history.append(out)
        return out


def run_pipeline(frame: pd.DataFrame) -> Dict[int, List[PipelineOutput]]:
    """Run the full pipeline over every cell of a simulated run.

    `frame` is the long-format DataFrame produced by
    data.battery_simulator.simulate_pack.  Returns {cell_index: [PipelineOutput]}.
    """
    results: Dict[int, List[PipelineOutput]] = {}
    for cell, group in frame.groupby("cell", sort=True):
        pipe = CellPipeline(cell=int(cell))
        group = group.sort_values("step")
        results[int(cell)] = [pipe.step(row) for _, row in group.iterrows()]
    return results
