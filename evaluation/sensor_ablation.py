"""
Phase 7B, Goal 5 — sensor-set ablation.

Run with:      python -m evaluation.sensor_ablation

Shows how DIFFERENT sensor sets change inference quality - not that "more
sensors always wins". Uses the existing, unchanged Cooling-System-Failure
synthetic scenario (a run that genuinely becomes dangerous) so accuracy/
false-alarm/miss numbers are meaningful; no scenario label or threshold is
touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import model_parameters as P
from data.battery_simulator import simulate_pack
from models.active_sensing import posterior_entropy
from models.bayesian_filter import BayesianFilter
from models.risk_forecast import dangerous_probability
from models.sensor_fusion import fuse_overlapping_sensors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "sensor_ablation"

CONFIGURATIONS: Dict[str, Dict[str, object]] = {
    "1_primary_only": {"temp_sensors": ("primary",), "extra": ()},
    "2_primary_backup": {"temp_sensors": ("primary", "backup"), "extra": ()},
    "3_all_three_temp": {"temp_sensors": ("primary", "surface", "backup"), "extra": ()},
    "4_temp_family_plus_voltage": {"temp_sensors": ("primary", "surface", "backup"), "extra": ("voltage_dev",)},
    "5_temp_family_plus_gas": {"temp_sensors": ("primary", "surface", "backup"), "extra": ("log_gas",)},
    "6_complete_supported_set": {
        "temp_sensors": ("primary", "surface", "backup"),
        "extra": ("voltage_dev", "log_gas", "neighbour_c"),
    },
}


def _frame(cell: int = 2, seed: int = 20240, n_steps: int = 180) -> pd.DataFrame:
    sim = simulate_pack(scenario="Cooling-System Failure", affected_cell=cell, seed=seed, n_steps=n_steps)
    return sim.frame[sim.frame["cell"] == cell].sort_values("step").reset_index(drop=True)


def run_configuration(frame: pd.DataFrame, temp_sensors, extra) -> Dict[str, object]:
    n = len(frame)
    dt_min = P.DT_MINUTES
    true_state = frame["true_state"].to_numpy()
    true_dangerous = frame["true_state"].isin(P.DANGEROUS_STATES).to_numpy()

    readings = {s: frame[f"temp_{s}"].to_numpy(dtype=float) for s in temp_sensors}
    temp_c = np.empty(n)
    for i in range(n):
        fusion = fuse_overlapping_sensors({s: readings[s][i] for s in temp_sensors})
        temp_c[i] = fusion.fused_temp_c
    temp_rate = np.zeros(n)
    temp_rate[1:] = (temp_c[1:] - temp_c[:-1]) / dt_min

    f = BayesianFilter()
    map_state = np.empty(n, dtype=int)
    p_dangerous = np.empty(n)
    entropy = np.empty(n)
    for i in range(n):
        obs = {"temp_c": float(temp_c[i]), "temp_rate": float(temp_rate[i])}
        if "voltage_dev" in extra:
            obs["voltage_dev"] = float(frame["voltage_dev"].iloc[i])
        if "log_gas" in extra:
            obs["log_gas"] = float(frame["log_gas"].iloc[i])
        if "neighbour_c" in extra:
            obs["neighbour_c"] = float(frame["neighbour_temp"].iloc[i])
        step = f.update(obs)
        map_state[i] = int(np.argmax(step.posterior))
        p_dangerous[i] = dangerous_probability(step.posterior)
        entropy[i] = posterior_entropy(step.posterior)

    alarm = p_dangerous > 0.5
    return {
        "accuracy": float(np.mean(map_state == true_state)),
        "false_alarms": int(np.sum(alarm & ~true_dangerous)),
        "missed_dangerous": int(np.sum(~alarm & true_dangerous)),
        "mean_entropy_nats": float(np.mean(entropy)),
        "max_p_dangerous": float(p_dangerous.max()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = _frame()

    rows: List[Dict[str, object]] = []
    for name, cfg in CONFIGURATIONS.items():
        result = run_configuration(frame, cfg["temp_sensors"], cfg["extra"])
        rows.append({"configuration": name, "sensors": ",".join(cfg["temp_sensors"]) + ("+" + ",".join(cfg["extra"]) if cfg["extra"] else ""), **result})

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "ablation.csv", index=False)
    (OUT_DIR / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nWritten to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
